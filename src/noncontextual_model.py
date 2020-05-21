import os

from data import read_corpus
from utils import extract_vocab, encode, split_into_sets, get_clusters, fixed_split

import argparse
import logging
import codecs
import torch
import time
import torch.nn as nn
import torch.optim as optim
from scorer import NeuralCoreferencePairScorer
import metrics
from visualization import build_and_display
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--dataset", type=str, default="coref149")
parser.add_argument("--embedding_size", type=int, default=300,
                    help="Size of word embeddings. Only used if use_pretrained_embs is None; "
                         "otherwise, supported modes have pre-set embedding sizes")
parser.add_argument("--use_pretrained_embs", type=str, default="word2vec",
                    help="Which (if any) pretrained embeddings to use. Supported modes are 'word2vec', 'fastText' and "
                         "None")
parser.add_argument("--freeze_pretrained", action="store_true")
parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--fixed_split", action="store_true")


logger = logging.getLogger()
logger.setLevel(logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODELS_SAVE_DIR = "noncontextual_model"
VISUALIZATION_GENERATE = True

if MODELS_SAVE_DIR and not os.path.exists(MODELS_SAVE_DIR):
    os.makedirs(MODELS_SAVE_DIR)
    logging.info(f"Created directory '{MODELS_SAVE_DIR}' for saving models.")


class NoncontextualController:
    def __init__(self, vocab,
                 embedding_size,
                 dropout,
                 # TODO: this really shouldn't be required (only needed for logging preds in correct format
                 dataset_name,
                 fc_hidden_size=150,
                 learning_rate=0.001,
                 max_span_size=10,
                 pretrained_embs=None,
                 freeze_pretrained=False,
                 name=None):
        self.name = name
        self.dataset_name = dataset_name
        if self.name is None:
            self.name = time.strftime("%Y%m%d_%H%M%S")

        self.path_model_dir = os.path.join(MODELS_SAVE_DIR, self.name)
        self.path_metadata = os.path.join(self.path_model_dir, "model_metadata.txt")
        self.path_pred_clusters = os.path.join(self.path_model_dir, "pred_clusters.txt")
        self.path_pred_scores = os.path.join(self.path_model_dir, "pred_scores.txt")
        self.path_log = os.path.join(self.path_model_dir, "log.txt")
        self.vocab_path = os.path.join(self.path_model_dir, "vocab.txt")

        self.vocab = vocab
        if os.path.exists(self.vocab_path):
            logging.info("Overriding provided embeddings with ones from model's directory...")
            with open(self.vocab_path, "r") as f_vocab:
                self.vocab = {token.strip(): i for i, token in enumerate(f_vocab)}
            # Make it so that a random embedding layer gets created, then later load the saved weights
            pretrained_embs = None

        self.max_span_size = max_span_size
        if pretrained_embs is not None:
            assert pretrained_embs.shape[1] == embedding_size
            logging.info(f"Using pretrained embeddings. freeze_pretrained = {freeze_pretrained}")
            self.embedder = nn.Embedding.from_pretrained(pretrained_embs, freeze=freeze_pretrained)
        else:
            logging.info(f"Initializing random embeddings as no pretrained embeddings were given")
            self.embedder = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embedding_size)

        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  hidden_size=fc_hidden_size,
                                                  dropout=dropout)

        self.scorer_optimizer = optim.Adam(list(self.scorer.parameters()) + list(self.embedder.parameters())
                                           if not freeze_pretrained else self.scorer.parameters(),
                                           lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()
        logging.debug(f"Initialized non-contextual model with name {self.name}.")
        self._prepare()

    def _prepare(self):
        """
        Prepares directories and files for the model. If directory for the model's name already exists, it tries to load
        an existing model. If loading the model was succesful, `self.loaded_from_file` is set to True.
        """
        self.loaded_from_file = False
        # Prepare directory for saving model for this run
        if not os.path.exists(self.path_model_dir):
            os.makedirs(self.path_model_dir)

            # All logs will also be written to a file
            open(self.path_log, "w", encoding="utf-8").close()
            logger.addHandler(logging.FileHandler(self.path_log, encoding="utf-8"))

            with open(self.vocab_path, "w") as f_vocab:
                # Write vocabulary by ascending token ID (assuming indexing from 0 to (|V| - 1))
                for token, _ in sorted(self.vocab.items(), key=lambda tup: tup[1]):
                    print(token, file=f_vocab)

            logging.info(f"Created directory '{self.path_model_dir}' for model files.")
        else:
            logging.info(f"Directory '{self.path_model_dir}' already exists.")
            path_to_model = os.path.join(self.path_model_dir, 'best_scorer.th')
            path_to_embeddings = os.path.join(self.path_model_dir, 'best_embs.th')
            if os.path.isfile(path_to_model):
                logging.info(f"Model with name '{self.name}' already exists. Loading model...")
                # Load trained embeddings, load scorer weights
                self.scorer.load_state_dict(torch.load(path_to_model))
                self.embedder.load_state_dict(torch.load(path_to_embeddings))

                logging.info(f"Model with name '{self.name}' loaded.")
                self.loaded_from_file = True
            else:
                logging.info(f"Existing weights were not found at {path_to_model}. Using random initialization...")

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        self.embedder.zero_grad()
        self.scorer.zero_grad()
        encoded_doc = []  # list of num_sents x [num_tokens_in_sent, embedding_size] tensors
        for curr_sent in curr_doc.raw_sentences():
            curr_encoded_sent = []
            for curr_token in curr_sent:
                curr_encoded_sent.append(self.vocab.get(curr_token.lower().strip(), self.vocab["<UNK>"]))
            encoded_doc.append(self.embedder(torch.tensor(curr_encoded_sent, device=DEVICE)))
        pad_embedding = self.embedder(torch.tensor([self.vocab["<PAD>"]], device=DEVICE))

        cluster_sets = []
        mention_to_cluster_id = {}
        for i, curr_cluster in enumerate(curr_doc.clusters):
            cluster_sets.append(set(curr_cluster))
            for mid in curr_cluster:
                mention_to_cluster_id[mid] = i

        logging.debug(f"**Sorting mentions...**")
        doc_loss, n_examples = 0.0, 0
        preds = {}

        logging.debug(f"**Processing {len(curr_doc.mentions)} mentions...**")
        for idx_head, (head_id, head_mention) in enumerate(curr_doc.mentions.items(), 1):
            logging.debug(f"**#{idx_head} Mention '{head_id}': {head_mention}**")

            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no features for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, encoded_candidates = [None], []
            gt_antecedents = []

            head_features = []
            for curr_token in head_mention.tokens:
                head_features.append(encoded_doc[curr_token.sentence_index][curr_token.position_in_sentence])
            head_features = torch.stack(head_features, dim=0).unsqueeze(0)  # shape: [num_tokens, embedding_size]

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), 1):
                if cand_id != head_id and cand_id in gt_antecedent_ids:
                    gt_antecedents.append(idx_candidate)

                # Obtain scores for candidates and select best one as antecedent
                if idx_candidate == idx_head:
                    if len(encoded_candidates) > 0:
                        encoded_candidates = torch.stack(encoded_candidates, dim=0)  # [num_candidates, self.max_span_size, embedding_size]
                        head_features = torch.repeat_interleave(head_features,
                                                                repeats=encoded_candidates.shape[0],
                                                                dim=0)  # [num_candidates, num_tokens, embedding_size]

                        cand_scores = self.scorer(encoded_candidates, head_features)  # [num_candidates - 1, 1]
                        cand_scores = torch.cat((torch.tensor([0.0], device=DEVICE),
                                                 cand_scores.flatten())).unsqueeze(0)  # [1, num_candidates]

                        # if no other antecedent exists for mention, then it's a first mention (GT is dummy antecedent)
                        if len(gt_antecedents) == 0:
                            gt_antecedents.append(0)

                        # Get index of max value. That index equals to mention at that place
                        curr_pred = torch.argmax(cand_scores)

                        # (average) loss over all ground truth antecedents
                        doc_loss += self.loss(torch.repeat_interleave(cand_scores, repeats=len(gt_antecedents), dim=0),
                                              torch.tensor(gt_antecedents, device=DEVICE))

                        n_examples += 1
                    else:
                        # Only one candidate antecedent = first mention
                        curr_pred = 0

                    # { antecedent: [mention(s)] } pair
                    existing_refs = preds.get(candidates[int(curr_pred)], [])
                    existing_refs.append(head_id)
                    preds[candidates[int(curr_pred)]] = existing_refs
                    break
                else:
                    # Add current mention as candidate
                    candidates.append(cand_id)
                    mention_features = []
                    for curr_token in cand_mention.tokens:
                        mention_features.append(encoded_doc[curr_token.sentence_index][curr_token.position_in_sentence])
                    mention_features = torch.stack(mention_features, dim=0)  # shape: [num_tokens, embedding_size]

                    num_tokens, num_features = mention_features.shape
                    # Pad/truncate current span to have self.max_span_size tokens
                    if num_tokens > self.max_span_size:
                        mention_features = mention_features[: self.max_span_size]
                    else:
                        pad_amount = self.max_span_size - num_tokens
                        mention_features = torch.cat((mention_features,
                                                      torch.repeat_interleave(pad_embedding, repeats=pad_amount, dim=0)))

                    encoded_candidates.append(mention_features)

        if not eval_mode:
            doc_loss.backward()
            self.scorer_optimizer.step()

        return preds, (float(doc_loss), n_examples)

    def train(self, epochs, train_docs, dev_docs):
        best_dev_loss = float("inf")
        logging.info("Starting training...")
        t_start = time.time()
        for idx_epoch in range(epochs):
            logging.info(f"\tRunning epoch {idx_epoch + 1}/{epochs}")

            # Make permutation of train documents
            shuffle_indices = torch.randperm(len(train_docs))

            logging.debug("\t\tModel training step")
            self.embedder.train()
            self.scorer.train()
            train_loss, train_examples = 0.0, 0
            for idx_doc in shuffle_indices:
                curr_doc = train_docs[idx_doc]

                _, (doc_loss, n_examples) = self._train_doc(curr_doc)

                train_loss += doc_loss
                train_examples += n_examples

            logging.debug("\t\tModel validation step")
            self.embedder.eval()
            self.scorer.eval()
            dev_loss, dev_examples = 0.0, 0
            for curr_doc in dev_docs:
                _, (doc_loss, n_examples) = self._train_doc(curr_doc, eval_mode=True)

                dev_loss += doc_loss
                dev_examples += n_examples

            mean_dev_loss = dev_loss / max(1, dev_examples)
            logging.info(f"\t\tTraining loss: {train_loss / max(1, train_examples): .4f}")
            logging.info(f"\t\tDev loss:      {mean_dev_loss: .4f}")

            if mean_dev_loss < best_dev_loss and MODELS_SAVE_DIR:
                logging.info(f"\tSaving new best model to '{self.path_model_dir}'")
                torch.save(self.embedder.state_dict(), os.path.join(self.path_model_dir, 'best_embs.th'))
                torch.save(self.scorer.state_dict(), os.path.join(self.path_model_dir, 'best_scorer.th'))

                best_dev_loss = dev_loss / dev_examples

        logging.info("Training noncontextual model complete")
        logging.info(f"Training took {time.time() - t_start:.2f}s")

    def evaluate(self, test_docs):
        # doc_name: <cluster assignments> pairs for all test documents
        logging.info("Evaluating baseline...")
        all_test_preds = {}

        muc_score = metrics.Score()
        b3_score = metrics.Score()
        ceaf_score = metrics.Score()

        logging.info("Evaluation with MUC, BCube and CEAF score...")
        for curr_doc in test_docs:

            test_preds, _ = self._train_doc(curr_doc, eval_mode=True)
            test_clusters = get_clusters(test_preds)

            # Save predicted clusters for this document id
            all_test_preds[curr_doc.doc_id] = test_clusters

            # gt = ground truth, pr = predicted by model
            gt_clusters = {k: set(v) for k, v in enumerate(curr_doc.clusters)}
            pr_clusters = {}
            for (pr_ment, pr_clst) in test_clusters.items():
                if pr_clst not in pr_clusters:
                    pr_clusters[pr_clst] = set()
                pr_clusters[pr_clst].add(pr_ment)

            muc_score.add(metrics.muc(gt_clusters, pr_clusters))
            b3_score.add(metrics.b_cubed(gt_clusters, pr_clusters))
            ceaf_score.add(metrics.ceaf_e(gt_clusters, pr_clusters))

        logging.info(f"----------------------------------------------")
        logging.info(f"**Test scores**")
        logging.info(f"**MUC:      {muc_score}**")
        logging.info(f"**BCubed:   {b3_score}**")
        logging.info(f"**CEAFe:    {ceaf_score}**")
        logging.info(f"**CoNLL-12: {metrics.conll_12(muc_score, b3_score, ceaf_score)}**")
        logging.info(f"----------------------------------------------")

        if MODELS_SAVE_DIR:
            # Save test predictions and scores to file for further debugging
            with open(self.path_pred_scores, "w", encoding="utf-8") as f:
                f.writelines([
                    f"Database: {self.dataset_name}\n\n",
                    f"Test scores:\n",
                    f"MUC:      {muc_score}\n",
                    f"BCubed:   {b3_score}\n",
                    f"CEAFe:    {ceaf_score}\n",
                    f"CoNLL-12: {metrics.conll_12(muc_score, b3_score, ceaf_score)}\n",
                ])
            with open(self.path_pred_clusters, "w", encoding="utf-8") as f:
                f.writelines(["Predictions:\n"])
                for doc_id, clusters in all_test_preds.items():
                    f.writelines([
                        f"Document '{doc_id}':\n",
                        str(clusters), "\n"
                    ])

    def visualize(self):
        # Build and display visualization
        if VISUALIZATION_GENERATE:
            build_and_display(self.path_pred_clusters, self.path_pred_scores, self.path_model_dir, display=False)
        else:
            logging.warning("Visualization is disabled and thus not generated. Set VISUALIZATION_GENERATE to true.")


if __name__ == "__main__":
    args = parser.parse_args()
    embedding_size = args.embedding_size

    if args.random_seed:
        torch.random.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    documents = read_corpus(args.dataset)
    if args.fixed_split:
        logging.info("Using fixed dataset split")
        train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
    else:
        train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)
    tok2id, id2tok = extract_vocab(train_docs)

    pretrained_embs = None
    # Note: pretrained word2vec embeddings we use are uncased
    if args.use_pretrained_embs == "word2vec":
        logging.info("Loading pretrained Slovene word2vec embeddings")
        with codecs.open(os.path.join("..", "data", "model.txt"), "r", encoding="utf-8", errors="ignore") as f:
            num_tokens, embedding_size = list(map(int, f.readline().split(" ")))
            embs = {}
            for line in f:
                stripped_line = line.strip().split(" ")
                embs[stripped_line[0]] = list(map(lambda num: float(num), stripped_line[1:]))

        pretrained_embs = torch.rand((len(tok2id), embedding_size))
        for curr_token, curr_id in tok2id.items():
            # leave out-of-vocab token embeddings as random [0, 1) vectors
            pretrained_embs[curr_id, :] = torch.tensor(embs.get(curr_token.lower(), pretrained_embs[curr_id, :]))
    elif args.use_pretrained_embs == "fastText":
        import fasttext
        logging.info("Loading pretrained Slovene fastText embeddings")
        ft = fasttext.load_model(os.path.join("..", "data", "cc.sl.300.bin"))

        embedding_size = 300
        pretrained_embs = torch.rand((len(tok2id), embedding_size))
        for curr_token, curr_id in tok2id.items():
            pretrained_embs[curr_id, :] = torch.tensor(ft.get_word_vector(curr_token))

        del ft
    else:
        embedding_size = args.embedding_size

    model = NoncontextualController(name=args.model_name,
                                    vocab=tok2id,
                                    embedding_size=embedding_size,
                                    dropout=args.dropout,
                                    fc_hidden_size=args.fc_hidden_size,
                                    learning_rate=args.learning_rate,
                                    pretrained_embs=pretrained_embs,
                                    freeze_pretrained=args.freeze_pretrained,
                                    dataset_name=args.dataset)
    if not model.loaded_from_file:
        model.train(epochs=args.num_epochs, train_docs=train_docs, dev_docs=dev_docs)
        # Reload best checkpoint
        model._prepare()

    model.evaluate(test_docs)
    model.visualize()

