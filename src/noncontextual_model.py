import argparse
import codecs
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common import ControllerBase, NeuralCoreferencePairScorer
from utils import extract_vocab, split_into_sets, fixed_split

from data import read_corpus

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


logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NoncontextualController(ControllerBase):
    def __init__(self, vocab,
                 embedding_size,
                 dropout,
                 dataset_name,
                 fc_hidden_size=150,
                 learning_rate=0.001,
                 max_span_size=10,
                 pretrained_embs=None,
                 freeze_pretrained=False,
                 model_name=None):
        # Note: a bit of a hack as model name automatically gets generated in super-class as well (if not provided)
        effective_model_name = time.strftime("%Y%m%d_%H%M%S") if model_name is None else model_name
        self.path_model_dir = os.path.join(self.model_base_dir, effective_model_name)

        self.vocab_path = os.path.join(self.path_model_dir, "vocab.txt")
        self.vocab = vocab
        self.max_span_size = max_span_size

        if os.path.exists(self.vocab_path):
            logging.info("Overriding provided embeddings with ones from model's directory...")
            with open(self.vocab_path, "r") as f_vocab:
                self.vocab = {token.strip(): i for i, token in enumerate(f_vocab)}
            # Make it so that a random embedding layer gets created, then later load the weights from checkpoint
            pretrained_embs = None

        if pretrained_embs is not None:
            assert pretrained_embs.shape[1] == embedding_size
            logging.info(f"Using pretrained embeddings. freeze_pretrained = {freeze_pretrained}")
            self.embedder = nn.Embedding.from_pretrained(pretrained_embs, freeze=freeze_pretrained).to(DEVICE)
        else:
            logging.info(f"Initializing random embeddings as no pretrained embeddings were given")
            self.embedder = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embedding_size).to(DEVICE)

        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  hidden_size=fc_hidden_size,
                                                  dropout=dropout).to(DEVICE)
        self.optimizer = optim.Adam(self.scorer.parameters() if freeze_pretrained else
                                    (list(self.scorer.parameters()) + list(self.embedder.parameters())),
                                    lr=learning_rate)

        super().__init__(learning_rate=learning_rate, dataset_name=dataset_name, model_name=effective_model_name)
        logging.info(f"Initialized non-contextual model with name {self.model_name}.")

        with open(self.vocab_path, "w", encoding="utf-8") as f_vocab:
            # Write vocabulary by ascending token ID (assuming indexing from 0 to (|V| - 1))
            for token, _ in sorted(self.vocab.items(), key=lambda tup: tup[1]):
                print(token, file=f_vocab)

    @property
    def model_base_dir(self):
        return "noncontextual_model"

    def train_mode(self):
        self.embedder.train()
        self.scorer.train()

    def eval_mode(self):
        self.embedder.eval()
        self.scorer.eval()

    def load_checkpoint(self):
        logging.info(f"Directory '{self.path_model_dir}' already exists.")
        path_to_model = os.path.join(self.path_model_dir, "best_scorer.th")
        path_to_embeddings = os.path.join(self.path_model_dir, "best_embs.th")

        if os.path.isfile(path_to_model):
            logging.info(f"Model with name '{self.model_name}' already exists. Loading model...")
            self.scorer.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
            self.embedder.load_state_dict(torch.load(path_to_embeddings, map_location=DEVICE))

            logging.info(f"Model with name '{self.model_name}' loaded.")
            self.loaded_from_file = True
        else:
            logging.info(f"Existing weights were not found at {path_to_model}. Using random initialization...")

    def save_checkpoint(self):
        logging.info(f"\tSaving new best model to '{self.path_model_dir}'")
        torch.save(self.embedder.state_dict(), os.path.join(self.path_model_dir, "best_embs.th"))
        torch.save(self.scorer.state_dict(), os.path.join(self.path_model_dir, "best_scorer.th"))

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        # Embed document tokens and a PAD token for use in coreference scorer.
        pad_embedding = self.embedder(torch.tensor([self.vocab["<PAD>"]], device=DEVICE))
        encoded_doc = []  # list of num_sents x [num_tokens_in_sent, embedding_size] tensors
        for curr_sent in curr_doc.raw_sentences():
            curr_encoded_sent = []
            for curr_token in curr_sent:
                curr_encoded_sent.append(self.vocab.get(curr_token.lower().strip(),
                                                        self.vocab["<UNK>"]))

            encoded_doc.append(self.embedder(torch.tensor(curr_encoded_sent, device=DEVICE)))

        cluster_sets = []
        mention_to_cluster_id = {}
        for i, curr_cluster in enumerate(curr_doc.clusters):
            cluster_sets.append(set(curr_cluster))
            for mid in curr_cluster:
                mention_to_cluster_id[mid] = i

        doc_loss, n_examples = 0.0, 0
        preds = {}

        for idx_head, (head_id, head_mention) in enumerate(curr_doc.mentions.items(), 1):
            logging.debug(f"**#{idx_head} Mention '{head_id}': {head_mention}**")

            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no features for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, encoded_candidates = [None], []
            gt_antecedents = []

            head_features = [encoded_doc[curr_token.sentence_index][curr_token.position_in_sentence]
                             for curr_token in head_mention.tokens]
            head_features = torch.stack(head_features, dim=0).unsqueeze(0)  # shape: [1, num_tokens, embedding_size]

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), 1):
                if cand_id != head_id and cand_id in gt_antecedent_ids:
                    gt_antecedents.append(idx_candidate)

                # Obtain scores for candidates and select best one as antecedent
                if idx_candidate == idx_head:
                    if len(encoded_candidates) > 0:
                        encoded_candidates = torch.stack(encoded_candidates, dim=0)  # shape: [num_candidates, self.max_span_size, embedding_size]
                        head_features = torch.repeat_interleave(head_features, repeats=encoded_candidates.shape[0],
                                                                dim=0)  # shape: [num_candidates, num_tokens, embedding_size]

                        cand_scores = self.scorer(encoded_candidates, head_features)  # shape: [num_candidates - 1, 1]
                        cand_scores = torch.cat((torch.tensor([0.0], device=DEVICE),
                                                 cand_scores.flatten())).unsqueeze(0)  # shape: [1, num_candidates]

                        # if no other antecedent exists for mention, then it's a first mention (GT is dummy antecedent)
                        if len(gt_antecedents) == 0:
                            gt_antecedents.append(0)

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
                    mention_features = torch.stack(
                        [encoded_doc[curr_token.sentence_index][curr_token.position_in_sentence]
                         for curr_token in cand_mention.tokens], dim=0)  # shape: [num_tokens, embedding_size]

                    num_tokens, num_features = mention_features.shape
                    # Pad/truncate current span to have `self.max_span_size` tokens
                    if num_tokens > self.max_span_size:
                        mention_features = mention_features[: self.max_span_size]
                    else:
                        pad_amt = self.max_span_size - num_tokens
                        mention_features = torch.cat((mention_features,
                                                      torch.repeat_interleave(pad_embedding, repeats=pad_amt, dim=0)))

                    encoded_candidates.append(mention_features)

        if not eval_mode:
            doc_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return preds, (float(doc_loss), n_examples)


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
    tok2id, id2tok = extract_vocab(train_docs, lowercase=True)

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
            pretrained_embs[curr_id, :] = torch.tensor(embs.get(curr_token.lower(), pretrained_embs[curr_id, :]),
                                                       device=DEVICE)
    elif args.use_pretrained_embs == "fastText":
        import fasttext
        logging.info("Loading pretrained Slovene fastText embeddings")
        ft = fasttext.load_model(os.path.join("..", "data", "cc.sl.300.bin"))

        embedding_size = 300
        pretrained_embs = torch.rand((len(tok2id), embedding_size))
        for curr_token, curr_id in tok2id.items():
            pretrained_embs[curr_id, :] = torch.tensor(ft.get_word_vector(curr_token), device=DEVICE)

        del ft
    else:
        embedding_size = args.embedding_size

    model = NoncontextualController(model_name=args.model_name,
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

