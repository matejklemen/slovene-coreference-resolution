import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

import metrics
from data import read_corpus
from scorer import NeuralCoreferencePairScorer
from utils import get_clusters, split_into_sets, fixed_split
from visualization import build_and_display

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_segment_size", type=int, default=256)
parser.add_argument("--dataset", type=str, default="coref149")
parser.add_argument("--bert_pretrained_name_or_dir", type=str, default=None)
parser.add_argument("--freeze_pretrained", action="store_true", default=True)
parser.add_argument("--fixed_split", action="store_true")


CUSTOM_PRETRAINED_BERT_DIR = os.environ.get("CUSTOM_PRETRAINED_BERT_DIR",
                                            os.path.join("..", "data", "slo-hr-en-bert-pytorch"))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODELS_SAVE_DIR = "contextual_model_bert"
VISUALIZATION_GENERATE = True
if MODELS_SAVE_DIR and not os.path.exists(MODELS_SAVE_DIR):
    os.makedirs(MODELS_SAVE_DIR)
    logging.info(f"Created directory '{MODELS_SAVE_DIR}' for saving models.")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def prepare_document(doc, tokenizer):
    """ Converts a sentence-wise representation of document (list of lists) into a document-wise representation
    (single list) and creates a mapping between the two position indices.

    E.g. a token that is originally in sentence#0 at position#3, might now be broken up into multiple subwords
    at positions [5, 6, 7] in tokenized document."""
    tokenized_doc, mapping = [], {}
    idx_tokenized = 0
    for idx_sent, curr_sent in enumerate(doc.raw_sentences()):
        for idx_inside_sent, curr_token in enumerate(curr_sent):
            tokenized_token = tokenizer.tokenize(curr_token)
            tokenized_doc.extend(tokenized_token)
            mapping[(idx_sent, idx_inside_sent)] = list(range(idx_tokenized, idx_tokenized + len(tokenized_token)))
            idx_tokenized += len(tokenized_token)

    return tokenized_doc, mapping


class ContextualControllerBERT:
    def __init__(self, embedding_size,
                 dropout,
                 pretrained_embs_dir,
                 dataset_name,
                 fc_hidden_size=150,
                 freeze_pretrained=True,
                 learning_rate=0.001,
                 max_segment_size=(512 - 2),
                 max_span_size=10,
                 name=None):
        if not freeze_pretrained and learning_rate >= 1e-4:
            logging.warning("WARNING: BERT weights are unfrozen with a relatively high learning rate (>= 1e-4)")

        self.name = name
        self.dataset_name = dataset_name
        if self.name is None:
            self.name = time.strftime("%Y%m%d_%H%M%S")

        self.path_model_dir = os.path.join(MODELS_SAVE_DIR, self.name)
        self.path_metadata = os.path.join(self.path_model_dir, "model_metadata.txt")
        self.path_pred_clusters = os.path.join(self.path_model_dir, "pred_clusters.txt")
        self.path_pred_scores = os.path.join(self.path_model_dir, "pred_scores.txt")
        self.path_log = os.path.join(self.path_model_dir, "log.txt")

        logging.info(f"Using device {DEVICE}")
        self.max_segment_size = max_segment_size
        self.max_span_size = max_span_size
        self.embedder = BertModel.from_pretrained(pretrained_embs_dir).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_embs_dir)

        self.freeze_pretrained = freeze_pretrained
        for param in self.embedder.parameters():
            param.requires_grad = not freeze_pretrained

        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  dropout=dropout,
                                                  hidden_size=fc_hidden_size).to(DEVICE)
        if freeze_pretrained:
            self.optimizer = optim.Adam(self.scorer.parameters(),
                                        lr=learning_rate)
        else:
            self.optimizer = optim.Adam(list(self.embedder.parameters()) + list(self.scorer.parameters()),
                                        lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()
        logging.debug(f"Initialized contextual BERT-based model with name {self.name}.")
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

            logging.info(f"Created directory '{self.path_model_dir}' for model files.")
        else:
            logging.info(f"Directory '{self.path_model_dir}' already exists.")
            path_to_model = os.path.join(self.path_model_dir, 'best_scorer.th')
            if os.path.isfile(path_to_model):
                self.scorer.load_state_dict(torch.load(path_to_model))
                self.loaded_from_file = True

            path_to_embeddings = os.path.join(self.path_model_dir, 'best_bert.th')
            if os.path.isfile(path_to_embeddings):
                self.embedder.load_state_dict(torch.load(path_to_embeddings))
                self.loaded_from_file = True

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        self.scorer.zero_grad()

        # maps from (idx_sent, idx_token) to (indices_in_tokenized_doc)
        tokenized_doc, mapping = prepare_document(curr_doc, tokenizer=self.tokenizer)
        encoded_doc = self.tokenizer.convert_tokens_to_ids(tokenized_doc)
        pad_embedding = self.embedder(torch.tensor([[self.tokenizer.pad_token_id]], device=DEVICE))[0][0]  # shape: [1, 768]

        # Break down long documents into smaller sub-documents and encode them
        num_total_segments = (len(encoded_doc) + self.max_segment_size - 1) // self.max_segment_size
        doc_segments = []  # list of `num_total_segments` tensors of shape [self.max_segment_size + 2, 768]
        for idx_segment in range(num_total_segments):
            curr_segment = self.tokenizer.prepare_for_model(
                encoded_doc[idx_segment * self.max_segment_size: (idx_segment + 1) * self.max_segment_size],
                max_length=(self.max_segment_size + 2), pad_to_max_length=True,
                return_tensors="pt").to(DEVICE)

            res = self.embedder(**curr_segment)
            last_hidden_states = res[0]
            # Note: [0] because assuming a batch size of 1 (i.e. processing 1 segment at a time)
            doc_segments.append(last_hidden_states[0])

        cluster_sets = []
        mention_to_cluster_id = {}
        for i, curr_cluster in enumerate(curr_doc.clusters):
            cluster_sets.append(set(curr_cluster))
            for mid in curr_cluster:
                mention_to_cluster_id[mid] = i

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
                positions_in_doc = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
                for curr_position in positions_in_doc:
                    idx_segment = curr_position // (self.max_segment_size + 2)
                    idx_inside_segment = curr_position % (self.max_segment_size + 2)
                    head_features.append(doc_segments[idx_segment][idx_inside_segment])
            head_features = torch.stack(head_features, dim=0).unsqueeze(0)  # shape: [1, num_tokens, embedding_size]

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), 1):
                if cand_id != head_id and cand_id in gt_antecedent_ids:
                    gt_antecedents.append(idx_candidate)

                # Obtain scores for candidates and select best one as antecedent
                if idx_candidate == idx_head:
                    if len(encoded_candidates) > 0:
                        # Prepare candidates and current head mention for batched scoring
                        encoded_candidates = torch.stack(encoded_candidates, dim=0)  # [num_candidates, self.max_span_size, 768]
                        head_features = torch.repeat_interleave(head_features,
                                                                repeats=encoded_candidates.shape[0],
                                                                dim=0)  # [num_candidates, num_tokens, 768]

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
                        positions_in_doc = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
                        for curr_position in positions_in_doc:
                            idx_segment = curr_position // (self.max_segment_size + 2)
                            idx_inside_segment = curr_position % (self.max_segment_size + 2)
                            mention_features.append(doc_segments[idx_segment][idx_inside_segment])
                    mention_features = torch.stack(mention_features, dim=0)  # [num_tokens, num_features]

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
            self.optimizer.step()

        return preds, (float(doc_loss), n_examples)

    def train(self, epochs, train_docs, dev_docs):
        best_dev_loss = float("inf")
        logging.info("Starting training...")
        for idx_epoch in range(epochs):
            logging.info(f"\tRunning epoch {idx_epoch + 1}/{epochs}")
            ts_epoch = time.time()

            # Make permutation of train documents
            shuffle_indices = torch.randperm(len(train_docs))

            logging.debug("\t\tModel training step")
            self.scorer.train()
            train_loss, train_examples = 0.0, 0
            for idx_doc in shuffle_indices:
                curr_doc = train_docs[idx_doc]

                _, (doc_loss, n_examples) = self._train_doc(curr_doc)

                train_loss += doc_loss
                train_examples += n_examples

            logging.debug("\t\tModel validation step")
            self.scorer.eval()
            dev_loss, dev_examples = 0.0, 0
            for curr_doc in dev_docs:
                _, (doc_loss, n_examples) = self._train_doc(curr_doc, eval_mode=True)

                dev_loss += doc_loss
                dev_examples += n_examples

            mean_dev_loss = dev_loss / max(1, dev_examples)
            logging.info(f"\t\tTraining loss: {train_loss / max(1, train_examples): .4f}")
            logging.info(f"\t\tDev loss:      {mean_dev_loss: .4f}")
            logging.info(f"\t\tTime taken: {time.time() - ts_epoch: .3f}s")

            if mean_dev_loss < best_dev_loss and MODELS_SAVE_DIR:
                logging.info(f"\tSaving new best model to '{self.path_model_dir}'")
                torch.save(self.scorer.state_dict(), os.path.join(self.path_model_dir, 'best_scorer.th'))
                if not self.freeze_pretrained:
                    torch.save(self.embedder.state_dict(), os.path.join(self.path_model_dir, 'best_bert.th'))

                best_dev_loss = mean_dev_loss

    def evaluate(self, test_docs):
        # doc_name: <cluster assignments> pairs for all test documents
        logging.info("Evaluating...")
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
    documents = read_corpus(args.dataset)

    if args.fixed_split:
        logging.info("Using fixed dataset split")
        train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
    else:
        train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

    used_bert = args.bert_pretrained_name_or_dir
    if used_bert is None:
        used_bert = CUSTOM_PRETRAINED_BERT_DIR

    controller = ContextualControllerBERT(name=args.model_name,
                                          embedding_size=768,
                                          fc_hidden_size=args.fc_hidden_size,
                                          dropout=args.dropout,
                                          pretrained_embs_dir=used_bert,
                                          learning_rate=args.learning_rate,
                                          max_segment_size=args.max_segment_size,
                                          dataset_name=args.dataset,
                                          freeze_pretrained=args.freeze_pretrained)
    if not controller.loaded_from_file:
        controller.train(epochs=args.num_epochs, train_docs=train_docs, dev_docs=dev_docs)
        # Reload best checkpoint
        controller._prepare()

    controller.evaluate(test_docs)
    controller.visualize()



