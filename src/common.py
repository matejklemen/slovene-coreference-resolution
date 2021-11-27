import logging
import os
import time

from tqdm import tqdm

import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_clusters
from visualization import build_and_display

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ControllerBase:
    def __init__(self, learning_rate, dataset_name, early_stopping_rounds=5, model_name=None):
        self.model_name = time.strftime("%Y%m%d_%H%M%S") if model_name is None else model_name
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds

        # Mention ranking model = always using cross-entropy
        self.loss = nn.CrossEntropyLoss()

        # Put various debugging/visualization related things in model dir
        self.path_model_dir = os.path.join(self.model_base_dir, self.model_name)
        self.path_metadata = os.path.join(self.path_model_dir, "model_metadata.txt")
        self.path_pred_clusters = os.path.join(self.path_model_dir, "pred_clusters.txt")
        self.path_pred_scores = os.path.join(self.path_model_dir, "pred_scores.txt")
        self.path_log = os.path.join(self.path_model_dir, "log.txt")

        self.loaded_from_file = False
        # self._prepare()

    @property
    def model_base_dir(self):
        """ Should return the directory where models of this type should be saved. """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(model_dir):
        raise NotImplementedError

    def save_pretrained(self, model_dir):
        raise NotImplementedError

    def _prepare(self):
        if os.path.exists(self.path_model_dir):
            self.load_checkpoint()
        else:
            os.makedirs(self.path_model_dir)
            logger.addHandler(logging.FileHandler(self.path_log, mode="w", encoding="utf-8"))
            logging.info(f"Created directory '{self.path_model_dir}' for model files")

    def load_checkpoint(self):
        """ Should load weights and other checkpoint-related data for underlying model of controller. """
        raise NotImplementedError

    def save_checkpoint(self):
        """ Should save weights and other checkpoint-related data for underlying model of controller. """
        pass

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document. Returns predictions, loss and number
            of examples evaluated. """
        raise NotImplementedError

    def train_mode(self):
        """ Set underlying modules to train mode. """
        raise NotImplementedError

    def eval_mode(self):
        """ Set underlying modules to eval mode. """
        raise NotImplementedError

    def train(self, epochs, train_docs, dev_docs):
        logging.info("Starting training")

        best_dev_loss, best_epoch = float("inf"), None
        t_start = time.time()
        for idx_epoch in range(epochs):
            t_epoch_start = time.time()
            shuffle_indices = torch.randperm(len(train_docs))

            self.train_mode()
            train_loss, train_examples = 0.0, 0
            for idx_doc in tqdm(shuffle_indices):
                curr_doc = train_docs[idx_doc]

                _, (doc_loss, n_examples) = self._train_doc(curr_doc)

                train_loss += doc_loss
                train_examples += n_examples

            self.eval_mode()
            dev_loss, dev_examples = 0.0, 0
            for curr_doc in dev_docs:
                _, (doc_loss, n_examples) = self._train_doc(curr_doc, eval_mode=True)

                dev_loss += doc_loss
                dev_examples += n_examples

            train_loss /= max(1, train_examples)
            dev_loss /= max(1, dev_examples)

            logging.info(f"[Epoch #{1 + idx_epoch}] "
                         f"training loss: {train_loss: .4f}, dev loss: {dev_loss: .4f} "
                         f"[took {time.time() - t_epoch_start:.2f}s]")

            if dev_loss < best_dev_loss:
                self.save_checkpoint()
                # Save this score as best
                best_dev_loss = dev_loss
                best_epoch = idx_epoch

            if idx_epoch - best_epoch == self.early_stopping_rounds:
                logging.info(f"Validation metric did not improve for {self.early_stopping_rounds} rounds, "
                             f"stopping early")
                break

        logging.info(f"Training complete: took {time.time() - t_start:.2f}s")

        # Add model train scores to model metadata
        with open(self.path_metadata, "a", encoding="utf-8") as f:
            logging.info(f"Saving best validation score to {self.path_metadata}")
            f.writelines([
                "\n",
                "Train model scores:\n",
                f"Best validation set loss: {best_dev_loss}\n",
            ])

        return best_dev_loss

    def evaluate_single(self, document):
        # doc_name: <cluster assignments> pairs for all test documents
        logging.info("Evaluating a single document...")

        predictions, _, probabilities = self._train_doc(document, eval_mode=True)
        clusters = get_clusters(predictions)
        scores = {m: probabilities[m] for m in clusters.keys()}

        return { "predictions": predictions, "clusters": clusters, "scores": scores }

    @torch.no_grad()
    def evaluate(self, test_docs):
        # doc_name: <cluster assignments> pairs for all test documents
        logging.info("Evaluating...")
        all_test_preds = {}

        # [MUC score]
        # The MUC score counts the minimum number of links between mentions
        # to be inserted or deleted when mapping a system response to a gold standard key set
        # [B3 score]
        # B3 computes precision and recall for all mentions in the document,
        # which are then combined to produce the final precision and recall numbers for the entire output
        # [CEAF score]
        # CEAF applies a similarity metric (either mention based or entity based) for each pair of entities
        # (i.e. a set of mentions) to measure the goodness of each possible alignment.
        # The best mapping is used for calculating CEAF precision, recall and F-measure
        muc_score = metrics.Score()
        b3_score = metrics.Score()
        ceaf_score = metrics.Score()

        for curr_doc in tqdm(test_docs):

            test_preds, _ = self._train_doc(curr_doc, eval_mode=True)
            test_clusters = get_clusters(test_preds)

            # Save predicted clusters for this document id
            all_test_preds[curr_doc.doc_id] = test_clusters

            # input into metric functions should be formatted as dictionary of {int -> set(str)},
            # where keys (ints) are clusters and values (string sets) are mentions in a cluster. Example:
            # {
            #  1: {'rc_1', 'rc_2', ...}
            #  2: {'rc_5', 'rc_8', ...}
            #  3: ...
            # }

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

        avg_score = metrics.conll_12(muc_score, b3_score, ceaf_score)
        logging.info(f"----------------------------------------------")
        logging.info(f"**Test scores**")
        logging.info(f"**MUC:      {muc_score}**")
        logging.info(f"**BCubed:   {b3_score}**")
        logging.info(f"**CEAFe:    {ceaf_score}**")
        logging.info(f"**CoNLL-12: {avg_score}**")
        logging.info(f"----------------------------------------------")

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

        return {
            "muc": muc_score,
            "b3": b3_score,
            "ceafe": ceaf_score,
            "avg": avg_score
        }

    def visualize(self):
        build_and_display(self.path_pred_clusters, self.path_pred_scores, self.path_model_dir, display=False)


class NeuralCoreferencePairScorer(nn.Module):
    def __init__(self, num_features, hidden_size=150, dropout=0.2):
        # Note: num_features is either hidden_size of a LSTM or 2*hidden_size if using biLSTM
        super().__init__()

        # Attempts to model head word (root) in a mention, e.g. "model" in "my amazing model"
        self.attention_projector = nn.Linear(in_features=num_features, out_features=1)
        self.dropout = nn.Dropout(p=dropout)

        # Converts [candidate_state, head_state, candidate_state * head_state] into a score
        # self.fc = nn.Linear(in_features=(3 * num_features) * 3, out_features=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=((3 * num_features) * 3), out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, candidate_features, head_features,
                candidate_attention_mask=None,
                head_attention_mask=None):
        """

        Args:
            candidate_features: [B, num_tokens_cand, num_features]
            head_features: [B, num_tokens_head, num_features]
        """
        eff_cand_attn = candidate_attention_mask.bool() if candidate_attention_mask is not None \
            else torch.ones(candidate_features.shape[:2], dtype=torch.bool)
        eff_head_attn = head_attention_mask.bool() if head_attention_mask is not None \
            else torch.ones(head_features.shape[:2], dtype=torch.bool)

        candidate_features[torch.logical_not(eff_cand_attn)] = 0.0
        head_features[torch.logical_not(eff_head_attn)] = 0.0

        candidate_lengths = torch.sum(eff_cand_attn, dim=1)
        head_lengths = torch.sum(eff_head_attn, dim=1)

        batch_index = torch.arange(candidate_features.shape[0], device=candidate_features.device)

        # Create candidate representation
        candidate_attn_weights = F.softmax(self.attention_projector(self.dropout(candidate_features)),
                                           dim=1)
        cand_attended_features = torch.sum(candidate_attn_weights * candidate_features, dim=1)
        candidate_repr = torch.cat((candidate_features[:, 0],  # first word of mention
                                    candidate_features[batch_index, candidate_lengths - 1],  # last word of mention
                                    cand_attended_features), dim=1)

        # Create head mention representation
        head_attn_weights = F.softmax(self.attention_projector(self.dropout(head_features)),
                                      dim=1)
        head_attended_features = torch.sum(head_attn_weights * head_features, dim=1)
        head_repr = torch.cat((head_features[:, 0],  # first word of mention
                               head_features[batch_index, head_lengths - 1],  # last word of mention
                               head_attended_features), dim=1)

        # Combine representations and compute a score
        pair_score = self.fc(self.dropout(torch.cat((candidate_repr,
                                                     head_repr,
                                                     candidate_repr * head_repr), dim=1)))
        return pair_score
