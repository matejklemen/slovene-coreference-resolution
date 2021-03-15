import argparse
import json
import logging
import os
import sys
from itertools import chain
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.model_selection import KFold

from common import ControllerBase, NeuralCoreferencePairScorer
from utils import split_into_sets, fixed_split, KFoldStateCache

from data import read_corpus, Document

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_segment_size", type=int, default=None,
                    help="Size of nonoverlapping segments into which a document will be split, with each segment being "
                         "processed independently. By default, a segment corresponds to a single sentence.")
parser.add_argument("--dataset", type=str, default="coref149")
parser.add_argument("--random_seed", type=int, default=13)
parser.add_argument("--freeze_pretrained", action="store_true")
parser.add_argument("--fixed_split", action="store_true")
parser.add_argument("--kfold_state_cache_path", type=str, default=None)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ContextualControllerELMo(ControllerBase):
    def __init__(self,
                 hidden_size,
                 dropout,
                 pretrained_embeddings_dir,
                 dataset_name,
                 fc_hidden_size=150,
                 freeze_pretrained=True,
                 learning_rate=0.001,
                 layer_learning_rate: Optional[Dict[str, float]] = None,
                 max_segment_size=None,  # if None, process sentences independently
                 max_span_size=10,
                 model_name=None):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.freeze_pretrained = freeze_pretrained
        self.fc_hidden_size = fc_hidden_size
        self.max_span_size = max_span_size
        self.max_segment_size = max_segment_size
        self.learning_rate = learning_rate
        self.layer_learning_rate = layer_learning_rate if layer_learning_rate is not None else {}

        self.pretrained_embeddings_dir = pretrained_embeddings_dir
        self.embedder = Elmo(options_file=os.path.join(pretrained_embeddings_dir, "options.json"),
                             weight_file=os.path.join(pretrained_embeddings_dir, "slovenian-elmo-weights.hdf5"),
                             dropout=(0.0 if freeze_pretrained else dropout),
                             num_output_representations=1,
                             requires_grad=(not freeze_pretrained)).to(DEVICE)
        embedding_size = self.embedder.get_output_dim()

        self.context_encoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                       batch_first=True, bidirectional=True).to(DEVICE)
        self.scorer = NeuralCoreferencePairScorer(num_features=(2 * hidden_size),
                                                  hidden_size=fc_hidden_size,
                                                  dropout=dropout).to(DEVICE)
        params_to_update = [
            {
                "params": self.scorer.parameters(),
                "lr": self.layer_learning_rate.get("lr_scorer", self.learning_rate)
            },
            {
                "params": self.context_encoder.parameters(),
                "lr": self.layer_learning_rate.get("lr_context_encoder", self.learning_rate)
            }
        ]
        if not freeze_pretrained:
            params_to_update.append({
                "params": self.embedder.parameters(),
                "lr": self.layer_learning_rate.get("lr_embedder", self.learning_rate)
            })

        self.optimizer = optim.Adam(params_to_update, lr=self.learning_rate)

        super().__init__(learning_rate=learning_rate, dataset_name=dataset_name, model_name=model_name)
        logging.info(f"Initialized contextual ELMo-based model with name {self.model_name}.")

    @property
    def model_base_dir(self):
        return "contextual_model_elmo"

    def train_mode(self):
        if not self.freeze_pretrained:
            self.embedder.train()
        self.context_encoder.train()
        self.scorer.train()

    def eval_mode(self):
        self.embedder.eval()
        self.context_encoder.eval()
        self.scorer.eval()

    def load_checkpoint(self):
        self.loaded_from_file = True
        self.context_encoder.load_state_dict(torch.load(os.path.join(self.path_model_dir, "context_encoder.th"),
                                                        map_location=DEVICE))
        self.scorer.load_state_dict(torch.load(os.path.join(self.path_model_dir, "scorer.th"),
                                               map_location=DEVICE))

        path_to_embeddings = os.path.join(self.path_model_dir, "embeddings.th")
        if os.path.isfile(path_to_embeddings):
            logging.info(f"Loading fine-tuned ELMo weights from '{path_to_embeddings}'")
            self.embedder.load_state_dict(torch.load(path_to_embeddings, map_location=DEVICE))

    @staticmethod
    def from_pretrained(model_dir):
        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "r", encoding="utf-8") as f_config:
            pre_config = json.load(f_config)

        instance = ContextualControllerELMo(**pre_config)
        instance.load_checkpoint()

        return instance

    def save_pretrained(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write controller config (used for instantiation)
        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "w", encoding="utf-8") as f_config:
            json.dump({
                "hidden_size": self.hidden_size,
                "dropout": self.dropout,
                "pretrained_embeddings_dir": self.pretrained_embeddings_dir,
                "dataset_name": self.dataset_name,
                "fc_hidden_size": self.fc_hidden_size,
                "freeze_pretrained": self.freeze_pretrained,
                "learning_rate": self.learning_rate,
                "layer_learning_rate": self.layer_learning_rate,
                "max_segment_size": self.max_segment_size,
                "max_span_size": self.max_span_size,
                "model_name": self.model_name
            }, fp=f_config, indent=4)

        torch.save(self.context_encoder.state_dict(), os.path.join(self.path_model_dir, "context_encoder.th"))
        torch.save(self.scorer.state_dict(), os.path.join(self.path_model_dir, "scorer.th"))

        # Save fine-tuned ELMo embeddings only if they're not frozen
        if not self.freeze_pretrained:
            torch.save(self.embedder.state_dict(), os.path.join(self.path_model_dir, "embeddings.th"))

    def save_checkpoint(self):
        logging.warning("save_checkpoint() is deprecated. Use save_pretrained() instead")
        self.save_pretrained(self.path_model_dir)

    def _prepare_doc(self, curr_doc: Document) -> Dict:
        """ Returns a cache dictionary with preprocessed data. This should only be called once per document, since
        data inside same document does not get shuffled. """
        ret = {}

        # By default, each sentence is its own segment, meaning sentences are processed independently
        if self.max_segment_size is None:
            def get_position(t):
                return t.sentence_index, t.position_in_sentence

            _encoded_segments = batch_to_ids(curr_doc.raw_sentences())
        # Optionally, one can specify max_segment_size, in which case segments of tokens are processed independently
        else:
            def get_position(t):
                doc_position = t.position_in_document
                return doc_position // self.max_segment_size, doc_position % self.max_segment_size

            flattened_doc = list(chain(*curr_doc.raw_sentences()))
            num_segments = (len(flattened_doc) + self.max_segment_size - 1) // self.max_segment_size
            _encoded_segments = \
                batch_to_ids([flattened_doc[idx_seg * self.max_segment_size: (idx_seg + 1) * self.max_segment_size]
                              for idx_seg in range(num_segments)])

        encoded_segments = []
        # Convention: Add a PAD word ([0] * max_chars vector) at the end of each segment, for padding mentions
        for curr_sent in _encoded_segments:
            encoded_segments.append(
                torch.cat((curr_sent, torch.zeros((1, ELMoCharacterMapper.max_word_length), dtype=torch.long)))
            )
        encoded_segments = torch.stack(encoded_segments)

        cluster_sets = []
        mention_to_cluster_id = {}
        for i, curr_cluster in enumerate(curr_doc.clusters):
            cluster_sets.append(set(curr_cluster))
            for mid in curr_cluster:
                mention_to_cluster_id[mid] = i

        all_candidate_data = []
        for idx_head, (head_id, head_mention) in enumerate(curr_doc.mentions.items(), 1):
            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no data for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, candidate_data = [None], []
            candidate_attention = []
            correct_antecedents = []

            curr_head_data = [[], []]
            num_head_words = 0
            for curr_token in head_mention.tokens:
                idx_segment, idx_inside_segment = get_position(curr_token)
                curr_head_data[0].append(idx_segment)
                curr_head_data[1].append(idx_inside_segment)
                num_head_words += 1

            if num_head_words > self.max_span_size:
                curr_head_data[0] = curr_head_data[0][:self.max_span_size]
                curr_head_data[1] = curr_head_data[1][:self.max_span_size]
            else:
                curr_head_data[0] += [curr_head_data[0][-1]] * (self.max_span_size - num_head_words)
                curr_head_data[1] += [-1] * (self.max_span_size - num_head_words)

            head_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
            head_attention[0, num_head_words:] = False

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), start=1):
                if idx_candidate >= idx_head:
                    break

                candidates.append(cand_id)

                # Maps tokens to positions inside segments (idx_seg, idx_inside_seg) for efficient indexing later
                curr_candidate_data = [[], []]
                num_candidate_words = 0
                for curr_token in cand_mention.tokens:
                    idx_segment, idx_inside_segment = get_position(curr_token)
                    curr_candidate_data[0].append(idx_segment)
                    curr_candidate_data[1].append(idx_inside_segment)
                    num_candidate_words += 1

                if num_candidate_words > self.max_span_size:
                    curr_candidate_data[0] = curr_candidate_data[0][:self.max_span_size]
                    curr_candidate_data[1] = curr_candidate_data[1][:self.max_span_size]
                else:
                    # padding tokens index into the PAD token of the last segment
                    curr_candidate_data[0] += [curr_candidate_data[0][-1]] * (self.max_span_size - num_candidate_words)
                    curr_candidate_data[1] += [-1] * (self.max_span_size - num_candidate_words)

                candidate_data.append(curr_candidate_data)
                curr_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
                curr_attention[0, num_candidate_words:] = False
                candidate_attention.append(curr_attention)

                is_coreferent = cand_id in gt_antecedent_ids
                if is_coreferent:
                    correct_antecedents.append(idx_candidate)

            if len(correct_antecedents) == 0:
                correct_antecedents.append(0)

            candidate_attention = torch.cat(candidate_attention) if len(candidate_attention) > 0 else []
            all_candidate_data.append({
                "head_id": head_id,
                "head_data": torch.tensor([curr_head_data]),
                "head_attention": head_attention,
                "candidates": candidates,
                "candidate_data": torch.tensor(candidate_data),
                "candidate_attention": candidate_attention,
                "correct_antecedents": correct_antecedents
            })

        ret["preprocessed_segments"] = encoded_segments
        ret["steps"] = all_candidate_data

        return ret

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        if not hasattr(curr_doc, "_cache_elmo"):
            curr_doc._cache_elmo = self._prepare_doc(curr_doc)
        cache = curr_doc._cache_elmo  # type: Dict

        encoded_segments = cache["preprocessed_segments"]
        if self.freeze_pretrained:
            with torch.no_grad():
                res = self.embedder(encoded_segments.to(DEVICE))
        else:
            res = self.embedder(encoded_segments.to(DEVICE))

        # Note: max_segment_size is either specified at instantiation or (the length of longest sentence + 1)
        embedded_segments = res["elmo_representations"][0]  # [num_segments, max_segment_size, embedding_size]
        (lstm_segments, _) = self.context_encoder(embedded_segments)  # [num_segments, max_segment_size, 2 * hidden_size]

        doc_loss, n_examples = 0.0, len(cache["steps"])
        preds = {}

        for curr_step in cache["steps"]:
            head_id = curr_step["head_id"]
            head_data = curr_step["head_data"]

            candidates = curr_step["candidates"]
            candidate_data = curr_step["candidate_data"]
            correct_antecedents = curr_step["correct_antecedents"]

            # Note: num_candidates includes dummy antecedent + actual candidates
            num_candidates = len(candidates)
            if num_candidates == 1:
                curr_pred = 0
            else:
                idx_segment = candidate_data[:, 0, :]
                idx_in_segment = candidate_data[:, 1, :]

                # [num_candidates, max_span_size, embedding_size]
                candidate_data = lstm_segments[idx_segment, idx_in_segment]
                # [1, head_size, embedding_size]
                head_data = lstm_segments[head_data[:, 0, :], head_data[:, 1, :]]
                head_data = head_data.repeat((num_candidates - 1, 1, 1))

                candidate_scores = self.scorer(candidate_data, head_data,
                                               curr_step["candidate_attention"],
                                               curr_step["head_attention"].repeat((num_candidates - 1, 1)))

                # [1, num_candidates]
                candidate_scores = torch.cat((torch.tensor([0.0], device=DEVICE),
                                              candidate_scores.flatten())).unsqueeze(0)

                curr_pred = torch.argmax(candidate_scores)
                doc_loss += self.loss(candidate_scores.repeat((len(correct_antecedents), 1)),
                                      torch.tensor(correct_antecedents, device=DEVICE))

            # { antecedent: [mention(s)] } pair
            existing_refs = preds.get(candidates[int(curr_pred)], [])
            existing_refs.append(head_id)
            preds[candidates[int(curr_pred)]] = existing_refs

        if not eval_mode:
            doc_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return preds, (float(doc_loss), n_examples)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    args = parser.parse_args()
    if args.random_seed:
        np.random.seed(args.random_seed)
        torch.random.manual_seed(args.random_seed)

    documents = read_corpus(args.dataset)

    def create_model_instance(model_name, **override_kwargs):
        return ContextualControllerELMo(model_name=model_name,
                                        fc_hidden_size=override_kwargs.get("fc_hidden_size", args.fc_hidden_size),
                                        hidden_size=override_kwargs.get("hidden_size", args.hidden_size),
                                        dropout=override_kwargs.get("dropout", args.dropout),
                                        pretrained_embeddings_dir="../data/slovenian-elmo",
                                        freeze_pretrained=override_kwargs.get("freeze_pretrained", args.freeze_pretrained),
                                        learning_rate=override_kwargs.get("learning_rate", args.learning_rate),
                                        max_segment_size=override_kwargs.get("max_segment_size", args.max_segment_size),
                                        layer_learning_rate={
                                            "lr_embedder": 10e-5} if not args.freeze_pretrained else None,
                                        dataset_name=override_kwargs.get("dataset", args.dataset))

    # Train model
    if args.dataset == "coref149":
        INNER_K, OUTER_K = 3, 10
        logging.info(f"Performing {OUTER_K}-fold (outer) and {INNER_K}-fold (inner) CV...")
        save_path = "cache_run_contextual_elmo_coref149.json"
        if args.kfold_state_cache_path is None:
            train_test_folds = KFold(n_splits=OUTER_K, shuffle=True).split(documents)
            train_test_folds = [{
                "train_docs": [documents[_i].doc_id for _i in train_dev_index],
                "test_docs": [documents[_i].doc_id for _i in test_index]
            } for train_dev_index, test_index in train_test_folds]

            fold_cache = KFoldStateCache(script_name="contextual_model_elmo.py",
                                         script_args=vars(args),
                                         main_dataset=args.dataset,
                                         additional_dataset=None,
                                         fold_info=train_test_folds)
        else:
            fold_cache = KFoldStateCache.from_file(args.kfold_state_cache_path)
            OUTER_K = fold_cache.num_folds

        for curr_fold_data in fold_cache.get_next_unfinished():
            curr_train_dev_docs = list(filter(lambda doc: doc.doc_id in set(curr_fold_data["train_docs"]), documents))
            curr_test_docs = list(filter(lambda doc: doc.doc_id in set(curr_fold_data["test_docs"]), documents))
            logging.info(f"Fold#{curr_fold_data['idx_fold']}...")

            best_metric, best_name = float("inf"), None
            for idx_inner_fold, (train_index, dev_index) in enumerate(KFold(n_splits=INNER_K).split(curr_train_dev_docs)):
                curr_train_docs = [curr_train_dev_docs[_i] for _i in train_index]
                curr_dev_docs = [curr_train_dev_docs[_i] for _i in dev_index]

                curr_model = create_model_instance(model_name=f"fold{curr_fold_data['idx_fold']}_{idx_inner_fold}")
                dev_loss = curr_model.train(epochs=args.num_epochs, train_docs=curr_train_docs, dev_docs=curr_dev_docs)
                logging.info(f"Fold {curr_fold_data['idx_fold']}-{idx_inner_fold}: {dev_loss: .5f}")

                if dev_loss < best_metric:
                    best_metric = dev_loss
                    best_name = curr_model.path_model_dir

            logging.info(f"Best model: {best_name}, best loss: {best_metric: .5f}")
            curr_model = ContextualControllerELMo.from_pretrained(best_name)
            curr_test_metrics = curr_model.evaluate(curr_test_docs)
            curr_model.visualize()

            curr_test_metrics_expanded = {}
            for metric, metric_value in curr_test_metrics.items():
                curr_test_metrics_expanded[f"{metric}_p"] = float(metric_value.precision())
                curr_test_metrics_expanded[f"{metric}_r"] = float(metric_value.recall())
                curr_test_metrics_expanded[f"{metric}_f1"] = float(metric_value.f1())
            fold_cache.add_results(idx_fold=curr_fold_data["idx_fold"], results=curr_test_metrics_expanded)
            fold_cache.save(save_path)

        logging.info(f"Final scores (over {OUTER_K} folds)")
        aggregated_metrics = {}
        for curr_fold_data in fold_cache.fold_info:
            for metric, metric_value in curr_fold_data["results"].items():
                existing = aggregated_metrics.get(metric, [])
                existing.append(metric_value)

                aggregated_metrics[metric] = existing

        for metric, metric_values in aggregated_metrics.items():
            logging.info(f"- {metric}: mean={np.mean(metric_values): .4f} +- sd={np.std(metric_values): .4f}\n"
                         f"\t all fold scores: {metric_values}")
    else:
        logging.info(f"Using single train/dev/test split...")
        if args.fixed_split:
            logging.info("Using fixed dataset split")
            train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
        else:
            train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

        model = create_model_instance(model_name=args.model_name)

        if not model.loaded_from_file:
            model.train(epochs=args.num_epochs, train_docs=train_docs, dev_docs=dev_docs)
            # Reload best checkpoint
            model = ContextualControllerELMo.from_pretrained(model.path_model_dir)

        model.evaluate(test_docs)
        model.visualize()
