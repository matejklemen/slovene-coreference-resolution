import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from transformers import BertModel, BertTokenizer

from common import ControllerBase, NeuralCoreferencePairScorer
from data import read_corpus, Document
from utils import split_into_sets, fixed_split, KFoldStateCache

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_segment_size", type=int, default=256)
parser.add_argument("--combine_layers", action="store_true",
                    help="Flag to determine if the sequence embeddings should be a learned combination of all "
                         "BERT hidden layers")
parser.add_argument("--dataset", type=str, default="coref149")
parser.add_argument("--pretrained_model_name_or_path", type=str, default="EMBEDDIA/crosloengual-bert")
parser.add_argument("--freeze_pretrained", action="store_true", help="If set, disable updates to BERT layers")
parser.add_argument("--random_seed", type=int, default=13)
parser.add_argument("--fixed_split", action="store_true")
parser.add_argument("--kfold_state_cache_path", type=str, default=None)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class WeightedLayerCombination(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.linear = nn.Linear(embedding_size, out_features=1)

    def forward(self, hidden_states):
        """ Args:
            hidden_states: shape [num_layers, B, seq_len, embedding_size]
        """
        attn_weights = torch.softmax(self.linear(hidden_states), dim=0)  # [num_layers, B, seq_len, 1]
        weighted_combination = torch.sum(attn_weights * hidden_states, dim=0)  # [B, seq_len, embedding_size]

        return weighted_combination, attn_weights


def prepare_document_bert(doc, tokenizer):
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


class ContextualControllerBERT(ControllerBase):
    def __init__(self,
                 dropout,
                 pretrained_model_name_or_path,
                 dataset_name,
                 fc_hidden_size=150,
                 freeze_pretrained=True,
                 learning_rate: float = 0.001,
                 layer_learning_rate: Optional[Dict[str, float]] = None,
                 max_segment_size=512,
                 max_span_size=10,
                 combine_layers=False,
                 model_name=None):
        self.dropout = dropout
        self.fc_hidden_size = fc_hidden_size
        self.freeze_pretrained = freeze_pretrained
        self.max_segment_size = max_segment_size - 3  # CLS, SEP, >= 1 PAD at the end (convention, for batching)
        self.max_span_size = max_span_size
        self.combine_layers = combine_layers
        self.learning_rate = learning_rate
        self.layer_learning_rate = layer_learning_rate if layer_learning_rate is not None else {}

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.embedder = BertModel.from_pretrained(pretrained_model_name_or_path,
                                                  output_hidden_states=combine_layers,
                                                  return_dict=True).to(DEVICE)
        for param in self.embedder.parameters():
            param.requires_grad = not self.freeze_pretrained

        embedding_size = self.embedder.config.hidden_size
        self.combinator = WeightedLayerCombination(embedding_size=embedding_size).to(DEVICE) \
            if self.combine_layers else None
        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  dropout=dropout,
                                                  hidden_size=fc_hidden_size).to(DEVICE)

        params_to_update = [{
                "params": self.scorer.parameters(),
                "lr": self.layer_learning_rate.get("lr_scorer", self.learning_rate)
        }]
        if not freeze_pretrained:
            params_to_update.append({
                "params": self.embedder.parameters(),
                "lr": self.layer_learning_rate.get("lr_embedder", self.learning_rate)
            })

        if self.combine_layers:
            params_to_update.append({
                "params": self.combinator.parameters(),
                "lr": self.layer_learning_rate.get("lr_combinator", self.learning_rate)
            })

        self.optimizer = optim.Adam(params_to_update, lr=self.learning_rate)

        super().__init__(learning_rate=self.learning_rate, dataset_name=dataset_name, model_name=model_name)
        logging.info(f"Initialized contextual BERT-based model with name {self.model_name}.")

    @property
    def model_base_dir(self):
        return "contextual_model_bert"

    def train_mode(self):
        if not self.freeze_pretrained:
            self.embedder.train()
        if self.combine_layers:
            self.combinator.train()
        self.scorer.train()

    def eval_mode(self):
        self.embedder.eval()
        if self.combine_layers:
            self.combinator.eval()
        self.scorer.eval()

    @staticmethod
    def from_pretrained(model_dir):
        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "r", encoding="utf-8") as f_config:
            pre_config = json.load(f_config)

        # If embeddings are not frozen, they are saved with the controller
        if not pre_config["freeze_pretrained"]:
            pre_config["pretrained_model_name_or_path"] = model_dir

        instance = ContextualControllerBERT(**pre_config)
        instance.path_model_dir = model_dir
        instance.load_checkpoint()

        return instance

    def save_pretrained(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write controller config (used for instantiation)
        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "w", encoding="utf-8") as f_config:
            json.dump({
                "dropout": self.dropout,
                "pretrained_model_name_or_path": self.pretrained_model_name_or_path if self.freeze_pretrained else model_dir,
                "dataset_name": self.dataset_name,
                "fc_hidden_size": self.fc_hidden_size,
                "freeze_pretrained": self.freeze_pretrained,
                "learning_rate": self.learning_rate,
                "layer_learning_rate": self.layer_learning_rate,
                "max_segment_size": self.max_segment_size,
                "max_span_size": self.max_span_size,
                "combine_layers": self.combine_layers,
                "model_name": self.model_name
            }, fp=f_config, indent=4)

        torch.save(self.scorer.state_dict(), os.path.join(self.path_model_dir, "scorer.th"))

        # Save fine-tuned BERT embeddings only if they're not frozen
        if not self.freeze_pretrained:
            self.embedder.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)

        if self.combine_layers:
            torch.save(self.combinator.state_dict(), os.path.join(model_dir, "combination.th"))

    def load_checkpoint(self):
        path_to_scorer = os.path.join(self.path_model_dir, "scorer.th")
        self.scorer.load_state_dict(torch.load(path_to_scorer, map_location=DEVICE))
        self.loaded_from_file = True

        if self.combine_layers:
            path_to_combination = os.path.join(self.path_model_dir, "combination.th")
            self.combinator.load_state_dict(torch.load(path_to_combination, map_location=DEVICE))
            self.loaded_from_file = True

    def save_checkpoint(self):
        logging.warning("save_checkpoint() is deprecated. Use save_pretrained() instead")
        self.save_pretrained(self.path_model_dir)

    def _prepare_doc(self, curr_doc: Document) -> Dict:
        """ Returns a cache dictionary with preprocessed data. This should only be called once per document, since
        data inside same document does not get shuffled. """
        ret = {}

        # maps from (idx_sent, idx_token) to (indices_in_tokenized_doc)
        tokenized_doc, mapping = prepare_document_bert(curr_doc, tokenizer=self.tokenizer)
        encoded_doc = self.tokenizer.convert_tokens_to_ids(tokenized_doc)

        num_total_segments = (len(tokenized_doc) + self.max_segment_size - 1) // self.max_segment_size
        segments = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        idx_token_to_segment = {}
        for idx_segment in range(num_total_segments):
            s_seg, e_seg = idx_segment * self.max_segment_size, (idx_segment + 1) * self.max_segment_size
            for idx_token in range(s_seg, e_seg):
                idx_token_to_segment[idx_token] = (idx_segment, 1 + idx_token - s_seg)  # +1 shift due to [CLS]

            curr_seg = self.tokenizer.prepare_for_model(ids=encoded_doc[s_seg: e_seg],
                                                        max_length=(self.max_segment_size + 2),
                                                        padding="max_length", truncation="longest_first",
                                                        return_token_type_ids=True, return_attention_mask=True)

            # Convention: add a PAD token at the end, so span padding points to that -> [CLS] <segment> [SEP] [PAD]
            segments["input_ids"].append(curr_seg["input_ids"] + [self.tokenizer.pad_token_id])
            segments["token_type_ids"].append(curr_seg["token_type_ids"] + [0])
            segments["attention_mask"].append(curr_seg["attention_mask"] + [0])

        # Shape: [num_segments, (max_segment_size + 3)]
        segments["input_ids"] = torch.tensor(segments["input_ids"])
        segments["token_type_ids"] = torch.tensor(segments["token_type_ids"])
        segments["attention_mask"] = torch.tensor(segments["attention_mask"])

        cluster_sets = []
        mention_to_cluster_id = {}
        for i, curr_cluster in enumerate(curr_doc.clusters):
            cluster_sets.append(set(curr_cluster))
            for mid in curr_cluster:
                mention_to_cluster_id[mid] = i

        all_candidate_data = []
        for idx_head, (head_id, head_mention) in enumerate(curr_doc.mentions.items(), start=1):
            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no data for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, candidate_data = [None], []
            candidate_attention = []
            correct_antecedents = []

            curr_head_data = [[], []]
            num_head_subwords = 0
            for curr_token in head_mention.tokens:
                indices_inside_document = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
                for _idx in indices_inside_document:
                    idx_segment, idx_inside_segment = idx_token_to_segment[_idx]
                    curr_head_data[0].append(idx_segment)
                    curr_head_data[1].append(idx_inside_segment)
                    num_head_subwords += 1

            if num_head_subwords > self.max_span_size:
                curr_head_data[0] = curr_head_data[0][:self.max_span_size]
                curr_head_data[1] = curr_head_data[1][:self.max_span_size]
            else:
                # padding tokens index into the PAD token of the last segment
                curr_head_data[0] += [curr_head_data[0][-1]] * (self.max_span_size - num_head_subwords)
                curr_head_data[1] += [-1] * (self.max_span_size - num_head_subwords)

            head_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
            head_attention[0, num_head_subwords:] = False

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), start=1):
                if idx_candidate >= idx_head:
                    break

                candidates.append(cand_id)

                # Maps tokens to positions inside segments (idx_seg, idx_inside_seg) for efficient indexing later
                curr_candidate_data = [[], []]
                num_candidate_subwords = 0
                for curr_token in cand_mention.tokens:
                    indices_inside_document = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
                    for _idx in indices_inside_document:
                        idx_segment, idx_inside_segment = idx_token_to_segment[_idx]
                        curr_candidate_data[0].append(idx_segment)
                        curr_candidate_data[1].append(idx_inside_segment)
                        num_candidate_subwords += 1

                if num_candidate_subwords > self.max_span_size:
                    curr_candidate_data[0] = curr_candidate_data[0][:self.max_span_size]
                    curr_candidate_data[1] = curr_candidate_data[1][:self.max_span_size]
                else:
                    # padding tokens index into the PAD token of the last segment
                    curr_candidate_data[0] += [curr_candidate_data[0][-1]] * (self.max_span_size - num_candidate_subwords)
                    curr_candidate_data[1] += [-1] * (self.max_span_size - num_candidate_subwords)

                candidate_data.append(curr_candidate_data)
                curr_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
                curr_attention[0, num_candidate_subwords:] = False
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

        ret["preprocessed_segments"] = segments
        ret["steps"] = all_candidate_data

        return ret

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        if not hasattr(curr_doc, "_cache_bert"):
            curr_doc._cache_bert = self._prepare_doc(curr_doc)
        cache = curr_doc._cache_bert  # type: Dict

        encoded_segments = cache["preprocessed_segments"]
        if self.freeze_pretrained:
            with torch.no_grad():
                embedded_segments = self.embedder(**{k: v.to(DEVICE) for k, v in encoded_segments.items()})
        else:
            embedded_segments = self.embedder(**{k: v.to(DEVICE) for k, v in encoded_segments.items()})

        # embedded_segments: [num_segments, max_segment_size + 3, embedding_size]
        if self.combine_layers:
            embedded_segments = torch.stack(embedded_segments["hidden_states"][-12:])
            embedded_segments, layer_weights = self.combinator(embedded_segments)
        else:
            embedded_segments = embedded_segments["last_hidden_state"]

        doc_loss, n_examples = 0.0, len(cache["steps"])
        preds = {}
        probs = {}

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
                curr_pred_prob = 1
            else:
                idx_segment = candidate_data[:, 0, :]
                idx_in_segment = candidate_data[:, 1, :]

                # [num_candidates, max_span_size, embedding_size]
                candidate_data = embedded_segments[idx_segment, idx_in_segment]
                # [1, head_size, embedding_size]
                head_data = embedded_segments[head_data[:, 0, :], head_data[:, 1, :]]
                head_data = head_data.repeat((num_candidates - 1, 1, 1))

                candidate_scores = self.scorer(candidate_data, head_data,
                                               curr_step["candidate_attention"],
                                               curr_step["head_attention"].repeat((num_candidates - 1, 1)))

                # [1, num_candidates]
                candidate_scores = torch.cat((torch.tensor([0.0], device=DEVICE),
                                              candidate_scores.flatten())).unsqueeze(0)

                candidate_probabilities = torch.softmax(candidate_scores, dim=-1)
                curr_pred_prob = torch.max(candidate_probabilities).item()

                curr_pred = torch.argmax(candidate_scores)
                doc_loss += self.loss(candidate_scores.repeat((len(correct_antecedents), 1)),
                                      torch.tensor(correct_antecedents, device=DEVICE))

            # { antecedent: [mention(s)] } pair
            existing_refs = preds.get(candidates[int(curr_pred)], [])
            existing_refs.append(head_id)
            preds[candidates[int(curr_pred)]] = existing_refs

            # { mention: probability } pair
            probs[head_id] = curr_pred_prob

        if not eval_mode:
            doc_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return preds, (float(doc_loss), n_examples), probs


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    args = parser.parse_args()
    documents = read_corpus(args.dataset)

    def create_model_instance(model_name, **override_kwargs):
        return ContextualControllerBERT(model_name=model_name,
                                        fc_hidden_size=override_kwargs.get("fc_hidden_size", args.fc_hidden_size),
                                        dropout=override_kwargs.get("dropout", args.dropout),
                                        combine_layers=override_kwargs.get("combine_layers", args.combine_layers),
                                        pretrained_model_name_or_path=override_kwargs.get("pretrained_model_name_or_path",
                                                                                          args.pretrained_model_name_or_path),
                                        learning_rate=override_kwargs.get("learning_rate", args.learning_rate),
                                        layer_learning_rate={"lr_embedder": 2e-5} if not args.freeze_pretrained else None,
                                        max_segment_size=override_kwargs.get("max_segment_size", args.max_segment_size),
                                        dataset_name=override_kwargs.get("dataset", args.dataset),
                                        freeze_pretrained=override_kwargs.get("freeze_pretrained", args.freeze_pretrained))

    # Train model
    if args.dataset == "coref149":
        INNER_K, OUTER_K = 3, 10
        logging.info(f"Performing {OUTER_K}-fold (outer) and {INNER_K}-fold (inner) CV...")
        save_path = "cache_run_contextual_bert_coref149.json"
        if args.kfold_state_cache_path is None:
            train_test_folds = KFold(n_splits=OUTER_K, shuffle=True).split(documents)
            train_test_folds = [{
                "train_docs": [documents[_i].doc_id for _i in train_dev_index],
                "test_docs": [documents[_i].doc_id for _i in test_index]
            } for train_dev_index, test_index in train_test_folds]

            fold_cache = KFoldStateCache(script_name="contextual_model_bert.py",
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
            curr_model = ContextualControllerBERT.from_pretrained(best_name)
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
            model = ContextualControllerBERT.from_pretrained(model.path_model_dir)

        model.evaluate(test_docs)
        model.visualize()



