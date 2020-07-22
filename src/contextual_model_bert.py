import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer

from data import read_corpus
from common import ControllerBase, NeuralCoreferencePairScorer
from utils import get_clusters, split_into_sets, fixed_split

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_segment_size", type=int, default=256)
parser.add_argument("--dataset", type=str, default="coref149")
parser.add_argument("--pretrained_model_name_or_path", type=str, default=os.path.join("..", "data",
                                                                                      "slo-hr-en-bert-pytorch"))
parser.add_argument("--embedding_size", type=int, default=768)
parser.add_argument("--freeze_pretrained", action="store_true")
parser.add_argument("--fixed_split", action="store_true")


logging.basicConfig(level=logging.INFO)
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
        attn_weights = F.softmax(self.linear(hidden_states), dim=0)  # [num_layers, B, seq_len, 1]
        weighted_states = torch.sum(attn_weights * hidden_states, dim=0)  # [B, seq_len, 768]

        return weighted_states, attn_weights


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
    def __init__(self, embedding_size,
                 dropout,
                 pretrained_model_name_or_path,
                 dataset_name,
                 fc_hidden_size=150,
                 freeze_pretrained=True,
                 learning_rate=0.001,
                 max_segment_size=512,
                 max_span_size=10,
                 combine_layers=True,  # TODO: should be False by default
                 model_name=None):
        self.max_segment_size = max_segment_size - 2  # CLS, SEP
        self.max_span_size = max_span_size

        self.embedder = BertModel.from_pretrained(pretrained_model_name_or_path, output_hidden_states=combine_layers).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.combinator = WeightedLayerCombination(embedding_size=embedding_size)

        self.freeze_pretrained = freeze_pretrained
        for param in self.embedder.parameters():
            param.requires_grad = not freeze_pretrained

        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  dropout=dropout,
                                                  hidden_size=fc_hidden_size).to(DEVICE)

        params_to_update = list(self.scorer.parameters())
        if not freeze_pretrained:
            params_to_update += list(self.embedder.parameters())

        if combine_layers:
            params_to_update += list(self.combinator.parameters())

        self.optimizer = optim.Adam(self.scorer.parameters(),
                                    lr=learning_rate)

        super().__init__(learning_rate=learning_rate, dataset_name=dataset_name, model_name=model_name)
        logging.info(f"Initialized contextual BERT-based model with name {self.model_name}.")

    @property
    def model_base_dir(self):
        return "contextual_model_bert"

    def train_mode(self):
        if not self.freeze_pretrained:
            self.embedder.train()
        self.scorer.train()

    def eval_mode(self):
        if not self.freeze_pretrained:
            self.embedder.eval()
        self.scorer.eval()

    def load_checkpoint(self):
        path_to_model = os.path.join(self.path_model_dir, "best_scorer.th")
        if os.path.isfile(path_to_model):
            self.scorer.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
            self.loaded_from_file = True

        if not self.freeze_pretrained and os.path.exists(self.path_model_dir):
            self.embedder = BertModel.from_pretrained(self.path_model_dir).to(DEVICE)
            self.tokenizer = BertTokenizer.from_pretrained(self.path_model_dir)
            self.loaded_from_file = True

    def save_checkpoint(self):
        # TODO: save and load linear combination as well
        torch.save(self.scorer.state_dict(), os.path.join(self.path_model_dir, "best_scorer.th"))
        if not self.freeze_pretrained:
            self.embedder.save_pretrained(self.path_model_dir)
            self.tokenizer.save_pretrained(self.path_model_dir)

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        # maps from (idx_sent, idx_token) to (indices_in_tokenized_doc)
        tokenized_doc, mapping = prepare_document_bert(curr_doc, tokenizer=self.tokenizer)
        encoded_doc = self.tokenizer.convert_tokens_to_ids(tokenized_doc)
        #
        pad_embedding = self.embedder(torch.tensor([[self.tokenizer.pad_token_id]], device=DEVICE))
        pad_embedding, _ = self.combinator(torch.cat(pad_embedding[2]))
        #
        # pad_embedding = self.embedder(torch.tensor([[self.tokenizer.pad_token_id]], device=DEVICE))[0][0]  # shape: [1, 768]

        # Break down long documents into smaller sub-documents and encode them
        num_total_segments = (len(encoded_doc) + self.max_segment_size - 1) // self.max_segment_size
        doc_segments = []  # list of `num_total_segments` tensors of shape [self.max_segment_size + 2, 768]
        for idx_segment in range(num_total_segments):
            curr_segment = self.tokenizer.prepare_for_model(
                encoded_doc[idx_segment * self.max_segment_size: (idx_segment + 1) * self.max_segment_size],
                max_length=(self.max_segment_size + 2), pad_to_max_length=True, truncation=True,
                truncation_strategy="longest_first", return_tensors="pt").to(DEVICE)

            res = self.embedder(**curr_segment)
            combined_hidden_states, _ = self.combinator(torch.cat(res[2]))
            # last_hidden_states = combined_hidden_states
            # last_hidden_states = res[0]
            # Note: [0] because assuming a batch size of 1 (i.e. processing 1 segment at a time)
            # doc_segments.append(last_hidden_states[0])
            doc_segments.append(combined_hidden_states)

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
                        encoded_candidates = torch.stack(encoded_candidates,
                                                         dim=0)  # shape: [num_cands, self.max_span_size, 768]
                        head_features = torch.repeat_interleave(head_features, repeats=encoded_candidates.shape[0],
                                                                dim=0)  # shape: [num_candidates, num_tokens, 768]

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
                    mention_features = []
                    for curr_token in cand_mention.tokens:
                        positions_in_doc = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
                        for curr_position in positions_in_doc:
                            idx_segment = curr_position // (self.max_segment_size + 2)
                            idx_inside_segment = curr_position % (self.max_segment_size + 2)
                            mention_features.append(doc_segments[idx_segment][idx_inside_segment])
                    mention_features = torch.stack(mention_features, dim=0)  # shape: [num_tokens, num_features]

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
            self.optimizer.zero_grad()

        return preds, (float(doc_loss), n_examples)


if __name__ == "__main__":
    args = parser.parse_args()
    documents = read_corpus(args.dataset)

    if args.fixed_split:
        logging.info("Using fixed dataset split")
        train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
    else:
        train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

    controller = ContextualControllerBERT(model_name=args.model_name,
                                          embedding_size=args.embedding_size,
                                          fc_hidden_size=args.fc_hidden_size,
                                          dropout=args.dropout,
                                          pretrained_model_name_or_path=args.pretrained_model_name_or_path,
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



