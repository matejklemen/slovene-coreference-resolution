import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import argparse

from transformers import BertModel, BertTokenizer
from data import read_corpus
from utils import split_into_sets
from scorer import NeuralCoreferencePairScorer


parser = argparse.ArgumentParser()
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_segment_size", type=int, default=256)
parser.add_argument("--dataset", type=str, default="coref149")
parser.add_argument("--bert_pretrained_name_or_dir", type=str, default=None)


CUSTOM_PRETRAINED_BERT_DIR = os.environ.get("CUSTOM_PRETRAINED_BERT_DIR",
                                            os.path.join("..", "data", "slo-hr-en-bert-pytorch"))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    def __init__(self, embedding_size, dropout, pretrained_embs_dir, fc_hidden_size=150, freeze_pretrained=True,
                 learning_rate=0.001, max_segment_size=(512 - 2)):
        logging.info(f"Using device {DEVICE}")
        self.max_segment_size = max_segment_size
        self.embedder = BertModel.from_pretrained(pretrained_embs_dir).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_embs_dir)
        for param in self.embedder.parameters():
            param.requires_grad = False

        if not freeze_pretrained:
            # TODO: enable unfreezing BERT
            raise NotImplementedError("The current implementation does not allow unfreezing BERT weights")

        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  dropout=dropout,
                                                  hidden_size=fc_hidden_size).to(DEVICE)
        self.scorer_optimizer = optim.Adam(self.scorer.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        self.scorer.zero_grad()

        # maps from (idx_sent, idx_token) to (indices_in_tokenized_doc)
        tokenized_doc, mapping = prepare_document(curr_doc, tokenizer=self.tokenizer)
        encoded_doc = self.tokenizer.convert_tokens_to_ids(tokenized_doc)

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
            head_features = torch.stack(head_features, dim=0)  # shape: [num_tokens, embedding_size]

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), 1):
                if cand_id != head_id and cand_id in gt_antecedent_ids:
                    gt_antecedents.append(idx_candidate)

                # Obtain scores for candidates and select best one as antecedent
                if idx_candidate == idx_head:
                    if len(encoded_candidates) > 0:
                        cand_scores = [torch.tensor([0.0], device=DEVICE)]
                        for candidate_features in encoded_candidates:
                            cand_scores.append(self.scorer(candidate_features, head_features))

                        assert len(cand_scores) == len(candidates)

                        # Concatenates the given sequence of seq tensors in the given dimension
                        cand_scores = torch.stack(cand_scores, dim=1)

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
                    mention_features = torch.stack(mention_features, dim=0)
                    encoded_candidates.append(mention_features)

        if not eval_mode:
            doc_loss.backward()
            self.scorer_optimizer.step()

        return preds, (float(doc_loss), n_examples)

    def train(self, epochs, train_docs, dev_docs):
        best_dev_loss = float("inf")
        logging.info("Starting training...")
        for idx_epoch in range(epochs):
            logging.info(f"\tRunning epoch {idx_epoch + 1}/{epochs}")

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

            logging.info(f"\t\tTraining loss: {train_loss / max(1, train_examples): .4f}")
            logging.info(f"\t\tDev loss:      {dev_loss / max(1, dev_examples): .4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    documents = read_corpus(args.dataset)
    train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)
    used_bert = args.bert_pretrained_name_or_dir
    if used_bert is None:
        used_bert = CUSTOM_PRETRAINED_BERT_DIR

    controller = ContextualControllerBERT(embedding_size=768,
                                          fc_hidden_size=args.fc_hidden_size,
                                          dropout=args.dropout,
                                          pretrained_embs_dir=used_bert,
                                          learning_rate=args.learning_rate,
                                          max_segment_size=args.max_segment_size)
    controller.train(epochs=args.num_epochs, train_docs=train_docs, dev_docs=dev_docs)



