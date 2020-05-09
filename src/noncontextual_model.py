from data import read_corpus
from utils import extract_vocab, encode

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


DATASET_NAME = 'coref149'  # Use 'coref149' or 'senticoref'
MODELS_SAVE_DIR = "baseline_model"
MAX_SEQ_LEN = 10
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
DROPOUT = 0.4  # higher value = stronger regularization
EMBEDDING_SIZE = 300
USE_PRETRAINED_EMBS = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class NoncontextualScorer(nn.Module):
    def __init__(self, vocab_size, pad_id, dropout, pretrained_embs=None, freeze_pretrained=False):
        super().__init__()

        if pretrained_embs is not None:
            assert pretrained_embs.shape[1] == EMBEDDING_SIZE
            logger.info(f"Using pretrained embeddings. freeze_pretrained = {freeze_pretrained}")
            self.embedder = nn.Embedding.from_pretrained(pretrained_embs, freeze=freeze_pretrained)
        else:
            logger.debug(f"Initializing random embeddings")
            self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE)

        self.fc = nn.Linear(in_features=(EMBEDDING_SIZE + EMBEDDING_SIZE), out_features=1)
        self.pad_id = pad_id
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, candidates, head_mentions):
        """ Average embeddings for each sequence, then concatenate the average vectors and put through fully connected
        layer.

        candidate, head_mention: [num_candidates, max_seq_length] tensors
        """
        cand_embs = self.dropout(self.embedder(candidates))
        cand_mask = (candidates != self.pad_id).unsqueeze(-1)
        masked_cand_embs = cand_mask * cand_embs
        cand_embs = torch.mean(masked_cand_embs, dim=1) / torch.sum(cand_mask, dim=1)

        head_embs = self.dropout(self.embedder(head_mentions))
        head_mask = (head_mentions != self.pad_id).unsqueeze(-1)
        masked_head_embs = head_mask * head_embs
        head_embs = torch.mean(masked_head_embs, dim=1) / torch.sum(head_mask, dim=1)

        combined = torch.cat((cand_embs, head_embs), dim=1)
        scores = self.fc(combined)
        return scores


class NoncontextualController:
    def __init__(self, vocab, dropout, pretrained_embs=None, freeze_pretrained=False):
        self.vocab = vocab
        self.model = NoncontextualScorer(vocab_size=len(vocab), dropout=dropout, pad_id=vocab["<PAD>"],
                                         pretrained_embs=pretrained_embs, freeze_pretrained=freeze_pretrained)

        self.model_optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.CrossEntropyLoss()

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

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
            self.model.zero_grad()

            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no features for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, encoded_candidates = [None], []
            gt_antecedents = []

            head_tokens = encode([t.raw_text for t in head_mention.tokens],
                                 vocab=self.vocab, max_seq_len=MAX_SEQ_LEN)

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), 1):
                if cand_id != head_id and cand_id in gt_antecedent_ids:
                    gt_antecedents.append(idx_candidate)

                # Obtain scores for candidates and select best one as antecedent
                if idx_candidate == idx_head:
                    if len(encoded_candidates) > 0:
                        encoded_candidates = torch.tensor(encoded_candidates, dtype=torch.long)
                        encoded_head = torch.repeat_interleave(torch.tensor([head_tokens]),
                                                               repeats=len(encoded_candidates), dim=0)
                        cand_scores = self.model(encoded_candidates, encoded_head)

                        # Concatenates the given sequence of seq tensors in the given dimension
                        cand_scores = torch.cat((torch.tensor([0.]), cand_scores.flatten())).unsqueeze(0)

                        # if no other antecedent exists for mention, then it's a first mention (GT is dummy antecedent)
                        if len(gt_antecedents) == 0:
                            gt_antecedents.append(0)

                        # Get index of max value. That index equals to mention at that place
                        curr_pred = torch.argmax(cand_scores)

                        # (average) loss over all ground truth antecedents
                        curr_loss = self.loss(torch.repeat_interleave(cand_scores, repeats=len(gt_antecedents), dim=0),
                                              torch.tensor(gt_antecedents))

                        doc_loss += float(curr_loss)

                        n_examples += 1

                        if not eval_mode:
                            curr_loss.backward()
                            self.model_optimizer.step()
                            self.model_optimizer.zero_grad()
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
                    mention_tokens = encode([t.raw_text for t in cand_mention.tokens],
                                            vocab=self.vocab, max_seq_len=MAX_SEQ_LEN)
                    encoded_candidates.append(mention_tokens)

        return preds, (doc_loss, n_examples)

    def train(self, epochs, train_docs, dev_docs):
        best_dev_loss = float("inf")
        logging.info("Starting training...")
        for idx_epoch in range(epochs):
            logging.info(f"\tRunning epoch {idx_epoch + 1}/{epochs}")

            # Make permutation of train documents
            shuffle_indices = torch.randperm(len(train_docs))

            logging.debug("\t\tModel training step")
            self.model.train()
            train_loss, train_examples = 0.0, 0
            for idx_doc in shuffle_indices:
                curr_doc = train_docs[idx_doc]

                _, (doc_loss, n_examples) = self._train_doc(curr_doc)

                train_loss += doc_loss
                train_examples += n_examples

            logging.debug("\t\tModel validation step")
            self.model.eval()
            dev_loss, dev_examples = 0.0, 0
            for curr_doc in dev_docs:
                _, (doc_loss, n_examples) = self._train_doc(curr_doc, eval_mode=True)

                dev_loss += doc_loss
                dev_examples += n_examples

            logging.info(f"\t\tTraining loss: {train_loss / max(1, train_examples): .4f}")
            logging.info(f"\t\tDev loss:      {dev_loss / max(1, dev_examples): .4f}")


if __name__ == "__main__":
    documents = read_corpus(DATASET_NAME)
    idx = np.arange(len(documents))
    np.random.shuffle(idx)
    # TODO: replace this with a call to split function (which needs to be moved from baseline into utils)
    train_docs = np.take(documents, idx[: -40])
    dev_docs = np.take(documents, idx[-40: -20])

    tok2id, id2tok = extract_vocab(train_docs)

    pretrained_embs = None
    if USE_PRETRAINED_EMBS:
        import fasttext
        logger.info("Loading pretrained Slovene fastText embeddings")
        # Load pre-trained fastText vectors and use them to initialize embeddings in our model
        ft = fasttext.load_model('../data/cc.sl.300.bin')
        vocab_embs = torch.tensor([ft.get_word_vector(curr_token) for curr_token in tok2id])
        del ft

    model = NoncontextualController(vocab=tok2id, dropout=DROPOUT, pretrained_embs=pretrained_embs)
    model.train(epochs=NUM_EPOCHS, train_docs=train_docs, dev_docs=dev_docs)

