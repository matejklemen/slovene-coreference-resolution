import os

from data import read_corpus
from utils import extract_vocab, encode, split_into_sets

import logging
import codecs
import torch
import torch.nn as nn
import torch.optim as optim
from scorer import NeuralCoreferencePairScorer


DATASET_NAME = 'coref149'  # Use 'coref149' or 'senticoref'
MODELS_SAVE_DIR = "baseline_model"
MAX_SEQ_LEN = 10
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
DROPOUT = 0.4  # higher value = stronger regularization
USE_PRETRAINED_EMBS = "word2vec"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NoncontextualController:
    def __init__(self, vocab, embedding_size, dropout, fc_hidden_size=150, learning_rate=0.001,
                 pretrained_embs=None, freeze_pretrained=False):
        self.vocab = vocab
        if pretrained_embs is not None:
            assert pretrained_embs.shape[1] == embedding_size
            logging.info(f"Using pretrained embeddings. freeze_pretrained = {freeze_pretrained}")
            self.embedder = nn.Embedding.from_pretrained(pretrained_embs, freeze=freeze_pretrained)
        else:
            logging.debug(f"Initializing random embeddings")
            self.embedder = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_size)

        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  hidden_size=fc_hidden_size,
                                                  dropout=dropout)

        # TODO: if freeze_pretrained=True, embedder params shouldn't be here
        self.scorer_optimizer = optim.Adam(list(self.scorer.parameters()) + list(self.embedder.parameters()) if not freeze_pretrained
                                           else self.scorer.parameters(),
                                           lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()

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
            encoded_doc.append(self.embedder(torch.tensor(curr_encoded_sent)))

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
                        mention_features.append(encoded_doc[curr_token.sentence_index][curr_token.position_in_sentence])
                    mention_features = torch.stack(mention_features, dim=0)  # shape: [num_tokens, embedding_size]

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
    embedding_size = 300

    documents = read_corpus(DATASET_NAME)
    train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

    tok2id, id2tok = extract_vocab(train_docs)

    pretrained_embs = None
    # Note: pretrained word2vec embeddings we use are (apparently) uncased
    if USE_PRETRAINED_EMBS == "word2vec":
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
    elif USE_PRETRAINED_EMBS == "fastText":
        import fasttext
        logging.info("Loading pretrained Slovene fastText embeddings")
        ft = fasttext.load_model(os.path.join("..", "data", "cc.sl.300.bin"))

        embedding_size = 300
        pretrained_embs = torch.rand((len(tok2id), embedding_size))
        for curr_token, curr_id in tok2id.items():
            pretrained_embs[curr_id, :] = torch.tensor(ft.get_word_vector(curr_token))

        del ft

    model = NoncontextualController(vocab=tok2id, embedding_size=embedding_size, dropout=DROPOUT,
                                    fc_hidden_size=512,
                                    learning_rate=LEARNING_RATE, pretrained_embs=pretrained_embs)
    model.train(epochs=NUM_EPOCHS, train_docs=train_docs, dev_docs=dev_docs)

