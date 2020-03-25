import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import shuffle

from data import read_corpus
from utils import extract_vocab, mention_dataset
from model import SimpleMentionScorer, SimpleCorefScorer

np.random.seed(1)
torch.manual_seed(1)

DUMMY_ANTECEDENT, DUMMY_IDX = "<DUMMY>", 0


# Turns list of document sentences into list of document tokens
def flatten_doc(sents_list):
    flat = []
    for sent in sents_list:
        flat.extend(sent)
    return flat


# TODO: first train a mention detector
# TODO: reference to tok2id is global
def train_epoch(train_docs, mention_scorer, pair_scorer, mention_opt, pair_opt, loss_fn):
    SAMPLE_SIZE = 1
    # Single epoch through the training set
    for curr_doc in train_docs:
        mention_opt.zero_grad()
        mapped_clusters = curr_doc.mapped_clusters
        # Skip documents that have no golden mentions
        if not curr_doc.mentions:
            continue

        # Get all mention candidates (up to MAX_SPAN_WIDTH)
        all_candidates, all_labels = [], []
        for curr_width in range(1, 1 + MAX_SPAN_WIDTH):
            curr_cands, curr_labels = mention_dataset(curr_doc, width=curr_width)
            all_candidates.extend(curr_cands)
            all_labels.extend(curr_labels)

        # Sort candidates by start of interval and end of interval (tiebreaker)
        sorted_candidates = sorted(enumerate(all_candidates),
                                   key=lambda idx_and_span: (idx_and_span[1][0], idx_and_span[1][1]))
        sort_indices = list(map(lambda tup: tup[0], sorted_candidates))
        all_candidates = np.take(all_candidates, sort_indices, axis=0)
        all_labels = np.take(all_labels, sort_indices)

        indices = np.arange(all_candidates.shape[0])
        pos_examples = indices[all_labels == 1]
        neg_examples = np.random.choice(indices[all_labels == 0], size=pos_examples.shape[0], replace=False)

        curr_loss = 0.0
        curr_flatdoc = flatten_doc(curr_doc.raw_sentences())
        for idx_ex in range(pos_examples.shape[0]):
            start_pos, end_pos = all_candidates[pos_examples[idx_ex]]
            mention = curr_flatdoc[start_pos: end_pos]
            print_score = False
            if mention[0] == "Milovanović":
                print_score = True
            mention = torch.tensor([[tok2id[t.lower()] for t in mention]])
            pos_score = mention_scorer(mention)
            curr_loss += loss_fn(pos_score, torch.tensor([1]))

            if print_score:
                print(pos_score)

            start_neg, end_neg = all_candidates[neg_examples[idx_ex]]
            non_mention = curr_flatdoc[start_neg: end_neg]
            non_mention = torch.tensor([[tok2id[t.lower()] for t in non_mention]])
            neg_score = mention_scorer(non_mention)
            curr_loss += loss_fn(neg_score, torch.tensor([0]))

        curr_loss /= (2 * pos_examples.shape[0])
        curr_loss.backward()
        mention_opt.step()

    return float(curr_loss)


# TODO: obtaining candidates and labels could be done just once per document (instead of once per document per epoch)
if __name__ == "__main__":
    DATA_DIR = "/home/matej/Documents/mag/2-letnik/obdelava_naravnega_jezika/coref149"
    # Consider spans up to this width as antecedent candidates
    MAX_SPAN_WIDTH = 10

    # Dataset loading and splitting into train/test
    all_docs = read_corpus(DATA_DIR)
    TR_TE_BOUNDARY = int(0.8 * len(all_docs))
    train_docs = all_docs[: TR_TE_BOUNDARY]
    test_docs = all_docs[TR_TE_BOUNDARY:]

    # Vocabulary extraction
    tok2id, id2tok = extract_vocab(train_docs)
    SAMPLE_SIZE = 1

    mention_scorer = SimpleMentionScorer(vocab_size=len(tok2id), emb_size=28)
    coref_scorer = SimpleCorefScorer(vocab_size=len(tok2id), emb_size=14)

    mention_optimizer = optim.SGD(mention_scorer.parameters(), lr=0.1)
    coref_optimizer = optim.SGD(coref_scorer.parameters(), lr=0.1)

    loss_fn = nn.CrossEntropyLoss()

    mention_scorer.train()
    coref_scorer.train()
    for idx_epoch in range(3):
        shuffle(train_docs)
        epoch_loss = train_epoch(train_docs, mention_scorer, coref_scorer, mention_optimizer, coref_optimizer, loss_fn)
        print(f"Epoch #{idx_epoch}: {epoch_loss:.4f}")

    test_gt = []
    test_preds = []
    mention_scorer.eval()
    for curr_doc in test_docs:
        curr_flatdoc = flatten_doc(curr_doc.raw_sentences())
        # Get all mention candidates (up to MAX_SPAN_WIDTH)
        all_candidates, all_labels = [], []
        for curr_width in range(1, 1 + MAX_SPAN_WIDTH):
            curr_cands, curr_labels = mention_dataset(curr_doc, width=curr_width)
            all_candidates.extend(curr_cands)
            all_labels.extend(curr_labels)

        test_gt.extend(all_labels)

        curr_preds = []
        for curr_cand in all_candidates:
            s, e = curr_cand
            raw_cand = torch.tensor([[
                tok2id.get(t.lower(), tok2id["<UNK>"]) for t in curr_flatdoc[s: e]
            ]])
            curr_score = mention_scorer(raw_cand)
            curr_preds.append(torch.argmax(curr_score, dim=1).item())

        test_preds.extend(curr_preds)


