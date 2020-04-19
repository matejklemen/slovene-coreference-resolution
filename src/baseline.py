import torch
import os

import numpy as np
import torch.optim as optim
import torch.nn as nn

from data import read_corpus
from collections import Counter


""" Useful resource for parsing the morphosyntactic properties: 
    http://nl.ijs.si/ME/V5/msd/html/msd-sl.html#msd.categories-sl """


def extract_category(msd_string):
    return msd_string[0]


def extract_gender(msd_string):
    gender = None
    if msd_string[0] == "S" and len(msd_string) >= 3:  # noun/samostalnik
        gender = msd_string[2]
    elif msd_string[0] == "G" and len(msd_string) >= 7:  # verb/glagol
        gender = msd_string[6]
    # P = adjective (pridevnik), Z = pronoun (zaimek), K = numeral (števnik)
    elif msd_string[0] in {"P", "Z", "K"} and len(msd_string) >= 4:
        gender = msd_string[3]

    return gender


def extract_number(msd_string):
    number = None
    if msd_string[0] == "S" and len(msd_string) >= 4:  # noun/samostalnik
        number = msd_string[3]
    elif msd_string[0] == "G" and len(msd_string) >= 6:  # verb/glagol
        number = msd_string[5]
    # P = adjective (pridevnik), Z = pronoun (zaimek), K = numeral (števnik)
    elif msd_string[0] in {"P", "Z", "K"} and len(msd_string) >= 5:
        number = msd_string[4]

    return number


def features_mention(doc, mention):
    """ Extract features for a mention. """
    # bs4 tags (metadata) for mention tokens (<=1 per token, punctuation currently excluded)
    mention_objs = doc.ssj_doc.findAll("w", {"xml:id": lambda val: val and val in mention.token_ids})
    # Index in which mention appears
    idx_sent = mention.positions[0][0]

    gender = None  # {None, 'm', 's', 'z'}
    number = None  # {None, 'e', 'd', 'm'}
    categories = Counter()  # {'S', 'G', 'P', 'Z', ...}
    lemmas = []

    for obj in mention_objs:
        lemmas.append(obj["lemma"])
        _, morphsyntax = obj["ana"].split(":")

        # Take gender of first token for which it can be determined
        if gender is None:
            curr_gender = extract_gender(morphsyntax)
            if curr_gender in {"m", "z", "s"}:
                gender = curr_gender

        # Take number of first token for which it can be determined
        if number is None:
            curr_number = extract_gender(morphsyntax)
            if curr_number in {"e", "d", "m"}:
                number = curr_number

        curr_category = extract_category(morphsyntax)
        categories[curr_category] = categories.get(curr_category, 0) + 1

    cat = categories.most_common(1)[0][0]

    return {
        "idx_sent": idx_sent,
        "lemmas": lemmas,
        "gender": gender,
        "number": number,
        "category": cat
    }


def features_mention_pair(doc, head_mention, cand_mention):
    """ Extracts features for a mention pair.
        - TODO: optional cache parameter? (where already constructed features would get stored)
        - TODO: additional features """

    head_features = features_mention(doc, head_mention)
    cand_features = features_mention(doc, cand_mention)

    # Inside same sentence
    is_same_sent = head_features["idx_sent"] == cand_features["idx_sent"]

    # Agreement in gender (None = can't determine)
    is_same_gender = None
    if head_features["gender"] is not None and cand_features["gender"] is not None:
        is_same_gender = head_features["gender"] == cand_features["gender"]

    # Agreement in number (None = can't determine)
    is_same_number = None
    if head_features["number"] is not None and cand_features["number"] is not None:
        is_same_number = head_features["number"] == cand_features["number"]

    # Not pronouns (these can be written same and refer to different objects) + exact match of lemmas
    str_match = head_features["category"] != "Z" and cand_features["category"] != "Z" and \
                " ".join(head_features["lemmas"]) == " ".join(cand_features["lemmas"])

    # TODO: transform constructed features into vectors (is_same_gender, is_same_number have 3 categories!)
    # ...

    return [int(is_same_sent), int(str_match)]


def train_doc(model, model_opt, loss, curr_doc, eval_mode=False):
    """ Trains/evaluates (if `eval_mode` is True) model on specific document.
        Returns predictions, loss and number of examples evaluated. """

    if len(curr_doc.mentions) == 0:
        return [], (0.0, 0)

    sorted_mentions = sorted(curr_doc.mentions.items(), key=lambda tup: (tup[1].positions[0][0],  # sentence
                                                                         tup[1].positions[0][1],  # start pos
                                                                         tup[1].positions[-1][1]))  # end pos
    doc_loss, n_examples = 0.0, 0
    preds = []

    for idx_head, (head_id, head_mention) in enumerate(sorted_mentions, 1):
        model.zero_grad()
        gt_antecedent_id = curr_doc.mapped_clusters[head_id]

        # Note: no features for dummy antecedent (len(`features`) is one less than `candidates`)
        candidates, features = [None], []
        gt_antecedent = torch.tensor([0])
        for idx_candidate, (cand_id, cand_mention) in enumerate(sorted_mentions, 1):
            if cand_id == gt_antecedent_id:
                gt_antecedent[:] = idx_candidate

            # obtain scores for candidates and select best one as antecedent
            if idx_candidate == idx_head:
                if len(features) > 0:
                    features = torch.tensor(np.array(features, dtype=np.float32))
                    cand_scores = model(features)
                    cand_scores = torch.softmax(torch.cat((torch.tensor([0.]), cand_scores.flatten())).unsqueeze(0),
                                                dim=-1)

                    curr_pred = torch.argmax(cand_scores)

                    curr_loss = loss(cand_scores, gt_antecedent)
                    doc_loss += float(curr_loss)
                    n_examples += 1

                    if not eval_mode:
                        curr_loss.backward()
                        model_opt.step()
                else:
                    # only one candidate antecedent = first mention
                    curr_pred = 0

                preds.append({
                    "mention_id": head_id,
                    "antecedent_id": candidates[int(curr_pred)]
                })
                break
            else:
                # add current mention as candidate
                candidates.append(cand_id)
                features.append(features_mention_pair(curr_doc, head_mention, cand_mention))

    return preds, (doc_loss, n_examples)


if __name__ == "__main__":
    DATA_DIR = "/home/matej/Documents/mag/2-letnik/obdelava_naravnega_jezika/coref149"
    SSJ_PATH = "/home/matej/Documents/mag/2-letnik/obdelava_naravnega_jezika/coref149/ssj500k-sl.TEI/ssj500k-reduced.xml"
    # Note: if you don't want to save model, set this to "" or None
    MODEL_SAVE_DIR = "baseline_model"
    if MODEL_SAVE_DIR and not os.path.exists(MODEL_SAVE_DIR):
        print(f"**Created directory '{MODEL_SAVE_DIR}' for saving model**")
        os.makedirs(MODEL_SAVE_DIR)

    documents = read_corpus(DATA_DIR, SSJ_PATH)
    train_docs, dev_docs = documents[: -40], documents[-40: -20]
    test_docs = documents[-20:]

    NUM_EPOCHS = 2
    NUM_FEATURES = 2  # TODO: set this appropriately based on number of features in `features_mention_pair(...)`

    model = nn.Linear(NUM_FEATURES, 1)
    model_optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()
    best_dev_loss = float("inf")

    for idx_epoch in range(NUM_EPOCHS):
        print(f"[EPOCH {1 + idx_epoch}]")

        shuffle_indices = torch.randperm(len(train_docs))
        model.train()
        # Train loop
        train_loss, train_examples = 0.0, 0
        for idx_doc in shuffle_indices:
            curr_doc = train_docs[idx_doc]
            preds, (doc_loss, n_examples) = train_doc(model, model_optimizer, loss, curr_doc)
            train_loss += doc_loss
            train_examples += n_examples

        model.eval()
        # Validation loop
        dev_loss, dev_examples = 0.0, 0
        for curr_doc in dev_docs:
            _, (doc_loss, n_examples) = train_doc(model, model_optimizer, loss, curr_doc, eval_mode=True)
            dev_loss += doc_loss
            dev_examples += n_examples

        print(f"**Training loss: {train_loss / max(1, train_examples): .4f}**")
        print(f"**Dev loss: {dev_loss / max(1, dev_examples): .4f}**")

        if (dev_loss / dev_examples) < best_dev_loss:
            print(f"**Saving new best model to '{MODEL_SAVE_DIR}'**")
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best.th'))

        print("-------------------")

    # TODO: test set prediction, form clusters, evaluate using MUC/Bcubed/CEAFe (and/or - if possible, all of these)
    # ...
