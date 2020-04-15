import torch

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


def features_mention(doc, mention):
    """ Extract features for a mention. """
    # bs4 tags (metadata) for mention tokens (<=1 per token, punctuation currently excluded)
    mention_objs = doc.ssj_doc.findAll("w", {"xml:id": lambda val: val and val in mention.token_ids})
    # Index in which mention appears
    idx_sent = mention.positions[0][0]

    genders = Counter()  # {m, z, s, None}
    categories = Counter()  # {S, G, P, Z, ...}
    lemmas = []

    for obj in mention_objs:
        lemmas.append(obj["lemma"])
        _, morphsyntax = obj["ana"].split(":")

        curr_gender = extract_gender(morphsyntax)
        curr_category = extract_category(morphsyntax)

        genders[curr_gender] = genders.get(curr_gender, 0) + 1
        categories[curr_category] = categories.get(curr_category, 0) + 1

    gender = genders.most_common(1)[0][0]
    cat = categories.most_common(1)[0][0]

    return {
        "idx_sent": idx_sent,
        "lemmas": lemmas,
        "gender": gender,
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

    # Agreement in most frequent genders (None = can't determine)
    is_same_gender = None
    if head_features["gender"] is not None and cand_features["gender"] is not None:
        is_same_gender = head_features["gender"] == cand_features["gender"]

    # Not pronouns (these can be written same and refer to different objects) + exact match of lemmas
    str_match = head_features["category"] != "Z" and cand_features["category"] != "Z" and \
                " ".join(head_features["lemmas"]) == " ".join(cand_features["lemmas"])

    # TODO: transform constructed features into vectors (is_same_gender has 3 categories!)
    # ...

    return [int(is_same_sent), int(str_match)]


if __name__ == "__main__":
    DATA_DIR = "/home/matej/Documents/mag/2-letnik/obdelava_naravnega_jezika/coref149"
    SSJ_PATH = "/home/matej/Documents/mag/2-letnik/obdelava_naravnega_jezika/coref149/ssj500k-sl.TEI/ssj500k-reduced.xml"

    documents = read_corpus(DATA_DIR, SSJ_PATH)
    train_docs, dev_docs = documents[: -40], documents[-40: -20]
    test_docs = documents[-20:]

    curr_doc = train_docs[0]

    NUM_FEATURES = 2

    model = nn.Linear(2, 1)
    model_optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    sorted_mentions = sorted(curr_doc.mentions.items(), key=lambda tup: (tup[1].positions[0][0],  # sentence
                                                                         tup[1].positions[0][1],  # start pos
                                                                         tup[1].positions[-1][1]))  # end pos

    for idx_head, (head_id, head_mention) in enumerate(sorted_mentions, 1):
        model.zero_grad()
        print(f"**Mention '{head_id}': {head_mention}**")
        gt_antecedent_id = curr_doc.mapped_clusters[head_id]

        # Note: no features for dummy antecedent (len(`features`) is one less than `candidates`)
        candidates = [None]
        features = []
        gt_antecedent = torch.tensor([0])
        for idx_candidate, (cand_id, cand_mention) in enumerate(sorted_mentions, 1):
            if cand_id == gt_antecedent_id:
                gt_antecedent[:] = idx_candidate

            # obtain scores for candidates and select best one as antecedent
            if idx_candidate == idx_head:
                if len(features) > 0:
                    features = torch.tensor(np.array(features, dtype=np.float32))
                    cand_scores = model(features)
                    cand_scores = torch.cat((torch.tensor([0.]), cand_scores.flatten())).unsqueeze(0)

                    prediction = torch.argmax(cand_scores)

                    curr_loss = loss(cand_scores, gt_antecedent)
                    curr_loss.backward()
                    model_optimizer.step()
                else:
                    prediction = 0

                break
            else:
                # add current mention as candidate
                mention_pair = (cand_id, head_id)
                print(f"**Processing {mention_pair} as a candidate**")

                features.append(features_mention_pair(curr_doc, head_mention, cand_mention))
