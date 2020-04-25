import logging
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import metrics
from data import DATA_DIR, SSJ_PATH, read_corpus
from utils import get_clusters
from visualization import build_and_display

#####################
# GLOBAL PARAMETERS
#####################
logging.basicConfig(level=logging.INFO)

NUM_FEATURES = 2  # TODO: set this appropriately based on number of features in `features_mention_pair(...)`
NUM_EPOCHS = 2
# Note: if you don't want to save model, set this to "" or None
MODELS_SAVE_DIR = "baseline_model"
VISUALIZATION_GENERATE = False
VISUALIZATION_OPEN_WHEN_DONE = False


# Useful resource for parsing the morphosyntactic properties:
# http://nl.ijs.si/ME/V5/msd/html/msd-sl.html#msd.categories-sl


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
            curr_number = extract_number(morphsyntax)
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
        - TODO: cache global var? (where already constructed features would get stored)
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


class BaselineModel:

    name: str

    path_model_dir: str
    path_model_metadata: str
    path_test_preds: str

    model: nn.Linear
    model_optimizer: optim.SGD
    loss: nn.CrossEntropyLoss

    # indicates whether a model was loaded from file (if it was, training phase can be skipped)
    loaded_from_file: bool

    def __init__(self, in_features, out_features=1, lr=0.01, name=None):
        """
        Initializes a new BaselineModel. baseline.prepare() should be called after initialization!
        """
        self.name = name
        if self.name is None:
            self.name = time.strftime("%Y%m9%d_%H%M%S")

        self.path_model_dir = os.path.join(MODELS_SAVE_DIR, self.name)
        self.path_model_metadata = os.path.join(self.path_model_dir, "model_metadata.txt")
        self.path_test_preds = os.path.join(self.path_model_dir, "test_preds.txt")

        self.model = nn.Linear(in_features=in_features, out_features=out_features)
        self.model_optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        logging.debug("Initialized new baseline model")
        pass

    def prepare(self):
        """
        Prepares directories and files for the model. If directory for the model's name already exists, it tries to load
        an existing model. If loading the model was succesful, `self.loaded_from_file` is set to True.
        """
        # Prepare directory for saving model for this run
        if MODELS_SAVE_DIR and not os.path.exists(self.path_model_dir):
            self.loaded_from_file = False
            os.makedirs(self.path_model_dir)
            logging.info(f"Created directory '{self.path_model_dir}' for saving model")

            # Save metadata for this run
            if MODELS_SAVE_DIR:
                with open(self.path_model_metadata, "w") as f:
                    print("Train model features:", file=f)
                    print(f"NUM_FEATURES: {self.model.in_features}", file=f)
                    print(f"NUM_EPOCHS: {NUM_EPOCHS}", file=f)
                    print("", file=f)

        else:
            logging.info(f"Directory '{self.path_model_dir}' already exists")
            path_to_model = os.path.join(self.path_model_dir, 'best.th')
            if os.path.isfile(path_to_model):
                logging.info(f"Model with name '{self.name}' already exists. Loading model...")
                self.model.load_state_dict(torch.load(path_to_model))
                logging.info(f"Model with name '{self.name}' loaded")
                self.loaded_from_file = True
        pass

    def train(self, epochs, train_docs, dev_docs):
        best_dev_loss = float("inf")
        logging.info("Starting baseline training...")
        for idx_epoch in range(epochs):
            logging.info(f"\tRunning epoch {idx_epoch+1}/{epochs}")

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

            # logging.info(f"----------------------------------------------")
            # logging.info(f"**Training loss: {train_loss / max(1, train_examples): .4f}**")
            # logging.info(f"**Dev loss: {dev_loss / max(1, dev_examples): .4f}**")
            # logging.info(f"----------------------------------------------")

            if ((dev_loss / dev_examples) < best_dev_loss) and MODELS_SAVE_DIR:
                logging.info(f"Saving new best model to '{self.path_model_dir}'")
                torch.save(self.model.state_dict(), os.path.join(self.path_model_dir, 'best.th'))

                # Save this score as best
                best_dev_loss = dev_loss / dev_examples

            logging.info("")
        logging.info("Training baseline complete")

        # Add model train scores to model metadata
        if MODELS_SAVE_DIR:
            with open(self.path_model_metadata, "w") as f:
                print("Train model scores:", file=f)
                print(f"Best validation set loss: {best_dev_loss}", file=f)
            logging.info(f"Saved best validation score to {self.path_model_metadata}")
        pass

    def evaluate(self, test_docs):
        ####################################
        # EVALUATION OF MODEL ON TEST DATA
        ####################################
        # doc_name: <cluster assignments> pairs for all test documents
        logging.info("Evaluating baseline...")
        all_test_preds = {}

        # [MUC score]
        # The MUC score counts the minimum number of links between mentions
        # to be inserted or deleted when mapping a system response to a gold standard key set
        muc_prec, muc_rec, muc_f1 = 0.0, 0.0, 0.0

        # [B3 score]
        # B3 computes precision and recall for all mentions in the document,
        # which are then combined to produce the final precision and recall numbers for the entire output
        b3_prec, b3_rec, b3_f1 = 0.0, 0.0, 0.0

        # [CEAF score]
        # CEAF applies a similarity metric (either mention based or entity based) for each pair of entities
        # (i.e. a set of mentions) to measure the goodness of each possible alignment.
        # The best mapping is used for calculating CEAF precision, recall and F-measure
        ceaf_prec, ceaf_rec, ceaf_f1 = 0.0, 0.0, 0.0

        logging.info("Evaluation with MUC, BCube and CEAF score...")
        for curr_doc in test_docs:

            test_preds, _ = self._train_doc(curr_doc, eval_mode=True)
            test_clusters = get_clusters(test_preds)

            # Save predicted clusters for this document id
            all_test_preds[curr_doc.doc_id] = test_clusters

            gt_clusters = {}  # ground truth / gold clusters
            for id_cluster, cluster in enumerate(curr_doc.clusters):
                for mention_id in cluster:
                    gt_clusters[mention_id] = {id_cluster}

            m_muc = metrics.muc(test_clusters, gt_clusters)
            m_b3 = metrics.b_cubed(test_clusters, gt_clusters)
            m_ceaf = metrics.ceaf_e(test_clusters, gt_clusters)

            b3_prec += m_b3[0]
            b3_rec += m_b3[1]
            b3_f1 += m_b3[2]

            muc_prec += m_muc[0]
            muc_rec += m_muc[1]
            muc_f1 += m_muc[2]

            ceaf_prec += m_ceaf[0]
            ceaf_rec += m_ceaf[1]
            ceaf_f1 += m_ceaf[2]

        # Calculate combined B3 score
        b3_prec /= len(test_docs)
        b3_rec /= len(test_docs)
        b3_f1 /= len(test_docs)
        muc_prec /= len(test_docs)
        muc_rec /= len(test_docs)
        muc_f1 /= len(test_docs)
        ceaf_prec /= len(test_docs)
        ceaf_rec /= len(test_docs)
        ceaf_f1 /= len(test_docs)

        logging.info(f"----------------------------------------------")
        logging.info(f"**Test scores**")
        logging.info(f"**MUC:    precision={muc_prec:.3f}, recall={muc_rec:.3f}, F1={muc_f1:.3f}**")
        logging.info(f"**BCubed: precision={b3_prec:.3f}, recall={b3_rec:.3f}, F1={b3_f1:.3f}**")
        logging.info(f"**CEAF:   precision={ceaf_prec:.3f}, recall={ceaf_rec:.3f}, F1={ceaf_f1:.3f}**")
        logging.info(f"----------------------------------------------")

        if MODELS_SAVE_DIR:
            # Save test predictions and scores to file for further debugging
            with open(self.path_test_preds, "w") as f:
                print(f"Test scores:", file=f)
                print(f"MUC:    precision={muc_prec:.3f}, recall={muc_rec:.3f}, F1={muc_f1:.3f}", file=f)
                print(f"BCubed: precision={b3_prec:.3f}, recall={b3_rec:.3f}, F1={b3_f1:.3f}", file=f)
                print(f"CEAF:   precision={ceaf_prec:.3f}, recall={ceaf_rec:.3f}, F1={ceaf_f1:.3f}\n", file=f)

                print("Predictions", file=f)
                for doc_id, clusters in all_test_preds.items():
                    print(f"Document '{doc_id}':", file=f)
                    print(clusters, file=f)

            # Build and display visualization
            if VISUALIZATION_GENERATE:
                build_and_display(self.path_test_preds, self.path_model_dir, VISUALIZATION_OPEN_WHEN_DONE)

        pass

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return [], (0.0, 0)

        logging.debug(f"**Sorting mentions...**")
        sorted_mentions = sorted(curr_doc.mentions.items(), key=lambda tup: (tup[1].positions[0][0],  # sentence
                                                                             tup[1].positions[0][1],  # start pos
                                                                             tup[1].positions[-1][1]))  # end pos
        doc_loss, n_examples = 0.0, 0
        preds = {}

        logging.debug(f"**Processing {len(sorted_mentions)} mentions...**")
        for idx_head, (head_id, head_mention) in enumerate(sorted_mentions, 1):
            logging.debug(f"**#{idx_head} Mention '{head_id}': {head_mention}**")
            self.model.zero_grad()

            gt_antecedent_id = curr_doc.mapped_clusters[head_id]

            # Note: no features for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, features = [None], []
            gt_antecedent = torch.tensor([0])

            for idx_candidate, (cand_id, cand_mention) in enumerate(sorted_mentions, 1):
                if cand_id == gt_antecedent_id:
                    gt_antecedent[:] = idx_candidate

                # Obtain scores for candidates and select best one as antecedent
                if idx_candidate == idx_head:
                    if len(features) > 0:
                        features = torch.tensor(np.array(features, dtype=np.float32))

                        cand_scores = self.model(features)

                        # Concatenates the given sequence of seq tensors in the given dimension
                        card_scores = torch.cat((torch.tensor([0.]), cand_scores.flatten())).unsqueeze(0)

                        cand_scores = torch.softmax(card_scores, dim=-1)

                        # Get index of max value. That index equals to mention at that place
                        curr_pred = torch.argmax(cand_scores)

                        curr_loss = self.loss(cand_scores, gt_antecedent)
                        doc_loss += float(curr_loss)

                        n_examples += 1

                        if not eval_mode:
                            curr_loss.backward()
                            self.model_optimizer.step()
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
                    features.append(features_mention_pair(curr_doc, head_mention, cand_mention))

        return preds, (doc_loss, n_examples)


if __name__ == "__main__":
    #################################
    # PREPARATION AND INITIALIZATION
    #################################
    # Prepare directory for saving trained models
    if MODELS_SAVE_DIR and not os.path.exists(MODELS_SAVE_DIR):
        os.makedirs(MODELS_SAVE_DIR)
        logging.info(f"**Created directory '{MODELS_SAVE_DIR}' for saving models**")

    # Read corpus. Documents will be of type 'Document'
    documents = read_corpus(DATA_DIR, SSJ_PATH)

    # Split documents to train and test set
    train_docs, dev_docs = documents[: -40], documents[-40: -20]
    test_docs = documents[-20:]
    logging.info(f"**{len(documents)} documents split to: training set ({len(train_docs)}), dev set ({len(dev_docs)}) "
                 f"and test set ({len(test_docs)})**")

    # if you'd like to reuse a model, give it a name, i.e.
    # baseline = BaselineModel(NUM_FEATURES, name="my_magnificent_model")
    baseline = BaselineModel(NUM_FEATURES)
    baseline.prepare()

    if not baseline.loaded_from_file:
        # train only if it was not loaded
        baseline.train(NUM_EPOCHS, train_docs, dev_docs)

    baseline.evaluate(test_docs)