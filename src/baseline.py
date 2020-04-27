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

NUM_FEATURES = 1  # TODO: set this appropriately based on number of features in `features_mention_pair(...)`
NUM_EPOCHS = 100
LEARNING_RATE = 30

MODELS_SAVE_DIR = "baseline_model"
VISUALIZATION_GENERATE = True
VISUALIZATION_OPEN_WHEN_DONE = True

# Cache features for mention pairs (useful for doing multiple epochs over data)
# Format: {doc1_id: {(mention1_id, mention2_id): <features>, ...}, ...}
_features_cache = {}


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


class FeatureMentionPair:

    @staticmethod
    def str_match(this_feats, other_feats):
        """
        True:  if neither mentions (this and other) are pronouns and mention's lemmas match
        False: otherwise
        """
        return int(this_feats["category"] != "Z" and other_feats["category"] != "Z" and \
               " ".join(this_feats["lemmas"]) == " ".join(other_feats["lemmas"]))

    @staticmethod
    def in_same_sentence(this_feats, other_feats):
        """
        True:  If mentions this and other in the same sentence
        False: otherwise
        """
        return int(this_feats["idx_sent"] == other_feats["idx_sent"])

    @staticmethod
    def is_same_gender(this_feats, other_feats):
        """
        One-hot encoded vector if this and other mention:
        [ match in gender, do not match in gender, gender can't be determined ]
        """
        is_same_gender = None
        if this_feats["gender"] is not None and other_feats["gender"] is not None:
            is_same_gender = this_feats["gender"] == other_feats["gender"]

        return [
            int(is_same_gender is True),
            int(is_same_gender is False),
            int(is_same_gender is None)
        ]

    @staticmethod
    def is_same_number(this_feats, other_feats):
        """
        One-hot encoded vector if this and other mention:
        [ match in number, do not match in number, number can't be determined]
        """
        is_same_number = None
        if this_feats["number"] is not None and other_feats["number"] is not None:
            is_same_number = this_feats["number"] == other_feats["number"]

        return [
            int(is_same_number is True),
            int(is_same_number is False),
            int(is_same_number is None),
        ]

    @staticmethod
    def is_appositive(this_feats, other_feats):
        """
        Two mentions are assumed appositive, if:
            - they are of NP, NN POS tag or other noun-related tag
            - previous mention is followed by comma (i.e. ...Janez Novak, predsednik drustva...)
        """
        print(this_feats, other_feats)
        # TODO implement
        return 0

    @staticmethod
    def is_alias():
        """
        One mention is considered an alias of another, if:
            - word or initials match exactly, or
            - mentions match partially (i.e. Marija Novak <-> gospa Novak)
            - one mention is a token subset of another (i.e. Janez Novak <-> Novak)
        """
        # TODO implement
        return 0

    # TODO: add more features (good source includes
    #       https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0100101 )


def features_mention_pair(doc, head_mention, cand_mention):

    # ------ LOAD FROM CACHE START
    head_id, cand_id = head_mention.mention_id, cand_mention.mention_id
    # if features for this mention pair have already been constructed, use them instead of constructing them again
    cached_doc_features = _features_cache.get(doc.doc_id)
    if cached_doc_features:
        cached_pair_features = cached_doc_features.get((head_id, cand_id))
        if cached_pair_features:
            return cached_pair_features
    else:
        _features_cache[doc.doc_id] = {}

    # ------ LOAD FROM CACHE END

    head_features = features_mention(doc, head_mention)
    cand_features = features_mention(doc, cand_mention)

    pair_features = [
        # FeatureMentionPair.in_same_sentence(head_features, cand_features),
        FeatureMentionPair.str_match(head_features, cand_features),

        # protip: add * if function returns a vector, but be wary of number of features added
        # *FeatureMentionPair.is_same_gender(head_features, cand_features)[0],  # 3 features
        # *FeatureMentionPair.is_same_number(head_features, cand_features),  # 3 features

        # TODO: implement in FeatureMentionPair
        # FeatureMentionPair.is_appositive(???),
        # FeatureMentionPair.is_alias(???),
    ]

    # add features calculated above to cache
    _features_cache[doc.doc_id][(head_id, cand_id)] = pair_features

    return pair_features


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

    def __init__(self, in_features, name=None):
        """
        Initializes a new BaselineModel.
        """
        self.name = name
        if self.name is None:
            self.name = time.strftime("%Y%m9%d_%H%M%S")

        self.path_model_dir = os.path.join(MODELS_SAVE_DIR, self.name)
        self.path_model_metadata = os.path.join(self.path_model_dir, "model_metadata.txt")
        self.path_test_preds = os.path.join(self.path_model_dir, "test_preds.txt")

        out_features = 1
        self.model = nn.Linear(in_features=in_features, out_features=out_features)
        self.model_optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.CrossEntropyLoss()
        logging.debug("Initialized new baseline model")
        self._prepare()

    def train(self, epochs, train_docs, dev_docs):
        best_dev_loss = float("inf")
        logging.info("Starting baseline training...")
        t_start = time.time()
        for idx_epoch in range(epochs):
            t_epoch_start = time.time()
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

            logging.info(f"\t\tParams:        {self.model.state_dict()}")
            logging.info(f"\t\tTraining loss: {train_loss / max(1, train_examples): .4f}")
            logging.info(f"\t\tDev loss:      {dev_loss / max(1, dev_examples): .4f}")

            if ((dev_loss / dev_examples) < best_dev_loss) and MODELS_SAVE_DIR:
                logging.info(f"\tSaving new best model to '{self.path_model_dir}'")
                torch.save(self.model.state_dict(), os.path.join(self.path_model_dir, 'best.th'))

                # Save this score as best
                best_dev_loss = dev_loss / dev_examples

            logging.info(f"\tEpoch #{1 + idx_epoch} took {time.time() - t_epoch_start:.2f}s")
            logging.info("")
        logging.info("Training baseline complete")
        logging.info(f"Training took {time.time() - t_start:.2f}s")

        # Add model train scores to model metadata
        if MODELS_SAVE_DIR:
            with open(self.path_model_metadata, "w") as f:
                print("Train model scores:", file=f)
                print(f"Best validation set loss: {best_dev_loss}", file=f)
            logging.info(f"Saved best validation score to {self.path_model_metadata}")

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
        # [B3 score]
        # B3 computes precision and recall for all mentions in the document,
        # which are then combined to produce the final precision and recall numbers for the entire output
        # [CEAF score]
        # CEAF applies a similarity metric (either mention based or entity based) for each pair of entities
        # (i.e. a set of mentions) to measure the goodness of each possible alignment.
        # The best mapping is used for calculating CEAF precision, recall and F-measure
        mucScore = metrics.Score()
        b3Score = metrics.Score()
        ceafScore = metrics.Score()

        logging.info("Evaluation with MUC, BCube and CEAF score...")
        for curr_doc in test_docs:

            test_preds, _ = self._train_doc(curr_doc, eval_mode=True)
            test_clusters = get_clusters(test_preds)

            # Save predicted clusters for this document id
            all_test_preds[curr_doc.doc_id] = test_clusters

            # input into metric functions should be formatted as dictionary of {int -> set(str)},
            # where keys (ints) are clusters and values (string sets) are mentions in a cluster. Example:
            # {
            #  1: {'rc_1', 'rc_2', ...}
            #  2: {'rc_5', 'rc_8', ...}
            #  3: ...
            # }

            # gt = ground truth, pr = predicted by model
            gt_clusters = {k: set(v) for k, v in enumerate(curr_doc.clusters)}
            pr_clusters = {}
            for (pr_ment, pr_clst) in test_clusters.items():
                if pr_clst not in pr_clusters:
                    pr_clusters[pr_clst] = set()
                pr_clusters[pr_clst].add(pr_ment)

            mucScore.add(metrics.muc(gt_clusters, pr_clusters))
            b3Score.add(metrics.b_cubed(gt_clusters, pr_clusters))
            ceafScore.add(metrics.ceaf_e(gt_clusters, pr_clusters))

        logging.info(f"----------------------------------------------")
        logging.info(f"**Test scores**")
        logging.info(f"**MUC:    {mucScore}**")
        logging.info(f"**BCubed: {b3Score}**")
        logging.info(f"**CEAFe:   {ceafScore}**")
        logging.info(f"----------------------------------------------")

        if MODELS_SAVE_DIR:
            # Save test predictions and scores to file for further debugging
            with open(self.path_test_preds, "w") as f:
                print(f"Test scores:", file=f)
                print(f"**MUC:    {mucScore}**", file=f)
                print(f"**BCubed: {b3Score}**", file=f)
                print(f"**CEAFe:  {ceafScore}**", file=f)

                print("Predictions", file=f)
                for doc_id, clusters in all_test_preds.items():
                    print(f"Document '{doc_id}':", file=f)
                    print(clusters, file=f)

            # Build and display visualization
            if VISUALIZATION_GENERATE:
                build_and_display(self.path_test_preds, self.path_model_dir, VISUALIZATION_OPEN_WHEN_DONE)

        pass

    def _prepare(self):
        """
        Prepares directories and files for the model. If directory for the model's name already exists, it tries to load
        an existing model. If loading the model was succesful, `self.loaded_from_file` is set to True.
        """
        self.loaded_from_file = False
        # Prepare directory for saving model for this run
        if MODELS_SAVE_DIR and not os.path.exists(self.path_model_dir):
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
                    features.append(features_mention_pair(curr_doc, head_mention, cand_mention))

        return preds, (doc_loss, n_examples)


def split_into_sets(documents, random_seed=None):
    """
    Splits documents array into three sets: learning, validation & testing.
    If random seed is given, documents selected for each set are randomly picked (but do not overlap, of course).
    """
    idx = np.arange(len(documents))

    # Setting a seed will always produce the same "shuffle"
    if random_seed is not None:
        np.random.seed(random_seed)

        # basically just shuffle indexes...
        np.random.shuffle(idx)

    train_idx, dev_idx, test_idx = idx[: -40], idx[-40: -20], idx[-20:]
    # ... and then select those indexes from list of documents
    documents = np.array(documents)
    train_docs, dev_docs, test_docs = documents[train_idx], documents[dev_idx], documents[test_idx]

    logging.info(f"**{len(documents)} documents split to: training set ({len(train_docs)}), dev set ({len(dev_docs)}) "
                 f"and test set ({len(test_docs)})**")

    return train_docs, dev_docs, test_docs


if __name__ == "__main__":
    # Prepare directory for saving trained models
    if MODELS_SAVE_DIR and not os.path.exists(MODELS_SAVE_DIR):
        os.makedirs(MODELS_SAVE_DIR)
        logging.info(f"**Created directory '{MODELS_SAVE_DIR}' for saving models**")

    # Read corpus. Documents will be of type 'Document'
    documents = read_corpus(DATA_DIR, SSJ_PATH)
    train_docs, dev_docs, test_docs = split_into_sets(documents)

    # if you'd like to reuse a model, give it a name, i.e.
    # baseline = BaselineModel(NUM_FEATURES, name="my_magnificent_model")
    baseline = BaselineModel(NUM_FEATURES)

    if not baseline.loaded_from_file:
        # train only if it was not loaded
        baseline.train(NUM_EPOCHS, train_docs, dev_docs)

    baseline.evaluate(test_docs)
