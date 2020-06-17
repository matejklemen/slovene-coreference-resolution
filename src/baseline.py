import argparse
import logging
import os
import time
from collections import Counter

import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyjarowinkler import distance as jwdistance
from utils import get_clusters, split_into_sets, fixed_split
from common import ControllerBase

from data import read_corpus

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--dataset", type=str, default="coref149")  # {'senticoref', 'coref149'}
parser.add_argument("--fixed_split", action="store_true")


logging.basicConfig(level=logging.INFO)


RANDOM_SEED = None
if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)

# Cache features for single mentions and mention pairs (useful for doing multiple epochs over data)
# Format for single: {doc1_id: {mention_id: <features>, ...}, ...}
_cached_MentionFeatures = {}
# Format for pair:   {doc1_id: {(mention1_id, mention2_id): <features>, ...}, ...}
_cached_MentionPairFeatures = {}


class MentionFeatures:

    # Useful resource for parsing the morphosyntactic properties:
    # http://nl.ijs.si/ME/V5/msd/html/msd-sl.html#msd.categories-sl

    @staticmethod
    def for_mention(document, mention, use_cache=True):
        # load from cache, if enabled and it exists
        if use_cache:
            if _cached_MentionFeatures.get(document.doc_id):
                if _cached_MentionFeatures[document.doc_id].get(mention.mention_id):
                    return _cached_MentionFeatures[document.doc_id][mention.mention_id]
            else:
                _cached_MentionFeatures[document.doc_id] = {}

        # otherwise create and store to cache
        mf = MentionFeatures(document, mention)
        if use_cache:
            _cached_MentionFeatures[document.doc_id][mention.mention_id] = mf
        return mf

    def __init__(self, document, mention):
        """
        Extract features for a given mention in the given document.
        Note: Some mention features are actually just properties of mention's tokens
        """
        self.mention = mention
        self.tokens = mention.tokens
        self.lemmas = [token.lemma for token in mention.tokens]  # Note: if token has no lemma, it is "None"!

        # Token index in which mention appears in the sentence
        self.sentence_index = mention.tokens[0].sentence_index

        # Index in which mention appears in the document (based on mentions). Example:
        # (">Janez Novak< je svoji >ženi >Mojci<< je kupil nov >kavomat<")
        # (  i=0                    i=1   i=2                   i=3
        self.mention_index = None
        for index, mention_id in enumerate(document.mentions.keys()):
            if mention_id == mention.mention_id:
                self.mention_index = index
                break

        # morphosyntactic description
        self.msd_desc = self.mention.tokens[0].msd

        # gender, number, main category
        self.gender = None  # {None, 'm', 's', 'z'}
        self.number = None  # {None, 'e', 'd', 'm'}
        self.categories = Counter()  # {'S', 'G', 'P', 'Z', ...}

        counted_categories = Counter()
        for token in mention.tokens:
            # Take gender of first token for which it can be determined
            if self.gender is None:
                if token.gender in {"m", "z", "s"}:
                    self.gender = token.gender

            # Take number of first token for which it can be determined
            if self.number is None:
                if token.number in {"e", "d", "m"}:
                    self.number = token.number

            # Count how many times each category appears in mention's tokens and take the most common one as the actual
            # mention's category
            counted_categories[token.category] += 1

        self.category = counted_categories.most_common(1)[0][0]


class MentionPairFeatures:

    @staticmethod
    def for_mentions(document, head_mention, cand_mention, use_cache=True):
        head_id, cand_id = head_mention.mention_id, cand_mention.mention_id
        doc_id = document.doc_id
        # load from cache, if enabled and it exists
        if use_cache:
            if _cached_MentionPairFeatures.get(doc_id):
                if _cached_MentionPairFeatures[doc_id].get((head_id, cand_id)):
                    return _cached_MentionPairFeatures[doc_id][(head_id, cand_id)]
            else:
                _cached_MentionPairFeatures[doc_id] = {}

        head_features = MentionFeatures.for_mention(document, head_mention)
        cand_features = MentionFeatures.for_mention(document, cand_mention)

        pair_features = [
            MentionPairFeatures.in_same_sentence(head_features, cand_features),
            MentionPairFeatures.str_match(head_features, cand_features),
            MentionPairFeatures.is_same_gender(head_features, cand_features),
            MentionPairFeatures.is_same_number(head_features, cand_features),
            MentionPairFeatures.is_prefix(head_features, cand_features),
            MentionPairFeatures.is_suffix(head_features, cand_features),
            MentionPairFeatures.jaro_winkler_dist(head_features, cand_features),
            MentionPairFeatures.is_appositive(head_features, cand_features, document),
            MentionPairFeatures.is_alias(head_features, cand_features),
            MentionPairFeatures.is_reflexive(head_features, cand_features),
        ]

        # add features for mention pair, constructed above, to cache
        _cached_MentionPairFeatures[doc_id][(head_id, cand_id)] = pair_features
        return pair_features

    @staticmethod
    def num_features():
        return 10  # corresponds to number of features returned by `MentionPairFeatures.for_mentions(...)`

    # !! Note
    # this_feats == head_features
    # other_feats == cand_features
    # meaning that, if order in document is important, "other" is before "this"!

    @staticmethod
    def str_match(this_feats, other_feats):
        """
        True:  if neither mentions (this and other) are pronouns and mention's lemmas match
        False: otherwise
        """
        return int(this_feats.category != "Z" and other_feats.category != "Z" and
                   this_feats.mention.lemma_text() == other_feats.mention.lemma_text())

    @staticmethod
    def in_same_sentence(this_feats, other_feats):
        """
        True:  If mentions this and other are in the same sentence
        False: otherwise
        """
        return int(this_feats.sentence_index == other_feats.sentence_index)

    @staticmethod
    def is_same_gender(this_feats, other_feats):
        """
        One-hot encoded vector if this and other mention:
        [ match in gender, do not match in gender, gender can't be determined ]
        """
        is_same_gender = None
        if this_feats.gender is not None and other_feats.gender is not None:
            is_same_gender = this_feats.gender == other_feats.gender

        return int(is_same_gender is True)

    @staticmethod
    def is_same_number(this_feats, other_feats):
        """
        One-hot encoded vector if this and other mention:
        [ match in number, do not match in number, number can't be determined]
        """
        is_same_number = None
        if this_feats.number is not None and other_feats.number is not None:
            is_same_number = this_feats.number == other_feats.number

        return int(is_same_number is True)

    @staticmethod
    def is_appositive(this_feats, other_feats, document):
        """
        Two mentions are assumed appositive, if:
            - they are of NP, NN POS tag or other noun-related tag
            - previous mention is followed by comma (i.e. ...Janez Novak, predsednik drustva...)
        """
        # TODO remarks: zadeva pozitivne primere vzame tudi naštevanja samostalnikov...
        # if both mentions are nouns
        if this_feats.category == "S" and other_feats.category == "S":
            # if both mentions are in same sentence
            if this_feats.sentence_index == other_feats.sentence_index:
                # "other" mention is positioned before "this" mention. to get distance in tokens between mentions,
                # we need distance from last token of "other" mention to first token in "this" mention
                # TODO: could generalize by comparing this and other first token position within sentence
                other_last_token_pos = other_feats.mention.tokens[0].position_in_sentence + len(other_feats.mention.tokens)
                this_first_token_pos = this_feats.mention.tokens[0].position_in_sentence
                if this_first_token_pos - other_last_token_pos == 1:
                    # there's exactly one token betwen, check if it's a comma
                    if document.tokens[document.sents[this_feats.sentence_index][other_last_token_pos]].raw_text == ",":
                        return int(True)
        return int(False)

    @staticmethod
    def is_alias(this_feats, other_feats):
        """
        One mention is considered an alias of another, if:
            - word or initials match exactly, or
            - mentions match partially (i.e. Marija Novak <-> gospa Novak)
            - one mention is a token subset of another (i.e. Janez Novak <-> Novak)
        """
        # TODO remarks: initials pade pri netrivialnih primerih.
        #               Primer: "Ministrstvo za kmetijstvo, gozdrastvo in prehrano" se inicira z "MKGIP", ne "MZK,GIP"
        this_initials = [tok.raw_text[0] for tok in this_feats.tokens]
        other_initials = [tok.raw_text[0] for tok in other_feats.tokens]
        this_words = this_feats.mention.raw_text()
        other_words = other_feats.mention.raw_text()

        # mentions are equal or one is initial of another
        if this_words == other_words or this_initials == other_initials or this_words == other_initials or this_initials == other_words:
            return int(True)

        # TODO remarks: return true on first match, which may not be specific enough (i.e. Janez Novak <-> Peter Novak
        #               sta lahko različni omembi...)
        for tok in this_feats.tokens:
            if tok in other_feats.tokens:
                return int(True)

        return int(False)

    @staticmethod
    def is_prefix(this_feats, other_feats):
        """
        True:  if this mention is prefix of other mention (or vice versa)
        False: otherwise
        """
        this_raw = this_feats.mention.raw_text()
        other_raw = other_feats.mention.raw_text()
        if this_raw.startswith(other_raw) or other_raw.startswith(this_raw):
            return int(True)
        return int(False)

    @staticmethod
    def is_suffix(this_feats, other_feats):
        """
        True:  if this mention is suffix of other mention (or vice versa)
        False: otherwise
        """
        this_raw = this_feats.mention.raw_text()
        other_raw = other_feats.mention.raw_text()
        if this_raw.endswith(other_raw) or other_raw.endswith(this_raw):
            return int(True)
        return int(False)

    @staticmethod
    def jaro_winkler_dist(this_feats, other_feats):
        """
        Result is a similarity value between this and other mention according to Jaro-Winkler metric.
        """
        return jwdistance.get_jaro_distance(this_feats.mention.raw_text(), other_feats.mention.raw_text())

    @staticmethod
    def is_reflexive(this_feats, other_feats):
        """
        True:  if this mention is reflexive and distance between this and other mentions is 0 (i.e. there are no other
               mentions between those two)
        False: otherwise

        Reflexive pronoun = povratni zaimek
        primer: "<Nueri> se, na primer, spominjajo <svojih> prednikov...", kjer je <svojih> povratni zaimek in se nanaša
        na omenitev takoj prej, <Nueri>

        note: izjeme so lahko dobsedeni navedki v navednicah!
        primer: ",,Prepričan <sem>, da ne bomo razočarali'', napoveduje Matjaž Brumen.", v tem primeru se <sem> nanaša
        na naslednjo omenitev t.j. <Matjaž Brumen>, ki je dobsedni navedek "izrekel".
        """
        if this_feats.msd_desc.startswith("Zp") and this_feats.mention_index - other_feats.mention_index == 1:
            return int(True)

        return int(False)


class BaselineController(ControllerBase):
    def __init__(self, in_features, dataset_name, model_name=None, learning_rate=0.001):
        self.num_features = in_features

        self.model = nn.Linear(in_features=in_features, out_features=1)
        self.model_optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        super().__init__(learning_rate=learning_rate, dataset_name=dataset_name, model_name=model_name)
        logging.info(f"Initialized baseline model with name {self.model_name}")

    @property
    def model_base_dir(self):
        return "baseline_model"

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load_checkpoint(self):
        path_to_model = os.path.join(self.path_model_dir, 'best.th')
        if os.path.isfile(path_to_model):
            logging.info(f"Model with name '{self.name}' already exists. Loading model...")
            self.model.load_state_dict(torch.load(path_to_model))
            logging.info(f"Model with name '{self.name}' loaded.")
            self.loaded_from_file = True

    def save_checkpoint(self):
        logging.info(f"Saving new best model to '{self.path_model_dir}'")
        torch.save(self.model.state_dict(), os.path.join(self.path_model_dir, 'best.th'))

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

        doc_loss, n_examples = 0.0, 0
        preds = {}

        for idx_head, (head_id, head_mention) in enumerate(curr_doc.mentions.items(), 1):
            logging.debug(f"**#{idx_head} Mention '{head_id}': {head_mention}**")

            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no features for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, features = [None], []
            gt_antecedents = []

            # TODO: this could be improved: only consider a window of antecedent candidates, not all preceding mentions
            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), 1):
                if cand_id != head_id and cand_id in gt_antecedent_ids:
                    gt_antecedents.append(idx_candidate)

                # Obtain scores for candidates and select best one as antecedent
                if idx_candidate == idx_head:
                    if len(features) > 0:
                        features = torch.tensor(np.array(features, dtype=np.float32))

                        cand_scores = self.model(features)
                        cand_scores = torch.cat((torch.tensor([0.]), cand_scores.flatten())).unsqueeze(0)  # [1, #cands]

                        # if no other antecedent exists for mention, then it's a first mention (GT is dummy antecedent)
                        if len(gt_antecedents) == 0:
                            gt_antecedents.append(0)

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
                    features.append(
                        MentionPairFeatures.for_mentions(curr_doc, head_mention, cand_mention))

        return preds, (doc_loss, n_examples)


class AllInOneModel:
    def __init__(self, baseline=None):
        self.baseline = baseline

    def evaluate(self, test_docs):
        muc_score = metrics.Score()
        b3_score = metrics.Score()
        ceaf_score = metrics.Score()

        for curr_doc in test_docs:
            # gt = ground truth, pr = predicted by model
            gt_clusters = {k: set(v) for k, v in enumerate(curr_doc.clusters)}
            pr_clusters = {0: set(curr_doc.mentions.keys())}

            muc_score.add(metrics.muc(gt_clusters, pr_clusters))
            b3_score.add(metrics.b_cubed(gt_clusters, pr_clusters))
            ceaf_score.add(metrics.ceaf_e(gt_clusters, pr_clusters))

        logging.info(f"All-in-one model: test scores")
        logging.info(f"MUC:      {muc_score}")
        logging.info(f"BCubed:   {b3_score}")
        logging.info(f"CEAFe:    {ceaf_score}")
        logging.info(f"CoNLL-12: {metrics.conll_12(muc_score, b3_score, ceaf_score)}")
        logging.info(f"\n")

        if self.baseline is not None:
            # Save test predictions and scores to file for further debugging
            with open(self.baseline.path_pred_scores, "a", encoding="utf-8") as f:
                f.writelines([
                    f"Test scores for All-in-one model:\n",
                    f"MUC:      {muc_score}\n",
                    f"BCubed:   {b3_score}\n",
                    f"CEAFe:    {ceaf_score}\n",
                    f"CoNLL-12: {metrics.conll_12(muc_score, b3_score, ceaf_score)}\n",
                ])


class EachInOwnModel:
    def __init__(self, baseline=None):
        self.baseline = baseline

    def evaluate(self, test_docs):
        muc_score = metrics.Score()
        b3_score = metrics.Score()
        ceaf_score = metrics.Score()

        for curr_doc in test_docs:
            # gt = ground truth, pr = predicted by model
            gt_clusters = {k: set(v) for k, v in enumerate(curr_doc.clusters)}
            pr_clusters = {cluster: {mention} for cluster, mention in enumerate(curr_doc.mentions.keys())}

            muc_score.add(metrics.muc(gt_clusters, pr_clusters))
            b3_score.add(metrics.b_cubed(gt_clusters, pr_clusters))
            ceaf_score.add(metrics.ceaf_e(gt_clusters, pr_clusters))

        logging.info(f"Each-in-own model: test scores")
        logging.info(f"MUC:      {muc_score}")
        logging.info(f"BCubed:   {b3_score}")
        logging.info(f"CEAFe:    {ceaf_score}")
        logging.info(f"CoNLL-12: {metrics.conll_12(muc_score, b3_score, ceaf_score)}")
        logging.info(f"\n")

        if self.baseline is not None:
            # Save test predictions and scores to file for further debugging
            with open(self.baseline.path_pred_scores, "a", encoding="utf-8") as f:
                f.writelines([
                    f"Test scores for Each-in-own model:\n",
                    f"MUC:      {muc_score}\n",
                    f"BCubed:   {b3_score}\n",
                    f"CEAFe:    {ceaf_score}\n",
                    f"CoNLL-12: {metrics.conll_12(muc_score, b3_score, ceaf_score)}\n",
                ])


if __name__ == "__main__":
    args = parser.parse_args()
    # if you'd like to reuse a model, provide a name, i.e. `baseline = BaselineController(..., model_name="model1")`
    baseline = BaselineController(MentionPairFeatures.num_features(),
                                  model_name=args.model_name,
                                  learning_rate=args.learning_rate,
                                  dataset_name=args.dataset)

    # Note: model should be initialized first as it also adds a logging handler to store logs into a file
    documents = read_corpus(args.dataset)
    if args.fixed_split:
        logging.info("Using fixed dataset split")
        train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
    else:
        train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

    if not baseline.loaded_from_file:
        # train only if it was not loaded
        baseline.train(args.num_epochs, train_docs, dev_docs)

    baseline.evaluate(test_docs)

    aioModel = AllInOneModel(baseline)
    aioModel.evaluate(test_docs)
    eioModel = EachInOwnModel(baseline)
    eioModel.evaluate(test_docs)

    baseline.visualize()
