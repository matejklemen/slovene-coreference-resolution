import json
from collections import Counter
import logging
import os
from typing import List, Optional, Mapping

from sklearn.model_selection import train_test_split


PAD_TOKEN, PAD_ID = "<PAD>", 0
BOS_TOKEN, BOS_ID = "<BOS>", 1
EOS_TOKEN, EOS_ID = "<EOS>", 2
UNK_TOKEN, UNK_ID = "<UNK>", 3


class KFoldStateCache:
    def __init__(self, script_name: str, main_dataset: str, fold_info: List[dict],
                 additional_dataset: Optional[str] = None,
                 script_args: Optional[Mapping] = None):
        self.script_name = script_name
        self.fold_info = fold_info
        self.num_folds = len(self.fold_info)

        self.script_args = script_args if script_args is not None else {}

        # The dataset that is being split with KFold CV
        self.main_dataset = main_dataset
        # For combined runners: documents, read with `read_corpus(additional_dataset)` should be placed in training set
        self.additional_dataset = additional_dataset

    def get_next_unfinished(self):
        for i, curr_fold in enumerate(self.fold_info):
            if curr_fold.get("results", None) is None:
                yield {
                    "idx_fold": i,
                    "train_docs": curr_fold["train_docs"],
                    "test_docs": curr_fold["test_docs"]
                }

    def add_results(self, idx_fold, results):
        self.fold_info[idx_fold]["results"] = results

    def save(self, path):
        _path = path if path.endswith(".json") else f"{path}.json"
        if os.path.exists(_path):
            logging.warning(f"Overwriting KFold cache at '{_path}'")
        with open(_path, "w", encoding="utf8") as f:
            json.dump({
                "script_name": self.script_name,
                "script_args": self.script_args,
                "main_dataset": self.main_dataset,
                "additional_dataset": self.additional_dataset,
                "fold_info": self.fold_info
            }, fp=f, indent=4)

    @staticmethod
    def from_file(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = KFoldStateCache(script_name=data["script_name"],
                                   script_args=data.get("script_args", None),
                                   main_dataset=data["main_dataset"],
                                   fold_info=data["fold_info"],
                                   additional_dataset=data.get("additional_dataset", None))
        return instance


def extract_vocab(documents, top_n=10_000, lowercase=False):
    token_counter = Counter()
    for curr_doc in documents:
        curr_sentences = curr_doc.raw_sentences()

        for sent_tokens in curr_sentences:
            processed = list(map(lambda s: s.lower() if lowercase else s, sent_tokens))
            token_counter += Counter(processed)

    tok2id, id2tok = {}, {}
    special_tokens = [(PAD_TOKEN, PAD_ID), (BOS_TOKEN, BOS_ID), (EOS_TOKEN, EOS_ID), (UNK_TOKEN, UNK_ID)]
    for t, i in special_tokens:
        tok2id[t] = i
        id2tok[i] = t

    for i, (token, _) in enumerate(token_counter.most_common(top_n), start=len(special_tokens)):
        tok2id[token] = i
        id2tok[i] = token

    return tok2id, id2tok


def encode(seq, vocab, max_seq_len):
    encoded_seq = []
    for i, curr_token in enumerate(seq):
        encoded_seq.append(vocab.get(curr_token, vocab["<UNK>"]))

    # If longer than max allowed length, truncate sequence; otherwise pad with a special symbol
    if len(seq) > max_seq_len:
        encoded_seq = encoded_seq[: max_seq_len]
    else:
        encoded_seq += [vocab["<PAD>"]] * (max_seq_len - len(seq))

    return encoded_seq


def get_clusters(preds):
    """ Convert {antecedent_id: mention_id} pairs into {mention_id: assigned_cluster_id} pairs. """
    cluster_assignments = {}

    for id_cluster, cluster_starter in enumerate(preds.get(None, [])):
        stack = [cluster_starter]
        curr_cluster = []
        while len(stack) > 0:
            cur = stack.pop()
            curr_cluster.append(cur)
            cluster_assignments[cur] = id_cluster
            mentions = preds.get(cur)
            if mentions is not None:
                stack.extend(mentions)

    return cluster_assignments


def split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15):
    """
    Splits documents array into three sets: learning, validation & testing.
    If random seed is given, documents selected for each set are randomly picked (but do not overlap, of course).
    """
    # Note: test_prop is redundant, but it's left in to make it clear this is a split into 3 parts
    test_prop = 1.0 - train_prop - dev_prop

    train_docs, dev_test_docs = train_test_split(documents, test_size=(dev_prop + test_prop))

    dev_docs, test_docs = train_test_split(dev_test_docs, test_size=test_prop/(dev_prop + test_prop))

    logging.info(f"{len(documents)} documents split to: training set ({len(train_docs)}), dev set ({len(dev_docs)}) "
                 f"and test set ({len(test_docs)}).")

    return train_docs, dev_docs, test_docs


def fixed_split(documents, dataset):
    tr, dev, te = read_splits(os.path.join("..", "data", "seeded_split", f"{dataset}.txt"))
    assert (len(tr) + len(dev) + len(te)) == len(documents)

    train_docs = list(filter(lambda doc: doc.doc_id in tr, documents))
    dev_docs = list(filter(lambda doc: doc.doc_id in dev, documents))
    te_docs = list(filter(lambda doc: doc.doc_id in te, documents))
    return train_docs, dev_docs, te_docs


def read_splits(file_path):
    with open(file_path, "r") as f:
        doc_ids = []
        # train, dev, test
        for _ in range(3):
            curr_ids = set(f.readline().strip().split(","))
            doc_ids.append(curr_ids)

        return doc_ids


if __name__ == "__main__":
    """ 'rc_1' and 'rc_3' are first mentions of some entity,
        'rc_2' and 'rc_5' refer to 'rc_1', etc. """
    preds = {
        None: ['rc_1', 'rc_3'],
        'rc_1': ['rc_2', 'rc_5'],
        'rc_2': ['rc_4'],
        'rc_5': ['rc_6', 'rc_11']
    }

    print(get_clusters(preds))
