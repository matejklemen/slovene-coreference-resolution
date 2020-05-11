from collections import Counter
import logging
import numpy as np
from sklearn.model_selection import train_test_split


PAD_TOKEN, PAD_ID = "<PAD>", 0
BOS_TOKEN, BOS_ID = "<BOS>", 1
EOS_TOKEN, EOS_ID = "<EOS>", 2
UNK_TOKEN, UNK_ID = "<UNK>", 3


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
