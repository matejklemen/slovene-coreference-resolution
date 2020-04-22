from collections import Counter

PAD_TOKEN, PAD_ID = "<PAD>", 0
BOS_TOKEN, BOS_ID = "<BOS>", 1
EOS_TOKEN, EOS_ID = "<EOS>", 2
UNK_TOKEN, UNK_ID = "<UNK>", 3


def extract_vocab(documents, top_n=10_000):
    token_counter = Counter()
    for curr_doc in documents:
        curr_sentences = curr_doc.raw_sentences()

        for sent_tokens in curr_sentences:
            # lowercase tokens to reduce vocabulary size
            processed = list(map(lambda s: s.lower(), sent_tokens))
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


def get_clusters(preds):
    """ Convert {antecedent_id: mention_id} pairs into {mention_id: assigned_cluster_id} pairs. """
    cluster_assignments = {}

    for id_cluster, cluster_starter in enumerate(preds[None]):
        stack = [cluster_starter]
        curr_cluster = []
        while len(stack) > 0:
            cur = stack.pop()
            curr_cluster.append(cur)
            cluster_assignments[cur] = {id_cluster}
            mentions = preds.get(cur)
            if mentions is not None:
                stack.extend(mentions)

    return cluster_assignments


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
