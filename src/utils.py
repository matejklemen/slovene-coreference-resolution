from collections import Counter

PAD_TOKEN, PAD_ID = "<PAD>", 0
BOS_TOKEN, BOS_ID = "<BOS>", 1
EOS_TOKEN, EOS_ID = "<EOS>", 2
UNK_TOKEN, UNK_ID = "<UNK>", 3


def mention_candidates(sentences, width, end=None):
    """ Extracts mention candidates, consisting of `width` tokens, that appear in interval [0, end) of tokens.
    For example, [['This', 'is', 'a', 'sentence']] with width=2 and end=3 returns mentions [0, 2]
    and [1, 3].

    If `end` is None, returns all mentions of requested width.

    Arguments
    ---------
    sentences: list
        list[list[str]], tokenized sentences (either raw or token IDs)
    width: int
        mention length
    end: int, optional
        boundary up to which mentions are extracted

    Returns
    -------
    list:
        list[list[int]], list of mentions (start and end position)
    """
    effective_end = sum([len(s) for s in sentences]) if end is None else end
    mentions = []
    for idx_start in range(0, effective_end - width + 1):
        mentions.append([idx_start, idx_start + width])

    return mentions


def mention_dataset(doc, width):
    """ Prepares mentions dataset with labels (based on ground truth information).
    Arguments
    ---------
    doc: data.Document
        Document instance

    Returns
    -------
    tuple:
        mention candidates and their labels (is mention / is not mention)
    """
    candidates = mention_candidates(doc.sents, width=width)
    mention_set = set(map(lambda start_end: (start_end[0], start_end[1]),
                          doc.mentions.values()))
    is_mention = [int((s, e) in mention_set) for s, e in candidates]
    return candidates, is_mention


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


if __name__ == "__main__":
    sample_sent = [["The", "way", "I", "see", "it", ",", "every", "life", "is", "a", "pile", "of", "good", "things",
                    "and", "bad", "things", "."],
                   ["The", "good", "things", "do", "n’t", "always", "soften", "the", "bad", "things", ",", "but", "vice",
                    "versa", ",", "the", "bad", "things", "do", "n’t", "always", "spoil", "the", "good", "things", "and",
                    "make", "them" "unimportant", "."]]

    # Ad-hoc test - check that 5 spans get returned in interval [0, 5)
    mentions = mention_candidates(sample_sent, end=5, width=1)
    assert mentions[0][0] == 0 and mentions[0][1] == 1
    assert mentions[-1][0] == 4 and mentions[-1][1] == 5
    assert len(mentions) == 5

