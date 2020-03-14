def extract_mentions(sentences, end, width):
    """ Extracts mentions, consisting of `width` tokens, that appear in interval [0, end) of tokens.
    For example, [['This', 'is', 'a', 'sentence']] with width=2 and end=3 returns mentions ['This', 'is']
    and ['is', 'a'].

    Arguments
    ---------
    sentences: list
        list[list[str]], tokenized sentences
    end: int
        boundary up to which mentions are extracted
    width: int
        mention length

    Returns
    -------
    tuple:
        (list[list[str]], list[list[int]]), list of mentions and sentence IDs, to which the mentions belong to
    """
    combined_sents, s_ids = [], []
    for idx_sent, s in enumerate(sentences):
        combined_sents.extend(s)
        s_ids.extend([idx_sent] * len(s))

    mentions, ids = [], []
    for idx_start in range(0, end - width + 1):
        mentions.append(combined_sents[idx_start: idx_start + width])
        ids.append(s_ids[idx_start: idx_start + width])

    return mentions, ids


if __name__ == "__main__":
    s = [["The", "way", "I", "see", "it", ",", "every", "life", "is", "a", "pile", "of", "good", "things", "and",
          "bad", "things", "."],
         ["The", "good", "things", "do", "n’t", "always", "soften", "the", "bad", "things", ",", "but", "vice",
          "versa", ",", "the", "bad", "things", "do", "n’t", "always", "spoil", "the", "good", "things", "and",
          "make", "them" "unimportant", "."]]

    # Ad-hoc tests
    mentions, s_ids = extract_mentions(s, end=5, width=1)
    assert len(mentions[0]) == 1
    assert len(mentions) == 5
    assert all([el == 0 for l in s_ids for el in l])

    # Sentence IDs should be different when crossing boundary between sentences
    mentions, s_ids = extract_mentions(s, end=19, width=2)
    assert len(mentions[0]) == 2
    assert s_ids[-1][0] == 0 and s_ids[-1][1] == 1

