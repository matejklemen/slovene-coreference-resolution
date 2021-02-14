""" This is an auxiliary script that extracts the fastText embeddings of only those subwords that actually appear
    in the data and saves them in a custom structure.
    While this means that the embeddings are only useful for this exact task (otherwise OOV is encountered, which is
    not supposed to happen with fastText), it makes the embeddings more practical to use since you reduce the memory
    use.

    However, the use of this script still requires you to have enough memory to hold all embeddings.
    Produces a directory `TARGET_DIR` with three files:
    - config.json: holds num_embeddings and embedding_dim for constructing a random EmbeddingBag
    - word2inds.json: holds mappings from words, encountered in data, to indices of subwords
    - embeddings.th: holds a state_dict() of EmbeddingBag, that is supposed to be loaded after instantiating the module
"""
import json
import os

import numpy as np
import torch

from src.data import read_corpus
from fasttext import load_model

if __name__ == "__main__":
    documents = read_corpus("coref149") + read_corpus("senticoref")

    model = load_model("/home/matej/Documents/projects/slovene-coreference-resolution/data/cc.sl.100.bin")
    EMBEDDING_DIM = 100
    TARGET_DIR = f"ft_sl_reduced{EMBEDDING_DIM}"

    word2subwords = {}
    subword2ft_index = {}
    ft_index2new = {}

    for token in ["<PAD>", "<UNK>"]:
        subwords, inds = model.get_subwords(token)
        if token not in word2subwords:
            word2subwords[token] = subwords
        for curr_sw, curr_ind in zip(subwords, inds):
            if curr_sw not in subword2ft_index:
                subword2ft_index[curr_sw] = curr_ind
                ft_index2new[curr_ind] = len(ft_index2new)

    for curr_doc in documents:
        for sent in curr_doc.raw_sentences():
            cased_sent = list(map(lambda s: s.strip(), sent))
            uncased_sent = list(map(lambda s: s.lower().strip(), sent))

            for tok in cased_sent:
                subwords, inds = model.get_subwords(tok)
                if tok not in word2subwords:
                    word2subwords[tok] = subwords

                for curr_sw, curr_ind in zip(subwords, inds):
                    if curr_sw not in subword2ft_index:
                        subword2ft_index[curr_sw] = curr_ind
                        ft_index2new[curr_ind] = len(ft_index2new)

            for tok in uncased_sent:
                subwords, inds = model.get_subwords(tok)
                if tok not in word2subwords:
                    word2subwords[tok] = subwords

                for curr_sw, curr_ind in zip(subwords, inds):
                    if curr_sw not in subword2ft_index:
                        subword2ft_index[curr_sw] = curr_ind
                        ft_index2new[curr_ind] = len(ft_index2new)

    word2new = {}
    for word, subwords in word2subwords.items():
        new_indices = [ft_index2new[subword2ft_index[sw]] for sw in subwords]
        word2new[word] = new_indices

    ft_weights = model.get_input_matrix()

    embeddings = np.zeros((len(ft_index2new), EMBEDDING_DIM), dtype=np.float32)
    for idx_ft, idx_new in ft_index2new.items():
        embeddings[idx_new] = ft_weights[idx_ft]

    # Just in case, free up some memory
    del ft_weights
    del model

    new_embeddings = torch.nn.EmbeddingBag(num_embeddings=len(embeddings), embedding_dim=EMBEDDING_DIM)
    new_embeddings.weight.data.copy_(torch.from_numpy(embeddings))

    print(f"Saving reduced embeddings to directory '{TARGET_DIR}'")
    os.makedirs(TARGET_DIR)

    with open(os.path.join(TARGET_DIR, "config.json"), "w", encoding="utf8") as f_config:
        json.dump({
            "num_embeddings": new_embeddings.num_embeddings,
            "embedding_dim": new_embeddings.embedding_dim
        }, fp=f_config, indent=4)

    with open(os.path.join(TARGET_DIR, "word2inds.json"), "w", encoding="utf8") as f:
        json.dump(word2new, fp=f, indent=4)

    torch.save(new_embeddings.state_dict(), os.path.join(TARGET_DIR, f"embeddings.th"))
