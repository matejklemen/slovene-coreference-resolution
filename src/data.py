import xml.etree.ElementTree as ET
import os
import logging

from bs4 import BeautifulSoup

DUMMY_ANTECEDENT = None

#####################
# GLOBAL PARAMETERS
#####################
# Path "./data/*" assumes you are running from root folder, i.e. (python /src/baseline.py)
# Use path "../data/*" if you are running from src folder, i.e. (cd src) and then (python baseline.py)
DATA_DIR = "./data/coref149"
SSJ_PATH = "./data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml"


def _read_tokens(corpus_soup):
    """ Obtain all tokens in current document.

    Arguments
    ---------
    corpus_soup: bs4.element.Tag
        Wrapped XML element containing the document (<tc:TextCorpus ...> tag).

    Returns
    -------
    dict[str, str]:
        Mapping of token IDs to raw tokens
    """
    id_to_tok = {}
    for i, el in enumerate(corpus_soup.findAll("tc:token")):
        token_id = el["id"]
        token = el.text.strip()
        id_to_tok[token_id] = token
    return id_to_tok


def _read_sentences(corpus_soup):
    """ Obtain all sentences in current document.

    Returns
    -------
    tuple:
        (list[list[str]], dict[str, list]):
            (1.) token IDs, organized into sentences
            (2.) token IDs to [index of sentence, index of token inside sentence]
    """
    sent_tok_ids = []
    tok_to_position = {}
    for idx_sent, el in enumerate(corpus_soup.findAll("tc:sentence")):
        token_ids = el["tokenids"].split(" ")
        for idx_tok, tok in enumerate(token_ids):
            tok_to_position[tok] = [idx_sent, idx_tok]
        sent_tok_ids.append(token_ids)
    return sent_tok_ids, tok_to_position


def _read_coreference(corpus_soup):
    """ Obtain all mentions and coreference clusters in current document.

    Returns
    -------
    tuple:
        (dict[str, list[str]], list[list[str]]):
            (1.) mentions
            (2.) mentions organized by coreference cluster
    """
    mentions = {}
    clusters = []
    for cluster_obj in corpus_soup.findAll("tc:entity"):
        curr_cluster = []
        for mention_obj in cluster_obj.findAll("tc:reference"):
            mention_id = mention_obj["id"]
            mention_tokens = mention_obj["tokenids"].split(" ")
            mentions[mention_id] = mention_tokens
            curr_cluster.append(mention_id)

        clusters.append(curr_cluster)
    return mentions, clusters


# Create a dictionary where each mention points to its antecedent (or the dummy antecedent)
def _coreference_chain(clusters_list):
    mapped_clusters = {}
    for curr_cluster in clusters_list:
        for i, curr_mention in enumerate(curr_cluster):
            mapped_clusters[curr_mention] = DUMMY_ANTECEDENT if i == 0 else curr_cluster[i - 1]
    return mapped_clusters


class Mention:
    def __init__(self, mention_id, raw, token_ids, positions=None):
        self.mention_id = mention_id
        self.raw = raw
        self.token_ids = token_ids
        # [idx sentence, idx token inside sentence] for each token
        self.positions = positions

    def __str__(self):
        return f"Mention(\"{' '.join(self.raw)}\")"


class Document:
    def __init__(self, doc_id, tokens, sentences, mentions, clusters, ssj_doc=None, tok_to_position=None):
        self.doc_id = doc_id  # type: str
        self.id_to_tok = tokens  # type: dict
        self.sents = sentences  # type: list
        self.mentions = mentions  # type: dict
        self.clusters = clusters  # type: list
        self.mapped_clusters = _coreference_chain(self.clusters)

        self.ssj_doc = ssj_doc  # type: bs4.element.Tag
        self.tok_to_positon = tok_to_position

    @staticmethod
    def read(file_path, ssj_doc):
        with open(file_path, encoding="utf8") as f:
            content = f.readlines()
            content = "".join(content)
            soup = BeautifulSoup(content, "lxml").find("tc:textcorpus")

        doc_id = file_path.split("/")[-1][:-4]  # = file name without ".tcf"

        # Read data as defined in coref149
        tokens = _read_tokens(soup)
        sents, tok_to_position = _read_sentences(soup)
        mentions, clusters = _read_coreference(soup)

        # Tokens have different IDs in ssj500k, so remap coref149 style to ssj500k style
        idx_sent_coref, idx_token_coref = 0, 0
        _coref_to_ssj = {}
        for curr_sent in ssj_doc.findAll("s"):

            for curr_token in curr_sent.findAll(["w", "pc"]):
                coref_token_id = sents[idx_sent_coref][idx_token_coref]
                ssj_token_id = curr_token["xml:id"]

                # Warn in case tokenization is different between datasets (we are slightly screwed in that case)
                if curr_token.text.strip() != tokens[coref_token_id]:
                    logging.warning(f"MISMATCH! '{curr_token.text.strip()}' (ssj500k ID: {ssj_token_id}) vs "
                                    f"'{tokens[coref_token_id]}' (coref149 ID: {coref_token_id})")

                _coref_to_ssj[coref_token_id] = ssj_token_id
                idx_token_coref += 1
                if idx_token_coref == len(sents[idx_sent_coref]):
                    idx_sent_coref += 1
                    idx_token_coref = 0

        # Correct coref149 token IDs to ssj500k token IDs
        fixed_tokens = {}
        for curr_id, curr_token in tokens.items():
            fixed_tokens[_coref_to_ssj[curr_id]] = curr_token

        fixed_sents = [[_coref_to_ssj[curr_id] for curr_id in curr_sent] for curr_sent in sents]
        fixed_tok_to_position = {_coref_to_ssj[token_id]: position for token_id, position in tok_to_position.items()}

        fixed_mentions = {}
        for mention_id, mention_tokens in mentions.items():
            fixed = list(map(lambda t: _coref_to_ssj[t], mention_tokens))
            fixed_mentions[mention_id] = fixed

        for mention_id, mention_tokens in fixed_mentions.items():
            raw_tokens = [fixed_tokens[t] for t in mention_tokens]
            token_positions = [fixed_tok_to_position[t] for t in mention_tokens]
            fixed_mentions[mention_id] = Mention(mention_id, raw_tokens, mention_tokens, token_positions)

        # __init__(self, doc_id, tokens, sentences, mentions, clusters, ssj_doc=None, tok_to_position=None)
        return Document(doc_id, fixed_tokens, fixed_sents, fixed_mentions, clusters, ssj_doc, tok_to_position)

    def raw_sentences(self):
        """ Returns list of sentences in document. """
        return [list(map(lambda t: self.id_to_tok[t], curr_sent)) for curr_sent in self.sents]

    def __len__(self):
        return len(self.id_to_tok)

    def __str__(self):
        return f"Document('{self.doc_id}', {len(self.id_to_tok)} tokens)"


def read_corpus(corpus_dir, ssj_path):
    logging.info(f"**Reading data from '{ssj_path}'**")
    with open(ssj_path, encoding="utf8") as ssj:
        content = ssj.readlines()
        content = "".join(content)
        ssj_soup = BeautifulSoup(content, "lxml")

    doc_to_soup = {}
    for curr_soup in ssj_soup.findAll("p"):
        doc_to_soup[curr_soup["xml:id"]] = curr_soup

    doc_ids = [f[:-4] for f in os.listdir(corpus_dir)
               if os.path.isfile(os.path.join(corpus_dir, f)) and f.endswith(".tcf")]
    return [Document.read(os.path.join(corpus_dir, f"{curr_id}.tcf"), doc_to_soup[curr_id]) for curr_id in doc_ids]


if __name__ == "__main__":
    documents = read_corpus(DATA_DIR, SSJ_PATH)
