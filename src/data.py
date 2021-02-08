import os
import logging
import csv
import pandas as pd
from collections import OrderedDict

from bs4 import BeautifulSoup

DUMMY_ANTECEDENT = None

#####################
# GLOBAL PARAMETERS
#####################
# Path "./data/*" assumes you are running from root folder, i.e. (python /src/baseline.py)
# Use path "../data/*" if you are running from src folder, i.e. (cd src) and then (python baseline.py)
COREF149_DIR = os.environ.get("COREF149_DIR", "../data/coref149")
SENTICOREF_DIR = os.environ.get("SENTICOREF149_DIR", "../data/senticoref1_0")
SENTICOREF_METADATA_DIR = "../data/senticoref_pos_stanza"
SSJ_PATH = os.environ.get("SSJ_PATH", "../data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml")


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
    id_to_tok = OrderedDict()
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


class Token:
    def __init__(self, token_id, raw_text, lemma, msd, sentence_index, position_in_sentence, position_in_document):
        self.token_id = token_id

        self.raw_text = raw_text
        self.lemma = lemma
        self.msd = msd

        self.sentence_index = sentence_index
        self.position_in_sentence = position_in_sentence
        self.position_in_document = position_in_document

        self.gender = self._extract_gender(msd)
        self.number = self._extract_number(msd)
        self.category = msd[0]

    def __str__(self):
        return f"Token(\"{self.raw_text}\")"

    def _extract_number(self, msd_string):
        number = None
        if msd_string[0] == "S" and len(msd_string) >= 4:  # noun/samostalnik
            number = msd_string[3]
        elif msd_string[0] == "G" and len(msd_string) >= 6:  # verb/glagol
            number = msd_string[5]
        # P = adjective (pridevnik), Z = pronoun (zaimek), K = numeral (števnik)
        elif msd_string[0] in {"P", "Z", "K"} and len(msd_string) >= 5:
            number = msd_string[4]

        return number

    def _extract_gender(self, msd_string):
        gender = None
        if msd_string[0] == "S" and len(msd_string) >= 3:  # noun/samostalnik
            gender = msd_string[2]
        elif msd_string[0] == "G" and len(msd_string) >= 7:  # verb/glagol
            gender = msd_string[6]
        # P = adjective (pridevnik), Z = pronoun (zaimek), K = numeral (števnik)
        elif msd_string[0] in {"P", "Z", "K"} and len(msd_string) >= 4:
            gender = msd_string[3]

        return gender


class Mention:
    def __init__(self, mention_id, tokens):
        self.mention_id = mention_id
        self.tokens = tokens

    def __str__(self):
        return f"Mention(\"{' '.join([tok.raw_text for tok in self.tokens])}\")"

    def raw_text(self):
        return " ".join([t.raw_text for t in self.tokens])

    def lemma_text(self):
        return " ".join([t.lemma for t in self.tokens if t.lemma is not None])


class Document:
    def __init__(self, doc_id, tokens, sentences, mentions, clusters,
                 metadata=None):
        self.doc_id = doc_id  # type: str
        self.tokens = tokens  # type: dict
        self.sents = sentences  # type: list
        self.mentions = mentions  # type: dict
        self.clusters = clusters  # type: list
        self.mapped_clusters = _coreference_chain(self.clusters)
        self.metadata = metadata

    def raw_sentences(self):
        """ Returns list of sentences in document. """
        return [list(map(lambda t: self.tokens[t].raw_text, curr_sent)) for curr_sent in self.sents]

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return f"Document('{self.doc_id}', {len(self.tokens)} tokens)"


def sorted_mentions_dict(mentions):
    # sorted() produces an array of (key, value) tuples, which we turn back into dictionary
    sorted_mentions = dict(sorted(mentions.items(),
                                  key=lambda tup: (tup[1].tokens[0].sentence_index,  # sentence
                                                   tup[1].tokens[0].position_in_sentence,  # start pos
                                                   tup[1].tokens[-1].position_in_sentence)))  # end pos

    return sorted_mentions


def read_senticoref_doc(file_path):
    # Temporary cluster representation:
    # {cluster1 index: { mention1_idx: ['mention1', 'tokens'], mention2_idx: [...] }, cluster2_idx: {...} }
    _clusters = {}
    # Temporary buffer for current sentence
    _curr_sent = []

    sents = []
    id_to_tok = {}
    tok_to_position = {}
    idx_sent, idx_inside_sent = 0, 0
    mentions, clusters = {}, []

    doc_id = file_path.split(os.path.sep)[-1][:-4]  # = file name without ".tsv"
    # Note: `quoting=csv.QUOTE_NONE` is required as otherwise some documents can't be read
    # Note: `keep_default_na=False` is required as there's a typo in corpus ("NA"), interpreted as <missing>
    curr_annotations = pd.read_table(file_path, comment="#", sep="\t", index_col=False, quoting=csv.QUOTE_NONE,
                                     names=["token_index", "start_end", "token", "NamedEntity", "Polarity",
                                            "referenceRelation", "referenceType"], keep_default_na=False)
    curr_metadata = pd.read_table(os.path.join(SENTICOREF_METADATA_DIR, f"{doc_id}.tsv"), sep="\t", index_col=False,
                                  quoting=csv.QUOTE_NONE, header=0, keep_default_na=False)

    metadata = {"tokens": {}}
    for i, (tok_id, ref_info, token) in enumerate(curr_annotations[["token_index", "referenceRelation", "token"]].values):
        # Token is part of some mention
        if ref_info != "_":
            # Token can be part of multiple mentions
            ref_annotations = ref_info.split("|")

            for mention_info in ref_annotations:
                cluster_idx, mention_idx = list(map(int, mention_info[3:].split("-")))  # skip "*->"

                curr_mentions = _clusters.get(cluster_idx, {})
                curr_mention_tok_ids = curr_mentions.get(mention_idx, [])
                curr_mention_tok_ids.append(tok_id)
                curr_mentions[mention_idx] = curr_mention_tok_ids

                _clusters[cluster_idx] = curr_mentions

        _curr_sent.append(tok_id)
        tok_to_position[tok_id] = [idx_sent, idx_inside_sent]
        id_to_tok[tok_id] = token
        idx_inside_sent += 1

        text, pos_tag, lemma = curr_metadata.iloc[i].values
        metadata["tokens"][tok_id] = {"ana": pos_tag, "lemma": lemma, "text": text}

        # Segment sentences heuristically
        if token in {".", "!", "?"}:
            idx_sent += 1
            idx_inside_sent = 0
            sents.append(_curr_sent)
            _curr_sent = []

    # If the document doesn't end with proper punctuation
    if len(_curr_sent) > 0:
        sents.append(_curr_sent)

    # --- generate token objects
    final_tokens = OrderedDict()
    for index, (tok_id, tok_raw) in enumerate(id_to_tok.items()):
        final_tokens[tok_id] = Token(
            tok_id,
            tok_raw,
            metadata["tokens"][tok_id]["lemma"] if "lemma" in metadata["tokens"][tok_id] else None,
            metadata["tokens"][tok_id]["ana"].split(":")[1],
            tok_to_position[tok_id][0],
            tok_to_position[tok_id][1],
            index
        )
    # ---

    mention_counter = 0
    for idx_cluster, curr_mentions in _clusters.items():
        curr_cluster = []
        for idx_mention, mention_tok_ids in curr_mentions.items():
            # assign coref149-style IDs to mentions
            mention_id = f"rc_{mention_counter}"
            mention_tokens = list(map(lambda tok_id: final_tokens[tok_id], mention_tok_ids))
            mentions[mention_id] = Mention(mention_id, mention_tokens)

            curr_cluster.append(mention_id)
            mention_counter += 1
        clusters.append(curr_cluster)

    return Document(doc_id, final_tokens, sents, sorted_mentions_dict(mentions), clusters, metadata=metadata)


def read_coref149_doc(file_path, ssj_doc):
    with open(file_path, encoding="utf8") as f:
        content = f.readlines()
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml").find("tc:textcorpus")

    doc_id = file_path.split(os.path.sep)[-1][:-4]  # = file name without ".tcf"

    # Read data as defined in coref149
    tokens = _read_tokens(soup)
    sents, tok_to_position = _read_sentences(soup)
    mentions, clusters = _read_coreference(soup)

    # Tokens have different IDs in ssj500k, so remap coref149 style to ssj500k style
    idx_sent_coref, idx_token_coref = 0, 0
    _coref_to_ssj = {} # mapping from coref ids to ssj ids
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

    # sentences are composed of ssj token IDs
    fixed_sents = [[_coref_to_ssj[curr_id] for curr_id in curr_sent] for curr_sent in sents]

    # Write all metadata for tokens
    # Note: currently not writing SRL/dependency metadata
    metadata = {"tokens": {}}
    for token in ssj_doc.findAll(["w", "c", "pc"]):
        token_id = token.get("xml:id", None)

        if token_id:
            metadata["tokens"][token_id] = token.attrs
            metadata["tokens"][token_id]["text"] = token.text

    final_tokens = OrderedDict()
    for index, (coref_token_id, raw_text) in enumerate(tokens.items()):
        ssj_token_id = _coref_to_ssj[coref_token_id]  # mapping of coref token ID to ssj token ID
        final_tokens[ssj_token_id] = Token(
            ssj_token_id,
            raw_text,
            metadata["tokens"][ssj_token_id]["lemma"] if "lemma" in metadata["tokens"][ssj_token_id] else None,
            metadata["tokens"][ssj_token_id]["ana"].split(":")[1],
            tok_to_position[coref_token_id][0],  # Note: tok_to_pos uses coref IDs, not ssj IDs
            tok_to_position[coref_token_id][1],
            index)

    final_mentions = {}
    for mention_id, mention_tokens in mentions.items():
        token_objs = [final_tokens[_coref_to_ssj[tok_id]] for tok_id in mention_tokens]
        final_mentions[mention_id] = Mention(mention_id, token_objs)

    # TODO: is metadata required here? metadata for tokens has been moved to token object
    return Document(doc_id, final_tokens, fixed_sents, sorted_mentions_dict(final_mentions), clusters, metadata=metadata)


def read_corpus(name):
    SUPPORTED_DATASETS = {"coref149", "senticoref"}
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset (must be one of {SUPPORTED_DATASETS})")

    if name == "coref149":
        with open(SSJ_PATH, encoding="utf8") as ssj:
            content = ssj.readlines()
            content = "".join(content)
            ssj_soup = BeautifulSoup(content, "lxml")

        doc_to_soup = {}
        for curr_soup in ssj_soup.findAll("p"):
            doc_to_soup[curr_soup["xml:id"]] = curr_soup

        doc_ids = [f[:-4] for f in os.listdir(COREF149_DIR)
                   if os.path.isfile(os.path.join(COREF149_DIR, f)) and f.endswith(".tcf")]
        return [read_coref149_doc(os.path.join(COREF149_DIR, f"{curr_id}.tcf"), doc_to_soup[curr_id]) for curr_id in doc_ids]
    else:
        doc_ids = [f[:-4] for f in os.listdir(SENTICOREF_DIR)
                   if os.path.isfile(os.path.join(SENTICOREF_DIR, f)) and f.endswith(".tsv")]

        return [read_senticoref_doc(os.path.join(SENTICOREF_DIR, f"{curr_id}.tsv")) for curr_id in doc_ids]


if __name__ == "__main__":
    DATASET_NAME = "senticoref"
    documents = read_corpus(DATASET_NAME)
    print(f"Read {len(documents)} documents")

    # http://nl.ijs.si/ME/Vault/V5/msd/html/msd-sl.html#msd.categories-sl
    if DATASET_NAME == "senticoref":
        # English tags - because tags are predicted with Stanza
        char_tag_to_pos = dict(zip(["N", "V", "A", "R", "P", "M", "S", "C", "Q", "I", "Y", "X", "Z"],
                                   ["samostalnik", "glagol", "pridevnik", "prislov", "zaimek", "števnik",
                                    "predlog", "veznik", "členek", "medmet", "okrajšava", "neuvrščeno", "ločilo"]))
    elif DATASET_NAME == "coref149":
        char_tag_to_pos = dict(zip(["S", "G", "P", "R", "Z", "K", "D", "V", "L", "M", "O", "N", "U"],
                                   ["samostalnik", "glagol", "pridevnik", "prislov", "zaimek", "števnik",
                                    "predlog", "veznik", "členek", "medmet", "okrajšava", "neuvrščeno", "ločilo"]))
    pos_to_idx = {c: i for i, c in enumerate(char_tag_to_pos.values())}
    pos_count = [0 for _ in range(len(pos_to_idx))]
    for doc in documents:
        for mention_id, mention in doc.mentions.items():
            first_token = mention.tokens[0]  # type: Token
            curr_tag = char_tag_to_pos[first_token.msd[0]]

            pos_count[pos_to_idx[curr_tag]] += 1

    print("besedna_vrsta,frekvenca")
    for curr_pos in pos_to_idx:
        print(f"{curr_pos},{pos_count[pos_to_idx[curr_pos]]}")

    entity_size_count = {}  # entity/cluster size -> number of such entities
    mentions_by_documents = {}  # number of mentions -> number of documents with this amount of mentions
    for doc in documents:
        num_mentions = 0
        for curr_cluster in doc.clusters:
            cluster_size = len(curr_cluster)
            num_mentions += cluster_size
            entity_size_count[cluster_size] = entity_size_count.get(cluster_size, 0) + 1

        mentions_by_documents[num_mentions] = mentions_by_documents.get(num_mentions, 0) + 1

    print("\nvelikost_entitete,frekvenca")
    for curr_size, num_mentions in sorted(entity_size_count.items(), key=lambda tup: tup[0]):
        print(f"{curr_size},{num_mentions}")

    print("\nštevilo_omenitev_v_dokumentu,frekvenca")
    for curr_num_mentions, num_docs in sorted(mentions_by_documents.items(), key=lambda tup: tup[0]):
        print(f"{curr_num_mentions},{num_docs}")

    dist_between_mentions = {}  # dist between consecutive mentions (in num. of mentions) -> frequency of this distance
    for doc in documents:
        sorted_mentions = sorted([(mention_id,
                                   curr_mention.tokens[0].position_in_document,
                                   curr_mention.tokens[-1].position_in_document)
                                  for mention_id, curr_mention in doc.mentions.items()],
                                 key=lambda triple: (triple[1], triple[2]))
        mention_id_to_rank = {mention_id: rank for rank, (mention_id, _, _) in enumerate(sorted_mentions)}

        for curr_cluster in doc.clusters:
            sorted_cluster = sorted(curr_cluster, key=lambda m_id: (doc.mentions[m_id].tokens[0].position_in_document,
                                                                    doc.mentions[m_id].tokens[-1].position_in_document))

            for m1_id, m2_id in zip(sorted_cluster, sorted_cluster[1:]):
                # Distance 0 = mentions right next to eachother when ordered by position
                rank_diff = mention_id_to_rank[m2_id] - mention_id_to_rank[m1_id] - 1

                dist_between_mentions[rank_diff] = dist_between_mentions.get(rank_diff, 0) + 1

        print("\nrazdalja_med_zaporednima_omenitvama_iste_entitete,frekvenca")
        for curr_dist, num_mentions in sorted(dist_between_mentions.items(), key=lambda tup: tup[0]):
            print(f"{curr_dist},{num_mentions}")

