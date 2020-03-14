import xml.etree.ElementTree as ET
import os

NAMESPACE = {"tc": "http://www.dspin.de/data/textcorpus"}


def _read_tokens(corpus_xml):
    """ Obtain all tokens in current document.

    Arguments
    ---------
    corpus_xml: xml.etree.ElementTree.Element
        Wrapped XML element containing the document (<tc:TextCorpus ...> tag).

    Returns
    -------
    tuple:
        (dict[str, str], dict[str, int]):
        (1.) mapping of token IDs to raw tokens
        (2.) mapping of token IDs to positions inside doc
    """
    id_to_tok, tok_to_pos = {}, {}
    for i, el in enumerate(corpus_xml.find("tc:tokens", NAMESPACE).findall("tc:token", NAMESPACE)):
        token_id = el.attrib["ID"]
        token = el.text
        id_to_tok[token_id] = token
        tok_to_pos[token_id] = i
    return id_to_tok, tok_to_pos


def _read_sentences(corpus_xml):
    """ Obtain all sentences in current document.

    Returns
    -------
    list:
        list[list[str]], containing token IDs, organized into sentences. """
    sent_tok_ids = []
    for el in corpus_xml.find("tc:sentences", NAMESPACE).findall("tc:sentence", NAMESPACE):
        token_ids = el.attrib["tokenIDs"].split(" ")
        sent_tok_ids.append(token_ids)
    return sent_tok_ids


def _read_coreference(corpus_xml):
    """ Obtain all mentions and coreference clusters in current document.

    Returns
    -------
    tuple:
        (dict[str, list[str]], list[list[str]]):
            (1.) mentions
            (2.) mentions organized by coreference cluster
    """
    id_to_mentions = {}
    coref_clusters = []
    for el in corpus_xml.find("tc:references", NAMESPACE).findall("tc:entity", NAMESPACE):
        mention_els = el.findall("tc:reference", NAMESPACE)
        curr_cluster = []
        for m in mention_els:
            # TODO? target mention ID (to make specific mention pairs if needed)
            mention_id = m.attrib["ID"]
            token_ids = m.attrib["tokenIDs"].split(" ")
            id_to_mentions[mention_id] = token_ids
            curr_cluster.append(mention_id)
        coref_clusters.append(curr_cluster)
    return id_to_mentions, coref_clusters


class Document:
    def __init__(self, file_name, tokens, sentences, mentions, clusters):
        self.name = file_name  # type: str
        self.id_to_tok = tokens  # type: dict
        self.sents = sentences  # type: list
        self.mentions = mentions  # type: dict
        self.clusters = clusters  # type: list

    @staticmethod
    def read(file_path):
        curr_doc = ET.parse(file_path).find("tc:TextCorpus", NAMESPACE)
        tokens, positions = _read_tokens(curr_doc)
        sents = _read_sentences(curr_doc)
        mentions, corefs = _read_coreference(curr_doc)
        # instead of saving all tokens, save start and end position of mentions inside document for easier comparison
        mapped_mentions = {}
        for m_id, m_tokens in mentions.items():
            start_pos = positions[m_tokens[0]]
            mapped_mentions[m_id] = [start_pos, start_pos + len(m_tokens)]

        return Document(file_path, tokens, sents, mapped_mentions, corefs)

    def raw_sentences(self):
        """ Returns list of sentences in document. """
        return [list(map(lambda t: self.id_to_tok[t], curr_sent)) for curr_sent in self.sents]

    def __len__(self):
        return len(self.id_to_tok)

    def __str__(self):
        return f"Document('{self.name}', {len(self.id_to_tok)} tokens)"


def read_corpus(corpus_dir):
    doc_fnames = [f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f)) and f.endswith(".tcf")]
    return [Document.read(os.path.join(corpus_dir, curr_fname)) for curr_fname in doc_fnames]


if __name__ == "__main__":
    DATA_DIR = "/home/matej/Documents/mag/2-letnik/obdelava_naravnega_jezika/coref149"
    print(f"**Reading data from '{DATA_DIR}'**")
    documents = read_corpus(DATA_DIR)
    print(f"**Read {len(documents)} documents**")
