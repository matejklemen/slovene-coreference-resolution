import os
import torch
from transformers import BertModel, BertTokenizer
from data import read_corpus
from utils import split_into_sets

CUSTOM_PRETRAINED_BERT_DIR = os.path.join("..", "data", "slo-hr-en-bert-pytorch")


def prepare_document(doc, tokenizer):
    """ Converts a sentence-wise representation of document (list of lists) into a document-wise representation
    (single list) and creates a mapping between the two position indices.

    E.g. a token that is originally in sentence#0 at position#3, might now be broken up into multiple subwords
    at positions [5, 6, 7] in tokenized document."""
    tokenized_doc, mapping = [], {}
    idx_tokenized = 0
    for idx_sent, curr_sent in enumerate(doc.raw_sentences()):
        for idx_inside_sent, curr_token in enumerate(curr_sent):
            tokenized_token = tokenizer.tokenize(curr_token)
            tokenized_doc.extend(tokenized_token)
            mapping[(idx_sent, idx_inside_sent)] = list(range(idx_tokenized, idx_tokenized + len(tokenized_token)))
            idx_tokenized += len(tokenized_token)

    return tokenized_doc, mapping


if __name__ == "__main__":
    all_docs = read_corpus("coref149")
    curr_doc = all_docs[0]

    tokenizer = BertTokenizer.from_pretrained(CUSTOM_PRETRAINED_BERT_DIR)
    context_encoder = BertModel.from_pretrained(CUSTOM_PRETRAINED_BERT_DIR)

    # TODO: enable unfreezing BERT (if there's time)
    for param in context_encoder.parameters():
        param.requires_grad = False

    # maps from (idx_sent, idx_token) to (indices_in_tokenized_doc)
    tokenized_doc, mapping = prepare_document(curr_doc, tokenizer=tokenizer)
    encoded_doc = tokenizer.convert_tokens_to_ids(tokenized_doc)

    # Break down long documents into smaller sub-documents and encode them
    SEGMENT_SIZE = 512 - 2  # 512 - <BOS> - <EOS>
    num_total_segments = (len(encoded_doc) + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    doc_segments = []  # list of `num_total_segments` tensors of shape [1, SEGMENT_SIZE + 2, 768]
    # TODO: segments could first be prepared, then processed in larger batches, but it requires some changes
    for idx_segment in range(num_total_segments):
        curr_segment = tokenizer.prepare_for_model(encoded_doc[idx_segment * SEGMENT_SIZE: (idx_segment + 1) * SEGMENT_SIZE],
                                                   max_length=(SEGMENT_SIZE + 2), pad_to_max_length=True,
                                                   return_tensors="pt")

        res = context_encoder(**curr_segment)
        last_hidden_states = res[0]
        # Note: [0] because assuming a batch size of 1 (i.e. processing 1 segment at a time)
        doc_segments.append(last_hidden_states[0])

    # For each token in mention, get the hidden states of subwords of that token and stack them together
    sample_mention = curr_doc.mentions["rc_1"]
    mention_features = []
    for curr_token in sample_mention.tokens:
        positions_in_doc = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
        for curr_position in positions_in_doc:
            idx_segment = curr_position // (SEGMENT_SIZE + 2)
            idx_inside_segment = curr_position % (SEGMENT_SIZE + 2)
            mention_features.append(doc_segments[idx_segment][idx_inside_segment])

    mention_features = torch.stack(mention_features, dim=0)

    # TODO: reuse the scorer from ELMo
    # ...



