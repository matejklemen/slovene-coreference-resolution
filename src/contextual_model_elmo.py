""" A coreference scorer that is a simplified version of the one described in End-to-end Neural Coreference Resolution.
    We are not performing end-to-end coreference resolution, so we only use the coreference scorer.
    We do not use character embeddings or additional features. For the word embeddings we use pretrained
    ELMo vectors. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids
from data import read_corpus

WEIGHTS_FILE = "../data/slovenian-elmo/slovenian-elmo-weights.hdf5"
OPTIONS_FILE = "../data/slovenian-elmo/options.json"
MAX_SEQ_LEN = 20
HIDDEN_SIZE = 256


class ContextualScorer(nn.Module):
    def __init__(self, num_features, dropout=0.2):
        # Note: num_features is either hidden_size of a LSTM or 2*hidden_size if using biLSTM
        super().__init__()

        # Attempts to model head word (""key word"") in a mention, e.g. [model] in "my amazing model"
        self.attention_projector = nn.Linear(in_features=num_features, out_features=1)
        # Converts [candidate_state, head_state, candidate_state * head_state] into a score
        self.fc = nn.Linear(in_features=(3 * num_features) * 3, out_features=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, candidate_features, head_features):
        """
        Args:
            candidate_features: [num_tokens_cand, num_features]
            head_features: [num_tokens_head, num_features]
        """

        # Create candidate representation
        candidate_attn_weights = F.softmax(self.attention_projector(self.dropout(candidate_features)),
                                           dim=0)
        cand_attended_features = torch.sum(candidate_attn_weights * candidate_features, dim=0)
        candidate_repr = torch.cat((candidate_features[0],  # first word of mention
                                    candidate_features[-1],  # last word of mention
                                    cand_attended_features))

        # Create head mention representation
        head_attn_weights = F.softmax(self.attention_projector(self.dropout(head_features)),
                                      dim=0)
        head_attended_features = torch.sum(head_attn_weights * head_features, dim=0)
        head_repr = torch.cat((head_features[0],  # first word of mention
                               head_features[-1],  # last word of mention
                               head_attended_features))

        # Combine representations and compute a score
        pair_score = self.fc(self.dropout(torch.cat((candidate_repr,
                                                     head_repr,
                                                     candidate_repr * head_repr))))
        return pair_score


if __name__ == "__main__":
    documents = read_corpus("coref149")
    curr_doc = documents[0]
    print(f"Document {curr_doc.doc_id}")
    sents = curr_doc.raw_sentences()
    print("SENTENCES:")
    for curr_sentence in sents:
        print(curr_sentence)

    # num_output_representations = number of different linear combinations of LSTM layers that the model outputs
    elmo = Elmo(OPTIONS_FILE, WEIGHTS_FILE, num_output_representations=1, dropout=0, requires_grad=False)
    encoded_sents = batch_to_ids(sents)

    lstm = nn.LSTM(input_size=1024, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)

    emb_obj = elmo(encoded_sents)
    embeddings = emb_obj["elmo_representations"][0]  # [batch_size, max_seq_len, embedding_size]
    masks = emb_obj["mask"]

    batch_size, max_seq_len = embeddings.shape[0], embeddings.shape[1]

    sample_antecedent = curr_doc.mentions['rc_1']
    sample_head = curr_doc.mentions['rc_2']

    (lstm_encoded_sents, _) = lstm(embeddings)  # shape: [batch_size, max_seq_len, 2 * hidden_size]
    antecedent_features = []
    for curr_token in sample_antecedent.tokens:
        antecedent_features.append(lstm_encoded_sents[curr_token.sentence_index, curr_token.position_in_sentence])

    antecedent_features = torch.stack(antecedent_features, dim=0)  # shape: [num_tokens, 2 * hidden_size]
    coref_scorer = ContextualScorer(2 * HIDDEN_SIZE)

    res = coref_scorer(antecedent_features, antecedent_features)
    print(res)
