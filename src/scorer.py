import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCoreferencePairScorer(nn.Module):
    def __init__(self, num_features, hidden_size=150, dropout=0.2):
        # Note: num_features is either hidden_size of a LSTM or 2*hidden_size if using biLSTM
        super().__init__()

        # Attempts to model head word (""key word"") in a mention, e.g. [model] in "my amazing model"
        self.attention_projector = nn.Linear(in_features=num_features, out_features=1)
        self.dropout = nn.Dropout(p=dropout)

        # Converts [candidate_state, head_state, candidate_state * head_state] into a score
        # self.fc = nn.Linear(in_features=(3 * num_features) * 3, out_features=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=((3 * num_features) * 3), out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, candidate_features, head_features):
        """
        Note: doesn't handle batches!
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
