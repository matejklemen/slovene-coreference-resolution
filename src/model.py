import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMentionScorer(nn.Module):
    # AverageEmbedding -> Linear
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.linear = nn.Linear(emb_size, 2)

    def forward(self, seq):
        # seq: [batch_size, max_len]
        avg_embedded = torch.mean(self.emb(seq), dim=1)
        return F.softmax(self.linear(avg_embedded), dim=-1)


class SimpleCorefScorer(nn.Module):
    # [AverageEmbedding(s1), AverageEmbedding(s2)] -> Linear
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.linear = nn.Linear(2 * emb_size, 1)

    def forward(self, seq1, seq2):
        # seq1, seq2: [batch_size, max_len]
        avg_emb1 = torch.mean(self.emb(seq1), dim=1)
        avg_emb2 = torch.mean(self.emb(seq2), dim=1)
        logits = self.linear(torch.cat((avg_emb1, avg_emb2), dim=1))
        return torch.sigmoid(logits)
