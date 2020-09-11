import torch
import torch.nn as nn
from .token import TokenEmbedding
from .positional import PositionalEmbedding


class TransformerEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information using sin, cos
    sum of all these features are output of Embedding
    """

    def __init__(self, vocab_size, d_model, embed_size, pad_idx=0, max_len=512):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, pad_idx=pad_idx)
        self.linear_layer = nn.Linear(embed_size, d_model)
        self.position = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence):
        x = self.linear_layer(self.token(sequence)) + self.position(sequence)
        return self.norm(x)

class TransformerEmbedding_bilinear(nn.Module):
    def __init__(self, vocab_size, d_model, embed_size, emb_mat, word2id, pad_idx=0, max_len=512):
        super().__init__()
        self.token_dict = dict()
        for i in range(len(emb_mat)):
            self.token_dict[i] = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, pad_idx=0).cuda()
            for word, id_ in word2id.items():
                try:
                    self.token_dict[i].token.weight.data[id_] = emb_mat[i][id_]
                except:
                    continue
        self.norm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Bilinear(embed_size, embed_size, d_model)
        self.king_embedding = nn.Embedding(27, embed_size)
    
    def forward(self, sequence, king_id):
        seq = torch.tensor([]).cuda()
        for i, king_ in enumerate(king_id):
            seq = torch.cat((seq, self.token_dict[king_.item()](sequence[i]).unsqueeze(0)), 0)
        x = self.linear_layer(seq, self.king_embedding(king_id).repeat(1, sequence.size(1), 1))
        return self.norm(x)