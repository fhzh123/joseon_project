# coding: utf-8
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

class NER_model(nn.Module):
    def __init__(self, emb_mat, word2id, pad_idx=0, bos_idx=1, eos_idx=2, max_len=150, d_model=512, d_embedding=256, n_head=8, 
                 dim_feedforward=2048, n_layers=10, dropout=0.1, device=None):
        super(NER_model, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Source embedding part
        # self.src_embedding = nn.Embedding(7559, d_model) # Need to fix number
        self.src_embedding = TransformerEmbedding(len(word2id.keys()), d_model, d_embedding, emb_mat, word2id)

        # Transformer
        # self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        # self.encoders = nn.ModuleList([
        #     nn.TransformerEncoderLayer(d_model, self_attn, dim_feedforward,
        #         activation='gelu', dropout=dropout) for i in range(n_layers)])
        encoder_layers = nn.TransformerEncoderLayer(d_model, 8, 512, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # Output Linear Part
        self.src_output_linear = nn.Linear(d_model, d_embedding)
        self.src_output_linear2 = nn.Linear(d_embedding, 9)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        # self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.src_output_linear.bias.data.zero_()
        self.src_output_linear.weight.data.uniform_(-initrange, initrange)
        self.src_output_linear2.bias.data.zero_()
        self.src_output_linear2.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, sequence, king_id):
        encoder_out = self.src_embedding(sequence, king_id).transpose(0, 1)
        src_key_padding_mask = sequence == self.pad_idx

        encoder_out = self.transformer_encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

        encoder_out1 = self.dropout(F.gelu(self.src_output_linear(encoder_out)))
        encoder_out2 = self.src_output_linear2(encoder_out1).transpose(0, 1)
        # encoder_out2 = F.softmax(encoder_out2, dim=2)

        return encoder_out2


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, pad_idx=0):
        super().__init__(vocab_size, embed_size, padding_idx=pad_idx)

class TransformerEmbedding(nn.Module):
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