# Import Modules
import math
import random
import numpy as np
from sru import SRU

# Import PyTorch
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from .embedding.transformer_embedding import TransformerEmbedding, TransformerEmbedding_bilinear

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, emb_mat, src_word2id,
                 n_layers=1, pad_idx=0, dropout=0.0, embedding_dropout=0.0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.sru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        # self.embedding = nn.Embedding(input_size, embed_size, padding_idx=pad_idx)  # embedding layer
        self.embedding = TransformerEmbedding_bilinear(len(src_word2id.keys()), 
                        self.hidden_size, self.embed_size, emb_mat, src_word2id)
        

    def forward(self, src, king_id, hidden=None, cell=None):
        # Source sentence embedding
        embeddings = self.embedding(src, king_id)  # (max_caption_length, batch_size, embed_dim)
        embedded = F.dropout(embeddings, p=self.embedding_dropout, inplace=True) # (max_caption_length, batch_size, embed_dim)
        # Bidirectional SRU
        outputs, hidden = self.sru(embedded, hidden)
        # sum bidirectional outputs
        outputs = torch.tanh(self.linear(outputs)) # (max_caption_length, batch_size, embed_dim)
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size, 1))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # Setting for attention mechanism
        timestep = encoder_outputs.size(0) # (max_caption_length)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) # (batch_size, max_caption_length, hidden)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch_size, max_caption_length, hidden)
        # Calculate attention energy score
        attn_energies = self.score(h, encoder_outputs) # (max_caption_length, batch_size)
        return torch.softmax(attn_energies, dim=1).unsqueeze(1) # (batch_size, 1, max_caption_length)

    def score(self, hidden, encoder_outputs):
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # (batch_size, max_caption_length, hidden)
        energy = (energy @ self.v).squeeze(2) # (batch_size, max_caption_length)
        # Old codes
        #energy = energy.transpose(1, 2)  # [B*H*T]
        #v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        #energy = torch.bmm(v, energy)  # [B*1*T]
        return energy

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, pad_idx=0, dropout=0.0, embedding_dropout=0.0):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.dropout = dropout
        self.embedding_dropout = embedding_dropout

        self.embed = nn.Embedding(output_size, embed_size, padding_idx=pad_idx)
        self.attention = Attention(hidden_size)
        self.sru = nn.GRU(hidden_size + embed_size, hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = F.dropout(self.embed(input), p=self.embedding_dropout, inplace=True).unsqueeze(0)  # (1, batch_size, embed_size)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs) # (max_caption_length, 1, batch_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))  # (batch_size, 1, embed_size)
        context = context.transpose(0, 1) # (1, batch_size, embed_size)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2) # (1, batch_size, embed + hidden)
        output, hidden = self.sru(rnn_input, last_hidden) # (1, batch_size, hidden)
        # context = context.squeeze(0)
        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output.squeeze(0))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, dropout=0.0):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(self.encoder.hidden_size, self.decoder.hidden_size)
        self.dropout = dropout
        self.device = device

    def forward(self, src, trg, king_id, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
        # Encoding source sentences
        encoder_output, hidden = self.encoder(src, king_id)
        hidden = hidden[-self.decoder.n_layers:]
        # hidden = torch.tanh(self.encoder.linear(hidden))
        hidden = torch.tanh(self.linear(hidden))
        # Decoding
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            # Teacher forcing mechanism
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if is_teacher:
                output = Variable(trg.data[t].to(self.device))
            else:
                output = top1.to(self.device)
        return outputs