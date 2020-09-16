import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .embedding.transformer_embedding import TransformerEmbedding, TransformerEmbedding_bilinear


class Transformer(nn.Module):
    def __init__(self, emb_mat, src_word2id, src_vocab_num, trg_vocab_num, pad_idx=0, bos_idx=1, 
            eos_idx=2, max_len=300, d_model=512, d_embedding=256, n_head=8, 
            dim_feedforward=2048, num_encoder_layer=8, num_decoder_layer=8, dropout=0.1,
            baseline=False, device=None):

        super(Transformer, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.baseline = baseline

        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        if self.baseline:
            self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding,
                                    pad_idx=self.pad_idx, max_len=self.max_len)
        else:
            self.src_embedding = TransformerEmbedding_bilinear(len(src_word2id.keys()), d_model, d_embedding, emb_mat, src_word2id)
        # Target embedding part
        self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding,
                                pad_idx=self.pad_idx, max_len=self.max_len)

        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num)
        
        # Transformer
        # self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        # self.encoders = nn.ModuleList([
        #     TransformerEncoderLayer(d_model, self_attn, dim_feedforward,
        #         activation='gelu', dropout=dropout) for i in range(num_encoder_layer)])
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, 
                                                    dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layer)

        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        decoder_mask_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(d_model, self_attn, decoder_mask_attn,
                dim_feedforward, activation='gelu', dropout=dropout) \
                    for i in range(num_decoder_layer)])

    def forward(self, src_input_sentence, trg_input_sentence, king_id, tgt_mask, non_pad_position=None):
        src_key_padding_mask = (src_input_sentence == self.pad_idx)
        tgt_key_padding_mask = (trg_input_sentence == self.pad_idx)

        if self.baseline:
            encoder_out = self.src_embedding(src_input_sentence).transpose(0, 1)
        else:
            encoder_out = self.src_embedding(src_input_sentence, king_id).transpose(0, 1)
        decoder_out = self.trg_embedding(trg_input_sentence).transpose(0, 1)

        # for i in range(len(self.encoders)):
        #     encoder_out = self.encoders[i](encoder_out, src_key_padding_mask=src_key_padding_mask)
        encoder_out = self.transformer_encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

        for i in range(len(self.decoders)):            
            decoder_out = self.decoders[i](decoder_out, encoder_out, tgt_mask=tgt_mask,
                                memory_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask)

        decoder_out = decoder_out.transpose(0, 1).contiguous()
        if non_pad_position is not None:
            decoder_out = decoder_out[non_pad_position]

        decoder_out = self.dropout(F.gelu(self.trg_output_linear(decoder_out)))
        decoder_out = self.trg_output_linear2(decoder_out)
        return decoder_out

    def translate_predict(self, src_sentence, device):
        if self.baseline:
            encoder_out = self.src_embedding(src_input_sentence).transpose(0, 1)
        else:
            encoder_out = self.src_embedding(src_input_sentence, king_id).transpose(0, 1)
        predicted = torch.LongTensor([[self.bos_idx]]).to(device)

        for _ in range(self.max_len):
            trg_embs = self.embedding(predicted).transpose(0, 1)
            decoder_out = self.transformer(src_embs, trg_embs)
            decoder_out = F.gelu(self.trg_output_linear(decoder_out))
            y_pred = self.output_linear2(decoder_out).transpose(0, 1).contiguous()
            y_pred_id = y_pred.max(dim=2)[1][-1, 0]

            if y_pred_id == self.eos_idx:
                break

            predicted = torch.cat([predicted, y_pred_id.view(1, 1)], dim=0)
            
        predicted = predicted[1:, 0].cpu().numpy() # remove bos token
        return predicted

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, self_attn, mask_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.multihead_attn = mask_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt