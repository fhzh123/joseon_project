# Import Module
import sentencepiece as spm
from itertools import chain
from random import random, randrange

import torch
from torch.utils.data.dataset import Dataset

# class HanjaKoreanDataset(Dataset):
#     def __init__(self, hanja_list, korean_list, king_list, min_len=4, max_len=300,
#                  bos_id=1, eos_id=2):
#         # hanja_list, korean_list = zip(*[(h, k) for h, k in zip(hanja_list, korean_list)\
#         #     if min_len <= len(h) <= src_max_len and min_len <= len(k) <= trg_max_len])
#         # King list version
#         hanja_list, korean_list, king_list = zip(*[(h, k, king) for h, k, king in zip(hanja_list, korean_list, king_list)\
#             if min_len <= len(h) <= max_len and min_len <= len(k) <= max_len])

#         print('hk', len(hanja_list))
#         self.hanja_korean = [(h, k, king) for h, k, king in zip(hanja_list, korean_list, king_list)]
#         self.hanja_korean = sorted(self.hanja_korean, key=lambda x: len(x[0])+len(x[1]))
#         self.hanja_korean = self.hanja_korean[-1000:] + self.hanja_korean[:-1000]
#         self.num_data = len(self.hanja_korean)
        
#     def __getitem__(self, index):
#         hanja, korean, king = self.hanja_korean[index]
#         return hanja, korean, king
    
#     def __len__(self):
#         return self.num_data

class CustomDataset(Dataset):
    def __init__(self, src_list, trg_list, king_list, 
                 pad_idx=0, min_len=4, max_len=300):
        src_list, trg_list = zip(*[(h, k) for h, k in zip(src_list, trg_list)\
             if len(h) <= max_len and len(k) <= max_len])
        self.hanja_korean = [(h, k) for h, k in zip(src_list, trg_list) \
            if len(h) >= min_len and len(k) >= min_len]
        self.king_list = king_list
        self.num_data = len(self.hanja_korean)
        
    def __getitem__(self, index):
        src_, trg_ = self.hanja_korean[index]
        king_ = self.king_list[index]
        return src_, trg_, king_
        
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0, max_len=None):
        self.dim = dim
        self.pad_index = pad_index
        self.max_len = max_len
        
    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)
        
        (src_, trg_, king_) = zip(*batch)
        batch_size = len(src_)
        
        ### for input_items desc
        # find longest sequence
        input_seq_len = max(map(lambda x: len(x), src_))
        # pad according to max_len
        src_ = [pad_tensor(torch.LongTensor(seq), input_seq_len, self.dim) for seq in src_]
        src_ = torch.cat(src_)
        src_ = src_.view(batch_size, input_seq_len)

        ### for target_items desc
        output_seq_len = max(map(lambda x: len(x), trg_))
        # pad according to max_len
        trg_ = [pad_tensor(torch.LongTensor(seq), output_seq_len, self.dim) for seq in trg_]
        trg_ = torch.cat(trg_)
        trg_ = trg_.view(batch_size, output_seq_len)
        
        ### for king_list
        king_ = torch.LongTensor(king_).unsqueeze(1)
        
        return src_, trg_, king_

    def __call__(self, batch):
        return self.pad_collate(batch)