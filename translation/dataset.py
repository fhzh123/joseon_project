# Import Module
import sentencepiece as spm
from itertools import chain
from random import random, randrange

import torch
from torch.utils.data.dataset import Dataset

class HanjaKoreanDataset(Dataset):
    def __init__(self, hanja_list, korean_list, king_list, min_len=4, src_max_len=300, trg_max_len=360,
                 bos_id=1, eos_id=2):
        # hanja_list, korean_list = zip(*[(h, k) for h, k in zip(hanja_list, korean_list)\
        #     if min_len <= len(h) <= src_max_len and min_len <= len(k) <= trg_max_len])
        # King list version
        hanja_list, korean_list, king_list = zip(*[(h, k, king) for h, k, king in zip(hanja_list, korean_list, king_list)\
            if min_len <= len(h) <= src_max_len and min_len <= len(k) <= trg_max_len])

        print('hk', len(hanja_list))
        self.hanja_korean = [(h, k, king) for h, k, king in zip(hanja_list, korean_list, king_list)]
        self.hanja_korean = sorted(self.hanja_korean, key=lambda x: len(x[0])+len(x[1]))
        self.hanja_korean = self.hanja_korean[-1000:] + self.hanja_korean[:-1000]
        self.num_data = len(self.hanja_korean)
        
    def __getitem__(self, index):
        hanja, korean, king = self.hanja_korean[index]
        return hanja, korean, king
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences

        sentences_list = zip(*batch)
        sentences_list = [pack_sentence(sentences) for sentences in sentences_list]        
        return tuple(sentences_list)

    def __call__(self, batch):
        return self.pad_collate(batch)