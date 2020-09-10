import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, str_list, ner_list, king_list, 
                 pad_idx=0, min_len=4, max_len=150):
        str_list, ner_list = zip(*[(h, k) for h, k in zip(str_list, ner_list)\
             if len(h) <= max_len and len(k) <= max_len])
        self.str_ner = [(h, k) for h, k in zip(str_list, ner_list) \
            if len(h) >= min_len and len(k) >= min_len]
        self.king_list = king_list
        self.num_data = len(self.str_ner)
        
    def __getitem__(self, index):
        str_, ner_ = self.str_ner[index]
        king_ = self.king_list[index]
        return str_, ner_, king_
        
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
        
        (str_, ner_, king_) = zip(*batch)
        batch_size = len(str_)
        
        ### for input_items desc
        # find longest sequence
        input_seq_len = max(map(lambda x: len(x), str_))
        # pad according to max_len
        str_ = [pad_tensor(torch.LongTensor(seq), input_seq_len, self.dim) for seq in str_]
        str_ = torch.cat(str_)
        str_ = str_.view(batch_size, input_seq_len)

        ### for target_items desc
        output_seq_len = max(map(lambda x: len(x), ner_))
        # pad according to max_len
        ner_ = [pad_tensor(torch.LongTensor(seq), output_seq_len, self.dim) for seq in ner_]
        ner_ = torch.cat(ner_)
        ner_ = ner_.view(batch_size, output_seq_len)
        
        ### for king_list
        king_ = torch.LongTensor(king_).unsqueeze(1)
        
        return str_, ner_, king_

    def __call__(self, batch):
        return self.pad_collate(batch)