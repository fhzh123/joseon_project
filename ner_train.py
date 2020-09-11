# Import Module
import os
import math
import time
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.utils.data import DataLoader

# Import Custom Module
from named_entity_recognition.dataset import CustomDataset, PadCollate
from named_entity_recognition.model import NER_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'ner_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        hj_train_indices = data_['hj_train_indices']
        hj_test_indices = data_['hj_test_indices']
        ner_train_indices = data_['ner_train_indices']
        ner_test_indices = data_['ner_test_indices']
        king_train_indices = data_['king_train_indices']
        king_test_indices = data_['king_test_indices']
        id2word = data_['id2word']
        word2id = data_['word2id']
        del data_

    with open(os.path.join(args.save_path, 'emb_mat.pkl'), 'rb') as f:
        emb_mat = pickle.load(f)

    dataset_dict = {
        'train': CustomDataset(hj_train_indices, ner_train_indices, king_train_indices,
                            min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(hj_test_indices, ner_test_indices, king_test_indices,
                            min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    print("Instantiating models...")
    model = NER_model(emb_mat=emb_mat, word2id=word2id, pad_idx=args.pad_idx, bos_idx=args.bos_idx, eos_idx=args.eos_idx, max_len=args.max_len,
                    d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                    dim_feedforward=args.dim_feedforward, n_layers=args.n_layers, dropout=args.dropout,
                    crf_loss=args.crf_loss, device=device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    print(model)

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_f1 = None
    total_train_loss_list = list()
    total_test_loss_list = list()
    freq = 0
    for e in range(args.num_epoch):
        start_time_e = time.time()
        print(f'Model Fitting: [{e+1}/{args.num_epoch}]')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
                val_f1 = 0
                val_loss = 0
            for src, trg, king_id in tqdm(dataloader_dict[phase]):
                # Sourcen, Target sentence setting
                src = src.to(device)
                trg = trg.to(device)
                king_id = king_id.to(device)
                
                # Optimizer setting
                optimizer.zero_grad()

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    if args.crf_loss:
                        mask = torch.where(src.cpu()!=0,torch.tensor(1),torch.tensor(0))
                        mask = torch.tensor(mask, dtype=torch.float).byte()
                        mask = mask.to(device)
                        output, loss = model(src, king_id, trg)
                        loss = -loss
                    else:
                        output = model(src, king_id)
                        output_flat = output.transpose(0,1)[1:].transpose(0,1).contiguous().view(-1, 9)
                        trg_flat = trg.transpose(0,1)[1:].transpose(0,1).contiguous().view(-1)
                        loss = criterion(output_flat, trg_flat)
                    if phase == 'valid':
                        val_loss += loss.item()
                        output_list = output_flat.max(dim=1)[1].tolist()
                        real_list = trg_flat.tolist()
                        f1_val = f1_score(real_list, output_list, average='macro')
                        val_f1 += f1_val
                # If phase train, then backward loss and step optimizer and scheduler
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    total_train_loss_list.append(loss.item())

                    # Print loss value only training
                    freq += 1
                    if freq == args.print_freq:
                        total_loss = loss.item()
                        if args.crf_loss:
                            # Mask vector processing
                            mask_ = list()
                            for x_ in mask.tolist():
                                mask_.append([1 if x==0 else x for x in x_])
                            mask_ = torch.tensor(mask_).byte()
                            output_list = model.crf.viterbi_decode(output, mask_)
                            output_list = list(itertools.chain.from_iterable(output_list))
                            real_list = trg.tolist()
                            real_list = list(itertools.chain.from_iterable(real_list))
                        else:
                            output_list = output_flat.max(dim=1)[1].tolist()
                            real_list = trg_flat.tolist()
                        f1_ = f1_score(real_list, output_list, average='macro')
                        print("[loss:%5.2f][pp:%5.2f][f1:%5.2f]" % (total_loss, math.exp(total_loss), f1_))
                        freq = 0

            # Finishing iteration
            if phase == 'valid':
                val_loss /= len(dataloader_dict['valid'])
                val_f1 /= len(dataloader_dict['valid'])
                total_test_loss_list.append(val_loss)
                print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS | val_f1:%5.2f | spend_time:%5.2fmin"
                        % (e+1, val_loss, math.exp(val_loss), val_f1, (time.time() - start_time_e) / 60))
                if not best_val_f1 or val_f1 > best_val_f1:
                    print("[!] saving model...")
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    torch.save(model.state_dict(), 
                               os.path.join(args.save_path, f'ner_model_{args.crf_loss}.pt'))
                    best_val_f1 = val_f1

        # Learning rate scheduler setting
        scheduler.step()

    pd.DataFrame(total_train_loss_list).to_csv(os.path.join(args.save_path, 'train_loss.csv'), index=False)
    pd.DataFrame(total_test_loss_list).to_csv(os.path.join(args.save_path, 'test_loss.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NER argparser')
    parser.add_argument('--save_path', default='./save', 
                        type=str, help='path of data pickle file (train)')
    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')

    parser.add_argument('--min_len', type=int, default=4, help='Minimum Length of Sentences; Default is 4')
    parser.add_argument('--max_len', type=int, default=150, help='Max Length of Source Sentence; Default is 150')

    parser.add_argument('--num_epoch', type=int, default=10, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size; Default is 48')
    parser.add_argument('--crf_loss', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate; Default is 5e-4')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')

    parser.add_argument('--d_model', type=int, default=512, help='Hidden State Vector Dimension; Default is 512')
    parser.add_argument('--d_embedding', type=int, default=256, help='Embedding Vector Dimension; Default is 256')
    parser.add_argument('--n_head', type=int, default=8, help='Multihead Count; Default is 256')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Embedding Vector Dimension; Default is 512')
    parser.add_argument('--n_layers', type=int, default=8, help='Model layers; Default is 8')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Ratio; Default is 0.5')

    parser.add_argument('--print_freq', type=int, default=300, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()

    main(args)