import os
import re
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import sentencepiece as spm
from glob import glob
from tqdm import tqdm
from gensim.corpora import Dictionary
from collections import deque, Counter

import torch

from dynamic_bernoulli_embeddings.training import train_model

def main(args):

    # Model Saving
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Data List Setting
    data_path = glob(os.path.join(args.data_path, '*.json'))
    data_path = sorted(data_path)[:-1] # 순종부록 제거

    king_list = list()
    king_index_list = list()
    total_record = list()
    comment_list = list()
    for_parsing_list = list()

    # start_time = time.time()
    for ix, path in enumerate(tqdm(data_path)):
        with open(path, 'r') as f:
            record_list = json.load(f)
            king_list.append(path.split(' ')[-1][:2])
            king_index_list.append(ix)

            total_record_by_king = list()
            for rc in record_list:
                total_record.append(rc['korean'])
                total_record_by_king.append(rc['korean'])
            for_parsing_list.append(total_record_by_king)
            total_record_by_king_str = ' '.join(total_record_by_king)
            comment_list.append(total_record_by_king_str)

    # 1) Make Korean text to train vocab
    with open(f'{args.save_path}/korean.txt', 'w') as f:
        for korean in total_record:
            f.write(f'{korean}\n')

    # 2) SentencePiece model training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.save_path}/korean.txt --model_prefix={args.save_path}/m_korean '
        f'--vocab_size={args.vocab_size} --character_coverage=0.995 --split_by_whitespace=true '
        f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx}')

    # 3) Korean vocabulary setting
    korean_vocab = list()
    with open(f'{args.save_path}/m_korean.vocab') as f:
        for line in f:
            korean_vocab.append(line[:-1].split('\t')[0])
    korean_word2id = {w: i for i, w in enumerate(korean_vocab)}

    # 4) SentencePiece model load
    sp_kr = spm.SentencePieceProcessor()
    sp_kr.Load(f"{args.save_path}/m_korean.model")

    # 5) Korean parsing by SentencePiece model
    korean_pieces = list()
    for comment_by_king_list in tqdm(for_parsing_list):
        id_encoded_list = [sp_kr.EncodeAsPieces(korean)for korean in comment_by_king_list]
        extended_id_list = list()
        for id_encoded in id_encoded_list:
            extended_id_list.extend(id_encoded)
        korean_pieces.append(extended_id_list)

    # Dataset Setting
    dataset = pd.DataFrame({
        'session': king_list,
        'text': korean_pieces,
        'time': king_index_list
    })
    dataset['bow'] = dataset['text'].apply(lambda x: [i for i in x])

    # Generate dictionary.
    dictionary = Dictionary(dataset.bow)
    dictionary.filter_extremes(no_below=15, no_above=1.)
    dictionary.compactify()

    # Model Training
    model, loss_history = train_model(dataset, korean_word2id, validation=None, m=args.minibatch_iteration,
                                      num_epochs=args.num_epochs, notebook=False)

    # Model Saving
    torch.save(model, os.path.join(args.save_path, 'dwe_kr_model.pt'))
    with open(os.path.join(args.save_path, 'kr_word2id.pkl'), 'wb') as f:
        pickle.dump(korean_word2id, f)
    with open(os.path.join(args.save_path, 'emb_mat.pkl'), 'wb') as f:
        pickle.dump(model.get_embeddings(), f)
    loss_history.to_csv(os.path.join(args.save_path, 'loss_history.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DWE argparser')
    # Path Setting
    parser.add_argument('--data_path', type=str, default='../joseon_word_embedding/data/', 
                        help='Data path setting')
    parser.add_argument('--save_path', type=str, default='./save_korean2',
                        help='Save path setting')
    # Training Setting
    parser.add_argument('--num_epochs', type=int, default=5, help='The number of epoch')
    parser.add_argument('--minibatch_iteration', type=int, default=600, help='Mini-batch size')
    parser.add_argument('--pad_idx', default=0, type=int, help='Padding index')
    parser.add_argument('--bos_idx', default=1, type=int, help='Start token index')
    parser.add_argument('--eos_idx', default=2, type=int, help='End token index')
    parser.add_argument('--unk_idx', default=3, type=int, help='Unknown token index')
    parser.add_argument('--vocab_size', default=24000, type=int, help='Korean vocabulary size')
    args = parser.parse_args()

    total_start_time = time.time()
    main(args)
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')