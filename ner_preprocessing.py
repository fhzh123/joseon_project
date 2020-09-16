# Import Module
import os
import time
import json
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

# Import Custom Modules
from utils import terminal_size, train_test_split

def main(args):
    #===================================#
    #============Data Load==============#
    #===================================#

    print('#'*terminal_size())
    print('Total list making...')
    # 1) Path setting
    data_list = glob(os.path.join(args.data_path, '*/*.json'))
    data_list = sorted(data_list)[:-1] # 순종부록 제거

    total_string_list = list()
    total_ner_list = list()
    total_king_list = list()

    # 2) Total data making
    for data_path in tqdm(data_list):
        # 2-1) Load data
        with open(data_path, 'r') as f:
            data_ = json.load(f)
        data_string_list = list()
        data_ner_list = list()
        # 2-2) Extract string data by length
        for x in data_['string']:
            if len(x) <= args.max_len:
                data_string_list.append(x)
        # 2-3) Extract NER data by length
        for x in data_['NER']:
            if len(x) <= args.max_len:
                data_ner_list.append(x)
        # 2-4) Total data setting
        total_string_list.extend(data_string_list)
        total_ner_list.extend(data_ner_list)
        # 2-5) King list setting
        king_id = int(data_path.split('NER/')[1][:2]) - 1
        total_king_list.extend([king_id for _ in range(len(data_string_list))])

    #===================================#
    #============Data Split=============#
    #===================================#

    split_string_record, split_ner_record, split_king_record = train_test_split(
        total_string_list, total_ner_list, total_king_list, split_percent=args.data_split_per)

    print('Paired data num:')
    print(f"train: {len(split_string_record['train'])}")
    print(f"test: {len(split_string_record['test'])}")

    #====================================#
    #==========DWE Results Open==========#
    #====================================#

    with open(os.path.join(args.save_path, 'hj_word2id.pkl'), 'rb') as f:
        word2id = pickle.load(f)

    #===================================#
    #=======Hanja Pre-processing========#
    #===================================#

    # 1) Hanja sentence parsing setting
    print('Hanja sentence parsing...')
    start_time = time.time()
    hj_parsed_indices_train = list()
    hj_parsed_indices_test = list()

    # 2) Parsing sentence
    # 2-1) Train data parsing
    print('Train data start...')
    for index in tqdm(split_string_record['train']):
        parsed_index = list()
        parsed_index.append(args.bos_idx) # Start token add
        for ind in index:
            try:
                parsed_index.append(word2id[ind])
            except KeyError:
                parsed_index.append(word2id['<unk>'])
        parsed_index.append(args.eos_idx) # End token add
        hj_parsed_indices_train.append(parsed_index)

    # 2-2) Test data parsing
    print('Test data start...')
    for index in tqdm(split_string_record['test']):
        parsed_index = list()
        parsed_index.append(args.bos_idx) # Start token add
        for ind in index:
            try:
                parsed_index.append(word2id[ind])
            except KeyError:
                parsed_index.append(word2id['<unk>'])
        parsed_index.append(args.eos_idx) # End token add
        hj_parsed_indices_test.append(parsed_index)
    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #========NER Pre-processing=========#
    #===================================#

    # 1) NER parsing setting
    print('NER parsing...')
    start_time = time.time()
    NER_label2id = {
        'O': 0,
        'B_PS': 1,
        'I_PS': 2,
        'B_LC': 3,
        'I_LC': 4,
        'B_BK': 5,
        'I_BK': 6,
        'B_BOOK': 5,
        'I_BOOK': 6,
        'B_ERA': 7,
        'I_ERA': 8
    }
    NER_id2label = {v: k for k, v in NER_label2id.items()}
    ner_indices_train = list()
    ner_indices_test = list()

    # 2) Parsing sentence
    # 2-1) Train data parsing
    print('Train data start...')
    for index in tqdm(split_ner_record['train']):
        parsed_index = list()
        parsed_index.append(0) # Start token add
        for ind in index:
            parsed_index.append(NER_label2id[ind])
        parsed_index.append(0) # End token add
        ner_indices_train.append(parsed_index)

    # 2-2) Test data parsing
    print('Test data start...')
    for index in tqdm(split_ner_record['test']):
        parsed_index = list()
        parsed_index.append(0) # Start token add
        for ind in index:
            parsed_index.append(NER_label2id[ind])
        parsed_index.append(0) # End token add
        ner_indices_test.append(parsed_index)
    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence save setting...')
    start_time = time.time()

    with open(os.path.join(args.save_path, 'ner_processed.pkl'), 'wb') as f:
        pickle.dump({
            'hj_train_indices': hj_parsed_indices_train,
            'hj_test_indices': hj_parsed_indices_test,
            'ner_train_indices': ner_indices_train,
            'ner_test_indices': ner_indices_test,
            'king_train_indices': split_king_record['train'],
            'king_test_indices': split_king_record['test'],
            'word2id': word2id,
            'id2word': {v: k for k, v in word2id.items()}
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    parser.add_argument('--max_len', default=150, type=int)
    parser.add_argument('--save_path', default='./save/', 
                        type=str)
    parser.add_argument('--data_path', default='../joseon_word_embedding/Crawl/crawledResults_NER', 
                        type=str, help='Crawling data path')
    parser.add_argument('--data_split_per', default=0.2, type=float,
                        help='Train / Validation split ratio')
    parser.add_argument('--pad_idx', default=0, type=int, help='Padding index')
    parser.add_argument('--bos_idx', default=1, type=int, help='Start token index')
    parser.add_argument('--eos_idx', default=2, type=int, help='End token index')
    parser.add_argument('--unk_idx', default=3, type=int, help='Unknown token index')
    args = parser.parse_args()

    main(args)
    print('All Done!')