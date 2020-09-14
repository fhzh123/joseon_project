import os
import re
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from gensim.corpora import Dictionary
from collections import deque, Counter

import torch

from dynamic_bernoulli_embeddings.training import train_model

def main(args):
    # Data List Setting
    data_path = glob(os.path.join(args.data_path, '*.json'))
    data_path = sorted(data_path)[:-1] # 순종부록 제거

    # Preprocessing
    total_counter = Counter()
    king_list = list()
    king_index_list = list()
    comment_list = list()

    # start_time = time.time()
    for ix, path in enumerate(tqdm(data_path)):
        with open(path, 'r') as f:
            record_list = json.load(f)
            king_list.append(path.split(' ')[-1][:2])
            king_index_list.append(ix)
            total_record = list()

            for rc in record_list:
                total_record.append(rc['hanja'])
            total_record = ' '.join(total_record)
            new_word = re.sub(pattern='[^\w\s]', repl='', string=total_record)
            new_word = re.sub(pattern='([ㄱ-ㅎㅏ-ㅣ]+)', repl='', string=new_word)
            new_word = re.sub(pattern='[\u3131-\u3163\uac00-\ud7a3]+', repl='', string=new_word)
            new_word = re.sub(pattern='[a-zA-Z0-9]+', repl='', string=new_word)
            comment_list.append(new_word)
            total_counter.update(new_word)

    vocab = list(total_counter.keys())
    vocab.insert(0, '<unk>')
    vocab.insert(0, '</s>')
    vocab.insert(0, '<s>')
    vocab.insert(0, '<pad>')
    word2id = {w: i for i, w in enumerate(vocab)}

    # Dataset Setting
    dataset = pd.DataFrame({
        'session': king_list,
        'text': comment_list,
        'time': king_index_list
    })
    dataset['bow'] = dataset['text'].apply(lambda x: [i for i in x])

    # Generate dictionary.
    dictionary = Dictionary(dataset.bow)
    dictionary.filter_extremes(no_below=15, no_above=1.)
    dictionary.compactify()

    # Model Training
    model, loss_history = train_model(dataset, word2id, validation=None, m=args.minibatch_iteration,
                                      num_epochs=args.num_epochs, notebook=False)

    # Model Saving
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    torch.save(model, os.path.join(args.save_path, 'dwe_hj_model.pt'))
    with open(os.path.join(args.save_path, 'hj_word2id.pkl'), 'wb') as f:
        pickle.dump(word2id, f)
    with open(os.path.join(args.save_path, 'emb_mat.pkl'), 'wb') as f:
        pickle.dump(model.get_embeddings(), f)
    loss_history.to_csv(os.path.join(args.save_path, 'loss_history.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DWE argparser')
    # Path Setting
    parser.add_argument('--data_path', type=str, default='../joseon_word_embedding/data/', 
                        help='Data path setting')
    parser.add_argument('--save_path', type=str, default='./save',
                        help='Save path setting')
    # Training Setting
    parser.add_argument('--num_epochs', type=int, default=5, help='The number of epoch')
    parser.add_argument('--minibatch_iteration', type=int, default=300, help='Mini-batch size')
    args = parser.parse_args()

    total_start_time = time.time()
    main(args)
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')