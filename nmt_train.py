# Import Module
import os
import math
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.utils.data import DataLoader

# Import Custom Module

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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