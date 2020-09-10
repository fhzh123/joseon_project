import fcntl
import struct
import random
import termios
import numpy as np

def train_test_split(record_list1, record_list2, record_list3, split_percent=0.2):
    assert len(record_list1) == len(record_list2) # Check paired data

    # Paired data split
    paired_data_len = len(record_list1)
    test_num = int(paired_data_len * split_percent)
    
    test_index = np.random.choice(paired_data_len, test_num, replace=False) 
    train_index = list(set(range(paired_data_len)) - set(test_index))
    random.shuffle(train_index)

    train_record_list1 = [record_list1[i] for i in train_index]
    train_record_list2 = [record_list2[i] for i in train_index]
    train_record_list3 = [record_list3[i] for i in train_index]
    test_record_list1 = [record_list1[i] for i in test_index]
    test_record_list2 = [record_list2[i] for i in test_index]
    test_record_list3 = [record_list3[i] for i in test_index]

    split_record1 = {'train': train_record_list1, 'test': test_record_list1}
    split_record2 = {'train': train_record_list2, 'test': test_record_list2}
    split_record3 = {'train': train_record_list3, 'test': test_record_list3}

    return split_record1, split_record2, split_record3

def terminal_size():
    th, tw, hp, wp = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return tw