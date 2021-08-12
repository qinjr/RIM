import logging
logging.basicConfig(level=logging.INFO)
import configparser
from tqdm import tqdm
import random
import sys
from utils import *
import numpy as np
import time

class DataloaderRIM(object):
    def __init__(self, 
                 batch_size,
                 remap_c_pos_list,
                 s_num,
                 target_file,
                 search_res_col_file,
                 search_res_label_file,
                 search_pool_file,
                 shuffle):
        
        logging.info('dataloader_rim initializing...')
        self.remap_c_pos_list = list(map(int, remap_c_pos_list.split(',')))
        self.batch_size = batch_size
        self.s_num = s_num
        self.shuffle = shuffle

        with open(target_file) as f:
            self.target = f.readlines()
            self.labels = [int(l[:-1].split(',')[-1]) for l in self.target]
        with open(search_res_col_file) as f:
            self.search_res_col = f.readlines()
        with open(search_res_label_file) as f:
            self.search_res_label = f.readlines()        
        with open(search_pool_file) as f:
            self.search_pool = f.readlines()
            self.search_pool.append(','.join(['0'] * len(self.search_pool[0].split(','))))
        
        logging.info('data loaded') 

        # shuffle
        if self.shuffle:
            logging.info('random shuffle...')
            self._shuffle_data()
            logging.info('random shuffle finished')

        self.dataset_size = len(self.target)
        
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration
        
        range_left = self.batch_size * self.step
        if self.step == self.total_step - 1:
            range_right = self.dataset_size
        else:
            range_right = self.batch_size * (self.step + 1)

        label_batch = self.labels[range_left:range_right]
        target_batch = self.target[range_left:range_right]
        target_batch = [select_pos_list(l.split(','), self.remap_c_pos_list) for l in target_batch]

        search_res_label_batch = self.search_res_label[range_left:range_right]
        search_res_label_batch = [list(map(int, l.split(','))) for l in search_res_label_batch]
        search_res_label_batch = [l + ['0'] * (self.s_num - len(l)) for l in search_res_label_batch]

        search_res_col_batch = self.search_res_col[range_left:range_right]
        search_res_col_batch = [list(map(int, l.split(','))) for l in search_res_col_batch]

        search_res_batch = []
        search_res_len_batch = []

        for search_res_linenos in search_res_col_batch:
            search_res_s = []
            for lineno in search_res_linenos:
                search_res_s.append(select_pos_list(self.search_pool[lineno].split(','), self.remap_c_pos_list))
                
            
            length = len(search_res_s)
            search_res_len_batch.append(length)

            if length < self.s_num:
                search_res_s += [search_res_s[0]] * (self.s_num - length)

            search_res_batch.append(search_res_s)        
        self.step += 1
        
        return search_res_batch, search_res_label_batch, search_res_len_batch, target_batch, label_batch

    def refresh(self):
        logging.info('refreshing...')
        self.step = 0
        if self.shuffle:
            self._shuffle_data()
        logging.info('refreshed')

    def _shuffle_data(self):
        zipped = list(zip(self.target, self.search_res_col, self.search_res_label, self.labels))
        random.shuffle(zipped)
        self.target, self.search_res_col, self.search_res_label, self.labels = zip(*zipped)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('PLEASE INPUT [DATASET]')
        sys.exit(0)
    dataset = sys.argv[1]
    batch_size = 100

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    dataloader_rim = DataloaderRIM(batch_size,
                                    cnf.get(dataset, 'remap_c_pos_list'),
                                    cnf.getint(dataset, 's_num'),
                                    cnf.get(dataset, 'target_train_file'),
                                    cnf.get(dataset, 'search_res_col_train_file'),
                                    cnf.get(dataset, 'search_res_label_train_file'),
                                    cnf.get(dataset, 'search_pool_file'),
                                    False)
    t = time.time()
    for batch_data in dataloader_rim:
        search_res_batch, search_res_label_batch, search_res_len_batch, target_batch, label_batch = batch_data
        print(np.array(search_res_batch).shape)
        print(np.array(search_res_label_batch).shape)
        print(np.array(search_res_len_batch).shape)
        print(search_res_len_batch)
        print(np.array(target_batch).shape)
        print(np.array(label_batch).shape)
        print(time.time() - t)
        t = time.time()
        break   
    print(time.time() - t)