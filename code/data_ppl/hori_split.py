import pickle as pkl
import random
import numpy as np
import datetime
import time
import sys
import logging
logging.basicConfig(level=logging.INFO)
import configparser
from tqdm import tqdm
from data_ppl_utils import *

random.seed(1111)

def timestr2timestamp(tstr, time_format):
    return int(time.mktime(datetime.datetime.strptime(tstr, time_format).timetuple()))

def split_tabular(tabular_file,
                  sampling_collection_file,
                  is_timestamp_flag,
                  neg_sample_flag,
                  label_flag,
                  timestamp_pos,
                  sync_c_pos,
                  split_time_points,
                  sampling_c_pos_list,
                  time_format,
                  search_pool_file,
                  target_train_file,
                  target_test_file,
                  sync_seq_dict_file,
                  dataset_summary_file,
                  sfh_rate,
                  neg_sample_num,
                  shuffle_target):

    sampling_c_pos_list = list(map(int, sampling_c_pos_list.split(',')))
    with open(dataset_summary_file, 'rb') as f:
        summary = pkl.load(f)

    # phase 1: split the tabular file horizontally using the given time points
    logging.info('phase 1: split the tabular file horizontally using split time points...')
    
    # get split time points
    split_tps = split_time_points.split(',')
    if is_timestamp_flag:
        split_tps = list(map(int, split_tps))
    else:
        split_tps = list(map(lambda x: timestr2timestamp(x, time_format), split_tps))

    logging.info('time points for cutting search pool, target train and target test is {} and {}'.format(split_tps[0], split_tps[1]))

    target_train_lines = []
    target_test_lines = []
    search_pool_lines = []
    
    ts_search_pool = []
    target_sync_ids_set = set()
    
    with open(tabular_file) as f:
        for line in tqdm(f):
            line_split = line[:-1].split(',')
            ts = line_split[timestamp_pos]
            if is_timestamp_flag:
                ts = int(ts)
            else:
                ts = timestr2timestamp(ts, time_format)

            if not label_flag:
                newline = ','.join(line_split + ['1']) + '\n'
            else:
                newline = line

            if ts < split_tps[0]:
                search_pool_lines.append(newline)
                ts_search_pool.append(ts)
            elif ts >= split_tps[1]:
                target_sync_ids_set.add(line_split[sync_c_pos])
                target_test_lines.append(newline)
            else:
                target_sync_ids_set.add(line_split[sync_c_pos])
                target_train_lines.append(newline)
    
    target_sync_ids = list(target_sync_ids_set)
    logging.info('number of target sync ids is {}'.format(len(target_sync_ids)))
    logging.info('original search pool size: {}'.format(len(search_pool_lines)))
    logging.info('original target train size: {}'.format(len(target_train_lines)))
    logging.info('original target test size: {}'.format(len(target_test_lines)))

    summary['ori_search_pool_size'] = len(search_pool_lines)
    summary['ori_target_train_size'] = len(target_train_lines)
    summary['ori_target_test_size'] = len(target_test_lines)

    logging.info('phase 1 completed...')

    # phase 2: generate sync seq file for target files and sort them using timestamp
    logging.info('phase 2: generate sync seq dict...')
    sync_seq_dict = {}
    
    logging.info('sync_seq_dict initializing...')
    for sync_id in tqdm(target_sync_ids):
        sync_seq_dict[sync_id] = []
    
    logging.info('sync_seq_dict inserting...')
    for i, line in tqdm(enumerate(search_pool_lines)):
        line_split = line[:-1].split(',')
        sync_id = line_split[sync_c_pos]
        if sync_id in sync_seq_dict:
            seq_part = ','.join(np.array(line_split)[sampling_c_pos_list].tolist())
            sync_seq_dict[sync_id].append((seq_part, ts_search_pool[i]))
    
    # sorting the sync_seq_dict for each sync id term
    logging.info('sync_seq_dict sorting...')
    for sync_id in tqdm(sync_seq_dict):
        sync_seq_dict[sync_id].sort(key = lambda x:x[1])
    
    dump_pkl(sync_seq_dict_file, sync_seq_dict)

    # (optinal) phase 3: negative sampling for target files and search pool file
    search_pool_lines_neg = []
    target_train_lines_dump = target_train_lines
    target_test_lines_dump = target_test_lines

    if neg_sample_flag:
        logging.info('phase 3: negative sampling for target and search pool files...')
        with open(sampling_collection_file, 'rb') as f:
            sampling_list = pkl.load(f)
        
        logging.info('generate neg samples for search pool...')
        for line in tqdm(search_pool_lines):
            line_split = line[:-1].split(',')
            sampling = random.choice(sampling_list).split(',')

            for i, pos in enumerate(sampling_c_pos_list):
                line_split[pos] = sampling[i]
            line_split[-1] = '0'
            search_pool_lines_neg.append(','.join(line_split) + '\n')
        
        logging.info('generate neg samples for target train...')
        cnt = 0
        for line in tqdm(target_train_lines):
            line_split = line[:-1].split(',')
            # sample from history or random sample
            for _ in range(neg_sample_num):
                r = random.randint(1, int(1 / sfh_rate))
                if r == 1:
                    sync_seq = sync_seq_dict[line_split[sync_c_pos]]
                    if sync_seq != []:
                        sampling = random.choice(sync_seq)[0].split(',')
                    else:
                        cnt += 1
                        sampling = random.choice(sampling_list).split(',')
                else:
                    sampling = random.choice(sampling_list).split(',')

                for i, pos in enumerate(sampling_c_pos_list):
                    line_split[pos] = sampling[i]
                line_split[-1] = '0'
                target_train_lines_dump.append(','.join(line_split) + '\n')
        logging.info('num of empty sync seq for target train {}'.format(cnt))

        logging.info('generate neg samples for target test...')
        cnt = 0
        for line in tqdm(target_test_lines):
            line_split = line[:-1].split(',')
            # sample from history or random sample
            for _ in range(neg_sample_num):
                r = random.randint(1, int(1 / sfh_rate))
                if r == 1:
                    sync_seq = sync_seq_dict[line_split[sync_c_pos]]
                    if sync_seq != []:
                        sampling = random.choice(sync_seq)[0].split(',')
                    else:
                        cnt += 1
                        sampling = random.choice(sampling_list).split(',')
                else:
                    sampling = random.choice(sampling_list).split(',')

                for i, pos in enumerate(sampling_c_pos_list):
                    line_split[pos] = sampling[i]
                line_split[-1] = '0'
                target_test_lines_dump.append(','.join(line_split) + '\n')
        logging.info('num of empty sync seq for target test {}'.format(cnt))

    # phase 4: random shuffle the lines
    search_pool_lines += search_pool_lines_neg

    summary['search_pool_size'] = len(search_pool_lines)
    summary['target_train_size'] = len(target_train_lines_dump)
    summary['target_test_size'] = len(target_test_lines_dump)
    logging.info('search pool size: {}'.format(len(search_pool_lines)))
    logging.info('target train size: {}'.format(len(target_train_lines_dump)))
    logging.info('target test size: {}'.format(len(target_test_lines_dump)))

    logging.info('phase 4: shuffle search pool (and possibly target files)...')
    random.shuffle(search_pool_lines)
    if shuffle_target:
        random.shuffle(target_train_lines_dump)
        random.shuffle(target_test_lines_dump)

    dump_lines(search_pool_file, search_pool_lines)
    dump_lines(target_train_file, target_train_lines_dump)
    dump_lines(target_test_file, target_test_lines_dump)

    logging.info('target and search pool generated')
    
    dump_pkl(dataset_summary_file, summary)
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('PLEASE INPUT [DATASET]')
        sys.exit(0)
    dataset = sys.argv[1]

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    # call functions
    split_tabular(cnf.get(dataset, 'remapped_tabular_file'),
                  cnf.get(dataset, 'sampling_collection_file'),
                  cnf.getboolean(dataset, 'is_timestamp_flag'),
                  cnf.getboolean(dataset, 'neg_sample_flag'),
                  cnf.getboolean(dataset, 'label_flag'),
                  cnf.getint(dataset, 'timestamp_pos'),
                  cnf.getint(dataset, 'sync_c_pos'),
                  cnf.get(dataset, 'split_time_points'),
                  cnf.get(dataset, 'sampling_c_pos_list'),
                  cnf.get(dataset, 'time_format'),
                  cnf.get(dataset, 'search_pool_file'),
                  cnf.get(dataset, 'target_train_file'),
                  cnf.get(dataset, 'target_test_file'),
                  cnf.get(dataset, 'sync_seq_dict_file'),
                  cnf.get(dataset, 'summary_dict_file'),
                  cnf.getfloat(dataset, 'sfh_rate'),
                  cnf.getint(dataset, 'neg_sample_num'), 
                  cnf.getboolean(dataset, 'shuffle_target'))

    sampling(cnf.get(dataset, 'target_train_file'), 
             cnf.get(dataset, 'target_train_sample_file'), 
             rate=cnf.getfloat(dataset, 'target_sample_rate'))
    sampling(cnf.get(dataset, 'target_test_file'), 
             cnf.get(dataset, 'target_test_sample_file'), 
             rate=cnf.getfloat(dataset, 'target_sample_rate'))
