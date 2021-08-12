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

def gen_remap_dicts(joined_tabular_file, 
                    used_cnum, 
                    remap_c_pos_list, 
                    remap_dicts_file, 
                    sep,
                    header):
    logging.info('remapping dicts begin...')
    # comes from the config file
    remap_c_pos_list = list(map(int, remap_c_pos_list.split(',')))
    assert used_cnum == len(remap_c_pos_list)

    # sets and remap dicts for each column that needs to be remapped
    c_sets = [set() for _ in range(used_cnum)]
    remap_dicts = [{} for _ in range(used_cnum)]

    # get all the unique feature values
    with open(joined_tabular_file) as f:
        if header == True:
            f.readline()
        for line in tqdm(f):
            line_split = line[:-1].split(sep=sep)
            for i, c_pos in enumerate(remap_c_pos_list):
                c_sets[i].add(line_split[c_pos])
    
    # generate remap dicts
    remap_id = 1
    for i, c_set in enumerate(c_sets):
        for c in c_set:
            remap_dicts[i][c] = str(remap_id)
            remap_id += 1
    logging.info('total feature number is: {}'.format(remap_id))

    dump_pkl(remap_dicts_file, remap_dicts)
    
def remap(joined_tabular_file,
          remap_dicts_file,
          remap_c_pos_list,
          sampling_c_pos_list,
          remapped_tabular_file,
          sampling_collection_file,
          dataset_summary_file, 
          sep,
          header):
    remap_c_pos_list = list(map(int, remap_c_pos_list.split(',')))
    sampling_c_pos_list = list(map(int, sampling_c_pos_list.split(',')))
    
    with open(remap_dicts_file, 'rb') as f:
        remap_dicts = pkl.load(f)
        logging.info('remap_dicts have been loaded')
    
    remapped_tabular_lines = []
    sampling_collection_set = set()

    with open(joined_tabular_file) as f:
        if header == True:
            f.readline()
        for line in tqdm(f):
            line_split = line[:-1].split(sep=sep)
            for i, c_pos in enumerate(remap_c_pos_list):
                line_split[c_pos] = remap_dicts[i][line_split[c_pos]]
            remapped_tabular_lines.append(','.join(line_split) + '\n')
            
            sampling_c_list = list(np.array(line_split)[sampling_c_pos_list])
            sampling_collection_set.add(','.join(sampling_c_list))
    
    sampling_collection_list = list(sampling_collection_set)
    
    dump_lines(remapped_tabular_file, remapped_tabular_lines)
    dump_pkl(sampling_collection_file, sampling_collection_list)
    
    logging.info('remapped and sampling collection files dumped')
    logging.info('generating dataset summary file...')

    # generate summary file: columns info
    summary_dict = {}
    total_feat_num = 0
    for i, c_pos in enumerate(remap_c_pos_list):
        summary_dict['C{}'.format(c_pos)] = len(remap_dicts[i])
        total_feat_num += len(remap_dicts[i])
        logging.info('the number of column C{}\'s unique values(features) is {}'.format(c_pos, len(remap_dicts[i])))
    summary_dict['feat_num'] = total_feat_num + 1
    logging.info('total feature number is {}'.format(total_feat_num + 1))

    dump_pkl(dataset_summary_file, summary_dict)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('PLEASE INPUT [DATASET]')
        sys.exit(0)
    dataset = sys.argv[1]
    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    # call functions
    gen_remap_dicts(cnf.get(dataset, 'joined_tabular_file'), 
                    cnf.getint(dataset, 'used_cnum'), 
                    cnf.get(dataset, 'remap_c_pos_list'), 
                    cnf.get(dataset, 'remap_dicts_file'),
                    sep=',',
                    header=cnf.getboolean(dataset, 'header'))
    
    remap(cnf.get(dataset, 'joined_tabular_file'),
          cnf.get(dataset, 'remap_dicts_file'),
          cnf.get(dataset, 'remap_c_pos_list'),
          cnf.get(dataset, 'sampling_c_pos_list'),
          cnf.get(dataset, 'remapped_tabular_file'),
          cnf.get(dataset, 'sampling_collection_file'),
          cnf.get(dataset, 'summary_dict_file'),
          sep=',',
          header=cnf.getboolean(dataset, 'header'))

