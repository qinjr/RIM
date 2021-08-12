from es_reader import *
from data_ppl_utils import *
from tqdm import tqdm
import sys
import configparser
import time
def pre_search_rim(query_generator, 
                    es_reader, 
                    search_res_col_file, 
                    search_res_label_file):
    search_res_col_lines = []
    search_res_label_lines = []
    
    for batch in tqdm(query_generator):
        q_batch, _ = batch
        res_lineno_batch, label_batch = es_reader.query_rim1(q_batch)
        

        search_res_col_lines += [(','.join(res) + '\n') for res in res_lineno_batch]
        search_res_label_lines += [(','.join(label) + '\n') for label in label_batch]

    dump_lines(search_res_col_file, search_res_col_lines)
    dump_lines(search_res_label_file, search_res_label_lines)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('PLEASE INPUT [DATASET] [BATCH_SIZE] [RETRIEVE_SIZE] [MODE]')
        sys.exit(0)
    dataset = sys.argv[1]
    batch_size = int(sys.argv[2])
    size = int(sys.argv[3])
    mode = sys.argv[4]

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    # query generator
    query_generator_train = queryGen(cnf.get(dataset, 'target_train_file'),
                                     batch_size,
                                     cnf.getint(dataset, 'sync_c_pos'),
                                     cnf.get(dataset, 'query_c_pos'))
    query_generator_test = queryGen(cnf.get(dataset, 'target_test_file'),
                                     batch_size,
                                     cnf.getint(dataset, 'sync_c_pos'),
                                     cnf.get(dataset, 'query_c_pos'))
    es_reader = ESReader(dataset, size)

    if mode == 'rim':
        print('target train pre searching...')
        pre_search_rim(query_generator_train, 
                        es_reader, 
                        cnf.get(dataset, 'search_res_col_train_file'), 
                        cnf.get(dataset, 'search_res_label_train_file'))
        
        print('target test pre searching...')
        pre_search_rim(query_generator_test, 
                        es_reader, 
                        cnf.get(dataset, 'search_res_col_test_file'), 
                        cnf.get(dataset, 'search_res_label_test_file'))