import pickle as pkl
import logging
logging.basicConfig(level=logging.INFO)
import os
import sys
import configparser


def get_dataset_summary(dataset_summary_file):
    if not os.path.exists(dataset_summary_file):
        logging.error('data summary file {} does not exists'.format(dataset_summary_file))
    with open(dataset_summary_file, 'rb') as f:
        summary_dict = pkl.load(f)
    
    for key in summary_dict:
        content = str(key) + ':' + str(summary_dict[key])
        logging.info(content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('PLEASE INPUT [DATASET]')
        sys.exit(0)
    dataset = sys.argv[1]

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    get_dataset_summary(cnf.get(dataset, 'summary_dict_file'))

