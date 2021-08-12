import sys
import configparser
from data_ppl_utils import *
from es_writer import *
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('PLEASE INPUT [DATASET]')
        sys.exit(0)
    dataset = sys.argv[1]

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    # ESWriter
    eswriter = ESWriter(cnf.get(dataset, 'search_pool_file'),
                        dataset,
                        cnf.getint(dataset, 'sync_c_pos'))
    eswriter.write()
