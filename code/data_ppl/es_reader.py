from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, MultiSearch
import elasticsearch.helpers
import time
import numpy as np
import pickle as pkl
from data_ppl_utils import *
import sys
import configparser

class ESReader(object):
    def __init__(self, 
                 index_name,
                 size,
                 host_url = 'localhost:9200'):
        
        self.es = Elasticsearch(host_url)
        self.index_name = index_name
        self.size = size

    # For UBR
    def query_ubr(self, queries, sync_ids):
        ms = MultiSearch(using=self.es, index=self.index_name)
        for i, q in enumerate(queries):
            s = Search().filter("terms", sync_id=[sync_ids[i]]).filter("terms", label=['1']).query("match", line=q)[:self.size]
            ms = ms.add(s)
        responses = ms.execute()

        res_lineno_batch = []
        # res_line_batch = []
        for response in responses:
            res_lineno = []
            res_line = []
            for hit in response:
                res_lineno.append(str(hit.line_no))
                # res_line.append(list(map(int, hit.line.split(','))))
            if res_lineno == []:
                res_lineno.append('-1')
            res_lineno_batch.append(res_lineno)
            # res_line_batch.append(res_line)
        return res_lineno_batch#, res_line_batch

    # For RIM: not groupping by label, query_rim1 is for sequential data setting
    def query_rim1(self, queries):
        ms = MultiSearch(using=self.es, index=self.index_name)
        for q in queries:
            s = Search().query("match", line=q)[:self.size]
            ms = ms.add(s)
        responses = ms.execute()

        res_lineno_batch = []
        # res_line_batch = []
        label_batch = []
        for response in responses:
            print("len of res:{}".format(len(response)))
            res_lineno = []
            # res_line = []
            labels = []
            for hit in response:
                res_lineno.append(str(hit.line_no))
                # res_line.append(list(map(int, hit.line.split(','))))
                labels.append(hit.label)
            if res_lineno == []:
                res_lineno.append('-1')
                labels.append('0')
            res_lineno_batch.append(res_lineno)
            # res_line_batch.append(res_line)
            label_batch.append(labels)
        return res_lineno_batch, label_batch#, res_line_batch

    # For RIM: avazu and criteo
    def query_rim_ac(self, queries, sync_ids):
        ms = MultiSearch(using=self.es, index=self.index_name)
        for i, q in enumerate(queries):
            s = Search().query("match", sync_id=sync_ids[i])[:self.size]
            ms = ms.add(s)
        responses = ms.execute()

        res_lineno_batch = []
        # res_line_batch = []
        label_batch = []
        for response in responses:
            res_lineno = []
            # res_line = []
            labels = []
            for hit in response:
                res_lineno.append(str(hit.line_no))
                # res_line.append(list(map(int, hit.line.split(','))))
                labels.append(hit.label)
            if res_lineno == []:
                res_lineno.append('-1')
                labels.append('0')
            res_lineno_batch.append(res_lineno)
            # res_line_batch.append(res_line)
            label_batch.append(labels)
        return res_lineno_batch, label_batch#, res_line_batch

class queryGen(object):
    def __init__(self,
                 target_file,
                 batch_size,
                 sync_c_pos,
                 query_c_pos):
        
        self.batch_size = batch_size
        self.sync_c_pos = sync_c_pos
        self.query_c_pos = list(map(int, query_c_pos.split(',')))

        with open(target_file) as f:
            self.target_lines = f.readlines()
        self.dataset_size = len(self.target_lines)
        
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
        print('data loaded')
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        q_batch = []
        sync_id_batch = []
        if self.step != self.total_step - 1:
            lines_batch = self.target_lines[self.step * self.batch_size: (self.step + 1) * self.batch_size]
        else:
            lines_batch = self.target_lines[self.step * self.batch_size:]
        
        q_batch = [select_pos_str(l, self.query_c_pos) for l in lines_batch]
        sync_id_batch = [l.split(',')[self.sync_c_pos] for l in lines_batch]

        self.step += 1

        return q_batch, sync_id_batch


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('PLEASE INPUT [DATASET] [BATCH_SIZE] [RETRIEVE_SIZE]')
        sys.exit(0)
    dataset = sys.argv[1]
    batch_size = int(sys.argv[2])
    size = int(sys.argv[3])

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    # query generator
    query_generator = queryGen(cnf.get(dataset, 'target_train_file'),
                               batch_size,
                               cnf.getint(dataset, 'sync_c_pos'),
                               cnf.get(dataset, 'query_c_pos'))
    es_reader = ESReader(dataset, size)

    t = time.time()
    for batch in query_generator:
        q_batch, sync_id_batch = batch

        res_lineno_batch, res_line_batch = es_reader.query_rim1(q_batch)
        # plist = zip(q_batch, res_line_batch, label_batch)
        # print(res_lineno_batch)
        # print(res_line_batch)

        print('time: %.4f seconds' % (time.time() - t))
        t = time.time()

