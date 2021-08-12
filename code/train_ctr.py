import sys
import os
import numpy as np
from dataloader import DataloaderRIM
from sklearn.metrics import *
import random
import time
import pickle as pkl
import math
from rim import *
import logging
logging.basicConfig(level=logging.INFO)
import configparser

random.seed(1111)

SAVE_PATH_PREFIX = '../models/'
LOG_PATH_PREFIX = '../logs/'

DECAY_FACTOR = 1

def get_elapsed(start_time):
    return time.time() - start_time

def eval(model, sess, dataloader, l2_norm):
    preds = []
    labels = []
    losses = []

    t = time.time()
    for batch_data in dataloader:
        pred, label, loss = model.eval(sess, batch_data, l2_norm)
        preds += pred
        labels += label
        losses.append(loss)

    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)
    
    logging.info("Time of evaluating on the test dataset: %.4fs" % (time.time() - t))
    return loss, logloss, auc

def train(dataset,
          model_type, 
          model_name,
          train_dataloader,
          test_dataloader,
          writer,
          max_epoch,
          eval_freq,
          log_freq,
          feature_size, 
          eb_dim,
          s_num, 
          c_num, 
          label_num,
          lr,
          l2_norm):
    tf.reset_default_graph()

    if model_type == 'RIM':
        model = RIM(feature_size, eb_dim, s_num, c_num, label_num)
    else:
        logging.info('WRONG MODEL TYPE')
        exit(1)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        start_time = time.time()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # before training process, get initial test evaluation
        step = 0 # training step
        log_step = 0 # training log step
        eval_step = 0 # eval step

        test_losses = []
        test_log_losses = []
        test_aucs = []

        test_loss, test_log_loss, test_auc = eval(model, sess, test_dataloader, l2_norm)

        test_losses.append(test_loss)
        test_log_losses.append(test_log_loss)
        test_aucs.append(test_auc)

        summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                    tf.Summary.Value(tag='test_log_loss', simple_value=test_log_loss),
                                    tf.Summary.Value(tag='test_auc', simple_value=test_auc)])
        writer.add_summary(summary, global_step=eval_step)
        eval_step += 1

        logging.info("STEP %d  LOSS TEST: %.4f  LOG_LOSS TEST: %.4f AUC TEST: %.4f  ELASPED: %.2fs" % (step, test_loss, test_log_loss, test_auc, get_elapsed(start_time)))

        early_stop = False

        # begin training process
        for epoch in range(max_epoch):
            # lr decay
            if epoch != 0:
                lr = lr * DECAY_FACTOR
                logging.info('DECAYED LR IS {}'.format(lr))
            if early_stop:
                break
            
            for batch_data in train_dataloader:
                if early_stop:
                    break
                train_loss, train_log_loss, train_l2_loss = model.train(sess, batch_data, lr, l2_norm)
                step += 1

                if step % log_freq == 0:
                    summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                tf.Summary.Value(tag='train_log_loss', simple_value=train_log_loss),
                                                tf.Summary.Value(tag='train_l2_loss', simple_value=train_l2_loss)])
                    writer.add_summary(summary, global_step=log_step)
                    log_step += 1
                    logging.info("STEP %d  LOSS TRAIN: %.4f  LOG_LOSS TRAIN: %.4f  L2_LOSS TRAIN: %.4f  Elasped: %.2fs" % (step, train_loss, train_log_loss, train_l2_loss, get_elapsed(start_time)))

                if step % eval_freq == 0:
                    test_dataloader.refresh()

                    test_loss, test_log_loss, test_auc = eval(model, sess, test_dataloader, l2_norm)
                    test_losses.append(test_loss)
                    test_log_losses.append(test_log_loss)
                    test_aucs.append(test_auc)

                    logging.info("STEP %d  LOSS TEST: %.4f  LOG_LOSS TEST: %.4f AUC TEST: %.4f  ELASPED: %.2fs" % (step, test_loss, test_log_loss, test_auc, get_elapsed(start_time)))
                    
                    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                    tf.Summary.Value(tag='test_log_loss', simple_value=test_log_loss),
                                    tf.Summary.Value(tag='test_auc', simple_value=test_auc)])
                    writer.add_summary(summary, global_step=eval_step)
                    eval_step += 1

                    if test_aucs[-1] > max(test_aucs[:-1]):
                        model_dir = os.path.join(SAVE_PATH_PREFIX, dataset, str(s_num), model_name)

                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        model_path = os.path.join(model_dir, 'ckpt')
                        model.save(sess, model_path)
                        
                    if len(test_losses) > 2 and epoch > 0:
                        if (test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3]):
                            early_stop = True
                        if (test_losses[-2] - test_losses[-1]) <= 0.0001 and (test_losses[-3] - test_losses[-2]) <= 0.0001:
                            early_stop = True
            
            # refresh dataloader if not early stop
            if not early_stop:
                train_dataloader.refresh()
        
        # write results
        log_dir = os.path.join(LOG_PATH_PREFIX, dataset, str(s_num))
        log_path = os.path.join(log_dir, '{}.txt'.format(model_type))
        with open(log_path, 'a') as f:
            results = [model_name, str(test_log_losses[np.argmax(test_aucs)]), str(max(test_aucs))]
            result_line = '\t'.join(results) + '\n'
            logging.info('Result: %s' % (result_line))
            f.write(result_line)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        logging.info("PLEASE INPUT [MODEL TYPE] [DATASET] [GPU]")
        sys.exit(0)
    model_type = sys.argv[1]
    dataset = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

    if not os.path.exists(SAVE_PATH_PREFIX):
        os.mkdir(SAVE_PATH_PREFIX)
    if not os.path.exists(LOG_PATH_PREFIX):
        os.mkdir(LOG_PATH_PREFIX)
    

    # read config file
    cnf_dataset = configparser.ConfigParser()
    cnf_dataset.read('../configs/config.ini')

    cnf_train = configparser.ConfigParser()
    cnf_train.read('../configs/train_params.ini')
    
    # get training params
    batch_sizes = list(map(int, cnf_train.get(dataset, 'batch_sizes').split(',')))
    lrs = list(map(float, cnf_train.get(dataset, 'lrs').split(',')))
    l2_norms = list(map(float, cnf_train.get(dataset, 'l2_norms').split(',')))
    eb_dim = cnf_train.getint(dataset, 'eb_dim')
    eval_batch_size = cnf_train.getint(dataset, 'eval_batch_size')

    log_freq_protion = cnf_train.getint(dataset, 'log_freq_protion')
    eval_freq_protion = cnf_train.getint(dataset, 'eval_freq_protion')
    max_epoch = cnf_train.getint(dataset, 'max_epoch')

    # get dataset stats
    feature_size = cnf_dataset.getint(dataset, 'feat_size')
    s_num = cnf_dataset.getint(dataset, 's_num')
    c_num = cnf_dataset.getint(dataset, 'c_num') 
    label_num = cnf_dataset.getint(dataset, 'label_num')
    dataset_size = cnf_dataset.getint(dataset, 'dataset_size')
    shuffle = cnf_dataset.getboolean(dataset, 'shuffle')

    # test dataloader
    if model_type == 'RIM_Random':
        test_dataloader = DataloaderRIM(eval_batch_size,
                                     cnf_dataset.get(dataset, 'remap_c_pos_list'),
                                     s_num,
                                     cnf_dataset.get(dataset, 'target_test_file'),
                                     cnf_dataset.get(dataset, 'search_res_col_test_random_file'),
                                     cnf_dataset.get(dataset, 'search_res_label_test_random_file'),
                                     cnf_dataset.get(dataset, 'search_pool_file'),
                                     False)
    else:
        test_dataloader = DataloaderRIM(eval_batch_size,
                                     cnf_dataset.get(dataset, 'remap_c_pos_list'),
                                     s_num,
                                     cnf_dataset.get(dataset, 'target_test_file'),
                                     cnf_dataset.get(dataset, 'search_res_col_test_file'),
                                     cnf_dataset.get(dataset, 'search_res_label_test_file'),
                                     cnf_dataset.get(dataset, 'search_pool_file'),
                                     False)
    
    logging.info('got test dataloader')

    for batch_size in batch_sizes:
        for lr in lrs:
            for l2_norm in l2_norms:
                model_name = '_'.join([model_type, str(batch_size), str(lr), str(l2_norm)])
                
                # log writters
                writer_dir = os.path.join(LOG_PATH_PREFIX, dataset, str(s_num), model_name)
                if not os.path.exists(writer_dir):
                    os.makedirs(writer_dir)

                writer = tf.summary.FileWriter(logdir=writer_dir)                
                train_dataloader = DataloaderRIM(batch_size,
                                                cnf_dataset.get(dataset, 'remap_c_pos_list'),
                                                s_num,
                                                cnf_dataset.get(dataset, 'target_train_file'),
                                                cnf_dataset.get(dataset, 'search_res_col_train_file'),
                                                cnf_dataset.get(dataset, 'search_res_label_train_file'),
                                                cnf_dataset.get(dataset, 'search_pool_file'),
                                                shuffle)


                logging.info('got train dataloader')
                # other training params
                log_freq = (dataset_size // batch_size) // log_freq_protion
                eval_freq = (dataset_size // batch_size) // eval_freq_protion

                logging.info('training begin...')
                train(dataset,
                      model_type, 
                      model_name,
                      train_dataloader,
                      test_dataloader,
                      writer,
                      max_epoch,
                      eval_freq,
                      log_freq,
                      feature_size, 
                      eb_dim,
                      s_num, 
                      c_num, 
                      label_num,
                      lr,
                      l2_norm)
                
                test_dataloader.refresh()