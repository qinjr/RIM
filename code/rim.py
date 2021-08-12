import tensorflow as tf
import numpy as np


class Base(object):
    def __init__(self, feature_size, eb_dim, s_num, c_num, label_num, pred_mode):
        # input placeholders
        with tf.name_scope('inputs'):
            self.search_res_ph = tf.placeholder(tf.int32, [None, s_num, c_num], name='search_res_ph')
            self.search_res_label_ph = tf.placeholder(tf.int32, [None, s_num], name='search_res_label_ph')
            self.search_res_len_ph = tf.placeholder(tf.int32, [None,], name='search_res_len_ph')
            self.target_ph = tf.placeholder(tf.int32, [None, c_num], name='target_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.l2_norm = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.pred_mode = pred_mode
            self.label_num = label_num

        # embedding
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
            self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

            self.label_emb_mtx = tf.get_variable('label_emb_mtx', [label_num, eb_dim], initializer=tf.truncated_normal_initializer)

        self.search_res = tf.nn.embedding_lookup(self.emb_mtx, self.search_res_ph) #[batch_size, s_num, c_num, eb_dim]
        self.search_res = tf.reshape(self.search_res, [-1, s_num, c_num * eb_dim])

        self.target = tf.nn.embedding_lookup(self.emb_mtx, self.target_ph)
        self.target = tf.reshape(self.target, [-1, c_num * eb_dim])
        
        self.search_res_label = tf.nn.embedding_lookup(self.label_emb_mtx, self.search_res_label_ph) #[batch_size, s_num, eb_dim]
        
    def _build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')

        if self.pred_mode == 'class':
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            # output
            self.y_pred = tf.reshape(score[:,0], [-1,])
        elif self.pred_mode == 'reg':
            # fc3 = tf.layers.dense(dp2, 1, activation=tf.nn.sigmoid, name='fc3')
            # fc3 = self.label_num * fc3
            fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
            self.y_pred = tf.reshape(fc3, [-1,])

    def _build_fc_net_res(self, inp, res_inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        concat_inp = tf.concat([dp2, res_inp], axis=-1)

        if self.pred_mode == 'class':
            fc3 = tf.layers.dense(concat_inp, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            # output
            self.y_pred = tf.reshape(score[:,0], [-1,])
        elif self.pred_mode == 'reg':
            # fc3 = tf.layers.dense(dp2, 1, activation=tf.nn.sigmoid, name='fc3')
            # fc3 = self.label_num * fc3
            fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
            self.y_pred = tf.reshape(fc3, [-1,])
    
    def _build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.l2_norm * tf.nn.l2_loss(v)
        self.l2_loss = self.loss - self.log_loss
    
    def _build_mseloss(self):
        # loss
        self.mse_loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        self.loss = self.mse_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.l2_norm * tf.nn.l2_loss(v)
        self.l2_loss = self.loss - self.mse_loss
        
    
    def _build_optimizer(self):    
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='optimizer')
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, l2_norm):
        if self.pred_mode == 'class':
            loss, log_loss, l2_loss, _ = sess.run([self.loss, self.log_loss, self.l2_loss, self.train_step], feed_dict = {
                    self.search_res_ph : batch_data[0],
                    self.search_res_label_ph : batch_data[1],
                    self.search_res_len_ph : batch_data[2],
                    self.target_ph : batch_data[3],
                    self.label_ph : batch_data[4],
                    self.lr : lr,
                    self.l2_norm : l2_norm,
                    self.keep_prob : 0.8
                })
            return loss, log_loss, l2_loss
        elif self.pred_mode == 'reg':
            loss, mse_loss, l2_loss, _ = sess.run([self.loss, self.mse_loss, self.l2_loss, self.train_step], feed_dict = {
                    self.search_res_ph : batch_data[0],
                    self.search_res_label_ph : batch_data[1],
                    self.search_res_len_ph : batch_data[2],
                    self.target_ph : batch_data[3],
                    self.label_ph : batch_data[4],
                    self.lr : lr,
                    self.l2_norm : l2_norm,
                    self.keep_prob : 0.8
                })
            return loss, mse_loss, l2_loss

    def eval(self, sess, batch_data, l2_norm):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.search_res_ph : batch_data[0],
                self.search_res_label_ph : batch_data[1],
                self.search_res_len_ph : batch_data[2],
                self.target_ph : batch_data[3],
                self.label_ph : batch_data[4],
                self.l2_norm : l2_norm,
                self.keep_prob : 1.
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))
    
    def _attention(self, key, value, query, mask):
        # key/value: [b, s_num, c_num * eb_dim], query: [b, c_num * eb_dim]
        _, s_num, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, s_num, 1])
        kq_inter = queries * key
        atten = tf.reduce_sum(kq_inter, axis=2)
        
        mask = tf.equal(mask, tf.ones_like(mask)) 
        paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
        atten = tf.nn.softmax(tf.where(mask, atten, paddings)) 
        atten = tf.expand_dims(atten, 2)

        res = tf.reduce_sum(atten * value, axis=1) #[b, c_num * eb_dim]
        return atten, res
    
    def _unroll_pairwise(self, xv, c_num):
        rows = []
        cols = []
        for i in range(c_num - 1):
            for j in range(i + 1, c_num):
                rows.append(i)
                cols.append(j)
        with tf.variable_scope('unroll_pairwise'):
            # [b, pair, dim]
            xv_p = tf.transpose(
                # [pair, b, dim]
                tf.gather(
                    # [c_num, b, dim]
                    tf.transpose(
                        xv, [1, 0, 2]),
                    rows),
                [1, 0, 2])
            # [b, pair, dim]
            xv_q = tf.transpose(
                tf.gather(
                    tf.transpose(
                        xv, [1, 0, 2]),
                    cols),
                [1, 0, 2])
        return xv_p, xv_q
    
    def _unroll_pairwise_new(self, xv, c_num):
        c_num_half = int(c_num / 2)

        rows = []
        cols = []
        for i in range(c_num_half):
            for j in range(i + 1, c_num):
                rows.append(i)
                cols.append(j)
        with tf.variable_scope('unroll_pairwise'):
            # [b, pair, dim]
            xv_p = tf.transpose(
                # [pair, b, dim]
                tf.gather(
                    # [c_num, b, dim]
                    tf.transpose(
                        xv, [1, 0, 2]),
                    rows),
                [1, 0, 2])
            # [b, pair, dim]
            xv_q = tf.transpose(
                tf.gather(
                    tf.transpose(
                        xv, [1, 0, 2]),
                    cols),
                [1, 0, 2])
        return xv_p, xv_q



class RIM(Base):
    def __init__(self, feature_size, eb_dim, s_num, c_num, label_num, pred_mode='class'):
        super(RIM, self).__init__(feature_size, eb_dim, s_num, c_num, label_num, pred_mode)

        mask = tf.sequence_mask(self.search_res_len_ph, s_num, dtype=tf.float32)

        # use attention to represent the searched results and labels
        self.atten_score, self.search_res_rep = self._attention(self.search_res, self.search_res, self.target, mask)
        self.label_rep = tf.reduce_sum(self.atten_score * self.search_res_label, axis=1)

        # featuren interaction
        feat_inter_inp = tf.concat([self.target, self.search_res_rep, self.label_rep], axis=1)
        feat_inter_inp = tf.reshape(feat_inter_inp, [-1, 2 * c_num + 1, eb_dim])
        inter_p, inter_q = self._unroll_pairwise(feat_inter_inp, 2 * c_num + 1)
        feat_inter = tf.reduce_sum(tf.multiply(inter_p, inter_q), axis=2)

        inp = tf.concat([self.target, self.search_res_rep, feat_inter, self.label_rep], axis=1)

        # fc layer
        self._build_fc_net_res(inp, self.label_rep)
        if self.pred_mode == 'class':
            self._build_logloss()
        elif self.pred_mode == 'reg':
            self._build_mseloss()
        self._build_optimizer()

