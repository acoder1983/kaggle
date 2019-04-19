import os
import time
from os.path import join

import numpy as np
import pandas as pd
import tensorflow as tf

import xdl

int_fea_num = 13
cat_fea_num = 26

tra_files = sorted([join('./raw_data/', f)
                    for f in os.listdir('./raw_data') if f.startswith('tra_')])

reader_tra = xdl.DataReader("reader_tra",
                            paths=tra_files,
                            enable_state=False)  # enable reader state

reader_tra.epochs(3).threads(1).batch_size(256).label_count(1)

reader_tst = xdl.DataReader("reader_tst",
                            paths=["./raw_data/tst.xdl"],
                            enable_state=False)  # enable reader state

reader_tst.epochs(1).threads(1).batch_size(1).label_count(1)


def init_reader(reader):
    for i in range(int_fea_num):
        reader.feature(name=str(i), type=xdl.features.dense, nvec=1)

    for i in range(int_fea_num, int_fea_num + cat_fea_num):
        reader.feature(name=str(i), type=xdl.features.sparse)

    reader.startup()


init_reader(reader_tra)
init_reader(reader_tst)


def fc(inputs, w_shape, b_shape, name):
    weight = tf.get_variable("%s_weights" % name,
                             w_shape,
                             initializer=tf.zeros_initializer(tf.float32))
    bias = tf.get_variable("%s_bias" % name,
                           b_shape,
                           initializer=tf.zeros_initializer(tf.float32))
    return tf.matmul(inputs, weight) + bias


k = 16  # embedding dim


def build_model(numerics, embeddings):

    input = tf.concat(numerics + embeddings, 1)
    x = fc(input, [len(numerics) + len(embeddings) * k, 256], [256], 'fc_1')
    x = fc(x, [256, 64], [64], 'fc_2')
    x = fc(x, [64, 16], [16], 'fc_3')
    y = fc(x, [16, 1], [1], 'fc_4')
    return y


@xdl.tf_wrapper(is_training=True)
def train_model(numerics, embeddings, labels):
    with tf.variable_scope("train"):
        y = build_model(numerics, embeddings)
        loss = tf.losses.sigmoid_cross_entropy(labels, y)
    return loss


@xdl.tf_wrapper(is_training=False)
def pred_model(numerics, embeddings):
    with tf.variable_scope("train", reuse=True):
        return build_model(numerics, embeddings)


emb_dim = {15: 11299105, 24: 9292738, 33: 7822987, 28: 6047969, 16: 2416541, 36: 303075, 38: 148165, 22: 95979, 27: 15210, 19: 12597,
           23: 5724, 30: 5721, 25: 3207, 31: 2178, 13: 1460, 20: 633, 14: 585, 17: 305, 37: 105, 26: 27, 18: 24, 34: 18, 35: 15, 29: 10, 32: 4, 21: 3}


def get_input(batch):
    numerics = [batch[str(c)] for c in range(int_fea_num)]
    embeddings = [xdl.embedding('emb%d' % c, batch[str(c)], xdl.TruncatedNormal(
        stddev=0.001), k, emb_dim[c], vtype='hash') for c in range(int_fea_num, int_fea_num + cat_fea_num)]
    return numerics, embeddings


def train_predict():
    batch_tra = reader_tra.read()
    batch_tst = reader_tst.read()

    with xdl.model_scope('train'):
        numerics, embeddings = get_input(batch_tra)

        loss = train_model(numerics, embeddings, batch_tra['label'])
        train_op = xdl.Adam().optimize()

        interval = 10
        log_hook = xdl.LoggerHook(loss, "loss:{0}", interval)
        sess = xdl.TrainSession(hooks=[log_hook])
        i = 0
        while not sess.should_stop():
            if i % interval == 0:
                print('train batch %d' % i)
            sess.run(train_op)
            i += 1

    with xdl.model_scope('predict'):
        result = pd.read_csv(
            'raw_data/random_submission.csv', dtype={'Id': 'str'})
        y_pred = np.zeros((result.shape[0],))

        numerics, embeddings = get_input(batch_tst)
        y = pred_model(numerics, embeddings)
        pred_sess = xdl.TrainSession()
        i = 0
        while True:
            val = pred_sess.run(y)
            if val is None:
                break
            val = np.array(val).ravel()
            y_pred[i:i + len(val)] = val
            i += len(val)
        print('pred %d' % i)

        result.Predicted = y_pred
        result[['Id', 'Predicted']].to_csv(
            'output/xdl_f39_k16_h256_h64_h16.csv', index=False)


t = time.time()
train_predict()
print('time cost %ds' % (time.time() - t))
