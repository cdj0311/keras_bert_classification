# coding:utf-8
import numpy as np
import tensorflow as tf


def non_zero_mean(np_arr):
    """ 求非零向量的均值 """
    exist = (np_arr != 0)
    num = np_arr.sum(axis=1)
    den = exist.sum(axis=1)
    return np.divide(num, den)

def _feed_dict_fn(samples, batch_size, NEG=4):
    """
    负采样：
    输入为batch_size大小的dict，{"query":[], "doc":[]}
    在batch_size内进行采样NEG个doc作为负样本
    """
    query_list = np.array(samples["query"])
    pos_doc_list = np.array(samples["doc"]).reshape((batch_size, 10, 768))
    pos_doc_list = non_zero_mean(pos_doc_list)
    neg_docs_list = [[] for _ in range(NEG)]

    for i in range(len(pos_doc_list)):
        poss = list(range(len(pos_doc_list)))
        poss.remove(i)
        negatives = np.random.choice(poss, NEG, replace=False)
        for j in range(NEG):
            negative = negatives[j]
            neg_docs_list[j].append(pos_doc_list[negative])
    for j in range(NEG):
        neg_docs_list[j] = np.array(neg_docs_list[j])
    y = np.zeros((len(query_list), NEG + 1))
    y[:, 0] = 1
    return query_list, pos_doc_list, neg_docs_list, y


def _parse_tfrecord(example):
    features = {"query": tf.FixedLenFeature([768], tf.float32),
                "doc": tf.FixedLenFeature([768*10], tf.float32)}
    features = tf.parse_single_example(example, features)
    return features


def train_input_fn(fin=None,
                   buffer_size=100,
                   batch_size=5,
                   NEG=4):
    data = tf.data.TFRecordDataset(fin)
    data = data.shuffle(buffer_size=buffer_size)
    data = data.map(_parse_tfrecord)
    data = data.prefetch(1)
    data = data.batch(batch_size).repeat()
    data = data.make_one_shot_iterator().get_next()
    sess = tf.Session()
    while True:
        feed_dict = sess.run(data)
        query_list, pos_doc_list, neg_docs_list, y = _feed_dict_fn(feed_dict, batch_size, NEG)
        yield ([query_list, pos_doc_list] + neg_docs_list, y)
    sess.close()


if __name__ == "__main__":
    train_input_fn(buffer_size=10, batch_size=4, NEG=2)
