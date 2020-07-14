# coding:utf-8
import codecs, sys, os, random
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras_bert import Tokenizer

class SaveModel(Callback):
    def __init__(self, q_encoder, d_encoder, query_encoder_path, doc_encoder_path, model_path):
        self.q_encoder = q_encoder
        self.d_encoder = d_encoder
        self.query_encoder_path = query_encoder_path
        self.doc_encoder_path = doc_encoder_path
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path[:-3] + "_%s.h5"%epoch)
        self.q_encoder.save(self.query_encoder_path[:-3] + "_%s.h5"%epoch)
        self.d_encoder.save(self.doc_encoder_path[:-3] + "_%s.h5"%epoch)


def _parse_data(lines, neg, max_len, tokenizer):
    query_t, pos_doc_t, neg_docs_t = [], [], [[] for _ in range(neg)]
    query_s, pos_doc_s, neg_docs_s = [], [], [[] for _ in range(neg)]

    for line in lines:
        line = line.strip().split("\t")
        qt, qs = tokenizer.encode(line[0], max_len=max_len)
        dt, ds = tokenizer.encode(line[1], max_len=max_len)
        query_t.append(qt)
        query_s.append(qs)
        pos_doc_t.append(dt)
        pos_doc_s.append(ds)

    # 负采样
    for i in range(len(pos_doc_t)):
        poss = list(range(len(pos_doc_t)))
        poss.remove(i)
        negatives = np.random.choice(poss, neg, replace=False)
        for j in range(neg):
            negative = negatives[j]
            neg_docs_t[j].append(pos_doc_t[negative])
            neg_docs_s[j].append(pos_doc_s[negative])
    # numpy
    query_t = np.array(query_t)
    query_s = np.array(query_s)
    pos_doc_t = np.array(pos_doc_t)
    pos_doc_s = np.array(pos_doc_s)
    neg_docs_t = [np.array(neg_docs_t[i]) for i in range(neg)]
    neg_docs_s = [np.array(neg_docs_s[i]) for i in range(neg)]

    # label
    y = np.zeros((len(query_t), neg+1))
    y[:, 0] = 1

    return query_t, query_s, pos_doc_t, pos_doc_s, neg_docs_t, neg_docs_s, y

def train_input_fn(corpus_path, batch_size=128, neg=4, max_len=10, tokenizer=None):
    fr = codecs.open(corpus_path, "r", "utf-8")
    lines = fr.readlines()
    fr.close()
    random.shuffle(lines)

    while True:
        for index in range(0, len(lines), batch_size):
            batch_samples = lines[index: index + batch_size]
            query_t, query_s, pos_doc_t, pos_doc_s, neg_docs_t, neg_docs_s, y = \
                _parse_data(batch_samples, neg, max_len, tokenizer)
            yield ([query_t, query_s, pos_doc_t, pos_doc_s] + neg_docs_t + neg_docs_s, y)
