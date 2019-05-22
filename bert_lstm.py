# coding:utf-8
import os
import codecs
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, LSTM, GRU, BatchNormalization, Dense, Dropout, GlobalAveragePooling1D, Masking, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from bert.extract_feature import BertVector

"""
将bert预训练模型（chinese_L-12_H-768_A-12）放到当前目录下
基于bert字向量的文本分类：基于GRU的微调
"""
class BertClassification(object):
    def __init__(self,
                 nb_classes=2,
                 gru_dim=128,
                 dense_dim=128,
                 max_len=100,
                 batch_size=128,
                 epochs=10,
                 train_corpus_path="data/sent.train",
                 test_corpus_path="data/sent.test",
                 save_weights_file="./model/weights_lstm.h5"):
        self.nb_classes = nb_classes
        self.gru_dim = gru_dim
        self.dense_dim = dense_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_corpus_path=train_corpus_path
        self.test_corpus_path=test_corpus_path
        self.save_weights_file = save_weights_file

        self.nb_samples = 25000 # 样本数
        self.bert_model = BertVector(pooling_strategy="NONE", 
                                     max_seq_len=self.max_len, 
                                     bert_model_path="./chinese_L-12_H-768_A-12/",
280                                  graph_tmpfile="./tmp_graph_xxx)

    def text2bert(self, text):
        """ 将文本转换为bert向量  """
        vec = self.bert_model.encode([text])
        return vec["encodes"][0]

    def data_format(self, lines):
        X, y = [], []
        for line in lines:
            line = line.strip().split("\t")
            label = int(line[0])
            content = line[1]
            vec = self.text2bert(content)
            X.append(vec)
            y.append(label)
        X = np.array(X)
        y = np_utils.to_categorical(np.asarray(y), num_classes=self.nb_classes)
        return X, y

    def data_iter(self):
        """ 数据生成器 """
        fr = codecs.open(self.train_corpus_path, "r", "utf-8")
        lines = fr.readlines()
        fr.close()
        random.shuffle(lines)
        while True:
            for index in range(0, len(lines), self.batch_size):
                batch_samples = lines[index: index+self.batch_size]
                X, y = self.data_format(batch_samples)
                yield (X, y)

    def data_val(self):
        """ 测试数据 """
        fr = codecs.open(self.test_corpus_path, "r", "utf-8")
        lines = fr.readlines()
        fr.close()
        random.shuffle(lines)
        X,y = self.data_format(lines)
        return X,y

    def create_model(self):
        x_in = Input(shape=(self.max_len, 768, ))
        x_out = Masking(mask_value=0.0)(x_in)
        x_out = GRU(self.gru_dim, dropout=0.25, recurrent_dropout=0.25)(x_out)
        x_out = Dense(self.dense_dim, activation="relu")(x_out)
        x_out = BatchNormalization()(x_out)
        x_out = Dense(self.nb_classes, activation="softmax")(x_out)
        model = Model(inputs=x_in, outputs=x_out)
        return model

    def train(self):
        model = self.create_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(self.save_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        x_test, y_test = self.data_val()
        steps_per_epoch = int(self.nb_samples/self.batch_size)+1
        model.fit_generator(self.data_iter(),
                            steps_per_epoch=steps_per_epoch,
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            validation_steps=None,
                            callbacks=[checkpoint]
                            )


if __name__ == "__main__":
    bc = BertClassification()
    bc.train()





