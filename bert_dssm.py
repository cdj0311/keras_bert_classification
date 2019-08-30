# coding:utf-8
import os, sys, random
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
import data_helper


os.environ["CUDA_VISIBLE_DEVICES"] = '2'


class BertDssm(object):
    def __init__(self,
                 train_corpus_path=None,
                 test_corpus_path=None,
                 model_path=None,
                 query_encoder_path=None,
                 doc_encoder_path=None,
                 n_samples=10000,
                 batch_size=128,
                 buffer_size=100,
                 NEG=10,
                 epochs=5,
                 docs_max=10,
                 doc_hidden_dim=512,
                 semantic_dim=128):
        self.train_corpus_path = train_corpus_path
        self.test_corpus_path = test_corpus_path
        self.model_path = model_path
        self.query_encoder_path = query_encoder_path
        self.doc_encoder_path = doc_encoder_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.NEG = NEG
        self.epochs = epochs
        self.docs_max = docs_max
        self.doc_hidden_dim = doc_hidden_dim
        self.semantic_dim = semantic_dim
        self.steps_per_epoch = self.n_samples // self.batch_size + 1
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def create_model(self):
        query_input = tf.keras.layers.Input(shape=(768, ), name="query_input")
        pos_doc_input = tf.keras.layers.Input(shape=(768, ), name="pos_doc_input")
        neg_docs_input = [tf.keras.layers.Input(shape=(768, ), name="neg_doc_input_%s"%j) for j in range(self.NEG)]

        # query
        query_sem = tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="query_sem")(query_input)

        # pos doc
        pos_doc_sem =tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="pos_doc_sem")(pos_doc_input)

        # neg doc
        neg_docs_sem = [tf.keras.layers.Dense(self.semantic_dim, activation="tanh")(neg_dense) for neg_dense in neg_docs_input]

        # cosine similarity
        query_pos_doc_cosine = tf.keras.layers.dot([query_sem, pos_doc_sem], axes=1, normalize=True)
        query_neg_docs_cosine = [tf.keras.layers.dot([query_sem, neg_sem], axes=1, normalize=True) for neg_sem in neg_docs_sem]
        concat_cosine = tf.keras.layers.concatenate([query_pos_doc_cosine] + query_neg_docs_cosine)
        concat_cosine = tf.keras.layers.Reshape((self.NEG + 1, 1))(concat_cosine)

        # gamma
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = tf.keras.layers.Conv1D(1, 1, padding="same", input_shape=(self.NEG + 1, 1), activation="linear", use_bias=False, weights=[weight])(concat_cosine)
        with_gamma = tf.keras.layers.Reshape((self.NEG + 1, ))(with_gamma)

        # softmax
        prob = tf.keras.layers.Activation("softmax")(with_gamma)

        # model
        model = tf.keras.models.Model(inputs=[query_input, pos_doc_input] + neg_docs_input, outputs=prob)
        
        return model

    def train(self):
        model, query_encoder, doc_encoder = self.create_model()
        model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        model.fit_generator(data_helper.train_input_fn(fin=self.train_corpus_path,
                                                       buffer_size=self.buffer_size,
                                                       batch_size=self.batch_size,
                                                       NEG=self.NEG),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs)
        model.save(self.model_path)


if __name__ == "__main__":
    bd = BertDssm()
    bd.train()



