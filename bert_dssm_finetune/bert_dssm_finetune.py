# coding:utf-8
import os, sys, codecs, json
import numpy as np
import tensorflow as tf
from keras.utils import Sequence, np_utils, multi_gpu_model
from keras.optimizers import Adam
from keras.layers import Dropout, Input, Dot, dot, concatenate, Reshape, GlobalAveragePooling1D
from keras.layers import CuDNNLSTM, LSTM, Bidirectional, Activation, Conv1D, Dense, GRU, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras_radam import RAdam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import bert_data_helper

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class BertDSSM(object):
    def __init__(self,
                 train_corpus_path=None,
                 test_corpus_path=None,
                 query_encoder_path=None,
                 doc_encoder_path=None,
                 model_path=None,
                 train_samples=10000,
                 batch_size=16,
                 epochs=1,
                 max_len=100,
                 NEG=4,
                 lr=0.00001,
                 checkpoint_path="chinese_L-12_H-768_A-12"):
        self.train_corpus_path = train_corpus_path
        self.test_corpus_path = test_corpus_path
        self.query_encoder_path = query_encoder_path
        self.doc_encoder_path = doc_encoder_path
        self.model_path = model_path
        self.train_samples = train_samples
        self.query_max_len = max_len
        self.doc_max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.NEG = NEG
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        self.vocab_path = os.path.join(checkpoint_path, 'vocab.txt')
        self.config_path = os.path.join(checkpoint_path, 'bert_config.json')
        self.ckpt_path = os.path.join(checkpoint_path, 'bert_model.ckpt')
        self.tokenizer = self.load_dict()

    def load_dict(self):
        token_dict = {}
        with codecs.open(self.vocab_path, "r", "utf-8") as fr:
            for line in fr:
                line = line.strip()
                token_dict[line] = len(token_dict)
        tokenizer = Tokenizer(token_dict)
        return tokenizer

    def create_model(self):
        # 0. input
        query_input_token = Input(shape=(self.query_max_len, ), name="query_input_token")
        query_input_segment = Input(shape=(self.query_max_len,), name="query_input_segment")
        pos_doc_input_token = Input(shape=(self.doc_max_len,), name="pos_doc_input_token")
        pos_doc_input_segment = Input(shape=(self.doc_max_len,), name="pos_doc_input_segment")
        neg_doc_input_token = [Input(shape=(self.doc_max_len,)) for _ in range(self.NEG)]
        neg_doc_input_segment = [Input(shape=(self.doc_max_len,)) for _ in range(self.NEG)]

        # 1. define bert model
        bert_model = load_trained_model_from_checkpoint(
            config_file=self.config_path,
            checkpoint_file=self.ckpt_path,
            seq_len=self.query_max_len
        )
        for l in bert_model.layers:
            l.trainable = True

        # 2. query
        query_bert = bert_model([query_input_token, query_input_segment])
        query_l = Lambda(lambda x:x[:, 0])(query_bert)
        query_sem = Dense(128, activation="tanh", name="query_sem")(query_l)
 
        # 3. doc
        doc_sem = Dense(128, activation="tanh", name="doc_sem")

        # 4. pos doc
        pos_doc_bert = bert_model([pos_doc_input_token, pos_doc_input_segment])
        pos_doc_l = Lambda(lambda x:x[:,0])(pos_doc_bert)
        pos_doc_sem = doc_sem(pos_doc_l)

        # 5. neg odc
        neg_docs_bert = [bert_model([neg_doc_t, neg_doc_s])
                         for neg_doc_t, neg_doc_s in zip(neg_doc_input_token, neg_doc_input_segment)]
        neg_docs_l = [Lambda(lambda x:x[:,0])(neg_doc_bert) for neg_doc_bert in neg_docs_bert]
        neg_docs_sem = [doc_sem(neg_doc_l) for neg_doc_l in neg_docs_l]

        # 6. cosine
        query_pos_cos = dot([query_sem, pos_doc_sem], axes=1, normalize=True)
        query_negs_cos = [dot([query_sem, neg_doc_sem], axes=1, normalize=True) for neg_doc_sem in neg_docs_sem]
        concat_cosine = concatenate([query_pos_cos] + query_negs_cos)
        concat_cosine = Reshape((self.NEG + 1, 1))(concat_cosine)

        # 7. gamma
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = Conv1D(1, 1, padding="same", input_shape=(self.NEG + 1, 1), activation="linear", use_bias=False,
                            weights=[weight])(concat_cosine)
        with_gamma = Reshape((self.NEG + 1,))(with_gamma)

        # 8. softmax
        prob = Activation("softmax")(with_gamma)

        # 9. output
        model = Model(inputs=[query_input_token, query_input_segment,
                              pos_doc_input_token, pos_doc_input_segment] + neg_doc_input_token + neg_doc_input_segment,
                      outputs=prob)

        query_encoder = Model(inputs=[model.get_layer("query_input_token").input,
                                      model.get_layer("query_input_segment").input],
                              outputs=model.get_layer("query_sem").output)

        doc_encoder = Model(inputs=[model.get_layer("pos_doc_input_token").input,
                                    model.get_layer("pos_doc_input_segment").input],
                            outputs=model.get_layer("doc_sem").get_output_at(0))
        return model, query_encoder, doc_encoder

    def train(self):
        model, query_encoder, doc_encoder = self.create_model()
        p_model = multi_gpu_model(model, gpus=4) 
        print(p_model.summary())
        p_model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.lr), metrics=["accuracy"])
        train_iter = bert_data_helper.train_input_fn(self.train_corpus_path,
                                                self.batch_size,
                                                self.NEG,
                                                self.query_max_len,
                                                self.tokenizer)
        p_model.fit_generator(train_iter,
                            steps_per_epoch=self.train_samples//self.batch_size,
                            epochs=self.epochs,
                            callbacks=[bert_data_helper.SaveModel(query_encoder, doc_encoder, self.query_encoder_path, self.doc_encoder_path, self.model_path)])



if __name__ == "__main__":
    bert_dssm = BertDSSM(train_corpus_path="./dssm_train_data.txt",
                         query_encoder_path="model/bert_dssm_query.h5",
                         doc_encoder_path="model/bert_dssm_doc.h5",
                         model_path="model/bert_dssm_model.h5",
                         epochs=3,
                         train_samples=2400000,
                         max_len=100,
                         batch_size=32,
                         NEG=6,
                         lr=0.00001,
                         checkpoint_path="../chinese_L-12_H-768_A-12"
                         )
    bert_dssm.train()

