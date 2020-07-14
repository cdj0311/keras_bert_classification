基于bert特征的文本分类与dssm语义表示
====
#0. ready
-------
  将bert预训练模型（chinese_L-12_H-768_A-12）放到当前目录下，下载地址：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
  
#1. bert_fc.py
------
  抽取bert句向量特征，接全连接层。
  训练命令：python bert_fc.py
  
#2. bert_lstm.py
-----
  抽取bert字向量特征，后面接LSTM/GRU和全连接层。
  训练命令：python bert_lstm.py
  
#3. bert_dssm.py
------
先将文本转换为bert句向量存在tfrecord中，这一步自行处理即可，
然后从tfrecord中读取数据，数据格式为feed_dict = {"query":[[1,2,3], [4,6,7]], "doc": [[1,2,3], [4,6,7]]}

#4. 基于bert微调DSSM向量
------
新增bert_dssm_finetune，基于bert微调的DSSM向量，使用https://github.com/CyberZHG/keras-bert 获取CLS向量然后接一层全连接。
data目录下有1000条样本数据，格式为：标题\t内容


Reference
=====
https://github.com/google-research/bert

https://github.com/CyberZHG/keras-bert
