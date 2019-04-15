基于bert的keras文本分类，RNN(LSTM/GRU)和Dense finetune
====
#0. ready
-------
  将bert预训练模型（chinese_L-12_H-768_A-12）放到当前目录下
  
#1. bert_fc.py
------
  抽取bert句向量特征，接全连接层。
  训练命令：python bert_fc.py
  
#2. bert_lstm.py
-----
  抽取bert字向量特征，后面接LSTM/GRU和全连接层。
  训练命令：python bert_lstm.py
  
#3. 结果对比
------
  分类任务：微博情感分析
  
  训练样本: 25000条
  
  测试样本：5000条
  
  textcnn accuracy: 93.6%
  
  bert_fc accuracy: 94.2%
  
  bert_lstm accuracy: 96.8%
