# pytorch_NER
### 模型
使用one-hot + Bi-LSTM + CRF, word2vec + Bi-LSTM + CRF, Albert-tiny +CRF以及BERT-Base +CRF四种模型训练NER任务并对比在两个训练集上的结果。其中在word2vec + Bi-LSTM + CRF模型中使用线性变换将word2vec中的词的向量进行变换，与one-hot embedding进行拼接作为Bi-LSTM的输入；
### 结果
data1
| 模型 | F1 | Precision | Recall |
| :-----: | :-----: | :----: | :----: |
| one-hot + Bi-LSTM + CRF | 0.911 | 0.911 | 0.911 |
| word2vec + BiLSTM + CRF | 0.928 | 0.934 | 0.923 |  
| Albert-tiny + CRF | 0.936 | 0.936 | 0.936 |  
| BERT-base + CRF | 0.962 | 0.958 | 0.966 |  

data2
| 模型 | F1 | Precision | Recall |
| :-----: | :-----: | :----: | :----: |
| one-hot + Bi-LSTM + CRF | 0.856 | 0.873 | 0.840 |
| word2vec + BiLSTM + CRF | 0.895 | 0.908 | 0.882 |  
| Albert-tiny + CRF | 0.936 | 0.899 | 0.882 |  
| BERT-base + CRF | 0.9445 | 0.942 | 0.947 |  

