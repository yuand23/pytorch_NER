# pytorch_NER
### 模型
使用one-hot + Bi-LSTM + CRF, word2vec + Bi-LSTM + CRF, Albert-tiny +CRF以及BERT-Base +CRF四种模型训练NER任务并对比在两个训练集上的结果。其中在word2vec + Bi-LSTM + CRF模型中使用线性变换将word2vec中的词的向量进行变换，与one-hot embedding进行拼接作为Bi-LSTM的输入；
### 数据集
data1: 简历数据集
| 训练集 | 总标记 | O标记 | 验证集 | 标签数 |
| :-----: | :-----: | :----: | :----: | :----: |
| 3821 | 13890 | 5446(占比39.21%) | 463 | 28 |

data2: 新闻数据集
| 训练集 | 总标记 | O标记 | 验证集 | 标签数 |
| :-----: | :-----: | :----: | :----: | :----: |
| 50658 | 172601 | 152505(占比88.36%) | 4631 | 7 | 
### 结果
data1
| 模型 | F1 | Precision | Recall |
| :-----: | :-----: | :----: | :----: | |
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
| BERT-base + CRF | 0.944 | 0.942 | 0.947 |  

评价标准为实体完全匹配时的得分。  

### run
one-hot + Bi-LSTM + CRF：  
python lstm_crf.py --batch-size 64  --lr 0.001 --embedding-size 256 --hidden-size 256  --trainset 训练集 --devset 验证集  
word2vec + BiLSTM + CRF：  
python lstm_crf_wv.py --batch-size 64  --lr 0.001 --embedding-size 128 --hidden-size 256  --trainset 训练集 --devset 验证集
Albert-tiny + CRF， BERT-base + CRF：  
python bert_crf.py --batch-size 64 --hidden-size 256  --trainset 训练集 --devset 验证集





