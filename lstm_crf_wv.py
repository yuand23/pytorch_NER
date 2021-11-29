# /usr/bin/env python
# coding=utf-8
from __future__ import unicode_literals
import argparse
import os
import json
import sys
import linecache
import pdb
from os.path import join
from collections import Counter
from gensim.models import KeyedVectors
import jieba

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import init
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch BiLSTM+CRF Sequence Labeling')
parser.add_argument('--model-name', type=str, default='model', metavar='S',
                    help='model name')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='test batch size')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--embedding-size', type=int, default=512, metavar='N',
                    help='embedding size')
parser.add_argument('--hidden-size', type=int, default=1024, metavar='N',
                    help='hidden size')
parser.add_argument('--rnn-layer', type=int, default=1, metavar='N',
                    help='RNN layer num')
parser.add_argument('--with-layer-norm', action='store_true', default=False,
                    help='whether to add layer norm after RNN')
parser.add_argument('--dropout', type=float, default=0, metavar='RATE',
                    help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed')
parser.add_argument('--save-interval', type=int, default=30, metavar='N',
                    help='save interval')
parser.add_argument('--valid-interval', type=int, default=60, metavar='N',
                    help='valid interval')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='log interval')
parser.add_argument('--patience', type=int, default=30, metavar='N',
                    help='patience for early stop')
parser.add_argument('--vocab', nargs='+', required=False, metavar='SRC_VOCAB TGT_VOCAB',
                    help='src vocab and tgt vocab')
parser.add_argument('--trainset', type=str, default=os.path.join('data', 'train_clean.csv'), metavar='TRAINSET',
                    help='trainset path')
parser.add_argument('--devset', type=str, default=os.path.join('data', 'test_clean.csv'), metavar='devset',
                    help='devset path')

START_TAG = "<start>"
END_TAG = "<end>"
PAD = "<pad>"
UNK = "<unk>"

def build_corpus(data_file, make_vocab=True):
    """读取数据"""
    word_lists = []
    tag_lists = []
    with open(data_file, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回token2id和tag2id
    if make_vocab:
        token2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return [" ".join(i) for i in word_lists], [" ".join(i) for i in tag_lists], token2id, tag2id
    else:
        return [" ".join(i) for i in word_lists], [" ".join(i) for i in tag_lists]
def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps  
def extend_maps(token2id, tag2id, for_crf=True):
    token2id[UNK] = len(token2id)
    token2id[PAD] = len(token2id)
    tag2id[UNK] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        tag2id[START_TAG] = len(tag2id)
        tag2id[END_TAG] = len(tag2id)

    return token2id, tag2id
class NERDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

class SequenceLabelingDataset(Dataset):
  def __init__(self, filename):
    self._filename = filename
    with open(filename, "r", encoding="utf-8") as f:
      self._lines_count = len(f.readlines())

  def __getitem__(self, idx):
    line = linecache.getline(self._filename, idx + 1)
    return line.strip().split(",")

  def __len__(self):
    return self._lines_count
            
def log_sum_exp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.
    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

class CRFLayer(nn.Module):
  def __init__(self, tag_size):
    super(CRFLayer, self).__init__()
    # transition[i][j] means transition probability from j to i
    self.transition = nn.Parameter(torch.randn(tag_size, tag_size))
    self.reset_parameters()

  def reset_parameters(self):
    init.normal_(self.transition)
    # initialize START_TAG, END_TAG probability in log space
    self.transition.detach()[tag2id[START_TAG], :] = -10000
    self.transition.detach()[:, tag2id[END_TAG]] = -10000

  def forward(self, feats, mask):
    """
    Arg:
      feats: (seq_len, batch_size, tag_size)
      mask: (seq_len, batch_size)
    Return:
      scores: (batch_size, )
    """
    seq_len, batch_size, tag_size = feats.size()
    # initialize alpha to zero in log space
    alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
    # alpha in START_TAG is 1
    alpha[:, tag2id[START_TAG]] = 0
    for t, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      # emit_score is the same regardless of current_tag, so we broadcast along current_tag
      emit_score = feat.unsqueeze(-1) # (batch_size, tag_size, 1)
      # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
      transition_score = self.transition.unsqueeze(0) # (1, tag_size, tag_size)
      # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
      alpha_score = alpha.unsqueeze(1) # (batch_size, 1, tag_size)
      alpha_score = alpha_score + transition_score + emit_score
      # log_sum_exp along current_tag dimension to get next_tag alpha
      mask_t = mask[t].unsqueeze(-1)
      alpha = log_sum_exp(alpha_score, -1) * mask_t + alpha * (1 - mask_t) # (batch_size, tag_size)
    # arrive at END_TAG
    alpha = alpha + self.transition[tag2id[END_TAG]].unsqueeze(0)

    return log_sum_exp(alpha, -1) # (batch_size, )

  def score_sentence(self, feats, tags, mask):
    """
    Arg:
      feats: (seq_len, batch_size, tag_size)
      tags: (seq_len, batch_size)
      mask: (seq_len, batch_size)
    Return:
      scores: (batch_size, )
    """
    seq_len, batch_size, tag_size = feats.size()
    scores = feats.new_zeros(batch_size)
    tags = torch.cat([tags.new_full((1, batch_size), fill_value=tag2id[START_TAG]), tags], 0) # (seq_len + 1, batch_size)
    for t, feat in enumerate(feats):
      emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])
      transition_score = torch.stack([self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])
      scores += (emit_score + transition_score) * mask[t]
    transition_to_end = torch.stack([self.transition[tag2id[END_TAG], tag[mask[:, b].sum().long()]] for b, tag in enumerate(tags.transpose(0, 1))])
    scores += transition_to_end
    return scores

  def viterbi_decode(self, feats, mask):
    """
    :param feats: (seq_len, batch_size, tag_size)
    :param mask: (seq_len, batch_size)
    :return best_path: (seq_len, batch_size)
    """
    seq_len, batch_size, tag_size = feats.size()
    # initialize scores in log space
    scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
    scores[:, tag2id[START_TAG]] = 0
    pointers = []
    # forward
    for t, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
      # max along current_tag to obtain: next_tag score, current_tag pointer
      scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
      scores_t += feat
      pointers.append(pointer)
      mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
      scores = scores_t * mask_t + scores * (1 - mask_t)
    pointers = torch.stack(pointers, 0) # (seq_len, batch_size, tag_size)
    scores += self.transition[tag2id[END_TAG]].unsqueeze(0)
    best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )
    # backtracking
    best_path = best_tag.unsqueeze(-1).tolist() # list shape (batch_size, 1)
    for i in range(batch_size):
      best_tag_i = best_tag[i]
      seq_len_i = int(mask[:, i].sum())
      for ptr_t in reversed(pointers[:seq_len_i, i]):
        # ptr_t shape (tag_size, )
        best_tag_i = ptr_t[best_tag_i].item()
        best_path[i].append(best_tag_i)
      # pop first tag
      best_path[i].pop()
      # reverse order
      best_path[i].reverse()
    return best_path

class BiLSTMCRF(nn.Module):
  def __init__(self, vocab_size, tag_size, embedding_size, wv_size, hidden_size, num_layers, dropout, with_ln):
    super(BiLSTMCRF, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=token2id[PAD])
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(wv_size, embedding_size)
    self.bilstm = nn.LSTM(input_size=embedding_size*2,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True)
    self.with_ln = with_ln
    if with_ln:
      self.layer_norm = nn.LayerNorm(hidden_size)
    self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
    self.crf = CRFLayer(tag_size)

    self.reset_parameters()

  def reset_parameters(self):
    init.xavier_normal_(self.embedding.weight)
    init.xavier_normal_(self.hidden2tag.weight)
    init.xavier_normal_(self.linear.weight)

  def get_lstm_features(self, seq_token, seq_word, mask):
    """
    :param seq: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    embed = self.embedding(seq_token) # (seq_len, batch_size, embedding_size)
#     print(embed.shape,seq_word.shape)
    embed_words = self.linear(seq_word)
#     print("*-"*10)
    enhanced_emb = torch.cat([embed,embed_words],dim=2)
    embed = self.dropout(enhanced_emb)
    embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).cpu())
    lstm_output, _ = self.bilstm(embed) # (seq_len, batch_size, hidden_size)
    lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
    lstm_output = lstm_output * mask.unsqueeze(-1)
    if self.with_ln:
      lstm_output = self.layer_norm(lstm_output)
    lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
    return lstm_features

  def neg_log_likelihood(self, seq_token, seq_word, tags, mask):
    """
    :param seq: (seq_len, batch_size)
    :param tags: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    lstm_features = self.get_lstm_features(seq_token, seq_word, mask)
    forward_score = self.crf(lstm_features, mask)
    gold_score = self.crf.score_sentence(lstm_features, tags, mask)
    loss = (forward_score - gold_score).sum()

    return loss

  def predict(self, seq_token, seq_word, mask):
    """
    :param seq: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    lstm_features = self.get_lstm_features(seq_token, seq_word, mask)
    best_paths = self.crf.viterbi_decode(lstm_features, mask)

    return best_paths

def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list  
class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""
    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = sorted(set(self.golden_tags),key=lambda x:x[::-1])
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()
    def cal_precision(self):
        precision_scores = {}
#         print(self.predict_tags_counter)
#         sys.exit()
        for tag in self.tagset:
            try:
              precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                  self.predict_tags_counter[tag]
            except:
              precision_scores[tag] = 0
        return precision_scores
    def cal_recall(self):
        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.golden_tags_counter[tag]
        return recall_scores
    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2*p*r / (p+r+1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores
    def report_scores(self):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # 计算并打印平均值
        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))
    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict
    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average
    def _remove_Otags(self):

        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))
    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size+1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))
                 
def main(args):
  global START_TAG
  global END_TAG
  global PAD
  global UNK
  global token2id
  global tag2id

  print("Args: {}".format(args))
  use_cuda = torch.cuda.is_available() and not args.no_cuda
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.manual_seed(args.seed)
  if use_cuda:
    torch.cuda.manual_seed(args.seed)
  print("loading word2vec")
  wv = KeyedVectors.load_word2vec_format("/disc1/juan.du/main_prod/word_vectors/sgns.wiki.bigram-char",binary=False)

  print("Loading data")

  train_x, train_y, token2id, tag2id  = build_corpus(data_file=args.trainset, make_vocab=True)
  dev_x, dev_y = build_corpus(data_file=args.devset, make_vocab=False)
  token2id, tag2id = extend_maps(token2id, tag2id, for_crf=True)
  id2tag = {v:k for k,v in tag2id.items()}
  trainset = NERDataset(train_x, train_y)
  devset = NERDataset(dev_x, dev_y) 
  
  trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
  devset_loader = DataLoader(devset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

  print("Building model")
  model = BiLSTMCRF(len(token2id), len(tag2id), args.embedding_size, 300, args.hidden_size, args.rnn_layer, args.dropout,
          args.with_layer_norm).to(device)
  print(model)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  print("Start training")
  model.train()
  step = 0

  def _prepare_data(samples, vocab, pad, device=None):
    samples = list(map(lambda s: s.strip().split(" "), samples))   
    batch_size = len(samples)
    sizes = [len(s) for s in samples]
    max_size = max(sizes)
    x_np = np.full((batch_size, max_size), fill_value=vocab[pad], dtype='int64')
    for i in range(batch_size):
      x_np[i, :sizes[i]] = [vocab[token] if token in vocab else vocab[UNK] for token in samples[i]]    
    return torch.LongTensor(x_np.T).to(device), x_np.shape[1]
  def _prepare_wvdata(samples, max_len, device=None):
    word_lst = [list(jieba.cut(i.replace(" ",""))) for i in samples] 
    word_idx_lst = []
    for i,isent in enumerate(word_lst):
      isent_lst = []
      for iword in isent:
        try:  
          isent_lst.extend([wv[iword]]*len(iword))
        except:
          isent_lst.extend([np.zeros(wv[0].shape)]*len(iword)) # oov 用0
      try:
        assert max_len-len(isent_lst)+1 > 0
      except:
        print("samples[i]: ",samples[i])
        print("max_len: ",max_len)
        print("isent: ",isent)
        sys.exit()
      isent_lst.extend([np.zeros(wv[0].shape)]*(max_len-len(isent_lst))) # pad:0
      word_idx_lst.append(isent_lst)
    return torch.FloatTensor(np.array(word_idx_lst).transpose(1,0,2)).to(device)
    
  def _compute_forward(seq_token, seq_word, tags, mask):
    loss = model.neg_log_likelihood(seq_token, seq_word, tags, mask)
    batch_size = seq.size(1)
    loss /= batch_size
    loss.backward()
    return loss.item()

  def _evaluate(if_report = False):
    def get_entity(tags):
      entity = []
      prev_entity = "O"
      start = -1
      end = -1
      for i, tag in enumerate(tags):
        if tag[0] == "O":
          if prev_entity != "O":
            entity.append((start, end))
          prev_entity = "O"
        if tag[0] == "B":
          if prev_entity != "O":
            entity.append((start, end))
          prev_entity = tag[2:]
          start = end = i
        if tag[0] in ["M","E","I"]:
          if prev_entity == tag[2:]:
            end = i
      return entity
    model.eval()
    correct_num = 0
    predict_num = 0
    truth_num = 0
    pred_lst, gold_lst = [],[]
    loss_lst = []
    with torch.no_grad():
      for bidx, batch in enumerate(devset_loader):
        seq, max_len = _prepare_data(batch[0], token2id, PAD, device)
        seq_words =  _prepare_wvdata(batch[0], max_len, device)
        tags = _prepare_data(batch[1], tag2id, END_TAG, device)
        mask = torch.ne(seq, float(token2id[PAD])).float()
        length = mask.sum(0)
        _, idx = length.sort(0, descending=True)
        seq = seq[:, idx]
        tags = tags[:, idx]
        seq_words = seq_words[:, idx]
        mask = mask[:, idx]
        best_path = model.predict(seq, seq_words, mask)
        loss = model.neg_log_likelihood(seq, tags, mask)
        loss /= seq.size(1)
        loss_lst.append(loss.item())
        ground_truth = [batch[1][i].strip().split(" ") for i in idx]
        if if_report:
          pred_lst.extend([list(map(lambda x: id2tag[x], hyp)) for hyp in best_path])
          gold_lst.extend(ground_truth)
        else:
          for hyp, gold in zip(best_path, ground_truth):
            hyp = list(map(lambda x: id2tag[x], hyp))
            predict_entities = get_entity(hyp)
            gold_entities = get_entity(gold)
            correct_num += len(set(predict_entities) & set(gold_entities))
            predict_num += len(set(predict_entities))
            truth_num += len(set(gold_entities))
    # calculate F1 on entity
    model.train()
    if not if_report:
      precision = correct_num / predict_num if predict_num else 0
      recall = correct_num / truth_num if truth_num else 0
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
      return sum(loss_lst)/len(loss_lst), f1,precision,recall
    else:
      metrics = Metrics(gold_lst, pred_lst, remove_O=True)
      metrics.report_scores()
      metrics.report_confusion_matrix()    
      return 0

  best_f1 = 0
  best_prec = 0
  best_rec = 0
  patience = 0
  early_stop = False
  for eidx in range(1, args.epochs + 1):
    if eidx == 2:
      model.debug = True
    if early_stop:
      print("Early stop. epoch {} step {} best f1 {} precision {} recall {}".format(eidx, step, best_f1, best_prec, best_rec))
      sys.exit(0)
    print("Start epoch {}".format(eidx))
    for bidx, batch in enumerate(trainset_loader):
      seq, max_len = _prepare_data(batch[0], token2id, PAD, device)
      tags, _ = _prepare_data(batch[1], tag2id, END_TAG, device)
      seq_words =  _prepare_wvdata(batch[0], max_len, device)
      
      mask = torch.ne(seq, float(token2id[PAD])).float()
      length = mask.sum(0)
      _, idx = length.sort(0, descending=True)
      seq = seq[:, idx]
      tags = tags[:, idx]
      seq_words = seq_words[:, idx]
      mask = mask[:, idx]
      
      optimizer.zero_grad()
      loss = _compute_forward(seq, seq_words, tags, mask)
      optimizer.step()
      step += 1
      if step % args.log_interval == 0:
        print("epoch {} step {} batch {} loss {}".format(eidx, step, bidx, loss))
      if step % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(args.model_name, "newest.model"))
        torch.save(optimizer.state_dict(), os.path.join(args.model_name, "newest.optimizer"))
      if step % args.valid_interval == 0:
        f1, precision, recall = _evaluate(if_report=False)
        print("[valid] epoch {} step {} f1 {} precision {} recall {}".format(eidx, step, f1, precision, recall))
        if f1 > best_f1:
          patience = 0
          best_f1 = f1
          torch.save(model.state_dict(), os.path.join(args.model_name, "best.model"))
          torch.save(optimizer.state_dict(), os.path.join(args.model_name, "best.optimizer"))
        else:
          patience += 1
          if patience == args.patience:
            early_stop = True
    _evaluate(if_report=True)

if __name__ == "__main__":
  main(parser.parse_args())