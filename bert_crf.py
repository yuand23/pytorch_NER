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
from pprint import pprint

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import init
import numpy as np

from transformers import BertTokenizerFast, AutoConfig, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='PyTorch BiLSTM+CRF Sequence Labeling')
parser.add_argument('--model-name', type=str, default='model', metavar='S',
                    help='model name')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dev-batch-size', type=int, default=20, metavar='N',
                    help='dev batch size')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--embedding-size', type=int, default=256, metavar='N',
                    help='embedding size')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
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
parser.add_argument('--trainset', type=str, default=os.path.join('data', 'train.csv'), metavar='TRAINSET',
                    help='trainset path')
parser.add_argument('--devset', type=str, default=os.path.join('data', 'dev.csv'), metavar='devset',
                    help='devset path')

START_TAG = "<start>"
END_TAG = "<end>"
PAD = "<pad>"
UNK = "<unk>"
# dataset1: resume data
def build_corpus(data_file, make_vocab=True):
    """????????????"""

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

    # ??????make_vocab???True??????????????????tag2id
    if make_vocab:
        tag2id = build_map(tag_lists,typ='tag')
        return ["".join(i) for i in word_lists], tag_lists, tag2id
    else:
        return ["".join(i) for i in word_lists], tag_lists
def build_map(lists,typ):
    maps = {PAD:0, START_TAG:1, END_TAG:2}
    else:
        print("typ not provided in build_map function")
        maps = {} 
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps  

class NERDataset(Dataset):
    """
    ???????????????????????????????????????????????????????????????
    """
    def __init__(self, sents_src, sents_tgt, tag2id, tokenizer_path = '') :
        ## ??????init???????????????????????????
        super(NERDataset, self).__init__()
        # ???????????????
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tag2id = tag2id
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    def __getitem__(self, i):
        ## ??????????????????
        # print(i)
        src = self.sents_src[i].strip()
        tgt = [tag2id[j] for j in self.sents_tgt[i]] 
        tokenized = self.tokenizer.encode_plus(src, return_offsets_mapping=True, add_special_tokens=True)
#         tokenized = tokenizer_fast([src], is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        # e.g.
        # {'input_ids': [101, 2769, 3221, 8604, 702, 1368, 2094, 102], 
        # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 
        # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1], 
        # 'offset_mapping': [(0, 0), (0, 1), (1, 2), (2, 5), (5, 6), (6, 7), (7, 8), (0, 0)]}
#         input_ids, token_type_ids, attention_mask = self.tokenizer(src)
        # ??????tokens?????????????????????
        offset_map = [max(i[1]-i[0],1) for i in tokenized["offset_mapping"]]
        input_ids = np.repeat(np.array(tokenized["input_ids"]), offset_map).tolist()
        token_type_ids = np.repeat(np.array(tokenized["token_type_ids"]), offset_map).tolist()
        attention_mask = np.repeat(np.array(tokenized["attention_mask"]), offset_map).tolist()
        # print(self.sents_src[i])
#         print(src)
#         print("len(src):",len(src))
#         print(input_ids)
#         print("len(input_ids):",len(input_ids))
#         sys.exit()
        try:
          assert len(src) == len(input_ids)-2
        except:
          print(len(src),src)
          print(len(input_ids),input_ids)
          sys.exit()
        output = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "target_ids": tgt #
        }
        return output

    def __len__(self):
        return len(self.sents_src) 
    def __offset_map(self, offset_mapping):
        return [max(i[1]-i[0],1) for i in offset_mapping]
def collate_fn(batch):
    """
    ??????padding??? batch????????????sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad ??????
        """
#         print("-"*10)
#         print(indice)
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.LongTensor(pad_indice)
    max_length = max([len(i["input_ids"]) for i in batch])
  
    token_ids_padded = padding([i["input_ids"] for i in batch], max_length, pad_idx=0)
    token_type_ids_padded = padding([i["token_type_ids"] for i in batch], max_length, pad_idx=0)
    att_mask_padded = padding([i["attention_mask"] for i in batch], max_length, pad_idx=0)
    target_ids_padded = padding([i["target_ids"] for i in batch], max_length - 2, pad_idx=tag2id[PAD]).transpose(0,1) # ?????????cls,sep??????
    crf_mask = att_mask_padded[:,2:].transpose(0,1)
#     print("= "*10)
#     print("targets: ",[i["target_ids"] for i in batch])
#     print("token_ids_padded: ",token_ids_padded.shape,token_ids_padded)
#     print("token_type_ids_padded: ",token_type_ids_padded.shape,token_type_ids_padded)
#     print("att_mask_padded: ",att_mask_padded.shape,att_mask_padded)
#     
#     print("target_ids_padded: ",target_ids_padded.shape,target_ids_padded)
#     sys.exit()
    try:
        assert token_ids_padded.shape[1] == target_ids_padded.shape[0]+2
    except:
        print("token_ids_padded.shape,target_ids_padded.shape: ",token_ids_padded.shape,target_ids_padded.shape)
        sys.exit()
    return token_ids_padded, token_type_ids_padded, att_mask_padded, target_ids_padded, crf_mask
            
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

class BERT_CRF(nn.Module):
  def __init__(self, bert_model_path, config, tag_size):
    super(BERT_CRF, self).__init__()
#     self.dropout = nn.Dropout(dropout)
    self.bert = AutoModel.from_pretrained(bert_model_path)
    self.hidden2tag = nn.Linear(config.hidden_size, tag_size)
    self.crf = CRFLayer(tag_size)
    self.reset_parameters()

  def reset_parameters(self):
    init.xavier_normal_(self.hidden2tag.weight)

  def get_bert_features(self, token_ids, token_type_ids, att_mask):
    bert_outputs = self.bert(token_ids, token_type_ids, att_mask).last_hidden_state[:,1:-1,:] # (batch_size, seq_len, embedding_size)   
    bert_features = self.hidden2tag(bert_outputs).transpose(0,1)   # (seq_len, batch_size, tag_size)
    return bert_features 

  def neg_log_likelihood(self, token_ids, token_type_ids, att_mask, target_ids, crf_mask):
    """
    :param tags: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    # tags: (seq_len, batch_size)
    """ 
    bert_features = self.get_bert_features(token_ids, token_type_ids, att_mask)
    forward_score = self.crf(bert_features, crf_mask) # mask: (seq_len, batch_size)
    gold_score = self.crf.score_sentence(bert_features, target_ids, crf_mask)
    loss = (forward_score - gold_score).sum()

    return loss

  def predict(self, token_ids, token_type_ids, att_mask, crf_mask):
    """
    :param mask: (seq_len, batch_size)
    """
    bert_features = self.get_bert_features(token_ids, token_type_ids, att_mask)
    best_paths = self.crf.viterbi_decode(bert_features, crf_mask)

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
    """??????????????????????????????????????????????????????????????????F1??????"""
    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # ???O????????????????????????????????????
            self._remove_Otags()

        # ?????????????????????
        self.tagset = sorted(set(self.golden_tags),key=lambda x:x[::-1])
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # ???????????????
        self.precision_scores = self.cal_precision()

        # ???????????????
        self.recall_scores = self.cal_recall()

        # ??????F1??????
        self.f1_scores = self.cal_f1()
    def cal_precision(self):
        precision_scores = {}
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
            f1_scores[tag] = 2*p*r / (p+r+1e-10)  # ?????????????????????????????????????????????0
        return f1_scores
    def report_scores(self):
        """????????????????????????????????????????????????????????????

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
        # ????????????
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # ????????????????????? ????????????????????????f1??????
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # ????????????????????????
        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))
    def count_correct_tags(self):
        """???????????????????????????????????????(?????????????????????????????????????????????tp)????????????????????????????????????????????????"""
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

        # ??????weighted precisions:
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
        print("??????????????????{}????????????{}???O???????????????{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))
    def report_confusion_matrix(self):
        """??????????????????"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # ????????????????????? matrix[i][j]?????????i???tag?????????????????????j???tag?????????
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # ??????tags??????
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # ?????????????????????????????????golden_tags???????????????predict_tags?????????????????????
                continue

        # ????????????
        row_format_ = '{:>7} ' * (tags_size+1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))
                 
def main(args):
  global START_TAG
  global END_TAG
  global PAD
  global UNK
  global tag2id
  global id2tag
  
#   bert_tokenizer_path = "/disc1/juan.du/bert_models/albert_chinese_tiny" 
  bert_tokenizer_path = "/home/juan.du/nlp/bert_models/bert-base-chinese" 
#   bert_model_path = "/disc1/juan.du/bert_models/albert_chinese_tiny"
  bert_model_path = "/home/juan.du/nlp/bert_models/bert-base-chinese"
  bert_config = AutoConfig.from_pretrained(bert_model_path) 

  print("Args: {}".format(args))
  use_cuda = torch.cuda.is_available() and not args.no_cuda
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.manual_seed(args.seed)
  if use_cuda:
    torch.cuda.manual_seed(args.seed)

  print("Loading data")
  # resume data1:
  train_x, train_y, tag2id = build_corpus(data_file=args.trainset, make_vocab=True)
  dev_x, dev_y = build_corpus(data_file=args.devset, make_vocab=False)
  id2tag = {v:k for k,v in tag2id.items()}
  print("tag2id")
  print(tag2id)
#   sys.exit()
  trainset = NERDataset(train_x, train_y, tag2id, bert_tokenizer_path)
  devset = NERDataset(dev_x, dev_y, tag2id, bert_tokenizer_path)

  trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
  devset_loader = DataLoader(devset, batch_size=args.dev_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

  print("Building model")
   
  model = BERT_CRF(bert_model_path, bert_config, len(tag2id)).to(device)
  print(model)
  
#   pprint([i for i,j in params])
  # for name, child in model.named_children():
#     print("--"*10)
#     print(name)
#     if name in 'crf':
# #       print(child.named_parameters[0])
#       for nme,param in child.named_parameters():
# #       print("=="*10)
#         print(nme)
#         print(param.requires_grad)
# # #       param.requires_grad = False
# #       print("*"*10)
# #       print(model)

#   sys.exit()
  def get_group_parameters(model):
    params = list(model.named_parameters())
    no_decay = ['bias,','LayerNorm']
    other = ['lstm','linear_layer']
    no_main = no_decay + other
    
    d1 = {'params':[p for n,p in params if n.split(".")[0] in ['bert']],'weight_decay':1e-4,'lr':1e-4}
    d2 = {'params':[p for n,p in params if n.split(".")[0] in ['hidden2tag']],'weight_decay':1e-4,'lr':5e-4}
    d3 = {'params':[p for n,p in params if n.split(".")[0] in ['crf']],'weight_decay':1e-4,'lr':5e-4}
    print("Optimizer parameters: ")
    print("bert, weight_decay {},lr {}".format(d1["weight_decay"],d1["lr"]))
    print("hidden2tag, weight_decay {},lr {}".format(d2["weight_decay"],d2["lr"]))
    print("crf, weight_decay {},lr {}".format(d3["weight_decay"],d3["lr"]))
    return [d1,d2,d3]
  grouped_paras = get_group_parameters(model)
#   print(grouped_paras)
  eps = 1e-5
  warmup_steps = 100
  optimizer = AdamW(grouped_paras,eps=eps)
  t_total = len(trainset)//args.batch_size * args.epochs
  
  print("eps: ", eps)
  print("warmup_steps: ",warmup_steps)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

  print("Start training")
  model.train()
  step = 0

  def _compute_forward(token_ids, token_type_ids, att_mask, target_ids, crf_mask):
    loss = model.neg_log_likelihood(token_ids, token_type_ids, att_mask, target_ids, crf_mask)
    batch_size = token_ids.size(0)
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
        token_ids, token_type_ids, att_mask, tags, crf_mask = batch
        # ??????tags pad
        ground_truth = []
        for tag_len, tag in zip(crf_mask.sum(0).tolist(),tags.transpose(0,1).tolist()):
          ground_truth.append(tag[:tag_len])
        token_ids = token_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        att_mask = att_mask.to(device)
        crf_mask = crf_mask.to(device)
        
        best_path = model.predict(token_ids, token_type_ids, att_mask, crf_mask)
        loss = model.neg_log_likelihood(token_ids, token_type_ids, att_mask, tags, crf_mask)
        loss /= token_ids.size(0)
        loss_lst.append(loss.item())
        if if_report:
          pred_lst.extend([list(map(lambda x: id2tag[x], i)) for i in best_path])
          gold_lst.extend([list(map(lambda x: id2tag[x], i)) for i in ground_truth])
        else:
          for hyp, gold in zip(best_path, ground_truth):
            hyp = list(map(lambda x: id2tag[x], hyp))
            gold = list(map(lambda x: id2tag[x], gold))
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
      return sum(loss_lst)/len(loss_lst),f1,precision,recall
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
      optimizer.zero_grad()
      token_ids, token_type_ids, att_mask, tags, crf_mask = batch
      
      token_ids = token_ids.to(device)
      token_type_ids = token_type_ids.to(device)
      att_mask = att_mask.to(device)
      tags = tags.to(device)
      crf_mask = crf_mask.to(device)
      
      loss = _compute_forward(token_ids, token_type_ids, att_mask, tags, crf_mask)
      optimizer.step()
      scheduler.step()
      step += 1
      if step % args.log_interval == 0:
        print("epoch {} step {} batch {} train loss {}".format(eidx, step, bidx, loss))
      if step % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(args.model_name, "newest.model"))
        torch.save(optimizer.state_dict(), os.path.join(args.model_name, "newest.optimizer"))
      if step % args.valid_interval == 0:
        dev_loss, f1, precision, recall = _evaluate(if_report=False)
        print("[valid] epoch {} step {} dev mean loss {} f1 {} precision {} recall {}".format(eidx, step, dev_loss, f1, precision, recall))
        if f1 > best_f1:
          patience = 0
          best_f1 = f1
          best_prec = precision
          best_rec = recall
          torch.save(model.state_dict(), os.path.join(args.model_name, "best.model"))
          torch.save(optimizer.state_dict(), os.path.join(args.model_name, "best.optimizer"))
        else:
          patience += 1
          if patience == args.patience:
            early_stop = True
    _evaluate(if_report=True)

if __name__ == "__main__":
  main(parser.parse_args())