# Copyright (C) 2021 #
# @Time    : 2023/6/26 10:53
# @Author  : YuAn_L
# @Email   : 2021200795@buct.edu.cn
# @File    : main.py
# @Software: PyCharm

import argparse
import random

import numpy as np
import torch


from exp.exp_DAN import Exp_DAN
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data_name', type=str, default='ccpp', help='dataset name')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of data')
parser.add_argument('--data_path', type=str, help='path of data')
parser.add_argument('--model', type=str, help='model name')

parser.add_argument('--if_ts', type=int, default=0, help='if time series dataset')
parser.add_argument('--if_re', type=int, default=0, help='if reconstruction_task')

parser.add_argument('--C_in', type=int, default=6, help='input dimension')
parser.add_argument('--seq_len', type=int, default=6, help='length of sequence')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--hidden_dim', type=int, default=45, help='the number of hidden unit')
parser.add_argument('--task', type=str, default='imputation', help='the number of hidden unit')
parser.add_argument('--scaler', type=str, default='Standard', help='type of Preprocessing data: Minax, Standard')

parser.add_argument('--batch_size', type=int, default=128, help='batch size of training input data')
parser.add_argument('--learning_rate', type=int, default=0.001, help='batch size of training input data')
parser.add_argument('--epoch', type=int, default=500, help='training epoch')

parser.add_argument('--use_cuda', type=bool, default=False, help='use gpu switch')
parser.add_argument('--gpu', type=int, default=0, help='name of gpu')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()


args.use_cuda = True
args.model = 'DAN'

print(args)
exp = Exp_DAN(args=args)

exp.train()
exp.test()
