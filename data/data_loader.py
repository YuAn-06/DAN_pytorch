# Copyright (C) 2021 #
# @Time    : 2023/6/28 11:03
# @Author  : YuAn_L
# @Email   : 2021200795@buct.edu.cn
# @File    : data_loader.py
# @Software: PyCharm


from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class Dataset_Custom(Dataset):
    def __init__(self,data,label,mode):

        self.data = data
        self.label = label
        self.mode = mode

    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode =='3D':
            return self. data.shape[1]

    def __getitem__(self, item):
        if self.mode == '2D':
            return self.data[item, :], self.label[item, :]
        elif self.mode == '3D':
            return self.data[:,item, :], self.label[:,item,:]

class Data_Process():

    def __init__(self,data_name,data_path,task=None):
        self.name = data_name # data name
        self.data_path = data_path
        self.task = task
        # self.process_data()
          
    def process_data(self):
        if self.name == 'ccpp':
            pp = pd.read_csv('C:\Study\Code\dataset\PP_GAS\gt_2011.csv').values

            x = pp[:, :-2]
            y = pp[:, -1]

            x_train = x[:6000, :]
            y_train = y[:6000].reshape(-1,1)
            x_test = x[6000:7500, :]
            y_test = y[6000:7500].reshape(-1,1)
        else:
            raise NotImplementedError(
                'Dataset doesnt exits, Please check if dataset name is one of pta, deb, tff, te, ccpp and sru.')

        return x_train, y_train, x_test, y_test
