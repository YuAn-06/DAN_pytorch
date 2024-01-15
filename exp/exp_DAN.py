"""
Copyright (C) 2024
@ Name: exp_DAN.py
@ Time: 2024/1/12 16:16
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""
from functools import partial

import torch.nn.functional
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.data_loader import Dataset_Custom, Data_Process
from exp.exp_basic import Exp_basic
from models.DAN import Extractor, Regressor
from utils import metrics
from utils.tools import *


class Exp_DAN(Exp_basic):

    def __init__(self, args):
        super(Exp_DAN, self).__init__(args)

    def _build_model(self):
        if self.args.if_re == 1:
            self.C_out = self.input_dim
        else:
            self.C_out = 1

        self.common_net = Extractor(input_dim=self.input_dim).to(self.device)
        self.src_net = Regressor().to(self.device)
        self.tgt_net = Regressor().to(self.device)
        print(self.common_net, self.src_net, self.tgt_net)

        return (self.common_net, self.src_net, self.tgt_net)

    def _get_data(self):
        # 获取训练数据和测试数据
        D = Data_Process(self.args.data_name, self.args.data_path)
        X_train, y_train, X_test, y_test = D.process_data()

        self.input_dim = X_train.shape[1]
        self.output_dim = y_train.shape[1]

        return X_train, y_train, X_test, y_test

    def _select_optimizer(self):

        model_optim = optim.SGD([{'params': self.common_net.parameters()},
                                 {'params': self.src_net.parameters()},
                                 {'params': self.tgt_net.parameters()}], lr=self.args.learning_rate, momentum=0.9)
        return model_optim

    def _select_criterion(self, src_feature, tgt_feature):
        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]

        if self.args.use_cuda:
            gaussian_kernel = partial(
                self.gaussian_kernel_matrix,
                sigmas=torch.autograd.Variable(torch.tensor(sigmas, dtype=torch.float32, device=self.device))
            )
        else:
            gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas=torch.autograd.Variable(torch.FloatTensor(sigmas))
            )

        loss_value = self.maximum_mean_discrepancy(src_feature, tgt_feature, kernel=gaussian_kernel)

        return loss_value

    def _select_scaler(self):

        if self.args.scaler == 'Minmax':
            scaler = MinMaxScaler()
        # scaler = MinMaxScaler()
        elif self.args.scaler == 'Standard':
            scaler = StandardScaler()
        else:
            raise NotImplementedError('Check your scaler if  it is Minmax or Standard')
        return scaler

    def gaussian_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_)

        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def pairwise_distance(self, x, y):

        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)

        return output

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):

        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))

        return cost

    def train(self):
        loss_hist = []
        re_hist = []
        KLD_hist = []
        self.y_scaler = self._select_scaler()
        self.X_scaler = self._select_scaler()

        self.y_train = self.y_scaler.fit_transform(self.y_train)
        self.X_train = self.X_scaler.fit_transform(self.X_train)

        self.X_test = self.X_scaler.transform(self.X_test)
        self.y_test = self.y_scaler.transform(self.y_test)

        src_data = Dataset_Custom(torch.tensor(self.X_train, dtype=torch.float32, device=self.device),
                                  torch.tensor(self.y_train, dtype=torch.float32, device=self.device), mode='2D')
        src_data_loader = DataLoader(dataset=src_data, batch_size=self.args.batch_size, drop_last=True)
        self.src_dataloader = src_data_loader
        tgt_data = Dataset_Custom(torch.tensor(self.X_test, dtype=torch.float32, device=self.device),
                                  torch.tensor(self.y_test, dtype=torch.float32, device=self.device), mode='2D')
        tgt_data_loader = DataLoader(dataset=tgt_data, batch_size=self.args.batch_size, drop_last=True)
        self.tgt_dataloader = tgt_data_loader
        optimizer = self._select_optimizer()

        self.common_net.train()
        self.src_net.train()
        self.tgt_net.train()

        for e in range(self.args.epoch):
            loss_hist.append(0)

            source_iter = iter(src_data_loader)
            target_iter = iter(tgt_data_loader)

            for batch_idx in range(min(len(src_data_loader), len(tgt_data_loader))):
                sdata = next(source_iter)
                tdata = next(target_iter)

                input1, label1 = sdata
                input2, label2 = tdata

                if self.args.use_cuda:
                    input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
                    input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
                else:

                    input1, label1 = Variable(input1), Variable(label1)
                    input2, label2 = Variable(input2), Variable(label2)

                optimizer.zero_grad()
                input = torch.cat((input1, input2), dim=0)
                common_feature = self.common_net(input)
                src_feature, tgt_feature = torch.split(common_feature, int(self.args.batch_size))
                src_output = self.src_net(src_feature)
                tgt_output = self.tgt_net(tgt_feature)

                mmd_loss = self._select_criterion(src_feature, tgt_feature) * 0.5 + self._select_criterion(src_output,
                                                                                                           tgt_output) * 0.5

                reg_loss = nn.MSELoss(reduction='mean')(src_output, label1)

                loss = mmd_loss + reg_loss

                loss.backward()
                optimizer.step()

                loss_hist[-1] += loss.item()

            print('Epoch:{}, Loss:{}'.format(e + 1, loss_hist[-1]))

        plt.figure()
        plt.plot(loss_hist, label='{} loss'.format(self.args.model))
        plt.legend()

    def test(self):
        # self.X_test = self.X_scaler.transform(self.X_test)
        # self.y_test = self.y_scaler.transform(self.y_test)

        # x_test, y_test = create_ts_data(seq_len=self.args.seq_len, data_X=self.X_test, data_y=self.y_test)

        # test_data = Dataset_Custom(torch.tensor(self.X_test, dtype=torch.float32),
        #                            torch.tensor(self.y_test, dtype=torch.float32),
        #                            mode='2D')

        src_preds_list = []
        src_trues_list = []

        tgt_preds_list = []
        tgt_trues_list = []

        self.common_net.eval()
        self.src_net.eval()

        with (torch.no_grad()):
            for batch_idx, sdata in enumerate(self.src_dataloader):
                input1, label1 = sdata
                input1, label1 = torch.tensor(input1, dtype=torch.float32, device=self.device), torch.tensor(label1,
                                                                                                             dtype=torch.float32,
                                                                                                             device=self.device)
                output1 = self.src_net(self.common_net(input1))

                src_pred = output1.detach().cpu().numpy()
                src_true = label1.detach().cpu().numpy()
                src_preds_list.append(src_pred)
                src_trues_list.append(src_true)

            for batch_idx, tdata in enumerate(self.tgt_dataloader):
                input2, label2 = tdata
                input2, label2 = torch.tensor(input2, dtype=torch.float32, device=self.device), torch.tensor(label2,
                                                                                                             dtype=torch.float32,
                                                                                                             device=self.device)
                output2 = self.src_net(self.common_net(input2))

                tgt_pred = output2.detach().cpu().numpy()
                tgt_true = label2.detach().cpu().numpy()
                tgt_preds_list.append(tgt_pred)
                tgt_trues_list.append(tgt_true)

        src_preds = np.array(src_preds_list)
        src_trues = np.array(src_trues_list)

        tgt_preds = np.array(tgt_preds_list)
        tgt_trues = np.array(tgt_trues_list)

        src_preds = src_preds.reshape(-1, src_preds.shape[-1])
        src_trues = src_trues.reshape(-1, src_trues.shape[-1])
        print("train shape:", src_preds.shape, src_trues.shape)

        mae, mse, rmse, mape, mspe = metrics.metric(src_preds, src_trues)
        r2 = r2_score(src_trues, src_preds)

        print("==== Train/Source Metrics ====")
        print("====== RMSE: {}, MAE: {}, R2: {} ======".format(rmse, mae, r2))

        tgt_preds = tgt_preds.reshape(-1, tgt_preds.shape[-1])
        tgt_trues = tgt_trues.reshape(-1, tgt_trues.shape[-1])
        print("test shape:", tgt_preds.shape, tgt_trues.shape)

        mae, mse, rmse, mape, mspe = metrics.metric(tgt_preds, tgt_trues)
        r2 = r2_score(tgt_trues, tgt_preds)

        print("==== Test/Target Metrics ====")
        print("====== RMSE: {}, MAE: {}, R2: {} ======".format(rmse, mae, r2))

        if self.args.if_ts:
            preds = tgt_preds
            trues = tgt_trues

        visual(tgt_preds, tgt_trues, self.args.if_re)
        visual(src_preds, src_trues, self.args.if_re)
