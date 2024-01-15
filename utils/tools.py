"""
Copyright (C) 2023
@ Name: tools.py
@ Time: 2023/10/31 22:48
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""

import math

import torch
import numpy as np
from matplotlib import pyplot as plt



def visual(preds, trues, if_re):
    """

    :param preds: predition values
    :param trues: trues value
    :param if_re: if reconstruction task?
    :param name: model name
    :return:
    """
    _, D = preds.shape
    print(if_re)
    if if_re != 0:

        n_column = 2
        n_row = (D + 1)  // 2
        fig, axs = plt.subplots(n_row,n_column,layout='constrained')
        for i in range(D):
            ax = axs.flat[i]
            ax.plot(preds[:, i], label="Prediction", linewidth=2)
            ax.plot(trues[:, i], label="GroundTruth", linewidth=2)
        if D % 2 != 0:
            plt.delaxes(axs[-1, -1])
    else:
        plt.figure(figsize=(20,10))

        plt.plot(trues, label="GroundTruth", linewidth=2)
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.show()
