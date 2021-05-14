#!/usr/bin/env python3
#-*- coding:utf-8 –*-

import numpy as np
from matplotlib import pyplot as plt
# 文件保存目录
directory = "./assets/"
loss_data_directory = directory + 'loss_data/loss.txt'
if __name__ == '__main__':
    loss_data = np.loadtxt(loss_data_directory)
    fig, ax = plt.subplots(2, 1, figsize=(10, 14))
    ax[0].set_title('Matplotlib demo')
    ax[0].set_xlabel("x axis caption")
    ax[0].set_ylabel("y axis caption")
    ax[0].plot(loss_data[:, 0], loss_data[:, 1])
    ax[1].set_title('Matplotlib demo')
    ax[1].set_xlabel("x axis caption")
    ax[1].set_ylabel("y axis caption")
    ax[1].plot(loss_data[:, 0], loss_data[:, 2])
    plt.show()
