from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json


class ImitationDataset(Dataset):
    def __init__(
        self,
        save_expert_path,
        device='cpu',
        is_val=False,
        is_test=False,
    ):
        super(ImitationDataset, self).__init__()

        # 初始化状态值
        self.state = None
        self.vel = None
        self.device = device

        # 整合所有的文件
        i = 0
        while True:
            if not os.path.exists(save_expert_path + 'trajectory_' + str(i) +
                                  '.json'):
                break
            with open(save_expert_path + 'trajectory_' + str(i) + '.json',
                      'r') as file:
                traj_dict = json.load(file)

            # 这里存储的一条轨迹中所有的数据拼接的数据，按行拼接
            traj_state = np.array(list(traj_dict.items())[0][-1])
            traj_vel = np.array(list(traj_dict.items())[1][-1])

            # 把所有文件的数据连接到一起（相当于两次拼接）
            if i == 0:
                self.state = traj_state
                self.vel = traj_vel
            else:
                # 按照行进行拼接
                self.state = np.concatenate((self.state, traj_state), axis=0)
                self.vel = np.concatenate((self.vel, traj_vel), axis=0)
            i += 1
        print('dataset done', self.state.shape[0])

        if not is_test:  # 不在测试阶段
            # 把数据集分为 (80%-20%)
            if is_val:
                start, end = -int(self.state.shape[0] * 0.2), -1
            else:
                start, end = 0, -int(self.state.shape[0] * 0.2)

            # 转换numpy到tensor
            self.state = torch.tensor(self.state).detach().to(
                self.device).type(torch.float32)
            self.vel = torch.tensor(self.vel).detach().to(self.device).type(
                torch.float32)

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, i):
        return self.state[i, :], self.vel[i, :]
