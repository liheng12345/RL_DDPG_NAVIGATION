#!/usr/bin/env python3
#-*- coding:utf-8 –*-
import torch
import torch.nn as nn
from norm_batch import BatchNorm


class Actor(nn.Module):
    def __init__(self, mode, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        if mode == "train":
            training_bool = True
        else:
            training_bool = False
        self.l1 = nn.Linear(state_dim, 300)
        self.l2 = nn.Linear(300, 200)
        self.l3 = nn.Linear(200, action_dim)

        self.max_speed = max_action[0]
        self.max_angular = max_action[1]

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        # 输出速度为-1~1之间，再通过max_action限制
        # 由于限制机器人只能前进所以线速度的输出采用sigmoid激活函数
        # 由于采用批量数据时需要对数据进行拼接，所有要把一维的向量扩维成二维的

        # -----------貌似直接一个reshape函数就可以---------------#
        x1 = torch.sigmoid(self.l3(x)[:, 0])
        x1 = x1.clamp(0, self.max_speed)
        x1 = x1.unsqueeze(1)
        x2 = torch.tanh(self.l3(x)[:, 1])
        x2 = x2.unsqueeze(1)
        x2 = x2.clamp(-self.max_angular, self.max_angular)
        # 参数0是安装行着拼接，1是安装列拼接（前提至少应该是一个二维向量）
        x = torch.cat((x1, x2), 1)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        # 这里应该让奖赏是负值，不能使用relu函数
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x
