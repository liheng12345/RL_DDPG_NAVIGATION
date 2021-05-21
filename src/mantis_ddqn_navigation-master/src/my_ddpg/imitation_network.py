# Parser and directory related
import argparse
import os
from os.path import exists
from rospkg import RosPack

# NN related
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImitationDataset
from network import Actor
#torch.manual_seed(10)
from ppo2 import ActorCritic


class ImitationNet(nn.Module):
    '''
    改类实现了神经网络的监督学习，输入是状态对，输出是机器人的速度指令
    '''
    def __init__(self, mode, state_dim, action_dim, max_action, device='cpu'):
        super(ImitationNet, self).__init__()
        # 定义一个神经网络从state到控制输入
        self.max_action = max_action
        self.ActorCritic = ActorCritic(state_dim,
                                       action_dim,
                                       has_continuous_action_space=True,
                                       action_std_init=0.1).to(device)

    def forward(self, state):
        # 使用ppo网络时候，这里state应该转换为1维数组
        state = state.squeeze(0)
        spotplus_function = nn.Softplus()
        tanh_function = nn.Tanh()
        # 防止梯度下降报错，需要定义新的action
        action = [0, 0]
        action_mean = self.ActorCritic.actor(state)
        action[0] = spotplus_function(action_mean[0]).clamp(
            0, self.max_action[0])
        action[1] = tanh_function(action_mean[1]).clamp(
            -self.max_action[1], self.max_action[1])
        return action_mean

    # 计算神经网络和目标控制率的差
    def step(self, state, target_control):
        control_predict = self.forward(state)
        # 计算均方误差
        total_loss = ((control_predict - target_control).pow(2)).sum()
        return total_loss

    def save(self, path):
        torch.save(self.ActorCritic.state_dict(), path)

    def load(self, path):
        self.ActorCritic.load_state_dict(torch.load(path))


def train(model,
          model_path,
          save_expert_path,
          lr,
          weight_decay,
          batch_size,
          device,
          mode='dagger'):
    if mode == 'dagger':
        epochs = 50
    elif mode == 'supervised':
        epochs = 150

    # 从模仿学习的数据集中加载数据
    train_data = ImitationDataset(save_expert_path, device)  # 这里加载数据的mode没什么用

    # Set network to start training
    opt = optim.Adam(model.ActorCritic.actor.parameters(),
                     lr,
                     weight_decay=weight_decay)
    losses = []
    # 设定模型到训练模式
    model.train()

    # Iterate through epochs
    for epoch in range(epochs):
        # 打乱数据集
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size,
                                                   shuffle=True)

        # Iterations
        for state, vel in train_loader:
            # Forward and Backward pass
            opt.zero_grad()
            loss = model.step(state, vel)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Print the current status
        print("-" * 25)
        print("Epoch:{:10}".format(epoch))
        print("Train Loss:{:10.6}\t".format(np.mean(losses)))

    # Save and update the model after every full training round
    model.save(model_path + "actor" + ".pth")

    return model
