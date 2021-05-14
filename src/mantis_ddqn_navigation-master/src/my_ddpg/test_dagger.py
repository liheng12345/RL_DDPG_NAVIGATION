#!/usr/bin/env python3
#-*- coding:utf-8 –*-
from gazebo_ddpg import Turtlebot3GymEnv
import argparse
from itertools import count
import os
import numpy as np
import torch
import torch.nn.functional as torch_function
import torch.optim as optim
from network import Actor, Critic
from replay_buffer import Replay_buffer
from apf import APF
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from dagger import Dagger
# 定义参数服务器
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--tau', default=0.5,
                    type=float)  # target smoothing coefficient
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=1000000,
                    type=int)  # replay buffer size
parser.add_argument('--batch_size', default=50, type=int)  # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--save_model_interval', default=1, type=int)  #
parser.add_argument('--max_episode', default=100000, type=int)  # num of games
parser.add_argument('--update_iteration', default=10, type=int)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 返回脚本的文件名称
script_name = os.path.basename(__file__)
# 设置随机种子，保证每次实验都是随机一定的
if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

# 文件保存目录
directory = "./assets/"
model_directory = directory + 'model/'
tensorboardX_directory = directory + 'tensorboardX/'
expert_directory = directory + 'expert_data/'
loss_data_directory = directory + 'loss_data/'

if __name__ == '__main__':
    # ---------机器人的状态【雷达数据, 到目标点的朝向, 到目标点的距离, 雷达数据最小值, 雷达数据最小值的索引-----#
    # ---------机器人的状态【Laser(arr), heading, distance, obstacleMinRange, obstacleAngle】-----#
    # ---------雷达的数据是360个数据--------------------------------------------------------------#
    # odom = env.getOdomData()
    args.mode = "train"
    env = Turtlebot3GymEnv()
    apf = APF(env)
    # 24+4
    state = env.reset()
    state_dim = len(state)
    action_dim = 2
    max_action = [4, 3]
    dagger = Dagger(apf, env, args.mode, state_dim, action_dim, max_action,
                    expert_directory, model_directory, device)
    dagger.run_dagger()
