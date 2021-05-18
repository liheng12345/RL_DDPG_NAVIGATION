#!/usr/bin/env python
#-*- coding:utf-8 –*-
import rospy
import json
import os
import random
from imitation_network import ImitationNet
import imitation_network
import torch
from apf import APF
from gazebo_ddpg import Turtlebot3GymEnv
import numpy as np


# json保存数据类型转换
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Dagger():
    def __init__(self, apf, env, mode, state_dim, action_dim, max_action,
                 save_expert_path, model_path, device):
        # dagger继承APF，APF又继承了Turtlebot3GymEnv
        self.apf = apf
        self.env = env
        self.laser = None
        self.pos = None
        self.true_pos = None
        self.currmode = None
        self.num_traj = 0
        self.lr = 0.002
        self.weight_decay = 0.1
        self.batch_size = 150
        self.beta = None
        self.beta_decay = None
        self.device = device
        # 模型、专家数据保存路径
        self.model_path = model_path
        self.save_expert_path = save_expert_path
        # 初始化环境类和人工势场专家类
        # self.env = Turtlebot3GymEnv()
        # self.apf_robot = APF(self.env)
        # 初始化数据集
        self.expert_data = None
        self.clearDataset()
        # 初始化模仿学校的模型
        self.model = ImitationNet(mode,
                                  state_dim,
                                  action_dim,
                                  max_action,
                                  device=device)
        # 获取专家数据
        self.expert_cmd = self.apf.apf_robot()
        self.robot_cmd = [0, 0]
        # dagger 模式
        self.COLLECT = 1
        self.TRAIN = 2
        self.EXECUTE = 3

    def save_expert_data(self):
        # 如果之前的专家文件已经存在，则对其进行递增
        file_name = self.save_expert_path + "trajectory_" + str(
            self.num_traj) + '.json'
        while os.path.exists(file_name):
            self.num_traj += 1
            file_name = self.save_expert_path + "trajectory_" + str(
                self.num_traj) + '.json'

        # 将数据写到文件里面
        if len(self.expert_data["robot_vel"]) > 0:

            with open(file_name, 'w') as fout:
                json.dump(self.expert_data, fout, cls=NpEncoder)
            self.num_traj += 1

    def clearDataset(self):
        self.expert_data = {'state': [], 'robot_vel': []}

    def calculate_action(self, state):
        # 将state数据无论是几维都装换为1行
        state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # 返回在gpu中计算的actor网络中的动作值只有一个，flatten没有起到作用
        return self.model(state).cpu().data.numpy().flatten()

    def collect(self):
        # Load model again
        if os.path.exists(self.model_path + 'actor.pth'):
            self.model.load(self.model_path + 'actor.pth')
        self.model = self.model.to(self.device)
        self.model.eval()
        # 本质上是将dropout作用去掉和不使用bn计算，（
        # 非训练模式）在测试模式（只有单个样本，没有了均值和方差的概念，
        # 这个时候就是固定的）

        print('starting expert planning')
        count = 0
        done = False
        rate = rospy.Rate(20)
        # 控制机器人直到到达目标点
        while True:
            # 注意这里专家是在背后指导机器人进行运动的
            # print('att:  ', self.apf.attractive(), '  rep:  ',
            #       self.apf.repulsion())
            self.expert_cmd = self.apf.apf_robot()
            if self.expert_cmd is not None:
                # 检查是否到达目标或者发生碰撞
                if done:
                    # 更新衰减因子 beta
                    self.beta *= self.beta_decay
                    print('Stopping expert planning.')
                    print('Collection done.')
                    # 存专家数据
                    if len(self.expert_data['state']) > 0:
                        self.save_expert_data()
                        self.clearDataset()
                        print('Going to train')
                        self.expert_cmd = None
                        self.currmode = self.TRAIN  #　训练完以后又开始新的收集数据
                    break
                # 定义使用专家还是模型策略
                if random.random() <= self.beta:
                    # 用当前动作与环境进行互动
                    _, _, done = self.env.step(self.expert_cmd)
                    print('using expert vel')
                else:
                    state, isCrash = self.env.calculateState(
                        self.env.getLaserData(), self.env.getOdomData())
                    # 运行神经网络发生速度指令
                    # 用模型计算的速度和原始模型比较来决策输出
                    vel = self.calculate_action(state)
                    # 如果模型计算的输出和专家测试的比较近，那么就使用模型输出的，否则就使用专家的数据
                    if abs(self.expert_cmd[0] - vel[0]) < 0.1 and abs(
                            self.expert_cmd[1] - vel[1]) < 5 / 180 * 3.14:
                        self.robot_cmd[0] = vel[0]
                        self.robot_cmd[1] = vel[1]
                        _, _, done = self.env.step(self.robot_cmd)
                        print('using model vel')
                    else:
                        print('model wrong, using expert instead')
                        if count < 0:
                            _, _, done = self.env.step(self.robot_cmd)
                            print("force robot_cmd")
                        else:
                            _, _, done = self.env.step(self.expert_cmd)
                    count += 1
                # 更新dagger数据集
                state, isCrash = self.env.calculateState(
                    self.env.getLaserData(), self.env.getOdomData())
                self.expert_data["state"] += [state]
                self.expert_data["robot_vel"] += [self.expert_cmd]
            rate.sleep()

    def train(self):
        # Load model again
        if os.path.exists(self.model_path + 'actor.pth'):
            self.model.load(self.model_path + 'actor.pth')
        self.model = self.model.to(self.device)
        # 训练神经网络
        self.model = imitation_network.train(self.model,
                                             self.model_path,
                                             self.save_expert_path,
                                             self.lr,
                                             self.weight_decay,
                                             self.batch_size,
                                             self.device,
                                             mode='dagger')
        self.model.eval()
        print("Training round complete. Resetting robot to start")
        self.env.reset()
        self.currmode = self.COLLECT

    def execute(self):
        # 加载模型
        if os.path.exists(self.model_path + 'actor.pth'):
            self.model.load(self.model_path + 'actor.pth')
        self.model = self.model.to(self.device)
        self.model.eval()
        rate = rospy.Rate(20)
        self.env.reset()
        while not rospy.is_shutdown():
            # 执行策略
            state, isCrash = self.env.calculateState(self.env.getLaserData(),
                                                     self.env.getOdomData())
            vel = self.calculate_action(state)
            _, _, done = self.env.step(vel)
            # 检查是否到达目标或者碰到障碍
            if done:
                self.env.reset()
                print("Goal is reached. Resetting robot to start")
                break
            rate.sleep()
        self.currmode = self.EXECUTE

    def run_dagger(self):
        self.beta = 1.0
        self.beta_decay = 0.1
        i = 0
        trajectory_file = self.save_expert_path + "trajectory_" + str(
            i) + '.json'
        while os.path.exists(trajectory_file):
            i += 1
            trajectory_file = self.save_expert_path + "trajectory_" + str(
                i) + '.json'
        self.beta = self.beta_decay**i
        self.currmode = self.COLLECT
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.currmode == self.COLLECT:
                self.collect()
            elif self.currmode == self.TRAIN:
                print("start train")
                self.train()
            elif self.currmode == self.EXECUTE:
                self.execute()
            rate.sleep()
