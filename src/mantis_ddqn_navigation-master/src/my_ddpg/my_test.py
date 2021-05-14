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


class DDPG(object):
    def __init__(self, mode, state_dim, action_dim, max_action):
        self.actor = Actor(mode, state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(mode, state_dim, action_dim,
                                  max_action).to(device)
        # 加载估计网络的模型参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=args.learning_rate)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=args.learning_rate)
        self.replay_buffer = Replay_buffer(args.capacity)
        # tensorbordx的可视化保存文件
        self.writer = SummaryWriter(tensorboardX_directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.save_loss_data = []

    def calculate_action(self, state):
        # 将state数据无论是几维都装换为1行
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # 返回在gpu中计算的actor网络中的动作值只有一个，flatten没有起到作用
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        for it in range(args.update_iteration):
            # 从replay buffer中采集数据,s是5维的
            s, s_next, action, reward, done = self.replay_buffer.sample(
                args.batch_size)  # 这里采用得到的numpy二维数组
            state = torch.FloatTensor(s).to(device)
            next_state = torch.FloatTensor(s_next).to(device)
            action = torch.FloatTensor(action).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # 依据s_next和actor网络计算以后的action来计算target_Q
            # 梯度计算
            self.critic_optimizer.zero_grad()
            # 这里特别注意这里的状态是多维的，大小是batch_size，为了计算loss的期望s
            target_Q = self.critic_target(next_state,
                                          self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()
            # 计算估计值current_Q
            current_Q = self.critic(state, action)
            # 计算critic loss
            critic_loss = torch_function.mse_loss(current_Q, target_Q)
            # 将每一步的loss值保存在tensorboard
            self.writer.add_scalar(
                'Loss/critic_loss',
                critic_loss,
                global_step=self.num_critic_update_iteration)
            critic_loss.backward()
            self.critic_optimizer.step()

            # 冻结critic网络参数
            next_state.cpu(), action.cpu(), reward.cpu(), done.cpu(
            ), target_Q.cpu(), current_Q.cpu()
            for Critic_param in self.critic.parameters():
                Critic_param.requires_grad = False

            # 梯度计算，应该冻结critic网络的参数
            self.actor_optimizer.zero_grad()
            # 计算 actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss',
                                   actor_loss,
                                   global_step=self.num_actor_update_iteration)
            actor_loss.backward()
            self.actor_optimizer.step()
            # 解冻critic网络参数
            for Critic_param in self.critic.parameters():
                Critic_param.requires_grad = True

            with torch.no_grad():
                # 更新目标网络的参数
                for param, target_param in zip(
                        self.critic.parameters(),
                        self.critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data +
                                            (1 - args.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(),
                                               self.actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data +
                                            (1 - args.tau) * target_param.data)

            # 绘图数据保存
            self.save_loss_data.append(
                [self.num_critic_update_iteration, critic_loss, actor_loss])

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self, i):
        # 绘图数据保存
        np.savetxt(loss_data_directory + +str(i) + "loss.txt",
                   self.save_loss_data)
        print(self.save_loss_data)
        # 模型参数保存
        torch.save(self.actor.state_dict(),
                   model_directory + str(i) + '_actor.pth')
        torch.save(self.critic.state_dict(),
                   model_directory + str(i) + '_critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


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
    max_action = [1, 1]
    agent = DDPG(args.mode, state_dim, action_dim, max_action)
    # 判断是进行训练还是测试
    if args.mode == "train":
        total_step = 0
        # 初始开始使用apf算法进行数据采集
        collect_data_step_max = 1
        # 设定训练时动作执行的方差
        var = 1.
        #保存好的数据
        savelist = []
        for i in range(1, 1000000):
            episode_reward = 0
            step = 0
            state = env.reset()
            if i < collect_data_step_max:
                for t in count():
                    action = apf.apf_robot()
                    # 用当前动作与环境进行互动
                    next_state, reward, done = env.step(action)
                    # 给当前环境存放训练用的数据
                    agent.replay_buffer.push(
                        (state, next_state, action, reward, np.float(done)))
                    savelist.append(
                        (state, next_state, action, reward, np.float(done)))
                    state = next_state
                    step += 1
                    total_step += 1
                    if done is True:
                        break
                torch.save(savelist, expert_directory + "apf_teacher.json")

            else:
                if i == collect_data_step_max:
                    # 加载保存好的数据
                    savelist = torch.load(expert_directory +
                                          "apf_teacher.json")
                    for index in range(len(savelist)):
                        agent.replay_buffer.push(savelist[index])

                for t in count():
                    # 依据状态在actor网络中计算动作
                    action = agent.calculate_action(state)
                    # 在训练的时候给动作加点噪声，保证都能采样到

                    action[0] = np.clip(np.random.normal(action[0], var), 0.,
                                        max_action[0])
                    action[1] = np.clip(np.random.normal(action[1], var),
                                        -max_action[1], max_action[1])

                    # 用当前动作与环境进行互动
                    next_state, reward, done = env.step(action)

                    # 给当前环境存放训练用的数据
                    agent.replay_buffer.push(
                        (state, next_state, action, reward, np.float(done)))

                    state = next_state
                    step += 1
                    total_step += 1
                    episode_reward += reward
                    if done is True or step > 200:
                        break

                    # 每隔几步方差缩减一点
                    if step % 5 == 0:
                        var *= 0.9999
                        # 机器人运动100步，刷新一次
                    if total_step % 100 == 0:
                        print(
                            "Total T:{}   Episode: {}   average reward: {:0.2f}"
                            .format(total_step, i, episode_reward / step))
                        agent.update()
                        # 显存释放
                        torch.cuda.empty_cache()
            if i % args.save_model_interval == 0:
                agent.save(i)
    if args.mode == 'test':
        agent.load()
        total_step = 0
        for i in range(700000):
            step = 0
            state = env.reset()
            episode_reward = 0
            for t in count():
                # 依据状态在actor网络中计算动作
                action = agent.calculate_action(state)
                # 用当前动作与环境进行互动
                next_state, reward, done = env.step(action)
                state = next_state
                step += 1
                episode_reward += reward
                if done is True:
                    break
            total_step += step + 1
            print("Total T:{}   Episode: {}   Total Reward: {:0.2f}".format(
                total_step, i, episode_reward))
