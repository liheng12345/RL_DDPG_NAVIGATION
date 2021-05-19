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
from network import PPO_Actor, PPO_Critic
from replay_buffer import Replay_buffer
from apf import APF
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import MultivariateNormal
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# 文件保存目录
directory = "./assets/"
model_directory = directory + 'model/'
tensorboardX_directory = directory + 'tensorboardX/'
expert_directory = directory + 'expert_data/'
loss_data_directory = directory + 'loss_data/'


class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    n_steps = 100  # 每隔100步更新一次，同时令buffer_capacity = n_steps
    training_step = 0  # 神经网络训练的次数
    buffer_capacity, batch_size = n_steps, 32
    mode = 'train'
    gamma = 0.99
    learning_rate = 1e-4
    seed = True
    random_seed = 888
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_action = [0.3, 1]
    action_std_init = 0.6
    action_var = torch.full((action_dim, ),
                            action_std_init * action_std_init).to(device)
    if seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __init__(self, mode, state_dim, action_dim, max_action):
        # 多个动作需要定义方差
        self.action_var = torch.full(
            (action_dim, ),
            self.action_std_init * self.action_std_init).to(self.device)
        self.actor = PPO_Actor(mode, state_dim, action_dim,
                               max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=self.learning_rate)
        self.critic = PPO_Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=3 * self.learning_rate)
        self.replay_buffer = Replay_buffer(self.self.capacity)

        self.buffer = []
        self.counter = 0

    def select_action(self, state):
        # 进行行扩围
        # 将state数据无论是几维都装换为1行
        state = torch.FloatTensor(state.reshape(1, -1)).to(
            self.device).unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.actor(state).cpu()
        cov_mat = torch.diag(sigma)
        dist = MultivariateNormal(mu, cov_mat)
        action = dist.sample()
        # 求在给定状态下，执行采样以后动作的概率大小（其实就是带入正态分布的密度函数中去计算概率的大小，再对其求log梯度
        action_log_prob = dist.log_prob(action)
        # 限定区间
        action[0], action[1] = action[0].clamp(
            0, max_action[0]), action[1].clamp(-max_action[1], max_action[1])
        action.numpy(), action_log_prob.numpy()
        # 最后动作是两个值，action_log_prob只有一个值
        return action, action_log_prob

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def calculate_action(self, state):
        # 将state数据无论是几维都装换为1行
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # 返回在gpu中计算的actor网络中的动作值只有一个，flatten没有起到作用
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        self.training_step += 1
        # 数据类型是transition
        s = torch.tensor([t.s for t in self.buffer],
                         dtype=torch.float).to(self.device)
        a = torch.tensor([t.a for t in self.buffer],
                         dtype=torch.float).to(self.device)
        r = torch.tensor([t.r for t in self.buffer],
                         dtype=torch.float).to(self.device)
        s_ = torch.tensor([t.s_ for t in self.buffer],
                          dtype=torch.float).to(self.device)

        old_action_log_probs = torch.tensor([t.a_log_p for t in self.buffer],
                                            dtype=torch.float).to(self.device)

        # 将奖赏标准化正态分布化
        r = (r - r.mean()) / (r.std() + 1e-5)
        # 这里的网络是cridit网络不含动作a的，在PPO算法中没有target网络
        with torch.no_grad():
            target_v = r + self.gamma * self.critic(s_)
        # 分离梯度，计算advant value
        advantage_value = (target_v - self.critic(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)),
                    self.batch_size, False):

                # 第一个actor网络的计算
                # 这里是同时计算batch个样本各自的均值和方差
                (mu, sigma) = self.actor(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                # 其实就是两个概率相除
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs[index])
                # PPO算法的核心，adv是优势函数，ratio和论文中设置的有点不一样，加了exp
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                # 控制梯度不要太大
                nn.utils.clip_grad_norm_(self.anet.parameters(),
                                         self.max_grad_norm)
                self.optimizer_a.step()

                # 第二个credit网络的计算，F:是numpy function
                value_loss = F.smooth_l1_loss(self.cnet(s[index]),
                                              target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(),
                                         self.max_grad_norm)
                self.optimizer_c.step()
        # 一定注意要清空buffer
        del self.buffer[:]

    def update(self):
        for it in range(self.ppo_epoch):
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
