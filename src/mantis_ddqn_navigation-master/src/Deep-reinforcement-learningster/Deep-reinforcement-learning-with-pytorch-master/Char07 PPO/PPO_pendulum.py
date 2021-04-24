import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
#这个metavar就是告诉--gamma 应该这么写 【--gamma G】
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * F.tanh(self.mu_head(x))
        #取值范围是【0-1】，强行平滑以后的Relu=max(0,x)函数
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value


class Agent():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 100, 32

    def __init__(self):
        self.training_step = 0
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.buffer = []
        self.counter = 0

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

    def select_action(self, state):
        # 进行行扩围
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        # 求在给定状态下，执行采样以后动作的概率大小（其实就是带入正态分布的密度函数中去计算概率的大小，再对其求log梯度
        action_log_prob = dist.log_prob(action)
        #限定区间
        action = action.clamp(-2.0, 2.0)
        return action.item(), action_log_prob.item()

    def get_value(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.cnet(state)
        return state_value.item()

    def save_param(self):
        torch.save(self.anet.state_dict(), 'param/ppo_anet_params.pkl')
        torch.save(self.cnet.state_dict(), 'param/ppo_cnet_params.pkl')

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        # 数据类型是transition
        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)

        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)


        # 将奖赏标准化正态分布化
        r = (r - r.mean()) / (r.std() + 1e-5)
        # 这里的网络是cridit网络不含动作a的，在PPO算法中没有target网络
        with torch.no_grad():
            target_v = r + args.gamma * self.cnet(s_)
        # 分离梯度，计算advant value
        adv = (target_v - self.cnet(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):


                # 第一个actor网络的计算
                # 这里是同时计算32个样本各自的均值和方差
                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                # 其实就是两个概率相除
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])
                # PPO算法的核心，adv是优势函数，ratio和论文中设置的有点不一样，加了exp
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                # 控制梯度不要太大
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()


                # 第二个credit网络的计算，F:是numpy function
                value_loss = F.smooth_l1_loss(self.cnet(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()
        # 一定注意要清空buffer
        del self.buffer[:]


def main():
    env = gym.make('Pendulum-v0')
    env.seed(args.seed)

    agent = Agent()

    training_records = []
    running_reward = -1000
    state = env.reset()
    for i_ep in range(1000):
        score = 0
        state = env.reset()

        for t in range(200):
            action, action_log_prob = agent.select_action(state)
            state_, reward, done, _ = env.step([action])
            if args.render:
                env.render()
            if agent.store(Transition(state, action, action_log_prob, (reward + 8) / 8, state_)):
                agent.update()
                ## 一定注意要清空buffer 在update已经实现了
            score += reward
            state = state_

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % args.log_interval == 0:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))
        if running_reward > -200:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            with open('log/ppo_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            break

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("img/ppo.png")
    plt.show()


if __name__ == '__main__':
    main()
