#!/usr/bin/env python3
#-*- coding:utf-8 –*-
from geometry_msgs.msg import Twist
from gazebo_turtlebot3_dqlearn import Turtlebot3GymEnv
import argparse
from itertools import count
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_function
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument('--tau', default=0.5,
                    type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=1000000,
                    type=int)  # replay buffer size
parser.add_argument('--batch_size', default=1000, type=int)  # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument(
    '--render_interval', default=100,
    type=int)  # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.5, type=float)
parser.add_argument('--max_episode', default=100000, type=int)  # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 返回脚本的文件名称
script_name = os.path.basename(__file__)
# 设置随机种子，保证每次实验都是随机一定的
if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
directory = './exp' + script_name + args.env_name + './'


class Replay_buffer():
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        # 索引容量的值，使得repaly buffer达到最大值后，从0继续开始存储
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        index = np.random.randint(0, len(self.storage), size=batch_size)
        s, s_next, action, reward, done = [], [], [], [], []

        for i in index:
            X, Y, U, R, D = self.storage[i]
            s.append(np.array(X, copy=False))
            s_next.append(np.array(Y, copy=False))
            action.append(np.array(U, copy=False))
            reward.append(np.array(R, copy=False))
            done.append(np.array(D, copy=False))

        return np.array(s), np.array(s_next), np.array(action), np.array(
            reward).reshape(-1, 1), np.array(done).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

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
        x1 = x1.unsqueeze(1)
        x2 = torch.tanh(self.l3(x)[:, 1])
        x2 = x2.unsqueeze(1)
        # 参数0是安装行着拼接，1是安装列拼接（前提至少应该是一个二维向量）
        x = torch.cat((x1, x2), 1)
        # 为了限幅需要对数据进行广播处理，让两个tensor的维度对应，输出是一个turble
        c = torch.broadcast_tensors(
            torch.Tensor([self.max_speed, self.max_angular]), x)
        x = c[0].to(device) * x
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
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        # 加载估计网络的模型参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        # tensorbordx的可视化保存文件
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def calculate_action(self, state):
        # 将state数据无论是几维都装换为1行
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # 返回在gpu中计算的actor网络中的动作值只有一个，flatten没有起到作用
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        for it in range(args.update_iteration):
            # 从replay buffer中采集数据,s是5维的
            s, s_next, action, reward, done = self.replay_buffer.sample(
                args.batch_size)
            state = torch.FloatTensor(s).to(device)
            next_state = torch.FloatTensor(s_next).to(device)
            action = torch.FloatTensor(action).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # 依据s_next和actor网络计算以后的action来计算target_Q

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

            # 梯度计算
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # 计算 actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss',
                                   actor_loss,
                                   global_step=self.num_actor_update_iteration)
            # 梯度计算
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新目标网络的参数
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data +
                                        (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data +
                                        (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


class Vector2d():
    """
    2维向量, 支持加减, 支持常量乘法(右乘)
    """
    def __init__(self, x, y):
        self.deltaX = x
        self.deltaY = y
        self.length = -1
        self.direction = [0, 0]
        self.vector2d_share()

    def vector2d_share(self):
        if type(self.deltaX) == type(list()) and type(self.deltaY) == type(
                list()):
            deltaX, deltaY = self.deltaX, self.deltaY
            self.deltaX = deltaY[0] - deltaX[0]
            self.deltaY = deltaY[1] - deltaX[1]
            self.length = math.sqrt(self.deltaX**2 + self.deltaY**2) * 1.0
            if self.length > 0:
                self.direction = [
                    self.deltaX / self.length, self.deltaY / self.length
                ]
            else:
                self.direction = None
        else:
            self.length = math.sqrt(self.deltaX**2 + self.deltaY**2) * 1.0
            if self.length > 0:
                self.direction = [
                    self.deltaX / self.length, self.deltaY / self.length
                ]
            else:
                self.direction = None

    def __add__(self, other):
        """
        + 重载
        :param other:
        :return:
        """
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.vector2d_share()
        return vec

    def __sub__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.vector2d_share()
        return vec

    def __mul__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX *= other
        vec.deltaY *= other
        vec.vector2d_share()
        return vec

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __repr__(self):
        return 'Vector deltaX:{}, deltaY:{}, length:{}, direction:{}'.format(
            self.deltaX, self.deltaY, self.length, self.direction)


#####################----APF参数设置-----#####################
"""
        :param start_posi_flag: 起点
        :param target_point: 终点
        :param obs: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param goal_threashold: 离目标点小于此值即认为到达目标点
"""


class APF(Turtlebot3GymEnv):
    def __init__(self, Turtlebot3GymEnv):
        super(APF, self).__init__()
        self.current_pos = Vector2d(0, 0)
        self.goal = Vector2d(0, 0)
        self.obstacles = [Vector2d(0, 0)]
        self.k_att = 8
        self.k_rep = 1
        self.rr = 0.9  # 斥力作用范围
        self.max_speed = 0.3
        self.max_angular = 0.8

    def attractive(self):
        """
        引力计算
        :return: 引力
        """
        att = (self.goal - self.current_pos) * self.k_att  # 方向由机器人指向目标点
        return att

    def repulsion(self):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            if obstacle.deltaX == 0 and obstacle.deltaY == 0:
                break
            else:
                # obstacle = Vector2d(0, 0)
                obs_to_rob = self.current_pos - obstacle
                rob_to_goal = self.goal - self.current_pos
                if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                    pass
                else:
                    rep_1 = Vector2d(
                        obs_to_rob.direction[0], obs_to_rob.direction[1]) * (
                            self.k_rep *
                            (1.0 / obs_to_rob.length - 1.0 / self.rr) /
                            (obs_to_rob.length**2) * (rob_to_goal.length**2))
                    rep_2 = Vector2d(
                        rob_to_goal.direction[0],
                        rob_to_goal.direction[1]) * (self.k_rep * (
                            (1.0 / obs_to_rob.length - 1.0 / self.rr)**2) *
                                                     rob_to_goal.length)
                    rep += (rep_1 + rep_2)
        return rep

    def out_put_velocity(self):
        # 初始化数据
        self.getOdomData()
        self.current_pos = Vector2d(self.position_x, self.position_y)
        self.goal = Vector2d(self.targetPointX, self.targetPointY)

        f_vec = self.attractive() + self.repulsion()
        ######----角度计算-----##########
        # 注意这里必须是atan2，否则角度会出问题
        angular = math.atan2(f_vec.direction[1],
                             f_vec.direction[0]) - self.angPos
        if angular > math.pi:
            angular = math.pi - 2 * math.pi
        if angular < -math.pi:
            angular = math.pi + 2 * math.pi
        speed_x = (self.current_pos - self.goal).length
        vel_msg = Vector2d(speed_x, angular)
        return vel_msg

    def apf_robot(self):
        motion_param = self.out_put_velocity()
        vel_msg = Twist()
        vel_msg.linear.x = motion_param.direction[0]
        vel_msg.angular.z = motion_param.direction[1]
        # 速度和角速度限制
        if vel_msg.linear.x > self.max_speed:
            vel_msg.linear.x = self.max_speed
        if vel_msg.angular.z > self.max_angular:
            vel_msg.angular.z = self.max_angular
        # 令机器人转向的的时候线速度为0
        if abs(vel_msg.angular.z) > 10.0 / 180 * math.pi:
            vel_msg.linear.x = 0
        action = [vel_msg.linear.x, vel_msg.angular.z]
        return action


if __name__ == '__main__':
    # ---------机器人的状态【雷达数据, 到目标点的朝向, 到目标点的距离, 雷达数据最小值, 雷达数据最小值的索引-----#
    # ---------机器人的状态【Laser(arr), heading, distance, obstacleMinRange, obstacleAngle】-----#
    # ---------雷达的数据是360个数据--------------------------------------------------------------#
    # odom = env.getOdomData()
    env = Turtlebot3GymEnv()
    apf = APF(env)
    # 24+4
    state = env.reset()
    state_dim = len(state)
    action_dim = 2
    max_action = [0.5, 0.5]
    agent = DDPG(state_dim, action_dim, max_action)
    # 判断是进行训练还是测试
    if args.mode == "train":
        total_step = 0
        collect_data_step_max = 100
        for i in range(100):
            episode_reward = 0
            step = 0
            state = env.reset()
            for t in count():
                if total_step < collect_data_step_max:
                    # 依据状态在actor网络中计算动作
                    # action = agent.calculate_action(state)
                    # action = [0, 0]
                    # # 给动作添加噪声
                    # action = (
                    #     action +
                    #     np.random.normal([0, 0], [0.7, 0.3], size=action_dim))
                    action = apf.apf_robot()
                    print(action[0], "  ", action[1])
                    # 用当前动作与环境进行互动
                    next_state, reward, done = env.step(action)
                    # 给当前环境存放训练用的数据
                    agent.replay_buffer.push(
                        (state, next_state, action, reward, np.float(done)))

                    state = next_state
                    step += 1
                    print(step)
                    if done is True:
                        break
                else:
                    if total_step == collect_data_step_max:
                        print("收集%d个数据完成", collect_data_step_max)
                    # 依据状态在actor网络中计算动作
                    action = agent.calculate_action(state)
                    # 用当前动作与环境进行互动
                    next_state, reward, done = env.step(action)
                    # 给当前环境存放训练用的数据
                    agent.replay_buffer.push(
                        (state, next_state, action, reward, np.float(done)))

                    state = next_state
                    step += 1
                    episode_reward += reward
                    if done is True or step > 200:
                        break

            total_step += step + 1
            print("Total T:{}   Episode: {}   Total Reward: {:0.2f}".format(
                total_step, i, episode_reward))
            agent.update()
            # agent.writer.close()
            print("更新完毕")

            if i % args.log_interval == 0:
                agent.save()