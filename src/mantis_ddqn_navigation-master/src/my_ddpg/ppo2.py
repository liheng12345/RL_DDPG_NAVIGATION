#!/usr/bin/env python3
#-*- coding:utf-8 –*-
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################

print(
    "============================================================================================"
)

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print(
    "============================================================================================"
)

################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space,
                 action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                (action_dim, ), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                       nn.Linear(64, 64), nn.Tanh(),
                                       nn.Linear(64, action_dim),
                                       nn.Softmax(dim=-1))

        # critic
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(),
                                    nn.Linear(64, 1))

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim, ),
                new_action_std * new_action_std).to(device)
        else:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print(
                "WARNING : Calling ActorCritic::set_action_std() on discrete action space policy"
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

    # 它的意思是如果这个方法没有被子类重写，但是调用了，就会报错。
    def forward(self):
        raise NotImplementedError

    def act(self, state, max_action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            spotplus_function = nn.Softplus()
            tanh_function = nn.Tanh()
            action_mean[0] = spotplus_function(action_mean[0])
            action_mean[0] = action_mean[0].clamp(0, max_action[0])
            action_mean[1] = tanh_function(action_mean[1])
            action_mean[1] = action_mean[1].clamp(-max_action[1],
                                                  max_action[1])
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor,
                 lr_critic,
                 gamma,
                 K_epochs,
                 eps_clip,
                 has_continuous_action_space,
                 action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim,
                                  has_continuous_action_space,
                                  action_std_init).to(device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(),
                                                lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr_critic)

        self.policy_old = ActorCritic(state_dim, action_dim,
                                      has_continuous_action_space,
                                      action_std_init).to(device)
        # 老策略每次重新加载新策略的值
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print(
                "WARNING : Calling PPO::set_action_std() on discrete action space policy"
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print(
            "--------------------------------------------------------------------------------------------"
        )

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ",
                      self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print(
                "WARNING : Calling PPO::decay_action_std() on discrete action space policy"
            )

        print(
            "--------------------------------------------------------------------------------------------"
        )

    def select_action(self, state, max_action):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state, max_action)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        # reversed对元素进行翻转，zip是对每个元素进行遍历,得到每一个状态对应的累记回报的reward
        for reward, is_terminal in zip(reversed(self.buffer.rewards),
                                       reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            #　第０个reward其实是最新的reward,替换
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states,
                                               dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions,
                                                dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs,
                                                 dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            print(torch.min(surr1, surr2))
            # take gradient step
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            self.critic_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic_optimizer.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

        # lambda就是定义一个函数，gpu -> cpu把数据加载到cpu
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path,
                       map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(
            torch.load(checkpoint_path,
                       map_location=lambda storage, loc: storage))
