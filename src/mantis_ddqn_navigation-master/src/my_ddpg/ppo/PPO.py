#!/usr/bin/env python3
#-*- coding:utf-8 –*-
import numpy as np
from stable_baselines3.ppo import MlpPolicy
from env import CustomEnv
from gazebo_ddpg import Turtlebot3GymEnv
from stable_baselines3 import PPO, A2C  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
import os


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # 找回分解出训练时机器人收到reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    env = Turtlebot3GymEnv()
    state = env.reset()
    state_dim = len(state)
    action_dim = 2
    max_action = np.array([1, 1]).astype(np.float32)

    ###### 基础训练机器人
    # env = CustomEnv(env, state_dim, action_dim, max_action)
    # env = make_vec_env(lambda: env, n_envs=1)
    # Retrieve reward, episode length, episode success and update the buffer
    # model = PPO(MlpPolicy, env, n_steps=20, batch_size=20, verbose=1)

    ###### 监视器
    # Create log dir
    log_dir = "./model/ppo/"
    os.makedirs(log_dir, exist_ok=True)
    env = CustomEnv(env, state_dim, action_dim, max_action)
    # Logs will be saved in log_dir/monitor.csv
    # 注意monitor.csv文件的r代表最好的episode对应的的reward，len代码改episode的长度，t代表流逝的时间
    env = Monitor(env, log_dir)
    env = make_vec_env(lambda: env, n_envs=1)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1, log_dir=log_dir)
    # Train the agent
    # n_steps更新网络的间隔时间，buffer_size = n_steps
    model = PPO(MlpPolicy, env, n_steps=20, batch_size=20, verbose=1)
    model.learn(total_timesteps=int(1000), callback=callback)

    # reward绘图
    plot_results(log_dir)