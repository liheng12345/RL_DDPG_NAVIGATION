import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from gazebo_ddpg import Turtlebot3GymEnv
from stable_baselines3 import PPO, A2C  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['turtlebot3']}

    def __init__(self, env, state_dim, action_dim, max_action):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.env = env
        self.action_space = spaces.Box(low=np.array([0, -max_action[1]]),
                                       high=np.array(
                                           [max_action[0], max_action[1]]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=float("-inf"),
                                            high=float("inf"),
                                            shape=(state_dim, ),
                                            dtype=np.float32)

    def step(self, action):
        """
            Important: the observation must be a numpy array
            :return: (np.array) 
        """
        observation, reward, done = self.env.step(action)
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        return observation.astype(np.float32), reward, done, info

    def reset(self):
        observation = self.env.reset()
        return observation.astype(
            np.float32)  # reward, done, info can't be included


# 测试环境是否配置好
# if __name__ == '__main__':
#     env = Turtlebot3GymEnv()
#     state = env.reset()
#     state_dim = len(state)
#     action_dim = 2
#     max_action = np.array([1, 1]).astype(np.float32)
#     env = CustomEnv(env, state_dim, action_dim, max_action)
#     # If the environment don't follow the interface, an error will be thrown
#     check_env(env, warn=True)

# 验证环境
# if __name__ == '__main__':
#     env = Turtlebot3GymEnv()
#     state = env.reset()
#     state_dim = len(state)
#     action_dim = 2
#     max_action = np.array([1, 1]).astype(np.float32)
#     env = CustomEnv(env, state_dim, action_dim, max_action)
#     obs = env.reset()

#     print(env.observation_space)
#     print(env.action_space)
#     print(env.action_space.sample())
#     n_steps = 20
#     for step in range(n_steps):
#         print("Step {}".format(step + 1))
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         print('obs=', obs, 'reward=', reward, 'done=', done)
#         if done:
#             print("Goal reached!", "reward=", reward)
#             break

# 测试算法
if __name__ == '__main__':
    env = Turtlebot3GymEnv()
    state = env.reset()
    state_dim = len(state)
    action_dim = 2
    max_action = np.array([1, 1]).astype(np.float32)
    env = CustomEnv(env, state_dim, action_dim, max_action)
    # 创建一个环境
    env = make_vec_env(lambda: env, n_envs=1)
    # Train the agent
    model = PPO(MlpPolicy, env, verbose=0)
    obs = env.reset()
    n_steps = 20
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break