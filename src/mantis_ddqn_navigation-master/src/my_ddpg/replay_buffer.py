#!/usr/bin/env python3
#-*- coding:utf-8 –*-
import numpy as np


class Replay_buffer():
    def __init__(self, max_size=1000000):
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
