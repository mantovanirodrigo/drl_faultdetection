# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:57:39 2023

@author: USUARIO
"""
import gym
import numpy as np
import random

x_train = np.zeros((200,52))
y_train = np.zeros(200)

class TEP_env(gym.Env):
    def __init__(self, min_values,max_values,
                 steps_per_episode=480,
                 dataset=(x_train, y_train),
                 random=False,
                 ):
        super().__init__()
        #changed images_per_episode to "steps per episode"
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=min_values, high=max_values,
                                                shape=(5,52),
                                                dtype=np.float64)

        self.steps_per_episode = steps_per_episode
        self.step_count = 0
        self.x, self.y = dataset
        self.random = random
        self.dataset_idx = 0
        self.done = False

    def step(self, action):

      done = False
      if action == self.expected_action:
        reward = 1
      else:
        reward = -2

      obs = self._next_obs()

      self.step_count += 1
      if self.step_count >= self.steps_per_episode:
          done = True

      return obs, reward, done, {}

    def reset(self):
        self.step_count = 0

        obs = self._next_obs()
        return np.array(obs)

    def _next_obs(self):
        if self.random:
            next_obs_idx = random.randint(0, len(self.x) - 1)
            self.expected_action = int(self.y[next_obs_idx])
            obs = self.x[next_obs_idx]

        else:
            obs = self.x[self.dataset_idx]
            self.expected_action = int(self.y[self.dataset_idx])

            
            if self.dataset_idx >= (len(self.x)-1):
                self.dataset_idx = -1

            self.dataset_idx += 1

        return np.array(obs)