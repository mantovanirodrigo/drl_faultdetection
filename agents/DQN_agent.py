# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:21:42 2023

@author: USUARIO
"""

from stable_baselines3 import DQN
from .agent_params import get_DQN_params

def create_DQN(train_env, log_path, arch):
    params = get_DQN_params(train_env, log_path, arch)
    model = DQN(**params)
    return model

def train_DQN(model, total_timesteps, eval_callback):
    
    model.learn(total_timesteps=total_timesteps,callback = eval_callback)
    return model