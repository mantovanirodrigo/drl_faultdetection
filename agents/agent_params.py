# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:17:01 2023

@author: USUARIO
"""

import torch

def get_DQN_params(train_env, log_path, arch):

    if arch == 1:
        policy_kwargs= dict(activation_fn=torch.nn.ReLU,net_arch=[256,128,64])
    elif arch == 2:
        policy_kwargs= dict(activation_fn=torch.nn.ReLU,net_arch=[64,128,64])
    else:
        raise ValueError("You have to choose between Architectures 1 and 2")
        
    DQN_params = {'policy':'MlpPolicy',
                  'env':train_env,
                  'verbose':1,
                  'learning_rate':0.0001,
                  'buffer_size': 48000,
                  'learning_starts':48000,
                  'batch_size':480,
                  'train_freq': (1,"step"),
                  'exploration_fraction':0.25,
                  'exploration_initial_eps':1.0,
                  'tensorboard_log':log_path,
                  'policy_kwargs':policy_kwargs}
    
    return DQN_params

def get_stop_callback_params():
    
    stop_cb_params = {'reward_threshold': 480,
                      'verbose': 1}
    
    return stop_cb_params

def get_eval_callback_params(eval_env, stop_callback, save_path):
    
    
    eval_cb_params = {'eval_env': eval_env,
                      'callback_on_new_best':stop_callback,
                      'n_eval_episodes': 600,
                      'eval_freq': 480*100,
                      'best_model_save_path':save_path,
                      'verbose':1}
    
    return eval_cb_params

