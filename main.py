# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:27:48 2023

@author: USUARIO


"""

# Define IDV
IDV_no = 17

# Define ANN architecture
Arch_no = 2

# Define paths
log_path = './logs/standard/idv' + str(IDV_no)
save_path = './models/standard/idv' + str(IDV_no) + '/arch' + str(Arch_no)
model_name = '/dqn_model_idv' + str(IDV_no) +'_arch' + str(Arch_no)

# Define number of training episodes
eps_no = 3000


from environment.custom_environment import TEP_env  # adjust as necessary
from stable_baselines3.common.monitor import Monitor  # if necessary
from agents.agent_params import get_eval_callback_params, get_stop_callback_params  
from agents.DQN_agent import create_DQN, train_DQN
from utils import get_files
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold



x_train, y_train, x_val, y_val, max_values, min_values = get_files(idv = IDV_no,
                                                                   n_train = 3000,
                                                                   n_val = 600,
                                                                   w = 5,
                                                                   standard = True)


# create your environments
train_env = TEP_env(min_values, max_values,
                    steps_per_episode=480, dataset=(x_train, y_train),
                    random=False)

eval_env = TEP_env(min_values, max_values,
                    steps_per_episode=480, dataset=(x_val, y_val),
                    random=False)


train_env = Monitor(train_env, log_path)
eval_env = Monitor(eval_env, log_path)

stop_cb_params = get_stop_callback_params()
stop_callback = StopTrainingOnRewardThreshold(**stop_cb_params)

eval_cb_params = get_eval_callback_params(eval_env, stop_callback, save_path)
eval_callback = EvalCallback(**eval_cb_params)

# create and train your DQN model
model = create_DQN(train_env, log_path, arch = Arch_no)
model = train_DQN(model, 480*eps_no, eval_callback)
model.save(save_path + model_name)  # adjust as necessary
