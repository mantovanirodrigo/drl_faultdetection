# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:58:46 2023

@author: USUARIO
"""


from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

log_dir = './logs/idv11/DQN_2/'
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

tags = event_acc.Tags()
print(tags)

episode_rewards = event_acc.Scalars('rollout/ep_rew_mean')
exploration_rate = event_acc.Scalars('rollout/exploration rate')

# Convert the metrics to a pandas dataframe
data = {'Steps': [episode.step for episode in episode_rewards],
        'Episode Reward': [episode.value for episode in episode_rewards],
        'Exploration Rate': [exploration.value for exploration in exploration_rate]}
df_1 = pd.DataFrame(data).astype(np.float32)

episode_rewards_eval = event_acc.Scalars('eval/mean_reward')

# Convert the metrics to a pandas dataframe
data = {'Steps': [episode.step for episode in episode_rewards_eval],
        'Episode Reward Eval': [episode.value for episode in episode_rewards_eval]}
df_eval_1 = pd.DataFrame(data).astype(np.float32)


sns.set_style('white')
fig, ax1 = plt.subplots(figsize=(8,5))

# Plot the first data series
ax1.plot(df_1['Steps'], df_1['Episode Reward'], 'k-')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Episode Mean Reward', color='k')
ax1.plot(df_eval_1['Steps'],df_eval_1['Episode Reward Eval'],'b--',lw=1,alpha=0.8,marker='o',markersize=2.5)
#ax1.grid(True, linewidth=0.5, alpha=0.5)
ax1.set_yticks(np.arange(-240, 481, 40))
ax1.tick_params(axis='y', labelsize=9)

#for i, label in enumerate(df_eval['Episode Reward Eval']):
#    if i % 2 == 0:
#        ax1.annotate('{:.1f}'.format(label), (df_eval.iloc[i,0], df_eval.iloc[i,1]),fontsize=8)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_yticks(np.arange(0.00, 1.01, 0.20))
ax2.tick_params(axis='y', labelsize=9)
# Plot the second data series
ax2.plot(df_1['Steps'], df_1['Exploration Rate'], 'r--',lw=1,alpha=0.5)
ax2.set_ylabel('Exploration Rate', color='r')
#plt.title('Average Reward for Training and Validation - IDV1 - DQN')
ax1.legend(['Mean ep. reward (train)','Mean ep. reward (validation)'],fontsize=10,loc=(0.63,0.1))
ax2.legend(['Exploration rate'],fontsize=10,loc=(0.75,0.25))

plt.show()
fig.show()
print('                                             ')
print('---------------------------------------------')
print(f'Current number of episodes: {df_1.shape[0]*4}')
best = np.max(df_eval_1['Episode Reward Eval'])
episode_best = np.argmax(df_eval_1['Episode Reward Eval'])
print(f'Maximum eval reward: {best:.2f} (Validation step #{episode_best+1})')
print('---------------------------------------------')