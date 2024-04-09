# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:19:38 2023

@author: USUARIO
"""

#from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import sliding_window, sliding_window_max
from utils import time_to_detection, time_to_detection_normal

from stable_baselines3 import DQN
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from utils import get_test_files

log_no = 1
idv_no = 14
arch = 1

print(f'... Starting IDV {idv_no}, arch {arch} ...')

best_model_path = 'models/idv'+str(idv_no)+'/arch'+str(arch) + '/best_model_arch' + str(arch)
best_model = DQN.load(best_model_path)

x_test_w, y_test_w = get_test_files(idv=idv_no,n_test=1000,w=5)


concat_x_test = np.concatenate(x_test_w)
concat_y_test = np.concatenate(y_test_w)

print('            ')
print('Test files concatenated!')
print('              ')

print('... Starting prediction ...')
y_pred_concat, _ = best_model.predict(concat_x_test)
print('... Prediction finished ...')
print('          ')
print('... Compiling performance metrics ...')

y_pred_cut = []
chunk_size = 956
total_chunks = 1000
for i in range(total_chunks):
    start_index = i*chunk_size
    end_index = (i+1)*chunk_size

    chunk_y_pred = y_pred_concat[start_index:end_index]
    y_pred_cut.append(chunk_y_pred)
    
cm_list = []
FAR_list = []
MDR_list = []
Accuracy_list = []
ttd_normal = []
ttd_russell = []


for i in range(1000):
    y_test = y_test_w[i]
    y_pred = y_pred_cut[i]
    accuracy = accuracy_score(y_test, y_pred)
    Accuracy_list.append(accuracy)
    
    cm = confusion_matrix(y_test,y_pred)
    cm_list.append(cm)
    
    far = cm[0,1]/(cm[0,1]+cm[0,0])
    FAR_list.append(far)
    
    mdr = cm[1,0]/(cm[1,0]+cm[1,1])
    MDR_list.append(mdr)
    
    ttd_ = time_to_detection(y_pred)
    ttd_russell.append(ttd_)
    ttd = time_to_detection_normal(y_pred)
    ttd_normal.append(ttd)
    
    
mean_far = np.mean(FAR_list)
mean_mdr = np.mean(MDR_list)
mean_ttd_normal = np.mean(ttd_normal)
mean_ttd_russell = np.mean(ttd_russell)
mean_accuracy = np.mean(Accuracy_list)

print('----------------------------------')
print(f'     Performance Metrics - IDV = {idv_no}  / Arch {arch}   ') 
print('----------------------------------')

print(f'False Alarm Rate: {100*mean_far:.5f}%')
print(f'Missed Detection Rate: {100*mean_mdr:.5f}%')
print(f'Model accuracy: {100*mean_accuracy:.5f}%')
print(f'Time to detection (1 obs.): {mean_ttd_normal:.5f} obs. = {3*mean_ttd_normal:.5f} min.')
print(f'Time to detection (6 obs.): {mean_ttd_russell:.5f} obs. = {3*mean_ttd_russell:.5f} min.')



















