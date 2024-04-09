# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:30:26 2023

@author: USUARIO """

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from utils import get_files
from scipy.stats import chi2
from utils import time_to_detection, time_to_detection_normal
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

idv_no = 10


print(f'... Collecting PCA results for IDV {idv_no} ...')
def T2(scores, model):
    import numpy as np
    eigenvalues = model.explained_variance_
    inv_eigenvalues = 1.0 / eigenvalues
    T2 = np.sum((scores**2)*inv_eigenvalues, axis = 1)
    return T2

def SPE(original_data, pca_data, model):
    reconstructed_data = model.inverse_transform(pca_data)
    SPE = ((original_data - reconstructed_data)**2).sum(axis=1)
    return SPE


train_files = []

for i in range(3000):
    
    if (i+1)%100 == 0:
        print(f'Train file #{i+1}')
    filename = 'data/idv0/train/train_' + str(i) + '.parquet'
    df = pd.read_parquet(filename)
    train_files.append(df)
    
test_files = []

for i in range(1000):
    if (i+1)%100 == 0:
        print(f'Test file #{i+1}')
    filename = 'data/idv' + str(idv_no) + '/test/test_' + str(i) + '.parquet'
    df = pd.read_parquet(filename)
    test_files.append(df)

# Training model for no fault

scaler = StandardScaler()
train_concat = pd.concat(train_files).values
X = scaler.fit_transform(train_concat)

pca = PCA(0.8)
pca_scores = pca.fit_transform(X)
print(f'Total number of PCs: {pca.components_.shape[0]}')


print('                    ')
print('-----------------------')
print('Normal PCA model was built. ')
print('-----------------------')
print('                  ')

print('... Obtaining thresholds for T2 and SPE...')
T2_normal = T2(pca_scores, pca)
SPE_normal = SPE(X, pca_scores, pca)

T2_threshold = np.percentile(T2_normal, 99)
SPE_threshold = np.percentile(SPE_normal, 99)

print('... Thresholds acquired! ...')
    
print(f'Collecting PCA model for IDV {idv_no}')
    
X_test_concat = pd.concat(test_files).values[:,:52]
X_test = scaler.transform(X_test_concat)
pca_scores_test = pca.transform(X_test)

#T2 test
T2_test = T2(pca_scores_test, pca)
T2_outliers = (T2_test > T2_threshold).astype(int)
print(f'                                ')
print(f'Finished T2 fault detection for IDV {idv_no}')


#SPE test
SPE_test = SPE(X_test, pca_scores_test, pca)
SPE_outliers = (SPE_test > SPE_threshold).astype(int)
print(f'                                ')
print(f'Finished SPE fault detection for IDV {idv_no}')

#Calculate metrics:
    
y_pred_cut_T2 = []
y_pred_cut_SPE = []
chunk_size = 960
total_chunks = 1000

for i in range(total_chunks):
    start_index = i*chunk_size
    end_index = (i+1)*chunk_size

    chunk_t2 = T2_outliers[start_index:end_index]
    chunk_spe = SPE_outliers[start_index:end_index]
    y_pred_cut_T2.append(chunk_t2)
    y_pred_cut_SPE.append(chunk_spe)

cm_list_T2 = []
FAR_list_T2 = []
MDR_list_T2 = []
Accuracy_list_T2 = []
ttd_normal_T2 = []
ttd_russell_T2 = []

cm_list_SPE = []
FAR_list_SPE = []
MDR_list_SPE = []
Accuracy_list_SPE = []
ttd_normal_SPE = []
ttd_russell_SPE = []


for i in range(1000):
    y_test = np.zeros(960)
    y_test[160:] = 1
    y_pred_t2 = y_pred_cut_T2[i]
    y_pred_spe = y_pred_cut_SPE[i]
    accuracy_t2 = accuracy_score(y_test, y_pred_t2)
    accuracy_spe = accuracy_score(y_test, y_pred_spe)
    Accuracy_list_T2.append(accuracy_t2)
    Accuracy_list_SPE.append(accuracy_spe)
    
    cm_t2 = confusion_matrix(y_test,y_pred_t2)
    cm_list_T2.append(cm_t2)
    cm_spe = confusion_matrix(y_test,y_pred_spe)
    cm_list_SPE.append(cm_spe)
    
    far_t2 = cm_t2[0,1]/(cm_t2[0,1]+cm_t2[0,0])
    FAR_list_T2.append(far_t2)
    far_spe = cm_spe[0,1]/(cm_spe[0,1]+cm_spe[0,0])
    FAR_list_SPE.append(far_spe)    
    
    mdr_t2 = cm_t2[1,0]/(cm_t2[1,0]+cm_t2[1,1])
    MDR_list_T2.append(mdr_t2)
    mdr_spe = cm_spe[1,0]/(cm_spe[1,0]+cm_spe[1,1])
    MDR_list_SPE.append(mdr_spe)
    
    ttd_t2_ = time_to_detection(y_pred_t2)
    ttd_russell_T2.append(ttd_t2_)
    ttd_t2 = time_to_detection_normal(y_pred_t2)
    ttd_normal_T2.append(ttd_t2)
    
    ttd_spe_ = time_to_detection(y_pred_spe)
    ttd_russell_SPE.append(ttd_spe_)
    ttd_spe = time_to_detection_normal(y_pred_spe)
    ttd_normal_SPE.append(ttd_spe)
    
    
mean_far_T2 = np.mean(FAR_list_T2)
mean_mdr_T2 = np.mean(MDR_list_T2)
mean_ttd_normal_T2 = np.mean(ttd_normal_T2)
mean_ttd_russell_T2 = np.mean(ttd_russell_T2)
mean_accuracy_T2 = np.mean(Accuracy_list_T2)

mean_far_SPE = np.mean(FAR_list_SPE)
mean_mdr_SPE = np.mean(MDR_list_SPE)
mean_ttd_normal_SPE = np.mean(ttd_normal_SPE)
mean_ttd_russell_SPE = np.mean(ttd_russell_SPE)
mean_accuracy_SPE = np.mean(Accuracy_list_SPE)

print(f'                ')
print(f'----------------------')
print(f'ROLLOUT IDV {idv_no}')
print(f'----------------------')
print(f'                ')
print(f'Peformance metrics [IDV = {idv_no}, T2 statistic]')
print(f'FAR: {100*mean_far_T2:.2f}%')
print(f'MDR: {100*mean_mdr_T2:.2f}%')
print(f'TTD (1 obs.): {mean_ttd_normal_T2:.2f} obs. ({3*mean_ttd_normal_T2:.2f} min.):')
print(f'TTD (6 obs.): {mean_ttd_russell_T2:.2f} obs. ({3*mean_ttd_russell_T2:.2f} min.):')
print(f'Accuracy: {100*mean_accuracy_T2:.2f}%')
print(f'----------------------')
print(f'                ')
print(f'Peformance metrics [IDV = {idv_no}, Q statistic]')
print(f'FAR: {100*mean_far_SPE:.2f}%')
print(f'MDR: {100*mean_mdr_SPE:.2f}%')
print(f'TTD (1 obs.): {mean_ttd_normal_SPE:.2f} obs. ({3*mean_ttd_normal_SPE:.2f} min.):')
print(f'TTD (6 obs.): {mean_ttd_russell_SPE:.2f} obs. ({3*mean_ttd_russell_SPE:.2f} min.):')
print(f'Accuracy: {100*mean_accuracy_SPE:.2f}%')


loaded_df = pd.read_excel('results_pca.xlsx')

data = {'IDV':[idv_no],
        'FAR (T2)': [mean_far_T2],
        'MDR (T2)': [mean_mdr_T2],
        'TTD - 1 obs. (T2)': [mean_ttd_normal_T2],
        'TTD - 6 obs. (T2)': [mean_ttd_russell_T2],
        'Accuracy (T2)': [mean_accuracy_T2],
        'FAR (SPE)': [mean_far_SPE],
        'MDR (SPE)': [mean_mdr_SPE],
        'TTD - 1 obs. (SPE)': [mean_ttd_normal_SPE],
        'TTD - 6 obs. (SPE)': [mean_ttd_russell_SPE],
        'Accuracy (SPE)': [mean_accuracy_SPE]
        }

dataframe = pd.DataFrame(data)
dataframe_export = pd.concat([loaded_df, dataframe],axis=0)
dataframe_export.to_excel('results_pca.xlsx')