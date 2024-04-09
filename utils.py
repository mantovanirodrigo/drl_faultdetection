# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:41:04 2023

@author: USUARIO
"""

def get_files(idv = 1, n_train = 6000, n_val = 2000, w = 5, standard = False):
    
    import pandas as pd
    import numpy as np

    # Defining train_files list
    train_files = []
    
    print('Loading train data')
    print('------------------------')
    for x in range(n_train):
      if (x+1) % (n_train/30) == 0:
        print(f'... {x+1}/{str(n_train)} files loaded ({int(100*(x+1)/n_train)}%) ...')
      file_name = 'data/idv' + str(idv) + '/train/train_' + str(x) + '.parquet'
      df = pd.read_parquet(file_name)
      train_files.append(df)
    print('------------------------') 
    print('                        ')
    

    # Defining val_files list
    val_files = []
    
    print('Loading val data')
    print('------------------------')
    for x in range(n_val):
      if (x+1) % (n_val/20) == 0:
        print(f'... {x+1}/{str(n_val)} files loaded ({int(100*(x+1)/n_val)}%) ...')
      file_name = 'data/idv' + str(idv) + '/val/val_' + str(x) + '.parquet'
      df = pd.read_parquet(file_name)
      val_files.append(df)
    print('------------------------')
    print('                        ')
    
    train = pd.concat(train_files)
    x_train = train.iloc[:,:-1].values
    y_train = train.iloc[:,-1].values

    val = pd.concat(val_files)
    x_val = val.iloc[:,:-1].values
    y_val = val.iloc[:,-1].values
    
    if standard:
        
        from sklearn.preprocessing import StandardScaler
        scaler_t = StandardScaler()
        scaler_t.fit(x_train)
        x_train = scaler_t.transform(x_train)
        
        scaler_v = StandardScaler()
        scaler_v.fit(x_val)
        x_val = scaler_v.transform(x_val)        
        
        
    print('Applying sliding window')
    
    x_train = np.array(list(sliding_window(x_train[:480*2000], w)))
    y_train = np.array(list(sliding_window_max(y_train[:480*2000],w)))
    x_val = np.array(list(sliding_window(x_val[:480*600],w)))
    y_val = np.array(list(sliding_window_max(y_val[:480*600],w)))
    
    # Getting min and max values of each variable

    max_values = np.max(x_train,axis=0)
    min_values = np.min(x_train,axis=0)
    
    return x_train, y_train, x_val, y_val, max_values, min_values

def get_test_files(idv = 1, n_test = 1000, w = 5, standard = False):
    
    import pandas as pd
    import numpy as np
      
    test_files = []
    for x in range(n_test):
        if (x+1) % (n_test/5) == 0:
            print(f'... Reading file #{x+1} ...')
        filename = 'data/idv' + str(idv) + '/test/test_' + str(x) + '.parquet'
        df = pd.read_parquet(filename).astype(np.float32)
        test_files.append(df)
        
    test = pd.concat(test_files).astype(np.float32)
    x_test = test.iloc[:,:-1].values.astype(np.float32)
    y_test = test.iloc[:,-1].values.astype(np.float32) 
    
    if standard:
        
        from sklearn.preprocessing import StandardScaler
        scaler_test = StandardScaler()
        scaler_test.fit(x_test)
        x_test = scaler_test.transform(x_test)
    
    x_test_cut = []
    y_test_cut = []
    chunk_size = 960
    total_chunks = n_test

    for i in range(total_chunks):
        if (i+1) % 100 == 0:
            print(f'Applying sliding window #{str(i+1)}')
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        chunk_x = x_test[start_index:end_index]
        chunk_y = y_test[start_index:end_index]
        x_test_cut.append(chunk_x)
        y_test_cut.append(chunk_y)
        
    x_test_w = []
    y_test_w = []

    for i in range(1000):
        x_w = np.array(list(sliding_window(x_test_cut[i], 5)))
        y_w = np.array(list(sliding_window_max(y_test_cut[i],5)))
        x_test_w.append(x_w)
        y_test_w.append(y_w)
        if (i+1) % 100 == 0:
            print(f'Final test transformation #{str(i+1)}')
            
    return x_test_w, y_test_w
        
    
def sliding_window(data, window_size):
    for i in range(len(data) - window_size + 1):
        yield data[i:i+window_size]

def sliding_window_max(data, window_size):
    import numpy as np
    
    for i in range(len(data) - window_size + 1):
        yield np.max(data[i:i+window_size])
        
def time_to_detection(array):
    arr = array[156:]
    for i in range(arr.shape[0]-5):
        if arr[i] == 1:
            if all(arr[j] == 1 for j in range(i, i+6)):
                return i
    return -1

def time_to_detection_normal(array):
    arr = array[156:]
    for i in range(arr.shape[0]):
        if arr[i] == 1:
            return i
    return -1

        
    