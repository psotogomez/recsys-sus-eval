import os
import pandas as pd
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split

SEED = 42
THRESHOLD = 1.5  # Mantener como en el script anterior
MIN_INTERACTIONS = 5
BALANCE_RATIO = 1.0
LIMIT = 50000

dataset_path = './yoochoose-dataset'
# En el caso de que no tengas los datos descomentar las dos líneas de abajo, se creara un nuevo directorio en esta misma ubicación llamado 'yoochoose-dataset' y se descargaran los datos
# subprocess.run(['curl', '-L', '-o', 'archive.zip', 'https://www.kaggle.com/api/v1/datasets/download/chadgostopp/recsys-challenge-2015'])
# subprocess.run(['unzip', 'archive.zip', '-d', dataset_path])

def read_yoochoose_data(limit=None):
    '''
    Función que lee los datos de clics y compras de YooChoose y los combina en un solo dataset.
    '''
    clicks = pd.read_csv(
        os.path.join(dataset_path, 'yoochoose-clicks.dat'),
        names=["session", "timestamp", "item", "category"],
        parse_dates=["timestamp"],
        dtype={"category": str},
        nrows=limit, 
        usecols=["session", "timestamp", "item", "category"]
    )
    clicks['rating'] = 1

    buys = pd.read_csv(
        os.path.join(dataset_path, 'yoochoose-buys.dat'),
        names=["session", "timestamp", "item", "price", "qty"],
        parse_dates=["timestamp"],
        nrows=limit, 
        usecols=["session", "timestamp", "item"]
    )
    buys['rating'] = 3 

    data = pd.concat([clicks[['session', 'item', 'rating']], buys[['session', 'item', 'rating']]])
    data = data.sort_values(by=["session", "item"]).reset_index(drop=True)
    
    return data

def transform_yoochoose_data(data, balance_ratio=1.0, balance_threshold=THRESHOLD):
    '''
    Función que transforma los datos de YooChoose en números enteros y balancea los datos.
    '''
    data_high = data[data['rating'] > balance_threshold]
    data_low = data[data['rating'] < balance_threshold]

    sample_size = int(len(data_high) * balance_ratio)
    if len(data_low) > sample_size:
        data_low = data_low.sample(n=sample_size, random_state=SEED)
    
    balanced_data = pd.concat([data_high, data_low]).reset_index(drop=True)
    
    balanced_data.rename(columns={'session': 'visitorid', 'item': 'itemid'}, inplace=True)
    
    return balanced_data[['visitorid', 'itemid', 'rating']]

def filter_users(data, min_interactions=MIN_INTERACTIONS):
    '''
    Función que elimina los usuarios
    con interacciones menores a 'min_interactions'.
    '''
    user_counts = data['visitorid'].value_counts()
    users_to_keep = user_counts[user_counts >= min_interactions].index
    return data[data['visitorid'].isin(users_to_keep)]

def split_train_test(data, test_prop=0.2):
    '''
    Función que divide los datos en conjuntos de entrenamiento y prueba, de tal manera que se garantice
    que cada usuario tenga al menos una interacción en el conjunto de entrenamiento y prueba.
    '''
    data_grouped_by_user = data.groupby('uid')
    tr_list, te_list = [], []

    np.random.seed(SEED)

    for uid, group in data_grouped_by_user:
        n_items_u = len(group)
        if n_items_u >= 5:
            n_test_items = max(1, int(test_prop * n_items_u))
            idx = np.zeros(n_items_u, dtype=bool)
            idx[np.random.choice(n_items_u, size=n_test_items, replace=False)] = True
            tr_list.append(group[~idx])
            te_list.append(group[idx])
        else:
            if n_items_u > 1:
                idx = np.zeros(n_items_u, dtype=bool)
                idx[np.random.choice(n_items_u, size=1, replace=False)] = True
                tr_list.append(group[~idx])
                te_list.append(group[idx])
            else:
                tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

raw_data = read_yoochoose_data(limit=LIMIT)  
transformed_data = transform_yoochoose_data(raw_data, balance_ratio=BALANCE_RATIO, balance_threshold=THRESHOLD)
transformed_data = filter_users(transformed_data, min_interactions=MIN_INTERACTIONS)

unique_users = transformed_data['visitorid'].unique()
user2id = {uid: i for i, uid in enumerate(unique_users)}

unique_items = transformed_data['itemid'].unique()
item2id = {iid: i for i, iid in enumerate(unique_items)}

transformed_data['uid'] = transformed_data['visitorid'].map(user2id)
transformed_data['sid'] = transformed_data['itemid'].map(item2id)
transformed_data = transformed_data[['uid', 'sid', 'rating']].dropna()

train_set, temp_set = train_test_split(transformed_data, test_size=0.2, random_state=SEED)
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=SEED)

val_data_tr, val_data_te = split_train_test(val_set)
test_data_tr, test_data_te = split_train_test(test_set)

output_dir = './processed_yoochoose'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_set.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_data_tr.to_csv(os.path.join(output_dir, 'validation_tr.csv'), index=False)
val_data_te.to_csv(os.path.join(output_dir, 'validation_te.csv'), index=False)
test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)
test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)

with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
    for item in unique_items:
        f.write(f'{item}\n')

with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
    for user in unique_users:
        f.write(f'{user}\n')

print("Preprocesamiento completado y archivos guardados en el directorio 'processed_yoochoose'.")
