import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
THRESHOLD = 1.5
MIN_INTERACTIONS = 5
BALANCE_RATIO = 1.0

dataset_path = './ecommerce-dataset'
# En el caso de que no tengas los datos descomentar las dos líneas de abajo, se creara un nuevo directorio en esta misma ubicación llamado 'ecommerce-dataset' y se descargaran los datos
# subprocess.run(['curl', '-L', '-o', 'archive.zip', 'https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset'])
# subprocess.run(['unzip', 'archive.zip', '-d', dataset_path])

raw_data = pd.read_csv(os.path.join(dataset_path, 'events.csv'))

def TransfromData(events, balance_ratio=1.0, balance_threshold=THRESHOLD):
    '''
    Función que transforma los eventos en números enteros y balancea los datos.
    '''
    data = events[['visitorid', 'event', 'itemid']].copy()
    event_to_rating = {'view': 1, 'addtocart': 2, 'transaction': 3}
    data['rating'] = data['event'].map(event_to_rating)
    
    data_high = data[data['rating'] > balance_threshold]
    data_low = data[data['rating'] < balance_threshold]
    
    sample_size = int(len(data_high) * balance_ratio)
    if len(data_low) > sample_size:
        data_low = data_low.sample(n=sample_size, random_state=SEED)
    
    balanced_data = pd.concat([data_high, data_low]).reset_index(drop=True)
    
    return balanced_data[['visitorid', 'itemid', 'rating']]

def RedundantData_DropDuplicatesFeature(data):
    '''
    Función que elimina duplicados de las columnas 'visitorid', 'itemid' y 'rating'.
    '''
    return data.drop_duplicates(subset=['visitorid', 'itemid', 'rating'])

def RedundantData_RemoveUsersWithFewInteractions(data, min_interactions=MIN_INTERACTIONS):
    '''
    Función que elimina los usuarios con interacciones menores a 'min_interactions'.
    '''
    user_counts = data['visitorid'].value_counts()
    users_to_keep = user_counts[user_counts >= min_interactions].index
    return data[data['visitorid'].isin(users_to_keep)]

def split_train_test_proportion(data, test_prop=0.2):
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
            # Nos aseguramos de que cada usuario tenga al menos una interacción en el conjunto de prueba
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

transformed_data = TransfromData(raw_data, balance_ratio=BALANCE_RATIO, balance_threshold=THRESHOLD)

# Correr línea siguiente si se desea eliminar las interacciones con rating menor a THRESHOLD
# transformed_data = transformed_data[transformed_data['rating'] > THRESHOLD]

transformed_data = RedundantData_DropDuplicatesFeature(transformed_data)
transformed_data = RedundantData_RemoveUsersWithFewInteractions(transformed_data, min_interactions=MIN_INTERACTIONS)

transformed_data.rename(columns={'visitorid': 'userID', 'itemid': 'itemID'}, inplace=True)

train_set, temp_set = train_test_split(
    transformed_data,
    test_size=0.2,
    random_state=SEED
)

val_set, test_set = train_test_split(
    temp_set,
    test_size=0.5,
    random_state=SEED
)

# Creamos un mapeo de los IDs únicos
unique_train_users = train_set['userID'].unique()
user2id = {uid: i for i, uid in enumerate(unique_train_users)}

unique_train_items = train_set['itemID'].unique()
item2id = {iid: i for i, iid in enumerate(unique_train_items)}

# Filtramos los conjuntos de validación y prueba para que solo contengan usuarios e items presentes en el conjunto de entrenamiento
val_set = val_set[val_set['userID'].isin(user2id.keys())]
val_set = val_set[val_set['itemID'].isin(item2id.keys())]

test_set = test_set[test_set['userID'].isin(user2id.keys())]
test_set = test_set[test_set['itemID'].isin(item2id.keys())]

# Numerizamos los conjuntos de datos
def numerize(tp):
    uid = tp['userID'].map(user2id)
    sid = tp['itemID'].map(item2id)
    tp = tp.copy()
    tp['uid'] = uid
    tp['sid'] = sid
    return tp[['uid', 'sid', 'rating']]

train_data = numerize(train_set)
val_data = numerize(val_set)
test_data = numerize(test_set)

val_data = val_data.dropna()
test_data = test_data.dropna()

# Convertimos los IDs a enteros
val_data['uid'] = val_data['uid'].astype(int)
val_data['sid'] = val_data['sid'].astype(int)
test_data['uid'] = test_data['uid'].astype(int)
test_data['sid'] = test_data['sid'].astype(int)

# Verificación de resultados
print(f"train_data shape: {train_data.shape}")
print(f"val_data shape: {val_data.shape}")
print(f"test_data shape: {test_data.shape}")

# Dividimos los conjuntos de validación y de prueba en dos partes
val_data_tr, val_data_te = split_train_test_proportion(val_data)
test_data_tr, test_data_te = split_train_test_proportion(test_data)

# Directorio en que se guardaran los datos procesados
output_dir = './processed_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_data_tr.to_csv(os.path.join(output_dir, 'validation_tr.csv'), index=False)
val_data_te.to_csv(os.path.join(output_dir, 'validation_te.csv'), index=False)
test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)
test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)

with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
    for item in unique_train_items:
        f.write(f'{item}\n')

with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
    for user in unique_train_users:
        f.write(f'{user}\n')

print("Preprocesamiento completado y archivos guardados en el directorio 'processed_data'.")