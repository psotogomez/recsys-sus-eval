import numpy as np
from scipy import sparse
import pandas as pd
import os
import bottleneck as bn

def load_train_data(csv_file, n_items, n_users, global_indexing=False):
    '''
    Función que carga los datos de entrenamiento en una matriz dispersa CSR de interacciones usuario-ítem.
    '''
    tp = pd.read_csv(csv_file)
    
    n_users = n_users if global_indexing else tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data

def load_tr_te_data(csv_file_tr, csv_file_te, n_items, n_users, unique_uid=None, global_indexing=False):
    '''
    Función que carga datos de entrenamiento y prueba en matrices dispersas, ajustando los índices según el modo de indexación.
    '''
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    if global_indexing:
        start_idx = 0
        end_idx = len(unique_uid) - 1
    else:
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

def get_data(dataset, global_indexing=False):
    '''
    Carga los datos de entrenamiento, validación y prueba y los mapeos únicos de ítems y usuarios.
    '''
    unique_sid = list()
    with open(os.path.join(dataset, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    
    unique_uid = list()
    with open(os.path.join(dataset, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
            
    n_items = len(unique_sid)
    n_users = len(unique_uid)
    
    train_data = load_train_data(os.path.join(dataset, 'train.csv'), n_items, n_users, global_indexing=global_indexing)

    vad_data_tr, vad_data_te = load_tr_te_data(
        os.path.join(dataset, 'validation_tr.csv'),
        os.path.join(dataset, 'validation_te.csv'),
        n_items, n_users, unique_uid=unique_uid, global_indexing=global_indexing)

    test_data_tr, test_data_te = load_tr_te_data(
        os.path.join(dataset, 'test_tr.csv'),
        os.path.join(dataset, 'test_te.csv'),
        n_items, n_users, unique_uid=unique_uid, global_indexing=global_indexing)
    
    data = train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
    data = [x.astype('float32') for x in data]
    
    return data, n_items, n_users, unique_sid, unique_uid

def ndcg(X_pred, heldout_batch, k=100):
    '''
    Calcula el NDCG@k, una métrica que mide la relevancia de las recomendaciones usando ganancia acumulada.
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() if n > 0 else 1 
                    for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def recall(X_pred, heldout_batch, k=100):
    '''
    Calcula el recall@k, una métrica que mide la proporción de ítems relevantes encontrados en las recomendaciones.
    '''
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)

    recall = tmp / np.maximum(1, np.minimum(k, X_true_binary.sum(axis=1)))
    return recall

def precision(X_pred, heldout_batch, k=100):
    '''
    Calcula el precision@k, que mide la proporción de ítems recomendados que son relevantes en las primeras k posiciones.
    '''
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    
    precision = tmp / np.maximum(1, k)
    return precision

def f1score(X_pred, heldout_batch, k=100):
    '''
    Calcula el F1-score@k, combinando precisión y recall para evaluar el rendimiento general de las recomendaciones.
    '''
    recall_at_k = recall(X_pred, heldout_batch, k)
    precision_at_k = precision(X_pred, heldout_batch, k)
    
    f1 = 2 * (precision_at_k * recall_at_k) / np.maximum(1e-8, precision_at_k + recall_at_k)
    return f1