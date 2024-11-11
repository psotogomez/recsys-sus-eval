import numpy as np
import torch
from torch import optim
import random
from copy import deepcopy

from utils_VAE import get_data, ndcg, recall, precision, f1score
from model_VAE import VAE
from codecarbon import OfflineEmissionsTracker, track_emissions, EmissionsTracker

SEED = 42
# para el dataset de ecommerce
# DATASET = './processed_data'
# para el dataset de YooChoose
DATASET = './processed_yoochoose' 
HIDDEN_DIM = 600
LATENT_DIM = 200
BATCH_SIZE = 500
BETA = 0.05
GAMMA = None
LR = 5e-4
N_EPOCHS = 20
N_ENC_EPOCHS = 3
N_DEC_EPOCHS = 1
NOT_ALTERNATING = False
COUNTRY_ISO_CODE = "CHL"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, n_items, n_users, unique_sid, unique_uid = get_data(DATASET, global_indexing=True)
train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data

print(f"train_data shape: {train_data.shape}")
print(f"valid_in_data shape: {valid_in_data.shape}, valid_out_data shape: {valid_out_data.shape}")
print(f"test_in_data shape: {test_in_data.shape}, test_out_data shape: {test_out_data.shape}")

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    '''
    Función que genera batches de datos que permite crear subconjuntos de entrenamiento y prueba con opción de mezcla.
    '''
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)

class Batch:
    '''
    Clase para manejar un batch de datos; facilita el acceso a índices y ratings de entrada y salida en el dispositivo.
    '''
    def __init__(self, device, idx, data_in, data_out=None):
        '''
        Función que inicializa la clase con el dispositivo, índices y datos de entrada y salida.
        '''
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        '''
        Devuelve los índices de los datos seleccionados en el batch actual.
        '''
        return self._idx
    
    def get_idx_to_dev(self):
        '''
        Función que devuelve los índices de los datos seleccionados en el batch actual en el dispositivo.
        '''
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        '''
        Función que devuelve los ratings de entrada o salida según el valor de is_out.
        '''
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        '''
        Función que devuelve los ratings de entrada o salida según el valor de is_out en el dispositivo.
        '''
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)

def evaluate(model, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500, exclude_seen=True):
    '''
    Función que evalúa el modelo en un conjunto de datos y devuelve las métricas de evaluación.
    '''
    metrics = deepcopy(metrics)
    model.eval()

    for m in metrics:
        m['score'] = []

    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch):
        
        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)
    
        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
        
        if exclude_seen:
            ratings_pred[batch.get_ratings().toarray() > 0] = -np.inf
        
        for m in metrics:
            m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()
    
    return [x['score'] for x in metrics]

@track_emissions(country_iso_code="CHL")
def run_with_tracking(model, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
    '''
    Función que entrena el modelo en un conjunto de datos durante un número de épocas, con seguimiento de emisiones.
    '''
    run(model, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate)

def run(model, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
    '''
    Función que entrena el modelo en un conjunto de datos durante un número de épocas.
    '''
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()
                
            _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
            loss.backward()
            
            for optimizer in opts:
                optimizer.step()

model_kwargs = {
    'hidden_dim': HIDDEN_DIM,
    'latent_dim': LATENT_DIM,
    'input_dim': train_data.shape[1]
}
metrics = [{'metric': ndcg, 'k': 100}]

best_ndcg = -np.inf
train_scores, valid_scores = [], []

model = VAE(**model_kwargs).to(device)
model_best = VAE(**model_kwargs).to(device)

learning_kwargs = {
    'model': model,
    'train_data': train_data,
    'batch_size': BATCH_SIZE,
    'beta': BETA,
    'gamma': GAMMA
}

decoder_params = set(model.decoder.parameters())
encoder_params = set(model.encoder.parameters())

optimizer_encoder = optim.Adam(encoder_params, lr=LR)
optimizer_decoder = optim.Adam(decoder_params, lr=LR)

# # Inicializar el tracker
# tracker = OfflineEmissionsTracker(country_iso_code=COUNTRY_ISO_CODE)
# tracker.start()

# tracker = EmissionsTracker(
#     project_name="RecSys",
#     measure_power_secs=10,  # Intervalo de medición
#     log_level="info"
# )

# tracker.start()
# try:
for epoch in range(N_EPOCHS):
    if NOT_ALTERNATING:
        run_with_tracking(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.7, **learning_kwargs)
    else:
        run_with_tracking(opts=[optimizer_encoder], n_epochs=N_ENC_EPOCHS, dropout_rate=0.7, **learning_kwargs)
        model.update_prior()
        run_with_tracking(opts=[optimizer_decoder], n_epochs=N_DEC_EPOCHS, dropout_rate=0, **learning_kwargs)

    # Evaluación
    train_ndcg = evaluate(model, train_data, train_data, metrics, samples_perc_per_epoch=0.01, exclude_seen=False)[0]
    valid_ndcg = evaluate(model, valid_in_data, valid_out_data, metrics, samples_perc_per_epoch=1)[0]
    train_scores.append(train_ndcg)
    valid_scores.append(valid_ndcg)
    
    if valid_scores[-1] > best_ndcg:
        best_ndcg = valid_ndcg
        model_best.load_state_dict(deepcopy(model.state_dict()))
        
    print(f'Epoch {epoch+1}/{N_EPOCHS} | Valid NDCG@100: {valid_ndcg:.4f} | ' +
            f'Best Valid NDCG@100: {best_ndcg:.4f} | Train NDCG@100: {train_ndcg:.4f}')
# finally:
#     # Detener el tracker
#     tracker.stop()
    
test_metrics = [
    {'metric': ndcg, 'k': 100}, 
    {'metric': recall, 'k': 20}, 
    {'metric': recall, 'k': 50},
    {'metric': precision, 'k': 20},
    {'metric': precision, 'k': 50},
    {'metric': f1score, 'k': 20},
    {'metric': f1score, 'k': 50}
]

final_scores = evaluate(model_best, test_in_data, test_out_data, test_metrics)

for metric, score in zip(test_metrics, final_scores):
    print(f"{metric['metric'].__name__}@{metric['k']}: {score:.4f}")