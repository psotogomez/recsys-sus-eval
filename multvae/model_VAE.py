import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    '''
    Función de activación Swish: devuelve x * sigmoid(x).
    '''
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    '''
    Calcula el logaritmo de la función de densidad de probabilidad normal para x dado mu y logvar.
    '''
    var = torch.exp(logvar) + 1e-8  
    log_scale = 0.5 * torch.log(2 * np.pi * var)
    return -0.5 * ((x - mu) ** 2) / var - log_scale


class CompositePrior(nn.Module):
    '''
    Prior compuesto para el modelo VAE que utiliza una mezcla de distribuciones Gaussianas para el espacio latente.
    '''
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        '''
        Función que inicializa el prior con pesos para cada componente de la mezcla y parámetros de Gaussianas.
        '''
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        '''
        Función que calcula el logaritmo de la densidad de la mezcla de prior, combinando múltiples Gaussianas.
        '''
        post_mu, post_logvar = self.encoder_old(x, 0)
        post_logvar = torch.clamp(post_logvar, min=-10, max=10)  
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    '''
    Codificador para el VAE que transforma las entradas en representaciones latentes.
    '''
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        '''
        Función que inicializa la arquitectura del codificador con capas de normalización y redes neuronales.
        '''
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        '''
        Función que genera los parámetros latentes (mu y logvar) a partir de las entradas aplicando Swish y normalización.
        '''
        norm = x.pow(2).sum(dim=-1).sqrt() + 1e-6
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        mu = self.fc_mu(h5)
        logvar = self.fc_logvar(h5)
        logvar = torch.clamp(logvar, min=-6, max=6)  #-10 a 10
        return mu, logvar
    

class VAE(nn.Module):
    '''
    Modelo de Autoencoder Variacional (VAE) para recomendación de ítems.
    '''
    def __init__(self, hidden_dim, latent_dim, input_dim):
        '''
        Función que inicializa el modelo VAE con un codificador, un prior y un decodificador.
        '''
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        '''
        Función que muestrea el espacio latente a partir de mu y logvar.
        '''
        if self.training:
            std = torch.exp(0.5 * logvar)
            std = torch.clamp(std, min=1e-8) 
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        '''
        Función que realiza la inferencia y generación de ítems a partir de las interacciones de usuario.
        '''
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        
        if calculate_loss:
            if gamma is not None:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta is not None:
                kl_weight = beta
            else:
                kl_weight = 1.0  
            
            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            prior_log_prob = self.prior(user_ratings, z)
            posterior_log_prob = log_norm_pdf(z, mu, logvar)
            kld = (posterior_log_prob - prior_log_prob).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)
            
            return (mll, kld), negative_elbo
                
        else:
            return x_pred

    def update_prior(self):
        '''
        Función que actualiza el prior con los parámetros del codificador.
        '''
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
