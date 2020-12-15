"""
This is an implementation of a VAE that can handle heterogeneous attribute data. See the hvae jupyter notebook for more details.
"""

from math import comb 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

m = D.LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))

m.log_prob(torch.tensor([2.]))

# GLOBAL
SEED = 42
VERBOSITY = 0

# VAE
NUM_EPOCHS_TUNE = 200 
NUM_EPOCHS = 500
HIDDEN_DIM = 64
Z_PRIOR_VAR = 0.5**2
X_POST_VAR = 0.1**2

def plot_history_vae(vae):
    history = vae.history
    num_epochs = vae.num_epochs
    
    fig, ax = plt.subplots(3, 1, figsize=(20, 20))
    
    ax[0].plot(range(1, num_epochs + 1),  history['avg_loss'], label='train epoch avg loss')
    ax[0].plot(range(1, num_epochs + 1),  history['val_avg_loss'], label='val avg loss')
    ax[0].set_title("Epoch average loss")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Average loss')
    ax[0].legend()
    
    ax[1].plot(range(1, num_epochs + 1),  history['recon_error'], label='avg recon error')
    ax[1].plot(range(1, num_epochs + 1),  history['val_recon_error'], label='avg val recon error')
    ax[1].set_title("Epoch average recon error")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Average reconstruction error')
    ax[1].legend()
    
    ax[2].plot(range(1, num_epochs + 1),  history['kl_penalty'], label='avg kl penalty')
    ax[2].plot(range(1, num_epochs + 1),  history['val_kl_penalty'], label='avg kl penalty')
    ax[2].set_title("Epoch average kl penalty")
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Average kl penalty')
    ax[2].legend()

    plt.show()
    plt.close()

    
def debug_nd_vae(vae, hist_kwargs={'density': True, 'bins':100, 'alpha': 0.9}, use_val=False):
    if use_val:
        X = vae.val_X
        X_numpy = vae.val_X_numpy
        
        y = vae.val_y
        y_numpy = vae.val_y_numpy
    else:
        X = vae.train_X
        X_numpy = vae.train_X_numpy
        
        y = vae.train_y
        y_numpy = vae.train_y_numpy
        
        
    with torch.no_grad():
        mu_z, logvar_z, sample_z, qz_x = vae.encode(X)
        mu_x, sample_x, _ = vae.decode(sample_z)

        sample_z_synthetic = vae.p_z.sample((len(y),))
        mu_x_synthetic, sample_x_synthetic, _ = vae.decode(sample_z_synthetic)
        


        
    fig, ax = plt.subplots(vae.z_dim, 1, figsize=(15, vae.z_dim*5))
    for i in range(vae.z_dim):
        if vae.z_dim == 1:
            ax.hist(sample_z.detach().numpy()[:, i],  **hist_kwargs)

            ax.set_title(f'Hist for marginal {i+1} of z')
        
        else:
            ax[i].hist(sample_z.detach().numpy()[:, i], **hist_kwargs)

            ax[i].set_title(f'Hist for marginal {i+1} of z')
        
    plt.show()
    plt.close()
    
    

    # histograms of attributes 
    fig, ax = plt.subplots(vae.x_dim, 3, figsize=(15, 4 * vae.x_dim))
    
    for i in range(vae.x_dim):
        ax[i][0].hist(X_numpy[:, i].flatten(),  **hist_kwargs)
        ax[i][0].set_title(f'Feature {vae.featnames[i]} dist orig')
        
        
        ax[i][1].hist(sample_x.detach().numpy()[:, i].flatten(),  **hist_kwargs)
        ax[i][1].set_title(f'Feature {vae.featnames[i]} dist reconstructed')
        
        ax[i][2].hist(sample_x_synthetic.detach().numpy()[:, i].flatten(),  **hist_kwargs)
        ax[i][2].set_title(f'Feature {vae.featnames[i]} dist synthetic')
        
        
    plt.show()
    plt.close()
    
    # relationships
    df_orig = pd.DataFrame(X_numpy)
    df_recon = pd.DataFrame(sample_x.detach().numpy())
    df_syn = pd.DataFrame(sample_x_synthetic.detach().numpy())
    
    print("original correlations")
    print(df_orig.corr())

    print("reconstructed correlations")
    print(df_recon.corr())
   
    
    print("synthetic correlations")
    print(df_syn.corr())
    
    df_orig.loc[:, 'class'] = y_numpy
    df_recon.loc[:, 'class'] = y_numpy
    df_syn.loc[:, 'class'] = y_numpy
    
    
    total_scatter_plots = comb(vae.x_dim, 2)
    if total_scatter_plots == 1:
        fig, ax = plt.subplots(1, 3)
        ax[0].scatter(X_numpy[:, 0].flatten(), 
                                X_numpy[:, 1].flatten(), 
                                alpha=0.5)

        ax[0].set_title(f'Scatter of features {vae.featnames[0]} and \n {vae.featnames[1]} orig')


        ax[1].scatter(sample_x.detach().numpy()[:, 0].flatten(), 
                            sample_x.detach().numpy()[:, 1].flatten(), 
                            alpha=0.5)

        ax[1].set_title(f'Scatter of features {vae.featnames[0]} and \n {vae.featnames[1]} reconstructed')

        ax[2].scatter(sample_x_synthetic.detach().numpy()[:, 0].flatten(), 
                            sample_x_synthetic.detach().numpy()[:, 1].flatten(), 
                        alpha=0.5)

        ax[2].set_title(f'Scatter of features {vae.featnames[0]} and \n {vae.featnames[1]} synthetic')
    else:
        fig, ax = plt.subplots(total_scatter_plots, 3, figsize=(15, 6 * total_scatter_plots))
        c = 0
        for i in range(vae.x_dim-1):
            for j in range(i+1, vae.x_dim): 
                ax[c][0].scatter(X_numpy[:, i].flatten(), 
                                    X_numpy[:, j].flatten(), 
                                    alpha=0.5)

                ax[c][0].set_title(f'Scatter of features {vae.featnames[i]} and \n {vae.featnames[j]} orig')


                ax[c][1].scatter(sample_x.detach().numpy()[:, i].flatten(), 
                                    sample_x.detach().numpy()[:, j].flatten(), 
                                    alpha=0.5)

                ax[c][1].set_title(f'Scatter of features {vae.featnames[i]} and \n {vae.featnames[j]} reconstructed')

                ax[c][2].scatter(sample_x_synthetic.detach().numpy()[:, i].flatten(), 
                                    sample_x_synthetic.detach().numpy()[:, j].flatten(), 
                                alpha=0.5)

                ax[c][2].set_title(f'Scatter of features {vae.featnames[i]} and \n {vae.featnames[j]} synthetic')

                c += 1

            
            
    plt.show()
    plt.tight_layout()
    plt.close()
    
    
    
    
class HVAE(nn.Module):
    def __init__(self, x_dim, z_dim, L=1, hidden_dim=64, 
                 z_prior_var=1., 
                 likelihood_partition=None,
                 likelihood_params = {'x_post_var' : 1., 'x_post_var_lognormal': 0.25**2}, 
                 optimizer=None, verbosity=0, device='cpu'):
        super().__init__()
        
        if likelihood_partition is None:
            # range for feature types
            self.likelihood_partition = {
                (0, x_dim-1): 'real'
            }
        else:
            self.likelihood_partition = likelihood_partition
            
        self.likelihood_params = likelihood_params

        p_z_loc = torch.zeros(z_dim).to(device)
        cov_diag = z_prior_var*torch.ones(z_dim).to(device)
 
        self.device = device
        
        self.p_z = D.Normal(loc=p_z_loc,
                            scale=cov_diag.sqrt())
        
        self.z_prior_var = z_prior_var

        
        # samples in stochastic estimate of reconstruction error
        self.L = L
        
        self.x_dim = x_dim
        self.z_dim = z_dim
       
        self.verbosity = verbosity
        
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim) # mean and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

    
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), 
                             lr=0.001, 
                             betas=(0.9, 0.999), 
                             eps=1e-08, 
                             weight_decay=0, 
                             amsgrad=False)
        else:
            self.optimizer = optimizer  
            
        self.train_dataset = None
        self.val_dataset = None
        
     
            
            
    def set_train_dataset(self, train_dataset, featnames=None):
        self.train_dataset = train_dataset
        self.train_X = train_dataset.data
        self.train_X_numpy = train_dataset.data.cpu().detach().numpy()
        
        self.train_y = train_dataset.targets
        self.train_y_numpy = train_dataset.targets.cpu().detach().numpy()
        
        if featnames is not None:
            self.featnames = featnames
        else:
            self.featnames = list(range(self.train_X_numpy.shape[1]))

    def set_val_dataset(self, val_dataset):
        self.val_dataset = val_dataset
        self.val_X = val_dataset.data
        self.val_X_numpy = val_dataset.data.cpu().detach().numpy()
        
        self.val_y = val_dataset.targets
        self.val_y_numpy = val_dataset.targets.cpu().detach().numpy()
    
    def encode(self, x):
        out = self.encoder(x)
        mu_z = out[:, :self.z_dim]
        logvar_z = out[:, self.z_dim:]  # decorrelated gaussian

        
        qz_x = torch.distributions.Normal(
                        loc=mu_z,
                        scale=(logvar_z/2).exp())


        sample_z = qz_x.rsample()
        return mu_z, logvar_z, sample_z, qz_x
                                       
    def decode(self, z):
        params = self.decoder(z)

        px_z = []
        samples = []
        
        for indices in self.likelihood_partition:
            data_type = self.likelihood_partition[indices]
            
            params_subset = params[:, indices[0]:indices[1] + 1]
            
            if data_type == 'real':
                cov_diag = self.likelihood_params['x_post_var']*torch.ones_like(params_subset).to(self.device)

                dist = torch.distributions.Normal(
                        loc=params_subset,
                        scale=cov_diag.sqrt())
                    
            elif data_type == 'categorical':
                dist = torch.distributions.OneHotCategorical(logits=params_subset)
            elif data_type == 'binary':
                dist = torch.distributions.Bernoulli(logits=params_subset)
            elif data_type == 'positive':
                lognormal_var = self.likelihood_params['x_post_var_lognormal']*torch.ones_like(params_subset).to(self.device)
                
                dist = torch.distributions.LogNormal(loc=params_subset, scale=lognormal_var.sqrt())
            elif data_type == 'count':
                positive_params_subset = F.softplus(params_subset)
                dist = torch.distributions.Poisson(rate=positive_params_subset)
            elif data_type == 'binomial':
                num_trials = self.likelihood_params['binomial_num_trials']
                dist = torch.distributions.Binomial(total_count=num_trials, logits=params_subset)
            elif data_type == 'ordinal':
                h = params_subset[:, 0:1]
                thetas = torch.cumsum(F.softplus(params_subset[:, 1:]), axis=1)
                                
                prob_lessthans = torch.sigmoid(thetas - h)
                probs = torch.cat((prob_lessthans, torch.ones(len(prob_lessthans), 1)), axis=1) - \
                        torch.cat((torch.zeros(len(prob_lessthans), 1), prob_lessthans), axis=1)
                                
                dist = torch.distributions.OneHotCategorical(probs=probs)
            else:
                raise NotImplementedError
                
                

            samples.append(dist.sample())
            px_z.append(dist)

        sample_x = torch.cat(samples, axis=1)
            
        return params, sample_x, px_z

    def forward(self, x):
        mu_z, logvar_z, sample_z, qz_x = self.encode(x)
        mu_x, sample_x, px_z = self.decode(sample_z)

        return mu_x, sample_x, px_z
    
    def recon_error(self, X, px_z):
        RL = None
        for indices, dist in zip(self.likelihood_partition, px_z):
            data_subset = X[:, indices[0]:indices[1] + 1]
            
            partial_RL = (-1. * dist.log_prob(data_subset))
            
            if len(partial_RL.shape) > 1:
                partial_RL = partial_RL.sum(axis=1)
            
            if RL is None:
                RL = partial_RL
            else: 
                RL = RL + partial_RL
        
        return RL
    
    def kl_penalty(self, qz_x):
        KLD = torch.distributions.kl.kl_divergence(qz_x, self.p_z)
        
        return KLD
    
    def train_on_batch(self, X):
        self.optimizer.zero_grad()
        
        mu_z, logvar_z, sample_z, qz_x = self.encode(X)
        params, sample_x, px_z = self.decode(sample_z)

        recon_error = self.recon_error(X, px_z)
        kl_penalty = self.kl_penalty(qz_x).sum(axis=1)
        
        loss = (recon_error + kl_penalty).mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().item()
      
                                    
    def train(self, num_epochs=100, batch_size=64):
        self.history = {
            'avg_loss': [],
            'recon_error': [],
            'kl_penalty': [],
            'val_avg_loss': [],
            'val_recon_error': [],
            'val_kl_penalty': []
        }
        
        self.num_epochs = num_epochs

        dataloader = DataLoader(dataset=self.train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=0)

        with trange(1, num_epochs+1) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')
                
                for batch_idx, (X, _)  in enumerate(dataloader):
                    batch_loss = self.train_on_batch(X)
            
                with torch.no_grad():
                    mu_z, logvar_z, sample_z, qz_x = self.encode(self.train_X)
                    mu_x, sample_x, px_z = self.decode(sample_z)
                    
                    recon_error = self.recon_error(self.train_X, px_z).mean().item()
                    kl_penalty = self.kl_penalty(qz_x).sum(axis=1).mean().item()
                    
                    avg_loss = recon_error + kl_penalty
                    
                    self.history['avg_loss'].append(avg_loss)
                    self.history['recon_error'].append(recon_error)
                    self.history['kl_penalty'].append(kl_penalty)

                    
                    mu_z, logvar_z, sample_z, qz_x = self.encode(self.val_X)
                    mu_x, sample_x, px_z = self.decode(sample_z)
                    
                    val_recon_error = self.recon_error(self.val_X, px_z).mean().item()
                    val_kl_penalty = self.kl_penalty(qz_x).sum(axis=1).mean().item()
                    
                    val_avg_loss = val_recon_error + val_kl_penalty

                    self.history['val_avg_loss'].append(val_avg_loss)
                    self.history['val_recon_error'].append(val_recon_error)
                    self.history['val_kl_penalty'].append(val_kl_penalty)
                   
                t.set_postfix(avg_loss=avg_loss, 
                              recon_error=recon_error,
                              kl_penalty=kl_penalty,
                              val_avg_loss=val_avg_loss,
                              val_recon_error=val_recon_error,
                              val_kl_penalty=val_kl_penalty
                             )
                


    def sample(self, S=10000):
        with torch.no_grad():
            z = self.p_z.sample((S,))
            z_numpy = z.detach().numpy()

            sample_means, samples, px_z = self.decode(z)
            sample_means = sample_means.detach().numpy()
            #samples = samples.detach().numpy()
            

        
        return samples, px_z


    def density(self, S, x_prime, debug=False):
        with torch.no_grad():
            samples, px_z = self.sample(S)
            
            if debug:
                print("x_prime", x_prime.shape, x_prime)
                print("samples", samples.shape)
                print("px_z loc ", px_z.loc.shape)

            x_prime_mod = x_prime.unsqueeze(1)
            if debug: print("x_prime_mod shape", x_prime_mod.shape)
            logprobs = px_z.log_prob(x_prime_mod)
            
            logsumexp = logprobs.exp().sum(axis=1).log() - np.log(S)
            logsumexp2 = logprobs.logsumexp(axis=1) - np.log(S)
            
            if debug:
                print("logprobs shape", logprobs.shape)
                print("logsumexp", logsumexp)
                print("logsumpexp2", logsumexp2)
                print("logsumpexp exp", logsumexp.exp())
                
            return logsumexp2