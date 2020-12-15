"""
This is an implementation of a CSVAE that can handle heterogeneous attribute data. See the hcsvae jupyter notebook for more details.
"""

from itertools import chain

from math import comb 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import trange



KL = torch.distributions.kl.kl_divergence

# GLOBAL
SEED = 42
VERBOSITY = 0


# CSVAE
CSVAE_NUM_EPOCHS_TUNE = 200
CSVAE_NUM_EPOCHS = 1000
Y_DIM = 2
W_DIM = 1
CSVAE_Z_DIM = 3
CSVAE_Z_PRIOR_VAR = 0.5**2
CSVAE_X_POST_VAR = 0.1**2
CSVAE_PW_Y_MEANS =  torch.tensor([[-1.], 
                                          [1.]])
CSVAE_PW_Y_VARS = torch.tensor([[0.5**2],
                                        [0.5**2]])
CSVAE_BETA1_TUNE = 10
CSVAE_BETA2_TUNE = 10
CSVAE_BETA3_TUNE = 5
CSVAE_BETA4_TUNE = 100
CSVAE_BETA5_TUNE = 10

CSVAE_BETA1 = 5 
CSVAE_BETA2 = 15 
CSVAE_BETA3 = 10 
CSVAE_BETA4 = 100
CSVAE_BETA5 = 10

BAD_OUTCOME = '#ff8c69'
GOOD_OUTCOME = '#40E0D0'
COUNTERFACTUAL = '#84218b'
FACTUAL = '#558f14'    # old: '#7bde74'
F_TO_CF_CMAP = 'PRGn_r'

def plot_history_csvae(csvae):
    history = csvae.history
    num_epochs = csvae.num_epochs
    
    
    fig, ax = plt.subplots(4, 2, figsize=(20, 20))
    
    ax[0][0].plot(range(1, num_epochs + 1),  history['avg_loss'], label='train epoch avg loss')
    ax[0][0].plot(range(1, num_epochs + 1),  history['val_avg_loss'], label='val avg loss')
    ax[0][0].set_title("Epoch average loss")
    ax[0][0].set_xlabel('Epoch')
    ax[0][0].set_ylabel('Average loss over batches')
    ax[0][0].legend()
    
    ax[0][1].plot(range(1, num_epochs + 1),  history['term1'], label='train term1')
    ax[0][1].plot(range(1, num_epochs + 1),  history['val_term1'], label='val term1')
    ax[0][1].set_title("Reconstruction error on entire data (term1)")
    ax[0][1].set_xlabel('Epoch')
    ax[0][1].set_ylabel('Reconstruction error')
    ax[0][1].legend()
    
            
    ax[1][0].plot(range(1, num_epochs + 1),  history['term2'], label='train term2')
    ax[1][0].plot(range(1, num_epochs + 1),  history['val_term2'], label='val term2')
    ax[1][0].set_title("KL between w_xy and w_y (term2)")
    ax[1][0].set_xlabel('Epoch')
    ax[1][0].set_ylabel('KL')
    ax[1][0].legend()
    

    ax[1][1].plot(range(1, num_epochs + 1),  history['term3'], label='train term3')
    ax[1][1].plot(range(1, num_epochs + 1),  history['val_term3'], label='val term3')
    ax[1][1].set_title("KL between z_x and z (term3)")
    ax[1][1].set_xlabel('Epoch')
    ax[1][1].set_ylabel('KL')       
    ax[1][1].legend()
        
    ax[2][0].plot(range(1, num_epochs + 1),  history['term4'], label='train term4')
    ax[2][0].plot(range(1, num_epochs + 1),  history['val_term4'], label='val term4')        
    ax[2][0].set_title("Negative conditional entropy of y given z (term4)")
    ax[2][0].set_xlabel('Epoch')
    ax[2][0].set_ylabel('Negative conditional entropy of y given z')    
    ax[2][0].legend()
    
    ax[2][1].plot(range(1, num_epochs + 1),  history['term5'], label='train term5')
    ax[2][1].plot(range(1, num_epochs + 1),  history['val_term5'], label='val term5')
    ax[2][1].set_title("Negative Approximate posterior likelihood y given z  (term5)")
    ax[2][1].set_xlabel('Epoch')
    ax[2][1].set_ylabel('Negative Approximate posterior likelihood y given z')       
    ax[2][1].legend()
    
    ax[3][0].plot(range(1, num_epochs + 1),  history['avg_y_pred_max'], label='train epoch avg y pred max')
    ax[3][0].plot(range(1, num_epochs + 1),  history['val_avg_y_pred_max'], label='val avg y pred max')
    ax[3][0].set_title("Average maximum y pred over batches by epoch")
    ax[3][0].set_xlabel('Epoch')
    ax[3][0].set_ylabel('Average maximum y pred over batches')   
    ax[3][0].legend()
    
    ax[3][1].plot(range(1, num_epochs + 1),  history['avg_y_pred_min'], label='epoch avg y pred min')
    ax[3][1].plot(range(1, num_epochs + 1),  history['val_avg_y_pred_min'], label='val avg y pred min')
    ax[3][1].set_title("Average minimum y pred over batches by epoch")
    ax[3][1].set_xlabel('Epoch')
    ax[3][1].set_ylabel('Average minimum y pred over batches')   
    ax[3][1].legend()
    
    plt.show()
    plt.close()

    
def debug_nd_csvae(csvae, hist_kwargs={'density': True, 'bins':100, 'alpha': 0.9}, use_val=False, do_scatterplots=True):
        
        if use_val:
            X = csvae.val_X
            X_numpy = csvae.val_X_numpy
            
            y = csvae.val_y
            y_numpy = csvae.val_y_numpy
        else:
            X = csvae.train_X
            X_numpy = csvae.train_X_numpy
            
            y = csvae.train_y
            y_numpy = csvae.train_y_numpy
            
        with torch.no_grad():
            mu_z, logvar_z, sample_z, qz_x = csvae.encode_z(X)

            mu_w, logvar_w, sample_w, qw_xy = csvae.encode_w(X, 
                                                            y.unsqueeze(-1).float())

            mu_x, sample_x, px_wz = csvae.decode_x(sample_w, sample_z)

            # sample z
            sample_z_synthetic = csvae.p_z.sample((len(y),))

            # sample w given y
            sample_w_y_synthetic = csvae.get_pw_y(y).sample()

            # decode
            _, sample_x_synthetic, _ = csvae.decode_x(sample_w_y_synthetic, sample_z_synthetic)
                

        

            
        fig, ax = plt.subplots(csvae.z_dim, 1, figsize=(15, csvae.z_dim*5))
        for i in range(csvae.z_dim):
            if csvae.z_dim == 1:
                ax.hist(sample_z.detach().numpy()[y_numpy == 0, i], label='y=0', color=BAD_OUTCOME,  **hist_kwargs)
                ax.hist(sample_z.detach().numpy()[y_numpy == 1, i], label='y=1', color=GOOD_OUTCOME, **hist_kwargs)

                ax.set_title(f'Hist for marginal {i+1} of z')
                ax.legend()
            
            else:
                ax[i].hist(sample_z.detach().numpy()[y_numpy == 0, i], label='y=0', color=BAD_OUTCOME, **hist_kwargs)
                ax[i].hist(sample_z.detach().numpy()[y_numpy == 1, i], label='y=1', color=GOOD_OUTCOME, **hist_kwargs)

                ax[i].set_title(f'Hist for marginal {i+1} of z')
                ax[i].legend()
            
        if not use_val:
            filename = 'csvae_z_marginals.png'
        else:
            filename = 'csvae_z_marginals_val.png'

        plt.show()
        plt.close()
        
        
        # assumes 1d w
        fig, ax = plt.subplots(figsize=(15, 5))
    
        ax.hist(sample_w.detach().numpy()[y_numpy.flatten() == 0],  color=BAD_OUTCOME, label='y=0',  **hist_kwargs)
        ax.hist(sample_w.detach().numpy()[y_numpy.flatten() == 1],  color=GOOD_OUTCOME, label='y=1',  **hist_kwargs)
        ax.set_title("encoded w + noise")
        ax.legend()


        if not use_val:
            filename = 'csvae_w.png'
        else:
            filename = 'csvae_w_val.png'

        plt.show()
        plt.close()

        # histograms of attributes 
        fig, ax = plt.subplots(csvae.x_dim, 3, figsize=(15, 4 * csvae.x_dim))
        
        for i in range(csvae.x_dim):
            ax[i][0].hist(X_numpy[y_numpy == 0, i].flatten(), color=BAD_OUTCOME, label='y=0',  **hist_kwargs)
            ax[i][0].hist(X_numpy[y_numpy == 1, i].flatten(), color=GOOD_OUTCOME, label='y=1',  **hist_kwargs)
            ax[i][0].set_title(f'Feature {csvae.featnames[i]} dist orig')
            ax[i][0].legend()
            
            
            ax[i][1].hist(sample_x.detach().numpy()[y_numpy == 0, i].flatten(), color=BAD_OUTCOME, label='y=0',  **hist_kwargs)
            ax[i][1].hist(sample_x.detach().numpy()[y_numpy == 1, i].flatten(), color=GOOD_OUTCOME, label='y=1',  **hist_kwargs)
            ax[i][1].set_title(f'Feature {csvae.featnames[i]} dist reconstructed')
            ax[i][1].legend()
            
            ax[i][2].hist(sample_x_synthetic.detach().numpy()[y_numpy == 0, i].flatten(), color=BAD_OUTCOME, label='y=0',  **hist_kwargs)
            ax[i][2].hist(sample_x_synthetic.detach().numpy()[y_numpy == 1, i].flatten(), color=GOOD_OUTCOME, label='y=1',  **hist_kwargs)
            ax[i][2].set_title(f'Feature {csvae.featnames[i]} dist synthetic')
            ax[i][2].legend()
            
            
        if not use_val:
            filename = 'csvae_debug_nd_marginals.png'
        else:
            filename = 'csvae_debug_nd_marginals_val.png'

        plt.show()
        plt.close()
        
        # relationships
        df_orig = pd.DataFrame(X_numpy)
        df_recon = pd.DataFrame(sample_x.detach().numpy())
        df_syn = pd.DataFrame(sample_x_synthetic.detach().numpy())
        
        print("original correlations")
        print(str(df_orig.corr()))

        print("reconstructed correlations")
        print(str(df_recon.corr()))
    
        
        print("synthetic correlations")
        print((str(df_syn.corr())))
        
        df_orig.loc[:, 'class'] = y_numpy
        df_recon.loc[:, 'class'] = y_numpy
        df_syn.loc[:, 'class'] = y_numpy
        
        if do_scatterplots:
            total_scatter_plots = comb(csvae.x_dim, 2)
            fig, ax = plt.subplots(total_scatter_plots, 3, figsize=(15, 6 * total_scatter_plots))
            c = 0
            for i in range(csvae.x_dim-1):
                for j in range(i+1, csvae.x_dim): 
                    try: 
                        ax1 = ax[c][0]
                        ax2 = ax[c][1]
                        ax3 = ax[c][2]
                    except TypeError:
                        ax1 = ax[0]
                        ax2 = ax[1]
                        ax3 = ax[2]

                    ax1.scatter(X_numpy[y_numpy == 0, i].flatten(), 
                                     X_numpy[y_numpy == 0, j].flatten(), 
                                     color=BAD_OUTCOME, label='y=0', alpha=0.5)

                    ax1.scatter(X_numpy[y_numpy == 1, i].flatten(),
                                     X_numpy[y_numpy == 1, j].flatten(),
                                     color=GOOD_OUTCOME, label='y=1', alpha=0.5)

                    ax1.set_title(f'Scatter of features {csvae.featnames[i]} and \n {csvae.featnames[j]} orig')
                    ax1.legend()


                    ax2.scatter(sample_x.detach().numpy()[y_numpy == 0, i].flatten(), 
                                     sample_x.detach().numpy()[y_numpy == 0, j].flatten(), 
                                     color=BAD_OUTCOME, label='y=0', alpha=0.5)

                    ax2.scatter(sample_x.detach().numpy()[y_numpy == 1, i].flatten(),
                                     sample_x.detach().numpy()[y_numpy == 1, j].flatten(),
                                     color=GOOD_OUTCOME, label='y=1', alpha=0.5)
                    ax2.set_title(f'Scatter of features {csvae.featnames[i]} and \n {csvae.featnames[j]} reconstructed')
                    ax2.legend()

                    ax3.scatter(sample_x_synthetic.detach().numpy()[y_numpy == 0, i].flatten(), 
                                     sample_x_synthetic.detach().numpy()[y_numpy == 0, j].flatten(), 
                                     color=BAD_OUTCOME, label='y=0', alpha=0.5)

                    ax3.scatter(sample_x_synthetic.detach().numpy()[y_numpy == 1, i].flatten(),
                                     sample_x_synthetic.detach().numpy()[y_numpy == 1, j].flatten(),
                                     color=GOOD_OUTCOME, label='y=1', alpha=0.5)

                    ax3.set_title(f'Scatter of features {csvae.featnames[i]} and \n {csvae.featnames[j]} synthetic')
                    ax3.legend()

                    c += 1
            
                


        plt.show()
        plt.tight_layout()
        plt.close()
        
        
def debug_class_change_csvae(csvae, x_orig, y_orig, w_range=None, clf=None, do_scatterplots=True):
    """
    evolution of marginals
    evolution of joint
    """

    with torch.no_grad():
        if w_range is None:
            w_range = torch.from_numpy(np.linspace(-2., 2., 100).reshape(-1, 1).astype('float32'))
            w_range_numpy = w_range.detach().numpy()

        # get its latent representation z
        mu_z, logvar_z, sample_z, _ = csvae.encode_z(x_orig)
        
        mu_w, logvar_w, sample_w, qw_xy = csvae.encode_w(x_orig, 
                                                        torch.from_numpy(y_orig).unsqueeze(-1).float())


        # decode it 
        z = torch.cat(len(w_range_numpy)*[sample_z.detach()])

        mu_x, sample_x, _ = csvae.decode_x(w_range, z)
        
        mu_x_numpy = mu_x.detach().numpy()
        sample_x = sample_x.detach().numpy()
        
        
    if clf is not None:
        fig, ax = plt.subplots()
        
        prob_y1 = clf.predict_proba(mu_x_numpy)[:, 1].flatten()
        
        ax.scatter(w_range_numpy.flatten(),
                    prob_y1,
                    c=w_range_numpy.flatten(),
                    cmap=F_TO_CF_CMAP
                    )
        
        ax.axhline(y=0.5)
        
        ax.set_xlabel("ws")
        ax.set_ylabel('classifier p(y=1)')
        ax.set_title("Evolution of classifier probability by w")
        
        
        plt.show()
        plt.close()
        
        
    fig, ax = plt.subplots(csvae.x_dim, 1, figsize=(15, 5*csvae.x_dim))
    for i in range(csvae.x_dim):

        
        ax[i].scatter(-1.5 + 0.05*np.random.randn((csvae.train_y_numpy == 0).sum()), 
                        csvae.train_X_numpy[csvae.train_y_numpy == 0, i], c=BAD_OUTCOME, alpha=0.1, s=5)
        ax[i].axhline(y=csvae.train_X_numpy[csvae.train_y_numpy == 0, i].mean(), c=FACTUAL, label='y=0 mean')            
        
        ax[i].scatter(1.5 + 0.05*np.random.randn((csvae.train_y_numpy == 1).sum()), 
                        csvae.train_X_numpy[csvae.train_y_numpy == 1, i], c=GOOD_OUTCOME, alpha=0.1, s=5)
        ax[i].axhline(y=csvae.train_X_numpy[csvae.train_y_numpy == 1, i].mean(), c=COUNTERFACTUAL, label='y=1 mean')
        
        ax[i].axvline(x=mu_w.item(), c='black', label='w orig')
        
        ax[i].scatter(w_range_numpy.flatten(), mu_x_numpy[:, i].flatten(), c=w_range_numpy.flatten(), cmap=F_TO_CF_CMAP)
        
        ax[i].set_xlabel('ws')
        ax[i].set_ylabel(f'feature {csvae.featnames[i]} values')
        ax[i].set_title(f"Interpolated marginal feature values for feature {csvae.featnames[i]}")
        ax[i].legend()

    plt.show()
    plt.close()


    if do_scatterplots:
        total_scatter_plots = comb(csvae.x_dim, 2)
        fig, ax = plt.subplots(total_scatter_plots, 1, figsize=(15, 7 * total_scatter_plots))



        c = 0
        for i in range(csvae.x_dim-1):
            for j in range(i+1, csvae.x_dim):         
                try:
                    ax_ = ax[c]
                except TypeError:
                    ax_ = ax

                divider = make_axes_locatable(ax_)
                cax = divider.append_axes('right', size='5%', pad=0.05)

                ax_.scatter(csvae.train_X_numpy[csvae.train_y_numpy == 0, i].mean(),
                                csvae.train_X_numpy[csvae.train_y_numpy == 0, j].mean(),
                                c=BAD_OUTCOME, label='y=0 mean', s=200)

                ax_.scatter(csvae.train_X_numpy[csvae.train_y_numpy == 1, i].mean(),
                                csvae.train_X_numpy[csvae.train_y_numpy == 1, j].mean(),
                                c=GOOD_OUTCOME, label='y=1 mean', s=200)   

                ax_.scatter(csvae.train_X_numpy[csvae.train_y_numpy == 0, i],
                                csvae.train_X_numpy[csvae.train_y_numpy == 0, j],
                                c=BAD_OUTCOME, alpha=0.1, s=5)

                ax_.scatter(csvae.train_X_numpy[csvae.train_y_numpy == 1, i],
                                csvae.train_X_numpy[csvae.train_y_numpy == 1, j],
                                c=GOOD_OUTCOME, alpha=0.1, s=5)   


                p = ax_.scatter(mu_x_numpy[:, i].flatten(), 
                                mu_x_numpy[:, j].flatten(),
                                c=w_range_numpy.flatten(),
                                cmap=F_TO_CF_CMAP
                                )

                fig.colorbar(p, cax=cax, orientation='vertical')



                ax_.set_xlabel(f'feat {csvae.featnames[i]}')
                ax_.set_ylabel(f'feat {csvae.featnames[j]}')
                ax_.set_title(f"Evolution of joint feature values for feats {csvae.featnames[i]} and \n {csvae.featnames[j]}")
                ax_.legend()


                c+= 1


        plt.show()
        plt.close()

        
        
class CSVAE(nn.Module):
    """
    y_dim corresponds to the number of classes
    """
    def __init__(self, 
                 x_dim, 
                 y_dim, 
                 z_dim, 
                 w_dim, 
                 predict_x_var=False, 
                 labels_mutually_exclusive=True,
                 use_bernoulli_y=False,
                 z_prior_var=1,
                 likelihood_partition=None,
                 likelihood_params = {'lik_var' : 0.1**2, 'lik_var_lognormal': 0.1**2}, 
                 pw_y_means=torch.tensor([[-1.], 
                                          [1.]]),
                 pw_y_vars=torch.tensor([[0.5**2],
                                        [0.5**2]]),
                 show_val_loss=False,
                 beta1=20, # reconstruction_error,
                 beta2=1, # how strongly we want the identification network for q(w|x,y) to mimic the prior p(w|y)
                 beta3=0.2, # how strongly we want the identification network q(z|x) to mimic the prior p(z)
                 beta4=10, # beta4 and beta5 control how strongly we want to enforce the mutual information minimization between
                 beta5=1, # w and z. If too small, z will encode information about w. should be tweaked.
                 optimizer1=None,
                 optimizer2=None,
                 device='cpu'
                ):
        
        super().__init__()
        
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        print(f"x_dim: {x_dim}")
        print(f"y_dim: {y_dim}")
        print(f"z_dim: {z_dim}")
        print(f"w_dim: {w_dim}")
        
        print(f"beta1: {beta1}")
        print(f"beta2: {beta2}")
        print(f"beta3: {beta3}")
        print(f"beta4: {beta4}")
        print(f"beta5: {beta5}")
        
        if likelihood_partition is None:
            # range for feature types
            self.likelihood_partition = {
                (0, x_dim-1): 'real'
            }
        else:
            self.likelihood_partition = likelihood_partition
            
        self.likelihood_params = likelihood_params
        
        self.z_prior_var = z_prior_var

        self.device = device

        self.predict_x_var = predict_x_var
        self.labels_mutually_exclusive = labels_mutually_exclusive
        
        # prior on z
        
        p_z_loc = torch.zeros(z_dim).to(device)
        cov_diag = z_prior_var*torch.ones(z_dim).to(device)
        self.p_z = D.Normal(loc=p_z_loc,
                        scale=cov_diag.sqrt())
        
        # prior on y
        if labels_mutually_exclusive:
            if not use_bernoulli_y:
                self.p_y = D.Categorical(probs=1 / y_dim * torch.ones(1,y_dim)) 
            else:
                if y_dim != 2:
                    raise ValueError("using bernoulli y with a y_dim != 2")
                self.p_y = D.Bernoulli(probs=0.5) 

          
        else:
            self.p_y = D.Bernoulli(probs=0.5 * torch.ones(y_dim)) 
            
        
        self.decoder_x = nn.Sequential(
            nn.Linear(z_dim + w_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, x_dim)
        )
        
        self.encoder_w = nn.Sequential(
            nn.Linear(x_dim + ((y_dim-1) if y_dim == 2 else y_dim), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * w_dim)
        )
        
        self.encoder_z = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * z_dim)
        )

        self.optim_1_params = chain(self.decoder_x.parameters(), 
                                    self.encoder_w.parameters(), 
                                    self.encoder_z.parameters())
                                    
        self.decoder_y = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, y_dim)
        )

        
        self.optim_2_params = chain(self.decoder_y.parameters())
        
        if optimizer1 is None:
            self.optimizer1 = torch.optim.Adam(self.optim_1_params, lr=1e-3)
        else:
            self.optimizer1 = optimizer1
            
        if optimizer2 is None:
            self.optimizer2 = torch.optim.Adam(self.optim_2_params, lr=1e-3)
        else:
            self.optimizer2 = optimizer2
            
            
        self.pw_y_means = pw_y_means
        self.pw_y_vars = pw_y_vars
        self.show_val_loss = show_val_loss
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.beta5 = beta5
            
            
        self.pw_y0 = self.get_pw_y(torch.tensor(0))
        self.pw_y1 = self.get_pw_y(torch.tensor(1))
            

    def set_train_dataset(self, dataset, featnames=None):
        self.train_dataset = dataset
        
        self.train_X = dataset.data
        self.train_X_numpy = dataset.data.cpu().detach().numpy()
        
        self.train_y = dataset.targets
        self.train_y_numpy = dataset.targets.cpu().detach().numpy()
        
        if featnames is not None:
            self.featnames = featnames
        else:
            self.featnames = list(range(self.train_X_numpy.shape[1]))
            
        self.p_y1 = np.mean(self.train_y_numpy)
        self.p_y0 = 1 - self.p_y1
        
        print(f"p_y1: {self.p_y1}")
        print(f"p_y0: {self.p_y0}")
        
        #self.p_y = D.Categorical(probs=torch.tensor([self.p_y0, self.p_y1]).float()) 
        
    def set_val_dataset(self, dataset):
        self.val_X = dataset.data
        self.val_X_numpy = dataset.data.cpu().detach().numpy()
        
        self.val_y = dataset.targets
        self.val_y_numpy = dataset.targets.cpu().detach().numpy()
        
    def decode_x(self, w, z):
        params = self.decoder_x(torch.cat((w, z), dim=-1))

        px_wz = []
        samples = []
        
        for indices in self.likelihood_partition:
            data_type = self.likelihood_partition[indices]
            
            params_subset = params[:, indices[0]:(indices[1] + 1)]
            
            
            if data_type == 'real':
                cov_diag = self.likelihood_params['lik_var']*torch.ones_like(params_subset).to(self.device)

                dist = D.Normal(
                        loc=params_subset,
                        scale=cov_diag.sqrt())
                    
            elif data_type == 'categorical':
                dist = D.OneHotCategorical(logits=params_subset)
            elif data_type == 'binary':
                dist = D.Bernoulli(logits=params_subset)
            elif data_type == 'positive':
                lognormal_var = self.likelihood_params['lik_var_lognormal']*torch.ones_like(params_subset).to(self.device)
                
                dist = D.LogNormal(loc=params_subset, scale=lognormal_var.sqrt())
            elif data_type == 'count':
                positive_params_subset = F.softplus(params_subset)
                dist = D.Poisson(rate=positive_params_subset)
            elif data_type == 'binomial':
                num_trials = self.likelihood_params['binomial_num_trials']
                dist = D.Binomial(total_count=num_trials, logits=params_subset)
            elif data_type == 'ordinal':
                h = params_subset[:, 0:1]
                thetas = torch.cumsum(F.softplus(params_subset[:, 1:]), axis=1)
                                
                prob_lessthans = torch.sigmoid(thetas - h)
                probs = torch.cat((prob_lessthans, torch.ones(len(prob_lessthans), 1)), axis=1) - \
                        torch.cat((torch.zeros(len(prob_lessthans), 1), prob_lessthans), axis=1)
                                
                dist = D.OneHotCategorical(probs=probs)
            else:
                raise NotImplementedError
            
                
            samples.append(dist.sample())
            px_wz.append(dist)

        sample_x = torch.cat(samples, axis=1)
            
        return params, sample_x, px_wz       
                                    
    def encode_w(self, x, y):
        xy = torch.cat((x, y), dim=-1)
        out = self.encoder_w(xy)
        
        mu = out[:, :self.w_dim]
        logvar = out[:, self.w_dim:]  # decorrelated gaussian

        
        qw_xy = D.Normal(loc=mu,
                        scale=(logvar/2).exp())
        
        sample = qw_xy.rsample()
        return mu, logvar, sample, qw_xy

    def encode_z(self, x):
        out = self.encoder_z(x)

        mu = out[:, :self.z_dim]
        logvar = out[:, self.z_dim:]  
        
        qz_x = D.Normal(loc=mu,
                        scale=(logvar/2).exp())

        sample = qz_x.rsample()
        return mu, logvar, sample, qz_x

    def decode_y(self, z):
        
        if self.labels_mutually_exclusive:
            probs = F.softmax(self.decoder_y(z), dim=-1)

            qy_z = D.Categorical(probs=probs)
        else:
            probs = F.sigmoid(self.decoder_y(z))

            qy_z = D.Bernoulli(probs=probs)
        
        return probs, qy_z

    def forward(self, x, y):
        mu_z, logvar_z, sample_z, qz_x = self.encode_z(x)
        mu_w, logvar_w, sample_w, qw_xy = self.encode_w(x, y)
        
        try:
            mu_x, logvar_x, pred_x, px_wz = self.decode_x(sample_w,
                                                           sample_z)
        except:
            mu_x, pred_x, px_wz = self.decode_x(sample_w, 
                                                 sample_z)

        pred_y, qy_z = self.decode_y(sample_z)
        return qz_x, qw_xy, px_wz, px_wz, qy_z
    
    def get_pw_y(self, y):
        """
        Right now this is not robust enough to handle a setting
         with more than 2 possibly non-mutually exclusive classes
        """
        
        loc = self.pw_y_means[y.long()]
        scale = self.pw_y_vars[y.long()].sqrt()

        pw_y = D.Normal(loc=loc,
                        scale=scale)

        
        return pw_y
    
    def recon_error(self, X, px_wz):
        RL = None
        for indices, dist in zip(self.likelihood_partition, px_wz):
            data_subset = X[:, indices[0]:indices[1] + 1]
            
            partial_RL = (-1. * dist.log_prob(data_subset))
            
            if len(partial_RL.shape) > 1:
                partial_RL = partial_RL.sum(axis=1)
            
            if RL is None:
                RL = partial_RL
            else: 
                RL = RL + partial_RL
        
        return RL

    def compute_M1(self, x, y, px_wz, qw_xy, pw_y, qz_x):

        M1 = (
            # reconstruction error
            self.beta1 * self.recon_error(x, px_wz) 
            # forcing posterior of w to be close to prior
          + self.beta2 * KL(qw_xy, pw_y).sum(axis=1) 
            # forcing posterior of z to be close to prior
          + self.beta3 * KL(qz_x, self.p_z).sum(axis=1)
            # doesn't affect optimization but is included for
            # correctness
          - self.p_y.log_prob(y) 
             )

        return M1

    def compute_M2(self, qy_z):
        # conditional entropy of y given z
        M2 = self.beta4 * -1 * qy_z.entropy()
        return M2

    def compute_N(self, y, qy_z):
        # this is the approximate posterior y_z.
        # we alternate learning this and M2 because
        # the two objectives are adversarial
        # we want to kind of learn the posterior py_z but also we want
        # z to not convey any information about y. 
        N = self.beta5 * qy_z.log_prob(y) 
        return N
    
    def get_loss_components(self, x, y, px_wz, qw_xy, pw_y, qz_x, qy_z):
        term1 = self.beta1 * self.recon_error(x, px_wz) 
        term2 = self.beta2 * KL(qw_xy, pw_y).sum(axis=1) 
        term3 = self.beta3 * KL(qz_x, self.p_z).sum(axis=1)
        term4 = self.beta4 * -1 * qy_z.entropy()
        term5 = self.beta5 * -1 * qy_z.log_prob(y) 
        
        return term1, term2, term3, term4, term5 
    
    def train_on_batch_orig(self, X, y):
      
        pw_y = self.get_pw_y(y)
                
        # encode
        _, _, sample_z, qz_x = self.encode_z(X)
        _, _, sample_w, qw_xy = self.encode_w(X, y.unsqueeze(-1).float())

        # decode
        _, _, px_wz = self.decode_x(sample_w, sample_z)

        if self.labels_mutually_exclusive:
            pred_y, qy_z = self.decode_y(sample_z)
        else:
            raise NotImplementedError

        self.optimizer1.zero_grad()
        
        M1 = self.compute_M1(
                                X, 
                                y, 
                                px_wz, 
                                qw_xy, 
                                pw_y, 
                                qz_x
                        )
      

        # forcing mutual information between z and y to be low (same as forcing conditional entropy to be high)
        if self.labels_mutually_exclusive:
            M2 = self.compute_M2(qy_z)
        else:
            raise NotImplementedError

        loss1 = (M1 + M2).mean()

        loss1.backward(retain_graph=True)
        
        

        self.optimizer2.zero_grad()

        if self.labels_mutually_exclusive:
            N = self.compute_N(y, qy_z) # Forcing the identification network for y to actually learn something
        else:
            raise NotImplementedError

        loss2 = (-1*N).mean()
        
        loss2.backward()
        
        self.optimizer1.step()
        self.optimizer2.step()


    def train_on_batch(self, X, y):
        # first update optimizer 1 parameters
        pw_y = self.get_pw_y(y)
                
        # encode
        _, _, sample_z, qz_x = self.encode_z(X)
        _, _, sample_w, qw_xy = self.encode_w(X, y.unsqueeze(-1).float())

        # decode
        _, _, px_wz = self.decode_x(sample_w, sample_z)

        if self.labels_mutually_exclusive:
            pred_y, qy_z = self.decode_y(sample_z)
        else:
            raise NotImplementedError

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        
        M1 = self.compute_M1(
                                X, 
                                y, 
                                px_wz, 
                                qw_xy, 
                                pw_y, 
                                qz_x
                        )
      

        # forcing mutual information between z and y to be low (same as forcing conditional entropy to be high)
        if self.labels_mutually_exclusive:
            M2 = self.compute_M2(qy_z)
        else:
            raise NotImplementedError

        loss1 = (M1 + M2).mean()

        loss1.backward()
        self.optimizer1.step()


        ###################################################



        # then recompute everything and perform step for second optimizer
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        # encode
        _, _, sample_z, qz_x = self.encode_z(X)


        if self.labels_mutually_exclusive:
            pred_y, qy_z = self.decode_y(sample_z)
        else:
            raise NotImplementedError

        if self.labels_mutually_exclusive:
            N = self.compute_N(y, qy_z) # Forcing the identification network for y to actually learn something
        else:
            raise NotImplementedError

        loss2 = (-1*N).mean()
        
        loss2.backward()
        
        self.optimizer2.step()
        
                                    

    def train(self, num_epochs=100, batch_size=128):
        self.history = {
            'term1': [],
            'term2': [],
            'term3': [],
            'term4': [],
            'term5': [],
            'avg_loss': [],
            'avg_y_pred_max': [],
            'avg_y_pred_min': [],
            'val_term1': [],
            'val_term2': [],
            'val_term3': [],
            'val_term4': [],
            'val_term5': [],
            'val_avg_loss': [],
            'val_avg_y_pred_max': [],
            'val_avg_y_pred_min': []
            
        }
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        train_dataloader = DataLoader(dataset=self.train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=0)
        
        with trange(1, num_epochs+1) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')

                for batch_idx, (X, y)  in enumerate(train_dataloader):
                    self.train_on_batch(X, y)
                    
                # train set metrics 
                with torch.no_grad():
                    pw_y = self.get_pw_y(self.train_y)

                    # encode
                    _, _, sample_z, qz_x = self.encode_z(self.train_X)
                    _, _, sample_w, qw_xy = self.encode_w(self.train_X, self.train_y.unsqueeze(-1).float())

                    # decode
                    _, _, px_wz = self.decode_x(sample_w, sample_z)

                    if self.labels_mutually_exclusive:
                        pred_y, qy_z = self.decode_y(sample_z)
                    else:
                        raise NotImplementedError

                    term1, term2, term3, term4, term5 = self.get_loss_components(self.train_X, 
                                                                                 self.train_y, 
                                                                                 px_wz, 
                                                                                 qw_xy, 
                                                                                 pw_y, 
                                                                                 qz_x, 
                                                                                 qy_z)
               
                    term1 = term1.mean().item()
                    term2 = term2.mean().item()
                    term3 = term3.mean().item()
                    term4 = term4.mean().item()
                    term5 = term5.mean().item()
                    avg_loss = term1 + term2 + term3 + term4 + term5
                    avg_y_pred_max = pred_y.detach().numpy().max(axis=1).mean()
                    avg_y_pred_min = pred_y.detach().numpy().min(axis=1).mean()
                    
                    self.history['term1'].append(term1)
                    self.history['term2'].append(term2)
                    self.history['term3'].append(term3)
                    self.history['term4'].append(term4)
                    self.history['term5'].append(term5)
                    self.history['avg_loss'].append(avg_loss)
                    self.history['avg_y_pred_max'].append(avg_y_pred_max)
                    self.history['avg_y_pred_min'].append(avg_y_pred_min)
                    
                    
                with torch.no_grad():
                    pw_y = self.get_pw_y(self.val_y)

                    # encode
                    _, _, sample_z, qz_x = self.encode_z(self.val_X)
                    _, _, sample_w, qw_xy = self.encode_w(self.val_X, self.val_y.unsqueeze(-1).float())

                    # decode
                    _, _, px_wz = self.decode_x(sample_w, sample_z)

                    if self.labels_mutually_exclusive:
                        pred_y, qy_z = self.decode_y(sample_z)
                    else:
                        raise NotImplementedError

                    val_term1, val_term2, val_term3, val_term4, val_term5 = self.get_loss_components(self.val_X, 
                                                                                                     self.val_y, 
                                                                                 px_wz, 
                                                                                 qw_xy, 
                                                                                 pw_y, 
                                                                                 qz_x, 
                                                                                 qy_z)
               
                    val_term1 = val_term1.mean().item()
                    val_term2 = val_term2.mean().item()
                    val_term3 = val_term3.mean().item()
                    val_term4 = val_term4.mean().item()
                    val_term5 = val_term5.mean().item()
                    val_avg_loss = val_term1 + val_term2 + val_term3 + val_term4 + val_term5
                    val_avg_y_pred_max = pred_y.detach().numpy().max(axis=1).mean()
                    val_avg_y_pred_min = pred_y.detach().numpy().min(axis=1).mean()
                    
                    self.history['val_term1'].append(val_term1)
                    self.history['val_term2'].append(val_term2)
                    self.history['val_term3'].append(val_term3)
                    self.history['val_term4'].append(val_term4)
                    self.history['val_term5'].append(val_term5)
                    self.history['val_avg_loss'].append(val_avg_loss)
                    self.history['val_avg_y_pred_max'].append(val_avg_y_pred_max)
                    self.history['val_avg_y_pred_min'].append(val_avg_y_pred_min)                    

                    

                t.set_postfix(avg_loss=avg_loss, 
                              avg_y_max=avg_y_pred_max, 
                              avg_y_min=avg_y_pred_min,
                              reconstruction_error=term1,
                              kl_wxy_wy=term2,
                              kl_zx_z=term3,
                              cond_entropy_y_z=term4,
                              post_y_approx=term5,
                              val_avg_loss=val_avg_loss, 
                              val_avg_y_max=val_avg_y_pred_max, 
                              val_avg_y_min=val_avg_y_pred_min,
                              val_reconstruction_error=val_term1,
                              val_kl_wxy_wy=val_term2,
                              val_kl_zx_z=val_term3,
                              val_cond_entropy_y_z=val_term4,
                              val_post_y_approx=val_term5
                             )
                     

    def unconditional_sample(self, y, S=1000):
        with torch.no_grad():
            sample_z_synthetic = self.p_z.sample((len(y),))

            # sample w given y
            sample_w_y_synthetic = self.get_pw_y(y).sample()

            # decode
            _, sample_x_synthetic, _ = self.decode_x(sample_w_y_synthetic, sample_z_synthetic)
            
            
        
        return sample_x_synthetic

    
        
    def sample(self, S, x, y_prime, sampling_method='one', multiple_zs=True, debug=False):
        """
        S: num_samples
        x: datapoint in need of recourse
        y_prime: target class (different from current class of x under classifier)
        """
        y_prime_torch = torch.tensor([y_prime]).unsqueeze(-1).float()
        y_torch = torch.tensor([1 - y_prime]).unsqueeze(-1).float()
        
        if debug:
            print(x)
            print(y_prime)
            
            with torch.no_grad():
                mu_z, logvar_z, sample_z, qz_x = self.encode_z(self.train_X)

                mu_w, logvar_w, sample_w, qw_xy = self.encode_w(self.train_X, 
                                                                self.train_y.unsqueeze(-1).float())

                mu_x, sample_x, px_wz = self.decode_x(sample_w, sample_z)

                # sample z
                sample_z_synthetic = self.p_z.sample((len(self.train_y),))

                # sample w given y
                sample_w_y_synthetic = self.get_pw_y(self.train_y).sample()

                # decode
                _, sample_x_synthetic, _ = self.decode_x(sample_w_y_synthetic, sample_z_synthetic)
        
        # sample many z_primes and w_primes
        _, _, _, qz_x = self.encode_z(x)

        if multiple_zs:
            z_prime = qz_x.sample((S,)).squeeze()
        else:
            z_prime = qz_x.sample((1,)).squeeze().repeat((S,1))


        if sampling_method == 'one':
            _, _, _, qw_xy_prime = self.encode_w(x, y_prime_torch)
            w_prime = qw_xy_prime.sample((S,)).squeeze(-1)

            if debug:
                _, _, _, qw_xy = self.encode_w(x, y_torch)
                w = qw_xy.sample((S,)).squeeze(-1)
                    
                       
        
        elif sampling_method == 'two':
            w_prime = self.get_pw_y(torch.tensor([y_prime]).int()).sample((S,)).squeeze(-1)
            
            if debug:
                #w = self.get_pw_y(torch.tensor([1 - y_prime]).int()).sample((S,)).squeeze(-1)
                _, _, _, qw_xy = self.encode_w(x, y_torch)
                w = qw_xy.sample((S,)).squeeze(-1)
                
        if len(z_prime.shape) == 1:
            z_prime = z_prime.view(-1, 1)
        
        _, sample_x_prime, px_wz = self.decode_x(w_prime, z_prime)
      
            
        return sample_x_prime, px_wz
        
        
    def density_x_prime(self, S, x, x_prime, y_prime, sampling_method='one', debug=False):
        with torch.no_grad():
            samples, px_wz = self.sample(S, x, y_prime, sampling_method=sampling_method)
            
            if debug:
                print("x", x.shape, x)
                print("x_prime", x_prime.shape, x_prime)
                print("y_prime", y_prime)
                print("samples", samples.shape)
                print("px_wz loc ", px_wz.loc.shape)

            x_prime_mod = x_prime.unsqueeze(1)
            if debug: print("x_prime_mod shape", x_prime_mod.shape)
            logprobs = px_wz.log_prob(x_prime_mod)
            
            logsumexp = logprobs.exp().sum(axis=1).log() - np.log(S)
            logsumexp2 = logprobs.logsumexp(axis=1) - np.log(S)
            
            if debug:
                print("logprobs shape", logprobs.shape)
                print("logsumexp", logsumexp)
                print("logsumpexp2", logsumexp2)
                print("logsumpexp exp", logsumexp.exp())
                
            return logsumexp2
        

    def py_given_x(self, x, num_samples=5000, debug=False, cond_on_x=False):
        with torch.no_grad():
            y_samples = self.p_y.sample((num_samples,)).float()
            x_duplicated = torch.cat(num_samples*[x])
            
            if debug:
                print(y_samples)
                print(f"y mean: {y_samples.detach().numpy().mean()}")
         
            _, _, w_samples, _ = self.encode_w(x_duplicated, y_samples)
            
            if debug:
                w_samples_numpy = w_samples.detach().numpy()
                y_samples_numpy = y_samples.detach().numpy()
                
                
                
            y0 = torch.zeros((num_samples, 1))
            y1 = torch.ones((num_samples, 1))
            
            logpy0 = np.log(0.5)
            logpy1 = np.log(0.5)
               
            if cond_on_x:
                if debug: print("conditioning w distribution on x")
                _, _, _, qw_xy0 = self.encode_w(x, torch.tensor([[0.]]).float())
                _, _, _, qw_xy1 = self.encode_w(x, torch.tensor([[1.]]).float())
            else:
                if debug: print("not conditioning w distribution on x")
                qw_xy0 = self.pw_y0
                qw_xy1 = self.pw_y1
            
            term1 = qw_xy1.log_prob(w_samples)
            term2 = logpy1
            term3 = qw_xy0.log_prob(w_samples)
            term4 = logpy0
            term5 = qw_xy1.log_prob(w_samples)
            term6 = logpy1
            
            if debug:
                print(term1.shape)
                print(term2.shape)
                print(term3.shape)
                print(term4.shape)
                print(term5.shape)
            
            red1 = (term3 + term4).exp()
            red2 = (term5 + term6).exp()
            red3 = (red1 + red2).log()
            
            red4 = term1 + term2 - red3
            red5 = red4.exp()
            red6 = red5.mean()
            
        return red6.item()