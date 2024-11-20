import torch
import time
import numpy as np
from latent_net import *

def load_data(path_to_data, device):
    # load actual neural (unit) activities from path
    all_data = np.load(path_to_data, allow_pickle=True).item()

    Y = all_data['Y']
    U = all_data['U']
    Z = all_data['Z']

    if device is not None: ## convert all arrays to tensors
        Y = torch.tensor(Y, device=device).float()
        Z = torch.tensor(Z, device=device).float()
        U = torch.tensor(U, device=device).float()
    
    return Y, Z, U

def load_q(path_to_data, device):
    # load actual neural (unit) activities from path
    all_data = np.load(path_to_data, allow_pickle=True).item()
    Q = all_data["Q"]

    if device is not None: ## convert all arrays to tensors
        Q = torch.tensor(Q, device=device).float()
    
    return Q

def load_data_deprecate(path_to_data, device):
    # load actual neural (unit) activities from path
    all_data = np.load(path_to_data, allow_pickle=True)

    Y = all_data['traces'] 
    U = all_data['inputs']
    Z = all_data['targets'] # task targets

    # get data dimensions
    n, t, tr = Y.shape 
    n_in, _, _ = U.shape
    n_out, _, _ = Z.shape
    
    # reorders data for training
    new_order = [2, 1, 0]
    Y = Y.transpose(new_order)
    U = U.transpose(new_order)
    Z = Z.transpose(new_order)
    assert(Y.shape == (tr, t, n))
    assert(U.shape == (tr, t, n_in))
    assert(Z.shape == (tr, t, n_out))
    
    if (device == torch.device('mps')): # requirement of mps
        Y = Y.astype(np.float32)
        Z = Z.astype(np.float32)
        U = U.astype(np.float32)
        
    if device is not None: ## convert all arrays to tensors
        Y = torch.tensor(Y, device=device).float()
        Z = torch.tensor(Z, device=device).float()
        U = torch.tensor(U, device=device).float()
    
    return Y, Z, U

def get_train_val_idx(n_samples, seed=0, split_frac=0.7):
    # set seed
    rng = np.random.default_rng(seed)

    # make training/validation binary mask
    tr_mask = np.zeros(n_samples, dtype=bool)
    ntrain = int(split_frac*n_samples)
    
    train_idx = rng.choice(n_samples, ntrain, replace=False)
    tr_mask[train_idx] = True
    val_mask = np.logical_not(tr_mask)
    assert np.sum(np.logical_and(tr_mask, val_mask)) == 0
    assert np.sum(np.logical_or(tr_mask, val_mask)) == n_samples

    return tr_mask, val_mask

# train an instance of a latent circuit
def train_lc(data_path, n=30, lr=.01, l_x=1, l_z=1, 
             alpha=0.2, sigma_rec=0.15, patience=20, stop_thresh=0.001,
             epochs=500, verbose=False, tr_val_split_seed=42, 
             pos_input=False, pos_output=False):

    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda available: ', torch.cuda.is_available())

    # load data
    Y, Z, U = load_data(data_path, device)
    
    # load the correct Q
    Q = load_q(data_path, device)

    # make training/validation binary mask
    tr_mask, val_mask = get_train_val_idx(Y.shape[0], seed=tr_val_split_seed)

    # set up model
    N = np.shape(Y)[2] # number of real/biological neurons
    isize = U.shape[2] # input size
    osize = Z.shape[2] # output size

    tic = time.time() # timer start

    # initialize model (RNN)
    model = LatentNet(n, N, alpha=alpha, sigma_rec=sigma_rec,
                      input_size=isize, output_size=osize, device=device,
                      pos_input=pos_input, pos_output=pos_output)
    model.to(device)
    model.set_q(Q)
    
    results = model.fit(U, Y, Z, tr_mask=tr_mask, val_mask=val_mask, 
                        l_x=l_x, l_z=l_z, lr=lr, patience=patience, stop_thresh=stop_thresh, 
                        epochs=epochs, verbose=verbose)

    toc = time.time()
    print("n = %d, time taken: %.2fs"%(n, toc-tic))

    # return train/val loss history
    return results
