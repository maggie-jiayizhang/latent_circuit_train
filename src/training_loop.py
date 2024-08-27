import numpy as np

from util_config import *
from util_train import *

# training loop given replicate index and config file
def training_loop(m, config_fn, job_id=0):

    # load config dictionary from json file
    CONFIG = json_to_config(config_fn)

    data_path = CONFIG['data_path'] # rnn/neural data
    base_path = CONFIG['base_path'] # base dir for lc
    save_dir = CONFIG['save_dir'] # save dir for lc

    lr = CONFIG['lr'] # learning rate
    alpha = CONFIG['alpha'] # alpha/time scale
    sigma_rec = CONFIG['sigma_rec']
    epochs = CONFIG['epochs']
    tr_val_split_seed = CONFIG['tr_val_split_seed']

    pos_input = CONFIG['pos_input']
    pos_output = CONFIG['pos_output']

    ns = CONFIG['ns']
    
    l_z = CONFIG['l_z']
    lx_steps = CONFIG['lx_steps']
    lxs = np.logspace(CONFIG['lx_min'], CONFIG['lx_max'], lx_steps)

    verbose = CONFIG['verbose']

    patience = CONFIG['patience']
    stop_thresh = CONFIG['stop_thresh']

    if not os.path.exists(f'{base_path}/{save_dir}/models'):
        os.makedirs(f'{base_path}/{save_dir}/models')

    for n in ns:
        for lxi in range(lx_steps):
            l_x = lxs[lxi]
            # train model
            print(f'n = {n}, model {m}, lx {l_x}')
            results = train_lc(data_path, n=n, lr=lr, alpha=alpha,
                               epochs=epochs, l_x=l_x, l_z=l_z, sigma_rec=sigma_rec,
                               tr_val_split_seed=tr_val_split_seed,
                               patience=patience, stop_thresh=stop_thresh,
                               pos_input=pos_input, pos_output=pos_output,
                               verbose=verbose)
            
            train_loss, val_loss, train_acc, val_acc, \
            x, zhat, val_mask, tr_mask, \
            wrec, win, wout, a_mat, q_mat, epochs_real = results
            
            # save loss history
            np.savez(f'{base_path}/{save_dir}/models/results_n{n}_m{m}_lx{lxi}_{job_id}.npz',
                     train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc,
                     x=x, zhat=zhat, wrec=wrec, win=win, wout=wout,
                     a_mat=a_mat, q_mat=q_mat, val_mask=val_mask, tr_mask=tr_mask, alpha=alpha, sigma_rec=sigma_rec,
                     l_x=l_x, l_z=l_z, lr=lr, epochs=epochs_real)
            # epochs here is the actual epoch (under patience and threshold)

    if (m==0): # save a copy of the config for each job_id
        CONFIG["job_id"] = job_id
        config_to_json(CONFIG, os.path.join(base_path, save_dir, 
                                            f"config_job{job_id}.json"))

def extract_RNN_name(RNN_fp, fmt="%s"):
    idx = RNN_fp[:-4].split("_")[-1] # extract RNN idx
    return fmt%idx

# create new config files from template (config_fn)
# such that data is the RNN specified in the RNN_list
def create_config(config_fn, RNN_list_fp, name_func, save_path, n=-1):

    RNN_list_file = open(RNN_list_fp, "r")
    RNN_list = RNN_list_file.read()
    RNN_list = RNN_list.strip().split("\n")
    RNN_list = RNN_list[:n]
    RNN_list_file.close()

    for data_fp in RNN_list:
        CONFIG_TMP = json_to_config(config_fn) # get config template
        name = name_func(data_fp)
        CONFIG_TMP["data_path"] += f"{data_fp}"
        CONFIG_TMP["save_dir"] += f"{name}"
        config_to_json(CONFIG_TMP, f'{save_path}/config_{name}.json')

def training_loop_batch(m, config_list):
    for config_fn in config_list:
        training_loop(m, config_fn)