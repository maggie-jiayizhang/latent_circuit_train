import os
from util_config import *

DATA_PATH  = "/scratch/gpfs/jz6521/latent_circuit_theory/rnn_data/rnn_perf_results_n6_m5_lx2_58489134.npy"
SAVE_DIR = "freeze_q"
BASE_PATH = "/scratch/gpfs/jz6521/latent_circuit_theory"
EPOCHS = 5000
LR = 0.01 # learning rate
ALPHA = 0.1
LZ = 1
LX_MIN = 0
LX_MAX = 1
LX_STEPS = 5
SIGMA_REC = 0.15
NS = [6]
VERBOSE = True
TR_VAL_SPLIT_SEED = 0
STOP_THRESH = 0.001
PATIENCE = 30
POS_INPUT = True
POS_OUTPUT = True
NMODELS = 200

def generate_config(fn="config.json"):
    config_dict = {"data_path": DATA_PATH, 
                   "save_dir": SAVE_DIR,
                   "base_path": BASE_PATH, 
                   "epochs": EPOCHS, 
                   "lr": LR,
                   "alpha": ALPHA,
                   "l_z": LZ,
                   "lx_min": LX_MIN,
                   "lx_max": LX_MAX,
                   "lx_steps": LX_STEPS,
                   "sigma_rec": SIGMA_REC,
                   "ns": NS,
                   "verbose": VERBOSE,
                   "tr_val_split_seed": TR_VAL_SPLIT_SEED,
                   "patience": PATIENCE,
                   "stop_thresh": STOP_THRESH,
                   "pos_input": POS_INPUT, 
                   "pos_output": POS_OUTPUT,
                   "nmodels": NMODELS
                   }
    
    batch_dir = os.path.join(BASE_PATH, SAVE_DIR)

    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    config_to_json(config_dict, os.path.join(batch_dir, fn))

    if not os.path.exists(f'{batch_dir}/models'):
        os.makedirs(f'{batch_dir}/models')

if __name__ == "__main__":
    generate_config()