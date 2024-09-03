import numpy as np
import torch
import torch.nn as nn
import copy
from util_misc import *

torch.autograd.set_detect_anomaly(True)

def torch_detach(ts):
    return ts.detach().cpu().numpy()


class RNN(torch.nn.Module):
    def __init__(self, n, alpha=.2, sigma_rec=.15, input_size=6,
                 output_size=2, pos_input=True, pos_output=True,
                 device='cpu'):
        super(RNN, self).__init__()
        self.device = device

        # time scale
        self.alpha = torch.tensor(alpha, device=device)
        # noise
        self.sigma_rec = torch.tensor(sigma_rec, device=device)

        self.n = n
        self.input_size = input_size
        self.output_size = output_size
        self.activation = torch.nn.ReLU()
        self.pos_input = pos_input
        self.pos_output = pos_output

        # initalize recurrent, input, output weights
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        self.recurrent_layer.weight.data.normal_(mean=0.0, std=1/n)

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.data.normal_(mean=1/np.sqrt(self.n), 
                                             std=1/np.sqrt(self.n))

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        self.output_layer.weight.data.normal_(mean=1/np.sqrt(self.n), 
                                              std=1/np.sqrt(self.n))

    def set_weights(self, wrec, win, wout):
        self.recurren_layer.weight.copy_(torch.from_numpy(wrec))
        self.input_layer.weight.copy_(torch.from_numpy(win))
        self.output_layer.weight.copy_(torch.from_numpy(wout))

    def loss_function(self):
        pass

    def forward(self, u):
        batch_size, t = u.shape[0], u.shape[1]
        states = torch.zeros(batch_size, t, self.n, device=self.device)
        # noise
        noise = torch.sqrt(2*self.alpha*self.sigma_rec**2) * torch.empty(batch_size, t, self.n).normal_(mean=0, std=1).to(device=self.device)

        for i in range(1, t):
            curr_state = states[:,i,:]
            curr_in = u[:,i,:]
            curr_noise = noise[:,i,:]
            
            leak = (1-self.alpha)*curr_state
            dx = self.alpha*(self.activation(self.recurrent_layer(curr_state) +
                                             self.input_layer(curr_in) + 
                                             curr_noise))
            states[:,i,:] = leak + dx
        
        return states


KEYS_LIST = ["n", "alpha", "sigma_rec", "input_size", "output_size",
             "pos_input", "pos_output", "device",
             "wrec", "win", "wout"]

def rnn_init_from_dict(params_dict, keys_list=KEYS_LIST):
    n, alpha, sigma_rec, input_size, output_size, pos_input, pos_output, device, \
    wrec, win, wout = parse_dict(params_dict, keys_list)

    rnn = RNN(n, alpha, sigma_rec, input_size, output_size, pos_input, pos_output, device=device)
    rnn.set_weights(wrec, win, wout)
    
    return rnn




    
        