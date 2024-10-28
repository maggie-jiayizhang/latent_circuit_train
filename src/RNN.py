import numpy as np
import torch
import torch.nn as nn
import copy
from util_misc import *

torch.autograd.set_detect_anomaly(True)

def torch_detach(ts):
    return ts.detach().cpu().numpy()

# a generic RNN
class RNN(torch.nn.Module):
    def __init__(self, N, alpha=.2, sigma_rec=.15, input_size=6,
                 output_size=2, pos_input=True, pos_output=True,
                 device='cpu'):

        super(RNN, self).__init__()
        self.device = device

        # time scale
        self.alpha = torch.tensor(alpha, device=device)
        # noise
        self.sigma_rec = torch.tensor(sigma_rec, device=device)

        self.N = N
        self.input_size = input_size
        self.output_size = output_size
        self.activation = torch.nn.ReLU()
        self.pos_input = pos_input
        self.pos_output = pos_output

        # initalize recurrent, input, output weights
        self.recurrent_layer = nn.Linear(self.N, self.N, bias=False)
        self.recurrent_layer.weight.data.normal_(mean=0.0, std=1/self.N)

        self.input_layer = nn.Linear(self.input_size, self.N, bias=False)
        self.input_layer.weight.data.normal_(mean=1/np.sqrt(self.N), 
                                             std=1/np.sqrt(self.N))

        self.output_layer = nn.Linear(self.N, self.output_size, bias=False)
        self.output_layer.weight.data.normal_(mean=1/np.sqrt(self.N), 
                                              std=1/np.sqrt(self.N))

    def set_weights(self, wrec, win, wout):
        with torch.no_grad():
            self.recurrent_layer.weight.copy_(torch.from_numpy(wrec).float())
            self.input_layer.weight.copy_(torch.from_numpy(win).float())
            self.output_layer.weight.copy_(torch.from_numpy(wout).float())

    def loss_function(self):
        pass

    def make_state_matrix(self, u):
        # hidden states matrix: #trials x time steps x #neurons
        batch_size, t = u.shape[0], u.shape[1]
        return torch.zeros(batch_size, t, self.n, device=self.device)

    def forward(self, u, states=None): # default RNN starts with 0 vectors
        batch_size, t = u.shape[0], u.shape[1]
        
        # output hidden states matrix: #trials x time steps x #neurons
        if (states is None):
            # allocate hidden states matrix
            states = self.make_state_matrix(u)
        else:
            assert(states.shape[0] == batch_size and states.shape[1] == t)

        # noise
        noise = torch.sqrt(2*self.alpha*self.sigma_rec**2) * torch.empty(batch_size, t, self.n).normal_(mean=0, std=1).to(device=self.device)

        for i in range(1, t):
            last_state = states[:,i-1,:].clone()
            curr_in = u[:,i,:]
            curr_noise = noise[:,i,:]
            
            # adheres to langdon & engel 2023, Eq26
            leak = (1-self.alpha)*last_state
            dx = self.alpha*(self.activation(self.recurrent_layer(last_state) +
                                             self.input_layer(curr_in) + 
                                             curr_noise))
            states[:,i,:] = leak + dx
            assert(torch.equal(last_state,states[:,i-1,:]))
        return states
    
    def output(self, x):
        return self.output_layer(x)

KEYS_LIST = ["n", "alpha", "sigma_rec", "input_size", "output_size",
             "pos_input", "pos_output", "device",
             "wrec", "win", "wout"]

def rnn_init_from_dict(params_dict, keys_list=KEYS_LIST):
    n, alpha, sigma_rec, input_size, output_size, pos_input, pos_output, device, \
    wrec, win, wout = parse_dict(params_dict, keys_list)

    rnn = RNN(n, alpha, sigma_rec, input_size, output_size, pos_input, pos_output, device=device)
    rnn.set_weights(wrec, win, wout)
    
    return rnn



    
        