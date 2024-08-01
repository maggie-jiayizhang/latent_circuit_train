'''
The following code was written by Chris Langdon;
Edited and adapted by Jiayi Zhang.
'''

import numpy as np
import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

def torch_detach(ts):
    return ts.detach().cpu().numpy()

# define the latent net class
class LatentNet(torch.nn.Module):
    def __init__(self, n, N, alpha=.2, sigma_rec=.15, input_size=6, 
                 output_size=2, pos_input=False, pos_output=False, 
                 device='cpu'):
        super(LatentNet, self).__init__()
        self.device = device

        # time scale
        self.alpha = torch.tensor(alpha, device=device)
        # noise
        self.sigma_rec = torch.tensor(sigma_rec, device=device)
        
        self.n = n # latent circuit size
        self.N = N # original system size
        self.input_size = input_size
        self.output_size = output_size
        self.activation = torch.nn.ReLU()
        self.pos_input = pos_input
        self.pos_output = pos_output

        # initialize latent circuit layers
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        self.recurrent_layer.weight.data.normal_(mean=0.0, std=1/n)

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.data.normal_(mean=1/np.sqrt(self.n), 
                                             std=1/np.sqrt(self.n))

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        self.output_layer.weight.data.normal_(mean=1/np.sqrt(self.n), 
                                              std=1/np.sqrt(self.n))

        self.a = torch.nn.Parameter(torch.rand(self.N, self.N, device=device),
                                    requires_grad=True)
        self.q = self.cayley_transform(self.a)

    # Tansform a square matric into a rectangular orthonormal matrix
    def cayley_transform(self, a):
        skew = (a - a.t()) / 2
        skew = skew.to(device=self.device)
        eye = torch.eye(self.N).to(device=self.device)
        o = (eye - skew) @ torch.inverse(eye + skew)
        return o[:self.n, :]

    # Forward pass of the latent RNN
    def forward(self, u):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=self.device)
        batch_size = states.shape[0]
        # noise outside input layer private to each node
        noise = torch.sqrt(2*self.alpha*self.sigma_rec**2) * torch.empty(batch_size, t, self.n).normal_(mean=0, std=1).to(device=self.device)
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states

    # Loss function for latent circuit model
    def loss_function(self, x, y, z, l_x, l_z):
        msez = self.mse_z(x, z) # behavioral loss
        msex = self.mse_x(x, y) # latent circuit loss
        return msez.item(), msex.item(), l_z * msez + l_x * msex
    
    # Behavior component of loss function
    def mse_z(self, x, z):
        mse = nn.MSELoss()
        z_bar = z - torch.mean(z, dim=[0,1], keepdim=True)
        return mse(self.output_layer(x), z) / mse(z_bar, torch.zeros_like(z_bar))

    # Latent circuit loss
    # || Qx - y ||2 / var(y)
    def mse_x(self, x, y):
        mse = nn.MSELoss()
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)
        x_proj = x @ self.q
        return mse(x_proj, y) / mse(y_bar, torch.zeros_like(y_bar))

    # Calculate task performance accuracy (needs to be fixed)
    def acc(self, true, pred):
        return torch.sum((pred[:,-1,0]>pred[:,-1,1])==true[:,-1,0]).item() / len(true)

    # Report current losses
    def report(self, e, epochs, mse_z, mse_x, orth, tr_loss, val_loss, tr_acc, val_acc):
        print('Epoch: {}/{}.............'.format(e, epochs), end=' ')
        print("val mse_z: {:.4f}".format(mse_z), end=' ')
        print("val mse_x: {:.4f}".format(mse_x), end=' ')
        print("val orth: {:.4f}".format(orth), end='  ')
        print("train_loss: {:.4f}".format(tr_loss), end=' ')
        print("val_loss: {:.4f}".format(val_loss), end=' ')
        print("train_acc: {:.4f}".format(tr_acc), end=' ')
        print("val_acc: {:.4f}".format(val_acc))

    # Fitting
    def fit(self, u, y, z, tr_mask, val_mask, lr=.01, l_x=1, l_z=1, 
            epochs=100, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, 
                                     weight_decay=0.001)
        # save loss history
        train_loss_history, val_loss_history = [],[]
        train_acc_history, val_acc_history = [],[]

        for i in range(epochs):
            optimizer.zero_grad() # zero out grad for each epoch
            
            x = self.forward(u) # forward pass

            tr_ez, tr_ex, tr_orth, loss = self.loss_function(x[tr_mask], y[tr_mask], z[tr_mask], l_x, l_z) # only compute on tr samples

            val_ez, val_ex, val_orth, val_loss = self.loss_function(x[val_mask], y[val_mask], z[val_mask], l_x, l_z) # compute on withheld samples

            loss.backward()  # compute gradient
            optimizer.step() # update params

            self.q = self.cayley_transform(self.a)

            # compute accuracy
            train_acc_history.append(self.acc(z[tr_mask], self.output_layer(x[tr_mask])))
            val_acc_history.append(self.acc(z[val_mask], self.output_layer(x[val_mask])))

            # save loss terms
            train_loss_history.append([tr_ez, tr_ex, tr_orth, loss.item()])
            val_loss_history.append([val_ez, val_ex, val_orth, val_loss.item()])
            
            if verbose:
                if i % 10 == 0:
                    self.report(i, epochs, val_ez, val_ex, val_orth, loss, val_loss, train_acc_history[-1], val_acc_history[-1])
        
            # clip self.input_layer.weight.data to be >= 0
            with torch.no_grad():
                if self.pos_input:
                    self.input_layer.weight.data = torch.clamp(self.input_layer.weight.data, min=0)
                # clip self.output_layer.weight.data to be >= 0
                if self.pos_output:
                    self.output_layer.weight.data = torch.clamp(self.output_layer.weight.data, min=0)
            
        # compute final predictions
        self.eval()
        x = self.forward(u)
        zhat = self.output_layer(x)

        # move x, z to cpu
        x = torch_detach(x)
        zhat = torch_detach(zhat)

        # move weight matrices to cpu
        recurrent_weights = torch_detach(self.recurrent_layer.weight.data)
        input_weights = torch_detach(self.input_layer.weight.data)
        output_weights = torch_detach(self.output_layer.weight.data)
        a_mat = torch_detach(self.a)
        q_mat = torch_detach(self.q)

        return  train_loss_history, val_loss_history, \
                train_acc_history, val_acc_history, \
                x, zhat, val_mask, tr_mask, \
                recurrent_weights, input_weights, output_weights, \
                a_mat, q_mat, self.alpha, self.sigma_rec, \
                l_x, l_z, lr, epochs
