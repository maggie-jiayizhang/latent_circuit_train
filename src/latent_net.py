'''
The following code was written by Chris Langdon;
Edited and adapted by Jiayi Zhang.

Note: latent circuit is a sub-class of RNN.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import hessian # compute hessian
from RNN import *

torch.autograd.set_detect_anomaly(True)

# define the latent net class (subclass of RNN)
### NOTE: give a specific q to the model so we only learn the latent circuit ###
class LatentNet(RNN):
    def __init__(self, n, N, alpha=.2, sigma_rec=.15, input_size=6, 
                 output_size=2, pos_input=False, pos_output=False, 
                 device='cpu'):

        super(LatentNet, self).__init__(N, alpha, sigma_rec, input_size,
              output_size, pos_input, pos_output, device)
        
        self.n = n # latent circuit size

        # initialize latent circuit layers
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        self.recurrent_layer.weight.data.normal_(mean=0.0, std=1/n)

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.data.normal_(mean=1/np.sqrt(self.n), 
                                             std=1/np.sqrt(self.n))

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        self.output_layer.weight.data.normal_(mean=1/np.sqrt(self.n), 
                                              std=1/np.sqrt(self.n))
        self.q = None

    def set_q(self, q):
        self.q = q

    # Tansform a square matric into a rectangular orthonormal matrix
    def cayley_transform(self, a):
        skew = (a - a.t()) / 2
        skew = skew.to(device=self.device)
        eye = torch.eye(self.N).to(device=self.device)
        o = (eye - skew) @ torch.inverse(eye + skew)
        return o[:self.n, :]

    # Forward pass of the latent circuit
    def forward(self, u, states):
        return super().forward(u, states)

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
    def report(self, e, epochs, mse_z, mse_x, tr_loss, val_loss, tr_acc, val_acc):
        print('Epoch: {}/{}.............'.format(e, epochs), end=' ')
        print("val mse_z: {:.4f}".format(mse_z), end=' ')
        print("val mse_x: {:.4f}".format(mse_x), end=' ')
        print("train_loss: {:.4f}".format(tr_loss), end=' ')
        print("val_loss: {:.4f}".format(val_loss), end=' ')
        print("train_acc: {:.4f}".format(tr_acc), end=' ')
        print("val_acc: {:.4f}".format(val_acc))

    # Fitting
    def fit(self, u, y, z, tr_mask, val_mask, lr, l_x=1, l_z=1, 
            epochs=100, patience=20, stop_thresh=0.001, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, 
                                     weight_decay=0.001)
        # save loss history
        train_loss_history, val_loss_history = [],[]
        train_acc_history, val_acc_history = [],[]

        # loss record for early stopping
        loss_rec, loss_acc = None, 0

        for i in range(epochs):
            optimizer.zero_grad() # zero out grad for each epoch
            
            # set t0 state to the output
            states = super().make_state_matrix(u)
            states[:,0,:] = y[:,0,:] @ self.q.T
            x = self.forward(u, states) # forward pass

            tr_ez, tr_ex, loss = self.loss_function(x[tr_mask], y[tr_mask], z[tr_mask], l_x, l_z) # only compute on tr samples

            val_ez, val_ex, val_loss = self.loss_function(x[val_mask], y[val_mask], z[val_mask], l_x, l_z) # compute on withheld samples

            loss.backward()  # compute gradient
            optimizer.step() # update params
            # no need to update q since it's frozen

            # compute accuracy
            train_acc_history.append(self.acc(z[tr_mask], self.output_layer(x[tr_mask])))
            val_acc_history.append(self.acc(z[val_mask], self.output_layer(x[val_mask])))

            # save loss terms
            train_loss_history.append([tr_ez, tr_ex, loss.item()])
            val_loss_history.append([val_ez, val_ex, val_loss.item()])
            
            if verbose:
                if i % 10 == 0:
                    self.report(i, epochs, val_ez, val_ex, loss, val_loss, train_acc_history[-1], val_acc_history[-1])
        
            # clip self.input_layer.weight.data to be >= 0
            with torch.no_grad():
                if self.pos_input:
                    self.input_layer.weight.data = torch.clamp(self.input_layer.weight.data, min=0)
                # clip self.output_layer.weight.data to be >= 0
                if self.pos_output:
                    self.output_layer.weight.data = torch.clamp(self.output_layer.weight.data, min=0)
            
            # for early stopping
            # when validation loss is not improving significantly
            # (stagnant) consider stopping training
            if (loss_rec is None or (val_loss + stop_thresh) < loss_rec):
                loss_rec = loss
                loss_acc = 0
            else:
                loss_acc += 1

            if (loss_acc == patience): # end if no improvement of training loss
                break
            
        # compute final predictions
        self.eval()
        states = super().make_state_matrix(u)
        states[:,0,:] = y[:,0,:] @ self.q.T
        x = self.forward(u, states) # forward pass
        zhat = self.output_layer(x)

        # move x, z to cpu
        x = torch_detach(x)
        zhat = torch_detach(zhat)

        # move weight matrices to cpu
        recurrent_weights = torch_detach(self.recurrent_layer.weight.data)
        input_weights = torch_detach(self.input_layer.weight.data)
        output_weights = torch_detach(self.output_layer.weight.data)
        q_mat = torch_detach(self.q)

        return  train_loss_history, val_loss_history, \
                train_acc_history, val_acc_history, \
                x, zhat, val_mask, tr_mask, \
                recurrent_weights, input_weights, output_weights, \
                q_mat, len(val_loss_history)