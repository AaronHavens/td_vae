import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class P_density(nn.Module):
    def __init__(self, D_in, hid_units):
        self.h1 = nn.Linear(D_in, hid_units)
        self.h2 = nn.Linear(D_in, hid_units)
        self.mu = nn.Linear(hid_units, D_in)
        self.log_std = nn.Linear(hid_units, D_in) 
    # according to TD VAE paper 
    # [mu, log(sig)] = W3*tanh(W1*x + B1)*sigmoid(W2*x + B2) + B3     
    def forward(self, x):
        x1 = F.tanh(self.h1(x))
        x2 = F.sig(self.h2(x))
        x3 = torch.mul(x1, x2)
        mu_out = self.mu(x3)
        log_std_out = self.log_std(x3)
        return mu_out, log_std_out


class belief_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, batch_size):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, 2)

        self.linear = nn.Linear(self.hidden_dim, self.latent_dim)

    def init_hidden(self):
        return (torch.zeros(2, self.batch_size, self.hidden_dim),
                torch.zeros(2, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

class inference_P(nn.Module):
    def __init__(self, latent_dim, hid_units):
        #backward and forward inference dependent on dt
model = LSTM(lstm_in
        
