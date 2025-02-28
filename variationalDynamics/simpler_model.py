import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.distributions import Normal


class GaussianModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(GaussianModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = nn.Linear(state_size + action_size, hidden_size)
        self.nn2 = nn.Linear(hidden_size, hidden_size)
        self.output_dim = state_size
        self.nn3 = nn.Linear(hidden_size, 2*self.output_dim)



    def forward(self, x):
        nn1_output = F.leaky_relu(self.nn1(x))
        nn2_output = F.leaky_relu(self.nn2(nn1_output))
        nn3_output = self.nn3(nn2_output)

        mean, log_var = torch.chunk(nn3_output, 2, dim=-1)
        log_var = torch.clamp(log_var, min=-5, max=1)
        # mean = nn3_output[:, :self.output_dim]

        # std = F.softplus(nn3_output[:, self.output_dim:]) + 1e-5
        return mean, log_var

    def loss(self, mean, std, labels):
        var = torch.square(std)

        mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) / var, dim=-1), dim=-1)
        var_loss = torch.mean(torch.mean(torch.log(var), dim=-1), dim=-1)
        total_loss = mse_loss + var_loss

        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class GaussianDynamicsModel():
    def __init__(self, state_size, action_size, device, hidden_size=200):
        self.state_size = state_size
        self.action_size = action_size
        self.gaussian_model = GaussianModel(state_size, action_size, hidden_size).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.gaussian_model.parameters(), lr=1e-3)

    def optimize_model(self, memory):
        state, action, reward, next_state, done = memory.sample(128)
        delta_state = next_state - state

        inputs = np.concatenate((state, action), axis=-1)
        labels = delta_state

        train_input = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        train_label = torch.tensor(labels, dtype=torch.float32, device=self.device)

        mean, log_var = self.gaussian_model(train_input)
        var = torch.exp(log_var)
        # var = torch.square(std)

        mse_loss = torch.mean(torch.mean(torch.pow(mean - train_label, 2) / var, dim=-1), dim=-1)
        var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
        total_loss = mse_loss + var_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


    def predict(self, inputs, batch_size=1024):
        gaussian_mean, gaussian_std = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(self.device)
            b_mean, b_std = self.gaussian_model(input)
            gaussian_mean.append(b_mean.detach().cpu().numpy())
            gaussian_std.append(b_std.detach().cpu().numpy())
        gaussian_mean = np.concatenate(gaussian_mean, axis=0)
        gaussian_std = np.concatenate(gaussian_std, axis=0)

        return gaussian_mean, gaussian_std


class VariationalGaussianDynamicsModel():
    def __init__(self, state_size, action_size, device, hidden_size=200):
        self.state_size = state_size
        self.action_size = action_size
        self.gaussian_model = GaussianModel(state_size, action_size, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.gaussian_model.parameters(), lr=1e-3)
        self.device = device

    def optimize_model(self, memory, p_model, critic, beta):
        state, action, reward, next_state, done = memory.sample(128)
        inputs = np.concatenate((state, action), axis=-1)
        state = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        # beta = torch.from_numpy(beta).float().to(self.device)
        train_input = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        mean_q, logvar_q = self.gaussian_model(train_input)
        var_q = torch.exp(logvar_q)
        std_q = torch.exp(logvar_q/2)

        mean_q = mean_q + train_input[:, :self.state_size]

        normal_q = Normal(mean_q, std_q)

        next_state_batch = normal_q.rsample()
        next_state_batch = torch.clamp(next_state_batch, min=-8.0, max=8.0)

        with torch.no_grad():
            mean_p, logvar_p = p_model.gaussian_model(train_input)
            mean_p = mean_p + train_input[:, :self.state_size]
            var_p = torch.exp(logvar_p)

        q_term = -torch.sum(torch.log(var_q), -1) / 2
        q_term = q_term.unsqueeze(-1)

        p_term = (torch.square(mean_q - mean_p) + var_q) / var_p
        p_term = -torch.sum(p_term, -1) / 2
        p_term = p_term.unsqueeze(-1)

        qf1_pi = critic(next_state_batch)

        not_done = (torch.abs(state[:, 0]) < 7.0) * (torch.abs(state[:, 1]) < 7.0)
        not_done = not_done.unsqueeze(-1)

        min_qf_pi = (qf1_pi / beta) * not_done

        # variational_loss = torch.mean(q_term)
        variational_loss = torch.mean(q_term - p_term - min_qf_pi)

        self.optimizer.zero_grad()
        variational_loss.backward()
        self.optimizer.step()

    def predict(self, inputs, batch_size=1024):
        gaussian_mean, gaussian_std = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(self.device)
            b_mean, b_std = self.gaussian_model(input)
            gaussian_mean.append(b_mean.detach().cpu().numpy())
            gaussian_std.append(b_std.detach().cpu().numpy())
        gaussian_mean = np.concatenate(gaussian_mean, axis=0)
        gaussian_std = np.concatenate(gaussian_std, axis=0)

        return gaussian_mean, gaussian_std
