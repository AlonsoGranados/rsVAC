import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class SAC(object):
    def __init__(self, num_inputs, action_size, args, device):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.device = device

        self.critic = QNetwork(num_inputs, action_size, 256).to(device=self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_size, 256).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(num_inputs,action_size, 256).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)


    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory):
        # print(memory)
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)


        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        (qf1_loss+qf2_loss).backward()
        self.critic_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)


class Reward_network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Reward_network, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, output_dim * 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        mean, log_var = torch.chunk(x, 2, dim=-1)
        log_var = torch.clamp(log_var, min=-1, max=2)
        return mean, log_var


class Reward_model():
    def __init__(self, input_dim, output_dim, device):
        self.device = device
        self.model = Reward_network(input_dim, output_dim).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, amsgrad=True)

    def optimize_model(self, args, memory):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(64)

        inputs = np.concatenate((state_batch, action_batch, next_state_batch), axis=-1)

        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        labels = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        mean, log_var = self.model(inputs)
        var = torch.exp(log_var)

        mse_loss = torch.pow(mean - labels, 2) / var
        var_loss = log_var

        total_loss = torch.mean(mse_loss + var_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def variance(self, args, memory):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(args.batch_size)

        action_one_hot_batch = np.zeros((action_batch.shape[0], 4))
        action_one_hot_batch[np.arange(action_batch.shape[0]), action_batch] = 1

        inputs = np.concatenate((state_batch, action_one_hot_batch, next_state_batch), axis=-1)

        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        mean, log_var = self.model(inputs)

        print(torch.min(log_var).item(), torch.mean(log_var).item(),torch.max(log_var).item())


class Variational_reward_model():
    def __init__(self, input_dim, output_dim, device, prior_model, beta):
        self.device = device
        self.prior_model = prior_model
        self.variational_model = Reward_network(input_dim, output_dim).to(device)
        self.optimizer = optim.AdamW(self.variational_model.parameters(), lr=0.0001, amsgrad=True)
        self.beta = beta

    def optimize_model(self, args, memory):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(64)
        inputs = np.concatenate((state_batch, action_batch, next_state_batch), axis=-1)
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        mean_q, log_var_q = self.variational_model(inputs)
        var_q = torch.exp(log_var_q)

        with torch.no_grad():
            mean_p, log_var_p = self.prior_model.model(inputs)
            var_p = torch.exp(log_var_p)

        q_term = -log_var_q

        p_term = ((torch.square(mean_q - mean_p) + var_q) / var_p)

        variational_loss = torch.mean(q_term + p_term - mean_q/self.beta)
        # variational_loss = torch.mean(q_term + p_term)

        self.optimizer.zero_grad()
        variational_loss.backward()
        self.optimizer.step()

        return variational_loss.item()


class Variational_cont():
    def __init__(self, args, n_observations, n_actions, device, env):
        self.agent = SAC(n_observations, n_actions, args, device)
        self.device = device
        self.env = env
        self.beta = args.beta
        self.reward_model = Reward_model(n_observations * 2 + n_actions, 1, device)
        self.variational_model = Variational_reward_model(n_observations * 2 + n_actions, 1, device, self.reward_model, args.beta)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.agent.policy.sample(state)
        else:
            _, _, action = self.agent.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def optimize_model(self, args, memory):
        self.reward_model.optimize_model(args, memory)
        self.variational_model.optimize_model(args, memory)

        batch_state, batch_action, r, batch_next_state, batch_done = memory.sample(args.batch_size)
        batch_done = (~batch_done).astype(int)
        inputs = np.concatenate((batch_state, batch_action, batch_next_state), axis=-1)
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mean, log_var = self.variational_model.variational_model(inputs)
            batch_reward = mean

        self.agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done))