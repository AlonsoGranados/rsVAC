import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim)
        )

        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.max_action * torch.tanh(x)
        return mu

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Critic, self).__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

    def Q1(self, state, action):
        x = torch.cat((state, action), 1)
        q1 = self.q1_net(x)
        return q1

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

    def optimize_model(self, memory):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = memory.sample(64)
        inputs = torch.cat((state_batch, action_batch, next_state_batch), dim = 1)
        labels = reward_batch
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
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = memory.sample(args.batch_size)

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

    def optimize_model(self, memory):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = memory.sample(64)
        inputs = torch.cat((state_batch, action_batch, next_state_batch), dim = 1)

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



class TD3Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            net_width,
            max_action,
            lr,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            save_dir = None,
    ):
        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.device = device
        self.critic = Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.save_dir = save_dir

        self.reward_model = Reward_model(state_dim * 2 + action_dim, 1, device)
        self.variational_model = Variational_reward_model(state_dim * 2 + action_dim, 1, device, self.reward_model, -8.0)

        self.total_it = 0
        self.online_rewards = deque(maxlen=int(1e4))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.reward_model.optimize_model(replay_buffer)
        self.variational_model.optimize_model(replay_buffer)
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        inputs = torch.cat((state, action, next_state), dim = 1)

        with torch.no_grad():
            mean, log_var = self.variational_model.variational_model(inputs)
            # print(reward, mean)
            reward = mean
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            
            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, n_stp):
        save_critic_name = self.save_dir + '/stp_{}_critic.pkl'.format(n_stp)
        save_critic_opt_name = self.save_dir + '/stp_{}_critic_opt.pkl'.format(n_stp)
        torch.save(self.critic.state_dict(), save_critic_name)
        torch.save(self.critic_optimizer.state_dict(), save_critic_opt_name)
        
        save_actor_name = self.save_dir + '/stp_{}_actor.pkl'.format(n_stp)
        save_actor_opt_name = self.save_dir + '/stp_{}_actor_opt.pkl'.format(n_stp)
        torch.save(self.actor.state_dict(), save_actor_name)
        torch.save(self.actor_optimizer.state_dict(), save_actor_opt_name)

    def save_best(self, risk=True):
        if risk:
            save_critic_name = self.save_dir + '/best_critic_var.pkl'
            torch.save(self.critic.state_dict(), save_critic_name)
            save_actor_name = self.save_dir + '/best_actor_var.pkl'
            torch.save(self.actor.state_dict(), save_actor_name)
        else:
            save_critic_name = self.save_dir + '/best_critic_mean.pkl'
            torch.save(self.critic.state_dict(), save_critic_name)
            save_actor_name = self.save_dir + '/best_actor_mean.pkl'
            torch.save(self.actor.state_dict(), save_actor_name)

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

