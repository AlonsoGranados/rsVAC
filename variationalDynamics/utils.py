import matplotlib.pyplot as plt
import numpy as np
import torch
import math
def exploration(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, exploration=True)
        env_pool.push(cur_state, action, reward, next_state, done)

def evaluation(args, env_sampler, agent):
    returns = []
    high = 0
    mid = 0
    low = 0
    walls = 0
    for i in range(20):
        env_sampler.current_state = None
        done = False
        test_step = 0
        G = 0
        while (not done) and (test_step != args.max_path_length):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
            G += reward
            test_step += 1
        if done:
            if reward > 80:
                high += 1
            elif reward > 40:
                mid += 1
            elif reward < -50:
                walls += 1
            else:
                low += 1
        returns.append(G)
    print(high/20, mid/20, low/20, walls/20)
    mean_return = np.mean(returns)
    variance_return = np.var(returns)
    returns.sort()
    cvar_min = np.mean(returns[:4])
    cvar_max = np.mean(returns[-4:])
    return mean_return, variance_return, cvar_max, cvar_min, high/20, mid/20, low/20, walls/20

from matplotlib.pyplot import cm
c = cm.rainbow(np.linspace(0, 1, 10))
import matplotlib.patches as patches
def evaluation_plot(args, env_sampler, predict_env, agent, true_env):
    fig, ax = plt.subplots()

    square = patches.Rectangle((7, 7), 100, -20, color='red', facecolor='none', alpha=0.5)
    ax.add_patch(square)
    square = patches.Rectangle((-7, 7), -20, -20, color='red', facecolor='none', alpha=0.5)
    ax.add_patch(square)
    square = patches.Rectangle((-11, 7.077), 100, 20, color='red', facecolor='none', alpha=0.5)
    ax.add_patch(square)
    square = patches.Rectangle((-7, -7), 14, -20, color='green', facecolor='none', alpha=0.5)
    ax.add_patch(square)
    # square = patches.Rectangle((-7+1.4, -7), 2.8, -20, color='blue', facecolor='none', alpha=0.5)
    # ax.add_patch(square)
    # square = patches.Rectangle((-7 + 1.4 + 2.8, -7), 2.8, -20, color='yellow', facecolor='none', alpha=0.5)
    # ax.add_patch(square)
    # square = patches.Rectangle((-7 + 1.4 + 5.6, -7), 2.8, -20, color='yellow', facecolor='none', alpha=0.5)
    # ax.add_patch(square)
    # square = patches.Rectangle((-7+1.4+5.6+2.8, -7), 2.8, -20, color='blue', facecolor='none', alpha=0.5)
    # ax.add_patch(square)
    # square = patches.Rectangle((7-1.4, -7), 1.4, -20, color='green', facecolor='none', alpha=0.5)
    # ax.add_patch(square)

    # square = patches.Rectangle((-7, -7), 2, -20, color='green', facecolor='none')
    # ax.add_patch(square)
    plt.plot([-7, -7], [-7, 7], color='black')
    plt.plot([-7, 7], [7, 7], color='black')
    plt.plot([7, 7], [7, -7], color='black')
    plt.plot([7, -7], [-7, -7], color='black')

    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    for i in range(10):
        env_sampler.current_state = None
        done = False
        test_step = 0
        if true_env:
            while (not done) and (test_step != args.max_path_length):
                cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                if not done:
                    plt.plot([cur_state[0], next_state[0]], [cur_state[1],next_state[1]], color= c[i])
                test_step += 1
        else:
            cur_state, _, _, _, _, _ = env_sampler.sample(agent, eval_t=True)
            while (not done) and (test_step != args.max_path_length):
                action = agent.select_action(cur_state, eval=True)
                next_state, _, done = predict_env.step(cur_state, action, deterministic=False)
                next_state = next_state[0]
                if not done:
                    plt.plot([cur_state[0], next_state[0]], [cur_state[1],next_state[1]], color= c[i])
                cur_state = next_state
                test_step += 1
    # plt.plot([-7,-7], [-7,7])
    # plt.plot([-7, 7], [7, 7])
    # plt.plot([7, 7], [7, -7])
    # plt.plot([7, -7], [-7, -7])
    plt.xlim(-9, 9)
    plt.ylim(-9, 9)


    # plt.show()

def generate_trajectories(args, env, agent, predict_env):
    mean_return = 0
    variance_return = 0
    for i in range(5):
        current_state, _ = env.reset()
        done = False
        test_step = 0
        G = 0
        while (not done) and (test_step != args.max_path_length):
            action = agent.select_action(current_state, True)
            current_state, reward, done = predict_env.step(current_state, action)
            if reward > 150:
                plt.plot(current_state[0, 0], current_state[0, 1], 'ko')
            if 200 > reward > 100:
                plt.plot(current_state[0, 0], current_state[0, 1], 'go')
            elif reward < -50:
                plt.plot(current_state[0, 0], current_state[0, 1], 'bo')
            else:
                plt.plot(current_state[0, 0], current_state[0, 1], 'r*')
            G += reward
            test_step += 1
        mean_return += G
        variance_return += G**2
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()
    mean_return = mean_return/5
    variance_return = variance_return/5
    variance_return -= mean_return**2
    print(mean_return, variance_return)
    return mean_return, variance_return

class adaptive_beta:
    def __init__(self, beta, epsilon, device):
        self.beta = beta
        self.epsilon = epsilon
        if beta > 0:
            self.is_positive = True
            self.log_beta = torch.tensor([math.log(beta)], device=device)
        else:
            self.is_positive = False
            self.log_beta = torch.tensor([math.log(-beta)], device=device)
        self.log_beta.requires_grad = True
        self.beta_optim = torch.optim.Adam([self.log_beta], lr=0.003)
    def dual_optimization(self, KL):
        beta_loss = (self.log_beta * (self.epsilon - KL)).mean()

        self.beta_optim.zero_grad()
        beta_loss.backward()
        self.beta_optim.step()

        beta = math.exp(self.log_beta.item())
        if self.is_positive:
            return beta
        else:
            return -beta

def variational_directions(predict_env):
    # board = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            #Sample next state from model
            state = np.array([[i * 1.6 - 8, j * 1.6 - 8]])
            action = np.array([[-0.5, 0]])

            inputs = np.concatenate((state, action), axis=-1)
            model_mean, model_stds = predict_env.var_model.predict(inputs)
            next_states = state + model_mean
            next_states = np.clip(next_states, -8, 8)

            plt.arrow(state[0,0], state[0,1], next_states[0,0] - state[0,0], next_states[0,1]-state[0,1],
                                color='r', head_width=0.2, width=0.05)
            # Sample next state from model
            state = np.array([[i * 1.6 - 8, j * 1.6 - 8]])
            action = np.array([[0, 0.5]])

            inputs = np.concatenate((state, action), axis=-1)
            model_mean, model_stds = predict_env.var_model.predict(inputs)
            next_states = state + model_mean
            next_states = np.clip(next_states, -8, 8)

            plt.arrow(state[0, 0], state[0, 1], next_states[0, 0] - state[0, 0], next_states[0, 1] - state[0, 1],
                      color='k', head_width=0.2, width=0.05)
            # Sample next state from model
            state = np.array([[i * 1.6 - 8, j * 1.6 - 8]])
            action = np.array([[0.5, 0]])

            inputs = np.concatenate((state, action), axis=-1)
            model_mean, model_stds = predict_env.var_model.predict(inputs)
            next_states = state + model_mean
            next_states = np.clip(next_states, -8, 8)

            plt.arrow(state[0, 0], state[0, 1], next_states[0, 0] - state[0, 0], next_states[0, 1] - state[0, 1],
                      color='g', head_width=0.2, width=0.05)
            # Sample next state from model
            state = np.array([[i * 1.6 - 8, j * 1.6 - 8]])
            action = np.array([[0, -0.5]])

            inputs = np.concatenate((state, action), axis=-1)
            model_mean, model_stds = predict_env.var_model.predict(inputs)
            next_states = state + model_mean
            next_states = np.clip(next_states, -8, 8)

            plt.arrow(state[0, 0], state[0, 1], next_states[0, 0] - state[0, 0], next_states[0, 1] - state[0, 1],
                      color='b', head_width=0.2, width=0.05)
            # model_stds = np.exp(model_stds/2)
            # board[10-j][i] = model_stds[0,0]
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    # plt.show()

    # plt.imshow(board)
    # plt.colorbar()
    # plt.show()


def get_gradient(agent, predict_env):
    board = np.zeros((11,11))
    for i in range(11):
        for j in range(11):
            #Sample next state from model
            # state = np.array([i * 1.3 - 6.5, j * 1.3 - 6.5])
            # action = np.array([-1, 0])
            # next_states, rewards, terminals = predict_env.step(state, action, deterministic=True)
            #
            # plt.arrow(state[0], state[1], next_states[0][0] - state[0], next_states[0][1]-state[1],
            #                     color='r', head_width=0.2, width=0.05)

            #Gradient w.r.t. V(s)
            state = np.array([i * 1.6 - 8, j * 1.6 - 8])
            state = torch.FloatTensor(state).to(torch.device('cuda')).unsqueeze(0)
            # state.requires_grad = True

            # agent.value_optim.zero_grad()
            V = agent.value.forward(state)
            # loss = -V
            # loss.backward()
            #
            # N = np.sqrt(state.grad[0][0].item() ** 2 + state.grad[0][1].item() ** 2)
            #
            # plt.arrow(i * 1.3 - 6.5, j * 1.3 - 6.5, -state.grad[0][0].item() / N, -state.grad[0][1].item() / N,
            #           color='b', head_width=0.2,
            #           width=0.05)

            #Gradient Q
            # state = np.array([i * 1.3 - 6.5, j * 1.3 - 6.5])
            # state = torch.FloatTensor(state).to(torch.device('cuda')).unsqueeze(0)
            # state.requires_grad = True
            #
            # agent.value_optim.zero_grad()
            # agent.policy_optim.zero_grad()
            # _, log_pi, pi = agent.policy.sample(state)
            # Q_1, Q_2 = agent.critic.forward(state, pi)
            # Q = torch.min(Q_1, Q_2)
            # loss = -Q
            # loss.backward()
            #
            # N = np.sqrt(state.grad[0][0].item() ** 2 + state.grad[0][1].item() ** 2)
            #
            # plt.arrow(i * 1.3 - 6.5, j * 1.3 - 6.5, -state.grad[0][0].item() / N, -state.grad[0][1].item() / N,
            #           color='b', head_width=0.2,
            #           width=0.05)

            #Action plot
            # _, log_pi, pi = agent.policy.sample(state)
            #
            # plt.arrow(i * 1.3 - 6.5, j * 1.3 - 6.5, pi[0][0].item(), pi[0][1].item(),
            #           color='k', head_width=0.2,
            #           width=0.05)
            board[10-j][i] = V.item()
    # plt.xlim(-8, 8)
    # plt.ylim(-8, 8)
    # plt.show()
    plt.imshow(board)
    plt.colorbar()
    plt.show()

def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state

    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=512, holdout_ratio=0.2)

def train_variational_predict_model(env_pool, model, Q, q_c, beta):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = delta_state
    # model.train(inputs, labels, state, action, reward, next_state, done, Q, q_c, batch_size=256, holdout_ratio=0.2, beta= beta)
    model.var_model.reverse_train(inputs, model.model, Q, q_c, batch_size=1024, beta=beta)
def train_combined_predict_model(env_pool, model, Q, q_c, beta):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
    model.train(inputs, labels, state, action, reward, next_state, done, Q, q_c, batch_size=256, holdout_ratio=0.2, beta= beta)


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        action = agent.select_action(state)
        next_states, rewards, terminals = predict_env.step(state, action, deterministic=False)
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, env_pool, predict_env, agent, beta):
    for i in range(args.num_train_repeat):
        batch_state, _, _, _, _ = env_pool.sample(128)
        batch_action = agent.select_action(batch_state)
        batch_next_state, batch_reward, batch_done = predict_env.step(batch_state, batch_action, deterministic=False)
        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)

        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), predict_env, beta, KL= True)

def train_policy_env(args, env_pool, agent):
    for i in range(args.num_train_repeat):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_pool.sample(128)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), None, None)

def pre_train_model(env_pool, predict_env):
    for i in range(10000):
        if i % 1000 == 0:
            print(i)
        predict_env.model.optimize_model(env_pool)
    for var_model, model in zip(predict_env.var_model.gaussian_model.parameters(), predict_env.model.gaussian_model.parameters()):
        var_model.data.copy_(model.data)



# def resize_model_pool(args, rollout_length, model_pool):
#     rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
#     model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
#     new_pool_size = args.model_retain_epochs * model_steps_per_epoch
#
#     sample_all = model_pool.return_all()
#     new_model_pool = ReplayMemory(new_pool_size)
#     new_model_pool.push_batch(sample_all)
#
#     return new_model_pool