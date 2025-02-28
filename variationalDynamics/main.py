import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import gymnasium as gym
from risk_environment import MultiGoalEnv
from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from simpler_model import GaussianDynamicsModel
from simpler_model import VariationalGaussianDynamicsModel
from predict_env import GaussianPredictEnv
from predict_env import VariationalGaussianPredictEnv
from sample_env import EnvSampler
from utils import exploration
from utils import train_policy_repeats
from utils import train_policy_env
from utils import evaluation
from utils import variational_directions
from utils import evaluation_plot
from utils import pre_train_model
import pickle


def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Risky",
                        help='Environment')
    parser.add_argument('--experiment_num', type=int, default= 1,
                        help='Experiment number')
    parser.add_argument('--beta', type=float, default=100.0,
                        help='Temperature parameter for KL')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Temperature parameter for KL (default: 1.0)')
    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--model', default="VMBPO",
                        help='Model Optimization')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--num_epoch', type=int, default=51, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--num_train_repeat', type=int, default=1, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=10000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--max_path_length', type=int, default=1000, metavar='A',
                        help='max length of path')
    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    return parser.parse_args()



def train(args, env_sampler, predict_env, agent, risky_agent, env_pool, model_pool):
    beta = args.beta
    exploration(args, env_sampler, env_pool, agent)
    mean_returns = []
    variance_returns = []
    cvar_min = []
    cvar_max = []
    KLs = []
    betas = []
    high_returns = []
    mid_returns = []
    low_returns = []
    wall_returns = []
    pre_train_model(env_pool, predict_env)

    for epoch_step in range(args.num_epoch):
        for i in range(args.epoch_length):
            predict_env.model.optimize_model(env_pool)

            predict_env.var_model.optimize_model(env_pool, predict_env.model, risky_agent.value, beta=beta)

            # Environment
            cur_state, action, next_state, reward, done, info = env_sampler.sample(risky_agent)
            env_pool.push(cur_state, action, reward, next_state, done)

            # Actor-learning
            train_policy_repeats(args, env_pool, predict_env, risky_agent, beta)

            # train_policy_env(args, env_pool, agent)

        # Evaluate
        mean_return, variance_return, max_return, min_return, high, mid, low, walls = evaluation(args, env_sampler, risky_agent)

        mean_returns.append(mean_return)
        variance_returns.append(variance_return)
        cvar_min.append(min_return)
        cvar_max.append(max_return)
        high_returns.append(high)
        mid_returns.append(mid)
        low_returns.append(low)
        wall_returns.append(walls)

        print(epoch_step, mean_return, variance_return, max_return, min_return)

        if epoch_step % 10 == 0 and args.experiment_num == 0:
            variational_directions(predict_env)
            plt.savefig('./Experiments/{0}_{1}_{2}_dynamics.png'.format(args.beta, args.epsilon, epoch_step+100), bbox_inches='tight')
            plt.show()
            evaluation_plot(args, env_sampler, predict_env, risky_agent, True)
            plt.savefig('./Experiments/{0}_{1}_{2}_trajectory.png'.format(args.beta, args.epsilon, epoch_step+100), bbox_inches='tight')
            plt.show()

        if (epoch_step + 1) % 10 == 0:
            save_runs(args, mean_returns, variance_returns, KLs, betas, cvar_max, cvar_min, high_returns, mid_returns, low_returns, wall_returns)


def save_runs(args, mean_returns, variance_returns, KLs, betas, max_return, min_return, high, mid, low, walls):
    with open("./Experiments/{0}/mean_return_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(mean_returns, fp)
    with open("./Experiments/{0}/variance_return_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(variance_returns, fp)
    with open("./Experiments/{0}/KL_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(KLs, fp)
    with open("./Experiments/{0}/Beta_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(betas, fp)
    with open("./Experiments/{0}/cvar_min_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(min_return, fp)
    with open("./Experiments/{0}/cvar_max_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(max_return, fp)
    with open("./Experiments/{0}/high_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(high, fp)
    with open("./Experiments/{0}/mid_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(mid, fp)
    with open("./Experiments/{0}/low_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(low, fp)
    with open("./Experiments/{0}/walls_{1}_{2}_{3}.pickle".format(args.env_name, args.beta, args.epsilon, args.experiment_num), "wb") as fp:
        pickle.dump(walls, fp)


def main(args=None):
    if args is None:
        args = readParser()

    if args.env_name == 'Risky':
        env = MultiGoalEnv()
    else:
       env = gym.make(args.env_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args, device)
    risky_agent = SAC(env.observation_space.shape[0], env.action_space, args, device)
    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)

    if (args.model == 'MBPO'):
        env_model = GaussianDynamicsModel(state_size, action_size, args.reward_size,
                                          args.pred_hidden_size,
                                          use_decay=args.use_decay)
        predict_env = GaussianPredictEnv(env_model, args.env_name)
    elif (args.model == 'VMBPO'):
        env_model = GaussianDynamicsModel(state_size, action_size, device,
                                          args.pred_hidden_size)
        var_model = VariationalGaussianDynamicsModel(state_size, action_size, device,
                                                     args.pred_hidden_size)
        predict_env = VariationalGaussianPredictEnv(env_model, var_model, args.env_name)
    else:
        predict_env = None

    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    model_pool = ReplayMemory(400000)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    train(args, env_sampler, predict_env, agent, risky_agent, env_pool, model_pool)


if __name__ == '__main__':
    main()
