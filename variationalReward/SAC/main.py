import gymnasium as gym
import argparse
import numpy as np
import torch
from replay_memory import ReplayMemory
from sample_env import EnvSampler
from Algorithms.Variational_continuous import Variational_cont
from utils import evaluation
from utils import exploration
import matplotlib.pyplot as plt
def readParser():
    parser = argparse.ArgumentParser(description='DZN')
    parser.add_argument('--env_name', default="Ant-v4",
                        help='Environment')
    parser.add_argument('--agent', default="SAC",
                        help='Network')
    parser.add_argument('--gamma', default=0.99,
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--experiment_num', type=int, default = 10,
                        help='experiment number')
    parser.add_argument('--tau', default=0.005,
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--beta', default=1.0,
                        help='Risk parameter')
    parser.add_argument('--alpha', default=0.2,
                        help='Risk parameter')
    parser.add_argument('--lr', default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--experience_replay_size', default=1000000,
                        help='size of experience buffer')
    parser.add_argument('--epoch_length', default=1000,
                        help='steps per epoch')
    parser.add_argument('--batch_size', default=128,
                        help='batch size for training policy')
    parser.add_argument('--max_path_length', default=500,
                        help='number of episodes per epoch')
    parser.add_argument('--init_exploration_steps', default=10000,
                        help='number of episodes per epoch')
    parser.add_argument('--num_epoch', default=1000,
                        help='total number of epochs')
    return parser.parse_args()


def experiment(args, env_sampler, trainer, memory):
    # test_losses = []
    # test_squared = []
    G = []
    right_r = []
    #Collect initial data
    exploration(env_sampler, memory, args.init_exploration_steps)

    #Training
    for epochs in range(args.num_epoch):
        for i in range(args.epoch_length):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(trainer)
            memory.push(cur_state, action, reward, next_state, done)
            trainer.optimize_model(args, memory)

        if (epochs+1)%10 == 0:
            mean_return, right = evaluation(args, env_sampler, trainer)
            G.append(mean_return)
            right_r.append(right)
            print(epochs, mean_return, right)



        if (epochs + 1) % 10 == 0:
            np.save('./Experiments/{0}/{1}/return_{2}_{3}'.format(args.env_name, args.agent, args.beta, int(args.experiment_num)), G)
            np.save('./Experiments/{0}/{1}/right_{2}_{3}'.format(args.env_name, args.agent, args.beta, int(args.experiment_num)), right_r)
            # np.save('./Experiments/{0}/{1}/test_{2}_{3}'.format(args.env_name, args.agent, int(args.experiment_num),
            #                                                     args.beta), test_losses)
            # np.save('./Experiments/{0}/{1}/squared_{2}_{3}'.format(args.env_name, args.agent, int(args.experiment_num),
            #                                                        args.beta), test_squared)
    print('Complete')


def main(args = None):
    if args is None:
        args = readParser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # env = gym.make(args.env_name)
    env = gym.make(args.env_name, exclude_current_positions_from_observation=False)

    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    memory = ReplayMemory(args.experience_replay_size)

    trainer = Variational_cont(args, n_observations, n_actions, device, env)
    experiment(args, env_sampler, trainer, memory)

if __name__ == '__main__':
    main()