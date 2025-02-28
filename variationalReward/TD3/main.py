import gymnasium as gym
import numpy as np
import torch
from TD3 import TD3Agent
import utils
import os
import sys
sys.path.append('..')
from utils import eval_model
from utils import get_save_dir

import argparse
def readParser():
    parser = argparse.ArgumentParser(description='MVPI')
    parser.add_argument('--env_name', default='Ant-v4', help='learning rate')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=4000,
                        help='total number of epochs')
    parser.add_argument('--experiment_num', default=10,
                        help='number of experiment')
    parser.add_argument('--num_episode', default=30,
                        help='total number of episodes per epoch')
    parser.add_argument('--eval_int', default=20,
                        help='number of epochs before evaluation')
    parser.add_argument('--max_episode_length', default=500,
                        help='maximum number of steps per episode')
    return parser.parse_args()


def experiment(args, env, eval_env, policy, replay_buffer, save_dir):
    max_timesteps = int(1e6)
    start_timesteps = 25e3
    exploration_noise = 0.1
    batch_size = 256
    eval_freq = 1e4

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    state, _ = env.reset()
    done = False
    episode_reward, episode_timesteps, episode_num = 0, 0, 0

    test_score_lst = []
    best_mean = -10000
    best_mean_variance = -10000

    eval_r_lst, eval_right_lst = [],[]

    for t in range(max_timesteps):
        episode_timesteps += 1

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * exploration_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, done, _ , info = env.step(action)
        # done_bool = float(done) if episode_timesteps < 500 else 1
        # store data in buffer
        replay_buffer.add(state, action, next_state, reward, done)
        # store reward in online reward buffer
        policy.online_rewards.append(reward)

        state = next_state
        episode_reward += reward

        # train after collecting sufficient data
        if t>=start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done or episode_timesteps == 500:
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t+1) % eval_freq == 0:
            eval_r, eval_right = eval_model(args, eval_env, policy)
            eval_r_mean = eval_r.mean()
            eval_right_mean = eval_right.mean()
            print((t+1)/1000, 'eval return:', eval_r_mean, eval_right_mean)

            eval_r_lst.append(eval_r_mean)
            eval_right_lst.append(eval_right_mean)
            # eval_len_lst.append(eval_len)
            # eval_xpos_vio_lst.append(eval_xpos_vio)

            if eval_r_mean > best_mean:
                best_mean = eval_r_mean
                policy.save_best(risk=False)

            with open(save_dir + 'eval_r.npy', 'wb') as f:
                np.save(f, np.array(eval_r_lst))
            with open(save_dir + 'eval_right.npy', 'wb') as f:
                np.save(f, np.array(eval_right_lst))
            # with open(save_dir + 'eval_xpos_vio.npy', 'wb') as f:
            #     np.save(f, np.array(eval_xpos_vio_lst))

def main():
    net_width = 256

    gamma = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    args = readParser()
    lr = args.lr
    env_id = args.env_name
    # env = gym.make(env_id)
    env = gym.make(env_id, exclude_current_positions_from_observation=False)
    # eval_env = gym.make(env_id)
    eval_env = gym.make(env_id, exclude_current_positions_from_observation=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # create save dir
    save_dir = get_save_dir(env_id, 0.2, args.lr, args.experiment_num)
    os.makedirs(save_dir, exist_ok=True)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "net_width": net_width,
        "max_action": max_action,
        "lr": lr,
        "discount": gamma,
        "tau": tau,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
        "policy_freq": policy_freq,
        "save_dir": save_dir,
    }

    print('env:', env_id, 'lr:', lr, 'seed:', args.experiment_num)

    policy = TD3Agent(**kwargs)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    experiment(args, env, eval_env, policy, replay_buffer, save_dir)

if __name__ == '__main__':
    main()

