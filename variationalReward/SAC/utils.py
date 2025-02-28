import numpy as np

def exploration(env_sampler, memory, num_steps):
    for i in range(num_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(None, exploration=True)
        memory.push(cur_state, action, reward, next_state, done)
    env_sampler.current_state = None
    env_sampler.path_length = 0
    print('Exploration completed')


def evaluation(args, env_sampler, agent):
    returns = []
    land_right = 0
    total = 0
    for i in range(5):
        env_sampler.current_state = None
        env_sampler.path_length = 0
        done = False
        test_step = 0
        G = 0
        while (not done) and (test_step != args.max_path_length):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True, clipping = False)
            if args.env_name == 'HalfCheetah-v4':
                if info['x_position'] < -3:
                    land_right += 1
            if args.env_name == 'Swimmer-v4':
                if info['x_position'] > 0.5:
                    land_right += 1
            if args.env_name == 'Ant-v4':
                if info['x_position'] > 0.5:
                    land_right += 1

            G += reward
            test_step += 1
        total += test_step
        returns.append(G)
    mean_return = np.mean(returns)
    return mean_return, land_right/total

