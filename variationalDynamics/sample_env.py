import gymnasium as gym
import numpy as np
class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length

    def sample(self, agent, eval_t=False, exploration = False):
        if self.current_state is None:
            self.current_state, _ = self.env.reset()

        cur_state = self.current_state

        if exploration:
            action = self.env.action_space.sample()
        else:
            action = agent.select_action(self.current_state, eval_t)

        next_state, reward, terminated, _, info = self.env.step(action)

        self.path_length += 1

        if terminated or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminated, info
