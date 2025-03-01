import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class RiskyEnv(gym.Env):
    def __init__(self, init_sigma=0.1):
        self.dynamics = PointDynamics(dim=2, sigma=0.5)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.reset()
        self.observation = None

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': []}
        self.spec = None

        super().__init__()

    def reset(self):
        unclipped_observation = self.init_mu + self.init_sigma * \
                                np.random.normal(size=self.dynamics.s_dim)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        self.observation = np.clip(unclipped_observation, o_lb, o_ub)
        return self.observation, 0

    @property
    def observation_space(self):
        return Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=None,
            dtype=np.float32
        )

    @property
    def action_space(self):
        return Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,),
            dtype=np.float32
        )


    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = self.action_space.low, self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation, action)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high

        reward = -0.1
        done = False
        if self.observation[0] > o_ub[0] or self.observation[0] < o_lb[0]:
            done = True
            reward = -100
        elif self.observation[1] > o_ub[0]:
            done = True
            reward = -100
        elif self.observation[1] < o_lb[0]:
            done = True
            reward = 100*(np.abs(self.observation[0])/7)
        self.observation = np.copy(next_obs)
        return next_obs, reward, done, done, {'pos': next_obs}

class PointDynamics(object):
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
                     np.random.normal(size=self.s_dim)
        return state_next
