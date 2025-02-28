import numpy as np
import torch

class PredictEnv:
    def __init__(self, model, env_name):
        self.model = model
        self.env_name = env_name

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v4":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v4":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]

        inputs = np.concatenate((obs, act), axis=-1)

        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)

        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(
                size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape

        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)

        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        # info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals

class GaussianPredictEnv:
    def __init__(self, model, env_name):
        self.model = model
        self.env_name = env_name

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v4":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v4":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == 'LunarLander-v2':
            x = next_obs[:, 0]
            y = next_obs[:, 1]
            not_done = (x > -1.0) \
                       * (x < 1.0) \
                       * (y > -0.03054)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == 'Risky':
            x = next_obs[:, 0]
            y = next_obs[:, 1]
            not_done = ((x > -7.0)
                        * (x < 7.0)
                        * (y > -7.0)
                        *(y < 7.0))
            done = ~not_done
            done = done[:, None]
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)

        model_means, model_vars = self.model.predict(inputs)

        model_means[:, 1:] += obs
        model_stds = np.sqrt(model_vars)

        if deterministic:
            samples = model_means
        else:
            samples = model_means + np.random.normal(size=model_means.shape) * model_stds

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)
        return next_obs, rewards, terminals


class VariationalGaussianPredictEnv:
    def __init__(self, model, var_model, env_name):
        self.model = model
        self.var_model = var_model
        self.env_name = env_name

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v4":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v4":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == 'LunarLander-v2':
            x = next_obs[:, 0]
            y = next_obs[:, 1]
            not_done = (x > -1.0) \
                       * (x < 1.0) \
                       * (y > -0.03054)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == 'Risky':
            x = obs[:, 0]
            y = obs[:, 1]
            not_done = ((x > -7.0)
                        * (x < 7.0)
                        * (y > -7.0)
                        *(y < 7.0))
            done = ~not_done
            done = done[:, None]

            rewards = -0.1 + (y < -7.0) * (abs(x) < 7.0) * (100/7)*np.abs(x) + (abs(x) > 7.0) * -100 + (abs(x) < 7.0) * (y > 7.0) * -100
            # rewards = -0.1 + (abs(x) > 7.0) * -100 + (y < -7.0) * (abs(x) < 7.0) * (abs(x) > 6.0) * 200 + (y < -7.0) * (abs(x) < 6.0) * (abs(x) > 3.0) * 100 + (y < -7.0) * (abs(x) < 3.0) * 50

            return done, rewards

    def evaluate_log_ratio(self, obs, act):
        inputs = np.concatenate((obs, act), axis=-1)

        q_means, q_log_vars = self.var_model.predict(inputs)
        q_vars = np.exp(q_log_vars)
        q_means += obs

        p_means, p_log_vars = self.model.predict(inputs)
        p_vars = np.exp(p_log_vars)
        p_means += obs

        next_obs = q_means + np.random.normal(size=q_means.shape) * np.sqrt(q_vars)

        log_q = -(np.log(q_vars).sum(-1) + (np.power(next_obs - q_means, 2) / q_vars).sum(-1)) / 2

        log_p = -(np.log(p_vars).sum(-1) + (np.power(next_obs - p_means, 2) / p_vars).sum(-1)) / 2

        return np.mean(log_q - log_p)

    def KL(self, obs, act, next_obs):
        with torch.no_grad():
            inputs = torch.cat((obs, act), dim=1)
            q_means, q_log_vars = self.var_model.gaussian_model(inputs)
            q_vars = torch.exp(q_log_vars)
            q_means += obs
            p_means, p_log_vars = self.model.gaussian_model(inputs)
            p_vars = torch.exp(p_log_vars)
            p_means += obs

            log_q = -(torch.log(q_vars).sum(-1) + (torch.square(next_obs - q_means) / q_vars).sum(-1)) / 2

            log_p = -(torch.log(p_vars).sum(-1) + (torch.square(next_obs - p_means) / p_vars).sum(-1)) / 2

        return torch.mean(log_q - log_p)

    def step(self, obs, act, deterministic=False):

        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
        inputs = np.concatenate((obs, act), axis=-1)
        model_means, model_log_var = self.var_model.predict(inputs)
        model_means += obs
        model_stds = np.exp(model_log_var/2)
        # model_means, model_stds = self.model.predict(inputs)

        if deterministic:
            next_obs = model_means
        else:
            next_obs = model_means + np.random.normal(size=model_means.shape) * model_stds

        next_obs = np.clip(next_obs, -8.0, 8.0)
        terminals, rewards = self._termination_fn(self.env_name, obs, act, next_obs)
        # rewards = rewards.reshape((-1,1))

        return next_obs, rewards, terminals
