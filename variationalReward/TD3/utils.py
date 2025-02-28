import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
def eval_model(args, env, agent, n_episodes=20):
	return_lst = []
	# ep_length_lst = []
	# xpos_vio_lst = []

	right_lst = []

	m = 0
	for _ in range(n_episodes):
		s, _ = env.reset()
		done = False
		ep_r, total_step, xpos_vio = 0, 0, 0
		while True:
			with torch.no_grad():
				a = agent.select_action(np.array(s))
			s_prime, r, done, _, info = env.step(a)

			xpos = info['x_position']
			if args.env_name == 'HalfCheetah-v4':
				if xpos < -3:
					xpos_vio += 1
			if args.env_name == 'Swimmer-v4':
				if xpos > 0.5:
					xpos_vio += 1
			if args.env_name == 'InvertedPendulum-v4':
				if xpos > 0.01:
					xpos_vio += 1
			ep_r += r
			total_step += 1

			if total_step == args.max_episode_length:
				done = True
			if done:
				break

			s = s_prime

		return_lst.append(ep_r)
		right_lst.append(xpos_vio / total_step)
		# ep_length_lst.append(total_step)
		# xpos_vio_lst.append(xpos_vio)
		m += xpos_vio/ total_step

	return np.array(return_lst), np.array(right_lst)


def get_save_dir(env_id, lam, lr, seed):
	save_dir = "./save/" + env_id + '/lam=' + str(lam)
	save_dir += "/lr=" + str(lr) + "/seed=" + str(seed) + "/"
	return save_dir