import gym

import torch

from episode import BatchEpisodes
from datetime import datetime, timezone
import time
from copy import deepcopy
from rl_utils import reinforce_loss
from envs.sync_vector_env import SyncVectorEnv

def make_env(env_name, env_kwargs={}, seed=None):
    def _make_env():
        env = gym.make(env_name, **env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return _make_env


class Runner(object):
    def __init__(self, env_name, env_kwargs, batch_size, policy_net, baseline, seed=None, env=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy_net = policy_net
        self.seed = seed
        self.baseline = baseline

        if env is None:
            env = gym.make(env_name, **env_kwargs)
        self.env = env
        if hasattr(env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        env_fns = [make_env(env_name, env_kwargs=env_kwargs)
                   for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_fns,
                                  observation_space=self.env.observation_space,
                                  action_space=self.env.action_space)
        self.closed = False
        # self.env.unwrapped.sample_tasks()

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)
    
    def sample(self, tasks, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        trains = []
        valids = []
        for index, task in enumerate(tasks):
            self.envs.reset_task(task)
            train, valid = self._sample(index, num_steps, fast_lr, gamma, gae_lambda, device)
            trains.append(train)
            valids.append(valid)

        return trains, valids

    def _sample(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        # Sample the training trajectories with the initial policy and adapt the
        # policy to the task, based on the REINFORCE loss computed on the
        # training trajectories. The gradient update in the fast adaptation uses
        # `first_order=True` no matter if the second order version of MAML is
        # applied since this is only used for sampling trajectories, and not
        # for optimization.
        params = None
        train_episodes_all = []
        for step in range(num_steps):
            train_episodes = self.create_episodes(params=params,
                                                  gamma=gamma,
                                                  gae_lambda=gae_lambda,
                                                  device=device)
            train_episodes.log('_enqueueAt', datetime.now(timezone.utc))
            # QKFIX: Deep copy the episodes before sending them to their
            # respective queues, to avoid a race condition. This issue would 
            # cause the policy pi = policy(observations) to be miscomputed for
            # some timesteps, which in turns makes the loss explode.
            # self.train_queue.put((index, step, deepcopy(train_episodes)))
            train_episodes_all.append(deepcopy(train_episodes))

            loss = reinforce_loss(self.policy_net, train_episodes, params=params)
            params = self.policy_net.update_params(loss, params=params, step_size=fast_lr, first_order=True)

        # Sample the validation trajectories with the adapted policy
        valid_episodes = self.create_episodes(params=params, gamma=gamma, gae_lambda=gae_lambda, device=device)
        valid_episodes.log('_enqueueAt', datetime.now(timezone.utc))

        return train_episodes_all, valid_episodes

    def create_episodes(self, params=None, gamma=0.95, gae_lambda=1.0, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size,
                                 gamma=gamma,
                                 device=device)
        episodes.log('_createdAt', datetime.now(timezone.utc))
        # episodes.log('process_name', self.name)

        t0 = time.time()
        for item in self.sample_trajectories(params=params):
            episodes.append(*item)
        episodes.log('duration', time.time() - t0)

        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline,
                                    gae_lambda=gae_lambda,
                                    normalize=True)
        return episodes

    def sample_trajectories(self, params=None):
        observations, infos = self.envs.reset()
        # print(observations.dtype)
        
        steps = []
        with torch.no_grad():
            cnt = 0
            while not (self.envs.terminateds.all() or self.envs.truncateds.all()):
                cnt += 1
                # print('Count: ', cnt)
                observations_tensor = torch.from_numpy(observations).to(torch.float64)
                # print(observations_tensor.dtype)
                policy = self.policy_net(observations_tensor, params=params)
                actions_tensor = policy.sample()
                actions = actions_tensor.cpu().numpy()

                new_observations, rewards, _, _, infos = self.envs.step(actions)
                batch_ids = infos['batch_ids']
                steps.append((observations, actions, rewards, batch_ids))
                observations = new_observations
        return steps

    