import numpy as np

from gym.vector import SyncVectorEnv as SyncVectorEnv_
from gym.vector.utils import concatenate, create_empty_array

from copy import deepcopy

class SyncVectorEnv(SyncVectorEnv_):
    def __init__(self, env_fns, observation_space, action_space):
        super(SyncVectorEnv, self).__init__(env_fns, observation_space, action_space)

        for env in self.envs:
            if not hasattr(env.unwrapped, 'reset_task'):
                raise ValueError('The environment provided is not a '
                                 'meta-learning environment. It does not have '
                                 'the method `reset_task` implemented.')

    @property
    def terminateds(self):
        return self._terminateds
    
    @property
    def truncateds(self):
        return self._truncateds

    def reset_task(self, task):
        for env in self.envs:
            env.unwrapped.reset_task(task)


    def step_wait(self):
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        batch_ids = []
        num_action = len(list(deepcopy(self._actions)))
        # print('num action: ', num_action, ' # of envs: ', len(self.envs))

        j = 0
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            if self._terminateds[i] or self._truncateds[i]:
                # print('terminated: ', self._terminateds[i], ' truncateds: ', self._truncateds[i])
                continue

            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            # if self._terminateds[i] or self._truncateds[i]:
            #     old_observation, old_info = observation, info
            #     observation, info = env.reset()
            #     info["final_observation"] = old_observation
            #     info["final_info"] = old_info
            batch_ids.append(i)
            
            if not (self._terminateds[i] or self._truncateds[i]):
                observations.append(observation)
                infos = self._add_info(infos, info, i)
            else:
                # print('After action >> terminated: ', self._terminateds[i], ' truncateds: ', self._truncateds[i], infos)
                pass
            j += 1
        # print('j: ', j)
        assert j == num_action
        # print(len(observations))
        if observations:
            self.observations = concatenate(
                self.single_observation_space, observations, self.observations
            )
        else:
            self.observations = None

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            {'batch_ids': batch_ids, 'infos': infos},
        )