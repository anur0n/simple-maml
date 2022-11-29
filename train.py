import gym
from policy import NormalPolicy
import envs
import os
from tqdm import trange
from tqdm import tqdm
import torch
from functools import reduce
from operator import mul

import torch.nn.functional as F
import config
from runner import Runner
from maml_trpo import MAMLTRPO
import rl_utils
import json
import argparse
from baseline import LinearFeatureBaseline
# for i in range(1000):
    # observation, reward, terminated, info
    # res = env.step(env.action_space.sample())
    # print('done: ', res)

    # print(i)
    # if terminated or truncated:
    #     observation, info = env.reset()

# env.close()

# policy = NormalPolicy()
# policy.share_memory()


torch.autograd.set_detect_anomaly(True)


def main(args):
    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        # config_filename = os.path.join(args.output_folder, 'config.json')

        # with open(config_filename, 'w') as f:
        #     config.update(vars(args))
        #     json.dump(config, f, indent=2)
        
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make("HalfCheetahVel-v2", render_mode='human')
    env.close()
    input_size = reduce(mul, env.observation_space.shape, 1)
    

    output_size = reduce(mul, env.action_space.shape, 1)
    policy = NormalPolicy(input_size, output_size, hidden_layer_size=tuple(config.hidden_sizes), activation=config.activation)
    policy.share_memory()
    
    baseline = LinearFeatureBaseline(input_size)

    runner = Runner(config.env_name, config.env_kwargs, config.fast_batch_size, policy_net=policy, env=env, baseline=baseline)
    meta_learner = MAMLTRPO(policy=policy, fast_rl=config.fast_lr, first_order=config.first_order)
    

    num_iterations = 0
    progress_bar = tqdm(total=config.num_batches)
    for batch in range(config.num_batches):
        tasks = runner.sample_tasks(config.meta_batch_size)
        trains, valids = runner.sample(tasks=tasks, num_steps=config.num_steps, fast_lr=config.fast_lr, gamma=config.gamma, gae_lambda=config.gae_lambda)
        # print('Sample collected')
        logs = meta_learner.step(trains=trains, valids=valids, max_kl=config.max_kl, cg_iters=config.cg_iters, cg_damping=config.cg_damping, ls_max_steps=config.ls_max_steps, ls_backtrack_ratio=config.ls_backtrack_ratio)
        # print('Meta updated')
        num_iterations += sum(sum(episode.lengths) for episode_list in trains for episode in episode_list)
        num_iterations += sum(sum(episode.lengths) for episode in valids)

        logs.update(tasks=tasks, num_iterations=num_iterations, train_returns=rl_utils.get_train_returns(trains), valid_returns=rl_utils.get_valid_returns(valids))
        
        progress_bar.write(f"Loss before: {logs['loss_before']} KL before: {logs['kl_before']} Num iterations: {logs['num_iterations']}")
        if 'loss_after' in logs.keys():
            progress_bar.write(f"Loss after: {logs['loss_after']} KL After: {logs['kl_after']}")
        # print('#########################################################################')

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)
        progress_bar.update(1)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed')

    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
