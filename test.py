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
import numpy as np

import PIL
from PIL import Image

import cv2


torch.autograd.set_detect_anomaly(True)


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make("HalfCheetahVel-v2", render_mode="human")
    env.close()
    input_size = reduce(mul, env.observation_space.shape, 1)

    output_size = reduce(mul, env.action_space.shape, 1)
    policy = NormalPolicy(
        input_size,
        output_size,
        hidden_layer_size=tuple(config.hidden_sizes),
        activation=config.activation,
    )
    with open(args.policy, "rb") as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    baseline = LinearFeatureBaseline(input_size)

    runner = Runner(
        config.env_name,
        config.env_kwargs,
        config.fast_batch_size,
        policy_net=policy,
        env=env,
        baseline=baseline,
    )

    config.num_batches = args.num_batches

    num_iterations = 0
    logs = {"tasks": []}
    train_returns, valid_returns = [], []
    progress_bar = tqdm(total=config.num_batches)
    for batch in range(config.num_batches):
        tasks = runner.sample_tasks(config.meta_batch_size)
        trains, valids = runner.sample(
            tasks=tasks,
            num_steps=config.num_steps,
            fast_lr=config.fast_lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        # print('Sample collected')
        logs["tasks"].extend(tasks)

        num_iterations += sum(
            sum(episode.lengths) for episode_list in trains for episode in episode_list
        )
        num_iterations += sum(sum(episode.lengths) for episode in valids)

        train_returns.append(rl_utils.get_train_returns(trains))
        valid_returns.append(rl_utils.get_valid_returns(valids))
        progress_bar.write(f'Train returns: {train_returns[-1].sum()} Valid returns: {valid_returns[-1].sum()}')
        progress_bar.update(1)
    logs["train_returns"] = np.concatenate(train_returns, axis=0)
    logs["valid_returns"] = np.concatenate(valid_returns, axis=0)

    # print(logs)


    ##################### Video recording ##################
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = None

    env = gym.make("HalfCheetahVel-v2", render_mode="rgb_array")
    # env.close()
    input_size = reduce(mul, env.observation_space.shape, 1)

    output_size = reduce(mul, env.action_space.shape, 1)
    observation, _ = env.reset()
    rewards_total = 0
    for _ in range(1000):
        frame = env.render()
        if video is None:
            video_file = os.path.join(os.getcwd(),'output','test_with_smapling_30.mp4')
            print(video_file, frame.shape)
            video = cv2.VideoWriter(video_file, fourcc, float(fps), (frame.shape[1], frame.shape[0]))
        video.write(frame)
        observations_tensor = torch.from_numpy(observation).unsqueeze(0)
        pi = policy(observations_tensor)
        actions_tensor = pi.sample()
        actions = actions_tensor.squeeze().cpu().numpy()
        # actions = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(actions)
        rewards_total+=reward

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    video.release()
    # video.close()
    print(rewards_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinforcement learning with "
        "Model-Agnostic Meta-Learning (MAML) - Test"
    )

    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        default="output_no_batch/policy.th",
        help="path to the policy checkpoint",
    )

    # Evaluation
    evaluation = parser.add_argument_group("Evaluation")
    evaluation.add_argument(
        "--num-batches", type=int, default=10, help="number of batches (default: 10)"
    )
    evaluation.add_argument(
        "--meta-batch-size",
        type=int,
        default=40,
        help="number of tasks per batch (default: 40)",
    )

    # Miscellaneous
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--output",
        type=str,
        required=True,
        default="output",
        help="name of the output folder (default: output)",
    )

    misc.add_argument("--seed", type=int, default=12, help="random seed")

    misc.add_argument(
        "--use-cuda",
        action="store_true",
        help="use cuda (default: false, use cpu). WARNING: Full upport for cuda "
        "is not guaranteed. Using CPU is encouraged.",
    )

    args = parser.parse_args()
    args.device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"

    main(args)
