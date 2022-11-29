from utils import weighted_mean
import numpy as np
from utils import to_numpy
import torch
import config


def reinforce_loss(policy_net, episodes, params=None, print_log=False):
    pi = policy_net(
        episodes.observations.view((-1, *episodes.observation_shape)),
        params=params,
        print_log=print_log,
    )

    log_probs = pi.log_prob(
        episodes.actions.view((-1, *episodes.action_shape)) + config.epsilon
    )
    log_probs = log_probs.view(len(episodes), episodes.batch_size)
    if print_log:
        print(
            "Reinforce loss: log probs: ",
            log_probs,
            " log_prob * advantages: ",
            log_probs * episodes.advantages,
            "Actions: ",
            torch.exp(pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))),
            "Rewards : ",
            episodes.rewards,
            "Advantages: ",
            episodes.advantages,
        )
    losses = -weighted_mean(
        log_probs * episodes.advantages, lengths=episodes.lengths, print_log=print_log
    )

    return losses.mean()


def get_valid_returns(episodes):
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])


def get_train_returns(episodes):

    return to_numpy(
        [
            episode.rewards.sum(dim=0)
            for episode_list in episodes
            for episode in episode_list
        ]
    )
