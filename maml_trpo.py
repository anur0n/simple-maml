import torch

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from utils import weighted_mean, detach_distribution, to_numpy, vector_to_parameters
from rl_utils import reinforce_loss
import config
import numpy as np


class MAMLTRPO(object):
    def __init__(self, policy, fast_rl=0.5, first_order=False, device="cpu"):
        self.device = device
        self.policy_net = policy
        self.policy_net.to(self.device)
        self.fast_rl = fast_rl
        self.first_order = first_order

    def adapt(self, train_steps, first_order=None, print_log=False):
        if first_order is None:
            first_order = self.first_order

        params = None
        for i, step in enumerate(train_steps):
            if print_log:
                print("------------------------------------------------------")
            inner_loss = reinforce_loss(
                self.policy_net, step, params=params, print_log=print_log
            )
            if i > 0 and print_log:
                # print(i, ':', 'Is params nan', params.values(), 'Inner loss: ', inner_loss)
                print(i, ":", "Inner loss: ", inner_loss)
            params = self.policy_net.update_params(
                inner_loss,
                params=params,
                step_size=self.fast_rl,
                first_order=first_order,
            )
            if print_log:
                print("#######################################################")

        return params

    def surrogate_loss(self, train, valid, old_policy=None):
        first_order = (old_policy is not None) or self.first_order
        params = self.adapt(train_steps=train, first_order=first_order, print_log=False)

        with torch.set_grad_enabled(old_policy is None):
            valid_episodes = valid
            
            policy = self.policy_net(valid_episodes.observations.view((-1, *valid_episodes.observation_shape)), params=params)

            if old_policy is None:
                old_policy = detach_distribution(policy)


            log_probs = policy.log_prob(
                valid_episodes.actions.view((-1, *valid_episodes.action_shape)) + config.epsilon
            )
            log_probs = log_probs.view(len(valid_episodes), valid_episodes.batch_size)

            log_ratio = log_probs - old_policy.log_prob(valid_episodes.actions.view((-1, *valid_episodes.action_shape)) + config.epsilon).view(len(valid_episodes), valid_episodes.batch_size)
            ratio = torch.exp(log_ratio)

            losses = -weighted_mean(
                ratio * valid_episodes.advantages, lengths=valid_episodes.lengths
            )
            kls = weighted_mean(
                kl_divergence(policy, old_policy).view(len(valid_episodes), valid_episodes.batch_size), lengths=valid_episodes.lengths
            )

        return losses.mean() + config.epsilon, kls.mean(), old_policy

    def step(
        self,
        trains,
        valids,
        max_kl=1e-3,
        cg_iters=10,
        cg_damping=1e-2,
        ls_max_steps=10,
        ls_backtrack_ratio=0.5,
    ):
        num_tasks = len(trains)

        logs = {}

        # print('num of tasks: ', num_tasks)
        # Compute the surrogate loss
        old_losses, old_kls, old_policies = [], [], []
        for (train, valid) in zip(trains, valids):
            old_l, old_kl_tmp, old_p = self.surrogate_loss(
                train, valid, old_policy=None
            )
            old_losses.append(old_l)
            old_kls.append(old_kl_tmp)
            old_policies.append(old_p)
            # print([self.surrogate_loss(train, valid, old_policy=None) for (train, valid) in zip(zip(*trains), valids)])

        old_loss = sum(old_losses) / num_tasks
        # print('Meta old loss: ', old_loss, ' num of tasks: ', num_tasks)
        grads = torch.autograd.grad(
            old_loss, self.policy_net.parameters(), retain_graph=True
        )
        # print('Meta grads: ', grads)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl, damping=cg_damping)
        step_dir = self.conjugate_gradient(
            hessian_vector_product, grads, cg_iters=cg_iters
        )

        logs["loss_before"] = to_numpy(old_loss)
        logs["kl_before"] = to_numpy(old_kl)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(
            step_dir, hessian_vector_product(step_dir, retain_graph=False)
        )
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = step_dir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy_net.parameters())

        # Line search
        step_size = 1.0

        for _ in range(ls_max_steps):
            vector_to_parameters(
                old_params - step_size * step, self.policy_net.parameters()
            )
            losses, kls = [], []
            for (train, valid, old_policy) in zip(trains, valids, old_policies):
                l, kl_tmp, p = self.surrogate_loss(train, valid, old_policy=old_policy)
                losses.append(l)
                kls.append(kl_tmp)

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs["loss_after"] = to_numpy(sum(losses) / num_tasks)
                logs["kl_after"] = to_numpy(kl)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy_net.parameters())
        return logs

    ##########################UTILITY FUNCTIONS######################################################
    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(
                grad_kl_v, self.policy_net.parameters(), retain_graph=retain_graph
            )
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def conjugate_gradient(self, f_Ax, b, cg_iters=10, residual_tol=1e-10):
        p = b.clone().detach()
        r = b.clone().detach()
        x = torch.zeros_like(b).double()
        rdotr = torch.dot(r, r)

        for i in range(cg_iters):
            z = f_Ax(p).detach()
            v = rdotr / torch.dot(p, z)
            x += v * p
            r -= v * z
            newdotr = torch.dot(r, r)
            mu = newdotr / rdotr
            p = r + mu * p
            rdotr = newdotr
            if rdotr.item() < residual_tol:
                break
        return x.detach()
