import torch
import numpy as np

from torch.distributions import Categorical, Independent, Normal
from torch.nn.utils.convert_parameters import _check_param_device

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (tuple, list)):
        return np.stack([to_numpy(t) for t in tensor], axis=0)
    else:
        raise NotImplementedError()

def detach_distribution(policy):
    if isinstance(policy, Independent):
        distribution = Independent(detach_distribution(policy.base_dist), policy.reinterpreted_batch_ndims)
    elif isinstance(policy, Categorical):
        distribution = Categorical(logits=policy.logits.detach())
    elif isinstance(policy, Normal):
        distribution = Normal(loc=policy.loc.detach(), scale=policy.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical`, `Independent` and '
                                    '`Normal` policies are valid policies. Got '
                                    '`{0}`.'.format(type(policy)))
    return distribution

def weighted_mean(tensor, lengths=None, print_log=False):
    if lengths is None:
        return torch.mean(tensor)
    if tensor.dim()<2:
        raise ValueError('Expected tensor with at least 2 dimensions '
                            '(trajectory_length x batch_size), got {0}D '
                            'tensor.'.format(tensor.dim()))
    
    for i, length in enumerate(lengths):
        tensor[length:, i].fill_(0.)
    
    extra_dims = (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float64)

    out = torch.sum(tensor, dim=0)
    if print_log:
        print('Weighted mean before div: ', out)
    out.div_(lengths.view(-1, *extra_dims))
    if print_log:
        print('Weighted mean after div: ', out)

    return out

def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    mean = weighted_mean(tensor, lengths=lengths)
    out = tensor - mean.mean()

    for i, length in enumerate(lengths):
        out[length:, i].fill_(0.)
    std = torch.sqrt(weighted_mean(out**2, lengths=lengths).mean())
    out.div_(std + epsilon)

    return out

def vector_to_parameters(vector, parameters):
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)

        num_param = param.numel()
        param.data.copy_(vector[pointer:pointer+num_param].view_as(param).data)
        
        pointer += num_param