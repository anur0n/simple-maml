from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

from torch.distributions import Independent, Normal
import config


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


class NormalPolicy(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layer_size=(),
        activation=F.relu,
        init_std=1.0,
        min_std=1e-6,
    ) -> None:
        super(NormalPolicy, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.input_size = input_size
        self.output_size = output_size
        # TODO change activation
        self.activation = torch.tanh
        self.min_log_std = math.log(min_std)
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters
        print("Policy: ", input_size, output_size)

        self.num_layers = len(hidden_layer_size) + 1

        layer_sizes = (input_size,) + hidden_layer_size
        self.batch_norm_layers = []
        for i in range(1, self.num_layers):
            self.add_module(
                "layer{0}".format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )
            # self.batch_norm_layers.append(nn.BatchNorm1d(layer_sizes[i]))
            # self.add_module("bn_layer{0}".format(i), self.batch_norm_layers[-1])
            # norm = nn.BatchNorm1d(10)
            # norm.run

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))

        self.apply(init_weights)

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print('Params: ', params)
        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
        # print('Gradients: ', grads)
        # print('Updating params: ', torch.isnan(grads).any())
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params

    def forward(self, input, params=None, print_log=False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            # print(i, ' :In shape: ', output.shape)
            output = F.linear(
                output,
                weight=params["layer{0}.weight".format(i)],
                bias=params["layer{0}.bias".format(i)],
            )
            # output = F.batch_norm(
            #     output,
            #     running_mean=self.batch_norm_layers[i - 1].running_mean,
            #     running_var=self.batch_norm_layers[i - 1].running_var,
            #     weight=params["bn_layer{0}.weight".format(i)],
            #     bias=params["bn_layer{0}.bias".format(i)],
            # )
            output = self.activation(output)
            # print('out shape: ', output.shape)

        mu = F.linear(output, weight=params["mu.weight"], bias=params["mu.bias"])
        # scale = torch.ones_like(params['sigma'])
        scale = torch.exp(torch.clamp(params["sigma"], min=self.min_log_std))

        if print_log:
            print("Policy net: mu: ", mu.shape, mu)
            print("Scales: ", scale.shape, scale)

        return Independent(Normal(loc=mu, scale=scale.abs()), 1)
