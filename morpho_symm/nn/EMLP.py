import logging
import math

import numpy as np
import torch
from emlp.reps.representation import Base as BaseRep

from morpho_symm.groups.SparseRepresentation import SparseRep
from .EquivariantModules import EquivariantModel, BasisLinear
log = logging.getLogger(__name__)

class EMLP(EquivariantModel):
    """ Equivariant MultiLayer Perceptron.
        If the input channels argument is an int, uses the hands off uniform_rep heuristic.
        If the channels argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.

        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            hidden_group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers

        Returns:
            Module: the EMLP objax module."""

    def __init__(self, rep_in, rep_out, ch=64, num_layers=3, with_bias=True, activation=torch.nn.ReLU,
                 cache_dir=None, init_mode="fan_in", inv_dims_scale=0.0):
        super().__init__(rep_in, rep_out, cache_dir)
        logging.info("Initing EMLP (PyTorch)")
        self.activations = activation
        self.hidden_channels = ch
        self.hidden_group = rep_in.G
        self.n_layers = num_layers
        self.init_mode = init_mode
        self.inv_dims_scale = inv_dims_scale
        # Parse channels as a single int, a sequence of ints, a single Rep, a sequence of Reps
        rep_inter_in = rep_in
        rep_inter_out = rep_out
        inv_in, inv_out = rep_in.G.n_inv_dims/rep_in.G.d, rep_out.G.n_inv_dims/rep_out.G.d
        inv_ratios = np.linspace(inv_in, inv_out, num_layers + 3, endpoint=True) * self.inv_dims_scale

        layers = []
        for n, inv_ratio in zip(range(num_layers + 1), inv_ratios[1:-1]):
            rep_inter_out = SparseRep(self.hidden_group.canonical_group(ch, inv_dims=math.ceil(ch * inv_ratio)))
            layer = EquivariantBlock(rep_in=rep_inter_in, rep_out=rep_inter_out, with_bias=with_bias,
                                     activation=self.activations)
            layers.append(layer)
            rep_inter_in = rep_inter_out
        # Add last layer
        linear_out = BasisLinear(rep_in=rep_inter_in, rep_out=rep_out, bias=False)
        layers.append(linear_out)

        self.net = torch.nn.Sequential(*layers)
        self.reset_parameters(init_mode=self.init_mode)
        EquivariantModel.test_module_equivariance(self, rep_in, rep_out)
        self.save_cache_file()

    def forward(self, x):
        return self.net(x)

    def get_hparams(self):
        return {'num_layers': len(self.net),
                'hidden_ch': self.hidden_channels,
                'activation': str(self.activations.__class__.__name__),
                'Repin': str(self.rep_in),
                'Repout': str(self.rep_in),
                'init_mode': str(self.init_mode),
                'inv_dim_scale': self.inv_dims_scale,
                }

    def reset_parameters(self, init_mode=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = init_mode if init_mode is not None else self.init_mode
        for module in self.net:
            if isinstance(module, EquivariantBlock):
                module.linear.reset_parameters(mode=self.init_mode, activation=module.activation.__class__.__name__.lower())
            elif isinstance(module, BasisLinear):
                module.reset_parameters(mode=self.init_mode, activation="Linear")
        log.info(f"EMLP initialized with mode: {self.init_mode}")

    def unfreeze_equivariance(self, num_layers=1):
        assert num_layers >= 1, num_layers
        # Freeze most of model parameters.
        for parameter in self.parameters():
            parameter.requires_grad = False

        for e_layer in self.net[-num_layers:]:
            e_layer.unfreeze_equivariance()


class MLP(torch.nn.Module):
    """ Standard baseline MLP. Representations and group are used for shapes only. """

    def __init__(self, d_in, d_out, ch=128, num_layers=3, activation=torch.nn.ReLU, with_bias=True, init_mode="fan_in"):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.init_mode = init_mode
        self.hidden_channels = ch
        self.activation = activation

        logging.info("Initing MLP")

        dim_in = self.d_in
        dim_out = ch
        layers = []
        for n in range(num_layers + 1):
            dim_out = ch
            layer = LinearBlock(dim_in=dim_in, dim_out=dim_out, with_bias=with_bias, activation=activation)
            # init
            # torch.nn.init.kaiming_uniform_(layer.linear.weight, mode=init_mode,
            #                                nonlinearity=activation.__name__.lower())
            dim_in = dim_out
            layers.append(layer)
        # Add last layer
        linear_out = torch.nn.Linear(in_features=dim_out, out_features=self.d_out, bias=with_bias)
        # init.
        # torch.nn.init.kaiming_uniform_(linear_out.weight, mode=init_mode, nonlinearity="linear")
        layers.append(linear_out)

        self.net = torch.nn.Sequential(*layers)
        self.reset_parameters(init_mode=self.init_mode)

    def forward(self, input):
        output = self.net(input)
        return output

    def get_hparams(self):
        return {'num_layers': len(self.net),
                'hidden_ch': self.hidden_channels,
                'init_mode': self.init_mode}

    def reset_parameters(self, init_mode=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = self.init_mode if init_mode is None else init_mode
        for module in self.net:
            if isinstance(module, LinearBlock):
                tensor = module.linear.weight if isinstance(module, LinearBlock) else module.weight
                activation = module.activation.__class__.__name__
            elif isinstance(module, torch.nn.Linear):
                tensor = module.weight
                activation = "Linear"
            else:
                raise NotImplementedError(module.__class__.__name__)

            if "fan_in" == self.init_mode or "fan_out" == self.init_mode:
                torch.nn.init.kaiming_uniform_(tensor, mode=self.init_mode, nonlinearity=activation.lower())
            elif 'normal' in self.init_mode.lower():
                split = self.init_mode.split('l')
                std = 0.1 if len(split) == 1 else float(split[1])
                tensor = module.linear.weight if isinstance(module, LinearBlock) else module.weight
                torch.nn.init.normal_(tensor, 0, std)
            else:
                raise NotImplementedError(self.init_mode)

        log.info(f"MLP initialized with mode: {self.init_mode}")


class LinearBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, with_bias=True, activation=torch.nn.Identity):
        super().__init__()

        # TODO: Optional Batch Normalization
        self.linear = torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=with_bias)
        self.activation = activation()
        self._preact = None  # Debug variable holding last linear activation Tensor, useful for logging.

    def forward(self, x, **kwargs):
        self._preact = self.linear(x)
        return self.activation(self._preact)


class EquivariantBlock(torch.nn.Module):

    def __init__(self, rep_in: BaseRep, rep_out: BaseRep, with_bias=True, activation=torch.nn.Identity):
        super(EquivariantBlock, self).__init__()

        # TODO: Optional Batch Normalization
        self.linear = BasisLinear(rep_in, rep_out, with_bias)
        self.activation = activation()
        self._preact = None  # Debug variable holding last linear activation Tensor, useful for logging.
        EquivariantModel.test_module_equivariance(self, rep_in, rep_out)

    def forward(self, x, **kwargs):
        self._preact = self.linear(x)
        return self.activation(self._preact)

    def unfreeze_equivariance(self):
        self.linear.unfreeze_equivariance()
