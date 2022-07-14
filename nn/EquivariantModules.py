#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
# Some code was adapted from https://github.com/ElisevanderPol/symmetrizer/blob/master/symmetrizer/nn/modules.py
import itertools
import logging
import math
import pathlib
from typing import Union, Optional
from zipfile import BadZipFile

import numpy as np
import torch
import torch.nn.functional as F
from emlp import Group
from emlp.reps.representation import Rep, Vector
from emlp.reps.representation import Base as BaseRep
from scipy.sparse import issparse

from groups.SemiDirectProduct import SemiDirectProduct, SparseRep
from utils.emlp_cache import EMLPCache
from utils.utils import slugify, coo2torch_coo

log = logging.getLogger(__name__)


class BasisLinear(torch.nn.Module):
    """
    Group-equivariant linear layer
    """
    def __init__(self, rep_in: BaseRep, rep_out: BaseRep, bias=True):
        super().__init__()

        # TODO: Add parameter for direct/whreat product
        G = SemiDirectProduct(Gin=rep_in.G, Gout=rep_out.G)
        self.repW = SparseRep(G)
        self.rep_in = rep_in
        self.rep_out = rep_out

        self._new_coeff, self._new_bias_coeff = True, True
        # Layer can be "unfreeze" and thus keep variable in case that happens.
        self.unfrozed_equivariance = False
        self.unfrozen_w = None
        self.unfrozen_bias = None

        # Compute the nullspace
        Q = self.repW.equivariant_basis()
        self._sum_basis_sqrd = Q.power(2).sum() if issparse(Q) else np.sum(np.power(Q, 2))
        basis = coo2torch_coo(Q) if issparse(Q) else torch.tensor(np.asarray(Q))
        self.basis = torch.nn.Parameter(basis, requires_grad=False)

        # Create the network parameters. Coefficients for each base and a b
        self.basis_coeff = torch.nn.Parameter(torch.randn((self.basis.shape[-1])))

        if bias:
            Qbias = rep_out.equivariant_basis()
            bias_basis = coo2torch_coo(Qbias) if issparse(Qbias) else torch.tensor(np.asarray(Qbias))
            self.bias_basis = torch.nn.Parameter(bias_basis, requires_grad=False)
            self.bias_basis_coeff = torch.nn.Parameter(torch.randn((self.bias_basis.shape[-1])))
            self._bias = self.bias
        else:
            self.bias_basis, self.bias_basis_coeff = None, None

        # TODO: Check if necessary
        # self.proj_b = torchify_fn(jit(lambda b: self.P_bias @ b))
        # Initialize parameters
        self.init_std = None
        self.reset_parameters()

        # Check Equivariance.
        EquivariantModel.test_module_equivariance(module=self, rep_in=self.rep_in, rep_out=self.rep_out)
        # Add hook to backward pass
        self.register_full_backward_hook(EquivariantModel.backward_hook)



    def forward(self, x):
        """
        Normal forward pass, using weights formed by the basis and corresponding coefficients
        """
        if x.device != self.weight.device:
            self._new_coeff, self._new_bias_coeff = True, True
        return F.linear(x, weight=self.weight, bias=self.bias)

    @property
    def weight(self):
        if not self.unfrozed_equivariance:
            # if self._new_coeff or self._weight is None:
            self._weight = torch.matmul(self.basis, self.basis_coeff).reshape((self.rep_out.G.d, self.rep_in.G.d))
            # self._new_coeff = False
            return self._weight
        else:
            return self.unfrozen_w

    @property
    def bias(self):
        if not self.unfrozed_equivariance:
            if self.bias_basis is not None:
                # if self._new_bias_coeff or self._bias is None:
                self._bias = torch.matmul(self.bias_basis, self.bias_basis_coeff).reshape((self.rep_out.G.d,))
                self._new_bias_coeff = False
                return self._bias
            return None
        else:
            return self.unfrozen_bias

    def reset_parameters(self, mode="fan_in", activation="ReLU"):
        if self.unfrozed_equivariance:
            raise BrokenPipeError("initialization called after unfrozed equivariance")
        # Compute the constant coming from the derivative of the activation. Torch return the square root of this value
        gain = torch.nn.init.calculate_gain(nonlinearity=activation.lower())
        # Get input out dimensions.
        dim_in, dim_out = self.rep_in.G.d, self.rep_out.G.d
        # Gain due to parameter sharing scheme from equivariance constrain
        lambd = self._sum_basis_sqrd
        if mode.lower() == "fan_in":
            basis_coeff_variance = dim_out / lambd
        elif mode.lower() == "fan_out":
            basis_coeff_variance = dim_in / lambd
        elif mode.lower() == "harmonic_mean":
            basis_coeff_variance = 2. / ((lambd / dim_out) + (lambd / dim_in))
        elif mode.lower() == "arithmetic_mean":
            basis_coeff_variance = ((dim_in + dim_out) / 2.) / lambd
        elif "normal" in mode.lower():
            split = mode.split('l')
            std = 0.1 if len(split) == 1 else float(split[1])
            torch.nn.init.normal_(self.basis_coeff, 0, std)
            return
        else:
            raise NotImplementedError(f"{mode} is not a recognized mode for Kaiming initialization")

        self.init_std = gain * math.sqrt(basis_coeff_variance)
        bound = math.sqrt(3.0) * self.init_std

        prev_basis_coeff = torch.clone(self.basis_coeff)
        torch.nn.init.uniform_(self.basis_coeff, -bound, bound)

        self._new_coeff, self._new_bias_coeff = True, True
        assert not torch.allclose(prev_basis_coeff, self.basis_coeff), "Ups, smth is wrong."

    def unfreeze_equivariance(self):
        w, bias = self.weight, self.bias
        self.unfrozed_equivariance = True
        self.unfrozen_w = torch.nn.Parameter(w, requires_grad=True)
        self.register_parameter('unfrozen_w', self.unfrozen_w)
        if bias is not None:
            self.unfrozen_bias = torch.nn.Parameter(bias, requires_grad=True)
            self.register_parameter('unfrozen_bias', self.unfrozen_bias)

    def __repr__(self):
        string = f"E-Linear G[{self.repW.G}]-W{self.rep_out.size() * self.rep_in.size()}-" \
                 f"Wtrain:{self.basis.shape[-1]}={self.basis_coeff.shape[0] / np.prod(self.repW.size()) * 100:.1f}%" \
                 f"-init_std:{self.init_std:.3f}"
        return string

    def to(self, *args, **kwargs):
        # When device or type changes tensors need updating.
        self._new_bias_coeff, self._new_coeff = True, True
        return super(BasisLinear, self).to(*args, **kwargs)


class EquivariantBlock(torch.nn.Module):

    def __init__(self, rep_in: BaseRep, rep_out: BaseRep, with_bias=True, activation=torch.nn.Identity):
        super(EquivariantBlock, self).__init__()

        # TODO: Optional Batch Normalization
        self.linear = BasisLinear(rep_in, rep_out, with_bias)
        self.activation = activation()
        self._preact = None   # Debug variable holding last linear activation Tensor, useful for logging.
        EquivariantModel.test_module_equivariance(self, rep_in, rep_out)

    def forward(self, x, **kwargs):
        self._preact = self.linear(x)
        return self.activation(self._preact)

    def unfreeze_equivariance(self):
        self.linear.unfreeze_equivariance()

class EquivariantModel(torch.nn.Module):

    def __init__(self, rep_in: BaseRep, rep_out: BaseRep, hidden_group: Group, cache_dir: Optional[Union[str, pathlib.Path]] = None):
        super(EquivariantModel, self).__init__()
        self.rep_in = rep_in
        self.rep_out = rep_out
        self.hidden_group = hidden_group
        self.cache_dir = cache_dir

        # Cache dir
        self.cache_dir = cache_dir if cache_dir is None else pathlib.Path(cache_dir).resolve(strict=True)
        if self.cache_dir is None:
            log.warning("No cache directory provided. Nothing will be saved")
        elif not self.cache_dir.exists():
            raise OSError(f"Cache dir {self.cache_dir} does not exists")
        else:
            log.info(f"Equivariant Module - Basis Cache dir {self.cache_dir}")
        self.load_cache_file()

    @staticmethod
    def backward_hook(module: torch.nn.Module, _inputs, _outputs):
        if hasattr(module, '_new_coeff') and hasattr(module, '_new_bias_coeff'):
            module._new_coeff, module._new_bias_coeff = True, True

    @property
    def _cache_file_name(self) -> str:
        EXTENSION = ".npz"
        model_rep = f'{self.rep_in.G}-{self.rep_out.G}'
        return slugify(model_rep) + EXTENSION

    def load_cache_file(self):
        if self.cache_dir is None:
            log.info("Cache Loading Failed: No cache directory provided")
            return
        model_cache_file = self.cache_dir.joinpath(self._cache_file_name)

        if not model_cache_file.exists():
            log.warning(f"Model cache {model_cache_file.stem} not found")
            return

        try:
            lazy_cache = np.load(model_cache_file, allow_pickle=True, mmap_mode='c')

            run_cache = self.rep_in.solcache
            if isinstance(run_cache, EMLPCache):
                cache = run_cache.cache
            else:
                cache = run_cache

            # Remove from memory cache all file-saved caches. Taking advantage of lazy loading.
            for k in list(cache.keys()):
                if str(k) in lazy_cache:
                    cache.pop(k)

            Rep.solcache = EMLPCache(cache, lazy_cache)
            log.info(f"Cache loaded for: {list(Rep.solcache.keys())}")
        except Exception as e:
            log.warning(f"Error while loading cache from {model_cache_file}: \n {e}")


    def save_cache_file(self):
        if self.cache_dir is None:
            log.info("Cache Saving Failed: No cache directory provided")
            return

        model_cache_file = self.cache_dir.joinpath(self._cache_file_name)

        run_cache = self.rep_in.solcache
        if isinstance(run_cache, EMLPCache):
            lazy_cache, cache = run_cache.lazy_cache, run_cache.cache
        else:
            lazy_cache, cache = {}, run_cache

        if len(run_cache) == 0:
            log.debug(f"Ignoring cache save as there is no new equivariant basis")
        try:
            combined_cache = {str(k): np.asarray(v) for k, v in itertools.chain(lazy_cache.items(), cache.items())}
            np.savez_compressed(model_cache_file, **combined_cache)

            # Since we moved all cache to disk with lazy loading. Remove from memory
            self.rep_in.solcache = EMLPCache(cache={}, lazy_cache=np.load(str(model_cache_file), allow_pickle=True))
            log.info(f"Saved cache from {list(self.rep_in.solcache.keys())} to {model_cache_file}")
        except BadZipFile as e:
            self.rep_in.solcache.lazy_cache = {}
            log.warning(f"Error while saving cache to {model_cache_file}: \n {e}")
        except Exception as e:
            log.warning(f"Error while saving cache to {model_cache_file}: \n {e}")

    @staticmethod
    def test_module_equivariance(module: torch.nn.Module, rep_in, rep_out, in_shape=None):
        module.eval()
        shape = (rep_in.G.d,) if in_shape is None else in_shape
        x = torch.randn(shape)
        for g_in, g_out in zip(rep_in.G.discrete_generators, rep_out.G.discrete_generators):
            g_in, g_out = (g_in.todense(), g_out.todense()) if issparse(g_in) else (g_in, g_out)
            g_in = torch.tensor(np.asarray(g_in), dtype=torch.float32).unsqueeze(0)
            g_out = torch.tensor(np.asarray(g_out), dtype=torch.float32).unsqueeze(0)

            y = module.forward(x)

            if x.ndim == 3:
                g_x = (g_in @ x.unsqueeze(1)).squeeze(1)
            else:
                g_x = g_in @ x

            g_y_pred = module.forward(g_x)
            g_y_true = g_out @ y
            if not torch.allclose(g_y_true, g_y_pred, atol=1e-4, rtol=1e-4):
                max_error = torch.max(g_y_true - g_y_pred).item()
                error = (g_y_true - g_y_pred).detach().numpy()
                raise RuntimeError(f"{module}\nis not equivariant to in/out group generators\n"
                                   f"max(f(g·x) - g·y) = {np.max(error)}")

            if torch.allclose(g_y_pred, y, atol=1e-4, rtol=1e-4):
                log.warning(f"\nModule {module} is INVARIANT! not EQUIVARIANT\n")
        module.train()

    @property
    def model_class(self):
        return self.__class__.__name__

    # def __repr__(self):
    #     return f'{self.model_class}: {self.rep_in.G}-{self.rep_out.G}'


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

    def __init__(self, rep_in, rep_out, hidden_group, ch=64, num_layers=3, with_bias=True, activation=torch.nn.ReLU,
                 cache_dir=None, init_mode="fan_in", inv_dims_scale=1.0):
        super().__init__(rep_in, rep_out, hidden_group, cache_dir)
        logging.info("Initing EMLP (PyTorch)")
        self.activations = activation
        self.hidden_channels = ch
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

        # input_layer = layers[0].linear
        # W = input_layer.weight.detach().numpy()
        # b = input_layer.bias.detach().numpy()

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

class LinearBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, with_bias=True, activation=torch.nn.Identity):
        super().__init__()

        # TODO: Optional Batch Normalization
        self.linear = torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=with_bias)
        self.activation = activation()
        self._preact = None   # Debug variable holding last linear activation Tensor, useful for logging.

    def forward(self, x, **kwargs):
        self._preact = self.linear(x)
        return self.activation(self._preact)


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

    def forward(self, x):
        y = self.net(x)
        return y

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
