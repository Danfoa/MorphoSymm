#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
# Some code was adapted from https://github.com/ElisevanderPol/symmetrizer/blob/master/symmetrizer/nn/modules.py
import functools
import itertools
import math
import os
import pathlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from jax import jit
from emlp import Group
from emlp.nn.pytorch import torchify_fn
from emlp.reps.representation import Rep, Vector
from emlp.reps.linear_operators import densify
import pickle5 as pickle
from pickle5 import PickleError

from groups.SemiDirectProduct import SemiDirectProduct
import logging

from utils.emlp_cache import EMLPCache
from utils.utils import slugify
import itertools

log = logging.getLogger(__name__)


class BasisLinear(torch.nn.Module):
    """
    Group-equivariant linear layer
    """
    def __init__(self, rep_in: Rep, rep_out: Rep, with_bias=True, cache_dir=None):
        super().__init__()

        # TODO: Add parameter for direct/whreat product
        self.G = SemiDirectProduct(Gin=rep_in.G, Gout=rep_out.G)
        self.repW = Vector(self.G)
        self.rep_in = rep_in
        self.rep_out = rep_out
        self.with_bias = with_bias

        self.cache_dir = cache_dir

        # Compute the nullspace
        self.basis = torch.nn.Parameter(torch.tensor(np.array(densify(self.repW.equivariant_basis()))),
                                        requires_grad=False)
        # Create the network parameters. Coefficients for each base and a b
        self.basis_coeff = torch.nn.Parameter(torch.randn((self.basis.shape[-1])))
        self._weight = self.weight

        if self.with_bias:
            self._bias = self.bias
            self.bias_basis = torch.nn.Parameter(torch.tensor(np.array(rep_out.equivariant_basis()))
                                                 , requires_grad=False)
            self.bias_basis_coeff = torch.nn.Parameter(torch.randn((self.bias_basis.shape[-1])))

        # TODO: Check if necessary
        self.proj_b = torchify_fn(jit(lambda b: self.P_bias @ b))
        self.reset_parameters()
        EMLP.test_module_equivariance(module=self, rep_in=self.rep_in, rep_out=self.rep_out)
        log.info(str(self))

    def forward(self, x):
        """
        Normal forward pass, using weights formed by the basis and corresponding coefficients
        """
        return F.linear(x, weight=self.weight, bias=self.bias)

    @property
    def weight(self):
        self._weight = torch.sum(self.basis * self.basis_coeff, dim=-1).reshape((self.rep_out.G.d, self.rep_in.G.d))
        return self._weight

    @property
    def bias(self):
        if self.with_bias:
            self._bias = torch.sum(self.bias_basis * self.bias_basis_coeff, dim=-1).reshape((self.rep_out.G.d,))
            return self._bias
        return None

    def reset_parameters(self, mode="fan_in", activation="ReLU"):
        # Compute the constant coming from the derivative of the activation. Torch return the square root of this value
        gain = torch.nn.init.calculate_gain(nonlinearity=activation.lower())
        # Get input out dimensions.
        dim_in, dim_out = self.rep_in.G.d, self.rep_out.G.d
        # Gain due to parameter sharing scheme from equivariance constrain
        lambd = torch.sum(torch.sum(self.basis**2, dim=1))
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

        std = gain * math.sqrt(basis_coeff_variance)
        bound = math.sqrt(3.0) * std

        torch.nn.init.uniform_(self.basis_coeff, -bound, bound)

    def __repr__(self):
        string = f"G[{self.G}]-W{self.rep_out.size() * self.rep_in.size()}-" \
                 f"Wtrain:{self.basis.shape[-1]}={self.basis_coeff.shape[0] / np.prod(self.repW.size()) * 100:.1f}%"
        return string


class EquivariantBlock(torch.nn.Module):

    def __init__(self, rep_in: Rep, rep_out: Rep, with_bias=True, activation=torch.nn.Identity):
        super(EquivariantBlock, self).__init__()

        # TODO: Optional Batch Normalization
        self.linear = BasisLinear(rep_in, rep_out, with_bias)
        self.activation = activation()
        self._preact = None   # Debug variable holding last linear activation Tensor, useful for logging.
        EMLP.test_module_equivariance(self, rep_in, rep_out)

    def forward(self, x, **kwargs):
        self._preact = self.linear(x)
        return self.activation(self._preact)


class EMLP(torch.nn.Module):
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

    def __init__(self, rep_in, rep_out, hidden_group, ch=128, num_layers=3, with_bias=True, activation=torch.nn.ReLU,
                 cache_dir=None, init_mode="fan_in"):
        super().__init__()
        logging.info("Initing EMLP (PyTorch)")
        self.rep_in = rep_in
        self.rep_out = rep_out
        self.activations = activation
        self.hidden_G = hidden_group

        self.hidden_channels = ch
        self.n_layers = num_layers
        self.init_mode = init_mode

        # Cache dir
        self.cache_dir = cache_dir if cache_dir is None else pathlib.Path(cache_dir).resolve(strict=True)
        self.load_cache_file()

        # Parse channels as a single int, a sequence of ints, a single Rep, a sequence of Reps
        rep_inter_in = rep_in
        rep_inter_out = rep_out

        layers = []
        for n in range(num_layers + 1):
            rep_inter_out = Vector(self.hidden_G.canonical_group(ch))
            layer = EquivariantBlock(rep_in=rep_inter_in, rep_out=rep_inter_out, with_bias=with_bias,
                                     activation=self.activations)
            layers.append(layer)
            rep_inter_in = rep_inter_out
        # Add last layer
        linear_out = BasisLinear(rep_in=rep_inter_in, rep_out=rep_out, with_bias=with_bias)
        layers.append(linear_out)

        self.net = torch.nn.Sequential(*layers)
        self.reset_parameters(init_mode=self.init_mode)
        EMLP.test_module_equivariance(self, rep_in, rep_out)
        self.save_cache_file()

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def test_module_equivariance(module: torch.nn.Module, rep_in, rep_out):
        x = torch.randn((rep_in.G.d,))
        for g_in, g_out in zip(rep_in.G.discrete_generators, rep_out.G.discrete_generators):
            g_in = torch.tensor(np.asarray(g_in), dtype=torch.float32)
            g_out = torch.tensor(np.asarray(g_out), dtype=torch.float32)
            # y  f(x)  --   g·y = f(g·x)
            y = module.forward(x)
            x_ = g_in @ x
            g_y_pred = module.forward(g_in @ x)
            g_y_true = g_out @ y
            if not torch.allclose(g_y_true, g_y_pred, atol=1e-4, rtol=1e-4):
                error = g_y_true - g_y_pred
                raise RuntimeError(f"{module} is not equivariant to in/out group generators f(g·x) - g·y:{error}")

    @property
    def _cache_file_name(self) -> str:
        EXTENSION = ".npz"
        model_rep = f'{self.rep_in.G}-{self.rep_out.G}'
        return slugify(model_rep) + EXTENSION

    def load_cache_file(self):
        if self.cache_dir is None:
            return
        model_cache_file = self.cache_dir.joinpath(self._cache_file_name)

        if not model_cache_file.exists():
            log.warning(f"Model cache {model_cache_file.stem} not found")
            return

        lazy_cache = np.load(model_cache_file)

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
        # for k, v in cache.items():
        #     if k in self.rep_in.solcache: continue
        #     Rep.solcache[k] = jax.device_put(v)
        log.info(f"Cache loaded for {list(Rep.solcache.keys())}")

    def save_cache_file(self):
        if self.cache_dir is None:
            warnings.warn("No cache directory provided. Nothing will be saved")
            return

        model_cache_file = self.cache_dir.joinpath(self._cache_file_name)

        run_cache = self.rep_in.solcache
        if isinstance(run_cache, EMLPCache):
            lazy_cache, cache = run_cache.lazy_cache, run_cache.cache
        else:
            lazy_cache, cache = {}, run_cache
        combined_cache = {str(k): np.asarray(v) for k, v in itertools.chain(lazy_cache.items(), cache.items())}
        np.savez_compressed(model_cache_file, **combined_cache)

        # Since we moved all cache to disk with lazy loading. Remove from memory
        self.rep_in.solcache = EMLPCache(cache={}, lazy_cache=np.load(model_cache_file))
        log.info(f"Saved cache from {list(self.rep_in.solcache.keys())} to {model_cache_file}")

    def get_hparams(self):
        return {'num_layers': len(self.net),
                'hidden_ch': self.hidden_channels,
                'Repin': str(self.rep_in),
                'Repout': str(self.rep_in),
                'init_mode': str(self.init_mode),
                }

    def reset_parameters(self, init_mode=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = self.init_mode if init_mode is None else init_mode
        for module in self.net:
            if isinstance(module, EquivariantBlock):
                module.linear.reset_parameters(mode=self.init_mode, activation=module.activation.__class__.__name__.lower())
            elif isinstance(module, BasisLinear):
                module.reset_parameters(mode=self.init_mode, activation="Linear")
        log.info(f"EMLP initialized with mode: {self.init_mode}")


class LinearBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, with_bias=True, activation=torch.nn.Identity):
        super(LinearBlock, self).__init__()

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
