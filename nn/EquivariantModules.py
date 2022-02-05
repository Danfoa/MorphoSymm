#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
# Some code was adapted from https://github.com/ElisevanderPol/symmetrizer/blob/master/symmetrizer/nn/modules.py
import math
import os
import pathlib
import warnings

import jax
import numpy as np
import torch
import torch.nn.functional as F
from jax import jit
from emlp.nn.pytorch import torchify_fn
from emlp.reps.representation import Rep
from emlp import Group
from emlp.reps.representation import Vector
from emlp.reps.linear_operators import densify
import pickle

from groups.SemiDirectProduct import SemiDirectProduct
import logging
from utils.utils import slugify

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
        if self.with_bias:
            self.bias_basis = torch.nn.Parameter(torch.tensor(np.array(rep_out.equivariant_basis()))
                                                 , requires_grad=False)
            self.bias_basis_coeff = torch.nn.Parameter(torch.randn((self.bias_basis.shape[-1])))

        # TODO: Check if necessary
        self.proj_b = torchify_fn(jit(lambda b: self.P_bias @ b))
        self.reset_parameters()
        log.info(str(self))

    def forward(self, x):
        """
        Normal forward pass, using weights formed by the basis and corresponding coefficients
        """
        return F.linear(x, weight=self.W, bias=self.bias)

    @property
    def W(self):
        return torch.sum(self.basis * self.basis_coeff, dim=-1).reshape((self.rep_out.G.d, self.rep_in.G.d))

    @property
    def bias(self):
        if self.with_bias:
            return torch.sum(self.bias_basis * self.bias_basis_coeff, dim=-1).reshape((self.rep_out.G.d,))
        return None

    def reset_parameters(self, weights_initializer=torch.nn.init.kaiming_uniform):
        # TODO: Compute initialization considering weight sharing distribution
        # Estimate the gain value considering the equivariance.
        # For now use Glorot initialization, must check later:
        gain = 1  # np.sqrt(2)

        fan_in, fan_out = self.W.shape
        if weights_initializer == torch.nn.init.xavier_uniform:
            # Xavier cannot find the in out dimensions because the tensor is not 2D
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(self.basis_coeff, -bound, bound)
        elif weights_initializer == torch.nn.init.kaiming_uniform:
            std = gain / math.sqrt(fan_out)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                return self.basis_coeff.uniform_(-bound, bound)
        else:
            weights_initializer(self.basis_coeff)
        if self.with_bias:
            self.bias_basis_coeff.zero_()

    def __repr__(self):
        string = f"G[{self.G}]-W{self.rep_out.size() * self.rep_in.size()}-" \
                 f"Wtrain:{self.basis.shape[-1]}={self.basis_coeff.shape[0] / np.prod(self.repW.size()) * 100:.1f}%"
        return string


class EBlock(torch.nn.Module):

    def __init__(self, rep_in: Rep, rep_out: Rep, with_bias=True, activation=torch.nn.Identity):
        super(EBlock, self).__init__()

        # TODO: Optional Batch Normalization
        self.linear = BasisLinear(rep_in, rep_out, with_bias)
        self.activation = activation()

    def forward(self, x, **kwargs):
        preact = self.linear(x)
        return self.activation(preact)



class EMLP(torch.nn.Module):
    """ Equivariant MultiLayer Perceptron.
        If the input channels argument is an int, uses the hands off uniform_rep heuristic.
        If the channels argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.

        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers

        Returns:
            Module: the EMLP objax module."""

    def __init__(self, rep_in, rep_out, group, ch=128, num_layers=3, with_bias=True, activation=torch.nn.SiLU,
                 cache_dir=None):
        super().__init__()
        logging.info("Initing EMLP (PyTorch)")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.activations = activation
        self.G = group

        self.hidden_channels = ch
        self.n_layers = num_layers

        # Cache dir
        self.cache_dir = cache_dir if cache_dir is None else pathlib.Path(cache_dir).resolve(strict=True)
        self.load_cache_file()

        # Parse channels as a single int, a sequence of ints, a single Rep, a sequence of Reps
        rep_inter_in = rep_in
        rep_inter_out = rep_out

        layers = []
        for n in range(num_layers):
            rep_inter_out = Vector(group.canonical_group(ch))
            layer = EBlock(rep_in=rep_inter_in, rep_out=rep_inter_out, with_bias=with_bias,
                           activation=self.activations)
            layers.append(layer)
            rep_inter_in = rep_inter_out
        # Add last layer
        linear_out = BasisLinear(rep_in=rep_inter_in, rep_out=rep_out, with_bias=with_bias)
        layers.append(linear_out)

        # logging.info(f"Reps: {reps}")
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @property
    def _cache_file_name(self) -> str:
        EXTENSION = ".emlp_cache"
        model_rep = f'{self.rep_in.G}-{self.rep_out.G}'
        return slugify(model_rep) + EXTENSION

    def load_cache_file(self):
        if self.cache_dir is None:
            return
        model_cache_file = self.cache_dir.joinpath(self._cache_file_name)
        if not model_cache_file.exists():
            warnings.warn(f"Model cache {model_cache_file.stem} not found")
            return
        with open(model_cache_file, 'rb') as handle:
            self.rep_in.solcache.update(pickle.load(handle))
            log.info(f"Model Cache found with Reps: {[k for k in self.rep_in.solcache.keys()]}")
            for k, v in self.rep_in.solcache.items():
                self.rep_in.solcache[k] = jax.device_put(v)


    def save_cache_file(self):
        if self.cache_dir is None:
            warnings.warn("No cache directory provided. Nothing will be saved")
        with open(self.cache_dir.joinpath(self._cache_file_name), 'wb') as handle:
            pickle.dump(self.rep_in.solcache, handle, protocol=pickle.HIGHEST_PROTOCOL)


class MLP(torch.nn.Module):
    """ Standard baseline MLP. Representations and group are used for shapes only. """

    def __init__(self, d_in, d_out, ch=384, num_layers=3, activation=torch.nn.SiLU):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        chs = [self.d_in] + num_layers * [ch]
        cout = self.d_out
        logging.info("Initing MLP")

        layers = []
        for cin, cout in zip(chs, chs[1:]):
            layers.append(torch.nn.Linear(cin, cout))
            layers.append(activation())
        layers.append(torch.nn.Linear(chs[-1], self.d_out))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y
