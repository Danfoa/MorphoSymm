#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
# Some code was adapted from https://github.com/ElisevanderPol/symmetrizer/blob/master/symmetrizer/nn/modules.py
import numpy as np
import torch
import torch.nn.functional as F
from jax import jit
from emlp.nn.pytorch import torchify_fn
from emlp.reps import Rep
from emlp import Group
from emlp.reps.representation import Vector
from emlp.reps.linear_operators import densify

from robot_kinematic_symmetries import SemiDirectProduct
import logging

log = logging.getLogger(__name__)


class BasisLinear(torch.nn.Module):
    """
    Group-equivariant linear layer
    """

    def __init__(self, rep_in: Rep, rep_out: Rep, with_bias=True):
        super().__init__()

        # TODO: Add parameter for direct/whreat product
        self.G = SemiDirectProduct(Gin=rep_in.G, Gout=rep_out.G)
        self.repW = Vector(self.G)

        self.rep_in = rep_in
        self.rep_out = rep_out

        self.with_bias = with_bias

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

    def __init__(self, rep_in, rep_out, group, ch=128, num_layers=3, with_bias=True, activation=torch.nn.SiLU):
        super().__init__()
        logging.info("Initing EMLP (PyTorch)")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.activations = activation
        self.G = group
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
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



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
