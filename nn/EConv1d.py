#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16/5/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import math
from typing import Union

import numpy as np
import torch
from emlp.reps.representation import Base as BaseRep
from scipy.sparse import issparse
from torch.nn import functional as F
from torch.nn.modules.utils import _single

from groups.SemiDirectProduct import SemiDirectProduct, SparseRep
from nn.EquivariantModules import EquivariantModel
from utils.utils import coo2torch_coo


class BasisConv1d(torch.nn.Module):
    from torch.nn.common_types import _size_1_t

    def __init__(self, rep_in: BaseRep, rep_out: BaseRep, kernel_size: _size_1_t, stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True,
                 dtype=None) -> None:
        super().__init__()

        # Original Implementation Parameters ___________________________________________________________
        self.kernel_size_ = int(kernel_size)
        self.stride_ = _single(stride)
        self.padding_ = padding if isinstance(padding, str) else _single(padding)
        self.dilation_ = _single(dilation)
        self.groups_ = groups

        # Custom parameters ____________________________________________________________________________
        G = SemiDirectProduct(Gin=rep_in.G, Gout=rep_out.G)
        self.repW = SparseRep(G)
        self.rep_in = rep_in
        self.rep_out = rep_out

        # Avoid recomputing W when basis coefficients have not changed.
        self._new_coeff, self._new_basis_coeff = True, True

        # Compute the nullspace
        Q = self.repW.equivariant_basis()
        self._sum_basis_sqrd = Q.power(2).sum() if issparse(Q) else np.sum(np.power(Q))
        basis = coo2torch_coo(Q) if issparse(Q) else torch.tensor(np.asarray(Q))
        self.basis = torch.nn.Parameter(basis, requires_grad=False)

        # Create the network parameters. Coefficients for each base, and kernel dim
        self.basis_coeff = torch.nn.Parameter(torch.rand(self.basis.shape[1], self.kernel_size_))

        self._weight = self.weight

        if bias:
            Qbias = rep_out.equivariant_basis()
            bias_basis = coo2torch_coo(Qbias) if issparse(Qbias) else torch.tensor(np.asarray(Qbias))
            self.bias_basis = torch.nn.Parameter(bias_basis, requires_grad=False)
            self.bias_basis_coeff = torch.nn.Parameter(torch.randn((self.bias_basis.shape[-1])))
            self._bias = self.bias
        else:
            self.bias_basis, self.bias_basis_coeff = None, None

        self.reset_parameters()
        # Check Equivariance.
        EquivariantModel.test_module_equivariance(module=self, rep_in=self.rep_in, rep_out=self.rep_out,
                                                  in_shape=(1, rep_in.G.d, 2))
        # Add hook to backward pass
        self.register_full_backward_hook(EquivariantModel.backward_hook)

    def forward(self, x):
        return F.conv1d(input=x, weight=self.weight, bias=self.bias, stride=self.stride_, padding=self.padding_,
                        dilation=self.dilation_, groups=self.groups_)

    @property
    def weight(self):
        if self._new_coeff:
            self._weight = torch.matmul(self.basis, self.basis_coeff).reshape((self.rep_out.G.d, self.rep_in.G.d, self.kernel_size_))
            self._new_coeff = False
        return self._weight

    @property
    def bias(self):
        if self.bias_basis is not None:
            if self._new_basis_coeff:
                self._bias = torch.matmul(self.bias_basis, self.bias_basis_coeff).reshape((self.rep_out.G.d,))
                self._new_basis_coeff = False
            return self._bias
        return None

    def reset_parameters(self, mode="fan_in", activation="ReLU"):
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
        if self.bias_basis is not None:
            torch.nn.init.zeros_(self.bias_basis_coeff)

        self._new_coeff, self._new_bias_coeff = True, True
        assert not torch.allclose(prev_basis_coeff, self.basis_coeff), "Ups, smth is wrong."