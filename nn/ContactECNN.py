#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/5/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import copy
from math import ceil

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from scipy.sparse import issparse

from groups.SemiDirectProduct import SparseRep
from groups.SymmetricGroups import C2
from nn.EquivariantModules import EquivariantBlock, BasisLinear, EMLP, EquivariantModel
from nn.EConv1d import BasisConv1d
from emlp.groups import Group
from emlp.reps.representation import Rep, Vector
from scipy.sparse import block_diag

import logging
log = logging.getLogger(__name__)

class ContactECNN(EquivariantModel):

    def __init__(self, rep_in: Rep, rep_out: Rep, hidden_group: Group, window_size=150, cache_dir=None, dropout=0.5,
                 init_mode="fan_in", inv_dim_scale=1.0):
        super(ContactECNN, self).__init__(rep_in, rep_out, hidden_group, cache_dir)
        self.rep_in = rep_in
        self.rep_out = rep_out
        self.hidden_G = hidden_group
        self.init_mode = init_mode
        self.window_size = window_size
        self.dropout = dropout
        self.inv_dims_scale = inv_dim_scale

        self.in_invariant_dims = self.rep_in.G.n_inv_dims
        inv_in, inv_out = rep_in.G.n_inv_dims / rep_in.G.d, rep_out.G.n_inv_dims / rep_out.G.d
        n_intermediate_layers = 6
        inv_ratios = np.linspace(inv_in, inv_out, n_intermediate_layers + 2, endpoint=True) * self.inv_dims_scale

        # CNN reps
        rep_ch_64_1 = SparseRep(self.hidden_G.canonical_group(64, inv_dims=ceil(64 * inv_ratios[1])))
        rep_ch_64_2 = SparseRep(self.hidden_G.canonical_group(64, inv_dims=ceil(64 * inv_ratios[2])))
        rep_ch_128_1 = SparseRep(self.hidden_G.canonical_group(128, inv_dims=ceil(128 * inv_ratios[3])))
        rep_ch_128_2 = SparseRep(self.hidden_G.canonical_group(128, inv_dims=ceil(128 * inv_ratios[4])))
        # Group of the flatten feature vector, must comply with the 2D symmetry.
        block2_out_window = int(window_size/4)
        G = C2(generator=block_diag([rep_ch_128_2.G.discrete_generators[0]]*block2_out_window))
        # MLP reps
        rep_in_mlp = SparseRep(G)
        rep_ch_2048 = SparseRep(self.hidden_G.canonical_group(2048, inv_dims=ceil(2048 * inv_ratios[5])))
        rep_ch_512 = SparseRep(self.hidden_G.canonical_group(512, inv_dims=ceil(512 * inv_ratios[6])))

        self.block1 = nn.Sequential(
            BasisConv1d(rep_in=self.rep_in, rep_out=rep_ch_64_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            BasisConv1d(rep_in=rep_ch_64_1, rep_out=rep_ch_64_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            BasisConv1d(rep_in=rep_ch_64_2, rep_out=rep_ch_128_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            BasisConv1d(rep_in=rep_ch_128_1, rep_out=rep_ch_128_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            EquivariantBlock(rep_in=rep_in_mlp,
                             rep_out=rep_ch_2048,
                             activation=nn.ReLU),
            nn.Dropout(p=self.dropout),
            EquivariantBlock(rep_in=rep_ch_2048,
                             rep_out=rep_ch_512,
                             activation=nn.ReLU),

            nn.Dropout(p=self.dropout),
            BasisLinear(rep_in=rep_ch_512, rep_out=self.rep_out),
        )

        # Test Each block equivariance.
        # EquivariantModel.test_module_equivariance(self.block1, rep_in=self.rep_in, rep_out=rep_ch_64,
        #                                           in_shape=(1, 54, 150))
        # EquivariantModel.test_module_equivariance(self.block2, rep_in=rep_ch_64, rep_out=rep_ch_128,
        #                                           in_shape=(1, 64, 75))
        # EquivariantModel.test_module_equivariance(self.fc, rep_in=rep_in_mlp, rep_out=self.rep_out)

        self.reset_parameters(init_mode=init_mode)
        # Test entire model equivariance.
        self.test_module_equivariance(module=self, rep_in=self.rep_in, rep_out=self.rep_out,
                                      in_shape=(1, 150, rep_in.G.d))
        self.save_cache_file()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        # Ensure flattening maintains symmetry constraints
        block2_out = block2_out.permute(0, 2, 1)
        block2_out_reshape = block2_out.reshape(block2_out.shape[0], -1)
        fc_out = self.fc(block2_out_reshape)
        return fc_out

    def get_hparams(self):
        return {'window_size': self.window_size,
                'rep_in': str(self.rep_in),
                'rep_out': str(self.rep_in),
                'hidden_group': str(self.hidden_G),
                'init_mode': self.init_mode,
                'inv_dims_scale': self.inv_dims_scale,
                'dropout': self.dropout}

    def reset_parameters(self, init_mode=None, model=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = init_mode if init_mode is not None else self.init_mode

        x = self if model is None else model
        for module in x.children():
            if isinstance(module, torch.nn.Sequential):
                for m in module.children():
                    if isinstance(m, BasisConv1d):
                        m.reset_parameters(mode=self.init_mode, activation="ReLU")
                    elif isinstance(m, EquivariantBlock):
                        m.linear.reset_parameters(mode=self.init_mode,
                                                  activation=m.activation.__class__.__name__.lower())
                    elif isinstance(m, BasisLinear):
                        m.reset_parameters(mode=self.init_mode, activation="Linear")

        log.info(f"{self.model_class} initialized with mode: {self.init_mode}")


    @staticmethod
    def test_module_equivariance(module: torch.nn.Module, rep_in, rep_out, in_shape=None):
        module.eval()
        shape = (rep_in.G.d) if in_shape is None else in_shape
        x = torch.randn(shape)
        for g_in, g_out in zip(rep_in.G.discrete_generators, rep_out.G.discrete_generators):
            g_in, g_out = (g_in.todense(), g_out.todense()) if issparse(g_in) else (g_in, g_out)
            g_in = torch.tensor(np.asarray(g_in), dtype=torch.float32).unsqueeze(0)
            g_out = torch.tensor(np.asarray(g_out), dtype=torch.float32).unsqueeze(0)

            y = module.forward(x)

            if x.ndim == 3:
                xx = x.permute(0, 2, 1)
                g_x = (g_in @ xx.unsqueeze(1)).squeeze(1)
                g_x = g_x.permute(0, 2, 1)
            else:
                g_x = g_in @ x

            g_y_pred = module.forward(g_x)

            g_y_true = (g_out[0, :, :] @ y.T).T

            if not torch.allclose(g_y_true, g_y_pred, atol=1e-4, rtol=1e-4):
                error = g_y_true - g_y_pred
                raise RuntimeError(f"{module} is not equivariant to in/out group generators f(g·x) - g·y:{error}")
        module.train()