#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/5/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import copy

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

class ContactECNN(EquivariantModel):

    def __init__(self, rep_in: Rep, rep_out: Rep, hidden_group: Group, window_size=150, cache_dir=None):
        super(ContactECNN, self).__init__(rep_in, rep_out, hidden_group, cache_dir)
        self.rep_in = rep_in
        self.rep_out = rep_out
        self.hidden_G = hidden_group
        self.window_size = window_size

        rep_ch_64 = SparseRep(self.hidden_G.canonical_group(64))
        rep_ch_128 = SparseRep(self.hidden_G.canonical_group(128))

        self.block1 = nn.Sequential(
            BasisConv1d(rep_in=self.rep_in, rep_out=rep_ch_64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            BasisConv1d(rep_in=rep_ch_64, rep_out=rep_ch_64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # TODO: Make equivariant version.
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            BasisConv1d(rep_in=rep_ch_64, rep_out=rep_ch_128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            BasisConv1d(rep_in=rep_ch_128, rep_out=rep_ch_128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Group of the flatten feature vector, must comply with the 2D symmetry.
        block2_out_window = int(window_size/4)
        G = C2(generator=block_diag([rep_ch_128.G.discrete_generators[0]]*block2_out_window))
        rep_in_mlp = SparseRep(G)

        self.fc = nn.Sequential(
            EquivariantBlock(rep_in=rep_in_mlp,
                             rep_out=SparseRep(self.hidden_G.canonical_group(2048)),
                             activation=nn.ReLU),
            nn.Dropout(p=0.5),
            EquivariantBlock(rep_in=SparseRep(self.hidden_G.canonical_group(2048)),
                             rep_out=SparseRep(self.hidden_G.canonical_group(512)),
                             activation=nn.ReLU),

            nn.Dropout(p=0.5),
            BasisLinear(rep_in=SparseRep(self.hidden_G.canonical_group(512)),
                        rep_out=SparseRep(self.hidden_G.canonical_group(16))),
        )
        self.save_cache_file()

        # Test Each block equivariance.
        EquivariantModel.test_module_equivariance(self.block1, rep_in=self.rep_in, rep_out=rep_ch_64,
                                                  in_shape=(1, 54, 150))
        EquivariantModel.test_module_equivariance(self.block2, rep_in=rep_ch_64, rep_out=rep_ch_128,
                                                  in_shape=(1, 64, 75))
        EquivariantModel.test_module_equivariance(self.fc, rep_in=rep_in_mlp, rep_out=rep_out)
        # Test entire model equivariance.
        self.test_module_equivariance(module=self, rep_in=self.rep_in, rep_out=self.rep_out,
                                      in_shape=(1, 150, rep_in.G.d))
        self.save_cache_file()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block2_out_reshape = block2_out.view(block2_out.shape[0], -1)
        fc_out = self.fc(block2_out_reshape)
        return fc_out

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

            # xa = x[0, 0, :].detach().numpy()
            # gx = g_x[0, 0, :].detach().numpy()
            # g = g_in[0, :, :].detach().numpy()
            # bb = g_y_true[0, :].detach().numpy()

            g_y_pred = module.forward(g_x)

            g_y_true = (g_out[0, :, :] @ y.T).T

            # a = y[0, :].detach().numpy()
            # bb = g_y_true[0, :].detach().numpy()
            # b = g_y_pred[0, :].detach().numpy()
            # g = g_out[0, :, :].detach().numpy()

            if not torch.allclose(g_y_true, g_y_pred, atol=1e-4, rtol=1e-4):
                error = g_y_true - g_y_pred
                raise RuntimeError(f"{module} is not equivariant to in/out group generators f(g·x) - g·y:{error}")
        module.train()