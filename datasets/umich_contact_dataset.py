#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/5/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from deep_contact_estimator.utils.data_handler import contact_dataset
from torch.utils.data._utils.collate import default_collate

from groups.SymmetricGroups import C2
from utils.utils import reflex_matrix, coo2torch_coo


class UmichContactDataset(contact_dataset):

    def __init__(self, data_path, label_path, window_size, device='cuda', augment=False):

        super().__init__(data_path, label_path, window_size, device=device)

        self.contact_state_freq = self.get_class_frequency()

        self.Gin, self.Gout = self.get_in_out_groups()
        self.augment = augment
        self.n_contact_states = 16
        if self.augment:
            # self._hin = coo2torch_coo(self.Gin.discrete_generators[0])
            # self._hout = coo2torch_coo(self.Gout.discrete_generators[0])
            self.hin = torch.tensor(self.Gin.discrete_generators[0].todense(), device=device)
            self.hout = torch.tensor(self.Gout.discrete_generators[0].todense(), device=device)

    def collate_fn(self, batch):
        collated_batch = default_collate(batch)

        if self.augment:
            # (Batch, Window size, features)
            x_batch = collated_batch['data']
            y_batch = F.one_hot(collated_batch['label'], num_classes=self.n_contact_states)
            g_x_batch = torch.matmul(x_batch.unsqueeze(1), self.hin.unsqueeze(0).to(x_batch.dtype)).squeeze()
            g_y_batch = torch.matmul(y_batch.unsqueeze(1), self.hout.unsqueeze(0).to(y_batch.dtype)).squeeze()
            # Convert back to numerical class label.
            _, g_y = torch.max(g_y_batch, dim=1)
            return {'data': g_x_batch, 'label': g_y}
        else:
            return collated_batch

    def get_class_frequency(self):
        classes, counts = torch.unique(self.label, return_counts=True, sorted=True)
        return counts / len(self)

    @staticmethod
    def get_in_out_groups():
        # Joint Space
        #        ____RF___|___LF____|___RH______|____LH____|
        # q    = [ 0, 1, 2,  3, 4, 5,  6,  7,  8, 9, 10, 11]
        perm_q = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        refx_q = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
        perm_q = np.concatenate((perm_q, np.array(perm_q) + len(perm_q))).tolist()
        refx_q = np.concatenate((refx_q, refx_q)).tolist()

        # IMU acceleration and angular velocity
        na = np.array([0, 1, 0])  # Normal vector to reflection/symmetry plane.
        Rr = reflex_matrix(na)  # Reflection matrix
        refx_a_IMU = np.squeeze(Rr @ np.ones((3, 1))).astype(np.int).tolist()
        refx_w_IMU = np.squeeze((-Rr) @ np.ones((3, 1))).astype(np.int).tolist()
        perm_a_IMU, perm_w_IMU = [24, 25, 26], [27, 28, 29]

        # Foot relative positions and velocities
        #            ____RF___|___LF____|___RH______|____LH____|
        # pf=        [0, 1, 2,  3, 4, 5,  6,  7,  8, 9, 10, 11]
        perm_foots = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        refx_foots = scipy.linalg.block_diag(*[Rr] * 4) @ np.ones((12, 1))  # Hips and IMU frames xz planes are coplanar
        refx_foots = np.squeeze(refx_foots).tolist()
        perm_foots = np.concatenate((perm_foots, np.array(perm_foots) + len(perm_foots))).astype(np.int)
        refx_foots = np.concatenate((refx_foots, refx_foots)).astype(np.int)

        # Final permutation and reflections
        perm = perm_q + perm_a_IMU + perm_w_IMU
        perm += (perm_foots + len(perm)).tolist()
        refx = refx_q + refx_a_IMU + refx_w_IMU + refx_foots.tolist()

        # Group instantiation.
        h_in = C2.oneline2matrix(oneline_notation=perm, reflexions=refx)
        Gin_data = C2(h_in)
        # One hot encoding of 16 contact states.
        h_out = C2.oneline2matrix(oneline_notation=[0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15])
        Gout_data = C2(h_out)

        return Gin_data, Gout_data

