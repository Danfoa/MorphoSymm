#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/5/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import pathlib

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F
from deep_contact_estimator.utils.data_handler import contact_dataset
from torch.utils.data._utils.collate import default_collate

from groups.SymmetricGroups import C2
from utils.utils import reflex_matrix, coo2torch_coo


class UmichContactDataset(contact_dataset):

    def __init__(self, data_path: pathlib.Path, label_path: pathlib.Path, window_size, use_class_imbalance_w=False, device='cuda', augment=False):
        assert data_path.exists(), data_path.absolute()
        assert label_path.exists(), label_path.absolute()

        super().__init__(str(data_path), str(label_path), window_size, device=device)
        self.device = device
        self.contact_state_freq = self.get_class_frequency()

        self.Gin, self.Gout = self.get_in_out_groups()
        self.augment = augment
        self.n_contact_states = 16
        if self.augment:
            # self._hin = coo2torch_coo(self.Gin.discrete_generators[0])
            # self._hout = coo2torch_coo(self.Gout.discrete_generators[0])
            self.hin = torch.tensor(self.Gin.discrete_generators[0].todense(), device=device)
            self.hout = torch.tensor(self.Gout.discrete_generators[0].todense(), device=device)

        class_weights = None if not use_class_imbalance_w else self.contact_state_freq
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        # def f(x,y):
        #     r = F.cross_entropy(x,y)
        #     return r

        # self.loss_fn = f

    def collate_fn(self, batch):
        collated_batch = default_collate(batch)

        if self.augment and np.random.rand() > 0.5:
            # (Batch, Window size, features)
            x_batch = collated_batch['data']
            y_batch = F.one_hot(collated_batch['label'], num_classes=self.n_contact_states).to(x_batch.dtype)
            g_x_batch = torch.matmul(x_batch.unsqueeze(1), self.hin.unsqueeze(0).to(x_batch.dtype)).squeeze()
            g_y_batch = torch.matmul(y_batch.unsqueeze(1), self.hout.unsqueeze(0).to(x_batch.dtype)).squeeze()
            # Convert back to numerical class label.
            _, g_y = torch.max(g_y_batch, dim=1)
            # return {"data": g_x_batch, "label": g_y}
            return g_x_batch, g_y
        else:
            # return collated_batch
            return collated_batch['data'], collated_batch['label']

    def get_class_frequency(self):
        classes, counts = torch.unique(self.label, return_counts=True, sorted=True)
        return counts / len(self)

    @torch.no_grad()
    def compute_metrics(self, output: torch.Tensor, gt_label: torch.Tensor) -> dict:
        _, prediction = torch.max(output, dim=1)

        bin_pred = self.decimal2binary(prediction)
        bin_gt = self.decimal2binary(gt_label)

        # P = bin_gt.sum(axis=0)   # Positives
        # N = torch.logical_not(bin_gt).sum(axis=0)  # Negatives

        acc_per_leg = (bin_pred == bin_gt).sum(axis=0) / output.shape[0]
        acc = (prediction == gt_label).sum() / output.shape[0]
        acc_per_leg_avg = torch.sum(acc_per_leg) / 4.0

        # precision_of_class = sklearn.metrics.precision_score(y_pred=prediction, y_true=gt_label, average='weighted',
        #                                                      zero_division=0)
        TP = [torch.logical_and(bin_pred[:, i], bin_gt[:, i]).sum() for i in range(4)]
        FP = [bin_pred[:, i].sum() - tp for i, tp in enumerate(TP)]
        TN = [torch.logical_and(torch.logical_not(bin_pred[:, i]), torch.logical_not(bin_gt[:, i])).sum() for i in range(4)]
        FN = [torch.logical_not(bin_pred[:, i]).sum() - tn for i, tn in enumerate(TN)]

        precision_legs = [tp/(tp + fp) for tp, fp in zip(TP, FP)]
        recall_legs = [tp/(tp + fn) for tp, fn in zip(TP, FN)]

        leg_names = ["RF", "LF", "RH", "LH"]
        precision = {f'precision_{leg}': v for leg, v in zip(leg_names, precision_legs)}
        recall = {f'recall_{leg}': v for leg, v in zip(leg_names, recall_legs)}
        acc_dir = {f"acc_{leg}": v for leg, v in zip(leg_names, acc_per_leg)}
        TP_dir = {f'TP_{leg}': v.float() for leg, v in zip(leg_names, TP)}
        FP_dir = {f'FP_{leg}': v.float() for leg, v in zip(leg_names, FP)}
        TN_dir = {f'TN_{leg}': v.float() for leg, v in zip(leg_names, TN)}
        FN_dir = {f'FN_{leg}': v.float() for leg, v in zip(leg_names, FN)}

        metrics = {'acc': acc, 'acc_legs_avg': acc_per_leg_avg}
        metrics.update(precision)
        metrics.update(recall)
        metrics.update(acc_dir)
        metrics.update(TP_dir)
        metrics.update(FP_dir)
        metrics.update(TN_dir)
        metrics.update(FN_dir)
        return metrics

    def decimal2binary(self, x):
        mask = 2 ** torch.arange(4 - 1, -1, -1).to(self.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

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

