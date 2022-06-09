#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/5/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import glob
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
import deep_contact_estimator


class UmichContactDataset(contact_dataset):
    dataset_path = pathlib.Path(deep_contact_estimator.__file__).parents[1].joinpath('dataset')
    leg_names = ["RF", "LF", "RH", "LH"]

    def __init__(self, data_name, label_name, window_size,
                 train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, loss_class_weights=None,
                 use_class_imbalance_w=False, device='cuda', augment=False):

        self.data_path, self.label_path = self.get_full_paths(data_name, label_name, train_ratio=train_ratio,
                                                              test_ratio=test_ratio, val_ratio=val_ratio)

        super().__init__(str(self.data_path), str(self.label_path), window_size, device=device)

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

        if use_class_imbalance_w:
            self.class_weights = 1 - self.contact_state_freq if loss_class_weights is None else loss_class_weights
        else:
            self.class_weights = None
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)

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
        # TP = [torch.logical_and(bin_pred[:, i], bin_gt[:, i]).sum() for i in range(4)]
        # FP = [bin_pred[:, i].sum() - tp for i, tp in enumerate(TP)]
        # TN = [torch.logical_and(torch.logical_not(bin_pred[:, i]), torch.logical_not(bin_gt[:, i])).sum() for i in
        #       range(4)]
        # FN = [torch.logical_not(bin_pred[:, i]).sum() - tn for i, tn in enumerate(TN)]

        # precision_legs = [tp / (tp + fp) for tp, fp in zip(TP, FP)]
        # recall_legs = [tp / (tp + fn) for tp, fn in zip(TP, FN)]


        # precision = {f'precision_{leg}': v for leg, v in zip(leg_names, precision_legs)}
        # recall = {f'recall_{leg}': v for leg, v in zip(leg_names, recall_legs)}
        acc_dir = {f"acc_{leg}": v for leg, v in zip(self.leg_names, acc_per_leg)}
        # TP_dir = {f'TP_{leg}': v.float() for leg, v in zip(leg_names, TP)}
        # FP_dir = {f'FP_{leg}': v.float() for leg, v in zip(leg_names, FP)}
        # TN_dir = {f'TN_{leg}': v.float() for leg, v in zip(leg_names, TN)}
        # FN_dir = {f'FN_{leg}': v.float() for leg, v in zip(leg_names, FN)}

        metrics = {'acc': acc, 'acc_legs_avg': acc_per_leg_avg}
        # metrics.update(precision)
        # metrics.update(recall)
        metrics.update(acc_dir)
        # metrics.update(TP_dir)
        # metrics.update(FP_dir)
        # metrics.update(TN_dir)
        # metrics.update(FN_dir)
        return metrics

    def decimal2binary(self, x):
        mask = 2 ** torch.arange(4 - 1, -1, -1).to(self.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    def test_model(self, model, dataloader):
        from deep_contact_estimator.src.test import compute_accuracy, compute_precision, compute_jaccard, compute_confusion_mat
        test_acc, acc_per_leg, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr = compute_accuracy(dataloader, model)
        precision_of_class, precision_of_legs, precision_of_all_legs = compute_precision(bin_pred_arr, bin_gt_arr,
                                                                                         pred_arr, gt_arr)
        jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs = compute_jaccard(bin_pred_arr, bin_gt_arr, pred_arr,
                                                                                 gt_arr)
        confusion_mat, fn_rate, fp_rate = compute_confusion_mat(bin_pred_arr, bin_gt_arr)

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

    def get_full_paths(self, data_name, label_name, train_ratio: float = 0.7,
                       val_ratio: float = 0.7, test_ratio: float = 0.7) -> (pathlib.Path, pathlib.Path):

        folder_path = pathlib.Path(self.dataset_path.joinpath(f'training/numpy_train_ratio={train_ratio:.3f}'))
        mat_path = pathlib.Path(self.dataset_path.joinpath('training/mat'))

        print(f'Contact Dataset path: {folder_path.absolute()}')
        data_path = folder_path.joinpath(data_name)
        label_path = folder_path.joinpath(label_name)

        if not data_path.exists():
            folder_path.mkdir(exist_ok=True)
            print(f"Generating dataset and saving it to: {folder_path}")
            UmichContactDataset.mat2numpy_split(data_path=mat_path, save_path=folder_path,
                                                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
            print(f"Created dataset partition and saved to: {folder_path}")
            assert data_path.exists(), f"Failed to create partition on {data_path}"

        return data_path, label_path

    @staticmethod
    def mat2numpy_split(data_path: pathlib.Path, save_path: pathlib.Path, train_ratio=0.7, val_ratio=0.15,
                        test_ratio=0.15):
        """
        Load data from .mat file, concatenate into numpy array, and save as train/val/test.
        Inputs:
        - data_pth: path to mat data folder
        - save_pth: path to numpy saving directory.
        - train_ratio: ratio of training data
        - val_ratio: ratio of validation data
        Data should be stored in .mat file, and contain:
        - q: joint encoder value (num_data,12)
        - qd: joint angular velocity (num_data,12)
        - p: foot position from FK (num_data,12)
        - v: foot velocity from FK (num_data,12)
        - imu_acc: linear acceleration from imu (num_data,3)
        - imu_omega: angular velocity from imu (num_data,3)
        - contacts: contact data (num_data,4)
                    contacts are stored as binary values, in the order
                    of right_front, left_front, right_hind, left_hind.
                    FRONT
                    1 0  RIGHT
                    3 2
                    BACK
                    Contact value will be treated as binary values in the
                    order of contacts[0], contacts[1], contacts[2], contacts[3]
                    and be converted to decimal value in this function.
                    Ex. [1,0,0,1] -> 9
                        [0,1,1,0] -> 6

        - tau_est (optional): estimated control torque (num_data,12)
        - F (optional): ground reaction force

        Output:
        -
        """
        import scipy.io as sio

        assert train_ratio + test_ratio + val_ratio <= 1.0, f'{(train_ratio, test_ratio, val_ratio)} > 1.0'

        num_features = 54
        train_data = np.zeros((0, num_features))
        val_data = np.zeros((0, num_features))
        test_data = np.zeros((0, num_features))
        train_label = np.zeros((0, 1))
        val_label = np.zeros((0, 1))
        test_label = np.zeros((0, 1))

        def binary2decimal(a, axis=-1):
            return np.right_shift(np.packbits(a, axis=axis), 8 - a.shape[axis]).squeeze()

        data_path = pathlib.Path(data_path)
        save_path = pathlib.Path(save_path)
        assert data_path.exists(), data_path.absolute()

        save_path.mkdir(exist_ok=True)

        print(f"Creating dataset partition train:{train_ratio}, test:{test_ratio}, val:{val_ratio}")

        # for all dataset in the folder
        for data_name in glob.glob(str(data_path.joinpath('*'))):
            print("loading... ", data_name)
            # load data
            raw_data = sio.loadmat(data_name)

            contacts = raw_data['contacts']
            q = raw_data['q']
            p = raw_data['p']
            qd = raw_data['qd']
            v = raw_data['v']
            acc = raw_data['imu_acc']
            omega = raw_data['imu_omega']

            # tau_est = raw_data['tau_est']
            # F = raw_data['F']

            # concatenate current data. First we try without GRF
            cur_data = np.concatenate((q, qd, acc, omega, p, v), axis=1)

            # separate data into train/val/test
            num_data = np.shape(q)[0]
            num_train = int(train_ratio * num_data)
            num_val = int(val_ratio * num_data)
            num_test = int(test_ratio * num_data)
            assert train_ratio + val_ratio + test_ratio <= 1.0
            cur_val = cur_data[:num_val, :]
            cur_test = cur_data[num_val:num_val + num_test, :]
            cur_train = cur_data[-num_train:, :]

            # stack with all other sequences
            train_data = np.vstack((train_data, cur_train))
            val_data = np.vstack((val_data, cur_val))
            test_data = np.vstack((test_data, cur_test))

            # convert labels from binary to decimal
            cur_label = binary2decimal(contacts).reshape((-1, 1))

            # stack labels
            val_label = np.vstack((val_label, cur_label[:num_val, :]))
            test_label = np.vstack((test_label, cur_label[num_val:num_val + num_test, :]))
            train_label = np.vstack((train_label, cur_label[-num_train:, :]))

            # break
        train_label = train_label.reshape(-1, )
        val_label = val_label.reshape(-1, )
        test_label = test_label.reshape(-1, )

        print(f"Saving data to {save_path.resolve()}")

        np.save(str(save_path.joinpath("train.npy")), train_data)
        np.save(str(save_path.joinpath("val.npy")), val_data)
        np.save(str(save_path.joinpath("test.npy")), test_data)
        np.save(str(save_path.joinpath("train_label.npy")), train_label)
        np.save(str(save_path.joinpath("val_label.npy")), val_label)
        np.save(str(save_path.joinpath("test_label.npy")), test_label)

        print("Generated ", train_data.shape[0], " training data.")
        print("Generated ", val_data.shape[0], " validation data.")
        print("Generated ", test_data.shape[0], " test data.")

        print(train_data.shape[0])
        print(val_data.shape[0])
        print(test_data.shape[0])

