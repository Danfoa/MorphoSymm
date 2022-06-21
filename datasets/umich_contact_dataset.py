#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/5/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import glob
import io
import pathlib
import time

import PIL
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn.functional as F
from deep_contact_estimator.utils.data_handler import contact_dataset
from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from groups.SymmetricGroups import C2
from utils.utils import reflex_matrix, coo2torch_coo
import deep_contact_estimator


class UmichContactDataset(contact_dataset):
    dataset_path = pathlib.Path(deep_contact_estimator.__file__).parents[1].joinpath('dataset')
    leg_names = ["RF", "LF", "RH", "LH"]
    states_names = ['-', 'LH', 'RH', 'RH-LH', 'LF', 'LF-LH', 'LF-RH', 'LF-RH-LH', 'RF', 'RF-LH', 'RF-RH', 'RF-RH-LH',
                    'RF-LF', 'RF-LF-LH', 'RF-LF-RH', 'RF-LF-RH-LH']

    def __init__(self, data_name, label_name, window_size,
                 train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, loss_class_weights=None,
                 use_class_imbalance_w=False, device='cuda', augment=False, debug=False):

        self.data_path, self.label_path = self.get_full_paths(data_name, label_name, train_ratio=train_ratio,
                                                              test_ratio=test_ratio, val_ratio=val_ratio)

        # super().__init__(str(self.data_path), str(self.label_path), window_size, device=device)
        trials = 5
        data, label = None, None
        while data is None and trials > 0:
            try:
                data = np.load(str(self.data_path))
                label = np.load(str(self.label_path))
            except:
                trials -= 1
                time.sleep(np.random.random())


        self.num_data = (data.shape[0]-window_size+1)
        self.window_size = window_size
        self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        self.label = torch.from_numpy(label).type('torch.LongTensor').to(device)
        # ----
        self.device = device

        self.contact_state_freq = self.get_class_frequency()

        self.Gin, self.Gout = self.get_in_out_groups()
        self.augment = augment
        self.n_contact_states = 16
        # if self.augment:
            # self._hin = coo2torch_coo(self.Gin.discrete_generators[0])
            # self._hout = coo2torch_coo(self.Gout.discrete_generators[0])
        self.hin = torch.tensor(self.Gin.discrete_generators[0].todense(), device=device)
        self.hout = torch.tensor(self.Gout.discrete_generators[0].todense(), device=device)

        if use_class_imbalance_w:
            self.class_weights = 1 - self.contact_state_freq if loss_class_weights is None else loss_class_weights
        else:
            self.class_weights = None
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        if debug:
            self.plot_statistics()

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
        _, prediction = torch.max(output, dim=-1)

        bin_pred = self.decimal2binary(prediction)
        bin_gt = self.decimal2binary(gt_label)

        acc_per_leg = (bin_pred == bin_gt).sum(axis=0) / output.shape[0]
        acc = (prediction == gt_label).sum() / output.shape[0]
        acc_per_leg_avg = torch.sum(acc_per_leg) / 4.0

        acc_dir = {f"{leg}/acc": v for leg, v in zip(self.leg_names, acc_per_leg)}
        metrics = {'contact_state/acc': acc, 'legs_avg/acc': acc_per_leg_avg}
        metrics.update(acc_dir)
        return metrics

    def test_metrics(self, y_pred, y_gt, trainer: Trainer, model, log_imgs=False, prefix="test_"):
        from deep_contact_estimator.src.test import compute_precision, compute_jaccard
        model.eval()

        # acc_dir = self.compute_metrics(y_pred, y_gt)
        _, prediction = torch.max(y_pred, dim=-1)
        bin_pred_arr = self.decimal2binary(prediction).detach().cpu().numpy()
        bin_gt_arr = self.decimal2binary(y_gt).detach().cpu().numpy()
        pred_arr = prediction.detach().cpu().numpy()
        gt_arr = y_gt.detach().cpu().numpy()

        # test_acc, acc_per_leg, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr = self.compute_accuracy(dataloader, model)
        precision_of_class, precision_of_legs, precision_of_all_legs = compute_precision(bin_pred_arr, bin_gt_arr,
                                                                                         pred_arr, gt_arr)
        jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs = compute_jaccard(bin_pred_arr, bin_gt_arr, pred_arr,
                                                                                 gt_arr)
        confusion_mat, rates = self.compute_confusion_mat(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr)

        TPR = [rates[f'{k}/TP'] / np.sum(bin_gt_arr[:, i] == 1) for i, k in enumerate(self.leg_names)]
        TNR = [rates[f'{k}/TN'] / np.sum(bin_gt_arr[:, i] == 0) for i, k in enumerate(self.leg_names)]

        balanced_acc_of_legs = [(tpr + tnr)/2 for tpr, tnr in zip(TPR, TNR)]
        recall_of_legs = [rates[f'{k}/TP']/(rates[f'{k}/TP'] + rates[f'{k}/FP']) for k in self.leg_names]
        f1_score_of_legs = [2*(p * r)/(r + p) for p, r in zip(precision_of_legs, recall_of_legs)]

        if log_imgs:
            import matplotlib.pyplot as plt
            import seaborn as sns
            def log_cm_img(label, classes, figsize=(4, 3), annot=True):
                # If Trainer is present log images
                df_cfm = pd.DataFrame(confusion_mat[label], index=classes, columns=classes)
                fig = plt.figure(figsize=figsize, dpi=80)
                cfm_plot = sns.heatmap(df_cfm, annot=annot, fmt=".2f", vmin=0, vmax=1)
                fig.tight_layout()
                # plt.show()
                buf = io.BytesIO()
                plt.savefig(buf, format='jpeg')
                buf.seek(0)
                img = PIL.Image.open(buf)
                plt.close(fig)

                if trainer.logger:
                    from torchvision.transforms import ToTensor
                    tensorboard = trainer.logger.experiment
                    tensorboard.add_image(label, ToTensor()(img))

            log_cm_img(label='cm/contact_state', classes=self.states_names, annot=False, figsize=(8, 6))
            for key, cm in confusion_mat.items():
                if 'contact_state' in key: continue
                log_cm_img(label=key, classes=[0, 1], annot=True)

        precision_dir = {f"{leg}/precision": v for leg, v in zip(self.leg_names, precision_of_legs)}
        jaccard_dir = {f"{leg}/jaccard": v for leg, v in zip(self.leg_names, jaccard_of_legs)}
        recall_dir = {f"{leg}/recall": v for leg, v in zip(self.leg_names, recall_of_legs)}
        f1_score_dir = {f"{leg}/f1": v for leg, v in zip(self.leg_names, f1_score_of_legs)}
        balanced_acc_dir = {f"{leg}/balanced_acc": a for leg, a in zip(self.leg_names, balanced_acc_of_legs)}

        metrics = {'contact_state/precision': precision_of_class,
                   'contact_state/jaccard': jaccard_of_class,
                   'legs_avg/precision': precision_of_all_legs,
                   'legs_avg/jaccard': jaccard_of_all_legs,
                   'legs_avg/f1': np.sum(f1_score_of_legs) / 4.0,
                   'legs_avg/recall': np.sum(recall_of_legs) / 4.0,
                   'legs_avg/balanced_acc': np.sum(balanced_acc_of_legs) / 4.0,
                   }

        metrics.update(precision_dir)
        metrics.update(jaccard_dir)
        metrics.update(recall_dir)
        metrics.update(f1_score_dir)
        metrics.update(balanced_acc_dir)

        model.log_metrics(metrics, prefix=prefix)
        model.train()

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
        # One hot encoding of 16 contact_hp_ecnn states.
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
        - contacts: contact_hp_ecnn data (num_data,4)
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

    def compute_accuracy(self, dataloader, model):
        # compute accuracy in batch

        num_correct = 0
        num_data = 0
        correct_per_leg = np.zeros(4)
        bin_pred_arr = np.zeros((0, 4))
        bin_gt_arr = np.zeros((0, 4))
        pred_arr = np.zeros((0))
        gt_arr = np.zeros((0))
        with torch.no_grad():
            for input_data, gt_label in tqdm(dataloader):

                output = model(input_data)

                _, prediction = torch.max(output, 1)

                bin_pred = self.decimal2binary(prediction)
                bin_gt = self.decimal2binary(gt_label)

                bin_pred_arr = np.vstack((bin_pred_arr, bin_pred.cpu().numpy()))
                bin_gt_arr = np.vstack((bin_gt_arr, bin_gt.cpu().numpy()))

                pred_arr = np.hstack((pred_arr, prediction.cpu().numpy()))
                gt_arr = np.hstack((gt_arr, gt_label.cpu().numpy()))

                correct_per_leg += (bin_pred == bin_gt).sum(axis=0).cpu().numpy()
                num_data += input_data.size(0)
                num_correct += (prediction == gt_label).sum().item()

        return num_correct / num_data, correct_per_leg / num_data, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr

    def compute_confusion_mat(self, bin_contact_pred_arr, bin_contact_gt_arr, pred_state, gt_state):

        confusion_mat = {}

        confusion_mat['cm/contact_state'] = confusion_matrix(pred_state, gt_state, labels=list(range(self.n_contact_states)), normalize='true')
        for leg_id, leg_name in enumerate(self.leg_names):
            confusion_mat[f'cm/leg_{leg_name}'] = confusion_matrix(bin_contact_gt_arr[:, leg_id], bin_contact_pred_arr[:, leg_id], labels=[0, 1], normalize='true')

        # false negative and false postivie rate
        # false negative = FN/P; false positive = FP/N
        rates = {}
        for key, cm in confusion_mat.items():
            if 'contact_state' in key: continue
            tn, fp, fn, tp = cm.ravel()
            leg_name = key.split('_')[1]
            rates[f'{leg_name}/TP'] = tp
            rates[f'{leg_name}/FN'] = fn
            rates[f'{leg_name}/FP'] = fp
            rates[f'{leg_name}/TN'] = tn

        return confusion_mat, rates

    def plot_statistics(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        y = self.label.detach().cpu().numpy()
        bin_gt = self.decimal2binary(self.label).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        sns.histplot(x=np.asarray(self.states_names)[y], bins=16, stat='probability', ax=ax, discrete=True, shrink=.9)
        ax.set_title(f'{self.data_path.stem} {len(self)} samples')
        ax.tick_params(axis='x', rotation=70)
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), dpi=150)
        for leg_name, i, ax in zip(self.leg_names, range(4), axs.flatten()):
            sns.histplot(x=bin_gt[:, i], bins=2, discrete=True, shrink=.9, stat='probability', ax=ax)
            ax.set_title(leg_name)
        fig.suptitle(f'{self.data_path.stem} {len(self)} samples')
        plt.show()


