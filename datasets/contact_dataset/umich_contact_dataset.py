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
from omegaconf import OmegaConf, DictConfig

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix

import pathlib
import sys

from ..utils.robot_utils import load_robot_and_symmetries, class_from_name

try:
    deep_contact_estimator_path = pathlib.Path(__file__).parent.absolute()
    assert deep_contact_estimator_path.exists()
    sys.path.append(str(deep_contact_estimator_path / 'deep-contact-estimator/utils'))
    sys.path.append(str(deep_contact_estimator_path / 'deep-contact-estimator/src'))
    from data_handler import contact_dataset
    from contact_cnn import contact_cnn   # just to trigger the error here
except ImportError as e:
    raise ImportError("Deep Contact Estimator submodule not initialized, run `git submodule update "
                      "--init --recursive --progress") from e

from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from groups.SymmetryGroups import C2
from utils.algebra_utils import reflection_matrix, coo2torch_coo


class UmichContactDataset(contact_dataset):
    dataset_path = pathlib.Path(__file__).parent
    leg_names = ["RF", "LF", "RH", "LH"]
    states_names = ['-', 'LH', 'RH', 'RH-LH', 'LF', 'LF-LH', 'LF-RH', 'LF-RH-LH', 'RF', 'RF-LH', 'RF-RH', 'RF-RH-LH',
                    'RF-LF', 'RF-LF-LH', 'RF-LF-RH', 'RF-LF-RH-LH']

    def __init__(self, data_name, label_name, window_size, robot_cfg,
                 train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, loss_class_weights=None,
                 use_class_imbalance_w=False, device='cuda', augment=False, debug=False, partition='training'):
        # Sub folder in dataset folder containing the mat/*.mat and numpy/*.npy
        self.partition = partition
        self.data_path, self.label_path = self.get_full_paths(data_name, label_name, train_ratio=train_ratio,
                                                              val_ratio=val_ratio)
        self.robot_cfg = robot_cfg
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

        self.rep_in, self.rep_out = self.get_in_out_symmetry_groups_reps(robot_cfg)
        self.augment = augment
        self.n_contact_states = 16
        # if self.augment:
            # self._hin = coo2torch_coo(self.Gin.discrete_generators[0])
            # self._hout = coo2torch_coo(self.Gout.discrete_generators[0])
        self.hin = torch.tensor(self.rep_in.G.discrete_generators[0].todense(), device=device)
        self.hout = torch.tensor(self.rep_out.G.discrete_generators[0].todense(), device=device)

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
        model.eval()

        # acc_dir = self.compute_metrics(y_pred, y_gt)
        _, prediction = torch.max(y_pred, dim=-1)
        bin_pred_arr = self.decimal2binary(prediction).detach().cpu().numpy()
        bin_gt_arr = self.decimal2binary(y_gt).detach().cpu().numpy()
        pred_arr = prediction.detach().cpu().numpy()
        gt_arr = y_gt.detach().cpu().numpy()

        # test_acc, acc_per_leg, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr = self.compute_accuracy(dataloader, model)
        precision_of_class, precision_of_legs, precision_of_all_legs = self.compute_precision(bin_pred_arr, bin_gt_arr,
                                                                                              pred_arr, gt_arr)
        f1score_of_class = sklearn.metrics.f1_score(gt_arr, pred_arr, average='weighted')

        jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs = self.compute_jaccard(bin_pred_arr, bin_gt_arr,
                                                                                      pred_arr, gt_arr)
        confusion_mat, rates = self.compute_confusion_mat(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr)

        TPR = [rates[f'{k}/TP'] / np.sum(bin_gt_arr[:, i] == 1) for i, k in enumerate(self.leg_names)]
        TNR = [rates[f'{k}/TN'] / np.sum(bin_gt_arr[:, i] == 0) for i, k in enumerate(self.leg_names)]
        FNR = [1 - tpr for tpr in TPR]
        FPR = [1 - tpr for tpr in TNR]

        balanced_acc_of_legs = [(tpr + tnr)/2 for tpr, tnr in zip(TPR, TNR)]
        recall_of_legs = [rates[f'{k}/TP']/(rates[f'{k}/TP'] + rates[f'{k}/FP']) for k in self.leg_names]
        f1score_of_legs = [2*(p * r)/(r + p) for p, r in zip(precision_of_legs, recall_of_legs)]
        pred_pos_cond_rate_of_legs = [(rates[f'{k}/TP'] + rates[f'{k}/FP']) /
                                       (rates[f'{k}/TP'] + rates[f'{k}/FP'] + rates[f'{k}/TN'] + rates[f'{k}/FN']) for k in self.leg_names]


        individual_state_metrics = {}
        if prefix == "test_":

            for state_id, state_name in enumerate(self.states_names):
                precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_true=gt_arr,
                                                                                                 y_pred=pred_arr,
                                                                                                 labels=[state_id])
                individual_state_metrics[f"contact_state/{state_name}/precision"] = float(precision)
                individual_state_metrics[f"contact_state/{state_name}/recall"] = float(recall)
                individual_state_metrics[f"contact_state/{state_name}/f1"] = float(f1)
                individual_state_metrics[f"contact_state/{state_name}/support"] = int(np.sum(gt_arr == state_id))

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
        f1_score_dir = {f"{leg}/f1": v for leg, v in zip(self.leg_names, f1score_of_legs)}
        balanced_acc_dir = {f"{leg}/balanced_acc": a for leg, a in zip(self.leg_names, balanced_acc_of_legs)}

        metrics = {'contact_state/f1': f1score_of_class,
                   'contact_state/precision': precision_of_class,
                   'contact_state/jaccard': jaccard_of_class,
                   'legs_avg/precision': precision_of_all_legs,
                   'legs_avg/jaccard': jaccard_of_all_legs,
                   'legs_avg/f1': np.sum(f1score_of_legs) / 4.0,
                   'legs_avg/recall': np.sum(recall_of_legs) / 4.0,
                   'legs_avg/balanced_acc': np.sum(balanced_acc_of_legs) / 4.0,
                   'legs_avg/specificity': np.sum(TNR) / 4.0,
                   'legs_avg/FPR': np.sum(FPR) / 4.0,
                   'legs_avg/FNR': np.sum(FNR) / 4.0,
                   'legs_avg/pred_pos_cond_rate': np.sum(pred_pos_cond_rate_of_legs) / 4.0,
                   }

        metrics.update(precision_dir)
        metrics.update(jaccard_dir)
        metrics.update(recall_dir)
        metrics.update(f1_score_dir)
        metrics.update(balanced_acc_dir)
        metrics.update(individual_state_metrics)

        model.log_metrics(metrics, prefix=prefix)
        model.train()

    def decimal2binary(self, x):
        mask = 2 ** torch.arange(4 - 1, -1, -1).to(self.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    @staticmethod
    def get_in_out_symmetry_groups_reps(robot_cfg: DictConfig):
        from groups.SparseRepresentation import SparseRep
        from utils.robot_utils import get_robot_params
        robot, rep_E3, rep_QJ = load_robot_and_symmetries(robot_cfg)
        G_class = class_from_name('groups.SymmetryGroups',
                                  robot_cfg.G if robot_cfg.gens_ids is None else robot_cfg.G_sub)

        # Configure input x and output y representations
        # x = [qj, dqj, a, w, pf, vf] - a: linear IMU acc, w: angular IMU velocity, pf:feet positions, vf:feet velocity
        rep_a = rep_E3.set_pseudovector(False)
        rep_w = rep_E3.set_pseudovector(True)

        # Configure pf, vf ∈ R^12  representations composed of reflections and permutations
        n_legs = 4
        rep_legs_reflected = n_legs * rep_E3.set_pseudovector(False)    # Same representation as the hips ref frames are collinear with base.
        G_legs_reflected = rep_legs_reflected.G
        g_q_perm = abs(rep_QJ.G.discrete_generators[0])  # Permutation swapping legs.
        G_pf = G_class(generators=g_q_perm @ G_legs_reflected.discrete_generators[0])
        rep_pf = SparseRep(G_pf)
        rep_vf = SparseRep(G_pf)

        # x   = [ qj   ,  dqj   ,   a   ,   w   ,   pf   ,   vf  ]
        rep_x = rep_QJ + rep_QJ + rep_a + rep_w + rep_pf + rep_vf
        # y : 16 dimensional contact state with following symmetry. See paper abstract.
        g_y = C2.oneline2matrix(oneline_notation=[0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15])
        G_y = C2(g_y)
        rep_y = SparseRep(G_y)

        rep_data_in, rep_data_out = rep_x, rep_y
        return rep_data_in, rep_data_out

    def get_full_paths(self, data_name, label_name, train_ratio: float = 0.7, val_ratio: float = 0.7) -> (pathlib.Path, pathlib.Path):

        folder_path = pathlib.Path(self.dataset_path / f'{self.partition}/numpy_train_ratio={train_ratio:.3f}')
        training_mat_path = pathlib.Path(self.dataset_path / f'{self.partition}/mat')
        test_mat_path = pathlib.Path(self.dataset_path / f'{self.partition}/mat_test')

        data_path = folder_path.joinpath(data_name)
        label_path = folder_path.joinpath(label_name)
        print(f'Contact Dataset path: \n\t- Data: {data_path} \n\t- Labels: {label_path}')

        if not data_path.exists():  # Data is not there generate it.
            folder_path.mkdir(exist_ok=True)
            print(f"Generating dataset and saving it to: {folder_path}")
            UmichContactDataset.mat2numpy_split(train_val_data_path=training_mat_path, test_data_path=test_mat_path,
                                                save_path=folder_path, train_ratio=train_ratio, val_ratio=val_ratio)
            print(f"Created dataset partition and saved to: {folder_path}")
            assert data_path.exists(), f"Failed to create partition on {data_path}"

        return data_path, label_path

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

    @staticmethod
    def mat2numpy_split(train_val_data_path: pathlib.Path, test_data_path: pathlib.Path, save_path: pathlib.Path,
                        train_ratio=0.7, val_ratio=0.15):
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
        assert train_ratio + val_ratio <= 1.0, f'{(train_ratio, val_ratio)} > 1.0'

        data_path = pathlib.Path(train_val_data_path)
        test_data_path = pathlib.Path(test_data_path)
        save_path = pathlib.Path(save_path)
        assert data_path.exists(), f"Train and Val .mat files required, and not found on {data_path.absolute()}"
        assert test_data_path.exists(), f"Test .mat files required, and not found on {test_data_path.absolute()}"
        assert data_path != test_data_path

        save_path.mkdir(exist_ok=True)

        print(f"\nCreating dataset partition: train:{train_ratio}, val:{val_ratio} from {data_path}")
        (train_data, val_data), (train_label, val_label) = UmichContactDataset.load_and_split_mat_files(data_path=train_val_data_path,
                                                                                                        partitions_ratio=(train_ratio, val_ratio),
                                                                                                        partitions_name=("train", "val"))
        print(f"\nCreating dataset test from {test_data_path}")
        (test_data,), (test_label,) = UmichContactDataset.load_and_split_mat_files(data_path=test_data_path,
                                                                                   partitions_ratio=(1.0,),
                                                                                   partitions_name=("test",))

        print(f"Saving data to {save_path.resolve()}")

        np.save(str(save_path.joinpath("train.npy")), train_data)
        np.save(str(save_path.joinpath("val.npy")), val_data)
        np.save(str(save_path.joinpath("test.npy")), test_data)
        np.save(str(save_path.joinpath("train_label.npy")), train_label)
        np.save(str(save_path.joinpath("val_label.npy")), val_label)
        np.save(str(save_path.joinpath("test_label.npy")), test_label)

    @staticmethod
    def load_and_split_mat_files(data_path: pathlib.Path, partitions_ratio=(0.85, 0.15),
                                 partitions_name=("train", "val")):
        partitions_data = [None] * len(partitions_ratio)
        partitions_labels = [None] * len(partitions_ratio)
        data_files = list(data_path.glob("*.mat"))
        assert len(data_files) > 1, f"No .mat files found in {data_path.absolute()}"
        assert sum(partitions_ratio) <= 1.0, f"the partitions should add up to less than 100% of the data"

        print(f"Loading data from {data_path}")
        print(f"Dataset .mat files found: {[pathlib.Path(d).name for d in data_files]}")
        # for all dataset in the folder
        all_samples = 0
        for data_name in glob.glob(str(data_path.joinpath('*'))):
            # load data
            raw_data = scipy.io.loadmat(data_name)

            contacts = raw_data['contacts']
            q = raw_data['q']
            p = raw_data['p']
            qd = raw_data['qd']
            v = raw_data['v']
            acc = raw_data['imu_acc']
            omega = raw_data['imu_omega']

            # concatenate current data. First we try without GRF
            cur_data = np.concatenate((q, qd, acc, omega, p, v), axis=1)
            # convert labels from binary to decimal
            def binary2decimal(a, axis=-1):
                return np.right_shift(np.packbits(a, axis=axis), 8 - a.shape[axis]).squeeze()
            cur_label = binary2decimal(contacts).reshape((-1, 1))

            # separate data into given paritions
            num_data = np.shape(q)[0]
            all_samples += num_data
            partitions_size = [int(ratio * num_data) for ratio in partitions_ratio]

            lower_lim = 0
            for partition_id, (partition_size, partition_name) in enumerate(zip(partitions_size, partitions_name)):
                if partitions_data[partition_id] is None:
                    partitions_data[partition_id] = []
                    partitions_labels[partition_id] = []
                partitions_data[partition_id].append(cur_data[lower_lim:lower_lim + partition_size, :])
                partitions_labels[partition_id].append(cur_label[lower_lim:lower_lim + partition_size, :])
                lower_lim += partition_size

        partitions_data = [np.vstack(data_list) for data_list in partitions_data]
        partitions_labels = [np.vstack(label_list).reshape(-1, ) for label_list in partitions_labels]

        for name, ratio, data in zip(partitions_name, partitions_ratio, partitions_data):
            # assert ratio * all_samples == data.shape[0]
            print(f"\t - {name} ({ratio*100:.1f}%) = {data.shape[0]:d} samples --> {data.shape[0]/all_samples*100:.1f}%")
        return partitions_data, partitions_labels

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

    @staticmethod
    def compute_precision(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr):

        precision_of_class = precision_score(gt_arr, pred_arr, average='weighted')
        precision_of_all_legs = precision_score(bin_gt_arr.flatten(), bin_pred_arr.flatten())
        precision_of_legs = []
        for i in range(4):
            precision_of_legs.append(precision_score(bin_gt_arr[:, i], bin_pred_arr[:, i]))

        return precision_of_class, precision_of_legs, precision_of_all_legs

    @staticmethod
    def compute_jaccard(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr):

        jaccard_of_class = jaccard_score(gt_arr, pred_arr, average='weighted')
        jaccard_of_all_legs = jaccard_score(bin_gt_arr.flatten(), bin_pred_arr.flatten())
        jaccard_of_legs = []
        for i in range(4):
            jaccard_of_legs.append(jaccard_score(bin_gt_arr[:, i], bin_pred_arr[:, i]))

        return jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs

    def plot_statistics(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        y = self.label.detach().cpu().numpy()
        bin_gt = self.decimal2binary(self.label).detach().cpu().numpy()

        df = pd.DataFrame({'contact_state': np.asarray(self.states_names)[y]})
        df['source'] = 'gt'

        df2 = pd.DataFrame({'contact_state': np.asarray(self.states_names)[y]})
        df2['source'] = 'pred'

        df = pd.concat((df, df2), axis=0, ignore_index=True)

        df['contact_state'] = pd.Categorical(df['contact_state'], self.states_names)
        n_colors = 1 if 'source' not in df else len(np.unique(df['source']))
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        sns.histplot(data=df, x='contact_state', hue='source' if 'source' in df else None, color='magma',
                     palette=sns.color_palette("magma_r", n_colors), bins=16, stat='probability', ax=ax, discrete=True,
                     multiple="dodge")
        ax.set_title(f'{self.data_path.stem} {len(self)} samples')
        ax.set(yscale='log')
        ax.tick_params(axis='x', rotation=70)
        plt.tight_layout()
        plt.show()

        # fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), dpi=150)
        # for leg_name, i, ax in zip(self.leg_names, range(4), axs.flatten()):
        #     sns.histplot(x=bin_gt[:, i], bins=2, discrete=True, shrink=.9, stat='probability', ax=ax)
        #     ax.set_title(leg_name)
        # fig.suptitle(f'{self.data_path.stem} {len(self)} samples')
        # plt.show()
        # pass


