# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import functools
import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn

import logging

from utils.utils import append_dictionaries

log = logging.getLogger(__name__)


#############################################################################################
#                           KFold Loop / Cross Validation Example                           #
# This example demonstrates how to leverage Lightning Loop Customization introduced in v1.5 #
# Learn more about the loop structure from the documentation:                               #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html                  #
#############################################################################################


#############################################################################################
#                           Step 1 / 5: Define KFold DataModule API                         #
# Our KFold DataModule requires to implement the `setup_folds` and `setup_fold_index`       #
# methods.                                                                                  #
#############################################################################################


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


@dataclass
class KFoldDataModule(BaseKFoldDataModule):
    """
     The `KFoldDataModule` will take a train and test dataset.
     On `setup_folds`, folds will be created depending on the provided argument `num_folds`
     Our `setup_fold_index`, the provided train dataset will be split accordingly to
     the current fold split.
    """
    dataset: Optional[Dataset] = None
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def __init__(self, dataset: Dataset, test_samples, num_folds=5, batch_size=256):
        super().__init__()
        self.dataset = dataset
        self.test_samples = test_samples
        self.num_folds = num_folds
        self.batch_size = batch_size
        # self.setup()

    def prepare_data(self) -> None:
        assert self.dataset is not None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.test_dataset = random_split(self.dataset,
                                                             [len(self.dataset) - self.test_samples, self.test_samples])
        self.setup_folds(self.num_folds)
        self.setup_fold_index(0)
        log.info(str(self))

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

        log.info(f"{self.__class__.__name__}: Switching to fold {fold_index}, n_train:{len(self.train_fold)}, "
                 f"n_val:{len(self.val_fold)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.__class__.__name__}: n_folds:{self.num_folds}, n_train:{len(self.train_fold)}, " \
               f"n_test:{len(self.test_dataset)}, n_val:{len(self.val_fold)}"


# #############################################################################################
# #                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# # The `EnsembleVotingModel` will take our custom LightningModule and                        #
# # several checkpoint_paths.                                                                 #
# #                                                                                           #
# #############################################################################################
#
#
# class EnsembleVotingModel(LightningModule):
#     def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
#         super().__init__()
#         # Create `num_folds` models with their associated fold weights
#         self.pl_models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
#         self.test_acc = Accuracy()
#
#     def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
#         # Compute the averaged predictions over the `num_folds` models.
#         logits = torch.stack([m(batch[0]) for m in self.pl_models]).mean(0)
#         loss = F.nll_loss(logits, batch[1])
#         self.test_acc(logits, batch[1])
#         self.log("test_acc", self.test_acc)
#         self.log("test_loss", loss)


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#                                                                                           #
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################


class KFoldValidation:
    def __init__(self, model: pl.LightningModule,
                 trainer_kwargs: dict, kfold_data_module: BaseKFoldDataModule,
                 num_folds: int, export_path: str, reinitialize=False, run_name=None) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path
        self.pl_model = model
        self.trainer_kwargs = trainer_kwargs
        self.kfold_datamodule = kfold_data_module
        self.run_name = '.' if run_name is None else run_name
        self.reset()
        # Store the weights of the model, to use the exact same initial parameters in all folds
        self.lightning_module_state_dict = deepcopy(self.pl_model.state_dict())
        self.trainer = None
        self.reinitialize = reinitialize


    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        """Nothing to reset in this loop."""
        # Reset data module to have a notion of train/test/val DataLoaders properties
        self.kfold_datamodule.setup()
        self.kfold_datamodule.setup_folds(num_folds=self.num_folds)
        self.kfold_datamodule.setup_fold_index(self.current_fold)

        self.metrics = {fold: {} for fold in range(self.num_folds)}

    def run(self):
        self.reset()

        while not self.done:
            self.on_fold_start()
            self.advance()
            self.on_fold_end()
            self.current_fold += 1  # increment fold tracking number.

        return self.on_run_end()

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.kfold_datamodule, BaseKFoldDataModule)
        self.kfold_datamodule.setup_folds(self.num_folds)

    def on_fold_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        log.info(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.kfold_datamodule, BaseKFoldDataModule)
        self.kfold_datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        # TODO: Almost impossible to reset trainer without reading 900 pages of documentation.
        tb_logger = TensorBoardLogger(save_dir=".", name=f'{self.run_name}', version=self.current_fold)
        ckpt_callback = ModelCheckpoint(monitor="hp/val_loss", mode='min', dirpath=f'{self.run_name}', 
                                        filename=f"best_model_of_fold_{self.current_fold}", save_weights_only=True)
        early_stop_callback = EarlyStopping(monitor="hp/val_loss",
                                            patience=int(self.trainer_kwargs['max_epochs'] * 0.2), mode='min')
        self.trainer_kwargs["logger"] = tb_logger
        self.trainer_kwargs["callbacks"] = [ckpt_callback, early_stop_callback]
        self.trainer_kwargs["deterministic"] = True
        self.trainer_kwargs['enable_progress_bar'] = False
        # self.trainer_kwargs['enable_model_summary'] = False

        self.trainer = Trainer(**self.trainer_kwargs)

        self.trainer.fit(model=self.pl_model, datamodule=self.kfold_datamodule)
        # Store the logged metrics
        self.metrics[self.current_fold].update({'train': self.trainer.logged_metrics})

        # Load the best val-detected parameters of fold training. Use these for validation and testing.
        self.pl_model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
        # Store the logged metrics
        self.trainer.validate(model=self.pl_model, datamodule=self.kfold_datamodule)
        self.metrics[self.current_fold].update({'val': self.trainer.logged_metrics})


    def on_fold_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        reduce_test_metrics = False
        if reduce_test_metrics:
            self.trainer.predict(model=self.pl_model, dataloaders=self.kfold_datamodule.test_dataloader(),)
            self.metrics[self.current_fold].update({'test': self.trainer.logged_metrics})
        else:
            torch.set_grad_enabled(False)
            metrics = {}
            for x, y in self.kfold_datamodule.test_dataloader():
                y_pred = self.pl_model(x)
                losses = torch.linalg.norm(y_pred - y, dim=-1)
                metrics['test_loss'] = losses if metrics.get('test_loss') is None else torch.hstack((losses,
                                                                                                     metrics['test_loss']))
                dat_metrics = self.kfold_datamodule.dataset.compute_metrics(y, y_pred)
                for k, v in dat_metrics.items():
                    metrics[f"test_{k}"] = v if metrics.get(f"test_{k}") is None else torch.hstack((metrics[f"test_{k}"],
                                                                                                    dat_metrics[k]))

        self.metrics[self.current_fold]['test'] = metrics

        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        torch.set_grad_enabled(True)

        # restore the original weights + optimizers and schedulers.
        self.pl_model.load_state_dict(self.lightning_module_state_dict)
        if self.reinitialize: # TODO: Avoid assuming custom LightningModel
            self.pl_model.model.reset_parameters()

    def on_run_end(self) -> dict:
        """Used to compute the performance of the ensemble model on the test set."""
        return self.metrics

    def summary(self) -> dict:
        add_dicts = lambda d1, d2, : {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) & set(d2)}
        scale_dict = lambda d, cnt: {k: v/cnt for k, v in d.items()}

        summary_metrics = copy.copy(self.metrics[0])
        for stage in ['train', 'test', 'val']:
            for fold in range(1, self.num_folds):
                fold_metrics = self.metrics[fold][stage]
                summary_metrics[stage] = add_dicts(summary_metrics[stage], fold_metrics)
            summary_metrics[stage] = scale_dict(summary_metrics[stage], self.num_folds)
        return summary_metrics

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer_kwargs.state.fn = TrainerFn.FITTING
        self.trainer_kwargs.training = True

    def _reset_testing(self) -> None:
        self.trainer_kwargs.state.fn = TrainerFn.TESTING
        self.trainer_kwargs.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    @staticmethod
    def summarize_results(results: dict):
        summaries = {}
        folds = []
        for run_key, run_results in results.items():
            folds = list(run_results.keys())
            assert len(folds) > 0 and isinstance(folds[0], int)
            run_summary = functools.reduce(append_dictionaries, list(run_results.values()))
            summaries[run_key] = run_summary

        return summaries