import math
import pathlib
import time
from typing import Union, Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from nn.EquivariantModules import BasisLinear, EquivariantBlock, LinearBlock, EMLP, MLP

import logging
log = logging.getLogger(__name__)

LossCallable = Callable[[torch.Tensor, torch.Tensor, ], torch.Tensor]
MetricCallable = Callable[[torch.Tensor, torch.Tensor, ], dict]

class LightningModel(pl.LightningModule):

    def __init__(self, lr, loss_fn: LossCallable, metrics_fn: MetricCallable, test_epoch_metrics_fn=None,
                 val_epoch_metrics_fn=None, log_preact=False, log_w=False):
        super().__init__()
        # self.model_type = model.__class__.__name__
        self.lr = lr

        # self.model = model
        self._loss_fn = loss_fn
        self.compute_metrics = metrics_fn
        self.test_epoch_metrics_fn = test_epoch_metrics_fn
        self.val_epoch_metrics_fn = val_epoch_metrics_fn
        self._log_w = log_w
        self._log_preact = log_preact
        # Save hyperparams in model checkpoint.
        # TODO: Fix this/home/dordonez/Projects/RobotEquivariantNN/launch/sample_eff
        self.save_hyperparameters()

    def set_model(self, model:[EMLP, MLP]):
        self.model = model
        self.model_type = model.__class__.__name__

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_pred = self.model(x)
        loss = self._loss_fn(y_pred, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=y.shape[0])

        metrics = self.compute_metrics(y_pred, y)
        self.log_metrics(metrics, prefix="train_", batch_size=y.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)
        loss = self._loss_fn(y_pred, y)
        metrics = self.compute_metrics(y_pred, y)

        self.log("val_loss", loss, prog_bar=False, on_epoch=True)
        self.log_metrics(metrics, prefix="val_", batch_size=y.shape[0])

        return {'out': y_pred, 'gt': y}

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)
        loss = self._loss_fn(y_pred, y)
        metrics = self.compute_metrics(y_pred, y)

        self.log("test_loss", loss, prog_bar=False, on_epoch=True)
        self.log_metrics(metrics, prefix="test_", batch_size=y.shape[0])

        return {'out': y_pred, 'gt': y}

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        return self.model(x)

    def log_metrics(self, metrics: dict, prefix='', batch_size=None):
        for k, v in metrics.items():
            name = f"{prefix}{k}"

            self.log(name, v, prog_bar=False, batch_size=batch_size)

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        self.log('time_per_epoch', time.time() - self.epoch_start_time, prog_bar=False, on_epoch=True)
        if self._log_w: self.log_weights()
        if self._log_preact: self.log_preactivations()

    def validation_epoch_end(self, outputs):
        if self.val_epoch_metrics_fn is not None:
            out = [o['out'] for o in outputs]
            gt = [o['gt'] for o in outputs]
            out = torch.cat(out, dim=0)
            gt = torch.cat(gt, dim=0)
            self.val_epoch_metrics_fn([out, gt, self.trainer, self, False, "val_"])

    def test_epoch_end(self, outputs):
        if self.test_epoch_metrics_fn is not None:
            out = [o['out'] for o in outputs]
            gt = [o['gt'] for o in outputs]
            out = torch.cat(out, dim=0)
            gt = torch.cat(gt, dim=0)
            self.test_epoch_metrics_fn([out, gt, self.trainer, self, True, "test_"])

    def on_train_start(self):
        # TODO: Add number of layers and hidden channels dimensions.
        hparams = {'lr': self.lr, 'model': self.model_type}
        if hasattr(self.model, "get_hparams"):
            hparams.update(self.model.get_hparams())
        if self.logger:
            self.logger.log_hyperparams(hparams, {"val_loss": np.NaN, "train_loss_epoch": np.NaN, "test_loss": np.NaN})

    def on_train_end(self) -> None:
        ckpt_call = self.trainer.checkpoint_callback
        if ckpt_call is not None:
            ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
            best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)
            if ckpt_path.exists() and best_path.exists():
                # Remove last model ckpt leave only best, to hint training successful termination.
                ckpt_path.unlink()
                log.info(f"Removing last ckpt {ckpt_path} from successful training run.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_weights(self):
        if not self.logger: return
        tb_logger = self.logger.experiment
        layer_index = 0  # Count layers by linear operators not position in network sequence
        for layer in self.model.net:
            layer_name = f"Layer{layer_index:02d}"
            if isinstance(layer, EquivariantBlock) or isinstance(layer, BasisLinear):
                lin = layer.linear if isinstance(layer, EquivariantBlock) else layer
                W = lin.weight.view(-1).detach()
                basis_coeff = lin.basis_coeff.view(-1).detach()
                tb_logger.add_histogram(tag=f"{layer_name}/c", values=basis_coeff, global_step=self.current_epoch)
                tb_logger.add_histogram(tag=f"{layer_name}/W", values=W, global_step=self.current_epoch)
                layer_index += 1
            elif isinstance(layer, LinearBlock) or isinstance(layer, torch.nn.Linear):
                lin = layer.linear if isinstance(layer, LinearBlock) else layer
                W = lin.weight.view(-1).detach()
                tb_logger.add_histogram(tag=f"{layer_name}/W", values=W, global_step=self.current_epoch)
                layer_index += 1

    def log_preactivations(self, ):
        if not self.logger: return
        tb_logger = self.logger.experiment
        layer_index = 0  # Count layers by linear operators not position in network sequence
        for layer in self.model.net:
            layer_name = f"Layer{layer_index:02d}"
            if isinstance(layer, EquivariantBlock) or isinstance(layer, LinearBlock):
                tb_logger.add_histogram(tag=f"{layer_name}/pre-act", values=layer._preact,
                                        global_step=self.current_epoch)
                layer_index += 1

    def get_metrics(self):
        # don't show the version number on console logs.
        items = super().get_metrics()
        items.pop("v_num", None)
        return items
