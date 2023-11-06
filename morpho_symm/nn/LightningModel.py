import logging
import pathlib
import time
from typing import Callable

import pytorch_lightning as pl
import torch
from escnn.nn import EquivariantModule

from morpho_symm.nn.EMLP import EMLP
from morpho_symm.nn.MLP import MLP
from morpho_symm.utils.mysc import flatten_dict

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
        # TODO: Fix this/home/dordonez/Projects/MorphoSymm/launch/sample_eff
        self.save_hyperparameters()

    def set_model(self, model: [EMLP, MLP]):
        self.model = model
        self.model_type = model.__class__.__name__
        self.equivariant = isinstance(model, EquivariantModule)

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x if not self.equivariant else self.model.in_type(x))
        y_pred = y_pred if not self.equivariant else y_pred.tensor
        loss = self._loss_fn(y_pred, y)

        metrics = self.compute_metrics(y_pred, y)
        self.log_metrics(metrics, suffix="train", batch_size=y.shape[0])
        self.log("loss/train", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x if not self.equivariant else self.model.in_type(x))
        y_pred = y_pred if not self.equivariant else y_pred.tensor
        loss = self._loss_fn(y_pred, y)

        metrics = self.compute_metrics(y_pred, y)
        self.log_metrics(metrics, suffix="val", batch_size=y.shape[0])
        self.log("loss/val", loss, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x if not self.equivariant else self.model.in_type(x))
        y_pred = y_pred if not self.equivariant else y_pred.tensor
        loss = self._loss_fn(y_pred, y)

        metrics = self.compute_metrics(y_pred, y)
        self.log_metrics(metrics, suffix="test", batch_size=y.shape[0])
        self.log("loss/test", loss, prog_bar=False)
        return loss

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        return self.model(x)

    def log_metrics(self, metrics: dict, suffix='', batch_size=None):
        flat_metrics = flatten_dict(metrics)
        for k, v in flat_metrics.items():
            name = f"{k}/{suffix}"
            self.log(name, v, prog_bar=False, batch_size=batch_size)

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        self.log('time_per_epoch', time.time() - self.epoch_start_time, prog_bar=False, on_epoch=True)
        if self._log_w: self.log_weights()
        if self._log_preact: self.log_preactivations()

    def on_train_end(self) -> None:
        ckpt_call = self.trainer.checkpoint_callback
        if ckpt_call is not None:
            ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(
                ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
            best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)
            if ckpt_path.exists() and best_path.exists():
                # Remove last model ckpt leave only best, to hint training successful termination.
                ckpt_path.unlink()
                log.info(f"Removing last ckpt {ckpt_path} from successful training run.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_metrics(self):
        # don't show the version number on console logs.
        items = super().get_metrics()
        items.pop("v_num", None)
        return items
