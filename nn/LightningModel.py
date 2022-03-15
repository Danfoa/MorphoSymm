import math
import time
from typing import Union, Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nn.EquivariantModules import BasisLinear, EquivariantBlock, LinearBlock, EMLP, MLP

LossCallable = Callable[[torch.Tensor, torch.Tensor, ], torch.Tensor]
MetricCallable = Callable[[torch.Tensor, torch.Tensor, ], dict]

class LightningModel(pl.LightningModule):

    def __init__(self, model: Union[EMLP, MLP], lr, loss_fn: LossCallable, metrics_fn: MetricCallable,
                 log_preact=True, log_w=True):
        super().__init__()
        self.model_type = model.__class__.__name__
        self.lr = lr

        self.model = model
        self.loss_fn = loss_fn
        self.compute_metrics = metrics_fn
        self._log_w = log_w
        self._log_preact = log_preact

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y, y_pred)
        # Logging to TensorBoard by default
        self.log("hp/train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        metrics = self.compute_metrics(y, y_pred)
        for k, v in metrics.items():
            self.log(f"hp/train_{k}", torch.mean(v), prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fn(y, y_pred)
            metrics = self.compute_metrics(y, y_pred)

            self.log("hp/val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            for k, v in metrics.items():
                self.log(f"hp/val_{k}", torch.mean(v), prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fn(y, y_pred)
            metrics = self.compute_metrics(y, y_pred)

            self.log("hp/test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            for k, v in metrics.items():
                self.log(f"hp/test_{k}", torch.mean(v), prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_epoch_end(self, outputs):
        # outs is a list of whatever you returned in `validation_step`
        pass

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        self.log('time_per_epoch', time.time() - self.epoch_start_time, logger=True, prog_bar=False)
        if self._log_w: self.log_weights()
        if self._log_preact: self.log_preactivations()

    def on_train_start(self):
        # TODO: Add number of layers and hidden channels dimensions.
        hparams = {'lr': self.lr, 'model': self.model_type}
        hparams.update(self.model.get_hparams())
        self.logger.log_hyperparams(hparams, {"hp/val_loss": 0., "hp/train_loss": 0., "hp/cosine_sim": 0.})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_weights(self):
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
