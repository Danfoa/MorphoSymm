import math
import time
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nn.EquivariantModules import BasisLinear, EBlock, LinearBlock, EMLP, MLP


class EquivariantModel(pl.LightningModule):

    def __init__(self, model: Union[EMLP, MLP], model_type, lr):
        super().__init__()
        self.model_type = model_type
        self.lr = lr

        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y, y_pred)
        # Logging to TensorBoard by default
        self.log("hp/train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_pred = self.model(x)
            loss = F.mse_loss(y, y_pred)

            cosine_sim = F.cosine_similarity(y, y_pred)

        self.log("hp/val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("hp/cosine_sim", cosine_sim, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_epoch_end(self, outputs):
        # outs is a list of whatever you returned in `validation_step`
        pass

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        self.log('time_per_epoch', time.time() - self.epoch_start_time, logger=True, prog_bar=False)
        self.log_weights()

    def on_train_start(self):
        hparams = {'lr': self.lr, 'model': self.model_type,}
        hparams.update(self.model.get_hparams())
        self.logger.log_hyperparams(hparams, {"hp/val_loss": 0., "hp/train_loss": 0.})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_weights(self,):
        tb_logger = self.logger.experiment
        layer_index = 0   # Count layers by linear operators not position in network sequence
        for layer in self.model.net:
            layer_name = f"Layer{layer_index:02d}"
            if isinstance(layer, EBlock):
                W = layer.linear.weight.view(-1).detach()
                basis_coeff = layer.linear.basis_coeff.view(-1).detach()
                tb_logger.add_histogram(tag=f"{layer_name}.c", values=basis_coeff, global_step=self.current_epoch)
                tb_logger.add_histogram(tag=f"{layer_name}.W", values=W, global_step=self.current_epoch)
                layer_index += 1
            elif isinstance(layer, LinearBlock):
                W = layer.linear.weight.view(-1).detach()
                tb_logger.add_histogram(tag=f"{layer_name}.W", values=W, global_step=self.current_epoch)
                layer_index += 1
            elif isinstance(layer, BasisLinear):
                W = layer.weight.view(-1).detach()
                basis_coeff = layer.basis_coeff.view(-1).detach()
                tb_logger.add_histogram(tag=f"{layer_name}.c", values=basis_coeff, global_step=self.current_epoch)
                tb_logger.add_histogram(tag=f"{layer_name}.W", values=W, global_step=self.current_epoch)
                layer_index += 1

    def get_metrics(self):
        # don't show the version number on console logs.
        items = super().get_metrics()
        items.pop("v_num", None)
        return items