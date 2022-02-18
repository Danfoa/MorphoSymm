import math
import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nn.EquivariantModules import BasisLinear, EBlock, LinearBlock


class EquivariantModel(pl.LightningModule):

    def __init__(self, model, model_type, lr):
        super().__init__()
        self.model_type = model_type
        self.lr = lr

        self.model = model
        self.model.apply(self.weights_initialization)
        # TODO: Assert Equivariance of network

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
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_pred = self.model(x)
            loss = F.mse_loss(y, y_pred)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        # outs is a list of whatever you returned in `validation_step`
        pass

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        self.log('time_per_epoch', time.time() - self.epoch_start_time, logger=True, prog_bar=False)
        self.log_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def weights_initialization(module,
                               weights_initializer=torch.nn.init.xavier_uniform,
                               bias_initializer=torch.nn.init.zeros_):
        # TODO: Place default initializer
        # TODO: Compute initialization considering weight sharing distribution
        # For now use Glorot initialization, must check later:
        if isinstance(module, BasisLinear):

            if weights_initializer == torch.nn.init.xavier_uniform:
                # Xavier cannot find the in out dimensions because the tensor is not 2D
                fan_in, fan_out = module.weight.shape
                gain = 1  # TODO: Check this
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
                return torch.nn.init._no_grad_uniform_(module.basis_coeff, -a, a)
            else:
                weights_initializer(module.basis_coeff)
            if module.with_bias:
                bias_initializer(module.bias_basis_coeff)

    def log_weights(self,):
        tb_logger = self.logger.experiment
        layer_index = 0   # Count layers by linear operators not position in network sequence
        for layer in self.model.net:
            layer_name = f"Layer{layer_index:02d}"
            # if isinstance(layer, torch.nn.Linear):
            #     for name, param in layer.named_parameters():
            #         if name.endswith(".bias"):
            #             continue
            #         tab = layer_name + ".W"
            #         tb_logger.add_histogram(tag=tab, values=param.detach().view(-1), global_step=self.current_epoch)
            #         layer_index += 1
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
