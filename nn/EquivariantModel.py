import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from nn.EquivariantModules import BasisLinear


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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.training_step(batch, batch_idx)

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
                fan_in, fan_out = module.W.shape
                gain = 1  # TODO: Check this
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
                return torch.nn.init._no_grad_uniform_(module.basis_coeff, -a, a)
            else:
                weights_initializer(module.basis_coeff)
            if module.with_bias:
                bias_initializer(module.bias_basis_coeff)
