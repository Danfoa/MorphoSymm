import logging
from typing import List, Union

import torch

log = logging.getLogger(__name__)


class MLP(torch.nn.Module):
    """Standard baseline MLP. Representations and group are used for shapes only."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_hidden_units: int = 64,
                 num_layers: int = 3,
                 bias: bool = True,
                 batch_norm: bool = False,
                 head_with_activation: bool = False,
                 activation: Union[torch.nn.Module, List[torch.nn.Module]] = torch.nn.ReLU,
                 init_mode="fan_in"):
        """Constructor of a Multi-Layer Perceptron (MLP) model.

        This utility class allows to easily instanciate a G-equivariant MLP architecture.

        Args:
        ----
            in_dim: Dimension of the input space.
            out_dim: Dimension of the output space.
            num_hidden_units: Number of hidden units in the intermediate layers.
            num_layers: Number of layers in the MLP including input and output/head layers. That is, the number of
            activation (escnn.nn.EquivariantModule, list(escnn.nn.EquivariantModule)): If a single activation module is
            provided it will be used for all layers except the output layer. If a list of activation modules is provided
            then `num_layers` activation equivariant modules should be provided.
            bias: Whether to include a bias term in the linear layers.
            init_mode: Not used until now. Will be used to initialize the weights of the MLP
        """
        super().__init__()
        logging.info("Instantiating MLP (PyTorch)")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.init_mode = init_mode if init_mode is not None else "fan_in"
        self.hidden_channels = num_hidden_units
        self.activation = activation if isinstance(activation, list) else [activation] * (num_layers - 1)

        self.num_layers = num_layers
        if self.num_layers == 1 and not head_with_activation:
            log.warning(f"{self} model with 1 layer and no activation. This is equivalent to a linear map")

        dim_in = self.in_dim
        dim_out = num_hidden_units

        self.net = torch.nn.Sequential()
        for n in range(self.num_layers - 1):
            dim_out = num_hidden_units

            block = torch.nn.Sequential()
            block.add_module(f"linear_{n}", torch.nn.Linear(dim_in, dim_out, bias=bias))
            if batch_norm:
                block.add_module(f"batchnorm_{n}", torch.nn.BatchNorm1d(dim_out))
            block.add_module(f"act_{n}", activation())

            self.net.add_module(f"block_{n}", block)
            dim_in = dim_out

        # Add last layer
        head_block = torch.nn.Sequential()
        head_block.add_module(f"linear_{num_layers - 1}", torch.nn.Linear(in_features=dim_out, out_features=self.out_dim, bias=bias))
        if head_with_activation:
            if batch_norm:
                head_block.add_module(f"batchnorm_{num_layers - 1}", torch.nn.BatchNorm1d(dim_out))
            head_block.add_module(f"act_{num_layers - 1}", activation())

        self.net.add_module("head", head_block)

        self.reset_parameters(init_mode=self.init_mode)

    def forward(self, input):
        """Forward pass of the MLP model."""
        output = self.net(input)
        return output

    def get_hparams(self):
        return {'num_layers': self.num_layers,
                'hidden_ch':  self.hidden_channels,
                'init_mode':  self.init_mode}

    def reset_parameters(self, init_mode=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = self.init_mode if init_mode is None else init_mode
        for module in self.net:
            if isinstance(module, torch.nn.Sequential):
                tensor = module[0].weight
                activation = module[-1].__class__.__name__
                activation = "linear" if activation == "Identity" else activation
            elif isinstance(module, torch.nn.Linear):
                tensor = module.weight
                activation = "Linear"
            else:
                raise NotImplementedError(module.__class__.__name__)

            if activation.lower() == "relu" or activation.lower() == "leakyrelu":
                torch.nn.init.kaiming_uniform_(tensor, mode=self.init_mode, nonlinearity=activation.lower())
            elif activation.lower() == "selu":
                torch.nn.init.kaiming_normal_(tensor, mode=self.init_mode, nonlinearity='linear')
            else:
                try:
                    torch.nn.init.kaiming_uniform_(tensor, mode=self.init_mode, nonlinearity=activation.lower())
                except ValueError as e:
                    log.info(f"Could not initialize {module.__class__.__name__} with {self.init_mode} mode. "
                              f"Using default Pytorch initialization")

        log.info(f"MLP initialized with mode: {self.init_mode}")