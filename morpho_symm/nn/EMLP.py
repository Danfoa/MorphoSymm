import logging
from typing import List, Union

import escnn
import numpy as np
import torch
from escnn.nn import EquivariantModule, FieldType

log = logging.getLogger(__name__)


class EMLP(EquivariantModule):
    """Equivariant Multi-Layer Perceptron (EMLP) model."""

    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 num_hidden_units: int = 64,
                 num_layers: int = 3,
                 bias: bool = True,
                 activation: Union[str, EquivariantModule] = "ELU",
                 head_with_activation: bool = False,
                 batch_norm: bool = True):
        """Constructor of an Equivariant Multi-Layer Perceptron (EMLP) model.

        This utility class allows to easily instanciate a G-equivariant MLP architecture. As a convention, we assume
        every internal layer is a map between the input space: X and an output space: Y, denoted as
        z = σ(y) = σ(W x + b). Where x ∈ X, W: X -> Y, b ∈ Y. The group representation used for intermediate layer
        embeddings ρ_Y: G -> GL(Y) is defined as a sum of multiple regular representations:
        ρ_Y := ρ_reg ⊕ ρ_reg ⊕ ... ⊕ ρ_reg. Therefore, the number of `hidden layer's neurons` will be a multiple of |G|.
        Being the multiplicities of the regular representation: ceil(num_hidden_units/|G|)

        Args:
        ----
            in_type (escnn.nn.FieldType): Input field type containing the representation of the input space.
            out_type (escnn.nn.FieldType): Output field type containing the representation of the output space.
            num_hidden_units: Number of hidden units in the intermediate layers. The effective number of hidden units
            will be ceil(num_hidden_units/|G|). Since we assume intermediate embeddings are regular fields.
            activation (escnn.nn.EquivariantModule, str): Name of pointwise activation function to use.
            num_layers: Number of layers in the MLP including input and output/head layers. That is, the number of
            hidden layers will be num_layers - 2.
            bias: Whether to include a bias term in the linear layers.
            activation (escnn.nn.EquivariantModule, list(escnn.nn.EquivariantModule)): If a single activation module is
            provided it will be used for all layers except the output layer. If a list of activation modules is provided
            then `num_layers` activation equivariant modules should be provided.
            head_with_activation: Whether to include an activation module in the output layer.
            init_mode: Not used until now. Will be used to initialize the weights of the MLP.
        """
        super(EMLP, self).__init__()
        logging.info("Instantiating EMLP (PyTorch)")
        self.in_type, self.out_type = in_type, out_type
        self.gspace = self.in_type.gspace
        self.group = self.gspace.fibergroup

        self.num_layers = num_layers
        if self.num_layers == 1 and not head_with_activation:
            log.warning(f"{self} model with 1 layer and no activation. This is equivalent to a linear map")

        if isinstance(activation, str):
            # Approximate the num of neurons as the num of signals in the space spawned by the irreps of the input type
            # To compute the signal over the group we use all elements for finite groups
            activation = self.get_activation(activation, in_type=in_type, desired_hidden_units=num_hidden_units)
            hidden_type = activation.in_type
        elif isinstance(activation, EquivariantModule):
            hidden_type = activation.in_type
        else:
            raise ValueError(f"Activation type {type(activation)} not supported.")

        input_irreps = set(in_type.representation.irreps)
        inner_irreps = set(out_type.irreps)
        diff = input_irreps.symmetric_difference(inner_irreps)
        if len(diff) > 0:
            log.warning(f"Irreps {list(diff)} of group {self.gspace.fibergroup} are not in the input/output types."
                        f"This represents an information bottleneck. Consider extracting invariant features.")

        layer_in_type = in_type

        self.net = escnn.nn.SequentialModule()
        for n in range(self.num_layers - 1):
            layer_out_type = hidden_type

            block = escnn.nn.SequentialModule()
            block.add_module(f"linear_{n}", escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias))
            if batch_norm:
                block.add_module(f"batchnorm_{n}", escnn.nn.IIDBatchNorm1d(layer_out_type)),
            block.add_module(f"act_{n}", activation)

            self.net.add_module(f"block_{n}", block)
            layer_in_type = layer_out_type

        # Add final layer
        head_block = escnn.nn.SequentialModule()
        head_block.add_module(f"linear_{num_layers - 1}", escnn.nn.Linear(layer_in_type, out_type, bias=bias))
        if head_with_activation:
            if batch_norm:
                head_block.add_module(f"batchnorm_{num_layers - 1}", escnn.nn.IIDBatchNorm1d(out_type)),
            head_block.add_module(f"act_{num_layers - 1}", activation)
        # head_layer.check_equivariance()
        self.net.add_module("head", head_block)
        # Test the entire model is equivariant.
        # self.net.check_equivariance()

    @staticmethod
    def get_activation(activation, in_type: FieldType, desired_hidden_units: int):
        gspace = in_type.gspace
        group = gspace.fibergroup
        grid_length = group.order() if not group.continuous else 20

        unique_irreps = set(in_type.irreps)
        unique_irreps_dim = sum([group.irrep(*id).size for id in set(in_type.irreps)])
        scale = in_type.size // unique_irreps_dim
        channels = int(np.ceil(desired_hidden_units // unique_irreps_dim // scale))
        if "identity" in activation.lower():
            raise NotImplementedError("Identity activation not implemented yet")
            # return escnn.nn.IdentityModule()
        else:
            return escnn.nn.FourierPointwise(gspace,
                                             channels=channels,
                                             irreps=list(unique_irreps),
                                             function=f"p_{activation.lower()}",
                                             inplace=True,
                                             type='regular' if not group.continuous else 'rand',
                                             N=grid_length)

    def forward(self, x):
        """Forward pass of the EMLP model."""
        return self.net(x)

    def reset_parameters(self, init_mode=None):
        """Initialize weights and biases of E-MLP model."""
        raise NotImplementedError()

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size


class MLP(torch.nn.Module):
    """Standard baseline MLP. Representations and group are used for shapes only."""

    def __init__(self, d_in, d_out, num_hidden_units=128, num_layers=3,
                 activation: Union[torch.nn.Module, List[torch.nn.Module]] = torch.nn.ReLU,
                 with_bias=True, init_mode="fan_in"):
        """Constructor of a Multi-Layer Perceptron (MLP) model.

        This utility class allows to easily instanciate a G-equivariant MLP architecture.

        Args:
            d_in: Dimension of the input space.
            d_out: Dimension of the output space.
            num_hidden_units: Number of hidden units in the intermediate layers.
            num_layers: Number of layers in the MLP including input and output/head layers. That is, the number of
            activation (escnn.nn.EquivariantModule, list(escnn.nn.EquivariantModule)): If a single activation module is
            provided it will be used for all layers except the output layer. If a list of activation modules is provided
            then `num_layers` activation equivariant modules should be provided.
            with_bias: Whether to include a bias term in the linear layers.
            init_mode: Not used until now. Will be used to initialize the weights of the MLP
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.init_mode = init_mode
        self.hidden_channels = num_hidden_units
        self.activation = activation

        logging.info("Initializing MLP")

        dim_in = self.d_in
        dim_out = num_hidden_units
        self.net = torch.nn.Sequential()
        for n in range(num_layers - 1):
            dim_out = num_hidden_units
            block = torch.nn.Sequential()
            block.add_module(f"linear_{n}", torch.nn.Linear(dim_in, dim_out, bias=with_bias))
            block.add_module(f"batchnorm_{n}", torch.nn.BatchNorm1d(dim_out))
            block.add_module(f"act_{n}", activation())

            self.net.add_module(f"block_{n}", block)
            dim_in = dim_out
        # Add last layer
        linear_out = torch.nn.Linear(in_features=dim_out, out_features=self.d_out, bias=with_bias)
        self.net.add_module("head", linear_out)

        self.reset_parameters(init_mode=self.init_mode)

    def forward(self, input):
        output = self.net(input)
        return output

    def get_hparams(self):
        return {'num_layers': len(self.net),
                'hidden_ch':  self.hidden_channels,
                'init_mode':  self.init_mode}

    def reset_parameters(self, init_mode=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = self.init_mode if init_mode is None else init_mode
        for module in self.net:
            if isinstance(module, torch.nn.Sequential):
                tensor = module[0].weight
                activation = module[-1].__class__.__name__
            elif isinstance(module, torch.nn.Linear):
                tensor = module.weight
                activation = "Linear"
            else:
                raise NotImplementedError(module.__class__.__name__)

            if "fan_in" == self.init_mode or "fan_out" == self.init_mode:
                torch.nn.init.kaiming_uniform_(tensor, mode=self.init_mode, nonlinearity=activation.lower())
            elif 'normal' in self.init_mode.lower():
                split = self.init_mode.split('l')
                std = 0.1 if len(split) == 1 else float(split[1])
                torch.nn.init.normal_(tensor, 0, std)
            else:
                raise NotImplementedError(self.init_mode)

        log.info(f"MLP initialized with mode: {self.init_mode}")
