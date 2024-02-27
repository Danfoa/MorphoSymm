from typing import Any, List, Tuple

import numpy as np
import torch
from escnn.gspaces import GSpace
from escnn.nn import EquivariantModule, FieldType, GeometricTensor, init
from escnn.nn.modules.basismanager import BasisManager, BlocksBasisExpansion
from torch.nn import Parameter


class CGTensorProductModule(EquivariantModule):

    def __init__(self, in_type: FieldType, out_type: FieldType, initialize: bool = True):
        r"""Applies a (learnable) quadratic non-linearity to the input features.

        The module requires its input and output types to be *uniform*, i.e. contain multiple copies of the same
        representation; see also :meth:`~escnn.nn.FieldType.uniform`.
        Moreover, the input and output field types must have the same number of fields, i.e.
        ```len(in_type) == len(out_type)```.

        The module computes the tensor product of each field with itself to generate an intermediate feature map.
        Note that this feature map will have size ```len(in_type) * in_type.representations[0].size**2```.
        To prevent the exponential growth of the model's width at each layer, the module includes also a learnable
        linear projection of each ```in_type.representations[0].size**2```-dimensional output field to a corresponding
        ```out_type.representations[0].size``` output field.
        Note that this layer applies an independent linear projection to each field individually but does not mix them.

        ..warning ::
            A model employing only this kind of non-linearities will effectively be a polynomial function.
            Moreover, the degree of the polynomial grows exponentially with the depth of the network.
            This may result in some instabilities during training.

        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            initialize (bool, optional): initialize the weights of the model (warning: can be slow). Default: ``True``

        Attributes:
            ~.weights (torch.Tensor): the learnable parameters which are used to expand the projection matrix
            ~.matrix (torch.Tensor): the matrix obtained by expanding the parameters in
            :attr:`~escnn.nn.TensorProductModule.weights`

        """
        super(CGTensorProductModule, self).__init__()

        self.in_type = in_type
        self.out_type = out_type

        assert self.in_type.gspace == self.out_type.gspace

        self.gspace: GSpace = self.in_type.gspace

        # assert in_type.uniform
        # assert out_type.uniform
        # assert len(in_type) == len(out_type), (len(in_type), len(out_type))

        in_repr = in_type.representation
        hid_repr = in_repr.tensor(in_repr)

        self.hidden_field_type = FieldType(in_type.gspace, [hid_repr])

        # Linear Projection back to output rep -------------------------------------------------------------------------
        # The tensor product result is linearly projected back to the output space. This is used to avoid the
        # exponential growth of the model's width at each layer.
        # BlocksBasisExpansion: submodule which takes care of building the linear projection matrix
        self._basisexpansion = BlocksBasisExpansion(
            in_reprs=[hid_repr],
            out_reprs=out_type.representations,
            basis_generator=self.gspace.build_fiber_intertwiner_basis,
            points=np.zeros((1, 1)),
            recompute=False
            )

        if self.basisexpansion.dimension() == 0:
            raise ValueError('''
                The basis for the steerable matrix is empty!
            ''')
        self.num_trainable_parameters = self.basisexpansion.dimension()

        self.weights = Parameter(torch.zeros(self.basisexpansion.dimension()), requires_grad=True)
        self.weights.data[:] = torch.randn_like(self.weights)
        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            init.generalized_he_init(self.weights.data, self.basisexpansion)

        matrix_size = (out_type.size, hid_repr.size)
        self.register_buffer("matrix", torch.zeros(*matrix_size))

    # @classmethod
    # def construct(cls, gspace: GSpace, channels: int, in_repr: Representation, out_repr: Representation,
    #               initialize: bool = True) -> 'TensorProductModule':
    #     '''Constructor method which provides an alternative way to instantiate a TensorProductModule.
    #     This method automatically builds the ```in_type``` (respectively, ```out_type```) as ```channels``` copies
    #     of ```in_repr``` (```out_repr```).
    #     '''
    #     assert in_repr.group == gspace.fibergroup
    #     assert out_repr.group == gspace.fibergroup
    #     in_type = gspace.type(*[in_repr] * channels)
    #     out_type = gspace.type(*[out_repr] * channels)
    #     return TensorProductModule(in_type, out_type, initialize=initialize)

    def forward(self, input: GeometricTensor):

        assert input.type == self.in_type

        coords = input.coords
        spatial_shape = input.shape[2:] if coords is not None else input.shape[-1]
        if coords is not None:
            input = input.tensor.view(input.shape[0], self._C, self.in_type.representations[0].size, *spatial_shape)
        else:
            input = input.tensor
        # compute tensor product
        tensor_features = torch.einsum('bi...,bo...->bio...', input, input)
        # tensor_features = tensor_features.view(input.shape[0], self._C, self.hidden_field_type.representations[
        # 0].size,
        #                                        *spatial_shape)
        tensor_features = tensor_features.view(input.shape[0], spatial_shape ** 2)

        # perform projection
        if not self.training:
            _matrix = self.matrix
        else:
            # retrieve the matrix and the bias
            _matrix = self.expand_parameters()

        # _matrix = _matrix.view(self._C, self.out_type.representations[0].size,
        #                        self.hidden_field_type.representations[0].size)
        output = torch.einsum('of,bf...->bo...', _matrix, tensor_features)

        # output = output.view(input.shape[0], self.out_type.size, *spatial_shape)

        return self.out_type(output)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size,) + input_shape[2:]
        else:
            return input_shape

    def export(self):
        raise NotImplementedError()

    @property
    def basisexpansion(self) -> BasisManager:
        r"""Submodule which takes care of building the matrix.

        It uses the learnt ``weights`` to expand a basis and returns a matrix in the usual form used by conventional
        convolutional modules.
        It uses the learned ``weights`` to expand the kernel in the G-steerable basis and returns it in the shape
        :math:`(c_\text{out}, c_\text{in}, s^d)`, where :math:`s` is the ``kernel_size`` and :math:`d` is the
        dimensionality of the base space.

        """
        return self._basisexpansion

    def expand_parameters(self) -> torch.Tensor:
        r"""Expand the matrix in terms of the :attr:`escnn.nn.TensorProductModule.weights`.

        Returns:
            the expanded projection matrix

        """
        _matrix = self.basisexpansion(self.weights)
        return _matrix.view(self.out_type.size, self.hidden_field_type.representations[0].size)

    def train(self, mode=True):
        r"""Modifies the module's mode from training to inference and vice versa.

        If ``mode=True``, the method sets the module in training mode and discards the
        :attr:`~escnn.nn.TensorProductModule.matrix` attribute.

        If ``mode=False``, it sets the module in evaluation mode. Moreover, the method builds the matrix
        using the current values of the trainable parameters and store them in
        :attr:`~escnn.nn.TensorProductModule.matrix`
        such that it is not recomputed at each forward pass.

        .. warning ::

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of this class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.

        Args:
            mode (bool, optional): whether to set training mode (``True``) or evaluation mode (``False``).
                                   Default: ``True``.

        """
        if not mode:
            _matrix = self.expand_parameters()
            self.register_buffer("matrix", _matrix)
        else:
            if hasattr(self, "matrix"):
                del self.matrix

        return super(CGTensorProductModule, self).train(mode)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-4, assert_raise: bool = True) -> List[
        Tuple[Any, float]]:

        c = self.in_type.size
        batch = 128
        x = torch.randn(batch, c)

        errors = []

        # for el in self.gspace.testing_elements:
        for _ in range(100):

            g = self.gspace.fibergroup.sample()
            if g == self.gspace.fibergroup.identity:
                continue
            x1 = GeometricTensor(x.clone(), self.in_type)
            x2 = GeometricTensor(x.clone(), self.in_type).transform_fibers(g)

            out1 = self(x1).transform_fibers(g)
            out2 = self(x2)

            # out1 = out1.tensor.view(batch, len(self.out_type), self.out_type.representations[0].size, *out1.shape[2:]).detach().numpy()
            # out2 = out2.tensor.view(batch, len(self.out_type), self.out_type.representations[0].size, *out2.shape[2:]).detach().numpy()

            errs = np.linalg.norm(out1 - out2, axis=2).reshape(-1)
            errs[errs < atol] = 0.
            norm = np.sqrt(np.linalg.norm(out1, axis=2).reshape(-1) * np.linalg.norm(out2, axis=2).reshape(-1))

            relerr = errs / norm

            # print(el, errs.max(), errs.mean(), relerr.max(), relerr.min())

            if assert_raise:
                assert relerr.mean() + relerr.std() < rtol, \
                    ('The error found during equivariance check with element "{}" is too high: max = {}, mean = {}, '
                     'std ={}') \
                        .format(g, relerr.max(), relerr.mean(), relerr.std())

            # errors.append((el, errs.mean()))
            errors.append(relerr)

        # return errors
        return np.concatenate(errors).reshape(-1)
