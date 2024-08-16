import logging
from typing import Union

import escnn
import numpy as np
import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.nn.EquivariantModules import IsotypicBasis

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
                 batch_norm: bool = False,
                 batch_norm_kwargs: dict = dict(affine=False, track_running_stats=True)):
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
            out_type (escnn.nn.FieldType): Output field type containing the representation of the output space. If the
            output type representation is composed only of multiples of the trivial representation then the this model
            will default to a G-invariant function. This G-invariant function is composed of a G-equivariant feature
            extractor, a G-invariant pooling layer, extracting invariant features from the equivariant features, and a
            last linear (unconstrained) layer to map the invariant features to the output space.
                TODO: Create a G-invariant EMLP class where we can control the network processing the G-invariant
                 features instead of defaulting to a single linear layer.
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

        if batch_norm:
            log.warning("Equivariant Batch norm affects the performance of the model. Dont use if for now!!!")
        # Check if the network is a G-invariant function (i.e., out rep is composed only of the trivial representation)
        out_irreps = set(out_type.representation.irreps)
        if len(out_irreps) == 1 and self.group.trivial_representation.id == list(out_irreps)[0]:
            self.invariant_fn = True
        else:
            self.invariant_fn = False
            input_irreps = set(in_type.representation.irreps)
            inner_irreps = set(out_type.irreps)
            diff = input_irreps.symmetric_difference(inner_irreps)
            if len(diff) > 0:
                log.warning(f"Irreps {list(diff)} of group {self.gspace.fibergroup} are not in the input/output types."
                            f"This represents an information bottleneck. Consider extracting invariant features.")

        if self.num_layers == 1 and not head_with_activation:
            log.warning(f"{self} model with 1 layer and no activation. This is equivalent to a linear map")

        if isinstance(activation, str):
            # Approximate the num of neurons as the num of signals in the space spawned by the irreps of the input type
            # To compute the signal over the group we use all elements for finite groups
            hidden_activation = self.get_activation(activation, in_type=in_type, desired_hidden_units=num_hidden_units)
            hidden_type = hidden_activation.in_type
        elif isinstance(activation, EquivariantModule):
            hidden_type = activation.in_type
        else:
            raise ValueError(f"Activation type {type(activation)} not supported.")

        layer_in_type = in_type
        self.net = escnn.nn.SequentialModule()
        for n in range(self.num_layers - 1):
            layer_out_type = hidden_type

            block = escnn.nn.SequentialModule()
            block.add_module(f"linear_{n}: in={layer_in_type.size}-out={layer_out_type.size}",
                             escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias))
            if batch_norm:
                block.add_module(f"batchnorm_{n}", escnn.nn.IIDBatchNorm1d(layer_out_type, **batch_norm_kwargs)),
            block.add_module(f"act_{n}", hidden_activation)

            self.net.add_module(f"block_{n}", block)
            layer_in_type = layer_out_type

        # Add final layer
        self.net_head = None
        if self.invariant_fn:
            self.net_head = torch.nn.Sequential()
            # TODO: Make the G-invariant pooling with Isotypic Basis a stand alone module.
            # Module describing the change of basis to an Isotypic Basis required for efficient G-invariant pooling
            self.change2isotypic_basis = IsotypicBasis(hidden_type)
            # Number of G-invariant features from net output equals the number of G-stable subspaces.
            num_inv_features = len(hidden_type.irreps)
            self.net_head.add_module(f"linear_{num_layers - 1}",
                                     torch.nn.Linear(num_inv_features, out_type.size, bias=bias))
            if head_with_activation:
                if batch_norm:
                    self.net_head.add_module(f"batchnorm_{num_layers - 1}", torch.nn.BatchNorm1d(out_type.size, **batch_norm_kwargs)),
                self.net_head.add_module(f"act_{num_layers - 1}", activation)
        else:  # Equivariant Network
            self.net_head = escnn.nn.SequentialModule()
            self.net_head.add_module(f"linear_{num_layers - 1}", escnn.nn.Linear(layer_in_type, out_type, bias=bias))
            if head_with_activation:
                if batch_norm:
                    self.net_head.add_module(f"batchnorm_{num_layers - 1}", escnn.nn.IIDBatchNorm1d(out_type, **batch_norm_kwargs)),
                self.net_head.add_module(f"act_{num_layers - 1}", escnn.nn.ELU(in_type=out_type))
        # Test the entire model is equivariant.
        # self.net.check_equivariance()

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Forward pass of the EMLP model."""
        equivariant_features = self.net(x)
        if self.invariant_fn:
            iso_equivariant_features = self.change2isotypic_basis(equivariant_features)
            invariant_features = self.irrep_norm_pooling(iso_equivariant_features.tensor, iso_equivariant_features.type)
            output = self.net_head(invariant_features)
            output = self.out_type(output)  # Wrap over invariant field type
        else:
            output = self.net_head(equivariant_features)
        return output

    def reset_parameters(self, init_mode=None):
        """Initialize weights and biases of E-MLP model."""
        raise NotImplementedError()

    @staticmethod
    def get_activation(activation, in_type: FieldType, desired_hidden_units: int) -> EquivariantModule:
        gspace = in_type.gspace
        group = gspace.fibergroup

        grid_kwargs = EMLP.get_group_kwargs(group)

        unique_irreps = set(in_type.irreps)
        unique_irreps_dim = sum([group.irrep(*id).size for id in set(in_type.irreps)])
        channels = int(np.ceil(desired_hidden_units // unique_irreps_dim))
        if "identity" in activation.lower():
            raise NotImplementedError("Identity activation not implemented yet")
            # return escnn.nn.IdentityModule()
        else:
            act = escnn.nn.FourierPointwise(gspace,
                                            channels=channels,
                                            irreps=list(unique_irreps),
                                            function=f"p_{activation.lower()}",
                                            inplace=True,
                                            **grid_kwargs)
        # assert (act.out_type.size - desired_hidden_units) <= unique_irreps_dim, \
        #     f"out_type.size {act.out_type.size} - des_hidden_units {desired_hidden_units} > {unique_irreps_dim}"
        return act

    @staticmethod
    def get_group_kwargs(group: escnn.group.Group):
        grid_type = 'regular' if not group.continuous else 'rand'
        N = group.order() if not group.continuous else 10
        kwargs = dict()

        if isinstance(group, escnn.group.DihedralGroup):
            N = N // 2
        elif isinstance(group, escnn.group.DirectProductGroup):
            G1_args = EMLP.get_group_kwargs(group.G1)
            G2_args = EMLP.get_group_kwargs(group.G2)
            kwargs.update({f"G1_{k}": v for k, v in G1_args.items()})
            kwargs.update({f"G2_{k}": v for k, v in G2_args.items()})

        return dict(N=N, type=grid_type, **kwargs)

    @staticmethod
    def irrep_norm_pooling(x: torch.Tensor, field_type: FieldType) -> torch.Tensor:
        from morpho_symm.utils.rep_theory_utils import irreps_stats
        n_inv_features = len(field_type.irreps)
        # TODO: Ensure isotypic basis i.e irreps of the same type are consecutive to each other.
        inv_features = []
        for field_start, field_end, rep in zip(field_type.fields_start,
                                               field_type.fields_end,
                                               field_type.representations):
            # Each field here represents a representation of an Isotypic Subspace. This rep is only composed of a single
            # irrep type.
            x_field = x[..., field_start:field_end]
            num_G_stable_spaces = len(rep.irreps)  # Number of G-invariant features = multiplicity of irrep
            # Again this assumes we are already in an Isotypic basis
            unique_irreps, _, _ = irreps_stats(rep.irreps)
            assert len(unique_irreps) >= 1, f"Field type is not an Isotypic Subspace irreps:{unique_irreps}"
            # This basis is useful because we can apply the norm in a vectorized way
            # Reshape features to [batch, num_G_stable_spaces, num_features_per_G_stable_space]
            x_field_p = torch.reshape(x_field, (x_field.shape[0], num_G_stable_spaces, -1))
            # Compute G-invariant measures as the norm of the features in each G-stable space
            inv_field_features = torch.norm(x_field_p, dim=-1)
            # Append to the list of inv features
            inv_features.append(inv_field_features)
        # Concatenate all the invariant features
        inv_features = torch.cat(inv_features, dim=-1)
        assert inv_features.shape[-1] == n_inv_features, f"Expected {n_inv_features} got {inv_features.shape[-1]}"
        return inv_features

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size


if __name__ == "__main__":
    # Load robot instance and its symmetry group
    robot_name = 'mini_cheetah-k4'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
    robot, G = load_symmetric_system(robot_name=robot_name)
    # We use ESCNN to handle the group/representation-theoretic concepts and for the construction of equivariant neural networks.
    gspace = escnn.gspaces.no_base_space(G)
    # Get the relevant group representations.
    rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
    rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
    rep_R3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
    rep_R3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3

    nom = torch.tensor([1, 0.7, -1.4, 1, 0.7, -1.4, 1, 0.7, -1.4, 1, 0.7, -1.4])
    print(nom)
    nom = torch.stack([torch.cos(nom), torch.sin(nom)], dim=-1).view(-1)
    for g in G.elements[1:]:
        sym = torch.from_numpy(rep_QJ(g)).float() @ nom.float()
    sym = sym.view(-1, 2)
    sym = torch.atan2(sym[..., 1], sym[..., 0])
    print(sym)

    nom = torch.tensor([1, 0.7, -1.4, 1, 0.7, -1.4, 1, 0.7, -1.4, 1, 0.7, -1.4])
    # Define the input and output FieldTypes using the representations of each geometric object.
    # in_type = escnn.nn.FieldType(gspace, [G.regular_representation] * 5)
    # Define the input and output FieldTypes using the representations of each geometric object.
    # Representation of x := [q, v] ∈ Q_js x TqQ_js => ρ_X_js(g) := ρ_Q_js(g) ⊕ ρ_TqQ_js(g) | g ∈ G
    in_type = FieldType(gspace, [rep_R3, rep_R3_pseudo, rep_R3, rep_R3, rep_R3_pseudo, rep_TqQJ, rep_TqQJ, rep_TqQJ])
    out_type = escnn.nn.FieldType(gspace, [G.trivial_representation] * 1)
    # Test Invariant EMLP
    emlp = EMLP(in_type, out_type,
                num_hidden_units=128,
                num_layers=3,
                activation="ReLU",
                head_with_activation=False)
    emlp.eval()  # Shut down batch norm
    x = in_type(torch.randn(1, in_type.size))
    y = emlp(x)
    import numpy as np



    # G = escnn.group.DihedralGroup(6)
    # gspace = escnn.gspaces.no_base_space(G)
    # # Test Invariant EMLP
    # in_type = escnn.nn.FieldType(gspace, [G.regular_representation] * 5)
    # out_type = escnn.nn.FieldType(gspace, [G.trivial_representation] * 6)
    # emlp = EMLP(in_type, out_type,
    #             num_hidden_units=128,
    #             num_layers=3,
    #             activation="ReLU",
    #             head_with_activation=False)
    # emlp.eval()  # Shut down batch norm
    # x = in_type(torch.randn(1, in_type.size))
    # y = emlp(x)
    #
    # for g in G.elements:
    #     g_x = in_type(in_type.transform_fibers(x.tensor, g))  # Compute g · x
    #     g_y = emlp(g_x)  # Compute g · y
    #     assert torch.allclose(y.tensor, g_y.tensor, rtol=1e-4, atol=1e-4), \
    #         f"{g} invariance failed {y.tensor} != {g_y.tensor}"
    #
    # # Test Equivariant EMLP
    # in_type = escnn.nn.FieldType(gspace, [G.regular_representation] * 5)
    # out_type = escnn.nn.FieldType(gspace, [G.regular_representation] * 2)
    # emlp = EMLP(in_type, out_type,
    #             num_hidden_units=128,
    #             num_layers=3,
    #             activation="ReLU",
    #             head_with_activation=False)
    # emlp.eval()  # Shut down batch norm
    #
    # x = in_type(torch.randn(1, in_type.size))
    # y = emlp(x)
    #
    # for g in G.elements:
    #     g_x = in_type(in_type.transform_fibers(x.tensor, g))  # Compute g · x
    #     g_y_gt = out_type(out_type.transform_fibers(y.tensor, g))  # Compute ground truth g · y
    #     g_y = emlp(g_x)  # Compute g · y
    #     assert torch.allclose(g_y_gt.tensor, g_y.tensor, rtol=1e-4, atol=1e-4), \
    #         f"{g} invariance failed {g_y_gt.tensor} != {g_y.tensor}"
