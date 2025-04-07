from typing import Tuple

import escnn
import numpy as np
import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from torch.nn import Parameter

from morpho_symm.utils.abstract_harmonics_analysis import isotypic_decomp_representation


class IsotypicBasis(EquivariantModule):
    r"""Utility non-trainable module to do the change of basis to a symmetry enabled basis (or isotypic basis)."""

    def __init__(self, in_type: FieldType):
        """Instanciate a non-trainable module effectively applying the change of basis to a symmetry enabled basis.

        Args:
            in_type: The representation of the input vector field.
        """
        super().__init__()
        self.in_type = in_type
        self.group = in_type.gspace.fibergroup
        # Representation iso_rep = Q2iso^-1 @ iso_basis @ Q2iso
        self.iso_rep = isotypic_decomp_representation(in_type.representation)
        # Output type is a symmetry enabled basis with "no change of basis" (i.e., identity matrix)
        self.out_type = FieldType(
            in_type.gspace, [iso_rep for iso_rep in self.iso_rep.attributes["isotypic_reps"].values()]
        )
        # Orthogonal transformation from the input basis to the isotypic basis
        self.Q2iso = Parameter(torch.from_numpy(self.iso_rep.change_of_basis_inv).float(), requires_grad=False)
        self.Q2ori = Parameter(torch.from_numpy(self.iso_rep.change_of_basis).float(), requires_grad=False)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Change of basis of the input field to a symmetry enabled basis (or isotypic basis)."""
        assert x.type == self.in_type, f"Input type {x.type} does not match module's input type {self.in_type}"
        x_iso = torch.einsum("ij,...j->...i", self.Q2iso, x.tensor)
        return self.out_type(x_iso)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Shape of the output field."""
        return input_shape

    def change2iso(self, x: GeometricTensor) -> GeometricTensor:
        """Change of basis of the input field to a symmetry enabled basis (or isotypic basis)."""
        return self.forward(x)

    def change2ori(self, x: GeometricTensor) -> GeometricTensor:
        """Change of basis of the input field to a symmetry enabled basis (or isotypic basis)."""
        assert x.type == self.out_type, f"Input type {x.type} does not match module's input type {self.out_type}"
        x_ori = torch.einsum("ij,...j->...i", self.Q2ori, x.tensor)
        return self.in_type(x_ori)


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(5)
    gspace = escnn.gspaces.no_base_space(G)

    in_type = escnn.nn.FieldType(gspace, [G.regular_representation] * 2)

    rep_iso_basis = isotypic_decomp_representation(in_type.representation)

    iso_module = IsotypicBasis(in_type)

    x_np = np.random.randn(1, in_type.size)
    x = in_type(torch.from_numpy(x_np).float())
    x_iso = iso_module(x)

    iso_rep = iso_module.iso_rep

    x_np_iso = (iso_rep.change_of_basis_inv @ x_np.T).T
    assert np.allclose(x_np_iso, x_iso.tensor.numpy()), f"{x_np_iso - x_iso.tensor.numpy()}!=0"
