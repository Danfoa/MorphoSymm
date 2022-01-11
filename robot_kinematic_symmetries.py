from typing import Tuple, Optional, Sequence

from emlp.reps import V, Vector, T, vis
from emlp.groups import Group, SO, Trivial
import jax.numpy as jnp
import numpy as np
from emlp.reps.linear_operators import lazify, LazyKron
from emlp.reps.product_sum_reps import ProductRep
from matplotlib import pyplot as plt

from utils.emlp_visualization import *

import torch
torch.set_default_dtype(torch.float32)
import emlp.nn.pytorch as nn


class JointSpaceSymmetry(Group):
    """ The alternating group in n dimensions"""
    def __init__(self, gen_permutations: Sequence, gen_reflexions: Optional[Sequence] = None):

        num_generators = len(gen_permutations)
        assert num_generators > 0, "Provide at least one generator"
        self.d = len(gen_permutations[0])
        for p in gen_permutations:
            assert self.d == len(np.unique(p)), np.unique(p, return_counts=True)
        self._gen_permutations = np.array(gen_permutations).astype(np.int)
        self.is_orthogonal = True
        if gen_reflexions is not None:  # Symmetries contain reflexions
            print(f"Creating the group of kinematic symmetries KO ⊆ O({self.d})")
            self.is_permutation = False
            self._gen_reflexions = np.array(gen_reflexions).astype(np.int)
            assert np.all(np.abs(self._gen_reflexions)) == 1, "One-line notation of reflexions must contain only {1,-1}"
            assert self._gen_reflexions.shape == self._gen_permutations.shape
        else:  # Symmetries are permutations_q of state variable dimensions
            print(f"Creating the group of kinematic symmetries KO ⊆ S({self.d})")
            self.is_permutation = True
            self._gen_reflexions = np.ones_like(gen_permutations)

        self.discrete_generators = np.zeros((num_generators, self.d, self.d))

        # Build permutation and reflexion group generators
        for i, (permutation, reflexion) in enumerate(zip(self._gen_permutations, self._gen_reflexions)):
            if np.any(permutation) < 0: self.is_permutation = False
            # Go from one-line notation to matrix form
            self.discrete_generators[i, range(self.d), np.abs(permutation)] = 1
            if not self.is_permutation:
                self.discrete_generators[i, range(self.d), np.abs(permutation)] *= reflexion

        super().__init__()

    def __eq__(self, G2):
        if not isinstance(G2, JointSpaceSymmetry): return False
        if self.d != G2.d or len(self.discrete_generators) != len(G2.discrete_generators): return False
        for h1, h2 in zip(self.discrete_generators, G2.discrete_generators):
            if not jnp.allclose(h1, h2): return False
        return True

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        outstr = f"KO({self.d})-P"
        for p in self._gen_permutations:
            outstr += f"{list(p)}"
        if self.args:
            outstr += '(' + ''.join(repr(arg) for arg in self.args) + ')'
        return outstr


class JointGroup(Group):
    """ The alternating group in n dimensions"""

    def __init__(self, G_in: Group, G_out: Group):

        assert len(G_in.discrete_generators) == len(G_out.discrete_generators)

        self.discrete_generators = []
        for h_in, h_out in zip(G_in.discrete_generators, G_out.discrete_generators):
            a = LazyKron([h_out, h_in.T])
            self.discrete_generators.append(densify(a))
        self.discrete_generators = np.array(self.discrete_generators)
        super().__init__()


if __name__ == "__main__":
    np.random.seed(1)

    # Test
    permutations_out = ((0, 1, 2), (1, 0, 2))
    permutations_in = ((0, 1), (1, 0))
    G_in = JointSpaceSymmetry(gen_permutations=permutations_in)
    G_out = JointSpaceSymmetry(gen_permutations=permutations_out)
    G12 = JointGroup(G_in, G_out)


    V12 = Vector(G12)

    assert G_in == G_in
    V_in = Vector(G_in)
    V_out = Vector(G_out)

    # # Bolt
    # nj = 6
    # permutations_q = ((3, 4, 5, 0, 1, 2),)
    # reflexions_q = ((-1, 1, 1, -1, 1, 1),)
    #
    # # Solo12
    # # nj = 12
    # # permutations_q = [(6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)]
    # # reflexions_q = [(-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
    #
    # G_js = JointSpaceSymmetry(gen_permutations=permutations_q, gen_reflexions=reflexions_q)
    #
    # V_in = Vector(G_js)
    # V_out = Vector(G_js)

    repin = V_in
    repout = V_out

    # V12 = V_in >> V_out

    basis = V12.equivariant_basis()

    visualize_basis(repin, repout, cluster=False)
    plt.show()

    visualize_basis_ind(repin, repout, cluster=True)
    plt.show()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = nn.EMLP(repin, repout, G_js).to(device)  # initialize the model
    #
    # x = torch.tensor([1., 2., 3., 4., 5., 6.])
    #
    # for gen in np.array(G_js.discrete_generators).astype(np.float32):
    #     x_symm = torch.tensor(gen @ x.numpy().astype(np.float32))
    #
    #     y = model(x)
    #     y_symm = model(x_symm)
    #
    #     y_hat = gen @ y_symm.numpy()
    #
    #     assert np.allclose(y.detach().numpy(), y_hat)

    print("Done")
