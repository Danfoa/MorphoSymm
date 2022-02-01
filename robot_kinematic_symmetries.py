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
            # print(f"Creating the group of kinematic symmetries KO ⊆ O({self.d})")
            self.is_permutation = False
            self._gen_reflexions = np.array(gen_reflexions).astype(np.int)
            assert np.all(np.abs(self._gen_reflexions)) == 1, "One-line notation of reflexions must contain only {1,-1}"
            assert self._gen_reflexions.shape == self._gen_permutations.shape
        else:  # Symmetries are permutations_q of state variable dimensions
            # print(f"Creating the group of kinematic symmetries KO ⊆ S({self.d})")
            self.is_permutation = True
            self._gen_reflexions = np.ones_like(gen_permutations)

        self.discrete_generators = np.zeros((num_generators, self.d, self.d))

        # Build permutation and reflexion group generator
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


class SemiDirectProduct(Group):
    """SemiDirectProduct"""

    def __init__(self, Gin: Group, Gout: Group):

        self.G1 = Gin
        self.G2 = Gout
        assert len(Gin.discrete_generators) == len(Gout.discrete_generators)

        self.discrete_generators = []
        for h_in, h_out in zip(Gin.discrete_generators, Gout.discrete_generators):
            a = LazyKron([h_out, h_in])
            self.discrete_generators.append(densify(a))
        self.discrete_generators = np.array(self.discrete_generators)
        super().__init__()

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        outstr = f'({repr(self.G1)})⋊({self.G2})'
        if self.args:
            outstr += '(' + ''.join(repr(arg) for arg in self.args) + ')'
        return outstr
#
# class SemidirectProduct(ProductRep):
#     """ Tensor product of representations ρ₁⋊ρ₂, but where the sub representations
#         ρ₁ and ρ₂ are representations of distinct groups (ie ρ₁⊗ρ₂ is a representation
#         of the direct product of groups G=G₁×G₂). As a result, the solutions for the two
#         sub representations can be solved independently and assembled together with the
#         kronecker product: basis = basis₁⊗basis₂ and P = P₁⊗P₂"""
#
#     def __init__(self, *reps, counter=None, extra_perm=None):
#         # Two variants of the constructor:
#         if counter is not None:  # one with counter specified directly
#             self.reps = counter
#             self.reps, perm = self.compute_canonical([counter], [np.arange(self.size())])
#             self.perm = extra_perm[perm] if extra_perm is not None else perm
#         else:  # other with list
#             reps, perms = zip(*[rep.canonicalize() for rep in reps])
#             # print([type(rep) for rep in reps],type(rep1),type(rep2))
#             rep_counters = [rep.reps if type(rep) == DirectProduct else {rep: 1} for rep in reps]
#             # Combine reps and permutations: Pi_a + Pi_b = Pi_{a x b}
#             reps, perm = self.compute_canonical(rep_counters, perms)
#             # print("dprod init",self.reps)
#             group_dict = defaultdict(lambda: 1)
#             for rep, c in reps.items():
#                 group_dict[rep.G] = group_dict[rep.G] * rep ** c
#             sub_products = {rep: 1 for G, rep in group_dict.items()}
#             self.reps = counter = sub_products
#             self.reps, perm2 = self.compute_canonical([counter], [np.arange(self.size())])
#             self.perm = extra_perm[perm[perm2]] if extra_perm is not None else perm[perm2]
#         self.invperm = np.argsort(self.perm)
#         self.canonical = (self.perm == self.invperm).all()
#         # self.G = tuple(set(rep.G for rep in self.reps.keys()))
#         # if len(self.G)==1: self.G= self.G[0]
#         self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())
#         assert all(count == 1 for count in self.reps.values())
#
#     def equivariant_basis(self):
#         a = [rep.equivariant_basis() for rep, c in self.reps.items()]
#         canon_Q = LazyKron([rep.equivariant_basis() for rep, c in self.reps.items()])
#         return LazyPerm(self.invperm) @ canon_Q
#
#     def equivariant_projector(self):
#         canon_P = LazyKron([rep.equivariant_projector() for rep, c in self.reps.items()])
#         return LazyPerm(self.invperm) @ canon_P @ LazyPerm(self.perm)
#
#     def rho(self, Ms):
#         canonical_lazy = LazyKron([rep.rho(Ms) for rep, c in self.reps.items() for _ in range(c)])
#         return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)
#
#     def drho(self, As):
#         canonical_lazy = LazyKronsum([rep.drho(As) for rep, c in self.reps.items() for _ in range(c)])
#         return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)
#
#     def __str__(self):
#         superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
#         return "⊗".join([str(rep) + f"_{rep.G}" for rep, c in self.reps.items()])
#
#     # # Bolt
#     # nj = 6
#     # permutations_q = ((3, 4, 5, 0, 1, 2),)
#     # reflexions_q = ((-1, 1, 1, -1, 1, 1),)
#     #
#     # Solo12
#     # nj = 12
#     # permutations_q = [(6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)]
#     # reflexions_q = [(-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
#
#     # G_js = JointSpaceSymmetry(gen_permutations=permutations_q, gen_reflexions=reflexions_q)
#     #
#     # Vin = Vector(G_js)
#     # Vout = Vector(G_js)

def is_equivariant_invariant(Q, G_in, G_out):
    W = Q @ np.random.rand(Q.shape[1], 1)
    W = W.reshape((G_out.d, G_in.d))

    g = G_in.sample()
    while np.allclose(g, np.eye(g.shape[0])):
        g = G_in.sample()
    x = np.random.rand(G_in.d)
    x_hat = g @ x

    y = W @ x
    y_hat = W @ x_hat
    print(f"x:\t{x}\nx':\t{x_hat}")
    print(f"y:\t{y}\ny':\t{y_hat}")
    invariant = np.allclose(y, y_hat)
    print("INVARIANT" if invariant else "EQUIVARIANT")

