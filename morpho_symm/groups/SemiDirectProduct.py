import scipy
import numpy as np
from emlp.reps.linear_operators import LazyKron
from emlp.reps.representation import Vector

from .SparseRepresentation import SparseRep
from .SymmetryGroups import C2, Sym
from scipy.sparse.linalg import LinearOperator

import logging

from utils.algebra_utils import dense

log = logging.getLogger(__name__)

class SemiDirectProduct(Sym):
    """SemiDirectProduct"""

    def __init__(self, Gin: Sym, Gout: Sym):

        assert len(Gin.discrete_generators) == len(Gout.discrete_generators)
        self.is_sparse = Gin.is_sparse and Gout.is_sparse
        self.is_orthogonal = Gin.is_orthogonal and Gout.is_orthogonal
        self.is_permutation = Gin.is_permutation and Gout.is_permutation

        self.discrete_generators = []
        for h_in, h_out in zip(Gin.discrete_generators, Gout.discrete_generators):
            if self.is_sparse:
                a = scipy.sparse.kron(h_out, h_in)
            else:
                a = LazyKron([dense(h_out), dense(h_in)])
            self.discrete_generators.append(a)

        self.G1 = Gin
        self.G2 = Gout
        self.d = Gin.d * Gout.d

        # TODO: Make functional for continuous groups
        self.lie_algebra = []

    @property
    def discrete_actions(self) -> list:
        actions = []
        for g_in, g_out in zip(self.G1.discrete_actions, self.G2.discrete_actions):
            if self.is_sparse:
                a = scipy.sparse.kron(g_out, g_in)
            else:
                a = LazyKron([dense(g_out), dense(g_in)])
            actions.append(a)
        return actions

    def get_inout_generators(self):
        return np.array(self.G1.discrete_generators, dtype=np.float32), \
               np.array(self.G2.discrete_generators, dtype=np.float32)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        outstr = f'{repr(self.G1)} ⋊ {repr(self.G2)}'
        return outstr


if __name__ == "__main__":

    # robot, Gin, Gout, _, _ = get_robot_params("solo")

    # Gin = Klein4.canonical_group(4)
    # Gout = Klein4.canonical_group(8)
    Gin = C2.canonical_group(4)
    Gout = C2.canonical_group(8)

    rep_in = Vector(Gin)
    rep_out = Vector(Gout)

    G = SemiDirectProduct(Gin, Gout)
    rep = Vector(G)
    rep2 = SparseRep(rep_in=Vector(Gin), rep_out=Vector(Gout))

    # C = rep.constraint_matrix()
    # Cin = rep_in.constraint_matrix().to_dense()
    # Cout = rep_out.constraint_matrix().to_dense()
    #
    # U, S, VH = jnp.linalg.svd(C.to_dense(), full_matrices=True)
    # Uin, Sin, VHin = jnp.linalg.svd(Cin, full_matrices=True)
    # Uout, Sout, VHout = jnp.linalg.svd(Cout, full_matrices=True)
    #
    # rank = np.sum(S > 1e-5)
    # rank_in = np.sum(Sin > 1e-5)
    # rank_out = np.sum(Sout > 1e-5)

    # C2 = np.kron(Cout, Cin)
    # S2 = np.kron(Sout, Sin)
    # V2 = np.kron(VHin, VHout)

    # print(S2[S2 < 1e-5])
    # Q2 = V2[:, S2 < 1e-4]

    Q = np.asarray(rep.equivariant_basis())
    Q2 = rep2.equivariant_basis()
    # rep = SemiDirectProductRep(Vector(Gin), Vector(Gout))
    # assert isinstance(C, LinearOperator)
    # u, s, vh = Qf = scipy.sparse.linalg.svds(C, k=16)
    # rep.equivariant_basis()
    print()

