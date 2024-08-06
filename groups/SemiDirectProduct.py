import sys
from typing import Union

import jax
import scipy
from emlp.groups import Group, Trivial
import jax.numpy as jnp
import numpy as np
from emlp.reps.linear_operators import lazify, LazyKron, densify, LazyPerm, LazyKronsum
from emlp.reps.product_sum_reps import ProductRep
from emlp.reps.representation import Vector, Scalar
from emlp.reps.representation import Base as BaseRep
from tqdm import tqdm

from groups.SymmetricGroups import C2, Klein4, Sym
from scipy.sparse.linalg import LinearOperator

from utils.robot_utils import get_robot_params

import logging

from utils.utils import dense

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


class SparseRep(BaseRep):

    def __init__(self, G: Union[Sym, Group]):
        super().__init__()
        self.G = G
        self.is_permutation = G.is_permutation

    def equivariant_basis(self):
        """
        Computes the equivariant solution basis for the given representation of size N. Allowing for
        sparse generator representations.
        TODO: Canonicalizes problems
        and caches solutions for reuse. Output [Q (N,r)] """
        if self == Scalar: return jnp.ones((1, 1))
        canon_rep, perm = self.canonicalize()
        invperm = np.argsort(perm)

        if canon_rep not in self.solcache:
            logging.info(f"{canon_rep} cache miss")
            logging.info(f"Solving basis for {self}" + (f", for G={self.G}" if hasattr(self, "G") else ""))
            # if self.G.is_sparse:
            #     C = self.constraint_matrix()
            #     U, S, VH = scipy.sparse.linalg.svds(C, min(C.shape) - 1)
            #     rank = (S > 1e-5).sum()
            #     return VH[rank:].conj().T
            if self.G.is_sparse:
                # P = self.G.discrete_generators[0]
                Q = self.sparse_equivariant_basis()
                # Q2 = self.sparse_equivariant_basis2()
                # assert np.allclose(Q, Q2)
            else:
                self.G.lie_algebra = []
                Q = Vector(self.G).equivariant_basis()
            self.solcache[canon_rep] = Q
        else:
            log.info(f"{canon_rep} cache found")

        if self.G.is_sparse:
            # TODO: Apply inv perm to sparse matrix, by modifying coordinates directly.
            return self.solcache[canon_rep]
        else:
            return self.solcache[canon_rep][invperm]

    def constraint_matrix(self):
        """ Constructs the equivariance constraint matrix (lazily) by concatenating
        the constraints (ρ(hᵢ)-I) for i=1,...M and dρ(Aₖ) for k=1,..,D from the generators
        of the symmetry group. """
        if self.G.is_sparse:
            n = self.size()
            constraints = []
            constraints.extend([self.rho(h) - scipy.sparse.eye(n) for h in self.G.discrete_generators])
            constraints.extend([self.drho(A) for A in self.G.lie_algebra])
            return scipy.sparse.vstack(constraints) if constraints else None # TODO: Check
        else:
            return super().constraint_matrix()

    def size(self):
        return self.G.d

    def sparse_equivariant_basis(self):
        """
        Custom code to obtain the equivariant basis, without the need to do eigendecomposition. Allowing to compute the
        basis of very large matrix without running into memory or complexity issues
        - Modified code from: shorturl.at/kuvBD
        TODO: Extend to cope with multi group actions, currently works only for C2
        :param P: (n,n) Generalized Permutation matrix with +-1 entries
        :return: Q: (n, b) `b` Eigenvectors of the fix-point equation
        """
        P = self.G.discrete_generators[0]
        log.info(f"Solving equivariant basis using single generalized permutation matrix {P.shape}")
        n = P.shape[0]

        idx = scipy.sparse.eye(n, format='coo', dtype=P.dtype)
        orbits = [idx]
        # First action is identity e
        for i, h in enumerate(self.G.discrete_actions[1:]):
            orbits.append((h @ orbits[0]).tocoo().astype(int))

        # cols_itr = np.asarray([m.col for m in orbits]).T
        rows_itr = np.asarray([m.row for m in orbits]).T
        data_itr = np.asarray([m.data for m in orbits]).T

        pending_dims = np.ones((n,), dtype=bool)

        pbar = tqdm(total=len(pending_dims), disable=False, dynamic_ncols=True, maxinterval=20, position=0, leave=True,
                    file=sys.stdout)
        pbar.set_description("Finding Eigenvectors")

        n_orbit = 0
        eig_cols, eig_rows, eig_data = [], [], []

        # TODO: Improve code readability and efficiency
        # TODO: COO format admits repeated entries, we just need to qualize after avoiding all this problem.
        # +1 is added in order to keep the sign of dimensions 0, otherwise its lost.
        u_orbits = [np.unique((r+1) * d) for r, d in zip(rows_itr, data_itr)]
        inv_dims = [u.item() for u in u_orbits if len(u) == 1]
        if len(inv_dims) > 0:
            eig_rows.extend(np.asarray(inv_dims) - 1)
            eig_cols.extend(range(len(inv_dims)))
            eig_data.extend(list(np.ones(len(inv_dims))))
            n_orbit = len(inv_dims)
            pending_dims[np.asarray(inv_dims) - 1] = False
            pbar.update(len(inv_dims))

        for idxs, vals in zip(rows_itr, data_itr):
            if pending_dims[idxs[0]]:
                eig_rows.extend(idxs)
                eig_cols.extend([n_orbit] * len(idxs))
                eig_data.extend(vals)
                pending_dims[idxs] = False
                n_orbit += 1
                pbar.update(len(idxs))

        assert not np.any(pending_dims), "There seems to be dimensions not included into the Null Space"
        pbar.set_description(f"{n_orbit} eigenvectors found")
        pbar.close()
        Q = scipy.sparse.coo_matrix((eig_data, (eig_rows, eig_cols)), shape=(n, n_orbit))
        return Q

    def __repr__(self):
        return str(self)  # f"T{self.rank+(self.G,)}"

    def __str__(self):
        return f"Vs[{str(self.G)}]"

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

