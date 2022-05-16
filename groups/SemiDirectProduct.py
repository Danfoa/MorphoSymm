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
                a = LazyKron([h_out, h_in])
            self.discrete_generators.append(a)

        self.G1 = Gin
        self.G2 = Gout
        self.d = Gin.d * Gout.d

        # TODO: Make functional for continuous groups
        self.lie_algebra = []

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
            if self.G.is_sparse and len(self.G.discrete_generators) == 1:
                # P = self.G.discrete_generators[0]
                Q = self.sparse_equivariant_basis()
            else:
                self.G.lie_algebra = []
                Q = Vector(self.G).equivariant_basis()
            self.solcache[canon_rep] = Q
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
        dtype = P.dtype
        n = P.shape[0]
        # compute the cyclic decomposition. a.k.a orbit for each dimension of the vec space acted by the gen permutation
        # w = np.abs(P) @ np.arange(n).astype(np.int32)
        w = P.row   # oneline notation

        pendind_dims = set(range(n))
        cycles = []
        pbar = tqdm(total=n, disable=False, dynamic_ncols=True, maxinterval=20, position=0, leave=True,
                    file=sys.stdout)
        pbar.set_description("Finding Orbits")
        while pendind_dims:
            a = pendind_dims.pop()  # Get the initial point of an orbit.
            pbar.update(1)
            cycles.append([a])
            while w[a] in pendind_dims:
                a = w[a]
                cycles[-1].append(a)
                pendind_dims.remove(a)
                pbar.update(1)
        pbar.refresh()
        pbar.close()

        # obtain the eigenvectors
        tmp = P.sum(axis=1)

        pbar = tqdm(total=len(cycles), disable=False, dynamic_ncols=True, maxinterval=20, position=0, leave=True,
                    file=sys.stdout)
        pbar.set_description("Finding Eigenvectors")
        eig_cols, eig_rows, eig_data = [], [], []
        for n_orbit, orbit in enumerate(cycles):
            p = tmp[orbit, 0]
            if np.prod(p) == 1:
                # Coordinates of the eigenvector values
                cols = [n_orbit] * len(orbit)
                rows = orbit
                # Values of the eigenvector
                data = [np.prod(p[j:-1]) for j in range(len(p))]

                eig_data.extend(data)
                eig_rows.extend(rows)
                eig_cols.extend(cols)
                pbar.update(1)
        pbar.refresh()
        pbar.close()
        Q = scipy.sparse.coo_matrix((eig_data, (eig_rows, eig_cols)), shape=(n, len(cycles)))
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

