import logging
import sys
from typing import Union

import numpy as np
import scipy
from emlp import Group
from emlp.reps.representation import Base as BaseRep, Scalar, Vector
from jax import numpy as jnp
from scipy import sparse
from tqdm import tqdm

from groups.SymmetryGroups import Sym
log = logging.getLogger(__name__)


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
                Q = self.sparse_equivariant_basis_gen_permutation()
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

    def sparse_equivariant_basis_gen_permutation(self):
        """
        Custom code to obtain the equivariant basis, without the need to do eigendecomposition. Allowing to compute the
        basis of very large matrix without running into memory or complexity issues.
        This code only works for regular representations that can be represented as a generalized permutation matrix.
        This covers most cases of interest in Discrete Morphological Symmetries, specially in the internal symmetry
        representations of equivariant NNs, where we have control over the nature of the representation.
        - Modified code from: shorturl.at/kuvBD
        :param P: (n,n) Generalized Permutation matrix with `+-c` entries where `c` is a scalar constant.
        :return: Q: (n, b) `b` Eigenvectors of the fix-point equation
        """
        P = self.G.discrete_generators[0]
        log.info(f"Solving equivariant basis using single generalized permutation matrix {P.shape}")
        n = P.shape[0]

        idx = scipy.sparse.eye(n, format='coo', dtype=P.dtype)
        orbits = [idx]
        # First action is identity e
        for i, h in enumerate(self.G.discrete_actions[1:]):
            orbits.append((h @ orbits[0]).tocoo().astype(np.int))

        # Extract the data from the orbit of w: G(w) = G(w) = [c_1*w_1, c_10*w_10, ... , c_50*w_50]
        # Rows iter describes the orbit G(w) locations/idx e.g., [w_1, w_10,...,w_50]
        rows_itr = np.asarray([m.row for m in orbits]).T
        # Data iter describes the orbit G(w) scalar coefficients e.g., [c_1, c_10,...,c_50]
        data_itr = np.asarray([m.data for m in orbits]).T

        # Orbits are permutation invariant: [c_1*w_1, c_10*w_10, ... , c_50*w_50] := [c_50*w_50,..., c_10*w_10, c_1*w_1]
        # By extracting the location of the orbits we can ignore repeated orbits
        # A set is viable alternative to account for repeated orbits but has a higher memory cost
        pending_dims = np.ones((n,), dtype=np.bool)

        pbar = tqdm(total=len(pending_dims), disable=False, dynamic_ncols=True, maxinterval=20, position=0, leave=True,
                    file=sys.stdout)
        pbar.set_description("Finding Eigenvectors")

        n_orbit = 0
        eig_cols, eig_rows, eig_data = [], [], []

        # # TODO: Improve code readability and efficiency
        # # TODO: COO format admits repeated entries, we just need to qualize after avoiding all this problem.
        # # +1 is added in order to keep the sign of dimensions 0, otherwise its lost.
        # u_orbits = [np.unique((r+1) * d) for r, d in zip(rows_itr, data_itr)]
        # inv_dims = [u.item() for u in u_orbits if len(u) == 1]
        # if len(inv_dims) > 0:
        #     eig_rows.extend(np.asarray(inv_dims) - 1)
        #     eig_cols.extend(range(len(inv_dims)))
        #     eig_data.extend(list(np.ones(len(inv_dims))))
        #     n_orbit = len(inv_dims)
        #     pending_dims[np.asarray(inv_dims, dtype=np.int) - 1] = False
        #     pbar.update(len(inv_dims))

        for w_idx, w_coeff in zip(rows_itr, data_itr):
            if pending_dims[w_idx[0]]:
                u_idx = set(w_idx)
                orbit = {}
                if len(u_idx) != len(w_idx):  # Dealing with invariant dimensions cases
                    for idx, val in zip(w_idx, w_coeff):
                        if idx in orbit:
                            orbit[idx] = 0 if val != orbit[idx] else val
                        else:
                            orbit[idx] = val
                else:                         # Equivariant dimensions alone
                    orbit = {idx: val for idx, val in zip(w_idx, w_coeff)}

                eig_rows.extend(orbit.keys())
                eig_cols.extend([n_orbit] * len(orbit))
                eig_data.extend(orbit.values())
                pending_dims[list(orbit.keys())] = False
                n_orbit += 1
                pbar.update(len(orbit))

        assert not np.any(pending_dims), "There seems to be dimensions not included into the Null Space"
        pbar.set_description(f"{n_orbit} eigenvectors found")
        pbar.close()
        Q = scipy.sparse.coo_matrix((eig_data, (eig_rows, eig_cols)), shape=(n, n_orbit))
        return Q

    def __repr__(self):
        return str(self)  # f"T{self.rank+(self.G,)}"

    def __str__(self):
        return f"ρ({str(self.G)})"

    def __add__(self, other):
        """ Direct sum (⊕) of representations. """
        # TODO: Keep lazy SumRep representation to partition the solution of the fix-point equation in the case of
        # non generalized-matrix representations.
        assert isinstance(self, SparseRep) and isinstance(other, SparseRep), f"{type(self)} != {type(other)}"
        G_class = type(self.G)
        G = G_class([sparse.block_diag((self.rho(g1), other.rho(g2))) for g1, g2 in zip(self.G.discrete_generators,
                                                                                        other.G.discrete_generators)])
        return SparseRep(G)

    def __mul__(self, other):
        """ For not convenient way to express sumation as multiplications """
        if type(other) == int:
            n_repetitions = other
            G_class = type(self.G)
            G = G_class([sparse.block_diag([self.rho(g)]*n_repetitions) for g in self.G.discrete_generators])
            return SparseRep(G)
        else:
            raise NotImplementedError(f"Mul operator not defined for {type(other)}")

    def __rmul__(self, other):
        """ For not convenient way to express sumation as multiplications """
        return self.__mul__(other)


# This will be called in the situation you brought up.
class SparseRepE3(SparseRep):

    def __init__(self, G: Union[Sym, Group], pseudovector=False):
        super().__init__(G)
        self.pseudovector = pseudovector

    def rho(self, M):
        if self.pseudovector:
            det = np.linalg.det(M.todense())
            return M * det
        return M