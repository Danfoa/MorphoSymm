#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
from typing import Optional, Sequence

import jax
import scipy.sparse
from scipy import sparse
from scipy.sparse import issparse
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp

import os
from emlp.groups import Group


class Sym(Group):

    def __init__(self, generators):
        """
        @param generators: (n, d, d) `n` generator in matrix form `(d, d)`, where `d` is the dimension
        of the Vector Space and action representations.
        """
        assert len(generators) > 0, "Zero generator provided"
        self.d = generators[0].shape[0]

        self.is_orthogonal = True
        self.is_permutation = True
        self.is_sparse = False

        self.discrete_generators = []
        # Ensure its orthogonal matrix
        for i, h in enumerate(generators):
            if issparse(h):
                assert np.allclose(sparse.linalg.norm(h, axis=0), 1), f"Generator {i} is not orthogonal: \n{h}"
                if h.min() < 0: self.is_permutation = False
                self.is_sparse = True
            else:
                assert np.allclose(np.linalg.norm(h, axis=0), 1), f"Generator {i} is not orthogonal: \n{h}"
                if np.any(h < 0): self.is_permutation = False

            self.discrete_generators.append(h)

        if not self.is_sparse:
            self.discrete_generators = jnp.asarray(self.discrete_generators)

        # Count number of dimensions that are invariant.
        h_diags = np.array([h.diagonal() for h in self.discrete_generators] * 4)
        inv_h_diags = np.array(h_diags) == 1
        inv_dims = np.all(inv_h_diags, axis=0)
        self.inv_dims = inv_dims
        self.n_inv_dims = np.sum(inv_dims).item()

        super().__init__()

    @property
    def discrete_actions(self) -> list:
        raise NotImplementedError()

    def __hash__(self):
        return hash(str(self.discrete_generators))

    def __repr__(self):
        return f"Sym({self.d})"

    def generators_characters(self):
        characters = []
        for h in self.discrete_generators:
            characters.append(np.trace(h))
        return characters

    @staticmethod
    def oneline2matrix(oneline_notation, reflexions: Optional[Sequence] = None):

        d = len(oneline_notation)
        # P = np.zeros((d, d)).astype(np.int8)
        assert d == len(np.unique(oneline_notation)), np.unique(oneline_notation, return_counts=True)

        reflexions = np.ones((d,), dtype=np.int8) if not reflexions else reflexions
        assert len(reflexions) == d, f"{len(reflexions)} != {d}"

        rows, cols = range(d), np.abs(oneline_notation)
        P = scipy.sparse.coo_matrix((reflexions, (rows, cols)), shape=(d, d))
        # P2 = np.zeros((d, d))
        # P2[range(d), np.abs(oneline_notation)] = 1 * np.array(reflexions).astype(np.int8)
        # assert np.allclose(P.todense(), P2)
        return P.astype(np.int8)

    @property
    def np_gens(self):
        return np.array([h.todense() for h in self.discrete_generators])

    @staticmethod
    def canonical_group(d, inv_dims: int = 0) -> 'Sym':
        raise NotImplementedError()


class C2(Sym):

    def __init__(self, generator):
        """
        @param generator: (d, d) generators in matrix form, where `d` is the dimension
        of the Vector Space and action representations.
        """
        generators = [generator] if not isinstance(generator, list) else generator
        super().__init__(generators)
        assert len(self.discrete_generators) == 1, "C2 must contain only one generator (without counting the identity)"

        h = self.discrete_generators[0]

        is_eye = np.isclose(sum(h.diagonal()), self.d) if self.is_sparse else jnp.isclose(jnp.trace(h), self.d)
        is_cyclic = np.isclose(sum((h @ h).diagonal()), self.d) if self.is_sparse else jnp.isclose(jnp.trace(h @ h), self.d)
        assert not is_eye, f"Generator must not be the identity: \n {h}"
        assert is_cyclic, f"Generator is not cyclic h @ h != I"


    @property
    def discrete_actions(self) -> list:
        return [sparse.eye(self.d, format='coo'), self.discrete_generators[0]]

    def __repr__(self):
        return f"C2[d:{self.d}]" if self.n_inv_dims == 0 else f"C2[d:{self.d}|inv:{self.n_inv_dims}]"

    @staticmethod
    def canonical_group(d, inv_dims: int = 0) -> 'C2':
        """
        @param d: Vector Space dimension
        """
        assert d > 0, "Vector space dimension must be greater than 0"
        assert inv_dims < d - 1, "At least a single dimension must be symmetric"

        id = inv_dims

        # Get fully equivariant representation.
        p = np.flip(np.arange(d))
        r = np.ones_like(p)

        if d % 2 > 0:   # Odd dimensional repr: Can have odd number of `inv_dims`
            if id % 2 == 0:
                r[d//2] *= -1
            else:
                id -= 1    # Count middle point invariance
        else:           # Even dimensional repr: Cannot have odd number of `inv_dims`
            if id % 2 > 0:
                inv_dims += 1
                id = inv_dims

        # Enforce `inv_dims` invariant dimensions.
        n = id // 2
        if n > 0:
            p_copy = np.copy(p)
            p[:n] = np.flip(p_copy[-n:])
            p[-n:] = np.flip(p_copy[:n])

        H = Sym.oneline2matrix(oneline_notation=p.tolist(), reflexions=r.tolist())
        # HH = np.asarray(H.todense())
        G = C2(generator=H)
        assert G.n_inv_dims == inv_dims, G.n_inv_dims
        return G

    @staticmethod
    def get_equivariant_basis(P):
        """
        Custom code to obtain the equivariant basis, without the need to do eigendecomposition. Allowing to compute the
        basis of very large matrix without running into memory or complexity issues
        :param P: (n,n) Generalized Permutation matrix with +-1 entries
        :return: Q: (n, b) `b` Eigenvectors of the fix-point equation
        """
        dtype = P.dtype
        # Modified code from: shorturl.at/kuvBD
        n = len(P)
        # compute the cyclic decomposition. a.k.a orbit for each dimension of the vec space acted by the gen permutation
        w = np.abs(P) @ np.arange(n).astype(np.int32)
        pendind_dims = set(range(n))
        cycles = []

        pbar = tqdm(total=n, disable=False,  dynamic_ncols=True, maxinterval=20, position=0, leave=True)
        while pendind_dims:
            a = pendind_dims.pop()  # Get the initial point of an orbit.
            pbar.update(1)
            # pendind_dims.remove(a)
            cycles.append([a])
            while w[a] in pendind_dims:
                a = w[a]
                cycles[-1].append(a)
                pendind_dims.remove(a)
                pbar.update(1)
        pbar.close()

        # obtain the eigenvectors
        Q = np.zeros((n, 0)).astype(dtype)
        for i, cyc in enumerate(cycles):
            p = np.sum(P[cyc, :], axis=1)
            if np.prod(p) == 1:
                Q = np.hstack((Q, np.zeros((n, 1), dtype=dtype)))
                Q[cyc, -1] = [np.prod(p[i:-1]) for i in range(len(p))]
        return Q

class Klein4(Sym):

    def __init__(self, generators):
        """
        @param generators: (2,d,d) Two generators in matrix form (excluding the identity), where `d` is the dimension
        of the Vector Space and action representations.
        """
        assert len(generators) == 2, "Provide only the non-trivial generators (2)"
        super().__init__(generators)

        # Assert generators and their composition is cylic. That is, assert generators produce an abelian group
        a, b = self.discrete_generators

        is_eye = lambda x: np.isclose(sum(x.diagonal()), self.d) if self.is_sparse else jnp.isclose(jnp.trace(x), self.d)
        a_is_eye, b_is_eye, ab_is_eye = is_eye(a), is_eye(b), is_eye(a @ b)
        is_cyclic = lambda x: np.isclose(sum((x @ x).diagonal()), self.d) if self.is_sparse else jnp.isclose(jnp.trace(x @ x), self.d)
        a_is_cyclic, b_is_cyclic, ab_is_cyclic = is_cyclic(a),  is_cyclic(b),  is_cyclic(a @ b)

        assert not a_is_eye and not b_is_eye, f"Generators cannot be the identity a != b != e"
        assert a_is_cyclic, f"Generator is not cyclic:\n{a @ a}"
        assert b_is_cyclic, f"Generator is not cyclic:\n{b @ b}"
        assert ab_is_cyclic, f"Generators composition a·b is not cyclic:\n{a @ b}"
        assert not ab_is_eye, f"Third action must be non-trivial: a·b != e"

    @property
    def discrete_actions(self) -> list:
        a, b = self.discrete_generators
        return [sparse.eye(self.d, format='coo'), a, b, a@b]

    def __hash__(self):
        return hash(str(self.discrete_generators))

    def __repr__(self):
        return f"V4[d:{self.d}]" if self.n_inv_dims == 0 else f"V4[d:{self.d}|inv:{self.n_inv_dims}]"

    @staticmethod
    def canonical_group(d, inv_dims: int = 0) -> 'Klein4':
        """
        @param d: Vector Space dimension
        """
        assert d > 0, "Vector space dimension must be greater than 0"

        # Representation reflections
        r_a = np.ones((d,))
        r_b = np.ones((d,))

        mod = d % 4

        # Find the appropiate number of invariant dimensions.
        feasible_inv_dims = inv_dims - mod if inv_dims > mod else mod
        for i in range(5):
            if (d - i - feasible_inv_dims) % 4 == 0:  # Prioratize invariance rather than equivariance
                feasible_inv_dims += i
                break
            elif (d + i - feasible_inv_dims) % 4 == 0:
                feasible_inv_dims -= i
                break

        if feasible_inv_dims % 2 > 0:  # Odd number of inv dims
            feasible_inv_dims -= 1

        idx = np.array(range(d))
        equiv_dims = idx[feasible_inv_dims//2: len(idx) - feasible_inv_dims//2]
        parts = np.array_split(equiv_dims, indices_or_sections=4)

        equiv_a = np.concatenate((parts[1], parts[0], parts[3], parts[2])).tolist()
        equiv_b = np.concatenate((parts[2], parts[3], parts[0], parts[1])).tolist()

        a = np.concatenate((idx[:feasible_inv_dims//2], equiv_a, idx[len(idx) - feasible_inv_dims//2:]))
        b = np.concatenate((idx[:feasible_inv_dims//2], equiv_b, idx[len(idx) - feasible_inv_dims//2:]))

        rep_a = C2.oneline2matrix(a,)
        rep_b = C2.oneline2matrix(b,)
        G = Klein4(generators=[rep_a, rep_b])

        assert G.n_inv_dims == feasible_inv_dims, G.n_inv_dims
        return G

    def is_canonical(self):
        return np.allclose(self.generators_characters(), 0.0)