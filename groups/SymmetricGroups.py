#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
from typing import Optional, Sequence

import numpy as np
import jax.numpy as jnp
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
        for i, h in enumerate(generators):
            # Ensure its orthogonal matrix
            assert np.allclose(np.linalg.norm(h, axis=0), 1), f"Generator {i} is not orthogonal: \n{h} "
            if np.any(h < 0): self.is_permutation = False
        # TODO: Make everything Sparse and Lazy. Avoid memory and runtime excess
        self.discrete_generators = np.array(generators).astype(np.int)
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
        P = np.zeros((d, d))
        assert d == len(np.unique(oneline_notation)), np.unique(oneline_notation, return_counts=True)
        reflexions = 1 if not reflexions else reflexions
        P[range(d), np.abs(oneline_notation)] = 1 * np.array(reflexions)
        return P

    @property
    def np_gens(self):
        return np.array(self.discrete_generators)

class C2(Sym):

    def __init__(self, generator):
        """
        @param generator: (d, d) generators in matrix form, where `d` is the dimension
        of the Vector Space and action representations.
        """
        super().__init__([generator])
        assert len(self.discrete_generators) == 1, "C2 must contain only one generator (without counting the identity)"

        h = self.discrete_generators[0]
        assert not np.allclose(h, np.eye(self.d)), "Generator must not be the identity"
        assert np.allclose(h @ h, np.eye(self.d)), "Generator is not cyclic"

    @property
    def discrete_actions(self) -> list:
        return [jnp.eye(self.d, dtype=jnp.int32), self.discrete_generators[0]]

    def __repr__(self):
        return f"C2[d:{self.d}]"

    @staticmethod
    def canonical_group(d) -> 'C2':
        """
        @param d: Vector Space dimension
        """
        assert d > 0, "Vector space dimension must be greater than 0"
        h = list(reversed(range(d)))
        H = C2.oneline2matrix(h)
        G = C2(generator=H)
        return G


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
        # assert not np.allclose(a, np.eye(self.d)) and not np.allclose(b, np.eye(self.d)), f"Provide only two non-trivial generators"
        assert np.allclose(a @ a, np.eye(self.d)), f"Generator is not cyclic:\n{a @ a}"
        assert np.allclose(b @ b, np.eye(self.d)), f"Generator is not cyclic:\n{b @ b}"
        assert np.allclose((a@b) @ (a@b), np.eye(self.d)), f"Generators composition a·b is not cyclic:\n{a@b}"
        assert not np.allclose(a@b, np.eye(self.d)), f"Third action must be non-trivial: a·b != e"

    @property
    def discrete_actions(self) -> list:
        a, b = self.discrete_generators
        return [jnp.eye(self.d, dtype=jnp.int32), a, b, a@b]

    def __hash__(self):
        return hash(str(self.discrete_generators))

    def __repr__(self):
        return f"V4[d:{self.d}]"

    @staticmethod
    def canonical_group(d) -> 'Klein4':
        """
        @param d: Vector Space dimension
        """
        assert d > 0, "Vector space dimension must be greater than 0"
        a = list(reversed(range(d)))

        #
        mod = d % 4
        idx = np.array_split(range(d - mod), indices_or_sections=4)
        b_r = np.ones((d,))
        if mod > 0:
            r_idx = np.array(range(d-mod, d))
            b = np.concatenate((idx[2], idx[3], idx[0], idx[1], r_idx)).tolist()
            b_r[-mod:] = -1
            raise NotImplementedError("TODO: Deal with case where impossible to get all representations irreducible")
        else:
            b = np.concatenate((idx[2], idx[3], idx[0], idx[1])).tolist()

        rep_a = C2.oneline2matrix(a)
        rep_b = C2.oneline2matrix(b, reflexions=b_r.tolist())
        G = Klein4(generators=[rep_a, rep_b])
        return G

    def is_canonical(self):
        return np.allclose(self.generators_characters(), 0.0)