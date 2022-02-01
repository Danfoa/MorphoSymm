#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import numpy as np
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

        self.discrete_generators = np.array(generators).astype(np.int)
        super().__init__()

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
    def oneline2matrix(oneline_notation, reflexions=1):
        d = len(oneline_notation)
        P = np.zeros((d, d))
        assert d == len(np.unique(oneline_notation)), np.unique(oneline_notation, return_counts=True)
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


class Sym3(Sym):

    def __init__(self, generators):
        """
        @param generators: (2,d,d) Two generators in matrix form (excluding the identity), where `d` is the dimension
        of the Vector Space and action representations.
        """
        assert len(generators) == 2, "Provide only the non-trivial generators (2)"
        super().__init__(generators)

        for h in self.discrete_generators:
            assert np.allclose(h @ h, np.eye(self.d)), f"Generator is not cyclic:\n{h @ h}"


    def __hash__(self):
        return hash(str(self.discrete_generators))

    def __repr__(self):
        return f"Sym3[d:{self.d}]"

    @staticmethod
    def canonical_group(d) -> 'Sym3':
        """
        @param d: Vector Space dimension
        """
        assert d > 0, "Vector space dimension must be greater than 0"
        h1 = list(reversed(range(d)))
        p = np.array(np.split(np.arange(d), 4))[[1, 0, 3, 2]]
        # h2p =
        h2 = np.concatenate(p)
        H1 = C2.oneline2matrix(h1)
        H2 = C2.oneline2matrix(h2)
        G = Sym3(generators=[H1, H2])
        assert G.is_canonical()
        return G

    def is_canonical(self):
        return np.allclose(self.generators_characters(), 0.0)