#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import numpy as np
from sympy import Symbol


def symbolic_matrix(base_name, rows, cols):
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    w = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            var_name = "%s%d,%d" % (base_name, r + 1, c + 1)
            w[r, c] = Symbol(var_name.translate(SUB))
    return w


def permutation_matrix(oneline_notation):
    d = len(oneline_notation)
    P = np.zeros((d, d))
    assert d == len(np.unique(oneline_notation)), np.unique(oneline_notation, return_counts=True)
    P[range(d), np.abs(oneline_notation)] = 1
    return P


def is_canonical_permutation(ph):
    return np.allclose(ph @ ph, np.eye(ph.shape[0]))


if __name__ == "__main__":

    a = (2, 3, 0, 1)
    P = permutation_matrix(a)
    assert is_canonical_permutation(P)