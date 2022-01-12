import numpy as np
from sympy import symbols, Symbol
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt


def symbolic_matrix(base_name, rows, cols):
    w = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            w[r, c] = Symbol("w_%d%d" % (r+1, c+1))
    return w


def permutation_matrix(oneline_notation):
    d = len(oneline_notation)
    assert np.unique(oneline_notation) == d

    p = np.zeros((d, d))
    p[range(d), np.abs(oneline_notation)] = 1
    return p


if __name__ == "__main__":

    H1 = [(2, 3, 0, 1), (0, 1, 2, 3), (0, 2, 1, 3)]
    H2 = [(2, 0, 1), (0, 1, 2), (1, 0, 2)]

    N1 = 4
    N2 = 3
    W = symbolic_matrix(base_name="w", rows=N2, cols=N1)

