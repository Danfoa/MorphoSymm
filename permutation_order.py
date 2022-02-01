import itertools

import matplotlib.pyplot as plt
import numpy as np
from sympy.interactive.printing import init_printing

from robot_kinematic_symmetries import JointSpaceSymmetry, SemiDirectProduct, is_equivariant_invariant
from utils.utils import permutation_matrix, is_canonical_permutation

init_printing(use_unicode=False, wrap_line=False)

from emlp.reps import Vector
from emlp.reps.linear_operators import lazify, LazyKron, densify

from utils.emlp_visualization import *

import torch

torch.set_default_dtype(torch.float32)


def distance(a, b):
    return np.sum(np.array(a) != np.array(b))


if __name__ == "__main__":
    np.random.seed(1)

    Din = 6
    Dout = 5
    idxIn = list(range(Din))
    idxOut = list(range(Dout))
    # gin = [2, 1, 0, ]  # [2, 0, 1] #list(reversed(idx))
    gin = list(reversed(idxIn))
    Pin = permutation_matrix(gin)
    assert is_canonical_permutation(Pin)

    Permutations = list(itertools.permutations(idxOut))
    # Permutations = [(4, 2, 3, 1, 0), (4, 2, 3, 0, 1)]

    fig, (ax, ax1) = plt.subplots(1, 2)
    for gout in Permutations:
        Pout = permutation_matrix(gout)
        # if not is_canonical_permutation(Pout):
        #     continue

        permutations_in = [gin]
        permutations_out = [gout]
        Gin = JointSpaceSymmetry(gen_permutations=permutations_in)
        Gout = JointSpaceSymmetry(gen_permutations=permutations_out)
        # print(np.array(Gin.discrete_generators[0]),"\n", np.array(Gin.discrete_generators[0]).T)
        G = SemiDirectProduct(Gin=Gin, Gout=Gout)
        Vin = Vector(Gin)
        Vout = Vector(Gout)
        Vin_out = Vector(G)

        Q_sub = np.array(densify(Vin_out.equivariant_basis()))
        C_sub = densify(Vin_out.constraint_matrix())
        Q2 = np.array(Matrix(C_sub).nullspace()).T

        W_sym = symbolic_matrix(base_name="w", rows=Gout.d, cols=Gin.d)
        w_sym_flat = np.reshape(W_sym, (np.prod(W_sym.shape), 1))
        #
        # visualize_basis_ind(Vin_out, Gin.d, Gout.d, title=str(Vin_out.G))
        # visualize_basis_sym(Vin_out, Gin.d, Gout.d, title=str(Vin_out.G))
        # plt.show()
        # is_equivariant_invariant(Q_sub, Gin, Gout)
        #
        # plot_system_of_equations(C_sub, w_sym_flat, w_sym_flat)
        assert Q2.shape == Q_sub.shape

        print(
            f"in:{gin}-out:{gout}-din:{distance(idxIn, gin)}-dout:{(distance(gin, gout), distance(idxOut, gout))}-N:{Q_sub.shape}")
        ax.scatter(distance(gin, gout), Q_sub.shape[1])
        ax1.scatter(distance(gout, idxOut), Q_sub.shape[1])
        # break
    fig.show()
