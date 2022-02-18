import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sympy
from emlp import Group
from emlp.reps.linear_operators import densify
from sklearn.cluster import KMeans
from sympy import Matrix

from .utils import symbolic_matrix


def visualize_basis_stats(basis):

    Q = np.asarray(basis)
    u_vals_Q = [np.unique(np.round(q, 3), return_counts=True) for q in Q.T]

    u_vals = [len(u_vals_q) for u_vals_q, count in u_vals_Q]

    # Plot a histogram showing the distribution of free parameters per basis.
    data = pd.DataFrame.from_dict({"Unique w per basis": np.array(u_vals, dtype=np.int)})
    plt.figure()
    ax = sns.histplot(data=data, x="Unique w per basis", discrete=True,
                      element="bars", stat="probability")
    ax.get_figure().show()
    print("Hey")

def visualize_basis(repin, repout, cluster=True):
    """ A function to visualize the basis of equivariant maps repin>>repout
        as an image. Only use cluster=True if you know Pv will only have
        r distinct values (true for G_js<S(n) but not true for many continuous groups)."""
    rep = (repin >> repout)
    P = rep.equivariant_projector()  # compute the equivariant basis
    Q = rep.equivariant_basis()

    n_basis = Q.shape[-1]


    v = np.random.randn(P.shape[1])  # sample random vector
    v = np.round(P @ v, decimals=4)  # project onto equivariant subspace (and round)
    u_vals = np.unique(v)

    cmap_name = 'nipy_spectral' #'tab20' if len(u_vals) <= 20 else 'hsv'

    if cluster:  # cluster nearby values for better color separation in plot
        v = KMeans(n_clusters=Q.shape[-1]).fit(v.reshape(-1, 1)).labels_
    plt.imshow(v.reshape(repout.size(), repin.size()), cmap=cmap_name, aspect='equal')
    plt.title(f"n_basis: {n_basis} - Free Params {len(u_vals):d}/{np.prod(v.shape):d}")
    plt.tight_layout()
    plt.axis('off')

def visualize_basis_ind(rep, dim_in, dim_out, title=None, param_names=False):
    """ A function to visualize the basis of equivariant maps repin>>repout
        as an image. Only use cluster=True if you know Pv will only have
        r distinct values (true for G_js<S(n) but not true for many continuous groups)."""
    # rep = (repin >> repout)
    P = rep.equivariant_projector()  # compute the equivariant basis
    Q = rep.equivariant_basis()
    v = np.random.randn(P.shape[1])  # sample random vector
    v = np.round(P @ v, decimals=4)  # project onto equivariant subspace (and round)

    # base = basis @ np.eye(basis.shape[0], dtype=basis.dtype)

    print("Ploting basis")
    n_basis = Q.shape[-1]
    n_cols = n_basis if n_basis < 4 else min(n_basis, int(n_basis/4))
    n_rows = int(n_basis / n_cols) + (1 if n_basis > n_cols and n_basis % 4 > 0 else 0)
    hf = max(dim_in, dim_out)  # W/H_solo

    W_sym = symbolic_matrix("w", dim_out, dim_in)
    w_sym = np.reshape(W_sym, (np.prod(W_sym.shape), 1))

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols*2*dim_in/hf, n_rows*2*dim_out/hf),
                            gridspec_kw={'wspace': 0.01, 'hspace': 0},
                            constrained_layout=True, dpi=160)
    basis = densify(Q)

    b_min, b_max = np.min(basis), np.max(basis)
    for i, ax in enumerate(axs.flatten()):
        # ax.axis("off")
        if i >= n_basis:
            break
        b = basis[:, i].reshape(dim_out, dim_in)
        u_vals = np.unique(np.round(b, 5))

        W_sym_a = None
        if param_names:
            # im = ax.imshow(b, aspect='equal', vmin=b_min, vmax=b_max, cmap='plasma')
            w_sym_a = ["0" if v == 0 else r"%s" % str(w) for w, v in zip(w_sym.flatten(), basis[:, i])]
            W_sym_a = np.array(w_sym_a).reshape(dim_out, dim_in)
        sns.heatmap(b, ax=ax, cmap='plasma', annot=W_sym_a, annot_kws={'fontsize': 9}, cbar=False, square=True,
                    linewidths=.01, fmt='', vmin=b_min, vmax=b_max)
        ax.set_title(f"b{i+1} {len(u_vals):d}/{np.prod(b.shape):d}", size=5)

    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize='small')
    fig.show()


def visualize_basis_sym(rep, dim_in, dim_out, title=None):
    """ A function to visualize the basis of equivariant maps repin>>repout
        as an image. Only use cluster=True if you know Pv will only have
        r distinct values (true for G_js<S(n) but not true for many continuous groups)."""
    # rep = (repin >> repout)
    P = rep.equivariant_projector()  # compute the equivariant basis
    Q = rep.equivariant_basis()
    v = np.random.randn(P.shape[1])  # sample random vector

    n_basis = Q.shape[-1]
    n_cols = n_basis if n_basis < 4 else min(n_basis, int(n_basis / 4))
    n_rows = int(n_basis / n_cols) + (1 if n_basis > n_cols and n_basis % 4 > 0 else 0)
    hf = max(dim_in, dim_out)  # W/H_solo

    W_sym = symbolic_matrix("w", dim_out, dim_in)
    w_sym = np.reshape(W_sym, (np.prod(W_sym.shape), 1))

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols * 2 * dim_in / hf, n_rows * 2 * dim_out / hf),
                            gridspec_kw={'wspace': 0.01, 'hspace': 0},
                            constrained_layout=True, dpi=160)
    basis = densify(Q)
    sym_basis = None
    for i in range(n_basis):
        w_sym_a = Matrix([0.0 if v == 0 else w for w, v in zip(w_sym.flatten(), basis[:, i])])
        if sym_basis:
            sym_basis += w_sym_a
        else:
            sym_basis = w_sym_a
    sym_basis = sum(sym_basis)
    b_min, b_max = np.min(basis), np.max(basis)
    for i, ax in enumerate(axs.flatten()):
        # ax.axis("off")
        if i >= n_basis:
            break
        b = basis[:, i].reshape(dim_out, dim_in)
        u_vals = np.unique(np.round(b, 5))

        w_sym_a = ["0" if v == 0 else r"%s" % str(w) for w, v in zip(w_sym.flatten(), basis[:, i])]
        W_sym_a = np.array(w_sym_a).reshape(dim_out, dim_in)
        # im = ax.imshow(b, aspect='equal', vmin=b_min, vmax=b_max, cmap='plasma')
        sns.heatmap(b, ax=ax, cmap='plasma', annot=W_sym_a, annot_kws={'fontsize': 9}, cbar=False, square=True,
                    linewidths=.01, fmt='', vmin=b_min, vmax=b_max)
        ax.set_title(f"b{i + 1} {len(u_vals):d}/{np.prod(b.shape):d}", size=5)

    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize='small')
    fig.show()



def visualize_group_generators(g: Group):

    fig, axs = plt.subplots(nrows=len(g.discrete_generators), ncols=1,
                            figsize=(3, 3*len(g.discrete_generators)),
                            gridspec_kw={'wspace': 0.01, 'hspace': 0},
                            constrained_layout=True)
    for ax, hi in zip(axs.flatten(), g.discrete_generators):
        ax.imshow(hi, cmap='tab20c')
        ax.axis("off")
    fig.show()


def plot_system_of_equations(A, x, b):
    import seaborn as sns
    Asym = Matrix(A)
    Arref = np.array(Asym.rref()[0].tolist(), dtype=np.float)
    Null_A = Asym.nullspace(simplify=True)
    Null_A_sym = Null_A * x.T

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    var_names = [r'%s' % str(s) for s in x.flatten()]
    eq_names = [r'$eq_{%s}$' % e for e in range(Arref.shape[-1])]
    sns.heatmap(A, ax=ax[0], cmap='inferno', xticklabels=var_names, yticklabels=eq_names, cbar=False, square=True, linewidths=.01)
    sns.heatmap(Arref, ax=ax[1], cmap='inferno', xticklabels=var_names, yticklabels=eq_names, cbar=False, square=True, linewidths=.01)

    fig.tight_layout()
    fig.show()

def visualize_equivariant_projector():
    pass