import numpy as np
import matplotlib.pyplot as plt
from emlp import Group
from emlp.reps.linear_operators import densify
from sklearn.cluster import KMeans


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

    cmap_name = 'tab20' if len(u_vals) <= 20 else 'hsv'

    if cluster:  # cluster nearby values for better color separation in plot
        v = KMeans(n_clusters=Q.shape[-1]).fit(v.reshape(-1, 1)).labels_
    plt.imshow(v.reshape(repout.size(), repin.size()), cmap=cmap_name, aspect='equal')
    plt.title(f"n_basis: {n_basis} - Free Params {len(u_vals):d}/{np.prod(v.shape):d}")
    plt.tight_layout()
    plt.axis('off')

def visualize_basis_ind(repin, repout, cluster=True):
    """ A function to visualize the basis of equivariant maps repin>>repout
        as an image. Only use cluster=True if you know Pv will only have
        r distinct values (true for G_js<S(n) but not true for many continuous groups)."""
    rep = (repin >> repout)
    P = rep.equivariant_projector()  # compute the equivariant basis
    Q = rep.equivariant_basis()
    v = np.random.randn(P.shape[1])  # sample random vector
    v = np.round(P @ v, decimals=4)  # project onto equivariant subspace (and round)

    # base = Q @ np.eye(Q.shape[0], dtype=Q.dtype)

    print("Ploting basis")
    n_basis = Q.shape[-1]
    n_cols = n_basis if n_basis < 4 else min(n_basis, int(n_basis/4))
    n_rows = int(n_basis / n_cols) + (1 if n_basis > n_cols and n_basis % 4 > 0 else 0)
    hf = max(repin.size(), repout.size())  # W/H

    proj = np.array(densify(P))
    plt.figure()
    plt.title("Equivariant Projector")
    plt.imshow(densify(P), cmap='tab20c')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols*2*repin.size()/hf, n_rows*2*repout.size()/hf),
                            gridspec_kw={'wspace': 0.01, 'hspace': 0},
                            constrained_layout=True, dpi=120)

    basis = densify(Q)
    b_min, b_max = np.min(basis), np.max(basis)
    for i, ax in enumerate(axs.flatten()):
        # ax.axis("off")
        if i >= n_basis:
            break
        b = basis[:, i].reshape(repout.size(), repin.size())
        u_vals = np.unique(np.round(b, 5))
        im = ax.imshow(b, aspect='equal', vmin=b_min, vmax=b_max, cmap='plasma')
        ax.set_title(f"b{i+1} {len(u_vals):d}/{np.prod(b.shape):d}", size=5)

    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
    plt.show()


def visualize_group_generators(g: Group):

    fig, axs = plt.subplots(nrows=len(g.discrete_generators), ncols=1,
                            figsize=(3, 3*len(g.discrete_generators)),
                            gridspec_kw={'wspace': 0.01, 'hspace': 0},
                            constrained_layout=True)
    for ax, hi in zip(axs.flatten(), g.discrete_generators):
        ax.imshow(hi, cmap='tab20c')
        ax.axis("off")
    plt.show()


def visualize_equivariant_projector():
    pass