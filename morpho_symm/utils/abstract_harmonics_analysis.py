import functools
from collections import OrderedDict

import numpy as np
from escnn.group import Representation, directsum

from morpho_symm.utils.algebra_utils import permutation_matrix


def decom_signal_into_isotypic_components(signal: np.ndarray, rep: Representation):
    """Decompose a signal into its isotypic components.

    This function takes a signal and a group representation and returns the signal decomposed into its isotypic
    components, in the isotypic basis and the original basis.

    Args:
        signal: (... , rep.size) array of signals to be decomposed.
        rep: (Representation) Representation of the vector space in which the signal lives in.

    Returns:
        iso_comp_signals: (OrderedDict) Dictionary of isotypic components of the signal in the isotypic basis.
        iso_comp_signals_orig_basis: (OrderedDict) Dictionary of isotypic components of the signal in the original
        basis.
    """
    rep_iso = isotypic_decomp_representation(rep)
    Q_iso2orig = rep_iso.change_of_basis  # Change of basis from isotypic basis to original basis
    Q_orig2iso = rep_iso.change_of_basis_inv  # Change of basis from original basis to isotypic basis
    assert signal.shape[-1] == rep.size, f"Expected signal shape to be (..., {rep.size}) got {signal.shape}"

    signal_iso = np.einsum("...ij,...j->...i", Q_orig2iso, signal)

    isotypic_representations = rep_iso.attributes["isotypic_reps"]

    # Compute the dimensions of each isotypic subspace
    cum_dim = 0
    iso_comp_dims = {}
    for irrep_id, iso_rep in isotypic_representations.items():
        iso_space_dims = range(cum_dim, cum_dim + iso_rep.size)
        iso_comp_dims[irrep_id] = iso_space_dims
        cum_dim += iso_rep.size

    # Separate the signal into isotypic components, by masking the signal outside of each isotypic subspace
    iso_comp_signals = OrderedDict()
    for irrep_id, _ in isotypic_representations.items():
        iso_dims = iso_comp_dims[irrep_id]
        iso_comp_signals[irrep_id] = signal_iso[..., iso_dims]

    iso_comp_signals_orig_basis = OrderedDict()
    # Compute the signals of each isotypic component in the original basis
    for irrep_id, _ in isotypic_representations.items():
        iso_dims = iso_comp_dims[irrep_id]
        Q_isocomp2orig = Q_iso2orig[:, iso_dims]  # Change of basis from isotypic component basis to original basis
        iso_comp_signals_orig_basis[irrep_id] = np.einsum(
            "...ij,...j->...i", Q_isocomp2orig, iso_comp_signals[irrep_id]
        )

    # Check that the sum of the isotypic components is equal to the original signal
    rec_signal = np.sum([iso_comp_signals_orig_basis[irrep_id] for irrep_id in isotypic_representations.keys()], axis=0)
    assert np.allclose(rec_signal, signal), (
        f"Reconstructed signal is not equal to original signal. Error: {np.linalg.norm(rec_signal - signal)}"
    )

    return iso_comp_signals, iso_comp_signals_orig_basis


def isotypic_decomp_representation(rep: Representation) -> Representation:
    """Returns a representation in a "symmetry enabled basis" (a.k.a Isotypic Basis).

    Takes a representation with an arbitrary basis (i.e., arbitrary change of basis and an arbitrary order of
    irreducible representations in the escnn Representation) and returns a new representation in which the basis
    is changed to a "symmetry enabled basis" (a.k.a Isotypic Basis). That is a representation in which the
    vector space is decomposed into a direct sum of Isotypic Subspaces. Each Isotypic Subspace is a subspace of the
    original vector space with a subspace representation composed of multiplicities of a single irreducible
    representation. In oder words, each Isotypic Subspace is a subspace with a subgroup of symmetries of the original
    vector space's symmetry group.

    Args:
        rep (Representation): Input representation in any arbitrary basis.

    Returns: A `Representation` with a change of basis exposing an Isotypic Basis (a.k.a symmetry enabled basis).
        The instance of the representation contains an additional attribute `isotypic_subspaces` which is an
        `OrderedDict` of representations per each isotypic subspace. The keys are the active irreps' ids associated
        with each Isotypic subspace.
    """
    symm_group = rep.group
    potential_irreps = rep.group.irreps()
    isotypic_subspaces_indices = {irrep.id: [] for irrep in potential_irreps}

    for pot_irrep in potential_irreps:
        cur_dim = 0
        for rep_irrep_id in rep.irreps:
            rep_irrep = symm_group.irrep(*rep_irrep_id)
            if rep_irrep == pot_irrep:
                isotypic_subspaces_indices[rep_irrep_id].append(list(range(cur_dim, cur_dim + rep_irrep.size)))
            cur_dim += rep_irrep.size

    # Remove inactive Isotypic Spaces
    for irrep in potential_irreps:
        if len(isotypic_subspaces_indices[irrep.id]) == 0:
            del isotypic_subspaces_indices[irrep.id]

    # Each Isotypic Space will be indexed by the irrep it is associated with.
    active_isotypic_reps = {}
    for irrep_id, indices in isotypic_subspaces_indices.items():
        irrep = symm_group.irrep(*irrep_id)
        multiplicities = len(indices)
        active_isotypic_reps[irrep_id] = Representation(
            group=rep.group,
            irreps=[irrep_id] * multiplicities,
            name=f"IsoSubspace {irrep_id}",
            change_of_basis=np.identity(irrep.size * multiplicities),
            supported_nonlinearities=irrep.supported_nonlinearities,
        )

    # Impose canonical order on the Isotypic Subspaces.
    # If the trivial representation is active it will be the first Isotypic Subspace.
    # Then sort by dimension of the space from smallest to largest.
    ordered_isotypic_reps = OrderedDict(sorted(active_isotypic_reps.items(), key=lambda item: item[1].size))
    if symm_group.trivial_representation.id in ordered_isotypic_reps.keys():
        ordered_isotypic_reps.move_to_end(symm_group.trivial_representation.id, last=False)

    # Required permutation to change the order of the irreps. So we obtain irreps of the same type consecutively.
    oneline_permutation = []
    for irrep_id, iso_rep in ordered_isotypic_reps.items():
        idx = isotypic_subspaces_indices[irrep_id]
        oneline_permutation.extend(idx)
    oneline_permutation = np.concatenate(oneline_permutation)
    P_in2iso = permutation_matrix(oneline_permutation)

    Q_iso = rep.change_of_basis @ P_in2iso.T
    rep_iso_basis = directsum(list(ordered_isotypic_reps.values()), name=rep.name + "-Iso", change_of_basis=Q_iso)

    iso_supported_nonlinearities = [iso_rep.supported_nonlinearities for iso_rep in ordered_isotypic_reps.values()]
    rep_iso_basis.supported_nonlinearities = functools.reduce(set.intersection, iso_supported_nonlinearities)
    rep_iso_basis.attributes["isotypic_reps"] = ordered_isotypic_reps

    return rep_iso_basis


def isotypic_basis(representation: Representation, multiplicity: int = 1, prefix=""):
    rep_iso = isotypic_decomp_representation(representation)

    iso_reps = OrderedDict()
    iso_range = OrderedDict()

    start_dim = 0
    for iso_irrep_id, reg_rep_iso in rep_iso.attributes["isotypic_reps"].items():
        iso_reps[iso_irrep_id] = directsum([reg_rep_iso] * multiplicity, name=f"{prefix}_IsoSpace{iso_irrep_id}")
        iso_range[iso_irrep_id] = range(start_dim, start_dim + iso_reps[iso_irrep_id].size)
        start_dim += iso_reps[iso_irrep_id].size

    assert rep_iso.size * multiplicity == sum([iso_rep.size for iso_rep in iso_reps.values()])

    return iso_reps, iso_range  # Dict[key:id_space -> value: rep_iso_space]
