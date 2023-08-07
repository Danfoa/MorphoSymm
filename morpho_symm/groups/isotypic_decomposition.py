import itertools
from typing import Callable, Dict, List, Union

import escnn
import networkx as nx
import numpy as np
import scipy
from escnn.group.group import Group, GroupElement
from escnn.group.representation import Representation
from networkx import Graph
from scipy.linalg import block_diag

from morpho_symm.utils.algebra_utils import permutation_matrix


def is_complex_irreducible(
        G: Group, representation: Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]]
        ):
    """Check if a representation is complex irreducible.

    We check this by asserting weather non-scalar (no multiple of
    identity ) Hermitian matrix `H` exists, such that `H` commutes with all group elements' representation.
    If rho is irreducible, this function returns (True, H=I)  where I is the identity matrix.
    Otherwise, returns (False, H) where H is a non-scalar matrix that commutes with all elements' representation.
    """
    if isinstance(representation, dict):
        def rep(g):
            return representation[g]
    else:
        rep = representation

    # Compute the dimension of the representation
    n = rep(G.sample()).shape[0]

    # Run through all r,s = 1,2,...,n
    for r in range(n):
        for s in range(n):
            # Define H_rs
            H_rs = np.zeros((n, n), dtype=complex)
            if r == s:
                H_rs[r, s] = 1
            elif r > s:
                H_rs[r, s] = 1
                H_rs[s, r] = 1
            else:  # r < s
                H_rs[r, s] = 1j
                H_rs[s, r] = -1j

            # Compute H
            H = sum([rep(g).conj().T @ H_rs @ rep(g) for g in G.elements]) / G.order()

            # If H is not a scalar matrix, then it is a matrix that commutes with all group actions.
            if not np.allclose(H[0, 0] * np.eye(H.shape[0]), H):
                return False, H
    # No Hermitian matrix was found to commute with all group actions. This is an irreducible rep
    return True, np.eye(n)


def decompose_representation(
        G: Group, representation: Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]]
        ):
    """Find the Hermitian matrix `Q` that block-diagonalizes the representation `rep` of group `G`.

    Such that `Q @ rep[g] @ Q^H = block_diag(rep_1[g], ..., rep_m[g])` for all `g` in `G`.
    """
    eps = 1e-12
    if isinstance(representation, dict):
        def rep(g):
            return representation[g]
    else:
        rep = representation
    # Compute the dimension of the representation
    n = rep(G.sample()).shape[0]

    for g in G.elements:  # Ensure the representation is unitary/orthogonal
        error = np.abs((rep(g) @ rep(g).conj().T) - np.eye(n))
        assert np.allclose(error, 0), f"Rep {rep} is not unitary: rep(g)@rep(g)^H=\n{(rep(g) @ rep(g).conj().T)}"

    # Find Hermitian matrix non-scalar `H` that commutes with all group actions
    is_irred, H = is_complex_irreducible(G, rep)
    if is_irred:
        return [rep], np.eye(n)

    # Eigen-decomposition of matrix `H = P·A·P^-1` reveals the G-invariant subspaces/eigenspaces of the representations.
    eivals, eigvects = np.linalg.eigh(H, UPLO='L')
    P = eigvects.conj().T
    assert np.allclose(P.conj().T @ np.diag(eivals) @ P, H)

    # Eigendcomposition is not guaranteed to block_diagonalize the representation. An additional permutation of the
    # rows and columns od the representation might be needed to produce a Jordan block canonical form.
    # First: We want to identify the diagonal blocks. To find them we use the trick of thinking of the representation
    # as an adjacency matrix of a graph. The non-zero entries of the adjacency matrix are the edges of the graph.
    edges = set()
    decomposed_reps = {}
    for g in G.elements:
        diag_rep = P @ rep(g) @ P.conj().T  # Obtain block diagonal representation
        diag_rep[np.abs(diag_rep) < eps] = 0  # Remove rounding errors.
        non_zero_idx = np.nonzero(diag_rep)
        edges.update([(x_idx, y_idx) for x_idx, y_idx in zip(*non_zero_idx)])
        decomposed_reps[g] = diag_rep

    # Each connected component of the graph is equivalent to the rows and columns determining a block in the diagonal
    graph = Graph()
    graph.add_edges_from(set(edges))
    connected_components = [sorted(list(comp)) for comp in nx.connected_components(graph)]
    connected_components = sorted(connected_components, key=lambda x: (len(x), min(x)))   # Impose a canonical order
    # If connected components are not adjacent dimensions, say subrep_1_dims = [0,2] and subrep_2_dims = [1,3] then
    # We permute them to get a jordan block canonical form. I.e. subrep_1_dims = [0,1] and subrep_2_dims = [2,3].
    oneline_notation = list(itertools.chain.from_iterable([list(comp) for comp in connected_components]))
    PJ = permutation_matrix(oneline_notation=oneline_notation)
    # After permuting the dimensions, we can assume the components are ordered in dimension
    ordered_connected_components = []
    idx = 0
    for comp in connected_components:
        ordered_connected_components.append(tuple(range(idx, idx + len(comp))))
        idx += len(comp)
    connected_components = ordered_connected_components

    # The output of connected components is the set of nodes/row-indices of the rep.
    subreps = [{} for _ in connected_components]
    for g in G.elements:
        for comp_id, comp in enumerate(connected_components):
            block_start, block_end = comp[0], comp[-1] + 1
            # Transform the decomposed representation into the Jordan Cannonical Form (jcf)
            jcf_rep = (PJ @ decomposed_reps[g] @ PJ.T)
            # Check Jordan Cannonical Form TODO: Extract this to a utils. function
            above_block = jcf_rep[0:block_start, block_start:block_end]
            below_block = jcf_rep[block_end:, block_start:block_end]
            left_block = jcf_rep[block_start:block_end, 0:block_start]
            right_block = jcf_rep[block_start:block_end, block_end:]

            assert np.allclose(above_block, 0) or above_block.size == 0, "Non zero elements above block"
            assert np.allclose(below_block, 0) or below_block.size == 0, "Non zero elements below block"
            assert np.allclose(left_block, 0) or left_block.size == 0, "Non zero elements left of block"
            assert np.allclose(right_block, 0) or right_block.size == 0, "Non zero elements right of block"
            sub_g = jcf_rep[block_start:block_end, block_start:block_end]
            subreps[comp_id][g] = sub_g

    # Decomposition to Jordan Canonical form is accomplished by (PJ @ P) @ rep @ (PJ @ P)^-1
    Q = PJ @ P

    # Test decomposition.
    for g in G.elements:
        jcf_rep = block_diag(*[subrep[g] for subrep in subreps])
        error = np.abs(jcf_rep - (Q @ rep(g) @ Q.conj().T))
        assert np.allclose(error, 0), f"Q @ rep[g] @ Q^-1 != block_diag[{[f'rep{i},' for i in range(len(subreps))]}]"

    return subreps, Q


def cplx_isotypic_decomposition(
        G: Group, representation: Callable[[GroupElement], np.ndarray]
        ):
    """Perform the isotypic decomposition of unitary representation, decomposing the rep into complex irreps.

    Args:
        G (Group): Symmetry group of the representation.
        representation (Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]]): dict/mapping from
        group elements to matrices, or a function that takes a group element and returns a matrix.

    Returns:
        sorted_irreps (List[Dict[GroupElement, np.ndarray]]): List of complex irreducible representations, sorted in
        ascending order of dimension.
        Q (ndarray): Hermitian matrix such that Q @ rep[g] @ Q^-1 is block diagonal, with blocks `sorted_irreps`.:

    """
    if isinstance(representation, dict):
        def rep(g):
            return representation[g]
    else:
        rep = representation

    n = rep(G.sample()).shape[0]
    subreps, Q_internal = decompose_representation(G, rep)

    found_irreps = []
    Qs = []

    # Check if each subrepresentation can be further decomposed.
    for subrep in subreps:
        n_sub = subrep(G.sample()).shape[0]   # Dimension of sub representation
        is_irred, _ = is_complex_irreducible(G, subrep)
        if is_irred:
            found_irreps.append(subrep)
            Qs.append(np.eye(n_sub))
        else:
            # Find Q_sub such that Q_sub @ subrep[g] @ Q_sub^-1 is block diagonal, with blocks `sub_subrep`
            sub_subreps, Q_sub = cplx_isotypic_decomposition(G, subrep)
            found_irreps += sub_subreps
            Qs.append(Q_sub)

    # Sort irreps by dimension.
    P, sorted_irreps = sorted_jordan_cann_form(G, found_irreps)

    # If subreps were decomposable, then these get further decomposed with an additional Hermitian matrix such that:
    # Q @ rep[g] @ Q^-1 = block_diag[irreps] | Q = (Q_external @ Q_internal)
    Q_external = block_diag(*Qs)
    Q = P @ Q_external @ Q_internal

    # Test isotypic decomposition.
    assert np.allclose(Q @ Q.conj().T, np.eye(n)), "Q is not unitary."
    for g in G.elements:
        g_iso = block_diag(*[irrep[g] if isinstance(irrep, dict) else irrep(g) for irrep in sorted_irreps])
        error = np.abs(g_iso - (Q @ rep(g) @ Q.conj().T))
        assert np.allclose(error, 0), f"Q @ rep[g] @ Q^-1 != block_diag[irreps[g]], for g={g}. Error \n:{error}"

    return sorted_irreps, Q


def sorted_jordan_cann_form(G: Group, reps: List[Callable[[GroupElement], np.ndarray]]):
    """Sorts a list of representations in ascending order of dimension, and returns a permutation matrix P such that.

    Args:
        G (Group): Symmetry group of the representation.
        reps: List of representations to sort by dimension.

    Returns:
        P (ndarray): Permutation matrix sorting the input reps.
        reps (List[Union[Dict[GroupElement, np.ndarray], Representation]]): Sorted list of representations.
    """
    reps_idx = range(len(reps))
    reps_size = [rep(G.sample()).shape[0] for rep in reps]
    sort_order = sorted(reps_idx, key=lambda idx: reps_size[idx])
    if sort_order == list(reps_idx):
        return np.eye(sum(reps_size)), reps
    irrep_dim_start = np.cumsum([0] + reps_size[:-1])
    oneline_perm = []
    for idx in sort_order:
        rep_size = reps_size[idx]
        oneline_perm += list(range(irrep_dim_start[idx], irrep_dim_start[idx] + rep_size))
    P = permutation_matrix(oneline_perm)

    return P, [reps[idx] for idx in sort_order]


def compute_character_table(G: Group, reps: List[Union[Dict[GroupElement, np.ndarray], Representation]]):
    """Computes the character table of a group for a given set of representations."""
    n_reps = len(reps)
    table = np.zeros((n_reps, G.order()), dtype=complex)
    for i, rep in enumerate(reps):
        for j, g in enumerate(G.elements):
            table[i, j] = rep.character(g) if isinstance(rep, Representation) else np.trace(rep(g))
    return table


def map_character_tables(in_table: np.ndarray, reference_table: np.ndarray):
    """Find a representation of a group in the set of irreducible representations."""
    n_in_reps = in_table.shape[0]
    out_ids, multiplicities = [], []
    for in_id in range(n_in_reps):
        character_orbit = in_table[in_id, :]
        orbit_error = np.isclose(np.abs(reference_table - character_orbit), 0)
        match_idx = np.argwhere(np.all(orbit_error, axis=1)).flatten()
        multiplicity = len(match_idx)
        out_ids.append(match_idx), multiplicities.append(multiplicity)
    return multiplicities, out_ids


def escnn_representation_form_mapping(
        G: Group, representation: Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]]
        ):
    """Get a ESCNN representation isntance from a mapping from group elements to unitary matrices.

    Args:
        G (Group): Symmetry group of the representation.
        representation (Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]]): Mapping from
            group elements to unitary matrices.

    Returns:
        representation (Representation): ESCNN representation instance.
    """

    def rep(g):
        return representation[g] if isinstance(representation, dict) else representation(g)

    rep(G.sample()).shape[0]  # Size of the representation
    # Find Q such that `iso_cplx(g) = Q @ rep(g) @ Q^-1` is block diagonal with blocks being complex irreps.
    cplx_irreps, Q = cplx_isotypic_decomposition(G, rep)
    # Get the size and location of each cplx irrep in `iso_cplx(g)`
    cplx_irreps_size = [irrep(G.sample()).shape[0] for irrep in cplx_irreps]
    irrep_dim_start = np.cumsum([0] + cplx_irreps_size[:-1])
    # Compute the character table of the found complex irreps and of all complex irreps of G
    irreps_char_table = compute_character_table(G, cplx_irreps)

    # We need to identify which real ESCNN irreps are present in rep(g).
    # First, we decompose the Group's ESCNN real irreps into complex irreps.
    escnn_cplx_irreps_data = {}
    for re_irrep in G.irreps():
        # Find Q_sub s.t. `block_diag([cplx_irrep_i1(g), cplx_irrep_i2(g)...]) = Q @ re_irrep_i(g) @ Q^-1`
        irreps, Q_sub = cplx_isotypic_decomposition(G, re_irrep)
        char_table = compute_character_table(G, irreps)
        escnn_cplx_irreps_data[re_irrep] = dict(subreps=irreps, Q=Q_sub, char_table=char_table)

    # Then, we find which of the Group complex irreps are present in the input representation, and determine
    # each group complex irrep multiplicity. As the complex irreps forming a real irrep can be spread over the
    # dimensions of the input rep, we find a permutation matrix P such that all complex irreps associated with a real
    # irrep are contiguous in dimensions.trifinger
    oneline_perm, Q_isore2isoimg = [], []
    escnn_real_irreps = []
    for escnn_irrep, data in escnn_cplx_irreps_data.items():
        # Match complex irreps by their character tables.
        multiplicities, irrep_locs = map_character_tables(data["char_table"], irreps_char_table)
        subreps_start_dims = [irrep_dim_start[i] for i in irrep_locs]  # Identify start of blocks in `rep(g)`
        data.update(multiplicities=multiplicities, subrep_start_dims=subreps_start_dims)
        assert np.unique(multiplicities).size == 1, "Multiplicities error"
        multiplicity = multiplicities[0]
        for m in range(multiplicity):
            Q_isore2isoimg.append(data['Q'])  # Add transformation from Real irrep to complex irrep
            escnn_real_irreps.append(escnn_irrep)  # Add escnn irrep to the list for instanciation
            for subrep, rep_start_dims in zip(data['subreps'], subreps_start_dims):
                rep_size = subrep[G.sample()].shape[0] if isinstance(subrep, dict) else subrep.size
                oneline_perm += list(range(rep_start_dims[m], rep_start_dims[m] + rep_size))
    # As the complex irreps forming a real irrep can be spread over the dimensions of the input rep, we find a
    # permutation matrix P such that all complex irreps of a real irrep are contiguous in dimensions / in the same block
    P = permutation_matrix(oneline_notation=oneline_perm)
    # Then we use the known transformations `Q_sub` for each real irrep, to create a mapping from cplx to real irreps.
    # s.t. `iso_re(g) = Q_iso_cplx2iso_re @ block_diag([cplx_irrep_11(g),...,cplx_irrep_ij(g)]) @ Q_iso_cplx2iso_re^-1`
    Q_iso_cplx2iso_re = block_diag(*[Q_sub.conj().T for Q_sub in Q_isore2isoimg])

    # Assert the matrix `P` and `Q_iso_cplx2iso_re` turn complex irreps into real irreps
    # `iso_re(g) = (Q_iso_cplx2iso_re @ P) @ iso_cplx(g) @ (Q_iso_cplx2iso_re @ P)^-1`,
    for g in G.elements:
        iso_re_g = block_diag(*[irrep(g) for irrep in escnn_real_irreps])
        iso_cplx_g = block_diag(*[cplx_irrep(g) for cplx_irrep in cplx_irreps])
        rec_iso_re_g = (Q_iso_cplx2iso_re @ P) @ iso_cplx_g @ (Q_iso_cplx2iso_re @ P).conj().T
        error = np.abs(iso_re_g - rec_iso_re_g)
        assert np.isclose(error, 0).all(), "Error in the conversion of Real irreps to Complex irreps"

    # Now we have an orthogonal transformation between the input `rep` and `iso_re`.
    #                        |     iso_cplx(g)     |
    # (Q_iso_cplx2iso_re @ P @ Q) @ rep(g) @ (Q^-1 @ P^-1 @ Q_iso_cplx2iso_re^-1) = Q_re @ rep(g) @ Q_re^-1 = iso_re(g)
    Q_re = Q_iso_cplx2iso_re @ P @ Q

    assert np.allclose(Q_re @ Q_re.conj().T, np.eye(Q_re.shape[0])), "Q_re is not an orthogonal transformation"
    if np.allclose(np.imag(Q_re), 0):
        Q_re = np.real(Q_re)  # Remove numerical noise and ensure rep(g) is of dtype: float instead of cfloat

    # Then we have that `Q_re^-1 @ iso_re(g) @ Q_re = rep(g)`
    reconstructed_rep = Representation(G, name="reconstructed", irreps=[irrep.id for irrep in escnn_real_irreps],
                                       change_of_basis=Q_re.conj().T)

    # Test ESCNN reconstruction
    for g in G.elements:
        g_true, g_rec = rep(g), reconstructed_rep(g)
        error = np.abs(g_true - g_rec)
        error[error < 1e-10] = 0
        assert np.allclose(error, 0), f"Reconstructed rep do not match input rep. g={g}, error:\n{error}"
        assert np.allclose(np.imag(g_rec), 0), f"Reconstructed rep not real for g={g}: \n{g_rec}"

    return reconstructed_rep


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    G = escnn.group.groups.klein4_group()
    # G = escnn.group.groups.CyclicGroup(3)
    # G = escnn.group.groups.DihedralGroup(3)
    test_rep = G.regular_representation + G.regular_representation

    rec_rep = escnn_representation_form_mapping(G, test_rep)

    print("Fucking yeah !!!")
