import itertools
from typing import Dict

import escnn.group
import numpy as np
from escnn.group import GroupElement, Group, CyclicGroup, DihedralGroup, DirectProductGroup
from escnn.group.representation import Representation

from utils.simultaneous_block_diagonalization import escnn_representation_form_mapping


def generate_cyclic_rep(G: CyclicGroup, rep):
    h = G.generators[0]
    # Check the given matrix representations comply with group axioms
    assert not np.allclose(rep[h], rep[G.identity]), f"Invalid generator: h=e"
    assert np.allclose(np.linalg.matrix_power(rep[h], G.order()), rep[G.identity]), f"Invalid rotation generator h_ref^{G.order()} != I"

    curr_g = h
    while len(rep) < G.order():  # Use generator to obtain all elements and element reps in group
        g = curr_g @ h
        rep[g] = rep[curr_g] @ rep[h]
        curr_g = g

    return rep


def generate_dihedral_rep(G: DihedralGroup, rep):
    h_rot, h_ref = G.generators
    # Check the given matrix representations comply with group axioms
    assert not np.allclose(rep[h_ref], rep[G.identity]), f"Invalid reflection generator: h_ref=e"
    assert not np.allclose(rep[h_rot], rep[G.identity]), f"Invalid rotation generator: h_rot=e"
    assert np.allclose(rep[h_ref] @ rep[h_ref], rep[G.identity]), f"Invalid reflection generator `h_ref @ h_ref != I`"
    assert np.allclose(np.linalg.matrix_power(rep[h_rot], G.order()//2), rep[G.identity]), f"Invalid rotation generator h_ref^{G.order} != I"

    curr_g, curr_ref_g = h_rot, h_ref @ h_rot
    rep[curr_ref_g] = rep[h_ref] @ rep[h_rot]
    while len(rep) < G.order():  # Use generator to obtain all elements and element reps in group
        g = curr_g @ h_rot
        gr = curr_ref_g @ h_rot
        rep[g] = rep[curr_g] @ rep[h_rot]
        rep[gr] = rep[curr_ref_g] @ rep[h_rot]
        curr_g, curr_ref_g = g, gr

    return rep


def generate_direct_product_rep(G: DirectProductGroup, rep1, rep2):
    rep = {}
    for h1, h2 in itertools.product(rep1.keys(), rep2.keys()):
        g = G.pair_elements(h1, h2)
        rep[g] = rep1[h1] @ rep2[h2]
    return rep


def group_rep_from_gens(G: Group, rep: Dict[GroupElement, np.ndarray]) -> Representation:
    if G.identity not in rep:
        rep[G.identity] = np.eye(list(rep.values())[0].shape[0])

    if isinstance(G, CyclicGroup):
        rep = generate_cyclic_rep(G, rep)
    elif isinstance(G, DihedralGroup):
        rep = generate_dihedral_rep(G, rep)
    elif isinstance(G, DirectProductGroup):
        # Extract the generators of first and second group, generate the groups independently and then combine them
        H1, H2 = zip(*[G.split_element(h) for h in G.generators])
        rep_G1 = {G.G1.identity: rep[G.pair_elements(G.G1.identity, G.G2.identity)]}
        rep_G2 = {G.G2.identity: rep[G.pair_elements(G.G1.identity, G.G2.identity)]}
        rep_G1.update({h1: rep[G.inclusion1(h1)] for h1 in H1})
        rep_G2.update({h2: rep[G.inclusion2(h2)] for h2 in H2})

        # generate each subgroup representation
        group_rep_from_gens(G.G1, rep_G1)
        group_rep_from_gens(G.G2, rep_G2)

        # Do direct product of the generated subgroups reps.
        rep = generate_direct_product_rep(G, rep_G1, rep_G2)
    else:
        raise NotImplementedError(f"Group {G} not implemented yet.")

    # Convert Dict[GroupElement, np.ndarray] to escnn `Representation`
    rep_escnn = escnn_representation_form_mapping(G, rep)

    return rep_escnn



