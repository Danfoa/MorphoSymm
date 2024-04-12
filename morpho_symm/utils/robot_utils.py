#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez
# @email   : daniels.ordonez@gmail.com

import re
from pathlib import Path
from typing import Optional

import escnn
import escnn.group
import numpy as np
from escnn.group import CyclicGroup, DihedralGroup, DirectProductGroup, Group, Representation
from omegaconf import DictConfig, OmegaConf

import morpho_symm
from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper
from morpho_symm.utils.algebra_utils import gen_permutation_matrix
from morpho_symm.utils.rep_theory_utils import escnn_representation_form_mapping, group_rep_from_gens
from morpho_symm.utils.pybullet_visual_utils import (
    change_robot_appearance,
    configure_bullet_simulation,
    listen_update_robot_sliders,
    setup_debug_sliders,
)


def get_escnn_group(cfg: DictConfig):
    """Get the ESCNN group object from the group label in the config file."""
    group_label = cfg.group_label
    label_pattern = r'([A-Za-z]+)(\d+)'
    assert cfg.group_label is not None, f'Group label unspecified. Not clear which symmetry group {cfg.name} has'
    match = re.match(label_pattern, group_label)
    if match:
        group_class = match.group(1)
        order = int(match.group(2))
    else:
        raise AttributeError(f'Group label {group_label} is not a known group label (Dn: Dihedral, Cn: Cyclic) order n')

    group_axis = np.array([0, 0, 1])
    subgroup_id = np.zeros_like(group_axis, dtype=bool).astype(object)
    if group_class.lower() == 'd':  # Dihedral
        # Define the symmetry space using presets from ESCNN
        # subgroup_id[group_axis == 1] = order
        symmetry_space = escnn.gspaces.dihedralOnR3(n=order // 2, axis=0.0)
    elif group_class.lower() == 'c':  # Cyclic
        assert order >= 2, f'Order of cyclic group must be greater than 2, got {order}'
        subgroup_id[group_axis == 1] = order
        symmetry_space = escnn.gspaces.GSpace3D(tuple(subgroup_id))
    elif group_class.lower() == "k":  # Klein four group
        is_planar_subgroup = True
        symmetry_space = escnn.gspaces.GSpace3D(sg_id=(is_planar_subgroup, False, 2))
    elif group_class.lower() == "dh":  # Dn x C2. Group of regular rectangular cuboid
        symmetry_space = escnn.gspaces.GSpace3D(sg_id=(True, True, order))
    else:
        raise NotImplementedError(f"We have to implement the case of {group_label}")

    return symmetry_space


def load_symmetric_system(
        robot_cfg: Optional[DictConfig] = None,
        robot_name: Optional[str] = None,
        debug=False
        ) -> [PinBulletWrapper, escnn.group.Group]:
    """Utility function to get the symmetry group and representations of a robotic system defined in config.

    This function loads the robot into pinocchio, and generate the symmetry group representations for the following
    spaces:
        1. The joint-space (Q_js), known as the space of generalized position coordinates.
        2. The joint-space tangent space (TqQ_js), known as the space of generalized velocities.
        3. The Euclidean space (Ed) in which the dynamical system evolves in.

    Args:
        robot_name: (str): (Optional) name of the robot configuration file in `cfg/robot/` to load.
        robot_cfg (DictConfig): (Optional) configuration parameters of the robot. Check `cfg/robot/`
        debug (bool): if true we load the robot into an interactive simulation session to visually inspect URDF

    Returns:
        robot (PinBulletWrapper): instance with the robot loaded in pinocchio and ready to be loaded in pyBullet
        G (escnn.group.Group): Instance of the symmetry Group of the robot. The representations for Q_js, TqQ_js and
        Ed are
        added to the list of representations of the group.
    """
    assert robot_cfg is not None or robot_name is not None, \
        "Either a robot configuration file or a robot name must be provided."
    if robot_cfg is None:
        # Only robot name is provided. Load the robot configuration file using compose API from hydra.
        # This allows to load the parent configuration files automatically.
        path_cfg = Path(morpho_symm.__file__).parent / 'cfg' / 'robot'
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(path_cfg), version_base='1.3'):
            robot_cfg = compose(config_name=robot_name)

    robot_name = str.lower(robot_cfg.name)
    # We allow symbolic expressions (e.g. `np.pi/2`) in the `q_zero` and `init_q`.
    q_zero = np.array([eval(str(s)) for s in robot_cfg.q_zero], dtype=float) if robot_cfg.q_zero is not None else None
    init_q = np.array([eval(str(s)) for s in robot_cfg.init_q], dtype=float) if robot_cfg.init_q is not None else None
    robot = PinBulletWrapper(robot_name=robot_name, init_q=init_q, q_zero=q_zero,
                             hip_height=robot_cfg.hip_height, endeff_names=robot_cfg.endeff_names)

    if debug:
        pb = configure_bullet_simulation(gui=True, debug=debug)
        robot.configure_bullet_simulation(pb, world=None)
        change_robot_appearance(pb, robot)
        setup_debug_sliders(pb, robot)
        listen_update_robot_sliders(pb, robot)

    symmetry_space = get_escnn_group(robot_cfg)

    G = symmetry_space.fibergroup

    # Select the field for the representations.
    rep_field = float if robot_cfg.rep_fields.lower() != 'complex' else complex

    if 'permutation_Q_js' not in robot_cfg:
        raise AttributeError(f"Configuration file for {robot_name} must define the field `permutation_Q_js`, "
                             f"describing the joint space permutation per each non-trivial group's generator.")
    if 'permutation_TqQ_js' not in robot_cfg:
        raise AttributeError(f"Configuration file for {robot_name} must define the field `permutation_TqQ_js`, "
                             f"describing the tangent joint-space permutation per each non-trivial group's generator.")

    reps_in_cfg = [k.split('permutation_')[1] for k in robot_cfg if "permutation" in k]
    for rep_name in reps_in_cfg:
        perm_list = list(robot_cfg[f'permutation_{rep_name}'])
        rep_dim = len(perm_list[0])
        reflex_list = list(robot_cfg[f'reflection_{rep_name}'])
        assert len(perm_list) == len(reflex_list), \
            f"Found different number of permutations and reflections for {rep_name}"
        assert len(perm_list) >= len(G.generators), \
            f"Found {len(perm_list)} element reps for {rep_name}, Expected {len(G.generators)} generators for {G}"
        # Generate ESCNN representation of generators
        gen_rep = {}
        for h, perm, refx in zip(G.generators, perm_list, reflex_list):
            assert len(perm) == len(refx) == rep_dim
            refx = np.array(refx, dtype=rep_field)
            gen_rep[h] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)
        # Generate the entire group
        rep = group_rep_from_gens(G, rep_H=gen_rep)
        rep.name = rep_name
        G.representations.update({rep_name: rep})

    rep_Q_js = G.representations['Q_js']
    rep_TqQ_js = G.representations.get('TqQ_js', None)
    rep_TqQ_js = rep_Q_js if rep_TqQ_js is None else rep_TqQ_js
    dimQ_js, dimTqQ_js = robot.nq - 7, robot.nv - 6
    assert dimQ_js == rep_Q_js.size
    assert dimTqQ_js == rep_TqQ_js.size

    # Create the representation of isometries on the Euclidean Space in d dimensions.
    rep_R3, rep_E3, rep_R3pseudo, rep_E3pseudo = generate_euclidean_space_representations(G)  # This adds `O3` and `E3` representations to the group.

    # Define the representation of the rotation matrix R that transforms the base orientation.
    rep_rot_flat = {}
    for h in G.elements:
        rep_rot_flat[h] = np.kron(rep_R3(h), rep_R3(~h).T)
    rep_rot_flat = escnn_representation_form_mapping(G, rep_rot_flat)
    rep_rot_flat.name = "SO3_flat"

    # Add representations to the group.
    G.representations.update(Q_js=rep_Q_js,
                             TqQ_js=rep_TqQ_js,
                             R3=rep_R3,
                             E3=rep_E3,
                             R3_pseudo=rep_R3pseudo,
                             E3_pseudo=rep_E3pseudo,
                             SO3_flat=rep_rot_flat)

    return robot, G


def generate_euclidean_space_representations(G: Group) -> tuple[Representation]:
    """Generate the E3 representation of the group G.

    This representation is used to transform all members of the Euclidean Space in 3D.
    I.e., points, vectors, pseudo-vectors, etc.
    TODO: List representations generated.

    Args:
        G (Group): Symmetry group of the robot.

    Returns:
        rep_E3 (Representation): Representation of the group G on the Euclidean Space in 3D.
    """
    # Configure E3 representations and group
    if isinstance(G, CyclicGroup):
        if G.order() == 2:  # Reflection symmetry
            rep_R3 = G.irrep(0) + G.irrep(1) + G.trivial_representation
        else:
            rep_R3 = G.irrep(1) + G.trivial_representation
    elif isinstance(G, DihedralGroup):
        rep_R3 = G.irrep(0, 1) + G.irrep(1, 1) + G.trivial_representation
    elif isinstance(G, DirectProductGroup):
        if G.name == "Klein4":
            rep_R3 = G.representations['rectangle'] + G.trivial_representation
            rep_R3 = escnn.group.change_basis(rep_R3, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), name="E3")
        elif G.name == "FullCylindricalDiscrete":
            rep_hx = np.array(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            rep_hy = np.array(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))
            rep_hz = np.array(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
            rep_E3_gens = {h: rep_h for h, rep_h in zip(G.generators, [rep_hy, rep_hx, rep_hz])}
            rep_E3_gens[G.identity] = np.eye(3)
            rep_R3 = group_rep_from_gens(G, rep_E3_gens)
        else:
            raise NotImplementedError(f"Direct product {G} not implemented yet.")
    else:
        raise NotImplementedError(f"Group {G} not implemented yet.")

    # Representation of unitary/orthogonal transformations in d dimensions.
    rep_R3.name = "R3"

    # We include some utility symmetry representations for different geometric objects.
    # We define a Ed as a (d+1)x(d+1) matrix representing a homogenous transformation matrix in d dimensions.
    rep_E3 = rep_R3 + G.trivial_representation
    rep_E3.name = "E3"

    # Build a representation of orthogonal transformations of pseudo-vectors.
    # That is if det(rep_O3(h)) == -1 [improper rotation] then we have to change the sign of the pseudo-vector.
    # See: https://en.wikipedia.org/wiki/Pseudovector
    pseudo_gens = {h: -1 * rep_R3(h) if np.linalg.det(rep_R3(h)) < 0 else rep_R3(h) for h in G.generators}
    pseudo_gens[G.identity] = np.eye(3)
    rep_R3pseudo = group_rep_from_gens(G, pseudo_gens)
    rep_R3pseudo.name = "R3_pseudo"

    rep_E3pseudo = rep_R3pseudo + G.trivial_representation
    rep_E3pseudo.name = "E3_pseudo"

    return rep_R3, rep_E3, rep_R3pseudo, rep_E3pseudo
