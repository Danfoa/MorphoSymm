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
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
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
        robot_cfg: Optional[DictConfig] = None, robot_name: Optional[str] = None, debug=False
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
        path_cfg = Path(morpho_symm.__file__).parent / 'cfg' / 'robot'
        path_robot_cfg = path_cfg / f'{robot_name}.yaml'
        assert path_robot_cfg.exists(), \
            f"Robot configuration {path_robot_cfg} does not exist."
        base_cfg = OmegaConf.load(path_cfg / 'base_robot.yaml')
        robot_cfg = OmegaConf.load(path_robot_cfg)
        robot_cfg = OmegaConf.merge(base_cfg, robot_cfg)

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

    # Get the dimensions of the spaces Q_js and TqQ_js
    dimQ_js, dimTqQ_js = robot.nq - 7, robot.nv - 6

    # Joint-Space Q_js (generalized position coordinates)
    rep_Q_js = {G.identity: np.eye(dimQ_js, dtype=rep_field)}
    # Check a representation for each generator is provided
    assert len(robot_cfg.permutation_Q_js) == len(robot_cfg.reflection_Q_js) >= len(G.generators), \
        f"Not enough representation provided for the joint-space `Q_js`. " \
        f"Found {len(robot_cfg.permutation_TqQ_js)} but symmetry group {G} has {len(G.generators)} generators."

    # Generate ESCNN representation of generators
    for g_gen, perm, refx in zip(G.generators, robot_cfg.permutation_Q_js, robot_cfg.reflection_Q_js):
        assert len(perm) == dimQ_js == len(refx), \
            f"Dimension of joint-space position coordinates dim(Q_js)={robot.n_js} != dim(rep_Q_JS): {len(refx)}"
        refx = np.array(refx, dtype=rep_field)
        rep_Q_js[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)
    # Generate the entire group
    rep_Q_js = group_rep_from_gens(G, rep_Q_js)

    # Joint-Space Tangent bundle TqQ_js (generalized velocity coordinates)
    rep_TqQ_js = {G.identity: np.eye(dimTqQ_js, dtype=rep_field)}
    if dimQ_js == dimTqQ_js:  # If position and velocity coordinates have the same dimensions
        rep_TqQ_js = rep_Q_js
    else:
        # Check a representation for each generator is provided
        assert robot_cfg.permutation_TqQ_js is not None and robot_cfg.reflection_TqQ_js is not None, \
            f"No representations provided for the joint-space tangent bundle rep_TqQ_js of {robot_name}"
        assert len(robot_cfg.permutation_TqQ_js) == len(robot_cfg.reflection_TqQ_js) >= len(G.generators), \
            f"Not enough representation provided for the joint-space tangent bundle `TqQ_js`. " \
            f"Found {len(robot_cfg.permutation_TqQ_js)} but symmetry group {G} has {len(G.generators)} generators."

        # Generate ESCNN representation of generators
        for g_gen, perm, refx in zip(G.generators, robot_cfg.permutation_TqQ_js, robot_cfg.reflection_TqQ_js):
            assert len(perm) == dimTqQ_js == len(refx), \
                f"Dimension of joint-space position coordinates dim(Q_js)={robot.n_js} != dim(rep_Q_JS): {len(refx)}"
            refx = np.array(refx, dtype=rep_field)
            rep_TqQ_js[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)
        # Generate the entire group
        rep_TqQ_js = group_rep_from_gens(G, rep_TqQ_js)
        rep_TqQ_js.name = 'TqQ_js'

    rep_Q_js.name = 'Q_js'

    # Create the representation of isometries on the Euclidean Space in d dimensions.
    generate_euclidean_space_representations(G)  # This adds `O3` and `E3` representations to the group.

    # Add representations to the group.
    G.representations.update(Q_js=rep_Q_js, TqQ_js=rep_TqQ_js)
    return robot, G


def generate_euclidean_space_representations(G: Group) -> Representation:
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

    # We include some utility symmetry representations for different geometric objects.
    # We define a Ed as a (d+1)x(d+1) matrix representing a homogenous transformation matrix in d dimensions.
    rep_E3 = rep_R3 + G.trivial_representation
    rep_E3.name = "E3"

    # Representation of unitary/orthogonal transformations in d dimensions.
    rep_R3.name = "R3"
    # Build a representation of orthogonal transformations of pseudo-vectors.
    # That is if det(rep_O3(h)) == -1 [improper rotation] then we have to change the sign of the pseudo-vector.
    # See: https://en.wikipedia.org/wiki/Pseudovector
    psuedo_gens = {h: -1 * rep_R3(h) if np.linalg.det(rep_R3(h)) < 0 else rep_R3(h) for h in G.generators}
    psuedo_gens[G.identity] = np.eye(3)
    rep_R3pseudo = group_rep_from_gens(G, psuedo_gens)
    rep_R3pseudo.name = "R3_pseudo"

    # Representation of quaternionic transformations in d dimensions.
    # TODO: Add quaternion representation
    # quat_gens = {h: Rotation.from_matrix(rep_O3(h)).as_quat() for h in G.generators}
    # quat_gens[G.identity] = np.eye(4)
    # TODO: Add quaternion pseudo representation
    # rep_O3pseudo = group_rep_from_gens(G, psuedo_gens)

    G.representations.update(Rd=rep_R3, Ed=rep_E3, Rd_pseudo=rep_R3pseudo)
