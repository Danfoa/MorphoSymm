#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com

import importlib
import re

import numpy as np
import escnn
import escnn.group
from escnn.group import Group, CyclicGroup, DihedralGroup, DirectProductGroup
from omegaconf import DictConfig

# from groups.SparseRepresentation import SparseRep, SparseRepE3
# from groups.SymmetryGroups import C2, Klein4, Sym
from robots.PinBulletWrapper import PinBulletWrapper
from .algebra_utils import configure_bullet_simulation, gen_permutation_matrix
from .group_utils import group_rep_from_gens
from .pybullet_visual_utils import setup_debug_sliders, listen_update_robot_sliders
from .simultaneous_block_diagonalization import escnn_representation_form_mapping


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_escnn_group(cfg: DictConfig):
    group_label = cfg.group_label
    label_pattern = r'([A-Za-z]+)(\d+)'
    match = re.match(label_pattern, group_label)
    if match:
        group_class = match.group(1)
        order = int(match.group(2))
    else:
        raise AttributeError(f'Group label {group_label} is not a known group label (Dn: Dihedral, Cn: Cyclic) order n')

    group_axis = np.array([0, 0, 1])
    subgroup_id = np.zeros_like(group_axis, dtype=np.bool).astype(object)
    if group_class.lower() == 'd':  # Dihedral
        # Define the symmetry space using presets from ESCNN
        # subgroup_id[group_axis == 1] = order
        symmetry_space = escnn.gspaces.dihedralOnR3(n=order//2, axis=0.0)
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


def load_robot_and_symmetries(robot_cfg, debug=False) -> [PinBulletWrapper, escnn.gspaces.GSpace3D]:
    """
    Utility function to get the symmetry group and representations of a robotic system defined in config.
    :param robot_cfg: Dictionary holding the configuration parameters of the robot. Check `cfg/robot/`
    :param debug: whether to load the robot into an interactive simulation session to visually debug URDF
    :return:
        robot: `PinBulletWrapper` instance with the robot loaded in pinocchio and ready to be loaded in pyBullet
        rep_Ed: Representation of the symmetry group of the robot on the Euclidean space of `d` dimensions.
        rep_QJ: Representation of the symmetry group of the robot on the Joint Space of the robot.
    """
    robot_name = str.lower(robot_cfg.name)
    robot = PinBulletWrapper(robot_name=robot_name, init_q=robot_cfg.init_q, hip_height=robot_cfg.hip_height,
                             endeff_names=robot_cfg.endeff_names)

    if debug:
        pb = configure_bullet_simulation(gui=True, debug=debug)
        robot.configure_bullet_simulation(pb, world=None)
        setup_debug_sliders(pb, robot)
        listen_update_robot_sliders(pb, robot)

    symmetry_space = get_escnn_group(robot_cfg)

    G = symmetry_space.fibergroup

    # Transformation required to obtain ρ_Q_js(g) ∈ G from the regular fields / permutation rep.
    num_symmetric_dof = robot.nj - robot_cfg.unique_bodies

    rep_field = np.float if robot_cfg.rep_fields.lower() != 'complex' else np.complex
    # Configuration file of robot provides oneline notations of the reps_QJ of generators of the group.
    rep_QJ = {G.identity: np.eye(robot.nj, dtype=rep_field)}
    for g_gen, perm, refx in zip(G.generators, robot_cfg.perm_qj, robot_cfg.refx_qj):
        assert len(perm) == robot.nj == len(refx), f"Perm {len(perm)} Reflx: {len(refx)}"
        refx = np.array(refx, dtype=rep_field)
        rep_QJ[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)

    # Add `Ed` and `QJ` representations to the group.
    rep_QJ = group_rep_from_gens(G, rep_QJ)
    rep_Ed = generate_E3_rep(G)

    # a = [rep_QJ(g) for g in G.elements]
    G.representations.update(Ed=rep_Ed, QJ=rep_QJ)
    return robot, symmetry_space


def generate_E3_rep(G: Group) -> None:
    # Configure E3 representations and group
    if isinstance(G, CyclicGroup):
        rep_E3 = G.irrep(0) + G.irrep(1) + G.trivial_representation
    elif isinstance(G, DihedralGroup):
        rep_E3 = G.irrep(0, 1) + G.irrep(1, 1) + G.trivial_representation
    elif isinstance(G, DirectProductGroup):
        if G.name == "Klein4":
            rep_E3 = G.representations['rectangle'] + G.trivial_representation
            rep_E3 = escnn.group.change_basis(rep_E3, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), name="E3")
        elif G.name == "FullCylindricalDiscrete":
            rep_hx = np.array(np.array([[-1, 0, 0], [0,  1, 0], [0, 0,  1]]))
            rep_hy = np.array(np.array([[ 1, 0, 0], [0, -1, 0], [0, 0,  1]]))
            rep_hz = np.array(np.array([[ 1, 0, 0], [0,  1, 0], [0, 0, -1]]))
            rep_E3_gens = {h: rep_h for h, rep_h in zip(G.generators, [rep_hy, rep_hx, rep_hz])}
            rep_E3 = group_rep_from_gens(G, rep_E3_gens)
    else:
        raise NotImplementedError(f"Group {G} not implemented yet.")
    rep_E3.name = "E3"
    return rep_E3