from pathlib import Path

import numpy as np
import pybullet
from escnn.group import Group, Representation, directsum
from pybullet_utils.bullet_client import BulletClient

from morpho_symm.data.DynamicsRecording import DynamicsRecording
from morpho_symm.utils.algebra_utils import permutation_matrix
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
from morpho_symm.utils.robot_utils import load_symmetric_system


def get_kinematic_three_rep(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(4, dtype=int)}
    gens = [permutation_matrix([1, 0, 3, 2]), permutation_matrix([2, 3, 0, 1]), permutation_matrix([0, 1, 2, 3])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three


def get_ground_reaction_forces_rep(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations['Rd']
    rep_F = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    rep_F.name = "R3_on_legs"
    return rep_F


def convert_mini_cheetah_raysim_recordings(data_path: Path):
    """Convertion script for the recordings of observations from the Mini-Cheetah Robot.

    This function takes recordings stored into a single numpy array of shape (time, state_dim) where the state is
    defined as [state]:
        base position,                           (3,)
        base_velocity,                           (3,)
        base_orientation,                        (3,)
        base_angular_velocity,                   (3,)
        feet_positions,                          (12,)
        joint position,                          (12,)
        joint velocities,                        (12,)
        joint torques,                           (12,)
        gait,                                    (4)
        reference base velocities,               (3,)
        reference base orientation,              (3,)
        reference base angular vel,              (3,)
        reference feet positions                 (12,)
        _________________________________________________________
        TOTAL:                                    85. Dimensions
    The conversion process takes these measurements and does the following:
        1. Stores them into a DynamicsRecording format, for easy loading
        2. Defines the group representation for each observation.
        3. Changes joint position to Pinocchio convention, used by MorphoSymm.
    """
    assert data_path.exists(), f"Path {data_path.absolute()} does not exist"
    state = np.load(data_path)
    assert state.shape[-1] == 85, "Something is fishy"
    # Load the Mini-Cheetah robot
    robot, G = load_symmetric_system(robot_name='mini_cheetah')
    rep_Q_js = G.representations['Q_js']  # Representation on joint space position coordinates
    rep_TqQ_js = G.representations['TqQ_js']  # Representation on joint space velocity coordinates
    rep_Rd = G.representations['Rd']  # Representation on vectors in R^d
    rep_Rd_pseudo = G.representations['Rd_pseudo']  # Representation on pseudo vectors in R^d

    # Define observation variables and their group representations z
    base_pos, base_pos_rep = state[:, :3], rep_Rd
    base_vel, base_vel_rep = state[:, 3:6], rep_Rd
    base_ori, base_ori_rep = state[:, 6:9], rep_Rd
    base_ang_vel, base_ang_vel_rep = state[:, 9:12], rep_Rd_pseudo  # Pseudo vector
    feet_pos, feet_pos_rep = state[:, 12:24], directsum([rep_Rd] * 4, name='Rd^4')
    joint_vel, joint_vel_rep = state[:, 36:48], rep_TqQ_js
    joint_torques, joint_torques_rep = state[:, 48:60], rep_TqQ_js
    rep_kin_three = get_kinematic_three_rep(G)
    gait, gait_rep = state[:, 60:64], rep_kin_three  # TODO
    ref_base_vel, ref_base_vel_rep = state[:, 64:67], rep_Rd
    ref_base_ori, ref_base_ori_rep = state[:, 67:70], rep_Rd
    ref_base_ang_vel, ref_base_ang_vel_rep = state[:, 70:73], rep_Rd_pseudo
    ref_feet_pos, rep_feet_pos = state[:, 73:85], get_ground_reaction_forces_rep(G, rep_kin_three)  # TODO

    # Joint positions need to be converted to the unit circle parametrization [cos(q), sin(q)].
    # For God’s sake, we need to avoid using PyBullet.
    bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
    robot.configure_bullet_simulation(bullet_client=bullet_client)
    # Get zero reference position.
    q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
    q_js_ms = state[:, 24:36] + q0[7:]  # Add offset to the measurements from UMich
    cos_q_js, sin_q_js = np.cos(q_js_ms), np.sin(q_js_ms)  # convert from angle to unit circle parametrization
    # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
    q_js_unit_circle_t = np.stack([cos_q_js, sin_q_js], axis=2)
    q_js_unit_circle_t = q_js_unit_circle_t.reshape(q_js_unit_circle_t.shape[0], -1)

    joint_pos, joint_pos_rep = q_js_unit_circle_t, rep_Q_js  # Joints in angle not unit circle representation

    data_recording = DynamicsRecording(
        description=f"Mini Cheetah {data_path.parent.parent.stem}",
        info=dict(num_traj=1,
                  trajectory_length=state.shape[0]),
        dynamics_parameters=dict(dt=0.001, group=dict(group_name=G.name, group_order=G.order())),
        recordings=dict(base_pos=base_pos.astype(np.float32),
                        base_vel=base_vel.astype(np.float32),
                        base_ori=base_ori.astype(np.float32),
                        base_ang_vel=base_ang_vel.astype(np.float32),
                        feet_pos=feet_pos.astype(np.float32),
                        joint_pos=joint_pos.astype(np.float32),
                        joint_vel=joint_vel.astype(np.float32),
                        joint_torques=joint_torques.astype(np.float32),
                        gait=gait.astype(np.float32),
                        ref_base_vel=ref_base_vel.astype(np.float32),
                        ref_base_ori=ref_base_ori.astype(np.float32),
                        ref_base_ang_vel=ref_base_ang_vel.astype(np.float32),
                        ref_feet_pos=ref_feet_pos.astype(np.float32),
                        ),
        state_obs=('base_pos', 'base_vel', 'base_ori', 'base_ang_vel',),
        action_obs=('joint_torques',),
        obs_representations=dict(base_pos=base_pos_rep,
                                 base_vel=base_vel_rep,
                                 base_ori=base_ori_rep,
                                 base_ang_vel=base_ang_vel_rep,
                                 ref_base_vel=ref_base_vel_rep,
                                 ref_base_ori=ref_base_ori_rep,
                                 ref_base_ang_vel=ref_base_ang_vel_rep,
                                 feet_pos=feet_pos_rep,
                                 joint_pos=joint_pos_rep,
                                 joint_vel=joint_vel_rep,
                                 joint_torques=joint_torques_rep,
                                 gait=gait_rep,
                                 ref_feet_pos=rep_feet_pos,
                                 ),
        )
    file_path = data_path.parent.parent / "recording"
    data_recording.save_to_file(file_path)
    print(f"Dynamics Recording saved to {file_path.absolute()}")


if __name__ == "__main__":
    data_path = Path("flat_terrain/forward_minus_0_4/heightmap_logger/state_reduced_nmpc.npy")
    convert_mini_cheetah_raysim_recordings(data_path)
