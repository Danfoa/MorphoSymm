from pathlib import Path

import numpy as np
import pybullet
from escnn.group import Group, Representation, directsum
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation

from morpho_symm.data.DynamicsRecording import DynamicsRecording, split_train_val_test
from morpho_symm.utils.algebra_utils import permutation_matrix
from morpho_symm.utils.rep_theory_utils import escnn_representation_form_mapping, group_rep_from_gens
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


def get_Rd_signals_on_kin_subchains(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations['R3']
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
        reference base height                    (1,)
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
    assert state.shape[-1] == 86, f"Expected {86} dimensions in the state, got {state.shape[-1]}"
    # Load the Mini-Cheetah robot
    robot, G = load_symmetric_system(robot_name='mini_cheetah')
    rep_Q_js = G.representations['Q_js']  # Representation on joint space position coordinates
    rep_TqQ_js = G.representations['TqQ_js']  # Representation on joint space velocity coordinates
    rep_Rd = G.representations['R3']  # Representation on vectors in R^d
    rep_Rd_pseudo = G.representations['R3_pseudo']  # Representation on pseudo vectors in R^d
    rep_euler_xyz = G.representations['euler_xyz']  # Representation on Euler angles
    rep_kin_three = get_kinematic_three_rep(G)  # Permutation of legs
    rep_Rd_on_limbs = get_Rd_signals_on_kin_subchains(G, rep_kin_three)  # Representation on R^3 on legs

    rep_z = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_z.name = "base_z"

    # Define observation variables and their group representations

    # Base body observations ___________________________________________________________________________________________
    base_pos = state[:, :3]  # Rep: rep_Rd
    base_z = state[:, [2]]  # Rep: rep_z
    base_vel = state[:, 3:6]  # Rep: rep_Rd
    base_ori = state[:, 6:9]  # Rep: rep_euler_xyz
    base_ang_vel = state[:, 9:12]  # Rep: rep_euler_xyz
    ref_base_z = state[:, [64]]  # Rep: rep_z
    ref_base_vel = state[:, 65:68]  # Rep: rep_Rd
    ref_base_ori = state[:, 68:71]  # Rep: rep_euler_xyz
    ref_base_ang_vel = state[:, 71:74]  # Rep: rep_euler_xyz
    base_z_error = base_z - ref_base_z  # Rep: rep_z
    base_vel_error = base_vel - ref_base_vel  # Rep: rep_Rd
    base_ang_vel_error = base_ang_vel - ref_base_ang_vel  # Rep: rep_euler_xyz
    base_ori_error = base_ori - ref_base_ori  # Rep: rep_euler_xyz
    # Define the representation of the rotation matrix R that transforms the base orientation.
    rep_rot_flat = {}
    # R = Rotation.from_euler("xyz", base_ori[2]).as_matrix()
    for h in G.elements:
        rep_rot_flat[h] = np.kron(rep_Rd(h), rep_Rd(~h).T)
    rep_rot_flat = escnn_representation_form_mapping(G, rep_rot_flat)
    rep_rot_flat.name = "SO(3)_flat"
    base_ori_R = np.asarray([Rotation.from_euler("xyz", ori).as_matrix() for ori in base_ori])
    base_ori_R_flat = base_ori_R.reshape(base_ori.shape[0], -1)

    # Euclidean space observations _____________________________________________________________________________________
    feet_pos = state[:, 12:24]  # Rep: rep_Rd_on_limbs
    gait = state[:, 60:64]  # Rep: rep_kin_three
    ref_feet_pos = state[:, 74:86]  # Rep: rep_Rd_on_limbs
    feet_pos_error = feet_pos - ref_feet_pos  # Rep: rep_Rd_on_limbs
    # Joint-Space observations _________________________________________________________________________________________
    joint_vel = state[:, 36:48]
    joint_torques = state[:, 48:60]
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
    joint_pos_S1, joint_pos_rep = q_js_unit_circle_t, rep_Q_js  # Joints in angle not unit circle representation
    joint_pos = q_js_ms  # Joints in angle representation

    # Subsample the data by skippig by ignoring odd frames. ============================================================
    dt_subsample = 3
    base_pos = base_pos[::dt_subsample]
    base_z = base_z[::dt_subsample]
    base_vel = base_vel[::dt_subsample]
    base_ori = base_ori[::dt_subsample]
    base_ori_R_flat = base_ori_R_flat[::dt_subsample]
    base_ang_vel = base_ang_vel[::dt_subsample]
    feet_pos = feet_pos[::dt_subsample]
    joint_pos = joint_pos[::dt_subsample]
    joint_pos_S1 = joint_pos_S1[::dt_subsample]
    joint_vel = joint_vel[::dt_subsample]
    joint_torques = joint_torques[::dt_subsample]
    gait = gait[::dt_subsample]
    ref_base_z = ref_base_z[::dt_subsample]
    ref_base_vel = ref_base_vel[::dt_subsample]
    ref_base_ori = ref_base_ori[::dt_subsample]
    ref_base_ang_vel = ref_base_ang_vel[::dt_subsample]
    ref_feet_pos = ref_feet_pos[::dt_subsample]
    feet_pos_error = feet_pos_error[::dt_subsample]
    base_z_error = base_z_error[::dt_subsample]
    base_vel_error = base_vel_error[::dt_subsample]
    base_ang_vel_error = base_ang_vel_error[::dt_subsample]
    base_ori_error = base_ori_error[::dt_subsample]
    # Define the dataset.
    data_recording = DynamicsRecording(
        description=f"Mini Cheetah {data_path.parent.parent.stem}",
        info=dict(num_traj=1,
                  trajectory_length=state.shape[0]),
        dynamics_parameters=dict(dt=0.001 * dt_subsample, group=dict(group_name=G.name, group_order=G.order())),
        recordings=dict(base_pos=base_pos[None, ...].astype(np.float32),
                        base_z=base_pos[None, :, [2]].astype(np.float32),
                        base_vel=base_vel[None, ...].astype(np.float32),
                        base_ori=base_ori[None, ...].astype(np.float32),
                        base_ori_R_flat=base_ori_R_flat[None, ...].astype(np.float32),
                        base_ang_vel=base_ang_vel[None, ...].astype(np.float32),
                        feet_pos=feet_pos[None, ...].astype(np.float32),
                        feet_pos_error=feet_pos_error[None, ...].astype(np.float32),
                        joint_pos=joint_pos[None, ...].astype(np.float32),
                        joint_pos_S1=joint_pos_S1[None, ...].astype(np.float32),
                        joint_vel=joint_vel[None, ...].astype(np.float32),
                        joint_torques=joint_torques[None, ...].astype(np.float32),
                        gait=gait[None, ...].astype(np.float32),
                        base_z_error=base_z_error[None, ...].astype(np.float32),
                        base_vel_error=base_vel_error[None, ...].astype(np.float32),
                        base_ang_vel_error=base_ang_vel_error[None, ...].astype(np.float32),
                        base_ori_error=base_ori_error[None, ...].astype(np.float32),
                        ),
        state_obs=('joint_pos', 'joint_vel', 'base_z_error', 'base_ori', 'base_ori_error', 'base_vel_error', 'base_ang_vel_error'),
        action_obs=('joint_torques',),
        obs_representations=dict(joint_pos=rep_TqQ_js,  # Joint-Space observations
                                 joint_pos_S1=rep_Q_js,
                                 joint_vel=rep_TqQ_js,
                                 joint_torques=rep_TqQ_js,
                                 # Base body observations
                                 base_pos=rep_Rd,
                                 base_z=rep_z,
                                 base_z_error=rep_z,
                                 base_vel=rep_Rd,
                                 base_vel_error=rep_Rd,
                                 base_ori=rep_euler_xyz,
                                 base_ori_R_flat=rep_rot_flat,
                                 base_ang_vel=rep_euler_xyz,
                                 base_ang_vel_error=rep_euler_xyz,
                                 base_ori_error=rep_euler_xyz,
                                 # Euclidean space observations
                                 feet_pos=rep_Rd_on_limbs,
                                 feet_pos_error=rep_Rd_on_limbs,
                                 gait=rep_kin_three,
                                 ),
        # Ensure the angles in the unit circle are not disturbed by the normalization.
        obs_moments=dict(joint_pos_S1=(np.zeros(q_js_unit_circle_t.shape[-1]), np.ones(q_js_unit_circle_t.shape[-1]),),
                         base_ori_R_flat=(np.zeros(base_ori_R_flat.shape[-1]), np.ones(base_ori_R_flat.shape[-1]),),
                         )
        )

    # Compute the mean and variance of all observations considering symmetry constraints.
    for obs_name in data_recording.recordings.keys():
        if obs_name in data_recording.obs_moments:
            continue
        data_recording.compute_obs_moments(obs_name=obs_name)

    file_name = (f"n_trajs={data_recording.info['num_traj']}"
                 f"-frames={data_recording.info['trajectory_length']}.pkl")
    data_recording.save_to_file(data_path.parent.parent / file_name)
    print(f"Dynamics Recording saved to {data_path.parent.parent / file_name}")


if __name__ == "__main__":
    terrains = ["flat", "uneven_easy", "uneven_medium", "uneven_hard_squares"]
    modes = ["forward_minus_0_4", "forward_minus_0_4_yawrate_0_4", "forward_minus_0_4_yawrate_minus_0_4"]
    for terrain in terrains:
        for mode in modes:
            data_path = Path(f"raysim_recordings/{terrain}/{mode}/heightmap_logger/state_reduced_nmpc.npy")
            convert_mini_cheetah_raysim_recordings(data_path)
