import time
from pathlib import Path

import hydra
import numpy as np
from escnn.group import directsum
from omegaconf import DictConfig
from pynput import keyboard

from morpho_symm.utils.abstract_harmonics_analysis import decom_signal_into_isotypic_components
from morpho_symm.utils.algebra_utils import matrix_to_quat_xyzw
from utils.pybullet_visual_utils import change_robot_appearance, spawn_robot_instances
from utils.robot_utils import load_symmetric_system

import morpho_symm
from morpho_symm.data.DynamicsRecording import DynamicsRecording
from morpho_symm.robot_symmetry_visualization_dynamic import load_mini_cheetah_trajs
from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper
from morpho_symm.utils.pybullet_visual_utils import configure_bullet_simulation
from morpho_symm.utils.rep_theory_utils import irreps_stats


def on_key_press(key):
    """TODO: Remove from here."""
    try:
        # Get the pressed key and add it to the global list of new characters
        if key.char.isdigit():
            num_pressed.append(int(key.char))
        else:
            new_command.append(key.char)
    except AttributeError:
        # Ignore special keys without a char attribute
        pass


# Set up the listener and the list to store new characters
new_command = []
num_pressed = []
listener = keyboard.Listener(on_press=on_key_press)
listener.start()


def get_motion_trajectory(robot: PinBulletWrapper, recording_name=None, angle_sweep=0.5):
    # Load a trajectory of motion and measurements from the mini-cheetah robot
    recordings_path = Path(
        morpho_symm.__file__).parent / f'data/{robot.robot_name}/raysim_recordings/flat/forward_minus_0_4/n_trajs=1-frames=7771-train.pkl'
    dyn_recordings = DynamicsRecording.load_from_file(recordings_path)
    return dyn_recordings
    #
    # if robot.robot_name == 'mini_cheetah':
    #     recordings_path = Path(morpho_symm.__file__).parent / 'data/contact_dataset/training_splitted/mat_test'
    #     recordings = load_mini_cheetah_trajs(recordings_path)
    #     recording_name = 'forest'
    #     recording = recordings[recording_name]
    #     # timestamps = recording['control_time'].flatten()
    #     q_js_t = recording['q']  # Joint space positions
    #     # v_js_t = recording['qd']  # Joint space velocities
    #     base_ori_t = recording['imu_rpy']
    #     # ground_reaction_forces = recording['F']
    #     # base_acc = recording['imu_acc']
    #     # base_ang_vel = recording['imu_omega']
    #     # feet_contact_states = recording['contacts']
    #
    #     q = []
    #     q0, _ = robot.pin2sim(robot._q_zero, np.zeros(robot.nv))
    #     for q_js, base_ori in zip(q_js_t, base_ori_t):
    #         # Define the recording base configuration.
    #         q_rec = np.concatenate([q0[:7], q_js])
    #         v_rec = np.zeros(robot.nv)
    #         # Just for Mit Cheetah.
    #         q_t, _ = robot.sim2pin(q_rec - q0, v_rec)
    #         q.append(q_t)
    #     return np.asarray(q)
    # else:
    #     n_dof = robot.nq - 7
    #     q0, _ = robot.get_init_config(False)
    #     phases = np.random.uniform(0, 2 * np.pi, n_dof)[..., None]
    #     amplitudes = np.random.uniform(0, angle_sweep, n_dof)[..., None]
    #     max_period, min_period = 6, 3
    #     periods = np.random.randint(min_period, max_period, n_dof)[..., None]
    #     q_period = np.lcm.reduce(periods).item()
    #     t = np.linspace(0, q_period, q_period * 10)[None, ...]
    #     q_js = q0[7:, None] + 0 * np.sin(2 * np.pi * (1 / periods) * t + phases) * amplitudes
    #     q_base = q0[:7, None] * np.ones_like(t)
    #     q = np.concatenate([q_base, q_js], axis=0).T
    #     return q


def fix_SO3_reprojection(R_iso):
    x = R_iso[0, :]
    y = R_iso[1, :]
    z = R_iso[2, :]
    x_norm, y_norm, z_norm = np.linalg.norm(x), np.linalg.norm(y), np.linalg.norm(z)
    if not np.isclose(x_norm, 1.0):
        x = np.array([1, 0, 0])
    if not np.isclose(y_norm, 1.0):
        y = np.array([0, 1, 0])
    if not np.isclose(z_norm, 1.0):
        z = np.array([0, 0, 1])
    R_iso = np.stack([x, y, z], axis=0)
    return R_iso


@hydra.main(config_path='cfg', config_name='config_visualization', version_base='1.2')
def main(cfg: DictConfig):
    robot, G = load_symmetric_system(robot_cfg=cfg.robot, debug=cfg.debug)
    rep_Q_js = G.representations['Q_js']
    rep_TqQ_js = G.representations['TqQ_js']
    rep_SO3_flat = G.representations['SO3_flat']
    rep_R3 = G.representations['R3']
    rep_R3_pseudo = G.representations['R3_pseudo']
    # Define the representation for the entire state configuration.
    rep_state = directsum([rep_Q_js, rep_TqQ_js, rep_SO3_flat, rep_R3, rep_R3_pseudo])

    # Load main robot in pybullet.
    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    q0, dq0 = robot.get_init_config(random=False)
    robot.configure_bullet_simulation(pb, base_pos=q0[:3], base_ori=q0[3:7])
    if cfg.robot.tint_bodies: change_robot_appearance(pb, robot)

    # For each isotypic component we spawn a robot instance to visualize the effect of the decomposition
    unique_irrep_ids, counts, indices = irreps_stats(rep_state.irreps)
    n_iso_comp = len(unique_irrep_ids)  # Number of isotypic subspaces equal to the unique irreps.
    base_positions = np.asarray([q0[:3]] * n_iso_comp)
    base_positions[:, 0] = -1.0  # Offset Iso Conf robots to the back.
    base_positions[:, 1] = np.linspace(0, 2.5 * robot.hip_height * n_iso_comp, n_iso_comp)
    base_positions[:, 1] -= np.max(base_positions[:, 1]) / 2
    iso_robots = spawn_robot_instances(robot,
                                       bullet_client=pb,
                                       base_positions=base_positions,
                                       base_orientations=[q0[3:7]] * n_iso_comp,
                                       tint=cfg.robot.tint_bodies,
                                       alpha=1.0)
    # Load and prepare data for visualization
    dyn_recordings = get_motion_trajectory(robot, recording_name=cfg.recording_name)
    q_js_t = dyn_recordings.recordings['joint_pos'][0]
    v_js_t = dyn_recordings.recordings['joint_vel'][0]
    R_flat_t = dyn_recordings.recordings['base_ori_R_flat'][0]
    base_vel_t = dyn_recordings.recordings['base_vel'][0]
    base_and_vel_t = dyn_recordings.recordings['base_ang_vel'][0]
    dt = dyn_recordings.dynamics_parameters['dt']
    traj_length = q_js_t.shape[0]
    signal_time = np.arange(0, traj_length * dt, dt)
    # Define state trajectory.
    state_traj = np.concatenate([q_js_t, v_js_t, R_flat_t, base_vel_t, base_and_vel_t], axis=1)

    def get_obs_from_state(state: np.array, state_rep):
        """Auxiliary function to extract the different observations of the state."""
        assert state.shape[0] == state_rep.size, f"Expected state of size {state_rep.size} but got {state.shape[0]}"
        q_js_end = robot.nq - 7
        v_js_end = q_js_end + (robot.nv - 6)
        q_js = state[:q_js_end]
        v_js = state[q_js_end:v_js_end]
        R_end = v_js_end + 9
        R_flat = state[v_js_end:R_end]
        R = R_flat.reshape(3, 3)
        base_vel = state[R_end:R_end + 3]
        base_ang_vel = state[R_end + 3:]
        return q_js, v_js, R, base_vel, base_ang_vel

    comp_iso_basis, comp_canonical_basis = decom_signal_into_isotypic_components(state_traj, rep_state)
    iso_robots = {irrep_id: iso_robot for irrep_id, iso_robot in zip(comp_canonical_basis.keys(), iso_robots)}
    iso_robots_pos = {irrep_id: pos for irrep_id, pos in zip(comp_canonical_basis.keys(), base_positions)}
    idx = 0
    fps = 30 # Control the animation update time.
    t_last_update = 0
    g, g_prev = G.identity, G.identity
    while True:
        if idx >= traj_length:
            idx = 0
        # t = signal_time[idx]
        # if t - t_last_update < 1 / fps:  # Dont update visualization too often.
        #     idx += dt
        #     continue
        # t_last_update = t

        state = state_traj[idx]
        q_js, v_js, base_SO3, base_vel, base_ang_vel = get_obs_from_state(state, rep_state)
        base_ori = matrix_to_quat_xyzw(base_SO3)
        q = np.concatenate((q0[:3], base_ori, q_js))
        v = np.concatenate((base_vel, base_ang_vel, v_js))
        robot.reset_state(q, v)

        # if g != g_prev:  # Apply symmetry transformation if required.
        #     state = rep_state(g) @ state
        #     for irrep_id, state_comp in comp_canonical_basis.items():
        #         comp_canonical_basis[irrep_id] = rep_state(g) @ state_comp
        #     g_prev = g

        for irrep_id, state_comp in comp_canonical_basis.items():
            iso_robot = iso_robots[irrep_id]
            q_js_iso, v_js_iso, R_iso, base_vel_iso, base_ang_vel_iso = get_obs_from_state(state_comp[idx], rep_state)
            R_iso_reformated = np.eye(3) # fix_SO3_reprojection(R_iso)
            base_ori_iso = matrix_to_quat_xyzw(R_iso_reformated)
            q_iso = np.concatenate((iso_robots_pos[irrep_id], base_ori_iso, q_js_iso))
            v_iso = np.concatenate((base_vel_iso, base_ang_vel_iso, v_js_iso))
            iso_robot.reset_state(q_iso, v_iso)
        # components_q_js = []
        # for iso_robot, (re_irrep, dims) in zip(iso_robots, iso_comp.items()):
        #     q, dq = iso_robot.get_state()
        #     # Get point in isotypic component and describe it in the basis of generalized coordinates.
        #     q_iso_masked = q_iso * dims
        #     # Transform back to generalized coordinates.
        #     q_js_comp = np.real(Q @ q_iso_masked)
        #     components_q_js.append(q_js_comp)
        #     # Apply selected symmetry action
        #     g_q_js_comp = np.real(rep_Q_js(g) @ q_js_comp)
        #     # Set the robot to desired state.
        #     g_q = np.concatenate((q[:7], g_q_js_comp))
        #     iso_robot.reset_state(g_q, dq)

        idx += 1
        # ========================================================================
        # Process new keyboard commands.
        if new_command:
            keys = new_command.copy()
            new_command.clear()
            if keys == ['p']:
                dt = 1 if dt == 0 else 0
        if num_pressed:
            if num_pressed[0] < G.order():
                g_idx = num_pressed[0]
                g = G.elements[g_idx]
                print(f"Group element selected {g}")
            else:
                print(f"Group element {num_pressed[0]} is larger than group order...ignoring")
            num_pressed.clear()

    pb.disconnect()


if __name__ == '__main__':
    main()
