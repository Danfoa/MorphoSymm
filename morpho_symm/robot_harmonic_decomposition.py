import time
from pathlib import Path

import hydra
import numpy as np
from escnn.group import Group, Representation
from omegaconf import DictConfig
from pynput import keyboard
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import morpho_symm
from morpho_symm.data.DynamicsRecording import DynamicsRecording
from morpho_symm.robot_symmetry_visualization_dynamic import load_mini_cheetah_trajs
from morpho_symm.utils.algebra_utils import matrix_to_quat_xyzw
from utils.pybullet_visual_utils import change_robot_appearance, spawn_robot_instances
from utils.robot_utils import load_symmetric_system

from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper
from morpho_symm.utils.pybullet_visual_utils import configure_bullet_simulation


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


def generate_dof_motions(robot: PinBulletWrapper, angle_sweep=0.5):
    """TODO: In construction."""
    if robot.robot_name == 'mini_cheetah':
        recordings_path = Path(morpho_symm.__file__).parent / 'data/contact_dataset/training_splitted/mat_test'
        recordings = load_mini_cheetah_trajs(recordings_path)
        recording_name = 'forest'
        recording = recordings[recording_name]
        # timestamps = recording['control_time'].flatten()
        q_js_t = recording['q']  # Joint space positions
        # v_js_t = recording['qd']  # Joint space velocities
        base_ori_t = recording['imu_rpy']
        # ground_reaction_forces = recording['F']
        # base_acc = recording['imu_acc']
        # base_ang_vel = recording['imu_omega']
        # feet_contact_states = recording['contacts']

        q = []
        q0, _ = robot.pin2sim(robot._q_zero, np.zeros(robot.nv))
        for q_js, base_ori in zip(q_js_t, base_ori_t):
            # Define the recording base configuration.
            q_rec = np.concatenate([q0[:7], q_js])
            v_rec = np.zeros(robot.nv)
            # Just for Mit Cheetah.
            q_t, _ = robot.sim2pin(q_rec - q0, v_rec)
            q.append(q_t)
        return np.asarray(q)
    else:
        n_dof = robot.nq - 7
        q0, _ = robot.get_init_config(False)
        phases = np.random.uniform(0, 2 * np.pi, n_dof)[..., None]
        amplitudes = np.random.uniform(0, angle_sweep, n_dof)[..., None]
        max_period, min_period = 6, 3
        periods = np.random.randint(min_period, max_period, n_dof)[..., None]
        q_period = np.lcm.reduce(periods).item()
        t = np.linspace(0, q_period, q_period * 10)[None, ...]
        q_js = q0[7:, None] + 0 * np.sin(2 * np.pi * (1 / periods) * t + phases) * amplitudes
        q_base = q0[:7, None] * np.ones_like(t)
        q = np.concatenate([q_base, q_js], axis=0).T
        return q



@hydra.main(config_path='cfg', config_name='config_visualization', version_base='1.1')
def main(cfg: DictConfig):
    """Visualize the effect of DMSs transformations in 3D animation.

    This script visualizes the DMSs transformations on robot state and on proprioceptive and exteroceptive measurements.
    """
    cfg.robot.seed = cfg.robot.seed if cfg.robot.seed >= 0 else np.random.randint(0, 1000)
    np.random.seed(cfg.robot.seed)
    # Get robot instance, along with representations of the symmetry group on the Euclidean space (in which the robot
    # base B evolves in) and Joint Space (in which the internal configuration of the robot evolves in).
    robot, G = load_symmetric_system(robot_cfg=cfg.robot, debug=cfg.debug)
    assert isinstance(G, Group)
    assert np.all(rep_name in G.representations for rep_name in ['Ed', 'Q_js', 'TqQ_js']), \
        f"Group {G} should have representations for Ed, Q_js and TqQ_js, found: {list(G.representations.keys())}"
    rep_Q_js = G.representations['Q_js']
    rep_QJ_iso = Representation(G, name="Q_js_iso", irreps=rep_Q_js.irreps, change_of_basis=np.eye(rep_Q_js.size))
    rep_TqJ = G.representations['TqQ_js']
    rep_Ed = G.representations['Ed']

    # Load main robot in pybullet.
    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    robot.configure_bullet_simulation(pb)
    n_dof = robot.nq - 7
    if cfg.robot.tint_bodies: change_robot_appearance(pb, robot)
    q0, dq0 = robot.get_init_config(random=True, angle_sweep=cfg.robot.angle_sweep, fix_base=cfg.robot.fix_base)
    orientation_0 = matrix_to_quat_xyzw(Rotation.from_euler('xyz', [0, 0, np.pi / 2]).as_matrix())
    base_pos = q0[:3]
    q0[3:7] = orientation_0
    robot.reset_state(q0, dq0)

    # Determine the number of isotypic components of the Joint-Space (JS) vector space.
    # This is equivalent to the number of unique irreps of the JS representation.
    iso_comp = {}  # TODO: Make a class for a Component.
    mask = []
    for re_irrep_id in rep_Q_js.irreps:
        mask.extend([re_irrep_id] * G.irrep(*re_irrep_id).size)
    for re_irrep_id in rep_Q_js.irreps:
        re_irrep = G.irrep(*re_irrep_id)
        dims = np.zeros(n_dof)
        dims[[i for i, x in enumerate(mask) if x == re_irrep_id]] = 1
        iso_comp[re_irrep] = dims
        # print(f"Re irrep: {re_irrep} - Trivial: {re_irrep.is_trivial()} - Mult: {multiplicities[idx]}")

    # For each isotypic component we spawn a robot instance in order to visualize the effect of the decomposition
    n_components = len(iso_comp)
    base_positions = np.asarray([base_pos] * n_components)
    base_positions[:, 0] = -1.0
    base_positions[:, 1] = np.linspace(0, 2.5 * robot.hip_height * n_components, n_components)
    base_positions[:, 1] -= np.max(base_positions[:, 1]) / 2
    # Base positions. Quaternion from (roll=0, pitch=0, yaw=90)

    iso_robots = spawn_robot_instances(
        robot, bullet_client=pb, base_positions=base_positions, base_orientations=[orientation_0] * n_components,
        tint=cfg.robot.tint_bodies, alpha=1.0,
        )

    # Load a trajectory of motion and measurements from the mini-cheetah robot
    recordings_path = Path(
        morpho_symm.__file__).parent / 'data/mini_cheetah/raysim_recordings/flat/forward_minus_0_4/n_trajs=1-frames=7771-train.pkl'
    dyn_recordings = DynamicsRecording.load_from_file(recordings_path)
    # Load and prepare data for visualization
    q_js_t = dyn_recordings.recordings['joint_pos']     # Joint space positions
    v_js_t = dyn_recordings.recordings['joint_vel']     # Joint space velocities
    base_ori_t = dyn_recordings.recordings['base_ori']  # Base orientation
    feet_pos = dyn_recordings.recordings['feet_pos']  # Feet positions  [x,y,z] w.r.t base frame
    # ground_reaction_forces = recording['F']
    # feet_contact_states = recording['contacts']
    # Prepare representations acting on proprioceptive and exteroceptive measurements.
    rep_kin_three = dyn_recordings.obs_representations['gait']    # Contact state is a 4D vector.
    rep_grf = dyn_recordings.obs_representations['ref_feet_pos']  #

    q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
    traj_q = generate_dof_motions(robot, angle_sweep=cfg.robot.angle_sweep * 2)
    traj_q_js = traj_q[:, 7:]
    # Add offset if needed

    # Go from basis of JS spawned by the generalized coordinates to the basis where isotypic components are separated.
    # rep_QJ = Q @ rep_QJ_iso @ Q_inv
    Q, Q_inv = rep_Q_js.change_of_basis, rep_Q_js.change_of_basis_inv
    # Separate JS trajectory into isotypic components.

    qj2iso, iso2qj = 'qj2iso', 'iso2qj'
    mode = qj2iso
    print(f"Mode changed to {mode}")
    g_idx = 0

    traj_q_iso = (Q_inv @ traj_q_js.T).T  # if mode == qj2iso else traj_q_js

    t, dt = 0, 1
    while True:
        if t >= len(traj_q_js):
            t = 0
        # for q_js, q_iso in tqdm(zip(traj_q_js, traj_q_iso), total=len(traj_q_js), desc="playback"):
        q_js, q_iso = traj_q_js[t], traj_q_iso[t]
        g = G.elements[g_idx]

        # Apply selected symmetry action
        q, dq = robot.get_state()
        g_q_js = np.real(rep_Q_js(g) @ q_js)
        g_q = np.concatenate((q[:7], g_q_js)).astype(float)
        robot.reset_state(g_q, dq)

        components_q_js = []
        for iso_robot, (re_irrep, dims) in zip(iso_robots, iso_comp.items()):
            q, dq = iso_robot.get_state()
            # Get point in isotypic component and describe it in the basis of generalized coordinates.
            q_iso_masked = q_iso * dims
            # Transform back to generalized coordinates.
            q_js_comp = np.real(Q @ q_iso_masked)
            components_q_js.append(q_js_comp)
            # Apply selected symmetry action
            g_q_js_comp = np.real(rep_Q_js(g) @ q_js_comp)
            # Set the robot to desired state.
            g_q = np.concatenate((q[:7], g_q_js_comp))
            iso_robot.reset_state(g_q, dq)

        # Get real robot generalized positions.
        q_iso_rec = sum(components_q_js)
        if mode == qj2iso:
            rec_error = q_js - q_iso_rec
            assert np.allclose(np.abs(rec_error), 0), f"Reconstruction error {rec_error}"
        elif mode == iso2qj:
            q_js = q_iso_rec
        else:
            raise NotImplementedError()

        t += dt
        time.sleep(0.05)

        # Process new keyboard commands.
        if new_command:
            keys = new_command.copy()
            new_command.clear()
            if keys == ['t']:
                dt = 1 if dt == 0 else 0
            if keys == ['m']:
                mode = qj2iso if mode == iso2qj else iso2qj
                print(f"Mode changed to {mode}")
        if num_pressed:
            if num_pressed[0] < G.order():
                g_idx = num_pressed[0]
                print(f"Group element selected {G.elements[g_idx]}")
            else:
                print(f"Group element {num_pressed[0]} is larger than group order...ignoring")
            num_pressed.clear()

    pb.disconnect()


if __name__ == '__main__':
    main()
