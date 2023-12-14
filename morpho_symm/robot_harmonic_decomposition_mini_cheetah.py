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
from morpho_symm.robot_symmetry_visualization_dynamic import load_mini_cheetah_trajs
from morpho_symm.utils.algebra_utils import matrix_to_quat_xyzw
from utils.pybullet_visual_utils import change_robot_appearance, spawn_robot_instances, render_camera_trajectory
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


def generate_dof_motions(robot: PinBulletWrapper, angle_sweep=0.5, recording_name = 'forest'):
    """TODO: In construction."""
    if robot.robot_name == 'mini_cheetah':
        # / home / danfoa / Projects / MorphoSymm / morpho_symm / data / contact_dataset / training_splitted / mat_test / forest.mat
        recordings_path = Path(morpho_symm.__file__).parent / 'data/contact_dataset/training_splitted/mat_test'
        recordings = load_mini_cheetah_trajs(recordings_path)
        recording = recordings[recording_name]
        # timestamps = recording['control_time'].flatten()
        q_js_t = recording['q']  # Joint space positions
        # v_js_t = recording['qd']  # Joint space velocities
        base_ori_t = recording['imu_rpy']
        # ground_reaction_forces = recording['F']
        # base_acc = recording['imu_acc']
        # base_ang_vel = recording['imu_omega']
        # feet_contact_states = recording['contacts']
        time = recording['control_time']
        q = []
        q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
        for q_js, base_ori in zip(q_js_t, base_ori_t):
            # Define the recording base configuration.
            q_rec = np.concatenate([q0[:7], -1 * q_js])
            v_rec = np.zeros(robot.nv)
            # Just for Mit Cheetah.
            q_t = q_rec + q0
            q.append(q_t)
        return np.asarray(q), time[0]
    else:
        n_dof = robot.nq - 7
        q0, _ = robot.get_init_config(False)
        phases = np.random.uniform(0, 2 * np.pi, n_dof)[..., None]
        amplitudes = np.random.uniform(0, angle_sweep, n_dof)[..., None]
        max_period, min_period = 6, 3
        periods = np.random.randint(min_period, max_period, n_dof)[..., None]
        q_period = np.lcm.reduce(periods).item()
        t = np.linspace(0, q_period, q_period * 10)[None, ...]
        q_js = np.sin(2 * np.pi * (1 / periods) * t + phases) * amplitudes
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
    rep_Q_js = G.representations['TqQ_js']
    rep_QJ_iso = Representation(G, name="TqQ_js_iso", irreps=rep_Q_js.irreps, change_of_basis=np.eye(rep_Q_js.size))
    # rep_TqJ = G.representations['TqQ_js']
    # rep_Ed = G.representations['Ed']

    # Load main robot in pybullet.
    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    robot.configure_bullet_simulation(pb)
    n_dof = robot.nv - 6
    if cfg.robot.tint_bodies: change_robot_appearance(pb, robot)
    q0, dq0 = robot.get_init_config(random=True, angle_sweep=cfg.robot.angle_sweep, fix_base=cfg.robot.fix_base)
    base_ori = q0[3:7]
    base_pos = q0[:3]
    base_oriR = Rotation.from_quat(base_ori)
    base_ori_rpg = Rotation.from_euler('xyz', base_oriR.as_euler(seq='xyz') - np.array([0, 0, np.pi/2]))
    base_ori = [ 0, 0, 0.7071068, 0.7071068 ]
    q0[3:7] = base_ori
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
    base_positions[:, 1] = np.linspace(0, 2.8 * robot.hip_height * n_components, n_components)
    base_positions[:, 1] -= np.max(base_positions[:, 1]) / 2
    # base_positions[:, 2] += 0.5

    iso_robots = spawn_robot_instances(
        robot,
        bullet_client=pb,
        base_positions=base_positions,
        base_orientations=[base_ori]*n_components,
        tint=cfg.robot.tint_bodies, alpha=.5,
        )

    # For the symmetries of the system some robots require centering of DoF domain.

    # Generate random DoF motions.
    recording_name = 'small_pebble'
    traj_q, time = generate_dof_motions(robot, recording_name=recording_name, angle_sweep=cfg.robot.angle_sweep * 2)
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

    t_idx = 0
    time_shift = 1
    fps = 30
    make_gif = True
    timescale = 1
    last_capture_time = time[0]
    frames = []
    while True:
        t_idx = t_idx + time_shift
        t = time[t_idx] * timescale
        dt = t - time[t_idx - 1]
        capture_frame = t - last_capture_time > 1 / fps
        if not capture_frame:
            continue
        # for q_js, q_iso in tqdm(zip(traj_q_js, traj_q_iso), total=len(traj_q_js), desc="playback"):
        q_js = traj_q_js[t_idx]
        q_iso = traj_q_iso[t_idx]
        g = G.elements[g_idx]

        # Apply selected symmetry action
        q, dq = robot.get_state()
        g_q_js = np.real(rep_Q_js(g) @ q_js)
        g_q = np.concatenate((q[:7], g_q_js)).astype(float)
        robot.reset_state(*robot.sim2pin(g_q, dq))

        components_q_js = []
        for iso_robot, (re_irrep, dims) in zip(iso_robots, iso_comp.items()):
            q, dq = iso_robot.get_state()
            # Get point in isotypic component and describe it in the basis of generalized coordinates.
            q_iso_masked = q_iso * dims
            # Transform back to generalized coordinates.
            q_js_comp = np.real(Q @ q_iso_masked)
            a = np.linalg.norm(q_js_comp[2:4])
            components_q_js.append(q_js_comp)
            # Apply selected symmetry action
            g_q_js_comp = np.real(rep_Q_js(g) @ q_js_comp)
            # Set the robot to desired state.
            g_q = np.concatenate((q[:7], g_q_js_comp))
            iso_robot.reset_state(*iso_robot.sim2pin(g_q, dq))

        # Get real robot generalized positions.
        q_iso_rec = sum(components_q_js)
        if mode == qj2iso:
            rec_error = q_js - q_iso_rec
            assert np.allclose(np.abs(rec_error), 0), f"Reconstruction error {rec_error}"
        elif mode == iso2qj:
            q_js = q_iso_rec
        else:
            raise NotImplementedError()



        if make_gif and capture_frame:
            init_roll_pitch_yaw = ([0], [-30], [90])
            roll, pitch, yaw = init_roll_pitch_yaw
            light_distance, light_directions = 4, (0.5, 0.5, 1)
            cam_distance, cam_target_pose = 0.7, base_pos
            aspect_ratio = 3.5
            width = 1024
            height = int(width / aspect_ratio)
            frame = render_camera_trajectory(pb, pitch, roll, yaw, 1, cam_distance, cam_target_pose,
                                             render_width=width, render_height=height,
                                             light_direction=light_directions, light_distance=light_distance,
                                             )[0]
            frames.append(frame)
            last_capture_time = t
            print(f"Frame {len(frames)} captured at time {t} [s]")

        # Process new keyboard commands.
        if new_command:
            keys = new_command.copy()
            new_command.clear()
            if keys == ['t']:
                time_shift = 1 if time_shift == 0 else 0
            if keys == ['p']:
                break
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

        if t_idx == len(traj_q_js) - 1:
            t_idx = 0
        if t > 4:
            break
    pb.disconnect()

    if len(frames) > 0:
        from moviepy.editor import ImageSequenceClip
        # Save animation
        root_path = Path(morpho_symm.__file__).parents[1].absolute()
        save_path = root_path / "docs/static/dynamic_animations/dynamics_harmonics_analyis"
        file_name = f"mini-cheetah_{G}-{recording_name}_harmonic_analysis"
        file_path = save_path / f'{file_name}.gif'
        file_count = 1
        while file_path.exists():
            file_path = save_path / f'{file_name}({file_count}).gif'
            file_count += 1
        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_gif(file_path, fps=fps, loop=0, fuzz=0.9)
        print(f"Animation with {len(frames)} ({len(frames)/fps}[s]) saved to {file_path.absolute()}")



if __name__ == '__main__':
    main()
