import logging
import time
from pathlib import Path

import hydra
import numpy as np
import scipy
from escnn.group import Group, Representation
from omegaconf import DictConfig
from pynput import keyboard
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import morpho_symm
from morpho_symm.data.DynamicsRecording import DynamicsRecording
from morpho_symm.utils.algebra_utils import matrix_to_quat_xyzw, permutation_matrix
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
from morpho_symm.utils.pybullet_visual_utils import configure_bullet_simulation
from utils.pybullet_visual_utils import (draw_vector, draw_plane, render_camera_trajectory,
                                         spawn_robot_instances)
from utils.robot_utils import load_symmetric_system

log = logging.getLogger(__name__)


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


def get_kinematic_three_rep(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(4, dtype=int)}
    gens = [permutation_matrix([1, 0, 3, 2]), permutation_matrix([2, 3, 0, 1]), permutation_matrix([0, 1, 2, 3])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)

    return rep_kin_three

def get_ground_reaction_forces_rep(G: Group, rep_kin_three: Representation):

    rep_R3 = G.representations['Rd']
    rep_F = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    return rep_F



def load_mini_cheetah_trajs(data_path: Path):
    """Load the mini-cheetah trajectories from the .mat files.

    The Umich dataset contains in the following order [RF, LF, RH, LH]:
    - q: joint encoder value (num0_data,12)
    - qd: joint angular velocity (num_data,12)
    - p: foot position from FK (num_data,12)
    - v: foot velocity from FK (num_data,12).

    - imu_acc: linear acceleration from imu (num_data,3)
    - imu_omega: angular velocity from imu (num_data,3)
    - contacts: contact data (num_data,4)
    - tau_est (optional): estimated control torque (num_data,12)
    - F (optional): ground reaction force
    Args:
        data_path (Path): Path to the .mat files.

    Returns:
        recordings (dict): Dictionary of recording name and data.
    """
    assert data_path.exists(), f"Path {data_path} does not exist."
    recordings = {}
    for data_name in data_path.glob('*.mat'):
        raw_data = scipy.io.loadmat(data_name)
        recordings[data_name.stem] = raw_data
    return recordings


def update_contacts(pb, feet_pos, prev_contact_state, contact_state, planes=None):
    out_of_view_pos = [0, 0, 100]

    if planes is None:
        planes = []
        for r in feet_pos:
            # Draw a dark purple plane (R, r) in the world frame.
            planes.append(draw_plane(pb, np.eye(3), out_of_view_pos,
                                     color=(0.9, 0, 0.9, 0.5), cylinder=True, size=(0.05, 0.005)))

    state_change = contact_state != prev_contact_state
    idx = np.where(state_change)[0]
    for i in idx:
        plane_pos = feet_pos[i] if contact_state[i] else out_of_view_pos
        pb.resetBasePositionAndOrientation(planes[i], plane_pos, np.array([0, 0, 0, 1]))

    stable_contact = contact_state == prev_contact_state
    idx = np.where(np.logical_and(stable_contact, contact_state))[0]
    for i in idx:
        plane_pos = feet_pos[i]
        pb.resetBasePositionAndOrientation(planes[i], plane_pos, np.array([0, 0, 0, 1]))

    return planes


def update_heading(pb, X_B, heading_arrow=None):
    pos = (X_B @ np.array([-0.09, 0, 0.045, 1]))[:3]
    if heading_arrow is None:
        heading_arrow = draw_plane(pb, X_B[:3, :3], pos, color=(0.504, 0.931, 0.970, 1.0),
                                   size=(0.01, 0.04, 0.01))
    else:
        pb.resetBasePositionAndOrientation(heading_arrow, pos, matrix_to_quat_xyzw(X_B[:3, :3]))
    return heading_arrow


def update_ground_reaction_forces(pb, feet_pos, forces, contact_state, vectors=None):
    vectors = [None] * 4 if vectors is None else vectors

    feet_in_contact = np.where(np.isclose(contact_state, 1))[0]
    for idx in feet_in_contact:
        # Remove old vector if present
        if vectors[idx] is not None:
            pb.removeBody(vectors[idx])
        # Draw a light purple vector representing the force in world frame.
        r_w, f_w = feet_pos[idx], forces[idx]
        force_vect = draw_vector(pb, origin=r_w, vector=f_w, v_color=(0.5, 0, 0.5, 0.5), scale=0.005)
        vectors[idx] = force_vect

    # Remove forces of legs not in contact
    feet_not_in_contact = np.where(np.isclose(contact_state, 0))[0]
    for idx in feet_not_in_contact:
        if vectors[idx] is not None:
            pb.removeBody(vectors[idx])
            vectors[idx] = None

    return vectors


@hydra.main(config_path='cfg', config_name='config_visualization', version_base='1.1')
def main(cfg: DictConfig):
    """Visualize the effect of DMSs transformations in 3D animation.

    This script visualizes the DMSs transformations on robot state and on proprioceptive and exteroceptive measurements.
    """
    cfg.robot.seed = cfg.robot.seed if cfg.robot.seed >= 0 else np.random.randint(0, 1000)
    np.random.seed(cfg.robot.seed)

    if 'mini_cheetah' not in cfg.robot.name:
        raise NotImplementedError("For the moment we have only real-world data from Mini-Cheetah.")

    # Get robot instance, along with representations of the symmetry group on the Euclidean space (in which the robot
    # base B evolves in) and Joint Space (in which the internal configuration of the robot evolves in).
    robot, G = load_symmetric_system(robot_cfg=cfg.robot, debug=cfg.debug)

    rep_Q_js = G.representations['Q_js']
    # rep_TqJ = G.representations['TqQ_js']
    rep_E3 = G.representations['Ed']
    rep_R3 = G.representations['Rd']

    offset = max(0.2, 1.8 * robot.hip_height)
    base_pos = np.array([-offset if G.order() != 2 else 0, -offset] + [robot.hip_height * 5.5])
    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    # Define the positions to spawn the |G| instances of the robot to represent the |G| symmetric states.
    X_B = np.eye(4)
    X_B[:3, 3] = base_pos
    orbit_X_B = [rep_E3(g) @ X_B @ np.linalg.inv(rep_E3(g)) for g in G.elements]
    robots = spawn_robot_instances(
        robot, bullet_client=pb, base_positions=[X[:3, 3] for X in orbit_X_B], tint=cfg.robot.tint_bodies,
        )
    robot = robots[0]
    end_effectors = robot.bullet_ids_allowed_floor_contacts

    # Load a trajectory of motion and measurements from the mini-cheetah robot
    recordings_path = Path(
        morpho_symm.__file__).parent / 'data/mini_cheetah/raysim_recordings/flat/forward_minus_0_4/n_trajs=1-frames=7771-train.pkl'
    dyn_recordings = DynamicsRecording.load_from_file(recordings_path)
    # Load and prepare data for visualization
    q_js_t = dyn_recordings.recordings['joint_pos']     # Joint space positions
    v_js_t = dyn_recordings.recordings['joint_vel']     # Joint space velocities
    base_ori_t = dyn_recordings.recordings['base_ori'][0]  # Base orientation
    feet_pos = dyn_recordings.recordings['feet_pos']  # Feet positions  [x,y,z] w.r.t base frame
    # ground_reaction_forces = recording['F']
    # feet_contact_states = recording['contacts']
    # Prepare representations acting on proprioceptive and exteroceptive measurements.
    rep_kin_three = dyn_recordings.obs_representations['gait']    # Contact state is a 4D vector.
    rep_grf = dyn_recordings.obs_representations['ref_feet_pos']  #

    q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))

    dt = dyn_recordings.dynamics_parameters['dt']
    timestamps = np.linspace(0, dt * q_js_t.shape[0], q_js_t.shape[0])
    frames = []
    robot_frames = [[] for _ in range(G.order())]
    symmetry_frames = []
    robot_terrain, robot_grf_vects = [None for _ in range(G.order())], [None for _ in range(G.order())]
    robot_heading = [None for _ in range(G.order())]
    prev_contact_state = np.asarray([0, 0, 0, 0], dtype=np.uint8)

    fps = 30
    prev_t = timestamps[0]
    t0 = timestamps[0]
    run_time0 = time.time()

    camera_view_point = Rotation.from_euler("xyz", [0, -25, 90], True)
    # rpy = camera_view_point.as_euler("xyz") * 180/np.pi
    for i, t in tqdm(enumerate(timestamps), total=len(timestamps), desc=f"Capturing frames from {cfg.robot.name}"):
        # if i < 21000 or i > 25000:
        #     continue
        # for i, t in tqdm(enumerate(timestamps[:int(len(timestamps) // 2)]), total = len(timestamps), desc=f"Capturing frames from {cfg.robot.name}"):
        run_time = time.time()
        if cfg.gui:
            if run_time - run_time0 < t - t0:
                time.sleep(((t - t0) - (run_time - run_time0))/2)
                continue
        else:
            if t - prev_t < (1 / fps):
                continue
        prev_t = t

        # Define the recording base configuration.
        X_B[:3, :3] = Rotation.from_euler("xyz", base_ori_t[i] - np.array([0, 0, 0*np.pi/2])).as_matrix()
        X_B[:3, 3] = base_pos
        # contact_state = feet_contact_states[i]
        orbit_X_B, orbit_cam_target = {}, {}
        q_rec = np.concatenate([X_B[:3, 3], matrix_to_quat_xyzw(X_B[:3, :3]), q_js_t[i]])
        v_rec = np.concatenate([np.zeros(6), v_js_t[i]])
        # Just for Mit Cheetah.
        q = q_rec
        feet_pos_fk = feet_pos[i]
        a = feet_pos_fk[:3]
        for robot_idx, g in enumerate(G.elements):
            g_X_B = np.real(rep_E3(g) @ X_B @ rep_E3(~g))
            orbit_X_B[g] = g_X_B
            g_q_js = np.real(rep_Q_js(g) @ q[7:])
            g_q = np.concatenate([g_X_B[:3, 3], matrix_to_quat_xyzw(g_X_B[:3, :3]), g_q_js]).astype(float)

            # Set the robot configuration to symmetric state.
            robots[robot_idx].reset_state(g_q, np.zeros(robot.nv))

            # Get the position of the 4 feet of the robot in the world frame [RF, LF, RH, LH].
            # g_rf_w = [pb.getLinkState(robots[robot_idx].robot_id, name)[0] for name in end_effectors]

            # Display contact terrain
            # g_contact_state = np.rint(np.real(rep_kin_three(g) @ contact_state)).astype(np.uint8)
            # g_prev_contact_state = np.rint(np.real(rep_kin_three(g) @ prev_contact_state)).astype(np.uint8)
            # robot_terrain[robot_idx] = update_contacts(pb,
            #                                            g_rf_w,
            #                                            g_prev_contact_state,
            #                                            g_contact_state,
            #                                            planes=robot_terrain[robot_idx])
            robot_heading[robot_idx] = update_heading(pb, g_X_B, heading_arrow=robot_heading[robot_idx])
            # Display ground reaction forces
            if not cfg.gui:
                # If in contacts draw the vect representing the ground reaction forces.
                # grf = ground_reaction_forces[i]
                # g_grf = np.real((rep_F(g) @ grf))
                # g_forces = np.split(g_grf, 4)

                # robot_grf_vects[robot_idx] = update_ground_reaction_forces(pb,
                #                                                            g_rf_w,
                #                                                            g_forces,
                #                                                            g_contact_state,
                #                                                            vectors=robot_grf_vects[robot_idx])

                g_camera_view_point = np.real(rep_R3(g) @ camera_view_point.as_matrix() @ np.linalg.inv(rep_R3(g)))
                roll, pitch, yaw = Rotation.from_matrix(g_camera_view_point).as_euler("xyz") * 180/np.pi
                cam_target = orbit_X_B[g][:3, 3]
                robot_frames[robot_idx].append(render_camera_trajectory(pb,
                                                                        roll=[roll],
                                                                        pitch=[-10 if robot_idx in [0, 1, 4, 5] else 20],
                                                                        yaw=[yaw + 90],
                                                                        render_width=512, render_height=512,
                                                                        n_frames=1, cam_distance=robot.hip_height * 3.0,
                                                                        farPlane=robot.hip_height * 5.0,
                                                                        cam_target_pose=cam_target
                                                                        )
                                               )

        if not cfg.gui:
            symmetry_frame = np.concatenate([robot_frames[robot_idx][-1] for robot_idx in range(G.order())], axis=2)
            symmetry_frames.append(symmetry_frame.squeeze())

            # frames.extend(render_camera_trajectory(pb, roll=[0], pitch=[-25], yaw=[0], n_frames=1,
            #                                        render_width=1000, render_height=500,
            #                                        cam_distance=robot.hip_height * 5.0,
            #                                        cam_distance=robot.hip_height * 3.5,
            #                                        cam_target_pose=[0, 0, robot.hip_height]))
        # prev_contact_state = contact_state


    frames = symmetry_frames

    if len(frames) > 0:
        from moviepy.editor import ImageSequenceClip
        # Save animation
        root_path = Path(morpho_symm.__file__).parents[1].absolute()
        save_path = root_path / "paper/tests"
        file_name = f"mini-cheetah_{G}-{recording_name}_dynamic_symmetries"
        file_path = save_path / f'{file_name}.gif'
        file_count = 1
        while file_path.exists():
            file_path = save_path / f'{file_name}({file_count}).gif'
            file_count += 1
        clip = ImageSequenceClip(list(frames), fps=fps//2)
        clip.write_gif(file_path, fps=fps//2, loop=0, fuzz=0.9)
        print(f"Animation with {len(frames)} ({len(frames)/fps//2}[s]) saved to {file_path.absolute()}")


    pb.disconnect()


if __name__ == '__main__':
    main()
