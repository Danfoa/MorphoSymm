import time

import numpy as np
from omegaconf import OmegaConf
from pynput import keyboard
from scipy.spatial.transform import Rotation
from utils.pybullet_visual_utils import change_robot_appearance, spawn_robot_instances
from utils.robot_utils import load_symmetric_system

from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper
from morpho_symm.utils.abstract_harmonics_analysis import isotypic_decomp_representation
from morpho_symm.utils.pybullet_visual_utils import configure_bullet_simulation


def on_key_press(key):
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
    q0, _ = robot.get_init_config(random=False)
    phases = np.random.uniform(0, 2 * np.pi, robot.nq)[..., None]
    amplitudes = np.random.uniform(0, angle_sweep, robot.nq)[..., None]
    max_period, min_period = 6, 3
    periods = np.random.randint(min_period, max_period, robot.nq)[..., None]
    q_period = np.lcm.reduce(periods).item()
    t = np.linspace(0, q_period, q_period * 10)[None, ...]

    s_t = (np.sin(2 * np.pi * (1 / periods) * t + phases) * amplitudes).T
    q = np.zeros((t.size, robot.nq - 1))
    q[:, :3] = s_t[..., :3] * 0.2 / s_t[..., :3].max()
    q[:, 3:6] = s_t[..., 3:6] * (30 * np.pi / 180) / s_t[..., 3:6].max()
    q[:, 7:] = s_t[..., 7:-1]
    return q[:, : robot.nq]  # We return orientation in euler angles


def coord_max2min(q):
    pos = q[:3]
    ori = Rotation.from_quat(q[3:7]).as_euler("xyz")
    return np.concatenate([pos, ori, q[7:]])


def coord_min2max(q):
    pos = q[:3]
    ori = Rotation.from_euler("xyz", q[3:6]).as_quat()
    return np.concatenate([pos, ori, q[6:]])


if __name__ == "__main__":
    """
    This example script shows how to apply the isotypic decomposition of a floating-base robot robot's configuration's
    space Q. The space is assumed to be composed of a MINIMAL set of generalized coordinates, defining a one-to-one map
    to the robot's degrees of freedom. Therefore we assume Q = [base_pos, base_ori, q_js] where:
        - base_pos: The base position (3,)
        - base_ori: The base orientation (3,) in Euler angles xyz (roll, pitch, yaw)
        - q_js: The joint space positions (n_dof,)

    Note: To select a different robot pass the robot name as an argument to the script.
    ```
    python robot_harmonic_decomposition_mini_cheetah.py robot=mini_cheetah  # solo, solo-4, atlas, baxter, anymal_c
    ```
    
    Note: Pinocchio defines the position configuration generalized coordinates in maximal/redundant coordinates.
    For instance the base orientation is defined in quaternions and positions of revolute unbounded joints are defined
    as points in the unit circle theta -> [cos(theta), sin(theta)]. For the isotypic decomposition we are required to
    work on minimal coordinates. So be mindful that the representation defined above is not the one acting on
    pinocchio's `q` vector.
    
    """
    # Get the command line arguments, the only expected is robot=<robot_name>
    conf = OmegaConf.from_cli()
    robot_name = conf.get("robot", "atlas")  # Default robot is mini_cheetah

    # Load the robot instance its symmetry group G ____________________________________________________________________
    robot, G = load_symmetric_system(robot_name=robot_name)
    # Now we obtain the group representations required to define the representation of the configuration space Q.
    rep_R3 = G.representations["R3"]
    rep_euler_xyz = G.representations["euler_xyz"]
    rep_Q_js = G.representations["TqQ_js"]

    # Define the group representation of the configuration space Q. ___________________________________________________
    # rep_Q := T @ (rep_Q_iso_1 + ..., rep_Q_iso_k) @ T_inv,                 T: Q -> Q
    rep_Q = isotypic_decomp_representation(rep_R3 + rep_euler_xyz + rep_Q_js)
    # Get the matrix change of basis mapping between the isotypic basis and the canonical basis of Q.
    T = rep_Q.change_of_basis  # T: Q -> Q  from isotypic basis to canonical basis
    T_inv = rep_Q.change_of_basis_inv  # T_inv: Q -> Q from canonical basis to isotypic basis
    num_iso_components = len(rep_Q.attributes["isotypic_reps"])

    # Define the orthogonal projectors of the state position to each isotypic subspace. _______________________________
    # The process of projecting the state into its isotypic components is done by applying a change to the isotypic
    # basis, isolating the values associated to the isotypic subspace and then returning to the canonical basis.
    # P_i = T @ diag(mask_iso_i) @ T_inv, such that q^(iso_i) = P_1 @ q
    iso_projectors = {}
    tmp_dim = 0
    for iso_id, iso_rep in rep_Q.attributes["isotypic_reps"].items():
        mask = np.zeros(rep_Q.size)
        mask[tmp_dim : tmp_dim + iso_rep.size] = 1
        iso_projectors[iso_id] = T @ np.diag(mask) @ T_inv
        print(f"Isotypic component {iso_id} has dimension {iso_rep.size}")
        tmp_dim += iso_rep.size
    assert rep_Q.size == robot.nv, f"Conf space rep dimension {rep_Q.size} != |Q|:={robot.nv}"

    # Illustrate how the mass matrix at any random configuration q can be block-diagonalized by change of basis T_inv.
    q0, dq0 = robot.get_init_config(random=True, angle_sweep=20 * np.pi / 180)

    # Ensure the robot's Joint-Space mass matrix is a G-equivariant matrix-valued function M(g * q) = g * M(q) * g^T
    for g in G.elements:
        if g == G.identity:
            continue
        Mq0 = robot.pinocchio_robot.mass(q0)
        g_q0 = coord_min2max(rep_Q(g) @ coord_max2min(q0))
        Mg_q0 = robot.pinocchio_robot.mass(g_q0)
        Mg_g0_pred = rep_Q(g) @ Mq0 @ rep_Q(~g)
        Mg_q0_iso = T @ Mg_q0 @ T_inv
        rep_Q_g_iso = T_inv @ rep_Q(g) @ T
        # Ignore the world position coordinates.
        assert np.allclose(Mg_q0[:3, :3], Mg_g0_pred[:3, :3], atol=1e-6, rtol=1e-6)

    # Configure visualization animation _______________________________________________________________________________
    pb = configure_bullet_simulation(gui=True)
    robot.configure_bullet_simulation(pb)
    change_robot_appearance(pb, robot)

    base_pos, base_ori_quat_xyzw = q0[:3], q0[3:7]
    robot.reset_state(q0, dq0)

    # For each isotypic component we spawn a robot instance in order to visualize the effect of the decomposition
    base_positions = np.asarray([base_pos] * num_iso_components)
    base_positions[:, 0] = -4.0 * robot.hip_height
    base_positions[:, 1] = np.linspace(0, 2.8 * robot.hip_height * num_iso_components, num_iso_components)
    base_positions[:, 1] -= np.max(base_positions[:, 1]) / 2
    # base_positions[:, 2] += 0.5

    iso_robots = spawn_robot_instances(
        robot,
        bullet_client=pb,
        base_positions=base_positions,
        base_orientations=[base_ori_quat_xyzw] * num_iso_components,
        tint=True,
        alpha=0.5,
    )
    # _________________________________________________________________________________________________________________

    # Define the state trajectories that will be used to visualize the isotypic decomposition. ________________________
    traj_q = generate_dof_motions(robot, angle_sweep=90 * np.pi / 180)
    time_horizon = traj_q.shape[0]
    # Project the trajectories into the isotypic components.
    traj_q_iso = {}
    for iso_id, iso_rep in rep_Q.attributes["isotypic_reps"].items():
        traj_q_iso[iso_id] = np.einsum("ij, tj -> ti", iso_projectors[iso_id], traj_q)
    # Check that q(t) := sum_i^n_iso q^(iso_i)(t)
    rec_traj_q_iso = np.sum([traj_q_iso[iso_id] for iso_id in rep_Q.attributes["isotypic_reps"].keys()], axis=0)
    assert np.allclose(rec_traj_q_iso, traj_q, atol=1e-6, rtol=1e-6), (
        f"Reconstruction error {np.linalg.norm(rec_traj_q_iso - traj_q)}"
    )

    # Visualize the robot state trajectory and its projection to each isotypic subspace _______________________________
    t_idx = 0
    fps = 30
    timescale = 0.5
    time_shift = 1
    frames = []
    while True:
        # Visualize the robot configuration
        traj_q_js = traj_q[t_idx, 6:]
        base_ori = Rotation.from_euler("xyz", traj_q[t_idx, 3:6]).as_quat()
        robot.reset_state(q=np.concatenate([base_pos + traj_q[t_idx, :3], base_ori, traj_q_js]), v=np.zeros(robot.nv))

        # Visualize the projections of the position configuration into the isotypic components.
        for n, iso_robot, iso_id in zip(range(len(iso_robots)), iso_robots, traj_q_iso.keys()):
            traj_q_js_iso_i = traj_q_iso[iso_id][t_idx, 6:]
            base_ori_iso_i = Rotation.from_euler("xyz", traj_q_iso[iso_id][t_idx, 3:6]).as_quat()
            iso_robot.reset_state(
                q=np.concatenate([base_positions[n] + traj_q_iso[iso_id][t_idx, :3], base_ori_iso_i, traj_q_js_iso_i]),
                v=np.zeros(robot.nv),
            )

        # Process new keyboard commands.
        if new_command:
            keys = new_command.copy()
            new_command.clear()
            if keys == ["t"]:  # Pauses/Resumes the replay
                time_shift = 1 if time_shift == 0 else 0

        # Update the time index
        t_idx += time_shift
        if t_idx == time_horizon:
            t_idx = 0
        time.sleep(1 / fps / timescale)
    pb.disconnect()
