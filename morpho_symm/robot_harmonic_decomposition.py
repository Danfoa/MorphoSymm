import time

import hydra
import numpy as np
from escnn.group import Group
from omegaconf import DictConfig
from pynput import keyboard
from utils.pybullet_visual_utils import change_robot_appearance, spawn_robot_instances
from utils.robot_utils import load_robot_and_symmetries

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


def generate_random_dof_motions(robot: PinBulletWrapper, angle_sweep=0.5):
    """TODO: In construction."""
    n_dof = robot.nq - 7
    q, dq = robot.get_init_config(random=False)
    phases = np.random.uniform(0, 2 * np.pi, n_dof)[..., None]
    amplitudes = np.random.uniform(0, angle_sweep, n_dof)[..., None]
    max_period, min_period = 6, 3
    periods = np.random.randint(min_period, max_period, n_dof)[..., None]
    q_period = np.lcm.reduce(periods).item()
    t = np.linspace(0, q_period, q_period * 20)[None, ...]

    q_js = q[7:, None] + np.sin(2 * np.pi * (1/periods) * t + phases) * amplitudes

    return q_js


@hydra.main(config_path='cfg/supervised', config_name='config_visualization', version_base='1.1')
def main(cfg: DictConfig):
    """Visualize the effect of DMSs transformations in 3D animation.

    This script visualizes the DMSs transformations on robot state and on proprioceptive and exteroceptive measurements.
    """
    cfg.robot.seed = cfg.robot.seed if cfg.robot.seed >= 0 else np.random.randint(0, 1000)
    np.random.seed(cfg.robot.seed)
    # Get robot instance, along with representations of the symmetry group on the Euclidean space (in which the robot
    # base B evolves in) and Joint Space (in which the internal configuration of the robot evolves in).
    robot, symmetry_space = load_robot_and_symmetries(robot_cfg=cfg.robot, debug=cfg.debug)

    G = symmetry_space.fibergroup
    assert isinstance(G, Group)
    assert 'QJ' in G.representations and 'Ed' in G.representations, "Missing QJ and Ed representations."
    rep_QJ = G.representations['QJ']

    # Load main robot in pybullet.
    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    robot.configure_bullet_simulation(pb)
    n_dof = robot.nq - 7
    if cfg.robot.tint_bodies: change_robot_appearance(pb, robot)
    q0, dq0 = robot.get_init_config(random=True, angle_sweep=cfg.robot.angle_sweep, fix_base=cfg.robot.fix_base)
    base_pos = q0[:3]

    # Determine the number of isotypic components of the Joint-Space (JS) vector space.
    # This is equivalent to the number of unique irreps of the JS representation.
    iso_comp = {}  # TODO: Make a class for a Component.
    mask = []
    for re_irrep_id in rep_QJ.irreps:
        mask.extend([re_irrep_id] * G.irrep(*re_irrep_id).size)
    for re_irrep_id in rep_QJ.irreps:
        re_irrep = G.irrep(*re_irrep_id)
        dims = np.zeros(n_dof)
        dims[[i for i, x in enumerate(mask) if x == re_irrep_id]] = 1
        iso_comp[re_irrep] = dims
        # print(f"Re irrep: {re_irrep} - Trivial: {re_irrep.is_trivial()} - Mult: {multiplicities[idx]}")

    # For each isotypic component we spawn a robot instance in order to visualize the effect of the decomposition
    n_components = len(iso_comp)
    base_positions = np.asarray([base_pos] * n_components)
    base_positions[:, 0] = -1.0
    base_positions[:, 1] = np.linspace(0, 2 * robot.hip_height * n_components, n_components)
    base_positions[:, 1] -= np.max(base_positions[:, 1]) / 2

    iso_robots = spawn_robot_instances(
        robot, bullet_client=pb, base_positions=base_positions, tint=cfg.robot.tint_bodies, alpha=0.5,
        )

    # For the symmetries of the system some robots require centering of DoF domain.
    q0 = np.asarray(cfg.robot.init_q) if cfg.robot.init_q is not None else q0

    # Generate random DoF motions.
    traj_q_js = generate_random_dof_motions(robot, angle_sweep=cfg.robot.angle_sweep * 2)
    # Add offset if needed

    # Go from basis of JS spawned by the generalized coordinates to the basis where isotypic components are separated.
    Q, Q_inv = rep_QJ.change_of_basis, rep_QJ.change_of_basis_inv
    # Separate JS trajectory into isotypic components.

    qj2iso, iso2qj = 'qj2iso', 'iso2qj'
    mode = qj2iso
    print(f"Mode changed to {mode}")
    t, dt = 0, 1
    g_idx = 0
    while True:
        g = G.elements[g_idx]
        q_iso_traj = Q_inv @ traj_q_js if mode == qj2iso else traj_q_js
        comp_q_js, comp_q_iso = [], []
        for iso_robot, (re_irrep, dims) in zip(iso_robots, iso_comp.items()):
            q, dq = iso_robot.get_state()
            # Get point in isotypic component and describe it in the basis of generalized coordinates.
            iso_q = q_iso_traj[:, t] * dims
            comp_q_iso.append(iso_q)
            # Transform back to generalized coordinates.
            iso_q_js = np.real(Q @ iso_q)
            comp_q_js.append(iso_q_js)
            # Apply selected symmetry action
            g_iso_q_js = np.real(rep_QJ(g) @ iso_q_js)
            # Set the robot to desired state.
            q = np.concatenate((q[:7], g_iso_q_js))
            iso_robot.reset_state(q, dq)

        comp_energy = np.abs(comp_q_js)
        comp_energy /= np.linalg.norm(comp_energy, axis=0, keepdims=True)

        for comp_id, (iso_robot, energy) in enumerate(zip(iso_robots, comp_energy)):
            change_robot_appearance(pb, iso_robot, change_color=False, alpha=energy[comp_id])

        # Get real robot generalized positions.
        rec_q_js = np.sum(comp_q_js, axis=0)
        q, dq = robot.get_state()
        if mode == qj2iso:
            q_js = traj_q_js[:, t]
            rec_error = q_js - rec_q_js
            assert np.allclose(np.abs(rec_error), 0), f"Reconstruction error {rec_error}"
        elif mode == iso2qj:
            q_js = rec_q_js
        else:
            raise NotImplementedError()

        # Apply selected symmetry action
        g_q_js = np.real(rep_QJ(g) @ q_js)
        q = np.concatenate((q[:7], g_q_js))
        robot.reset_state(q, dq)

        time.sleep(0.01)

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

        t += dt
        t = t % traj_q_js.shape[1]
    pb.disconnect()


if __name__ == '__main__':
    main()
