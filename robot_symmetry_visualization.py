import os
import pathlib
import time

import escnn.group
import hydra
import numpy as np
from omegaconf import DictConfig
from pytransform3d import transformations as tr

from utils.pybullet_visual_utils import render_orbiting_animation, tint_robot, \
    display_robots_and_vectors, get_mock_ground_reaction_forces
from utils.robot_utils import load_robot_and_symmetries
from utils.algebra_utils import configure_bullet_simulation, quat_xyzw_to_SO3

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@hydra.main(config_path='cfg/supervised', config_name='config_visualization', version_base='1.1')
def main(cfg: DictConfig):
    np.random.seed(100 if not 'seed' in cfg.robot else cfg.robot.seed)
    # Get robot instance, along with representations of the symmetry group on the Euclidean space (in which the robot
    # base B evolves in) and Joint Space (in which the internal configuration of the robot evolves in).
    robot, symmetry_space = load_robot_and_symmetries(robot_cfg=cfg.robot, debug=cfg.debug)

    G = symmetry_space.fibergroup
    assert isinstance(G, escnn.group.Group)
    assert 'QJ' in G.representations and 'Ed' in G.representations, "We need these two reps to do visualizations"
    rep_QJ = G.representations['QJ']
    rep_Ed = G.representations['Ed']

    # Configuration of the robot. Check cfg/robot/
    robot_cfg = cfg.robot
    # Configuration of the 3D visualization -------------------------------------------------------------------------
    offset = 1.8 * robot.hip_height

    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    robot.configure_bullet_simulation(pb, world=None)
    if cfg.robot.tint_bodies:
        tint_robot(pb, robot)

    # Get initial random configuration of the system
    q, dq = robot.get_init_config(random=True, angle_sweep=robot_cfg.angle_sweep)
    q_offset = np.zeros_like(q) if robot_cfg.offset_q is None else np.array([eval(str(s)) for s in robot_cfg.offset_q])
    q = q + q_offset
    rB0 = np.array([-offset if G.order() != 2 else 0, -offset] + [robot.hip_height * 1.5])
    q[:3] = rB0
    robot.reset_state(q - q_offset, dq)
    # robot.reset_state(q + q_offset, dq)
    # --------------------------------------------------------------------------------------------------------------

    # NOTATION:
    # We use the following code notation which tries to replicate as much as possible the notation of the paper:
    # [Variable][Frame]_[reference frame of variable] e.g, RB_w : Rotation (R) of Base (B) in World (w) coordinates.
    # - `Ed`: Euclidean space of d dimensions
    # - `QJ`: Joint Space of the robot
    # - `TQJ`: Tangent space to the joint space.
    # - `r`: Vector in `Ed` describing the position of a frame of the robot bodies. e.g. rB: Robot base position
    # - `R`: SO3 matrix describing a rotation in E3. e.g. RB: Robot base orientation
    # - `T`: Homogenous Matrix 4x4 transformation holding a rotation matrix and a translation T = [[R , r], [0, 1]]

    # Specify the configuration of the original robot base
    # Keep a "null" notation for computation of hb in Base reference frame.
    rB_w = np.array(q[:3])  # Base position in R^3
    RB_w = quat_xyzw_to_SO3(q[3:7])  # Base Rotation in SO3
    XB_w = tr.transform_from(R=RB_w, p=rB_w)  # Base Homogenous Transformation

    # x: is the concatenation of q_js and dq_js. That is the joint-space state-space vector.
    x = np.concatenate((q[7:], dq[6:]))
    x = x.astype(np.float64)
    # Get the representation of the symmetry group of the robot on the joint space (JS) state-space x=[q_js, dq_js]
    rep_x = rep_QJ + rep_QJ  # Since rep_QJ acts linearly on q, the representation on dq is the same (Eq.2 of paper)

    # To visualize how the symmetries of the robotic system affect propioceptive and exteroceptive measurements we use:
    # - Propioceptive: `hg_B` The center of mass linear and angular momentum of the robot, and `f1/2` contact forces
    # - Exteroceptive: We will assume to have measurements of the terrain elevation and orientation.
    # Consider the robot Center of Mass (CoM) momentum in base `B` coordinates. A vector and a pseudo-vector.
    hg_B = np.array([0.1, 0.1, 0.0, -0.0, -0.2, -0.2])
    # The representation for the linear `l` and angular `k` components  of Center of Mass (CoM) momentum `h=[l,k]` is:
    rep_h = rep_Ed + rep_Ed  # Additions of representations amounts to block-diagonal matrix concatenation.

    # Define mock surface orientations `Rf_w`, contact points `rf_w` and contact forces `f_w`
    Rf1_w, Rf2_w, f1_w, f2_w, rf1_w, rf2_w = get_mock_ground_reaction_forces(pb, robot, robot_cfg)

    # Main part of the script. =======================================================================================
    # Start by defining lists holding the orbits of all the proprioceptive and exteroceptive measurements
    # An orbit is all unique symmetric states of a variable G·x = {g·x | g in G}
    Gx, Ghg_B, Gy_B, GXB_w, GX_g_bar = [x], [hg_B], [hg_B], [XB_w], [tr.transform_from(R=np.eye(3), p=np.zeros(3))]
    Gf1_w, Gf2_w, Gr1_w, Gr2_w, GRf1_w, GRf2_w = [f1_w], [f2_w], [rf1_w], [rf2_w], [Rf1_w], [Rf2_w]

    # For each symmetry `g` of the system, get the representations of the action in the relevant vector spaces to
    # compute the symmetric states of robot configuration and data
    for g in G.elements[1:]:
        # Let x = [q, dq], and h = [l, k]
        # g_e3 = rep_Ed(g)
        # g_qj = rep_QJ(g)
        # Get symmetric g.x=[g.q, g.dq], g·h=[g·l, g·k] -------------------------------------------------------
        gx_w, gh_B = rep_x(g) @ x, rep_h(g) @ hg_B
        if np.linalg.det(rep_Ed(g)) < 0:  # If g is a reflection, we need to flip the sign of the angular momentum
            gh_B[3:] = -gh_B[3:]
        Gx.append(gx_w), Ghg_B.append(gh_B)

        # Get the Euclidean rotation/reflection of the symmetry in Ed -------------------------------------------------
        X_gbar = tr.transform_from(R=np.asarray(rep_Ed(g)), p=np.zeros(3))
        GX_g_bar.append(X_gbar)  # Homogeous matrix transformation of the Euclidean Isometry gbar

        # Compute new robot base configuration -----------------------------------------------------------------------
        gRB_w = rep_Ed(g) @ RB_w @ np.linalg.inv(rep_Ed(g))
        gr_w = tr.transform(X_gbar, XB_w[:, 3], strict_check=False)[:3]  # Transform the base position
        # Add new robot base configuration (homogenous matrix) to the orbit of base configs.
        GXB_w.append(tr.transform_from(R=gRB_w, p=gr_w))

        # Use symmetry representations to get symmetric versions of Euclidean vectors, representing measurements of data
        # We could also add some pseudo-vectors e.g. torque, and augment them as we did with `k`
        gf1_w, gf2_w = rep_Ed(g) @ f1_w, rep_Ed(g) @ f2_w
        gr_f1_w, gr_f2_w = rep_Ed(g) @ rf1_w, rep_Ed(g) @ rf2_w
        Gf1_w.append(gf1_w[:3]), Gf2_w.append(gf2_w[:3]), Gr1_w.append(gr_f1_w), Gr2_w.append(gr_f2_w)

        # The environment is theoretically truly reflected/rotated, but we use "cuboids" for representing estimations of
        GRf1_w.append(rep_Ed(g) @ Rf1_w @ np.linalg.inv(rep_Ed(g)))
        GRf2_w.append(rep_Ed(g) @ Rf2_w @ np.linalg.inv(rep_Ed(g)))
    # =============================================================================================================

    # Visualization of orbits of robot states and of data ==========================================================
    # Use Ctrl and mouse-click+drag to rotate the 3D environment.
    # Get the robot joint state (q_js, dq_js) from the state x for all system configurations.
    splited_orbits = [np.split(x, 2) for x in Gx]
    Gq_js, Gdq_js = [x[0] for x in splited_orbits], [x[1] for x in splited_orbits]
    Gq_js = [qj - q_offset[7:] for qj in Gq_js]  # Remove the offset from the joint angles
    display_robots_and_vectors(pb, robot, base_confs=GXB_w, Gq_js=Gq_js, Gdq_js=Gdq_js, Ghg=Ghg_B,
                               forces=[Gf1_w, Gf2_w], forces_points=[Gr1_w, Gr2_w], surface_normals=[GRf1_w, GRf2_w],
                               GX_g_bar=GX_g_bar, offset=offset, tint=cfg.robot.tint_bodies)


    if cfg.make_gif:
        # To get the optimal visualization you might need to play a bit with the rendering parameters for each robot.
        # this params seem to work mostly ok for all.
        cam_distance = offset * 5
        cam_target_pose = [0, 0, 0]
        save_path = pathlib.Path("paper/animations")
        save_path.mkdir(exist_ok=True)
        render_orbiting_animation(pb, cam_target_pose=cam_target_pose, cam_distance=cam_distance,
                                  save_path=save_path, anim_time=10, fps=15, periods=1,
                                  init_roll_pitch_yaw=(0, 35, 0), invert_roll="dh" in robot_cfg.group_label.lower(),
                                  pitch_sin_amplitude=20,
                                  file_name=f"{robot.robot_name}-{G.name}-symmetries_anim_static",
                                  gen_gif=True, gen_imgs=False)
        print("Done enjoy your gif :). I hope you learned something new")
    elif cfg.make_imgs:
        cam_distance = offset * 6
        cam_target_pose = [0, 0, 0]
        save_path = pathlib.Path(f"paper/images/{cfg.robot.name}")
        save_path.mkdir(exist_ok=True)
        render_orbiting_animation(pb, cam_target_pose=cam_target_pose, cam_distance=cam_distance,
                                  anim_time=2, fps=2, periods=1, pitch_sin_amplitude=0,
                                  init_roll_pitch_yaw=(0, 90, 0) if G.order() > 2 else (0, 0, 0),
                                  invert_roll="dh" in robot_cfg.group_label.lower(),
                                  save_path=save_path, gen_gif=False, gen_imgs=True)
    if cfg.gui:
        for _ in range(500):
            time.sleep(0.1)

        pb.disconnect()


if __name__ == '__main__':
    main()
