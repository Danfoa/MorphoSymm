"""PinBulletWrapper.

Code based on: https://github.com/machines-in-motion/bullet_utils.

Pybullet interface using pinocchio's convention.

License: BSD 3-Clause License
Copyright (C) 2018-2019, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
import copy
import logging
from dataclasses import dataclass
from typing import Collection, Iterable, Optional

import numpy as np
import scipy
from pinocchio import JointModelFreeFlyer, RobotWrapper
from pinocchio import pinocchio_pywrap as pin
from pinocchio.utils import zero
from pybullet_utils.bullet_client import BulletClient
from robot_descriptions.loaders.pinocchio import load_robot_description as pin_load_robot_description
from robot_descriptions.loaders.pybullet import load_robot_description as pb_load_robot_description
from scipy.linalg import inv

log = logging.getLogger(__name__)


@dataclass
class JointInfo:
    """Joint information."""
    pin_id: -1
    bullet_id: -1
    idx_q: -1
    idx_dq: -1
    pos_lims: (0, 0)
    vel_lim: np.Inf
    acc_lim: np.Inf
    tau_lim: np.Inf

class PinBulletWrapper:
    """Bridge class between pybullet and pinocchio.

    Attributes:
        nq (int): Dimension of the generalized coordiantes.
        nv (int): Dimension of the generalized velocities.
        nj (int): Number of joints.
        nf (int): Number of end-effectors.
        robot_id (int): PyBullet id of the robot.
        pinocchio_robot (Pinocchio.RobotWrapper): Pinocchio RobotWrapper for the robot.
        useFixedBase (bool): Determines if the robot base if fixed.
        nb_dof (int): The degrees of freedom excluding the base.
    """

    def __init__(self, robot_name:str, endeff_names: Optional[Iterable] = None, useFixedBase=False,
                 reference_robot: Optional['PinBulletWrapper'] = None, init_q=None, hip_height=1.0):
        """Initializes the wrapper.

        Args:
            robot_name (str): Name of this robot instance
            endeff_names (List[str]): Names of the end-effectors.
            useFixedBase (bool, optional): Determines if the robot base if fixed. Defaults to False.
            reference_robot (PinBulletWrapper, optional): Instance to copy the pinocchio model from. Defaults to None.
            init_q (List[float]): Initial configuration of the robot of length nq.
            hip_height (float): Height of the hip of the robot. Hip height on resting configuration.
        """
        self.robot_name = str.lower(robot_name)
        self.hip_height = hip_height
        self._joint_names = None
        self._endeff_names = endeff_names
        # Default to URDF values
        self._qj_high_limit, self._qj_low_limit, self._dqj_limit = None, None, None
        # Initialize Pinocchio Robot.
        self.pinocchio_robot = self.load_pinocchio_robot(reference_robot)
        self.nq = self.pinocchio_robot.nq
        self.nv = self.pinocchio_robot.nv
        self.nj = self.nq - 7
        # assert self.nj == len(self.joint_names), f"{len(self.joint_names)} != {self.nj}"
        self.nf = len(self.endeff_names) if self.endeff_names is not None else -1
        self.useFixedBase = useFixedBase
        self.nb_dof = self.nv - 6

        self._init_q = np.concatenate((np.zeros(6), [1], np.zeros(self.nj))) if init_q is None else np.array(init_q)
        assert len(self._init_q) == self.nq, f"Expected |q0|=3+4+nj={self.nq}, but received {len(self._init_q)}"
        self.base_linvel_prev = None
        self.base_angvel_prev = None
        self.base_linacc = np.zeros(3, dtype=np.float32)
        self.base_angacc = np.zeros(3, dtype=np.float32)

        # IMU pose offset in base frame
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        # Mappings between joint names pin and bullet ids, and pinocchio generalized q and dq coordinates
        self.joint_aux_vars = {}
        for joint, joint_name in zip(self.pinocchio_robot.model.joints, self.pinocchio_robot.model.names):
            if joint.idx_q == -1: continue  # Ignore universe
            if joint.nq == 7: continue      # Ignore floating-base
            log.debug(f"Joint[{joint_name}] - DoF(nq):{joint.nq}, idx_q:{joint.idx_q}, idx_v:{joint.idx_v}")
            vel_limit = self.pinocchio_robot.model.velocityLimit[joint.idx_v:joint.idx_v + joint.nv]
            upper_pos_limit = self.pinocchio_robot.model.upperPositionLimit[joint.idx_q:joint.idx_q + joint.nq]
            lower_pos_limit = self.pinocchio_robot.model.lowerPositionLimit[joint.idx_q:joint.idx_q + joint.nq]
            self.joint_aux_vars[joint_name] = JointInfo(pin_id=joint.id, bullet_id=np.NAN, idx_q=joint.idx_q,
                                                        idx_dq=joint.idx_v, pos_lims=(lower_pos_limit, upper_pos_limit),
                                                        vel_lim=vel_limit, acc_lim=np.Inf, tau_lim=np.Inf)
        self.joint_names = list(self.joint_aux_vars.keys())

        self._pb = None
        self.world = None
        self.robot_id = None
        self.bullet_endeff_ids = {}
        self.bullet_ids_allowed_floor_contacts = []

    def configure_bullet_simulation(self, bullet_client: BulletClient, world,
                                    base_pos=(0, 0, 0), base_ori=(0, 0, 0, 1)):
        """Configures the bullet simulation and loads this robot URDF description."""
        # Load robot to simulation
        self._pb = bullet_client
        self.world = world
        self.robot_id = self.load_bullet_robot(base_pos, base_ori)
        assert self.robot_id is not None
        log.debug("Configuring Bullet Robot")
        bullet_joint_map = {}  # Key: joint name - Value: joint id

        if self._qj_low_limit is None or self._qj_high_limit is None:
            self._qj_low_limit, self._qj_high_limit, self._dqj_limit = (np.empty(self.nj) for _ in
                                                                        range(3))
        auto_end_eff = False
        if self.endeff_names is None:
            auto_end_eff = True
            self._endeff_names = []

        for bullet_joint_id in range(self.bullet_client.getNumJoints(self.robot_id)):
            joint_info = self.bullet_client.getJointInfo(self.robot_id, bullet_joint_id)
            joint_name = joint_info[1].decode("UTF-8")
            if joint_name in self.joint_aux_vars.keys():
                self.joint_aux_vars[joint_name].bullet_id = bullet_joint_id
                # Fill default joint pos vel limits
                lower_limit, upper_limit = joint_info[8], joint_info[9]
                tau_max, dq_max = joint_info[10], joint_info[11]
                self.joint_aux_vars[joint_name].pos_lims = (lower_limit, upper_limit)
                self.joint_aux_vars[joint_name].vel_lim = dq_max
                self.joint_aux_vars[joint_name].tau_lim = tau_max
            elif auto_end_eff and not np.any([s in joint_name for s in ["base", "imu", "hip", "camera", "accelero"]]):
                log.info(f"Adding end-effector {joint_name}")
                self._endeff_names.append(joint_name)
            else:
                log.debug(f"unrecognized joint semantic type {joint_name}")
            bullet_joint_map[joint_name] = bullet_joint_id  # End effector joints.

        # In pybullet, the contact wrench is measured at action joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = {name: bullet_joint_map[name] for name in self.endeff_names}
        self.bullet_ids_allowed_floor_contacts = [bullet_joint_map[name] for name in self.endeff_names]

        # Configure end effectors
        self.nf = len(self.endeff_names)
        self.pinocchio_endeff_ids = {name: self.pinocchio_robot.model.getFrameId(name) for name in self.endeff_names}

        self._qj_low_limit = np.array([joint.pos_lims[0] for joint in self.joint_aux_vars.values()])
        self._qj_high_limit = np.array([joint.pos_lims[1] for joint in self.joint_aux_vars.values()])
        self._dqj_limit = np.array([joint.vel_lim for joint in self.joint_aux_vars.values()])

    def get_force(self):
        """Returns the force readings as well as the set of active contacts.

        Returns:
            (:obj:`list` of :obj:`int`): List of active contact frame ids.
            (:obj:`list` of np.array((6,1))) List of active contact forces.
        """
        active_contacts_frame_ids = []
        active_bullet_link_id = []

        contact_forces = []
        undesired_contacts = False  # Weather contact with other body parts and floor occured.

        # Get the contact model using the pybullet.getContactPoints() api.
        contact_points = self.bullet_client.getContactPoints(bodyA=self.robot_id, bodyB=self.world.floor_id)

        bullet_endeff_ids = list(self.bullet_endeff_ids.values())
        endeff_names = list(self.bullet_endeff_ids.keys())

        for contact_info in reversed(contact_points):
            robot_link_in_contact = contact_info[3]

            # if robot_link_in_contact not in bullet_endeff_ids:
            if robot_link_in_contact not in self.bullet_ids_allowed_floor_contacts:
                # link_name = self.bullet_client.getJointInfo(self.robot_id, robot_link_in_contact)[12].decode("UTF-8")
                undesired_contacts = True
                # continue
            contact_normal = contact_info[7]
            normal_force = contact_info[9]
            lateral_friction_direction_1 = contact_info[11]
            lateral_friction_force_1 = contact_info[10]
            lateral_friction_direction_2 = contact_info[13]
            lateral_friction_force_2 = contact_info[12]

            if robot_link_in_contact in active_bullet_link_id:
                continue

            if robot_link_in_contact in bullet_endeff_ids:
                active_endeff_name = endeff_names[bullet_endeff_ids.index(robot_link_in_contact)]
                active_contacts_frame_ids.append(self.pinocchio_endeff_ids[active_endeff_name])

            active_bullet_link_id.append(robot_link_in_contact)

            force = np.zeros(6)
            force[:3] = (normal_force * np.array(contact_normal)
                         + lateral_friction_force_1 * np.array(lateral_friction_direction_1)
                         + lateral_friction_force_2 * np.array(lateral_friction_direction_2))

            contact_forces.append(force)

        return active_contacts_frame_ids[::-1], contact_forces[::-1], undesired_contacts

    def get_state(self):
        """Returns action pinocchio-like representation of the q, dq matrices.

        Note that the base velocities are expressed in the base frame.

        Returns:
            ndarray: Generalized positions.
            ndarray: Generalized velocities.
        """
        q = zero(self.nq)
        dq = zero(self.nv)

        if not self.useFixedBase:
            base_inertia_pos, base_inertia_quat = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
            # Get transform between inertial frame and link frame in base
            base_stat = self.bullet_client.getDynamicsInfo(self.robot_id, -1)
            base_inertia_link_pos, base_inertia_link_quat = self.bullet_client.invertTransform(base_stat[3],
                                                                                               base_stat[4])
            pos, orn = self.bullet_client.multiplyTransforms(base_inertia_pos, base_inertia_quat,
                                                             base_inertia_link_pos, base_inertia_link_quat)

            q[:3] = pos
            q[3:7] = orn

            vel, orn = self.bullet_client.getBaseVelocity(self.robot_id)  # Return in "world" inertial reference frame
            dq[:3] = vel
            dq[3:6] = orn

            # Pinocchio assumes the base velocity to be in the body frame -> rotate.
            rot_base2world = np.array(self.bullet_client.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            dq[0:3] = rot_base2world.T.dot(dq[0:3])
            dq[3:6] = rot_base2world.T.dot(dq[3:6])

        # Query the joint readings.
        joint_states = self.bullet_client.getJointStates(self.robot_id,
                                                         [self.joint_aux_vars[m].bullet_id for m in self.joint_names])

        for i, joint_name in enumerate(self.joint_names):
            q[self.joint_aux_vars[joint_name].idx_q] = joint_states[i][0]
            dq[self.joint_aux_vars[joint_name].idx_dq] = joint_states[i][1]

        self.last_q = q
        self.last_dq = dq
        return q, dq

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.

        This includes updating:
        - kinematics
        - joint and frame jacobian
        - centroidal momentum

        Args:
          q: Pinocchio generalized position vector.
          dq: Pinocchio generalize velocity vector.
        """
        # pin.forwardKinematics(self.pinocchio_robot.model, self.pinocchio_robot.data, q, dq)

        # Must go first !!!!!
        # self.pinocchio_robot.forwardKinematics(q, dq)
        # self.pinocchio_robot.framesForwardKinematics(q)

        pin.ccrba(self.pinocchio_robot.model, self.pinocchio_robot.data, q, dq)
        pin.computeKineticEnergy(self.pinocchio_robot.model, self.pinocchio_robot.data)
        pin.computePotentialEnergy(self.pinocchio_robot.model, self.pinocchio_robot.data)
        vg = inv(self.pinocchio_robot.data.Ig) @ self.pinocchio_robot.data.hg
        # h = [Ij * hj for hj, Ij in zip(self.pinocchio_robot.data.h, self.pinocchio_robot.data.Ycrb)]
        self.rel_kinetic_energy = self.pinocchio_robot.data.kinetic_energy - (
                    0.5 * vg.T @ self.pinocchio_robot.data.Ig @ vg)

    def get_state_update_pinocchio(self):
        """Get state from pybullet and update pinocchio robot internals.

        This gets the state from the pybullet simulator and forwards
        the kinematics, jacobians, centroidal moments on the pinocchio robot
        (see forward_pinocchio for details on computed quantities).
        """
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        """Reset the robot to the desired states.

        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """
        def vec2list(m):
            return np.array(m.T).reshape(-1).tolist()

        if not self.useFixedBase:
            for joint_name in self.joint_names:
                joint_info = self.joint_aux_vars[joint_name]
                self.bullet_client.resetJointState(
                    self.robot_id,
                    joint_info.bullet_id,
                    q[joint_info.idx_q],
                    dq[joint_info.idx_dq],
                )

            # Get transform between inertial frame and link frame in base
            base_stat = self.bullet_client.getDynamicsInfo(self.robot_id, -1)
            base_pos, base_quat = self.bullet_client.multiplyTransforms(vec2list(q[:3]), vec2list(q[3:7]),
                                                                        base_stat[3], base_stat[4])
            self.bullet_client.resetBasePositionAndOrientation(self.robot_id, base_pos, base_quat)

            # Pybullet assumes the base velocity to be aligned with the world frame.
            rot = np.array(self.bullet_client.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            self.bullet_client.resetBaseVelocity(self.robot_id, vec2list(rot.dot(dq[:3])), vec2list(rot.dot(dq[3:6])))

        else:
            for joint_name in self.joint_names:
                joint_info = self.joint_aux_vars[joint_name]
                self.bullet_client.resetJointState(
                    self.robot_id,
                    joint_info.bullet_id,
                    q[joint_info.idx_q],
                    dq[joint_info.idx_dq],
                )

    def load_pinocchio_robot(self, reference_robot: Optional['PinBulletWrapper'] = None) -> RobotWrapper:
        """Function to load and configure the pinocchio instance of your robot.

        Returns:
            RobotWrapper: Instance of your pinocchio robot.
        """
        if reference_robot is not None:
            import sys
            assert np.all(self.joint_names == reference_robot.joint_names), "Invalid reference RobotWrapper"
            pin_robot = copy.copy(reference_robot.pinocchio_robot)
            pin_robot.data = copy.deepcopy(reference_robot.pinocchio_robot.data)
            assert sys.getrefcount(pin_robot.data) <= 2
        else:
            pin_robot = pin_load_robot_description(f"{self.robot_name}_description", root_joint=JointModelFreeFlyer())
        self._mass = float(np.sum([i.mass for i in pin_robot.model.inertias]))  # [kg]
        return pin_robot

    def load_bullet_robot(self, base_pos=None, base_ori=None) -> int:
        """Function to load and configure the pinocchio instance of your robot.

        Returns:
            int: Bullet robot body id.
        """
        self.robot_id = pb_load_robot_description(f"{self.robot_name}_description",
                                                  basePosition=base_pos, baseOrientation=base_ori,
                                                  flags=self.bullet_client.URDF_USE_INERTIA_FROM_FILE |
                                                        self.bullet_client.URDF_USE_SELF_COLLISION,
                                                  useFixedBase=self.useFixedBase)
        return self.robot_id

    def get_init_config(self, random=False, angle_sweep=None):
        """Get initial configuration of the robot.

        Args:
            random: if True, randomize the initial configuration.
            angle_sweep: if not None, randomize the initial configuration within the given angle sweep.

        Returns:
            q (ndarray): generalized positions.
            dq (ndarray): generalized velocities.
        """
        q = self._init_q
        qj = self._init_q[7:]
        dqj = np.zeros_like(qj)
        base_pos, base_ori = q[:3], q[3:7]

        if random:
            pitch = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25))
            roll = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25))
            yaw = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25))
            base_ori = scipy.spatial.transform.Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()

            if angle_sweep is not None:
                low_lim, high_lim = -angle_sweep, angle_sweep
            else:
                low_lim, high_lim = self._qj_low_limit, self._qj_high_limit
            qj_offset = np.random.uniform(low=low_lim, high=high_lim, size=(self.nj,))
            qj += qj_offset  # [rad]
            dqj_lim = np.minimum(2 * np.pi, self._dqj_limit)
            dqj_offset = np.random.uniform(low=-dqj_lim, high=dqj_lim, size=(self.nj,))
            dqj += dqj_offset  # [rad/s]

        q = np.concatenate([base_pos, base_ori, qj])
        dq = np.concatenate([np.zeros(6), dqj])
        return q, dq

    @property
    def mass(self) -> float:
        """Get the total mass of the robot.

        Returns:
            mass (float): Total mass of the robot in [kg] computed from the robot's URDF file.
        """
        return float(np.sum([i.mass for i in self.pinocchio_robot.model.inertias]))  # [kg]

    @property
    def velocity_limits(self, q=None, dq=None) -> np.ndarray:
        """Get the velocity limits of the robot.

        Args:
            q: Generalized positions (nq,).
            dq: Generalized velocities (nv,).

        Returns:
            vel_lims (ndarray): Velocity limits of shape (self.nv,).
        """
        if self.pinocchio_robot is None:
            raise AttributeError("Pinocchio robot has not been loaded")
        elif self._dqj_limit is None:
            self._dqj_limit = []
            for joint_name in self.joint_names:
                vel_limit = self.joint_aux_vars[joint_name].vel_lim
                self._dqj_limit.append(vel_limit)
            self._dqj_limit = np.asarray(self._dqj_limit).flatten()
        return self._dqj_limit

    @property
    def joint_pos_limits(self, q=None, dq=None):
        """Get the joint position limits of the robot.

        Args:
            q: generalized positions (nq,).
            dq: generalized velocities (nv,).

        Returns:
            qj_low_limit (ndarray): Lower joint position limits of shape (self.nj,).
        """
        if self.pinocchio_robot is None:
            raise AttributeError("Pinocchio robot has not been loaded")
        elif self._qj_low_limit is None or self._qj_high_limit is None:
            self._qj_high_limit, self._qj_low_limit = [], []
            for joint_name in self.joint_names:
                low, high = self.joint_aux_vars[joint_name].pos_lims
                self._qj_low_limit.append(low)
                self._qj_high_limit.append(high)
            self._qj_high_limit = np.asarray(self._qj_high_limit).flatten()
            self._qj_low_limit = np.asarray(self._qj_low_limit).flatten()
        return self._qj_low_limit, self._qj_high_limit

    @property
    def endeff_names(self) -> Collection:
        """Returns the names of the end-effectors of the robot.

        Returns:
            Collection: List of end-effector names.
        """
        return self._endeff_names

    @property
    def bullet_client(self):
        """Returns the bullet client instance this robot is associated with.

        Returns:
            BulletClient (int): Bullet client instance.
        """
        if self._pb is None:
            raise RuntimeError("Robot bullet simulation not configured, remember to call `configure_bullet_simulation`")
        return self._pb

    def get_base_position_world(self):
        """Returns the position of the base in the world frame.

        Returns:
            np.array((3,)) with the translation
            np.array((4,)) with angular position in quaternion.
        """
        base_inertia_pos, base_inertia_quat = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
        return base_inertia_pos, base_inertia_quat

    def get_base_velocity_world(self):
        """Returns the velocity of the base in the world frame.

        Returns:
            np.array((6,1)) with the translation and angular velocity
        """
        vel, orn = self.bullet_client.getBaseVelocity(self.robot_id)
        return np.array(vel + orn).reshape(6, 1)

    def get_base_acceleration_world(self):
        """Returns the numerically-computed acceleration of the base in the world frame.

        Returns:
            np.array((6,1)) vector of linear and angular acceleration
        """
        return np.concatenate((self.base_linacc, self.base_angacc))
