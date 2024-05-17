"""PinBulletWrapper.

Code based on: https://github.com/machines-in-motion/bullet_utils.

Pybullet interface using pinocchio's convention.

License: BSD 3-Clause License
Copyright (C) 2018-2019, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
import logging
from typing import Iterable, Optional

import numpy as np
import pinocchio
from pinocchio.utils import zero
from pybullet_utils.bullet_client import BulletClient
from robot_descriptions.loaders.pybullet import load_robot_description as pb_load_robot_description

from morpho_symm.robots.PinSimWrapper import JointWrapper, PinSimWrapper, SimPinJointWrapper, State

log = logging.getLogger(__name__)


class PinBulletWrapper(PinSimWrapper):
    """Bridge class between pybullet and pinocchio.

    Attributes:
        nq (int): Dimension of the generalized coordiantes.
        nv (int): Dimension of the generalized velocities.
        n_js (int): Number of joints.
        nf (int): Number of end-effectors.
        robot_id (int): PyBullet id of the robot.
        pinocchio_robot (Pinocchio.RobotWrapper): Pinocchio RobotWrapper for the robot.
        fixed_base (bool): Determines if the robot base if fixed.
    """

    def __init__(self, robot_name: str, endeff_names: Optional[Iterable] = None, fixed_base=False,
                 reference_robot: Optional['PinBulletWrapper'] = None, hip_height=1.0, init_q=None, q_zero=None):
        """Initializes the wrapper.

        Args:
            robot_name (str): Name of this robot instance
            endeff_names (List[str]): Names of the end-effectors.
            fixed_base (bool, optional): Determines if the robot base if fixed. Defaults to False.
            reference_robot (PinBulletWrapper, optional): Instance to copy the pinocchio model from. Defaults to None.
            hip_height (float): Height of the hip of the robot. Hip height on resting configuration.
            init_q (List[float]): Initial configuration of the robot of length nq.
            q_zero (List[float]): Zero configuration. This can be different from the Zero config defined in the URDF.
        """
        super().__init__(robot_name, endeff_names, fixed_base, reference_robot, hip_height, init_q, q_zero)

        self._pb = None
        self.world = None
        self.robot_id = None
        self.bullet_endeff_ids = {}
        self.bullet_ids_allowed_floor_contacts = []

    def configure_bullet_simulation(self, bullet_client: BulletClient, world=None,
                                    base_pos=(0, 0, 0), base_ori=(0, 0, 0, 1)):
        """Configures the bullet simulation and loads this robot URDF description."""
        # Load robot to simulation
        self._pb = bullet_client
        self.world = world
        self.robot_id = self.load_bullet_robot(base_pos, base_ori)
        assert self.robot_id is not None

        log.debug("Configuring Bullet Robot")
        bullet_joint_map = {}  # Key: joint name - Value: joint id

        auto_end_eff = False
        if self.endeff_names is None:
            auto_end_eff = True
            self._endeff_names = []

        for bullet_joint_id in range(self.bullet_client.getNumJoints(self.robot_id)):
            joint_info = self.bullet_client.getJointInfo(self.robot_id, bullet_joint_id)
            joint_name = joint_info[1].decode("UTF-8")
            bullet_joint_map[joint_name] = bullet_joint_id
            if joint_name in self.pin_joint_space:
                pin_joint = self.pin_joint_space[joint_name]
                # Create Joint Wrapper with parameter convention from pybullet
                bullet_joint_type = joint_info[2]
                if bullet_joint_type in [self.bullet_client.JOINT_REVOLUTE, self.bullet_client.JOINT_PRISMATIC]:
                    nq, nv = 1, 1
                elif bullet_joint_type == self.bullet_client.JOINT_SPHERICAL:
                    nq, nv = 4, 3
                elif bullet_joint_type == self.bullet_client.JOINT_FIXED:
                    nq, nv = 0, 0
                elif bullet_joint_type == self.bullet_client.JOINT_PLANAR:
                    nq, nv = 2, 2
                else:
                    raise NotImplementedError(f"Joint type {bullet_joint_type} not handled")
                bullet_joint = JointWrapper(type=bullet_joint_type, idx_q=joint_info[3], idx_v=joint_info[4],
                                            nq=nq, nv=nv, pos_limit_low=joint_info[8], pos_limit_high=joint_info[9])
                # Integrate the two convention in a utility class handling the conversions accordingly
                pb_pin_joint = BulletJointWrapper(pin_joint=pin_joint, bullet_joint=bullet_joint,
                                                  bullet_idx=joint_info[0], bullet_client=self.bullet_client,
                                                  damping=joint_info[6], friction=joint_info[7],
                                                  max_force=joint_info[10], max_vel=joint_info[11], axis=joint_info[13],
                                                  link_name=joint_info[12], parent_frame_pos=joint_info[14],
                                                  parent_frame_ori=joint_info[15], parent_link_idx=joint_info[16])
                # Store the SimPinJointWrapper for all joint-space joints.
                self.joint_space[joint_name] = pb_pin_joint

            elif auto_end_eff:
                if not np.any([s in joint_name for s in ["base", "imu", "hip", "camera", "accelero"]]):
                    log.debug(f"Adding end-effector {joint_name}")
                    self._endeff_names.append(joint_name)
            else:
                log.debug(f"unrecognized joint semantic type {joint_name}")

        # In pybullet, the contact wrench is measured at action joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = {name: bullet_joint_map[name] for name in self.endeff_names}
        self.bullet_ids_allowed_floor_contacts = [bullet_joint_map[name] for name in self.endeff_names]

        # Configure end effectors
        self.pb_nq = 7 + np.sum([self.joint_space[name].sim_joint.nq for name in self.joint_space_names])
        self.pb_nv = 6 + np.sum([self.joint_space[name].sim_joint.nv for name in self.joint_space_names])

        self.nf = len(self.endeff_names)
        self.pinocchio_endeff_ids = {name: self.pinocchio_robot.model.getFrameId(name) for name in self.endeff_names}
        self.bullet_joint_map = bullet_joint_map

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

    def get_state_sim(self) -> State:
        """Fetch state from bullet (q, dq).

        Obtains the system state from the simulator [bullet] and returns the generalized positions and velocities in the
        simulator convenction.

        Returns:
            q (ndarray): Generalized position coordinates of shape in bullet convention.
            v (ndarray): Generalized velocity coordinates of shape in bullet convention.
        """
        q_sim = zero(self.pb_nq)
        v_sim = zero(self.pb_nv)

        if not self.fixed_base:
            base_inertia_pos, base_inertia_quat = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
            # Get transform between inertial frame and link frame in base
            base_stat = self.bullet_client.getDynamicsInfo(self.robot_id, -1)
            base_inertia_link_pos, base_inertia_link_quat = self.bullet_client.invertTransform(base_stat[3],
                                                                                               base_stat[4])
            pos, orn = self.bullet_client.multiplyTransforms(base_inertia_pos, base_inertia_quat,
                                                             base_inertia_link_pos, base_inertia_link_quat)

            q_sim[:3] = pos
            q_sim[3:7] = orn

            vel, orn = self.bullet_client.getBaseVelocity(self.robot_id)  # Return in "world" inertial reference frame
            v_sim[:3] = vel
            v_sim[3:6] = orn

            # Pinocchio assumes the base velocity to be in the body frame -> rotate.
            rot_base2world = np.array(self.bullet_client.getMatrixFromQuaternion(q_sim[3:7])).reshape((3, 3))
            v_sim[0:3] = rot_base2world.T.dot(v_sim[0:3])
            v_sim[3:6] = rot_base2world.T.dot(v_sim[3:6])

        # Fetch joint state from bullet
        joint_states = self.bullet_client.getJointStates(self.robot_id,
                                                         [self.joint_space[m].bullet_idx for m in
                                                          self.joint_space_names])
        for joint_name, joint_state in zip(self.joint_space_names, joint_states):
            joint = self.joint_space[joint_name]
            q_sim[joint.sim_joint.idx_q: joint.sim_joint.idx_q + joint.sim_joint.nq] = joint_state[0]
            v_sim[joint.sim_joint.idx_q: joint.sim_joint.idx_q + joint.sim_joint.nq] = joint_state[1]

        return q_sim, v_sim

    def pin2sim(self, q, v) -> State:

        pb_q = np.zeros(self.pb_nq)
        pb_v = np.zeros(self.pb_nv)

        for joint_name, joint in self.joint_space.items():
            q_joint = q[joint.pin_joint.idx_q: joint.pin_joint.idx_q + joint.pin_joint.nq]
            dq_joint = v[joint.pin_joint.idx_v: joint.pin_joint.idx_v + joint.pin_joint.nv]
            pb_q_joint, pb_v_joint = joint.pin2sim(q_joint, dq_joint)
            # Place value in simulator joint index
            pb_q[joint.sim_joint.idx_q: joint.sim_joint.idx_q + joint.sim_joint.nq] = pb_q_joint
            pb_v[joint.sim_joint.idx_v: joint.sim_joint.idx_v + joint.sim_joint.nv] = pb_v_joint

        # Base configuration
        pb_q[:7] = q[:7]
        pb_v[:6] = v[:6]

        return pb_q, pb_v

    def sim2pin(self, pb_q, pb_v) -> State:

        q = np.zeros(self.nq)
        v = np.zeros(self.nv)
        for joint_name, joint in self.joint_space.items():
            # Extract position and velocity coordinates from simulator
            pb_q_joint = pb_q[joint.sim_joint.idx_q: joint.sim_joint.idx_q + joint.sim_joint.nq]
            pb_v_joint = pb_v[joint.sim_joint.idx_v: joint.sim_joint.idx_v + joint.sim_joint.nv]
            # Convert position and velocity coordinates to pinocchio convention
            q_joint, v_joint = joint.sim2pin(pb_q_joint, pb_v_joint)
            # Place values in Pinocchio joint index ordering
            q[joint.pin_joint.idx_q: joint.pin_joint.idx_q + joint.pin_joint.nq] = q_joint
            v[joint.pin_joint.idx_v: joint.pin_joint.idx_v + joint.pin_joint.nv] = v_joint

        # Base configuration
        q[:7] = pb_q[:7]
        v[:6] = pb_v[:6]

        return q, v

    def reset_state_sim(self, q, v) -> None:
        """Reset robot state in bullet to the described (q, dq) configuration in pinocchio convention.

        Args:
            q (ndarray): Generalized position coordinates of shape in bullet convention.
            v (ndarray): Generalized velocity coordinates of shape in bullet convention.
        """
        assert not np.iscomplexobj(q), "q must be real valued"
        assert not np.iscomplexobj(v), "v must be real valued"

        q, v = np.array(q), np.array(v)

        def vec2list(m):
            return np.asarray(m.T).reshape(-1).tolist()

        for joint_name, joint in self.joint_space.items():
            self.bullet_client.resetJointState(
                self.robot_id,
                joint.bullet_idx,
                q[joint.sim_joint.idx_q: joint.sim_joint.idx_q + joint.sim_joint.nq],
                v[joint.sim_joint.idx_v: joint.sim_joint.idx_v + joint.sim_joint.nv],
                )

        if not self.fixed_base:
            # Get transform between inertial frame and link frame in base
            base_stat = self.bullet_client.getDynamicsInfo(self.robot_id, -1)
            base_pos, base_quat = self.bullet_client.multiplyTransforms(vec2list(q[:3]), vec2list(q[3:7]),
                                                                        base_stat[3], base_stat[4])
            self.bullet_client.resetBasePositionAndOrientation(self.robot_id, base_pos, base_quat)

            # Pybullet assumes the base velocity to be aligned with the world frame.
            rot = np.array(self.bullet_client.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            self.bullet_client.resetBaseVelocity(self.robot_id, vec2list(rot.dot(v[:3])),
                                                 vec2list(rot.dot(v[3:6])))

    def load_bullet_robot(self, base_pos=None, base_ori=None) -> int:
        """Function to load and configure the pinocchio instance of your robot.

        Returns:
            int: Bullet robot body id.
        """
        self.robot_id = pb_load_robot_description(f"{self.robot_name}_description",
                                                  basePosition=base_pos, baseOrientation=base_ori,
                                                  flags=self.bullet_client.URDF_USE_INERTIA_FROM_FILE |
                                                        self.bullet_client.URDF_USE_SELF_COLLISION,
                                                  useFixedBase=self.fixed_base)
        return self.robot_id

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

    def __repr__(self):
        """."""
        bullet_id = f"({self.robot_id})" if hasattr(self, 'robot_id') else ""
        return f"{self.robot_name}{bullet_id}-nq:{self.nq}-nv:{self.nv}"

    @staticmethod
    def from_instance(other: 'PinBulletWrapper') -> 'PinBulletWrapper':
        """Creates another instance of this robot wrapper without duplicating the model or data from pinocchio robot.

        This is usefull when we want to spawn multiple instances of the same robot on the physics simulator.


        Args:
            other (PinBulletWrapper): The instance from which to get the model and data for the pinocchio model

        Returns:
            PinBulletWrapper: A new instance of this robot wrapper
        """
        return PinBulletWrapper(
            robot_name=other.robot_name,
            endeff_names=other.endeff_names,
            fixed_base=other.fixed_base,
            # reference_robot=other,
            init_q=other._init_q,
            hip_height=other.hip_height,
            )


class BulletJointWrapper(SimPinJointWrapper):
    """Auxiliary class to integrate Bullet and Pinocchio Joint models."""

    def __init__(self, pin_joint: JointWrapper, bullet_joint: JointWrapper, bullet_client: BulletClient,
                 bullet_idx: int,
                 damping: float, friction: float, max_force: float, max_vel: float, link_name: str, axis: np.ndarray,
                 parent_frame_pos: np.ndarray, parent_frame_ori: np.ndarray, parent_link_idx: int):
        self.pin_joint = pin_joint
        self.sim_joint = bullet_joint
        self.bullet_client = bullet_client
        self.bullet_idx = bullet_idx
        self.damping = damping
        self.friction = friction
        self.max_force = max_force
        self.max_vel = max_vel
        self.link_name = link_name
        self.axis = axis
        self.parent_frame_pos = parent_frame_pos
        self.parent_frame_ori = parent_frame_ori
        self.parent_link_idx = parent_link_idx

    def sim2pin(self, q, v):
        if self.pin_joint.nq == 1:
            return q, v
        if self.pin_joint.nq == 2 and self.pin_joint.nv == 1:  # Unit circle
            return np.asarray([np.cos(q), np.sin(q)]).flatten(), v
        else:
            raise NotImplementedError()

    def pin2sim(self, q, v):
        if self.pin_joint.nq == 1:
            return q, v
        if self.pin_joint.nq == 2 and self.pin_joint.nv == 1:  # Unit circle
            theta = np.arctan2(q[1], q[0])
            return np.asarray([theta]).flatten(), v
        else:
            raise NotImplementedError()
