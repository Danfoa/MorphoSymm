"""PinBulletWrapper
Code based on: https://github.com/machines-in-motion/bullet_utils

Pybullet interface using pinocchio's convention.

License: BSD 3-Clause License
Copyright (C) 2018-2019, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
import pathlib
from abc import ABC, abstractmethod
from enum import auto
from strenum import StrEnum
from math import pi
from typing import Collection, Optional, List, Tuple, Union
from pinocchio import pinocchio_pywrap as pin
from scipy.linalg import inv, pinv
import gym
import numpy as np
import pinocchio
from dataclasses import dataclass
from numpy.random import default_rng
from pinocchio import RobotWrapper
from pinocchio.utils import zero
from pybullet_utils.bullet_client import BulletClient

import logging

from robots.AccelerationBounds import ContrainAccelerationBound

log = logging.getLogger(__name__)

@dataclass
class JointInfo:
    pin_id: -1
    bullet_id: -1
    idx_q: -1
    idx_dq: -1

# TODO: Remove dependency StrEnum
class ControlMode(StrEnum):
    TORQUE = 'torque'
    ACC = 'acceleration'
    BOUNDED_ACC = 'bounded_acceleration'
    POSITION_PD = 'position_pd'
    IMPEDANCE = 'impedance'

class PinBulletWrapper(ABC):
    """[summary]

    Attributes:
        nq (int): Dimension of the generalized coordiantes.
        nv (int): Dimension of the generalized velocities.
        nj (int): Number of joints.
        nf (int): Number of end-effectors.
        robot_id (int): PyBullet id of the robot.
        pinocchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
        useFixedBase (bool): Determines if the robot base if fixed.
        nb_dof (int): The degrees of freedom excluding the base.
        joint_names (:obj:`list` of :obj:`str`): Names of the joints.
        endeff_names (:obj:`list` of :obj:`str`): Names of the end-effectors.
    """

    def __init__(self, resources: pathlib.Path, control_mode=ControlMode('torque'),
                 useFixedBase=False, reference_robot: Optional['PinBulletWrapper'] = None,
                 feet_lateral_friction=0.95, feet_spinning_friction=0.5, **kwargs):
        """Initializes the wrapper.

        Args:
            robot_id (int): PyBullet id of the robot.
            pinocchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
            joint_names (:obj:`list` of :obj:`str`): Names of the joints.
            endeff_names (:obj:`list` of :obj:`str`): Names of the end-effectors.
            useFixedBase (bool, optional): Determines if the robot base if fixed.. Defaults to False.
        """
        self._joint_names = None
        self._endeff_names = None
        self._mirror_joint_idx, self._mirror_endeff_idx = None, None  # Used for mirroring robot state
        self._mirror_obs_idx, self._mirror_obs_signs = None, None
        # Default to URDF values
        self._joint_upper_limits, self._joint_lower_limits, self._joint_vel_limits = None, None, None
        self._feet_lateral_friction = feet_lateral_friction
        self._feet_spinning_friction = feet_spinning_friction
        # Initialize Pinocchio Robot.
        self.resources = resources
        self.pinocchio_robot = self.load_pinocchio_robot(reference_robot)
        self.control_mode = control_mode
        self.nq = self.pinocchio_robot.nq
        self.nv = self.pinocchio_robot.nv
        self.nj = self.nq - 7
        assert self.nj == len(self.joint_names), f"{len(self.joint_names)} != {self.nj}"
        self.nf = len(self.endeff_names)
        self.useFixedBase = useFixedBase
        self.nb_dof = self.nv - 6

        self.base_linvel_prev = None
        self.base_angvel_prev = None
        self.base_linacc = np.zeros(3, dtype=np.float32)
        self.base_angacc = np.zeros(3, dtype=np.float32)

        # IMU pose offset in base frame
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        self.rng = default_rng()

        self.last_q, self.last_dq, self.last_tau = None, None, np.NAN
        self.rel_kinetic_energy = np.NAN

        for j_name in self.joint_names:
            assert self.pinocchio_robot.model.existJointName(j_name), j_name

        self.pinocchio_endeff_ids = {name: self.pinocchio_robot.model.getFrameId(name) for name in self.endeff_names}
        # Mappings between joint names pin and bullet ids, and pinocchio generalized q and dq coordinates
        self.joint_aux_vars = {}
        for joint, joint_name in zip(self.pinocchio_robot.model.joints, self.pinocchio_robot.model.names):
            if joint_name in self.joint_names:
                self.joint_aux_vars[joint_name] = JointInfo(pin_id=joint.id, bullet_id=-1,
                                                            idx_q=joint.idx_q, idx_dq=joint.idx_v)
        # Bullet simulation attributes TODO: Documentation.
        self._pb = None
        self.world = None
        self.robot_id = None
        self.bullet_endeff_ids = {}
        self.bullet_ids_allowed_floor_contacts = []
        # PD control variables
        kps = [3.5] * self.nj
        kds = [0.2] * self.nj
        self._Kp = np.diagflat(kps)
        self._Kd = np.diagflat(kds)



    def configure_bullet_simulation(self, bullet_client: BulletClient, world, base_pos=None, base_ori=None):
        # Load robot to simulation
        self._pb = bullet_client
        self.world = world
        self.robot_id = self.load_bullet_robot(base_pos, base_ori)
        assert self.robot_id is not None
        log.debug("Configuring Bullet Robot")
        bullet_joint_map = {}  # Key: joint name - Value: joint id

        if self._joint_lower_limits is None or self._joint_upper_limits is None:
            self._joint_lower_limits, self._joint_upper_limits, self._joint_vel_limits = (np.empty(self.nj) for _ in range(3))

        a = list(range(self.bullet_client.getNumJoints(self.robot_id)))
        for bullet_joint_id in range(self.bullet_client.getNumJoints(self.robot_id)):
            joint_info = self.bullet_client.getJointInfo(self.robot_id, bullet_joint_id)
            joint_name = joint_info[1].decode("UTF-8")

            if joint_name in self.joint_names:
                self.joint_aux_vars[joint_name].bullet_id = bullet_joint_id
                # Fill default joint pos vel limits
                lower_limit, upper_limit = joint_info[8], joint_info[9]
                tau_max, dq_max = joint_info[10], joint_info[11]
                idx = self.joint_names.index(joint_name)
                self._joint_lower_limits[idx] = lower_limit
                self._joint_upper_limits[idx] = upper_limit
                self._joint_vel_limits[idx] = dq_max

            bullet_joint_map[joint_name] = bullet_joint_id  # End effector joints.

        # Disable the velocity control on the joints as we use torque control.
        self.bullet_client.setJointMotorControlArray(self.robot_id,
                                                     [j.bullet_id for j in self.joint_aux_vars.values()],
                                                     self.bullet_client.VELOCITY_CONTROL,
                                                     forces=np.zeros(self.nj))
        self.bullet_client.setJointMotorControlArray(self.robot_id, [j.bullet_id for j in self.joint_aux_vars.values()],
                                                     controlMode=self.bullet_client.TORQUE_CONTROL,
                                                     forces=np.zeros(self.nj))

        # In pybullet, the contact wrench is measured at action joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = {name: bullet_joint_map[name] for name in self.endeff_names}
        self.bullet_ids_allowed_floor_contacts = [bullet_joint_map[name] for name in self.endeff_names]

        # Enforce similar actuator dynamics and ensure friction forces:
        joints_lower_limit, joints_upper_limit = self.joint_pos_limits
        for joint_id in range(self.nj):
            joint_info = self.bullet_client.getJointInfo(self.robot_id, joint_id)
            link_name = joint_info[12].decode("UTF-8")
            joint_name = joint_info[1].decode("UTF-8")
            parent_id = joint_info[16]
            lower_limit, upper_limit = joint_info[8], joint_info[9]
            joint_type = joint_info[2]

            # if joint_type == self.bullet_client.JOINT_REVOLUTE:
            #     # Override URDF values if configured in robot specific child class
            #     dq_max = self.velocity_limits[joint_id]
            #     lower_limit, upper_limit = joints_lower_limit[joint_id], joints_upper_limit[joint_id]
            #     self.bullet_client.changeDynamics(self.robot_id, joint_id,
            #                                       linearDamping=0.04, angularDamping=0.04,
            #                                       restitution=0.3, lateralFriction=0.5,
            #                                       jointLowerLimit=lower_limit,
            #                                       jointUpperLimit=upper_limit,
            #                                       maxJointVelocity=dq_max
            #                                       )
            # elif joint_type == self.bullet_client.JOINT_FIXED:  # Ensure end effectors have contact friction
            #     self.bullet_client.changeDynamics(self.robot_id, joint_id,
            #                                       restitution=0.0,
            #                                       spinningFriction=self._feet_spinning_friction,
            #                                       # improve collision of round objects
            #                                       contactStiffness=30000,
            #                                       contactDamping=1000,
            #                                       lateralFriction=self._feet_lateral_friction)
            # elif joint_type == self.bullet_client.JOINT_PRISMATIC:
            #     self.bullet_client.changeDynamics(self.robot_id, joint_id, linearDamping=0, angularDamping=0, )

            log.debug("- id:{:<4} j_name: {:<10} l_name: {:<15} parent_id: {:<4} lim:[{:.1f},{:.1f}]".format(
                joint_id, joint_name, link_name, parent_id, np.rad2deg(lower_limit), np.rad2deg(upper_limit)))

        # Compute Observation and Action Space sizes
        n_obs = len(self.get_observation())
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.nj,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,))

    def get_force(self):
        """Returns the force readings as well as the set of active contacts

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
        """
        Returns action pinocchio-like representation of the q, dq matrices.
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
        self.rel_kinetic_energy = self.pinocchio_robot.data.kinetic_energy - (0.5 * vg.T @ self.pinocchio_robot.data.Ig @ vg)

    def get_state_update_pinocchio(self):
        """Get state from pybullet and update pinocchio robot internals.

        This gets the state from the pybullet simulator and forwards
        the kinematics, jacobians, centroidal moments on the pinocchio robot
        (see forward_pinocchio for details on computed quantities)."""
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        """Reset the robot to the desired states.

        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """
        vec2list = lambda m: np.array(m.T).reshape(-1).tolist()

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

    def send_joint_command(self, tau: np.ndarray):
        """Apply the desired torques to the joints.
        Args:
            tau (ndarray): Torque to be applied assumed to be in the order of self.joint_names
        """
        # TODO: Apply the torques on the base towards the simulator as well.
        if not self.useFixedBase:
            assert tau.shape[0] == self.nv - 6
        # else:
            # assert tau.shape[0] == self.nv

        tau_clipped = np.clip(tau, -self.torque_limits, self.torque_limits)
        zeroGains = tau.shape[0] * (0.0,)

        self.last_tau = tau_clipped
        self.bullet_client.setJointMotorControlArray(
            self.robot_id,
            [self.joint_aux_vars[j_name].bullet_id for j_name in self.joint_names],
            self.bullet_client.TORQUE_CONTROL,
            forces=tau_clipped,
            positionGains=zeroGains,
            velocityGains=zeroGains,
        )

    def apply_action(self, a, q=None, dq=None, ddq_lb=None, ddq_ub=None):
        """
        Transforms the action commands in rance [-1, 1] to torque values using any of the ControlModes selected
        Args:
            a: (nj,) actions
            q: [Optional] robot position state
            dq: [Optional] robot velocity state
            ddq_lb: Optional (nj,) joint acceleration lower bounds
            ddq_ub: Optional (nj,) joint acceleration upper bounds

        Returns:

        """
        assert (np.isfinite(a).all()), "Invalid action: {}".format(a)

        if self.control_mode == ControlMode.TORQUE:
            tau = np.clip(a, -1, +1) * self.torque_limits
            self.send_joint_command(np.squeeze(tau))
            return tau
        elif self.control_mode == ControlMode.ACC:
            # Scale action to the acceleration range
            ddqa_des = np.squeeze(np.clip(a, -1, +1)) * self.acceleration_limits
            # Integrate current states to obtain the desired postion and velocity setpoints.
            dt = self.world.timestep
            dqa_des = dq[6:] + ddqa_des * dt
            qa_des = q[7:] + dqa_des * dt
            # Compute stable PD torque control values
            tau = self.stable_pd_control(q, dq, qa_des=qa_des, dqa_des=dqa_des, dt=self.world.timestep)
            self.send_joint_command(np.squeeze(tau))
            return qa_des, dqa_des
        elif self.control_mode == ControlMode.BOUNDED_ACC:
            # Scale action to the constrained acceleration range
            if ddq_lb is None or ddq_ub is None:
                raise AttributeError("For bound acc control you must provide the acceleration upper and lower bounds")
            ddqat_0 = (ddq_ub - ddq_lb) / 2 + ddq_lb
            ddqa_des = np.squeeze(np.clip(a, -1, +1)) * ((ddq_ub - ddq_lb) / 2) + ddqat_0
            ddqa_des = np.clip(ddqa_des, a_min=-self.acceleration_limits, a_max=self.acceleration_limits)

            # Integrate current states to obtain the desired position and velocity setpoints.
            dt = self.world.timestep
            dqa_des = dq[6:] + ddqa_des * dt
            qa_des = q[7:] + dqa_des * dt
            # Compute stable PD torque control values
            tau = self.stable_pd_control(q, dq, qa_des=qa_des, dqa_des=dqa_des, dt=self.world.timestep)
            self.send_joint_command(np.squeeze(tau))
            return qa_des, dqa_des, ddqa_des
        elif self.control_mode == ControlMode.POSITION_PD:
            raise NotImplementedError()
            # qa: actuated dof
            # qa_pos_min, qa_pos_max = self.joint_pos_limits
            # # TODO: Non symmetric nor centered bounds
            # qa_des = np.squeeze(np.clip(action, -1, +1)) * qa_pos_max
            # if q is None or dq is None:
            #     q, dq = self.get_state()
            # tau = self.stable_pd_control(q, dq, qa_des=qa_des, dt=self.world.timestep)
            # tau = np.clip(tau, -self._max_servo_torque, self._max_servo_torque)
        else:
            raise NotImplementedError()



    def load_pinocchio_robot(self, reference_robot: Optional['PinBulletWrapper'] = None) -> RobotWrapper:
        """
        Function to load and configure the pinocchio instance of your robot.
        Returns:
            RobotWrapper: Instance of your pinocchio robot.
        """
        raise NotImplementedError("You forgot to implement function returning the `pinocchio.RobotWrapper`")

    def load_bullet_robot(self, base_pos=None, base_ori=None) -> int:
        """
        Function to load and configure the pinocchio instance of your robot.
            int: Bullet robot body id.
        """
        raise NotImplementedError("You forgot to implement function loading the robot to bullet")

    @abstractmethod
    def get_init_config(self, random=False):
        raise NotImplementedError()

    def get_observation(self, q=None, dq=None) -> Collection:
        """
        Used in RL settings where the observation stands for the MDP agent observation state.
        This function should be overwritten for maximum flexibility, the default observation is composed of:
         - COM Linear and angular momentum (6,)
         - Robot base height (1,)
         - Robot base orientation in world coordinates (6,)
         - Robot joints positions q (nj,)
        Returns:
            Robot state observation
        """
        # TODO: Enforce all observation is invariant to the direction of motion
        # Get robot state in Pinocchio format.
        q_pin, dq_pin = self.get_state() if q is None else (q, dq)
        # Get only actuated joints states _________________________________________________________________________
        q_obs = q_pin[7:7 + self.nj]
        # dq = dq_pin[7:]

        # Get robot position related observation variables
        base_pos_obs = q_pin[[2]]  # [Z]
        # Get robot orientation related observation variables
        base_ori_quaternion = q_pin[3:7]
        base_ori_so3 = np.array(self.bullet_client.getMatrixFromQuaternion(base_ori_quaternion)).reshape((3, 3))
        # Get NN friendly continuous representation of rotations in 3D
        # 6D - Paper:(On the Continuity of Rotation Representations in Neural Networks)
        base_ori_obs = base_ori_so3[:, :2].flatten(order='F')  #

        self.update_pinocchio(q_pin, dq_pin)
        hg = self.pinocchio_robot.data.hg #self.pinocchio_robot.centroidalMomentum(q_pin, dq_pin)
        com_momentum_obs = hg.np

        return np.concatenate((com_momentum_obs, base_pos_obs, base_ori_obs, q_obs))

    def get_mirrored_observation(self, q, dq, obs=None):
        """
        Mirrors the robot observation variables along the sagittal plane and the ZX plane.
        i.e. The robot joints positions (nj) and velocities are mirrored across the robot sagittal plane (assuming XZ)
        while the robot base position, orientation, linear and angular velocities are mirrored across the world XZ
        plane, assuming this is the direction of motion in experiments.
        TODO: Parameterize for programmable mirror planes
        Note: If you override the `get_observation` function of your robot, you should also override this function.
        Returns:
            Mirrored observation of the same length as the original observation
        """
        if obs is not None:  # Avoid computing again centroidal momentum.
            base_pos_obs = q[[2]]
            # Mirror base orientation w.r.t world ZX plane
            base_ori_quat_mirror = q[3:7] * np.array([1, -1, 1, -1])
            base_ori_so3_mirror = np.array(self.bullet_client.getMatrixFromQuaternion(base_ori_quat_mirror)).reshape(
                (3, 3))
            base_ori_mirror_obs = base_ori_so3_mirror[:, :2].flatten(order='F')  #
            # Mirror COM linear and angular velocities and momentum w.r.t world XZ plane
            com_momentum_mirror_obs = obs[:6] * np.array([1, -1, 1, -1, 1, -1])
            com_vel = dq[0:6] * np.array([1, -1, 1, -1, 1, -1])
            # Mirror robot joints w.r.t sagittal plane symmetry
            q_mirror, dq_mirror = self.mirror_joints_sagittal(q, dq)
            # Get only actuated joints states
            q_mirror_obs = q_mirror[7:7 + self.nj]
            return np.concatenate((com_momentum_mirror_obs, base_pos_obs, base_ori_mirror_obs, q_mirror_obs))
        else:
            q_mirror, dq_mirror = self.mirror_base(q, dq)
            # Compute mirror observation analytically
            q_mirror, dq_mirror = self.mirror_joints_sagittal(q_mirror, dq_mirror)
            return self.get_observation(q=q_mirror, dq=dq_mirror)

    @property
    def mass(self) -> float:
        return float(np.sum([i.mass for i in self.pinocchio_robot.model.inertias]))  # [kg]

    @property
    @abstractmethod
    def joint_names(self) -> List:
        raise NotImplementedError()

    @property
    @abstractmethod
    def mirrored_joint_names(self) -> List:
        """
        List indicating the joint names of the sagittal plane reflected joints following the order of `joint_names`
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def mirror_joint_signs(self) -> List:
        """
        List indicating the direction changes in reflected joints w.r.t. the sagittal plane following the order
        of `joint_names`
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def torque_limits(self, q=None, dq=None) -> Union[float, Collection]:
        """
        Returns:
            max torques per dof (nj,)
        """

    @property
    @abstractmethod
    def acceleration_limits(self, q=None, dq=None) -> Union[float, Collection]:
        """
        Returns:
            maximum acceleration per dof (nj,)
        """

    @property
    @abstractmethod
    def velocity_limits(self, q=None, dq=None) -> np.array:
        """
        Returns:
            maximum velocity per dof (nj,)
        """
        if self._joint_vel_limits is None:
            raise NotImplementedError()
        return self._joint_vel_limits

    @property
    @abstractmethod
    def joint_pos_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the jactuated joints positional limits and neutral positions.
        Returns:
            - (nj,) lower_limits: Actuated joints lower positional limits
            - (nj,) upper_limits: Actuated joints upper positional limits
        """
        if self._joint_lower_limits == None or self._joint_upper_limits == None:
            raise NotImplementedError()
        return self._joint_lower_limits, self._joint_upper_limits

    @property
    @abstractmethod
    def endeff_names(self) -> Collection:
        """
        Getter for end effector names (usually robot feet)
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def mirrored_endeff_names(self) -> Collection:
        """
        Getter for end effector names (usually robot feet)
        """
        raise NotImplementedError()

    def mirror_base(self, q, dq, plane='XZ') -> [Collection, Collection]:
        if plane != 'XZ': raise NotImplementedError()

        q_mirror, dq_mirror = np.array(q), np.array(dq)
        # Mirror base orientation w.r.t world ZX plane
        q_mirror[3:7] = q[3:7] * np.array([1, -1, 1, -1])
        # Mirror base position w.r.t world ZX plane
        q_mirror[:3] = q[:3] * np.array([1, -1, 1])
        # Mirror COM linear and angular velocities and momentum w.r.t world XZ plane
        dq_mirror[:6] = dq[:6] * np.array([1, -1, 1, -1, 1, -1])
        return q_mirror, dq_mirror

    def mirror_joints_sagittal(self, q, dq) -> [Collection, Collection]:
        q_mirror, dq_mirror = np.array(q), np.array(dq)
        q_mirror[7:] = np.array(q[7:])[np.asarray(self.mirror_joint_idx)] * self.mirror_joint_signs
        dq_mirror[6:] = np.array(dq[6:])[np.asarray(self.mirror_joint_idx)] * self.mirror_joint_signs
        return q_mirror, dq_mirror

    def mirror_action(self, a) -> Collection:
        return np.array(a)[np.asarray(self.mirror_joint_idx)] * self.mirror_joint_signs

    def mirror_observation(self, a) -> Collection:
        return np.array(a)[np.asarray(self.mirror_obs_idx)] * self.mirror_obs_signs

    @property
    def mirror_endeff_idx(self) -> Collection:
        if self._mirror_endeff_idx is None:
            assert len(self.mirrored_endeff_names) == len(self.endeff_names)
            assert np.all([j in self.endeff_names for j in self.mirrored_endeff_names])
            self._mirror_endeff_idx = [self.endeff_names.index(j) for j in self.mirrored_endeff_names]
        return self._mirror_endeff_idx

    @property
    def mirror_obs_idx(self) -> List:
        """
        Provides the permutation indices for obtaining the obs symmetrical equivalent
        """
        if self._mirror_obs_idx is None:
            # obs = np.concatenate((com_momentum_obs, base_pos_obs, base_ori_obs, q_obs))
            momentum_idx = list(range(0, 6))
            base_pos_idx = [6]
            base_pos_ori_idx = list(range(7, 7 + 6))
            q_idx = list(np.array(self.mirror_joint_idx) + 13)
            self._mirror_obs_idx = momentum_idx + base_pos_idx + base_pos_ori_idx + q_idx
            assert len(np.unique(self._mirror_obs_idx)) == len(self._mirror_obs_idx), \
                "Appears to be missing indices: %s" % self._mirror_obs_idx
        return self._mirror_obs_idx

    @property
    def mirror_obs_signs(self) -> List:
        """
        Provides the signs to apply to the symmetric permutation of an observation
        """
        if self._mirror_obs_signs is None:
            # obs = np.concatenate((com_momentum_obs, base_pos_obs, base_ori_obs, q_obs))
            momentum_signs = [1., -1., 1., -1., 1., -1.]
            base_pos_signs = [1.]
            base_pos_ori_signs = [1., -1., 1., -1., 1., -1.]
            q_signs = self.mirror_joint_signs
            signs = momentum_signs + base_pos_signs + base_pos_ori_signs + q_signs
            assert np.logical_or(np.array(signs) == 1.0, np.array(signs) == -1.0).all() , \
                "Appears to be invalid sign: %s" % signs
            self._mirror_obs_signs = signs

        return self._mirror_obs_signs

    @property
    def mirror_joint_idx(self) -> Tuple:
        """
        Provides the permutation indices for obtaining the joints sagittal plane mirror equivalents in the order of
        `joint_names`
        """
        if self._mirror_joint_idx is None:
            assert len(self.mirrored_joint_names) == len(self.joint_names)
            assert np.all([j in self.joint_names for j in self.mirrored_joint_names])
            assert np.all([s == 1 or s == -1 for s in self.mirror_joint_signs]), self.mirror_joint_signs
            self._mirror_joint_idx = [self.joint_names.index(j) for j in self.mirrored_joint_names]
            assert len(np.unique(self._mirror_joint_idx)) == len(self._mirror_joint_idx), \
                "Appears to be missing indices: %s" % self._mirror_joint_idx

        return tuple(self._mirror_joint_idx)

    @property
    def bullet_client(self):
        if self._pb is None:
            raise RuntimeError("Robot bullet simulation not configured, remember to call `configure_bullet_simulation`")
        return self._pb

    def get_base_position_world(self):
        """Returns the position of the base in the world frame.
        Returns:
            np.array((3,)) with the translation
            np.array((4,)) with angular position in quaternion
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

    def stable_pd_control(self, q, dq, qa_des, dt, dqa_des=None):
        if dqa_des is None:
            dqa_des = [0.0] * self.nj
        # Code based on https://github.com/bulletphysics/bullet3/pull/3246

        # Compute -Kp((q + qdot * dt) - qdes)
        p_term = self._Kp.dot(qa_des - (q[7:] + dq[6:]*dt))
        # Compute -Kd(qdot - qdotdes)
        d_term = self._Kd.dot(dqa_des - dq[6:])

        # Compute/update all the joints and frames
        # Computing the mass matrix M and the dynamic drift b
        M = self.pinocchio_robot.mass(q)  # compute the mass matrix
        b = self.pinocchio_robot.nle(q, dq)  # compute the dynamic drift

        # TODO: For ideal case we could introduce GRF effect on generalized forces. Have to generate the
        #  Jacobian of contacts COP to generalized forces.

        # Obtain estimated generalized accelerations, considering Coriolis and Gravitational forces, and stable PD
        # actions
        # TODO: use pinocchio integrate method which i believe takes advantage of the sparcity.
        ddq_estimated = np.linalg.solve(a=(M[6:, 6:] + self._Kd * dt),
                                b=(-b[6:] + p_term + d_term))

        tau = p_term + (d_term - (self._Kd.dot(ddq_estimated) * dt))
        np.set_printoptions(precision=3)
        # log.debug(tau)
        return tau