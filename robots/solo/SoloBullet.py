# coding: utf8
import copy
import pathlib
import time
from math import pi
from typing import Collection, Tuple, Union, Optional, List

import numpy as np  # Numpy library
import scipy.spatial.transform
import torch
from hydra.utils import get_original_cwd
from pinocchio import Quaternion, Force, JointModelFreeFlyer
from pinocchio.robot_wrapper import RobotWrapper
from pybullet_utils.bullet_client import BulletClient

from ..AccelerationBounds import BatchedContraintAccelerationBound
from ..PinBulletWrapper import PinBulletWrapper, ControlMode

class Solo8Bullet(PinBulletWrapper):
    urdf_subpath = 'urdf/solo8.urdf'
    # Servo variables.
    # tau = I * Kt * N
    MOTOR_TORQUE_COST = 0.025  # [Nm/A]
    MOTOR_GEAR_REDUCTION = 9.0  # [] -> N
    MAX_CURRENT = 12  # [A] Servo max current.
    MAX_QREF = pi
    MOTOR_INERTIA = 0.0000045

    max_joint_acc = np.array([605, 4000, 605, 4000, 605, 4000, 605, 4000])  # [rad/s^2]
    # max_joint_acc = [100] * 8  # [rad/s^2]
    cam_distance = 0.8

    # Robot model variables
    BASE_LINK_NAME = "base_link"
    EEF_NAMES = ["HL_ANKLE", "HR_ANKLE", "FL_ANKLE", "FR_ANKLE"]
    ALLOWED_CONTACT_JOINTS_NAMES = ["FL_KFE", "FR_KFE", "HL_KFE", "HR_KFE"]
    JOINT_NAMES = ["FL_HFE", "FL_KFE", "FR_HFE", "FR_KFE", "HL_HFE", "HL_KFE", "HR_HFE", "HR_KFE"]

    def __init__(self, resources:pathlib.Path, control_mode=ControlMode('torque'), power_coeff=1.0,
                 reference_robot: Optional['PinBulletWrapper']=None, gen_xacro=False, useFixedBase=False, **kwargs):
        self._mass = np.NAN
        self.power_coeff = power_coeff
        if gen_xacro:
            import robot_properties_solo.utils
            robot_properties_solo.utils.build_xacro_files(self.resources.resources_dir)
        # Super initialization: will call load_bullet_robot and load_pinocchio_robot
        super(Solo8Bullet, self).__init__(resources=resources, control_mode=control_mode, useFixedBase=useFixedBase,
                                          reference_robot=reference_robot, **kwargs)

        self._max_servo_torque = np.array([self.MOTOR_TORQUE_COST * self.MOTOR_GEAR_REDUCTION *
                                           self.MAX_CURRENT * self.power_coeff] * self.nj)
        self._max_servo_speed = 4 * pi

        self._masses = [M.mass for M in self.pinocchio_robot.model.inertias][1:]
        # Stable PD control variables ____________
        kps = [8.5, 8.5] * 4
        kds = [0.3, 0.3] * 4
        self._Kp = np.diagflat(kps)
        self._Kd = np.diagflat(kds)
        self._pos_upper_limit = None
        self._pos_neutral = None

    def load_pinocchio_robot(self, reference_robot: Optional['PinBulletWrapper']=None) -> RobotWrapper:
        if reference_robot is not None:
            import sys
            assert np.all(self.joint_names == reference_robot.joint_names), "Invalid reference RobotWrapper"
            pin_robot = copy.copy(reference_robot.pinocchio_robot)
            pin_robot.data = copy.deepcopy(reference_robot.pinocchio_robot.data)
            assert sys.getrefcount(pin_robot.data) <= 2
        else:
            # time.sleep(np.random.rand() * 5)
            urdf_path = self.resources / self.urdf_subpath
            assert urdf_path.exists(), f"Cannot find urdf file {urdf_path.absolute()}"
            meshes_path = self.resources
            pin_robot = RobotWrapper.BuildFromURDF(str(urdf_path.absolute()), str(meshes_path.absolute()),
                                                   JointModelFreeFlyer(), verbose=True)
            pin_robot.model.rotorInertia[6:] = self.MOTOR_INERTIA
            pin_robot.model.rotorGearRatio[6:] = self.MOTOR_GEAR_REDUCTION
            print(" -- Done")

        self._mass = float(np.sum([i.mass for i in pin_robot.model.inertias]))  # [kg]
        return pin_robot

    def load_bullet_robot(self, base_pos=None, base_ori=None) -> int:
        if base_ori is None: base_ori = [0, 0, 0, 1]
        if base_pos is None: base_pos = [0, 0, 0.95]

        urdf_path = self.resources / self.urdf_subpath
        assert urdf_path.exists(), f"Cannot find urdf file {urdf_path.absolute()}"
        meshes_path = self.resources

        # Load the robot for PyBullet
        self.bullet_client.setAdditionalSearchPath(str(meshes_path.absolute()))
        self.robot_id = self.bullet_client.loadURDF(str(urdf_path.absolute()),
                                                    basePosition=base_pos,
                                                    baseOrientation=base_ori,
                                                    flags=self.bullet_client.URDF_USE_INERTIA_FROM_FILE |
                                                          self.bullet_client.URDF_USE_SELF_COLLISION,
                                                    useFixedBase=self.useFixedBase)

        return self.robot_id

    def configure_bullet_simulation(self, bullet_client: BulletClient, world, base_pos=None, base_ori=None):
        super(Solo8Bullet, self).configure_bullet_simulation(bullet_client, world, base_pos, base_ori)
        # Initial config
        q0, dq0 = self.get_init_config(random=False)
        q0[:2] += base_pos[:2] if base_pos is not None else [0, 0]
        self.reset_state(q0, dq0)

        self.bullet_ids_allowed_floor_contacts += [self.joint_aux_vars[j].bullet_id for j in self.ALLOWED_CONTACT_JOINTS_NAMES]

        num_joints = self.bullet_client.getNumJoints(self.robot_id)
        robot_color = [0.227, 0.356, 0.450, 1.0]
        left_leg_color = [0.427, 0.521, 0.592, 1.0]
        endeff_color = [0, 0, 0, 1]
        for i in range(num_joints):
            link_name = self.bullet_client.getJointInfo(self.robot_id, i)[12].decode("UTF-8")
            joint_name = self.bullet_client.getJointInfo(self.robot_id, i)[1].decode("UTF-8")

            if "FOOT" in link_name:
                color = endeff_color
            elif 'L' in joint_name:
                color = left_leg_color
            elif 'R' in joint_name:
                color = robot_color
            else:
                color = robot_color

            self.bullet_client.changeVisualShape(objectUniqueId=self.robot_id,
                                                 linkIndex=i,
                                                 rgbaColor=color,
                                                 specularColor=[0, 0, 0])

        self.bullet_client.changeVisualShape(objectUniqueId=self.robot_id,
                                             linkIndex=-1,
                                             rgbaColor=robot_color,
                                             specularColor=[0, 0, 0])

    def get_observation(self, q=None, dq=None) -> Collection:
        return super(Solo8Bullet, self).get_observation(q, dq)

    @property
    def torque_limits(self, q=None, dq=None) -> Union[float, Collection]:
        return self._max_servo_torque

    @property
    def acceleration_limits(self, q=None, dq=None) -> Union[float, Collection]:
        return self.max_joint_acc

    @property
    def velocity_limits(self, q=None, dq=None) -> Union[float, Collection]:
        return np.array([self._max_servo_speed] * self.nj),

    @property
    def joint_pos_limits(self) -> Tuple[Collection, Collection]:
        # Override URDF values
        return np.array([-pi] * self.nj), np.array([pi] * self.nj)

    @property
    def hip_height(self) -> float:
        return 0.24  # [m]

    @property
    def joint_names(self) -> List:
        return Solo8Bullet.JOINT_NAMES

    @property
    def mirrored_joint_names(self) -> List:
        return ["FR_HFE", "FR_KFE", "FL_HFE", "FL_KFE", "HR_HFE", "HR_KFE", "HL_HFE", "HL_KFE"]

    @property
    def mirrored_endeff_names(self) -> Collection:
        return ["HR_ANKLE", "HL_ANKLE", "FR_ANKLE", "FL_ANKLE"]

    @property
    def mirror_joint_signs(self) -> List:
        return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    @property
    def endeff_names(self) -> List:
        return Solo8Bullet.EEF_NAMES

    def get_init_config(self, random=False):

        leg_pos = np.array([0.8, -1.6])
        leg_pos_offset1 = np.random.rand(2) * [-np.deg2rad(15), -np.deg2rad(15)] if random else [0, 0]
        leg_pos_offset2 = np.random.rand(2) * [np.deg2rad(15), np.deg2rad(15)] if random else [0, 0]
        leg_vel = np.array([0.0, 0.0])
        leg_vel_offset1 = np.random.uniform(-np.deg2rad(3), np.deg2rad(3), 2) if random else [0, 0]
        leg_vel_offset2 = np.random.uniform(-np.deg2rad(3), np.deg2rad(3), 2) if random else [0, 0]

        base_ori = [0, 0, 0, 1]
        if random:
            pitch = np.random.uniform(low=-np.deg2rad(5), high=np.deg2rad(5))
            roll = np.random.uniform(low=-np.deg2rad(5), high=np.deg2rad(5))
            base_ori = scipy.spatial.transform.Rotation.from_euler("xyz", [roll, pitch, 0]).as_quat()

        q_legs = np.concatenate((leg_pos + leg_pos_offset1, leg_pos + leg_pos_offset2, leg_pos + leg_pos_offset2,
                                 leg_pos + leg_pos_offset1))
        dq_legs = np.concatenate((leg_vel + leg_vel_offset1, leg_vel + leg_vel_offset2, leg_vel + leg_vel_offset2,
                                  leg_vel + leg_vel_offset1))

        q0 = np.array([0., 0., self.hip_height + 0.02] + list(base_ori) + list(q_legs))
        dq0 = np.array([0, 0., 0., 0., 0., 0.] + list(dq_legs))

        if random and np.random.rand() > 0.5:
            q0, dq0 = self.mirror_base(q0, dq0)
            q0, dq0 = self.mirror_joints_sagittal(q0, dq0)

        return q0, dq0

    @property
    def mass(self) -> float:
        if np.isnan(self._mass):
            return super().mass
        return self._mass