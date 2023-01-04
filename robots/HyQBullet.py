import copy
import os
import pathlib
import time
from contextlib import contextmanager
from math import pi
from typing import Collection, Tuple, Union, Optional, List

import numpy as np  # Numpy library
import scipy.spatial.transform
import torch
from pinocchio import Quaternion, Force, JointModelFreeFlyer
from pinocchio.robot_wrapper import RobotWrapper
from pybullet_utils.bullet_client import BulletClient

from robot_descriptions.loaders.pinocchio import load_robot_description as pin_load_robot_description
from robot_descriptions.loaders.pybullet import load_robot_description as pb_load_robot_description
from robots.PinBulletWrapper import PinBulletWrapper

class HyQBullet(PinBulletWrapper):

    def __init__(self,  resources: pathlib.Path, control_mode=ControlMode('torque'), power_coeff=1.0,
                 reference_robot: Optional['PinBulletWrapper'] = None,
                 gen_xacro=False, useFixedBase=False, **kwargs):
        self._mass = np.NAN
        self.power_coeff = power_coeff
        if gen_xacro:
            raise NotImplementedError()

        # Super initialization: will call load_bullet_robot and load_pinocchio_robot
        super(HyQBullet, self).__init__(resources=resources, recontrol_mode=control_mode, useFixedBase=useFixedBase,
                                        reference_robot=reference_robot, **kwargs)

        self._masses = [M.mass for M in self.pinocchio_robot.model.inertias][1:]

    def load_pinocchio_robot(self, reference_robot: Optional['PinBulletWrapper'] = None) -> RobotWrapper:

        pin_robot = pin_load_robot_description("hyq_description", root_joint=JointModelFreeFlyer())
                # pin_robot = RobotWrapper.BuildFromURDF(str(urdf_path.absolute()), str(meshes_path.absolute()),
                #                                        JointModelFreeFlyer(), verbose=True)
        self._mass = float(np.sum([i.mass for i in pin_robot.model.inertias]))  # [kg]
        return pin_robot


    def load_bullet_robot(self, base_pos=None, base_ori=None) -> int:
        if base_ori is None: base_ori = [0, 0, 0, 1]
        if base_pos is None: base_pos = [0, 0, .4]

        robot_id = pb_load_robot_description("hyq_description", basePosition=base_pos,
                                             baseOrientation=base_ori,
                                             flags=self.bullet_client.URDF_USE_INERTIA_FROM_FILE |
                                                   self.bullet_client.URDF_USE_SELF_COLLISION,
                                             useFixedBase=self.useFixedBase)
        return robot_id

    def configure_bullet_simulation(self, bullet_client: BulletClient, world, base_pos=None, base_ori=None):
        super(HyQBullet, self).configure_bullet_simulation(bullet_client, world, base_pos, base_ori)
        # Initial config
        q0, dq0 = self.get_init_config(random=True)
        q0[:2] += base_pos[:2] if base_pos is not None else [0, 0]
        self.reset_state(q0, dq0)

        self.bullet_ids_allowed_floor_contacts += []

        num_joints = self.bullet_client.getNumJoints(self.robot_id)
        robot_color = [0.054, 0.415, 0.505, 1.0]
        FL_leg_color = [0.698, 0.376, 0.082, 1.0]
        FR_leg_color = [0.260, 0.263, 0.263, 1.0]
        HL_leg_color = [0.800, 0.480, 0.000, 1.0]
        HR_leg_color = [0.710, 0.703, 0.703, 1.0]
        endeff_color = [0, 0, 0, 1]
        for i in range(num_joints):
            link_name = self.bullet_client.getJointInfo(self.robot_id, i)[12].decode("UTF-8")
            joint_name = self.bullet_client.getJointInfo(self.robot_id, i)[1].decode("UTF-8")
            print(f"{joint_name}-{link_name}")
            if "foot" in link_name:
                color = endeff_color
            elif 'lf' in joint_name:
                color = FL_leg_color
            elif 'rf' in joint_name:
                color = FR_leg_color
            elif 'lh' in joint_name:
                color = HL_leg_color
            elif 'rh' in joint_name:
                color = HR_leg_color
            else:
                color = robot_color
            self.bullet_client.changeVisualShape(objectUniqueId=self.robot_id,
                                                 linkIndex=i,
                                                 rgbaColor=color,
                                                 specularColor=[0, 0, 0])


    @property
    def hip_height(self) -> float:
        return 0.85  # [m]

    @property
    def joint_names(self) -> List:
        return ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint', 'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint', 'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint',]

    @property
    def endeff_names(self) -> List:
        return ['lf_foot_joint', 'rf_foot_joint', 'lh_foot_joint', 'rh_foot_joint']

    def get_init_config(self, random=False):
        f_leg_pos = np.array([0.0, 0.7, -1.5])
        h_leg_pos = np.array([0.0, -0.7, 1.5])
        leg_pos_offset = lambda: (2*np.random.rand(3) - 1) * [np.deg2rad(30),
                                                              np.deg2rad(30),
                                                              np.deg2rad(25)] if random else np.zeros(3)
        leg_vel = np.array([0.0, 0.0, 0.0])
        leg_vel_offset1 = np.random.uniform(-np.deg2rad(3), np.deg2rad(3), 3) if random else np.zeros(3)
        leg_vel_offset2 = np.random.uniform(-np.deg2rad(3), np.deg2rad(3), 3) if random else np.zeros(3)

        base_ori = [0, 0, 0, 1]
        if random:
            pitch = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25))
            roll = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25))
            yaw = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25))
            base_ori = scipy.spatial.transform.Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()

        q_legs = np.concatenate((f_leg_pos + leg_pos_offset(),
                                 h_leg_pos + leg_pos_offset(),
                                 f_leg_pos + leg_pos_offset(),
                                 h_leg_pos + leg_pos_offset()))
        dq_legs = np.concatenate((leg_vel + leg_vel_offset1, leg_vel + leg_vel_offset2, leg_vel + leg_vel_offset2,
                                  leg_vel + leg_vel_offset1))

        q0 = np.array([0., 0., self.hip_height + 0.02] + list(base_ori) + list(q_legs))
        dq0 = np.array([0., 0., 0., 0., 0., 0.] + list(dq_legs))

        return q0, dq0



