import copy
import functools
import itertools
import random
from typing import Optional

import numpy as np
from emlp import Group
from torch.utils.data import Dataset

np.set_printoptions(precision=4)


class COMMomentum(Dataset):

    def __init__(self, robot, size=5000, angular_momentum=False, Gin: Optional[Group] = None,
                 Gout: Optional[Group] = None,
                 augment=False, dtype=np.float32):

        # self.robot = robot
        self.size = size
        self.dtype = dtype
        self.angular_momentum = angular_momentum
        self.robot = robot
        dq_max = robot.velocity_limits
        q_min, q_max = robot.joint_pos_limits
        q, dq = robot.get_init_config(random=False)
        # TODO: Temporal for debugging
        # base_ori = [0.0871557, 0, 0, 0.9961947]
        # q[3:7] = base_ori
        # We will assume base is fixed at initial config position for the entire dataset.
        self.base_q = q[:7]
        self.base_dq = dq[:6]

        Y = np.zeros((size, 6), dtype=self.dtype)
        X = np.zeros((size, robot.nj * 2), dtype=self.dtype)
        for i in range(size):
            q[7:] = np.random.uniform(q_min, q_max, robot.nj).astype(dtype=self.dtype)
            dq[6:] = np.random.uniform(-dq_max, dq_max, robot.nj).astype(dtype=self.dtype)
            hg = robot.pinocchio_robot.centroidalMomentum(q, dq).np.astype(dtype=self.dtype)

            Y[i, :] = hg
            X[i, :] = np.concatenate((q[7:], dq[6:]))

        self.X = X
        self.Y = Y

        self.augment = augment
        self.Gin = Gin
        self.Gout = Gout
        self.H = [(np.array(gin, dtype=self.dtype), np.array(gout, dtype=self.dtype)) for gin, gout in
                  zip(Gin.discrete_generators, Gout.discrete_generators)]
        self._pb = None  # GUI debug
        self.test_equivariance()

    def test_equivariance(self):
        augment = copy.copy(self.augment)

        self.augment = False  # Temporarily
        x, y = self[np.random.randint(0, self.size)]

        # TODO: Should be replaced by a correct Group representation and the `sample` method
        # Get all possible group actions
        samples_g_sequences = itertools.combinations_with_replacement(self.H, len(self.H))
        for g_sequence in samples_g_sequences:
            g_sequence = np.array(g_sequence, dtype=np.object)
            g_in_sequence, g_out_sequence = g_sequence[:, 0], g_sequence[:, 1]
            g_in = functools.reduce(np.dot, g_in_sequence)
            g_out = functools.reduce(np.dot, g_out_sequence)
            gx, gy = g_in @ x, g_out @ y
            gy_true = self.get_hg(*np.split(gx, 2))
            error = gy_true - gy
            if not np.allclose(error, 0.0, rtol=1e-4, atol=1e-4):
                self.gui_debug(*np.split(x, 2), *np.split(gx, 2), hg1=y, hg2=gy_true, ghg2=gy)
                raise AttributeError(f"Ground truth hg(q,dq) = Ag(q)dq is not equivariant to provided groups: \n" +
                                     f"x:{x}\ng*x:{gx}\ny:{y} \ng*y:{gy}\n" +
                                     f"Aq(g*q)g*dq:{gy_true}\nError:{error}")

        self.augment = augment  # Restore agumentation flag

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.angular_momentum:
            x, y = self.X[i, :], self.Y[i, :],
        else:
            x, y = self.X[i, :], self.Y[i, :3],

        if self.augment:
            if random.random() < 0.5:
                g_in, g_out = random.choice(self.H)
                x, y = g_in @ x, g_out @ y
        return x, y

    def get_hg(self, q, dq):
        hg = self.robot.pinocchio_robot.centroidalMomentum(q=np.concatenate((self.base_q, q)),
                                                           v=np.concatenate((self.base_dq, dq)))
        hg = np.array(hg).astype(self.dtype)
        if self.angular_momentum:
            return hg
        return hg[:3]

    def gui_debug(self, q1, dq1, q2, dq2, hg1, hg2, ghg2):
        from utils.utils import configure_bullet_simulation

        def tint_robot(robot, color=(0.227, 0.356, 0.450), alpha=0.5):
            num_joints = self.robot.bullet_client.getNumJoints(self.robot.robot_id)
            for i in range(num_joints):
                self._pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=i, rgbaColor=color + (alpha,))
            self._pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=-1, rgbaColor=color + (alpha,))

        def draw_momentum_vector(p1, p2, v_color, scale=1.0, show_axes=True, text=None):
            linewidth = 4
            x_color, y_color, z_color = (0, 1, 0), (1, 0, 0), (0, 0, 1)
            if show_axes:
                # x
                self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + np.array([p2[0], 0., 0.]) * scale,
                                          lineColorRGB=x_color, lineWidth=linewidth, lifeTime=0)
                # y
                self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + np.array([0., p2[1], 0.]) * scale,
                                          lineColorRGB=y_color, lineWidth=linewidth, lifeTime=0)
                # z
                self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + np.array([0., 0., p2[2]]) * scale,
                                          lineColorRGB=z_color, lineWidth=linewidth, lifeTime=0)
            # v
            self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + p2 * scale,
                                      lineColorRGB=v_color, lineWidth=linewidth, lifeTime=0)
            if text:
                self._pb.addUserDebugText(text=text, textPosition=p1 + p2 * scale + np.array([0, 0, 0.05]),
                                          textSize=1.2, lifeTime=0, textColorRGB=(0, 0, 0))

        if self._pb is None:
            self._pb = configure_bullet_simulation(gui=True)

        robot1 = self.robot
        robot2 = copy.copy(self.robot)
        offset = 1.0
        robot1.configure_bullet_simulation(self._pb, world=None)
        robot2.configure_bullet_simulation(self._pb, world=None)
        tint_robot(robot2, alpha=0.4)
        tint_robot(robot1, alpha=0.4)
        # Place robots in env
        q, dq = robot1.get_init_config(random=True)
        q[:7] = self.base_q
        dq[:6] = self.base_dq
        base_q1 = q[:7]
        base_q2 = np.array(base_q1)
        base_q2[1] += offset

        # Set positions:
        robot1.reset_state(np.concatenate((base_q1, q1)), dq)
        robot2.reset_state(np.concatenate((base_q2, q2)), dq)
        # Draw linear momentum
        draw_momentum_vector(base_q1[:3], hg1[:3], v_color=(0, 0, 0), scale=1.0, text=f"hg(q,dq)={hg1}")
        draw_momentum_vector(base_q2[:3], hg2[:3], v_color=(0, 0, 0), scale=1.0, text=f"hg(g*q, g*dq)={hg2}")
        draw_momentum_vector(base_q2[:3], ghg2[:3], v_color=(0.125, 0.709, 0.811), scale=1.0,
                             text=f"g*hg(q, dq)={ghg2}", show_axes=False)
        print("a")
