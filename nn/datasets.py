import copy
import functools
import itertools
import logging
import random
from typing import Optional

import numpy as np
import torch
from emlp import Group
from torch.utils.data import Dataset
import torch.nn.functional as F

import logging
log = logging.getLogger(__name__)

np.set_printoptions(precision=4)


class COMMomentum(Dataset):

    def __init__(self, robot, size=5000, angular_momentum=False, Gin: Optional[Group] = None,
                 Gout: Optional[Group] = None, normalize=True,
                 augment=False, dtype=np.float32):

        # self.robot = robot
        self.size = size
        self.dtype = dtype
        self.angular_momentum = angular_momentum
        self.robot = robot
        self._normalize = normalize

        self.augment = augment
        self.Gin = Gin
        self.Gout = Gout
        self.H = [(np.array(gin, dtype=self.dtype), np.array(gout, dtype=self.dtype)) for gin, gout in
                  zip(Gin.discrete_generators, Gout.discrete_generators)]
        self._pb = None  # GUI debug

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

        self.test_equivariance()
        X, Y = self.compute_normalization(X, Y)

        self.X = X
        self.Y = Y[:, :(6 if angular_momentum else 3)]

        self.loss_fn = F.mse_loss

        log.info(f"CoM[{robot.__class__.__name__}]-Samples:{size}-Aug:{'True' if augment else 'False'}-"
                 f"Normalize:{normalize}")

    def compute_normalization(self, X, Y):
        idx = 6 if self.angular_momentum else 3
        self._X_mean, self._Y_mean, self._X_std, self._Y_std = 0., 0., 1., 1.
        if self._normalize:
            X_aug = np.vstack([X] + [np.asarray(g @ X.T).T for g in self.Gin.discrete_generators])
            Y_aug = np.vstack(
                [Y[:, :idx]] + [np.asarray(g @ Y[:, :idx].T).T for g in self.Gout.discrete_generators])
            self._X_mean = np.mean(X_aug, axis=0)
            self._Y_mean = np.mean(Y_aug[:, :idx], axis=0)

            self._X_std = np.std(X_aug, axis=0)
            self._Y_std = np.std(Y_aug, axis=0)

            X, Y = self.normalize(X, Y[:, :idx])
        return X, Y
        # X_mean = np.mean(self.X, axis=0)
        # Y_mean = np.mean(self.Y[:, :idx], axis=0)

        # X_std = np.std(self.X, axis=0)
        # Y_std = np.std(self.Y[:, :idx], axis=0)
        #
        # # TODO: Change for orbit
        # gX_mean = [np.asarray(g @ X_mean) for g in self.Gin.discrete_generators]
        # gX_mean.append(X_mean)
        # gY_mean = [np.asarray(g @ Y_mean) for g in self.Gout.discrete_generators]
        # gY_mean.append(Y_mean)
        #
        # gX_std = [np.asarray(g @ X_std) for g in self.Gin.discrete_generators]
        # gX_std.append(X_std)
        # gY_std = [np.asarray(g @ Y_std) for g in self.Gout.discrete_generators]
        # gY_std.append(Y_std)
        #
        # GX_mean, GY_mean = np.mean(np.asarray(gX_mean), axis=0), np.mean(np.asarray(gY_mean), axis=0)
        # GX_std, GY_std = np.mean(np.asarray(gX_std), axis=0), np.mean(np.asarray(gY_std), axis=0)

    def normalize(self, x, y):
        return (x - self._X_mean) / self._X_std, (y - self._Y_mean) / self._Y_std

    def denormalize(self, xn, yn):
        return xn*self._X_std + self._X_mean, yn*self._Y_std + self._Y_mean

    def test_equivariance(self):
        augment = copy.copy(self.augment)

        self.augment = False  # Temporarily
        decimals = 15
        q, dq = self.robot.get_init_config(random=True)
        x = np.concatenate((q[7:]*1.4, dq[6:] * 10))
        x = np.round(x, decimals).astype(np.float64)
        y = self.get_hg(*np.split(x, 2))
        y = np.round(y, decimals).astype(np.float64)

        # TODO: Should be replaced by a correct Group representation and the `sample` method
        # Get all possible group actions
        samples_g_sequences = self.H # itertools.combinations_with_replacement(self.H, len(self.H))
        for g_sequence in [[h] for h in self.H]:
            g_sequence = np.array(g_sequence, dtype=np.object)
            g_in_sequence, g_out_sequence = g_sequence[:, 0], g_sequence[:, 1]
            g_in = functools.reduce(np.dot, g_in_sequence).astype(int)
            g_in_q = g_in[:12,:12].astype(np.int)
            g_out = functools.reduce(np.dot, g_out_sequence).astype(int)
            gx, gy = np.round(g_in @ x, decimals), np.round(g_out @ y, decimals)
            gy_true = self.get_hg(*np.split(gx, 2))
            error = gy_true - gy
            if not np.allclose(error, 0.0, rtol=1e-3, atol=1e-3):
                try:
                    self.gui_debug(*np.split(x, 2), *np.split(gx, 2), hg1=y, hg2=gy_true, ghg2=gy)
                except Exception as e:
                    logging.warning(f"Unable to start GUI of pybullet: {str(e)}")
                raise AttributeError(f"Ground truth hg(q,dq) = Ag(q)dq is not equivariant to provided groups: \n" +
                                     f"x:{x}\ng*x:{gx}\ny:{y} \ng*y:{gy}\n" +
                                     f"Aq(g*q)g*dq:{gy_true}\nError:{error}")

        self.augment = augment  # Restore agumentation flag

    def compute_metrics(self, y, y_pred) -> dict:
        with torch.no_grad():
            metrics = {}
            # eps = 1e-9
            metrics["lin_cos_sim"] = F.cosine_similarity(y[:, :3], y_pred[:, :3], dim=-1)
            metrics["lin_err"] = torch.linalg.norm((y[:, :3] - y_pred[:, :3]) * y.new(self._Y_std[:3]), dim=-1)
            if self.angular_momentum:
                metrics["ang_cos_sim"] = F.cosine_similarity(y[:, 3:], y_pred[:, 3:], dim=-1)
                metrics["ang_err"] = torch.linalg.norm((y[:, 3:] - y_pred[:, 3:]) * y.new(self._Y_std[3:]), dim=-1)
        return metrics

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

        def draw_momentum_vector(p1, p2, v_color, scale=1.0, show_axes=True, text=None, offset=0.0):
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
                self._pb.addUserDebugText(text=text, textPosition=p1 + p2 * scale + np.array([0, 0, 0.2 + offset]),
                                          textSize=1.2, lifeTime=0, textColorRGB=(0, 0, 0))
        if self._pb:
            self._pb.resetSimulation()
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
        draw_momentum_vector(base_q1[:3], hg1[:3], v_color=(0, 0, 0), scale=5.0, text=f"hg(q,dq)={hg1}")
        draw_momentum_vector(base_q2[:3], hg2[:3], v_color=(0, 0, 0), scale=5.0, text=f"hg(g*q, g*dq)={hg2}")
        draw_momentum_vector(base_q2[:3], ghg2[:3], v_color=(0.125, 0.709, 0.811), scale=5.0,
                             text=f"g*hg(q, dq)={ghg2}", show_axes=False, offset=0.2)
        print("a")
