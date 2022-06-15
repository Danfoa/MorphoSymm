import copy
import logging
import pathlib
import random
import time
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import issparse
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from groups.SymmetricGroups import Sym
from utils.utils import dense

log = logging.getLogger(__name__)

np.set_printoptions(precision=4)


class Standarizer:

    def __init__(self, X_mean, X_std, Y_mean, Y_std, device):
        self.X_mean, self.X_std = torch.tensor(X_mean).to(device), torch.tensor(X_std).to(device)
        self.Y_mean, self.Y_std = torch.tensor(Y_mean).to(device), torch.tensor(Y_std).to(device)

    def transform(self, x=None, y=None):
        if isinstance(x, np.ndarray):
            X_mean, X_std = self.X_mean.cpu().numpy(), self.X_std.cpu().numpy()
            Y_mean, Y_std = self.Y_mean.cpu().numpy(), self.Y_std.cpu().numpy()
        else:
            X_mean, X_std = self.X_mean, self.X_std
            Y_mean, Y_std = self.Y_mean, self.Y_std

        if x is not None and y is not None:
            return (x - X_mean) / X_std, (y - Y_mean) / Y_std
        elif x is not None:
            return (x - X_mean) / X_std
        elif y is not None:
            return (y - Y_mean) / Y_std

    def unstandarize(self, xn=None, yn=None):
        if isinstance(xn, np.ndarray):
            X_mean, X_std = self.X_mean.cpu().numpy(), self.X_std.cpu().numpy()
            Y_mean, Y_std = self.Y_mean.cpu().numpy(), self.Y_std.cpu().numpy()
        else:
            X_mean, X_std = self.X_mean, self.X_std
            Y_mean, Y_std = self.Y_mean, self.Y_std

        if xn is not None and yn is not None:
            return xn * X_std + X_mean, yn * Y_std + Y_mean
        elif xn is not None:
            return xn * X_std + X_mean
        elif yn is not None:
            return yn * Y_std + Y_mean


class COMMomentum(Dataset):

    def __init__(self, robot, Gin: Sym, Gout: Sym, type='train',
                 angular_momentum=True, standarizer: Union[bool, Standarizer] = True, augment=False,
                 train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, samples=100000,
                 dtype=torch.float32, data_path="datasets/com_momentum", device='cpu'):

        self.dataset_type = type
        self.dtype = dtype
        self.angular_momentum = angular_momentum
        self.robot = robot
        self.normalize = True if isinstance(standarizer, Standarizer) else standarizer

        self.Gin = Gin
        self.Gout = Gout
        self.group_actions = [(np.asarray(gin), np.asarray(gout)) for gin, gout in zip(self.Gin.discrete_actions,
                                                                                       self.Gout.discrete_actions)]
        augmentation_actions = []
        for gin, gout in zip(self.Gin.discrete_actions, self.Gout.discrete_actions):
            augmentation_actions.append((torch.tensor(np.asarray(dense(gin))).to(device),
                                         torch.tensor(np.asarray(dense(gout))).to(device)))
        self.t_group_actions = augmentation_actions
        self._pb = None  # GUI debug
        self.augment = augment

        self._samples = samples
        self.dataset_path = pathlib.Path(data_path).joinpath(f"data_{samples:d}.npz")
        self.ensure_dataset_existance()
        partition_path = self.ensure_dataset_partition(train_ratio, test_ratio, val_ratio)

        # Load data
        assert type.lower() in ["train", "test", "val"], f"type must be one of these [train, test, val]"
        file_path = partition_path.joinpath(f"{type}.npz")
        assert file_path.exists(), file_path.absolute()
        # Loading on multiple threads might lead to issues -------------------
        trials = 5
        data = None
        while data is None and trials > 0:
            try:
                data = np.load(str(file_path))
            except:
                trials -= 1
                time.sleep(np.random.random())
        #  -------------------------------------------------------------------
        X, Y = data['X'], data['Y']

        q, dq = robot.get_init_config(random=False)
        self.base_q = q[:7]
        self.base_dq = dq[:6]

        # Normalize Data
        if isinstance(standarizer, Standarizer):
            self.standarizer = standarizer
        elif isinstance(standarizer, bool):
            X_mean, X_std, Y_mean, Y_std = self.compute_normalization(X, Y)
            self.standarizer = Standarizer(X_mean, X_std, Y_mean, Y_std, device=device)

        X, Y = self.standarizer.transform(X, Y)
        self.X = torch.from_numpy(X).type('torch.FloatTensor').to(device)
        self.Y = torch.from_numpy(Y).type('torch.FloatTensor').to(device)

        self.loss_fn = F.mse_loss

        log.info(str(self))

    def compute_normalization(self, X, Y):
        idx = 6 if self.angular_momentum else 3
        X_mean, Y_mean, X_std, Y_std = 0., 0., 1., 1.
        if self.normalize:
            # TODO: Obtain analytic formula for mean and std along orbit of discrete and continuous groups.
            X_aug = np.vstack([X] + [np.asarray(g @ X.T).T for g in self.Gin.discrete_actions])
            Y_aug = np.vstack([Y[:, :idx]] + [np.asarray(g @ Y[:, :idx].T).T for g in self.Gout.discrete_actions])
            X_mean = np.mean(X_aug, axis=0)
            Y_mean = np.mean(Y_aug[:, :idx], axis=0)
            X_std = np.std(X_aug, axis=0)
            Y_std = np.std(Y_aug, axis=0)

        return X_mean, X_std, Y_mean, Y_std

    def test_equivariance(self):
        trials = 10
        for trial in range(trials):
            q, dq = self.robot.get_init_config(random=True)

            x = np.concatenate((q[7:], dq[6:]))
            x = x.astype(np.float64)
            y = self.get_hg(*np.split(x, 2))
            y = y.astype(np.float64)

            # Get all possible group actions
            for g_in, g_out in zip(self.Gin.discrete_actions[1:], self.Gout.discrete_actions[1:]):
                g_in, g_out = (g_in.todense(), g_out.todense()) if issparse(g_in) else (g_in, g_out)
                gx, gy = np.asarray(g_in) @ x, np.asarray(g_out) @ y
                assert gx.dtype == x.dtype, (gx.dtype, x.dtype)
                assert gy.dtype == y.dtype, (gy.dtype, y.dtype)
                ggx, ggy = g_in.astype(np.float64) @ gx, g_out.astype(np.float64) @ gy
                # Check if there is numerical error in group actions application.
                action_error_x = ggx - x
                action_error_y = ggy - y
                gy_true = self.get_hg(*np.split(gx, 2))
                assert gy_true.dtype == y.dtype, (gy_true.dtype, y.dtype)
                error = gy_true - gy
                rel_error_norm = np.linalg.norm(error) / np.linalg.norm(gy_true)
                cos_sim = np.dot(gy, gy_true) / (np.linalg.norm(gy_true) * np.linalg.norm(gy))
                if rel_error_norm > 0.05 and cos_sim <= 0.95:
                    try:
                        self.gui_debug(*np.split(x, 2), *np.split(gx, 2), hg1=y, hg2=gy_true, ghg2=gy)
                    except Exception as e:
                        logging.warning(f"Unable to start GUI of pybullet: {str(e)}")
                    raise AttributeError(f"Ground truth hg(q,dq) = Ag(q)dq is not equivariant to provided groups: \n" +
                                         f"x:{x}\ng*x:{gx}\ny:{y} \ng*y:{gy}\n" +
                                         f"Aq(g*q,g*dq):{gy_true}\nError:{error}")
        return None

    def compute_metrics(self, y, y_pred) -> dict:
        with torch.no_grad():
            metrics = {}

            y_dn = self.standarizer.unstandarize(yn=y)
            y_pred_dn = self.standarizer.unstandarize(yn=y_pred)

            lin, lin_pred = y_dn[:, :3], y_pred_dn[:, :3]
            metrics["lin_cos_sim"] = torch.mean(F.cosine_similarity(lin, lin_pred, dim=-1))
            metrics["lin_err"] = torch.mean(torch.linalg.norm(lin - lin_pred, dim=-1))

            ang, ang_pred = y_dn[:, 3:], y_pred_dn[:, 3:]
            metrics["ang_cos_sim"] = torch.mean(F.cosine_similarity(ang, ang_pred, dim=-1))
            metrics["ang_err"] = torch.mean(torch.linalg.norm(ang - ang_pred, dim=-1))
        return metrics

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, i):
        if self.angular_momentum:
            x, y = self.X[i, :], self.Y[i, :],
        else:
            x, y = self.X[i, :], self.Y[i, :3],

        # if self.augment:  # Sample uniformly among symmetry actions including identity
        #     g_in, g_out = random.choice(self.group_actions)
        #     x, y = (g_in @ x).astype(self.dtype), (g_out @ y).astype(self.dtype)
        return x, y

    def collate_fn(self, batch):
        # Enforce data type in batched array
        # Small hack to do batched augmentation. TODO: Although efficient this should be done somewhere else.
        x_batch, y_batch = default_collate(batch)

        if self.augment:  # Sample uniformly among symmetry actions including identity
            g_in, g_out = random.choice(self.t_group_actions)
            g_x_batch = torch.matmul(x_batch.unsqueeze(1), g_in.unsqueeze(0).to(x_batch.dtype)).squeeze()
            g_y_batch = torch.matmul(y_batch.unsqueeze(1), g_out.unsqueeze(0).to(y_batch.dtype)).squeeze()
            # x, xx = x_batch[0], g_x_batch[0]
            # y, yy = y_batch[0], g_y_batch[0]
            x_batch, y_batch = g_x_batch, g_y_batch
        return x_batch.to(self.dtype), y_batch.to(self.dtype)

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, augment: bool):
        self._augment = augment
        log.debug(f"Dataset Group Augmentation {'ACTIVATED' if self.augment else 'DEACTIVATED'}")

    def get_hg(self, q, dq):
        hg = self.robot.pinocchio_robot.centroidalMomentum(q=np.concatenate((self.base_q, q)),
                                                           v=np.concatenate((self.base_dq, dq)))
        hg = np.array(hg)
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
        offset = 2 * self.robot.hip_height
        robot1.configure_bullet_simulation(self._pb, world=None)
        robot2.configure_bullet_simulation(self._pb, world=None)
        # tint_robot(robot2, alpha=0.9)
        # tint_robot(robot1, alpha=0.9)
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
        draw_momentum_vector(base_q1[:3], hg1[:3], v_color=(0, 0, 0), scale=1 / np.linalg.norm(hg1[:3]),
                             text=f"hg(q,dq)={hg1}")
        draw_momentum_vector(base_q2[:3], hg2[:3], v_color=(0, 0, 0), scale=1 / np.linalg.norm(hg2[:3]),
                             text=f"hg(g*q, g*dq)={hg2}")
        draw_momentum_vector(base_q2[:3], ghg2[:3], v_color=(0.125, 0.709, 0.811), scale=1 / np.linalg.norm(ghg2[:3]),
                             text=f"g*hg(q, dq)={ghg2}", show_axes=False, offset=0.2)
        draw_momentum_vector(base_q2[:3], hg2[3:], v_color=(0, .5, .5), show_axes=False,
                             scale=1 / np.linalg.norm(hg2[3:]))
        draw_momentum_vector(base_q2[:3], ghg2[3:], v_color=(0.125, 0.709, 0.811), scale=1 / np.linalg.norm(ghg2[3:]),
                             show_axes=False, offset=0.2)

        print("a")

    def ensure_dataset_existance(self):
        q, dq = self.robot.get_init_config(random=False)
        self.base_q = q[:7]
        self.base_dq = dq[:6]
        if self.dataset_path.exists():
            log.debug(f"Loading dataset of size {self._samples}")
        else:
            log.info(f"Generating dataset for {self.robot.__class__.__name__} of size {self._samples} samples")
            self.dataset_path.parent.mkdir(exist_ok=True, parents=True)
            # Ensure deterministic generation
            np.random.seed(29081995)
            # Get joint limits.
            dq_max = np.asarray(self.robot.velocity_limits)
            q_min, q_max = (np.asarray(lim) for lim in self.robot.joint_pos_limits)
            # Generate random configuration samples.
            Y = np.zeros((self._samples, 6))
            X = np.zeros((self._samples, self.robot.nj * 2))
            for i in range(self._samples):
                q[7:] = np.random.uniform(q_min, q_max, size=None)
                dq[6:] = np.random.uniform(-dq_max, dq_max, size=None)
                hg = self.robot.pinocchio_robot.centroidalMomentum(q, dq)

                Y[i, :] = hg.np
                X[i, :] = np.concatenate((q[7:], dq[6:]))

            np.savez_compressed(str(self.dataset_path), X=X, Y=Y)
            log.info(f"Dataset saved to {self.dataset_path.absolute()}")
            self.test_equivariance()

        assert self.dataset_path.exists(), "Something went wrong"

    def ensure_dataset_partition(self, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15) -> pathlib.Path:
        partition_folder = f"{self._samples}_train={train_ratio:.2f}_test={test_ratio:.2f}_val={val_ratio:.2f}"
        partition_path = self.dataset_path.parent.joinpath(partition_folder)
        partition_path.mkdir(exist_ok=True)

        train_path, test_path, val_path = partition_path.joinpath("train.npz"), partition_path.joinpath("test.npz"), \
                                          partition_path.joinpath("val.npz")

        if not train_path.exists() or not val_path.exists() or not test_path.exists():
            data = np.load(str(self.dataset_path))
            X, Y = data["X"], data["Y"]

            num_data = X.shape[0]
            num_train = int(train_ratio * num_data)
            num_val = int(val_ratio * num_data)
            num_test = int(test_ratio * num_data)
            assert train_ratio + val_ratio + test_ratio <= 1.0

            X_test, Y_test = X[:num_test, :], Y[:num_test, :]
            X_val, Y_val = X[num_test:num_val + num_test, :], Y[num_test:num_val + num_test, :]
            X_train, Y_train = X[-num_train:, :], Y[-num_train:, :]

            np.savez_compressed(str(partition_path.joinpath("train.npz")), X=X_train, Y=Y_train)
            np.savez_compressed(str(partition_path.joinpath("test.npz")), X=X_test, Y=Y_test)
            np.savez_compressed(str(partition_path.joinpath("val.npz")), X=X_val, Y=Y_val)

            log.info(f"Saving dataset partition {partition_folder} on {partition_path.absolute()}")
        else:
            log.debug(f"Loaded dataset partition {partition_folder}")
        return partition_path

    def to(self, device):
        log.info(f"Moving data to device:{device}")
        self.X.to(device)
        self.Y.to(device)

    def __repr__(self):
        msg = f"CoM[{self.robot.__class__.__name__}]-{self.dataset_type}-Aug:{self.augment}" \
              f"-X:{self.X.shape}-Y:{self.Y.shape}"
        return msg

    def __str__(self):
        return self.__repr__()
