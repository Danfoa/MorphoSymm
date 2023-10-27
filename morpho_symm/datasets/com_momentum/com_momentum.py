import copy
import logging
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from escnn import gspaces
from escnn.group import Group, Representation
from escnn.nn import FieldType
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

import morpho_symm.utils.pybullet_visual_utils
from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper

# from morpho_symm.utils.algebra_utils import dense
from morpho_symm.utils.robot_utils import load_symmetric_system

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

    def __init__(self, robot_cfg, type='train',
                 angular_momentum=True, standarizer: Union[bool, Standarizer] = True, augment=False,
                 train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, samples=100000,
                 dtype=torch.float32, data_path="dataset/com_momentum", device='cpu', debug=False):

        self.dataset_type = type
        self.dtype = dtype
        self.angular_momentum = angular_momentum
        self.normalize = True if isinstance(standarizer, Standarizer) else standarizer

        # Load robot, symmetry group and input-output field types/representations
        self.robot, self.G, self.in_type, self.out_type = self.define_input_output_field_types(robot_cfg)

        self._pb = None  # GUI debug
        self.augment = augment if isinstance(augment, bool) else False

        self._samples = samples
        self.dataset_path = pathlib.Path(data_path).joinpath(f"data_{samples:d}.npz")
        self.ensure_dataset_existance()
        partition_path = self.ensure_dataset_partition(train_ratio, test_ratio, val_ratio)

        # Load data
        assert type.lower() in ["train", "test", "val"], "type must be one of these [train, test, val]"
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

        q, dq = self.robot.get_init_config(random=False)
        self.base_q = q[:7]
        self.base_dq = dq[:6]

        # Normalize Data
        if isinstance(standarizer, Standarizer):
            self.standarizer = standarizer
        elif isinstance(standarizer, bool):
            X_mean, X_std = self.compute_normalization(X)
            Y_mean, Y_std = self.compute_normalization(Y)
            self.standarizer = Standarizer(X_mean, X_std, Y_mean, Y_std, device=device)

        self.X = torch.from_numpy(X).type('torch.FloatTensor').to(device)
        self.Y = torch.from_numpy(Y).type('torch.FloatTensor').to(device)

        if debug:
            self.plot_statistics()

        self.X, self.Y = self.standarizer.transform(self.X, self.Y)

        if isinstance(augment, str) and augment.lower() == "hard":
            for g in self.G.elements[1:]:
                rep_X = self.in_type.fiber_representation(g).to(self.X.device)
                rep_Y = self.out_type.fiber_representation(g).to(self.Y.device)
                gX = (rep_X @ self.X.T).T
                gY = (rep_Y @ self.Y.T).T
                self.X = torch.vstack([self.X, gX])
                self.Y = torch.vstack([self.Y, gY])

        self.loss_fn = F.mse_loss
        log.info(str(self))

        # TODO: Remove
        # self.test_equivariance()

    def compute_normalization(self, data_matrix, rep_data: Optional[Representation] = None):
        """Compute normalization of potential symmetric data distribution.

        This function takes a data matrix of shape X = (samples, dim(x)) where dim(x) represents the dimensions of a
        data sample. The average of the distribution is computed only over the trivial irrep dimensions, since the
        expectation value of a symmetric distribution is restricted to the space invariant to all group actions.

        The covariance is computed for each of irreducible representation sub-space separatedly, considering that
        vectors in an irrep subspace is orthogonal to all other irreps subspaces.

        Args:
            data_matrix: (samples, dim(x))
            rep_data (Representation): symmetry representation of the data. None if data has no known symmetry group.

        Returns:
            mean: Empirical expected value over the data matrix (dim(x),)
            std: Empirical variance over the data matrix (dim(x),)
        """
        if rep_data is None:
            X_mean = np.mean(data_matrix, axis=0)
            X_std = np.std(data_matrix, axis=0)
        else:
            raise NotImplementedError()
            # Q_inv = rep_data.change_of_basis_inv
            # X_iso = Q_inv @ X[..., None]
            # X_mean =

        return X_mean, X_std,  # Y_mean, Y_std

    def test_equivariance(self):
        trials = 10
        for trial in range(trials):
            q, dq = self.robot.get_init_config(random=True)

            x = np.concatenate((q[7:], dq[6:]))
            x = x.astype(np.float64)
            y = self.get_hg(*np.split(x, 2))
            y = y.astype(np.float64)

            x_orbit, y_orbit = [x], [y]
            y_true = [y]

            non_equivariance_detected = False
            rep_X = self.in_type.representation
            rep_Y = self.out_type.representation
            # Get all possible group actions
            for g in self.G.elements:
                gx, gy = rep_X(g) @ x, rep_Y(g) @ y
                x_orbit.append(gx)
                y_orbit.append(gy)
                assert gx.dtype == x.dtype, (gx.dtype, x.dtype)
                assert gy.dtype == y.dtype, (gy.dtype, y.dtype)

                gy_true = self.get_hg(*np.split(gx, 2))
                y_true.append(gy_true)

                assert gy_true.dtype == y.dtype, (gy_true.dtype, y.dtype)

                error = gy_true - gy
                rel_error_norm = np.linalg.norm(error) / np.linalg.norm(gy_true)
                cos_sim = np.dot(gy, gy_true) / (np.linalg.norm(gy_true) * np.linalg.norm(gy))

                # TODO: Change true
                if rel_error_norm > 0.05 and cos_sim <= 0.95:
                    non_equivariance_detected = True

            if non_equivariance_detected:
                try:
                    splited_orbits = [np.split(x, 2) for x in x_orbit]
                    qs = [x[0] for x in splited_orbits]
                    dqs = [x[1] for x in splited_orbits]
                    self.gui_debug(qs=qs, dqs=dqs, hgs=y_true, ghgs=y_orbit)
                except Exception as e:
                    raise e
                    logging.warning(f"Unable to start GUI of pybullet: {str(e)}")
                # raise AttributeError(f"Ground truth hg(q,dq) = Ag(q)dq is not equivariant to provided groups: \n" +
                #                      f"x:{x}\ng*x:{gx}\ny:{y} \ng*y:{gy}\n" +
                #                      f"Aq(g*q,g*dq):{gy_true}\nError:{error}")
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
        return x, y

    def collate_fn(self, batch):
        # Enforce data type in batched array
        # Small hack to do batched augmentation. TODO: Although efficient this should be done somewhere else.
        x_batch, y_batch = default_collate(batch)

        if self.augment:  # Sample uniformly among symmetry actions including identity
            g = self.G.sample()

            g_x_batch = self.in_type.transform_fibers(x_batch, g)
            g_y_batch = self.out_type.transform_fibers(y_batch, g)
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

    def gui_debug(self, qs, dqs, hgs, ghgs):
        """Plot side by side robots with different configutations, CoM momentums and expected CoM after an action g."""
        from utils.algebra_utils import configure_bullet_simulation
        def tint_robot(robot, color=(0.227, 0.356, 0.450), alpha=0.5):
            num_joints = self.robot.bullet_client.getNumJoints(self.robot.robot_id)
            for i in range(num_joints):
                self._pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=i, rgbaColor=color + (alpha,))
            self._pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=-1, rgbaColor=color + (alpha,))

        def draw_momentum_vector(p1, p2, v_color, scale=1.0, show_axes=False, text=None, offset=0.0):
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

        robot = self.robot
        for i in range(len(qs)):
            q, dq, hg, ghg = qs[i], dqs[i], hgs[i], ghgs[i]
            robot = self.robot if i == 0 else copy.copy(self.robot)
            offset = 1.5 * self.robot.hip_height
            morpho_symm.utils.pybullet_visual_utils.configure_bullet_simulation(self._pb, world=None)
            # tint_robot(robot2, alpha=0.9)
            # Place robots in env
            random_q, random_dq = robot.get_init_config(random=True)
            random_q[:7] = self.base_q
            # random_dq[:6] = self.base_dq
            base_q = random_q[:7]
            base_q[1] += offset * i

            # Set positions:
            robot.reset_sim_state(np.concatenate((base_q, q)), np.concatenate((self.base_dq, dq)))

            # Draw linear momentum
            linear_color, angular_color = (0.682, 0.576, 0.039), (0.392, 0.047, 0.047)
            # draw_momentum_vector(base_q[:3], hg[:3], v_color=linear_color, scale=.2 / np.linalg.norm(hg[:3]))
            # draw_momentum_vector(base_q[:3], hg[3:], v_color=angular_color, scale=.2 / np.linalg.norm(hg[3:]))
            # draw_momentum_vector(base_q[:3], ghg[:3], v_color=(0, 0, 0), scale=.2 / np.linalg.norm(ghg[:3]))
            # draw_momentum_vector(base_q[:3], ghg[3:], v_color=(0, 0, 0), scale=.2 / np.linalg.norm(ghg[3:]))

        print("a")

    def ensure_dataset_existance(self):
        # if self.robot.
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
            dq_max = np.minimum(dq_max, np.pi)
            q_min, q_max = self.robot.joint_pos_limits
            q_min = np.maximum(q_min, -np.pi)
            q_max = np.minimum(q_max, np.pi)

            x = np.zeros((self._samples, self.robot.n_js * 2))
            y = np.zeros((self._samples, 6))
            for i in range(self._samples):
                q[7:] = np.random.uniform(q_min, q_max, size=None)
                dq[6:] = np.random.uniform(-dq_max, dq_max, size=None)
                hg = self.robot.pinocchio_robot.centroidalMomentum(q, dq)
                y[i, :] = hg.np
                x[i, :] = np.concatenate((q[7:], dq[6:]))

            # Pinnochio introduces small but considerable equivariance numerical error, even when the robot kinematics
            # and dynamics are completely equivariant. So we make the gt the avg of the augmented predictions.
            ys_pin = [y]
            for g in self.G.elements[1:]:
                gx = np.squeeze(self.in_type.representation(g) @ x.T).T
                # gy = np.squeeze(y @ g_out)
                gy_pin = np.zeros((self._samples, 6))
                # Generate random configuration samples.
                for i, x_sample in enumerate(gx):
                    gy_pin[i, :] = self.get_hg(x_sample[:self.robot.nq - 7], x_sample[self.robot.nq - 7:])
                ys_pin.append((self.out_type.representation(
                    ~g) @ gy_pin.T).T)  # inverse is not needed for the groups we use (C2, V4).

            y_pin_avg = np.mean(ys_pin, axis=0)
            y = y_pin_avg  # To mitigate numerical error

            # From the augmented dataset take the desired samples.
            X, Y = x, y
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

            log.info(f"Saving dataset partition {partition_folder} on {partition_path.parent}")
        else:
            log.debug(f"Loaded dataset partition {partition_folder}")
        return partition_path

    def to(self, device):
        log.info(f"Moving data to device:{device}")
        self.X.to(device)
        self.Y.to(device)

    def define_input_output_field_types(self, robot_cfg: DictConfig) -> (PinBulletWrapper, Group, FieldType, FieldType):
        """Define the input-output symmetry representations for the CoM function g·y = f(g·x) | g ∈ G.

        Define the symmetry representations for the Center of Mass momentum y := (l, k) ∈ R^3 x R^3, where l is the
        linear momentum and k is the angular momentum. The CoM momentum is a function of the system joint state
        x := (q_js, v_js) ∈ Q_js x TqQ_js, where Q_js is the joint space and TqQ_js is the tangent space of the
        joint space.

        Args:
            robot_cfg: The robot configuration (see cfg/robots).

        Returns:
            robot (PinBulletWrapper): The robot wrapper.
            gspace (GSpace): The total symmetry space of the system.
            input_type (FieldType): The input feature field type, describing the system state (q_js, v_js) and its
            symmetry transformations.
            output_type (FieldType): The output feature field type, describing the system CoM (hg) and its symmetry
            transformations.
        """
        robot, G = load_symmetric_system(robot_cfg=robot_cfg, debug=False)
        # For this application we compute the CoM w.r.t base frame, meaning that we ignore the fiber group Ed in which
        # the system evolves in:
        gspace = gspaces.no_base_space(G)
        # Get the relevant representations.
        rep_Q_js = G.representations["Q_js"]
        rep_TqQ_js = G.representations["TqQ_js"]
        rep_R3 = G.representations["Rd"]
        rep_R3_pseudo = G.representations["Rd_pseudo"]

        # Rep for x := [q, dq] ∈ Q_js x TqQ_js     =>    ρ_Q_js(g) ⊕ ρ_TqQ_js(g)  | g ∈ G
        in_type = FieldType(gspace, [rep_Q_js, rep_TqQ_js])

        # Rep for center of mass momentum y := [l, k] ∈ R3 x R3  =>    ρ_R3(g) ⊕ ρ_R3pseudo(g)  | g ∈ G
        out_type = FieldType(gspace, [rep_R3, rep_R3_pseudo])

        # TODO: handle subgroup cases.
        # if robot_cfg.gens_ids is not None and self.dataset_type in ["test", "val"]:
        #     Loaded symmetry group is not the full symmetry group of the robot
        #     Generate the true underlying data representations
        # robot_cfg = copy.copy(robot_cfg)
        # robot_cfg['gens_ids'] = None
        # _, rep_Ed_full, rep_QJ_full = load_robot_and_symmetries(robot_cfg)
        # rep_data_in = get_rep_data_in(rep_QJ_full)
        # rep_data_out = get_rep_data_out(rep_Ed_full)
        # log.info(f"[{self.dataset_type}] Dataset using the system's full symmetry group {type(rep_QJ_full.G)}")
        # return robot, rep_data_in, rep_data_out

        log.info(f"[{self.dataset_type}] Dataset using the symmetry group {type(G)}")
        return robot, G, in_type, out_type

    def __repr__(self):
        msg = f"CoM Dataset: [{self.robot.robot_name}]-{self.dataset_type}-Aug:{self.augment}" \
              f"-X:{self.X.shape}-Y:{self.Y.shape}"
        return msg

    def __str__(self):
        return self.__repr__()

    def plot_statistics(self):
        import seaborn as sns
        from matplotlib import pyplot as plt

        x_orbits = [self.X.detach().cpu().numpy()]
        y_orbits = [self.Y.detach().cpu().numpy()]

        for g_in, g_out in self.t_group_actions[1:]:
            g_x = torch.matmul(self.X.unsqueeze(1), g_in.unsqueeze(0).to(self.X.dtype)).squeeze()
            g_y = torch.matmul(self.Y.unsqueeze(1), g_out.unsqueeze(0).to(self.Y.dtype)).squeeze()
            y_orbits.append(g_y.detach().cpu().numpy())
            x_orbits.append(g_x.detach().cpu().numpy())

        y_true = []
        for x_orbit in x_orbits:
            y_orbit = []
            for x in x_orbit:
                y = self.get_hg(*np.split(x, 2))
                y_orbit.append(y)
            y_true.append(np.vstack(y_orbit))

        i = 0
        errors = []
        for g_y, y_gt in zip(y_orbits, y_true):
            lin_error = np.linalg.norm(y_gt[:, :3] - g_y[:, :3], axis=1)
            ang_error = np.linalg.norm(y_gt[:, 3:] - g_y[:, 3:], axis=1)
            errors.append((lin_error, ang_error))
            print(f"action {i}, lin_error: {np.mean(lin_error):.3e} ang_error: {np.mean(ang_error):.3e}")
            i += 1

        fig, axs = plt.subplots(nrows=len(errors), ncols=2, figsize=(8, 8), dpi=150)
        for orbit_id in range(len(errors)):
            lin_err, ang_err = errors[orbit_id]
            ax_lin, ax_ang = axs[orbit_id, :]
            sns.histplot(x=lin_err, bins=50, stat='probability', ax=ax_lin, kde=True)
            sns.histplot(x=ang_err, bins=50, stat='probability', ax=ax_ang, kde=True)
            ax_lin.set_title(f'Lin error g:{orbit_id}')
            ax_ang.set_title(f'Ang error g:{orbit_id}')
        plt.tight_layout()
        plt.show()
