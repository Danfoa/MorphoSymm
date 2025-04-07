import logging
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from escnn import gspaces
from escnn.group import Group, directsum
from escnn.nn import FieldType
from pinocchio import pinocchio_pywrap as pin
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from morpho_symm.data.DynamicsRecording import DynamicsRecording
from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper

# from morpho_symm.utils.algebra_utils import dense
from morpho_symm.utils.robot_utils import load_symmetric_system

log = logging.getLogger(__name__)

np.set_printoptions(precision=4)


class ProprioceptiveDataset(Dataset):
    def __init__(
        self,
        robot_name,
        type="train",
        angular_momentum=True,
        augment=False,
        train_ratio=0.7,
        test_ratio=0.15,
        val_ratio=0.15,
        n_total_samples=10000,
        kinetic_energy=False,
        dtype=torch.float32,
        data_path="dataset/proprioceptive",
        device="cpu",
        debug=False,
    ):
        self.dataset_type = type
        self.dtype = dtype
        self.angular_momentum = angular_momentum
        self.kin_energy = kinetic_energy

        # Load robot, symmetry group and input-output field types/representations
        self.robot, self.G, self.in_type, self.out_type = self.define_input_output_field_types(robot_name)

        self._pb = None  # GUI debug
        self.augment = augment if isinstance(augment, bool) else False

        self._samples = n_total_samples
        self.dataset_path = pathlib.Path(data_path).joinpath(f"{self.robot.name}/n_samples_{n_total_samples:d}.plk")
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
        if self.kin_energy:
            X, Y = data["X"], data["KinE"]
        else:
            X, Y = data["X"], data["Y"]

        q, dq = self.robot.get_init_config(random=False, fix_base=True)
        self.base_q = q[:7]
        self.base_dq = dq[:6]

        self.X = torch.from_numpy(X).type("torch.FloatTensor").to(device)
        self.Y = torch.from_numpy(Y).type("torch.FloatTensor").to(device)

        if debug:
            self.plot_statistics()

        # if isinstance(augment, str) and augment.lower() == "hard":
        #     for g in self.G.elements[1:]:
        #         rep_X = self.in_type.fiber_representation(g).to(self.X.device)
        #         rep_Y = self.out_type.fiber_representation(g).to(self.Y.device)
        #         gX = (rep_X @ self.X.T).T
        #         gY = (rep_Y @ self.Y.T).T
        #         self.X = torch.vstack([self.X, gX])
        #         self.Y = torch.vstack([self.Y, gY])

        self.loss_fn = F.mse_loss
        log.info(str(self))

        # TODO: Remove
        # self.test_equivariance()

    def test_equivariance(self):
        trials = 50
        for trial in range(trials):
            q, dq = self.robot.get_init_config(random=True, fix_base=True)
            dq = np.random.uniform(-np.pi, np.pi, size=dq.shape)
            kin_E = pin.computeKineticEnergy(self.robot.pinocchio_robot.model, self.robot.pinocchio_robot.data, q, dq)

            x = np.concatenate((q[7:], dq[6:]))
            x = x.astype(np.float64)
            y = self.get_hg(q[7:], dq[6:])
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

                qj, dqj = gx[: self.robot.nq - 7], gx[self.robot.nq - 7 :]
                gq, gdq = q, dq
                gq[7:] = qj
                gdq[6:] = dqj
                g_kinE = pin.computeKineticEnergy(
                    self.robot.pinocchio_robot.model, self.robot.pinocchio_robot.data, gq, gdq
                )

                # Ensure kinetic energy is preserved, up to 5% relative error
                kin_E_rel_error = np.abs((g_kinE - kin_E) / kin_E)
                if kin_E_rel_error > 0.05:
                    raise AttributeError(f"Kinetic energy G-invariance violated: relative error {g_kinE} != {kin_E}")

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
        # Original scales and means.
        with torch.no_grad():
            metrics = {}

            assert y.shape == y_pred.shape, (y.shape, y_pred.shape)

            if self.kin_energy:
                metrics["kin_energy_err"] = torch.mean(torch.abs(y - y_pred))
                return metrics

            lin, lin_pred = y[:, :3], y_pred[:, :3]
            metrics["lin_cos_sim"] = torch.mean(F.cosine_similarity(lin, lin_pred, dim=-1))
            metrics["lin_err"] = torch.mean(torch.linalg.norm(lin - lin_pred, dim=-1))

            ang, ang_pred = y[:, 3:], y_pred[:, 3:]
            metrics["ang_cos_sim"] = torch.mean(F.cosine_similarity(ang, ang_pred, dim=-1))
            metrics["ang_err"] = torch.mean(torch.linalg.norm(ang - ang_pred, dim=-1))
        return metrics

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, i):
        if self.kin_energy:
            x, y = (
                self.X[i, :],
                self.Y[i],
            )
        else:
            if self.angular_momentum:
                x, y = (
                    self.X[i, :],
                    self.Y[i, :],
                )
            else:
                x, y = (
                    self.X[i, :],
                    self.Y[i, :3],
                )
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

    def get_hg(self, qj, dqj):
        hg = self.robot.pinocchio_robot.centroidalMomentum(
            q=np.concatenate((self.base_q, qj)), v=np.concatenate((self.base_dq, dqj))
        )
        hg = np.array(hg)
        if self.angular_momentum:
            return hg
        return hg[:3]

    def ensure_dataset_existance(self):
        # if self.robot.
        q, dq = self.robot.get_init_config(random=False, fix_base=True)
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
            print(f"vel max {dq_max}")
            print(f"qmin {q_min} \n qmax {q_max}")
            rep_Q = self.G.representations["Q_js"]
            rep_TqQ = self.G.representations["TqQ_js"]
            rep_R3 = self.G.representations["R3"]
            rep_R3_pseudo = self.G.representations["R3_pseudo"]
            # Rep for center of mass momentum y := [l, k] ∈ R3 x R3  =>    ρ_R3(g) ⊕ ρ_R3pseudo(g)  | g ∈ G
            com_rep = directsum([rep_R3, rep_R3_pseudo])

            x = np.zeros((self._samples, self.robot.nq + self.robot.nv - 7 - 6))
            y = np.zeros((self._samples, 6))
            kinE = np.zeros((self._samples, 1))
            for i in range(self._samples):
                q[7:] = np.random.uniform(q_min, q_max, size=None)
                dq[6:] = np.random.uniform(-dq_max, dq_max, size=None)
                hg = self.robot.pinocchio_robot.centroidalMomentum(q, dq)
                kin_energy = pin.computeKineticEnergy(
                    self.robot.pinocchio_robot.model, self.robot.pinocchio_robot.data, q, dq
                )
                kinE[i, :] = kin_energy
                y[i, :] = hg.np
                x[i, :] = np.concatenate((q[7:], dq[6:]))

            # Pinnochio introduces small but considerable equivariance numerical error, even when the robot kinematics
            # and dynamics are completely equivariant. So we make the gt the avg of the augmented predictions.
            ys_pin = [y]
            G_kinE = [kinE]
            for g in self.G.elements[1:]:
                gx = np.squeeze(self.in_type.representation(g) @ x.T).T
                # gy = np.squeeze(y @ g_out)
                gy_pin = np.zeros((self._samples, 6))
                g_kinE = np.zeros((self._samples, 1))
                # Generate random configuration samples.
                for i, x_sample in enumerate(gx):
                    qj, vj = x_sample[: self.robot.nq - 7], x_sample[self.robot.nq - 7 :]
                    q[7:] = qj
                    dq[6:] = vj
                    hg = self.robot.pinocchio_robot.centroidalMomentum(q, dq)
                    kin_energy = pin.computeKineticEnergy(
                        self.robot.pinocchio_robot.model, self.robot.pinocchio_robot.data, q, dq
                    )
                    gy_pin[i, :] = hg.np
                    g_kinE[i, :] = kin_energy

                # Inverse is not needed for the groups we use (C2, V4).
                ys_pin.append((com_rep(~g) @ gy_pin.T).T)
                G_kinE.append(g_kinE)

            y_pin_avg = np.mean(ys_pin, axis=0)
            y = y_pin_avg  # To mitigate numerical error
            mean_kinE = np.mean(G_kinE, axis=0)
            max_err = np.max(np.abs(kinE - mean_kinE))
            kinE = mean_kinE
            # From the augmented dataset take the desired samples.
            X, Y, KinE = x, y, kinE

            print(X.shape)
            print(np.expand_dims(X, 1).shape)
            data_record = DynamicsRecording(
                description=f"Proprioceptive data from {self.robot.name} robot",
                recordings={
                    "X": np.expand_dims(X, 1),
                    "Y": np.expand_dims(Y, 1),
                    "kinetic_energy": np.expand_dims(KinE, 1),
                    "qjs": np.expand_dims(X[..., : self.robot.nq - 7], 1),
                    "vjs": np.expand_dims(X[..., self.robot.nq - 7 :], 1),
                    "com_lin_momentum": np.expand_dims(Y[..., :3], 1),
                    "com_ang_momentum": np.expand_dims(Y[..., 3:], 1),
                },
                state_obs=("X"),
                obs_representations={
                    "X": rep_Q + rep_TqQ,
                    "Y": com_rep,
                    "kinetic_energy": self.G.trivial_representation,
                    "qjs": rep_Q,
                    "vjs": rep_TqQ,
                    "com_lin_momentum": rep_R3,
                    "com_ang_momentum": rep_R3_pseudo,
                },
            )
            data_record.save_to_file(self.dataset_path)
            print(f"Saved data record to {self.dataset_path}")
            # np.savez_compressed(str(self.dataset_path), X=X, Y=Y, KinE=KinE)
            log.info(f"Dataset saved to {self.dataset_path.absolute()}")
            self.test_equivariance()
        assert self.dataset_path.exists(), "Something went wrong"

    def ensure_dataset_partition(self, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15) -> pathlib.Path:
        assert train_ratio + test_ratio + val_ratio <= 1.0
        partition_folder = f"{self._samples}_train={train_ratio:.2f}_test={test_ratio:.2f}_val={val_ratio:.2f}"
        partition_path = self.dataset_path.parent.joinpath(partition_folder)
        partition_path.mkdir(exist_ok=True)

        train_path, test_path, val_path = (
            partition_path.joinpath("train.npz"),
            partition_path.joinpath("test.npz"),
            partition_path.joinpath("val.npz"),
        )

        if not train_path.exists() or not val_path.exists() or not test_path.exists():
            data = np.load(str(self.dataset_path))
            X, Y, KinE = data["X"], data["Y"], data["KinE"]

            num_data = X.shape[0]
            num_train = int(train_ratio * num_data)
            num_val = int(val_ratio * num_data)
            num_test = int(test_ratio * num_data)
            assert train_ratio + val_ratio + test_ratio <= 1.0

            X_test, Y_test = X[:num_test, :], Y[:num_test, :]
            X_val, Y_val = X[num_test : num_val + num_test, :], Y[num_test : num_val + num_test, :]
            X_train, Y_train = X[-num_train:, :], Y[-num_train:, :]

            KinE_test = KinE[:num_test, :]
            KinE_val = KinE[num_test : num_val + num_test, :]
            KinE_train = KinE[-num_train:, :]

            # Use full group orbit on test set to account for equivariance/invariance error.
            G_X, G_Y, G_KinE = [X_test], [Y_test], [KinE_test]
            for g in self.G.elements[1:]:
                rep_X = self.in_type.representation
                rep_Y = self.out_type.representation
                gX = (rep_X(g) @ X_test.T).T
                gY = (rep_Y(g) @ Y_test.T).T
                G_X.append(gX)
                G_Y.append(gY)
                G_KinE.append(KinE_test)

            X_test = np.vstack(G_X)
            Y_test = np.vstack(G_Y)
            KinE_test = np.vstack(G_KinE)

            np.savez_compressed(str(partition_path.joinpath("train.npz")), X=X_train, Y=Y_train, KinE=KinE_train)
            np.savez_compressed(str(partition_path.joinpath("test.npz")), X=X_test, Y=Y_test, KinE=KinE_test)
            np.savez_compressed(str(partition_path.joinpath("val.npz")), X=X_val, Y=Y_val, KinE=KinE_val)

            log.info(f"Saving dataset partition {partition_folder} on {partition_path.parent}")
        else:
            log.debug(f"Loaded dataset partition {partition_folder}")
        return partition_path

    def to(self, device):
        log.info(f"Moving data to device:{device}")
        self.X.to(device)
        self.Y.to(device)

    def define_input_output_field_types(self, robot_name) -> (PinBulletWrapper, Group, FieldType, FieldType):
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
        robot, G = load_symmetric_system(robot_name=robot_name, debug=False)
        # For this application we compute the CoM w.r.t base frame, meaning that we ignore the fiber group Ed in which
        # the system evolves in:
        gspace = gspaces.no_base_space(G)
        # Get the relevant representations.
        rep_Q_js = G.representations["Q_js"]
        rep_TqQ_js = G.representations["TqQ_js"]
        rep_R3 = G.representations["R3"]
        rep_R3_pseudo = G.representations["R3_pseudo"]

        # Rep for x := [q, dq] ∈ Q_js x TqQ_js     =>    ρ_Q_js(g) ⊕ ρ_TqQ_js(g)  | g ∈ G
        in_type = FieldType(gspace, [rep_Q_js, rep_TqQ_js])

        if self.kin_energy:
            out_type = FieldType(gspace, [G.trivial_representation])
        else:  # CoM momentum
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
        msg = (
            f"CoM Dataset: [{self.robot.robot_name}]-{self.dataset_type}-Aug:{self.augment}"
            f"-X:{self.X.shape}-Y:{self.Y.shape}"
        )
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
            sns.histplot(x=lin_err, bins=50, stat="probability", ax=ax_lin, kde=True)
            sns.histplot(x=ang_err, bins=50, stat="probability", ax=ax_ang, kde=True)
            ax_lin.set_title(f"Lin error g:{orbit_id}")
            ax_ang.set_title(f"Ang error g:{orbit_id}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    robot_name = "anymal_c"

    dataset = ProprioceptiveDataset(
        robot_name=robot_name,
        type="train",
        n_total_samples=100000,
        angular_momentum=True,
        augment=False,
        kinetic_energy=True,
    )
