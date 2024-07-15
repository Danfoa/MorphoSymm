import copy
import logging
import math
import pickle
from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import datasets
import numpy as np
from datasets import Features, IterableDataset
from escnn.group import Representation, groups_dict

log = logging.getLogger(__name__)


@dataclass
class DynamicsRecording:
    """Data structure to store recordings of a Markov Dynamics."""

    description: Optional[str] = None
    info: dict[str, object] = field(default_factory=dict)
    dynamics_parameters: dict = field(default_factory=lambda: {'dt': None})

    # Ordered list of observations composing to the state and action space of the Markov Process.
    state_obs: tuple[str, ...] = field(default_factory=list)
    action_obs: tuple[str, ...] = field(default_factory=list)

    # Map from observation name to the observation representation name. This name should be in `group_representations`.
    obs_representations: dict[str, Representation] = field(default_factory=dict)
    recordings: dict[str, Iterable] = field(default_factory=dict)
    # Map from observation name to the observation moments (mean, var) of the recordings.
    obs_moments: dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        # Check provided state observables are in the recordings
        for obs_name in self.state_obs:
            assert obs_name in self.recordings.keys(), \
                f"State observable {obs_name} not in the provided recordings: {self.recordings.keys()}"
        for obs_name in self.recordings.keys():
            # Scalar observations should have shape (traj, time, 1)
            obs_shape = self.recordings[obs_name].shape
            if len(obs_shape) == 2:
                self.recordings[obs_name] = self.recordings[obs_name][..., None]
            # If group representation of observation is provided, check the representation has the same dimension
            if obs_name in self.obs_representations.keys():
                rep = self.obs_representations[obs_name]
                assert rep.size == self.obs_dims[obs_name], \
                    (f"Invalid dimension of representation {rep.name} {(rep.size, rep.size)} associated to "
                     f"the observation {obs_name} ({self.obs_dims[obs_name]},)")

    @property
    def obs_dims(self):
        """Dictionary providing the map between observation name and observation dimension."""
        return {k: v.shape[-1] for k, v in self.recordings.items()}

    @property
    def state_dim(self):
        """Return the dimension of the state vector."""
        return sum([self.obs_dims[obs] for obs in self.state_obs])

    _obs_idx = None
    def obs_idx_in_state(self, obs_name) -> tuple[int]:
        """returns a tuple of indices of each observation in the state vector."""
        assert obs_name in self.state_obs, f"Observation {obs_name} not in state observations"
        if self._obs_idx is None:
            self._obs_idx = {}
            # Iterate over all state observables in order and store the indices of each observation given its dimension
            curr_dim = 0
            for obs in self.state_obs:
                obs_dim = self.obs_dims[obs]
                self._obs_idx[obs] = tuple(range(curr_dim, curr_dim + obs_dim))
                curr_dim += obs_dim
        return self._obs_idx[obs_name]

    def state_representations(self) -> list[Representation]:
        """Return the ordered list of representations of the state vector."""
        return [self.obs_representations[m] for m in self.state_obs]

    def state_moments(self) -> [np.ndarray, np.ndarray]:
        """Compute the mean and standard deviation of the state observations."""
        mean, var = [], []
        for obs_name in self.state_obs:
            if obs_name not in self.obs_moments.keys():
                self.compute_obs_moments(obs_name)
            obs_mean, obs_var = self.obs_moments[obs_name]
            mean.append(obs_mean)
            var.append(obs_var)
        mean, var = np.concatenate(mean), np.concatenate(var)
        return mean, var
    
    def compute_obs_moments(self, obs_name: str) -> [np.ndarray, np.ndarray]:
        """Compute the mean and standard deviation of observations."""
        assert obs_name in self.recordings.keys(), f"Observation {obs_name} not found in recordings"
        is_symmetric_obs = obs_name in self.obs_representations.keys()
        if is_symmetric_obs:
            rep_obs = self.obs_representations[obs_name]
            obs_original_basis = np.asarray(self.recordings[obs_name])
            G = rep_obs.group
            # Allocate the mean and variance arrays.
            mean, var = np.zeros(rep_obs.size), np.ones(rep_obs.size)
            # Change basis of the observation to expose the irrep G-stable subspaces
            Q_inv = rep_obs.change_of_basis_inv  # Orthogonal transformation to irrep basis (Q^T = Q^-1)
            Q = rep_obs.change_of_basis
            # Get the dimensions of each irrep.

            S = np.zeros((rep_obs.size, rep_obs.size))
            irreps_dimension = []
            cum_dim = 0
            for irrep_id in rep_obs.irreps:
                irrep = G.irrep(*irrep_id)
                # Get dimensions of the irrep in the original basis
                irrep_dims = range(cum_dim, cum_dim + irrep.size)
                irreps_dimension.append(irrep_dims)
                if irrep_id == G.trivial_representation.id:
                    S[irrep_dims, irrep_dims] = 1
                cum_dim += irrep.size

            # Compute the mean of the observation.
            # The mean of a symmetric random variable (rv) lives in the subspaces associated with the trivial irreps.
            has_trivial_irreps = G.trivial_representation.id in rep_obs.irreps
            if has_trivial_irreps:
                avg_projector = rep_obs.change_of_basis @ S @ rep_obs.change_of_basis_inv
                # Compute the mean in a single vectorized operation
                mean_empirical = np.mean(obs_original_basis, axis=(0, 1))
                mean = np.einsum('...ij,...j->...i', avg_projector, mean_empirical)

            # Compute the variance of the observable by computing a single variance per irrep G-stable subspace.
            # To do this, we project the observations to the basis exposing the irreps, compute the variance per
            # G-stable subspace, and map the variance back to the original basis.
            centered_obs_irrep_basis = np.einsum('...ij,...j->...i', Q_inv, obs_original_basis - mean)
            var_irrep_basis = np.ones_like(var)
            for irrep_id, irrep_dims in zip(rep_obs.irreps, irreps_dimension):
                irrep = G.irrep(*irrep_id)
                centered_obs_irrep = centered_obs_irrep_basis[..., irrep_dims]
                assert centered_obs_irrep.shape[-1] == irrep.size, \
                    f"Obs irrep shape {centered_obs_irrep.shape} != {irrep.size}"

                # Since the irreps are unitary/orthogonal transformations, we are constrained compute a unique variance
                # for all dimensions of the irrep G-stable subspace, as scaling the dimensions independently would break
                # the symmetry of the rv. As a centered rv the variance is the expectation of the squared rv.
                var_irrep = np.mean(centered_obs_irrep ** 2)  # Single scalar variance per G-stable subspace
                # Store the irrep mean and variance in the entire representation mean and variance
                var_irrep_basis[irrep_dims] = var_irrep
            # Convert the variance from the irrep basis to the original basis
            Cov = Q @ np.diag(var_irrep_basis) @ Q_inv
            var = np.diagonal(Cov)

            # TODO: Move this check to Unit test as it is computationally demanding to check this at runtime.
            # Ensure the mean is equivalent to computing the mean of the orbit of the recording under the group action
            aug_obs = []
            for g in G.elements:
                g_obs = np.einsum('...ij,...j->...i', rep_obs(g), obs_original_basis)
                aug_obs.append(g_obs)

            aug_obs = np.concatenate(aug_obs, axis=0)   # Append over the trajectory dimension
            mean_emp = np.mean(aug_obs, axis=(0, 1))
            assert np.allclose(mean, mean_emp, rtol=1e-3, atol=1e-3), f"Mean {mean} != {mean_emp}"

            var_emp = np.var(aug_obs, axis=(0, 1))
            assert np.allclose(var, var_emp, rtol=1e-2, atol=1e-2), f"Var {var} != {var_emp}"
        else:
            mean = np.mean(np.asarray(self.recordings[obs_name]), axis=(0, 1))
            var = np.var(np.asarray(self.recordings[obs_name]), axis=(0, 1))
        assert mean.shape == (self.obs_dims[obs_name],), \
            f"Obs {obs_name} dim ({self.obs_dims[obs_name]},) diff from estimated mean dim ({mean.shape},)!= "
        assert var.shape == (self.obs_dims[obs_name],), \
            f"Obs {obs_name} dim ({self.obs_dims[obs_name]},) diff from estimated var dim ({var.shape},)!= "

        self.obs_moments[obs_name] = mean, var

    def get_state_trajs(self, standardize: bool = False):
        """Returns a single array containing the concatenated state observations trajectories.

        Given the state observations `self.state_obs` this method concatenates the trajectories of each observation
        into a single array of shape [traj, time, state_dim]. If standardize is set to True, the state observations
        are standardized to have zero mean and unit variance.

        Returns:
            A single array containing the concatenated state observations trajectories of shape [traj, time, state_dim].
        """
        obs = [self.recordings[obs_name] for obs_name in self.state_obs]
        state_trajs = np.concatenate(obs, axis=-1)
        if standardize:
            state_mean, var = self.state_moments()
            state_std = np.sqrt(var)
            state_trajs = (state_trajs - state_mean) / state_std

        return state_trajs

    def get_state_dim_names(self):
        dim_names = []
        for obs_name in self.state_obs:
            obs_dim = self.obs_dims[obs_name]
            dim_names += [f"{obs_name}:{i}" for i in range(obs_dim)]
        return dim_names
    
    def get_obs_from_vector(self, vector: np.ndarray, obs_names: tuple[str,...] = None) -> dict[str, np.ndarray]:
        """ Extract the observation values from a vector given the ordered observation names."""
        if obs_names is None:
            obs_names = self.state_obs
        else:
            for name in obs_names:
                assert name in self.recordings.keys(), f"Observation {name} not in recordings {self.recordings.keys()}"
        obs_dims = [self.obs_dims[obs] for obs in obs_names]
        assert vector.shape[-1] == sum(obs_dims), \
            f"Vector dim {vector.shape[-1]} differs from expected dimension {sum(obs_dims)}"
        obs_idx = [0] + [sum(obs_dims[:i]) for i in range(1, len(obs_dims))]
        obs_values = {obs_name: vector[...,obs_idx[i]:obs_idx[i] + obs_dims[i]] for i, obs_name in enumerate(obs_names)}
        return obs_values

    def save_to_file(self, file_path: Path):
        # Store representations and groups without serializing
        if len(self.obs_representations) > 0:
            self._obs_rep_irreps = {}
            self._obs_rep_names = {}
            self._obs_rep_Q = {}
            for k, rep in self.obs_representations.items():
                self._obs_rep_irreps[k] = rep.irreps if rep is not None else None
                self._obs_rep_names[k] = rep.name if rep is not None else None
                self._obs_rep_Q[k] = rep.change_of_basis if rep is not None else None
            group = self.obs_representations[self.state_obs[0]].group
            self._group_keys = group._keys
            self._group_name = group.__class__.__name__
            # Remove non-serializable objects
            del self.obs_representations
            self.dynamics_parameters.pop('group', None)

        with file_path.with_suffix(".pkl").open('wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(file_path: Path, only_metadata=False,
                       obs_names: Optional[Iterable[str]] = None,
                       ignore_other_obs: bool = True) -> 'DynamicsRecording':
        """ Load a DynamicsRecording object from a file.

        Args:
            file_path: Path to the file containing the DynamicsRecording object.
            only_metadata: If True, the recordings are not loaded, only the metadata.
            obs_names: List of observation names to load. If None, all observations are loaded.
            ignore_other_obs: If True, observations not in `obs_names` are ignored.

        Returns:
            A DynamicsRecording object containing the loaded data.
        """
        with file_path.with_suffix(".pkl").open('rb') as file:
            data = pickle.load(file)
            if only_metadata:
                del data.recordings
            else:
                data_obs_names = list(data.recordings.keys())
                if obs_names is not None:
                    data.state_obs = tuple(obs_names)
                    if ignore_other_obs:
                        irrelevant_obs = [k for k in data_obs_names if k not in obs_names and obs_names is not None]
                        for k in irrelevant_obs:
                            del data.recordings[k]

            if hasattr(data, '_group_name'):
                group = groups_dict[data._group_name]._generator(**data._group_keys)  # Instanciate symmetry group
                data.dynamics_parameters['group'] = group
                data.obs_representations = {}
                for obs_name in data._obs_rep_irreps.keys():
                    irreps_ids = data._obs_rep_irreps[obs_name]
                    rep_name = data._obs_rep_names[obs_name]
                    rep_Q = data._obs_rep_Q[obs_name]
                    if rep_name is None:
                        data.obs_representations[obs_name] = None
                    elif rep_name in group.representations:
                        data.obs_representations[obs_name] = group.representations[rep_name]
                    else:
                        data.obs_representations[obs_name] = Representation(group, name=rep_name,
                                                                            irreps=irreps_ids, change_of_basis=rep_Q)
                    group.representations[rep_name] = data.obs_representations[obs_name]
        data.__post_init__()
        return data

    @staticmethod
    def load_data_generator(dynamics_recordings: list["DynamicsRecording"],
                            frames_per_step: int = 1,
                            prediction_horizon: Union[int, float] = 1,
                            state_obs: Optional[list[str]] = None,
                            action_obs: Optional[list[str]] = None):
        """Generator that yields observation samples of length `n_frames_per_state` from the Markov Dynamics recordings.

        Args:
            recordings (list[DynamicsRecording]): List of DynamicsRecordings.
            frames_per_step: Number of frames to compose a single observation sample at time `t`. E.g. if `f` is
            provided
            the state samples will be of shape [f, obs_dim].
            prediction_horizon (int, float): Number of future time steps to include in the next time samples.
                E.g: if `n` is an integer the samples will be of shape [n, frames_per_state, obs_dim]
                If `n` is a float, then the samples will be of shape [int(n*traj_length), frames_per_state, obs_dim]
            state_obs: Ordered list of observations names composing the state space.
            action_obs: Ordered list of observations names composing the action space.

        Returns:
            A dictionary containing the observations of shape (time_horizon, frames_per_step, obs_dim)
        """
        for file_data in dynamics_recordings:
            recordings = file_data.recordings
            relevant_obs = set(file_data.state_obs).union(set(file_data.action_obs))
            if state_obs is not None:
                relevant_obs = set(state_obs)
            if action_obs is not None:
                relevant_obs = relevant_obs.union(set(action_obs))

            # Get any observation list of trajectories and count the number of trajectories
            # We assume all observations have the same number of trajectories
            n_trajs = len(next(iter(recordings.values())))

            # Since we assume trajectories can have different lengths, we iterate over each trajectory
            # and generate samples of length `n_frames_per_state` from each trajectory.
            for traj_id in range(n_trajs):
                traj_length = next(iter(recordings.values()))[traj_id].shape[0]
                if isinstance(prediction_horizon, float):
                    steps_in_pred_horizon = floor((prediction_horizon * traj_length) // frames_per_step) - 1
                else:
                    steps_in_pred_horizon = prediction_horizon
                assert steps_in_pred_horizon > 0, f"Invalid prediction horizon {steps_in_pred_horizon}"

                remnant = traj_length % frames_per_step
                frames_in_pred_horizon = steps_in_pred_horizon * frames_per_step
                # Iterate over the frames of the trajectory
                for frame in range(traj_length - frames_per_step):
                    # Collect the next steps until the end of the trajectory. If the prediction horizon is larger than
                    # the remaining steps (prediction outside trajectory length), then we continue to the next traj.
                    # This is better computationally for avoiding copying while batch processing data later.
                    if frame + frames_per_step + frames_in_pred_horizon > (traj_length - remnant):
                        continue
                    sample = {}
                    for obs_name, trajs in recordings.items():
                        if obs_name not in relevant_obs:  # Do not process unrequested observations
                            continue
                        num_steps = (frames_in_pred_horizon // frames_per_step) + 1
                        # Compute the indices for the start and end of each "step" in the time horizon
                        start_indices = np.arange(0, num_steps) * frames_per_step + frame
                        end_indices = start_indices + frames_per_step
                        # Use these indices to slice the relevant portion of the trajectory
                        obs_time_horizon = trajs[traj_id][start_indices[0]:end_indices[-1]]
                        # Reshape the slice to have the desired shape (time, frames_per_step, obs_dim)
                        obs_dim = file_data.obs_dims[obs_name]
                        obs_time_horizon = obs_time_horizon.reshape((num_steps, frames_per_step, obs_dim))

                        # Test no copy is being made (too costly to do at runtime)
                        # assert np.shares_memory(obs_time_horizon, trajs[traj_id])
                        assert len(obs_time_horizon) == steps_in_pred_horizon + 1, \
                            f"{len(obs_time_horizon)} != {steps_in_pred_horizon + 1}"
                        sample[obs_name] = obs_time_horizon
                    # print(frame)
                    yield sample

    @staticmethod
    def map_state_next_state(sample: dict,
                             state_observations: List[str],
                             state_mean: Optional[np.ndarray] = None,
                             state_std: Optional[np.ndarray] = None) -> dict:
        """Map composing multiple frames of observations into a flat vectors `state` and `next_state` samples.

        This method constructs the state `s_t` and history of nex steps `s_t+1` of the Markov Process.
        The state is defined as a set of observations within a window of fps=`frames_per_state`.
        E.g.: Consider the state is defined by the observations [m=momentum, p=position] at `fps` consecutive frames.
            Then the state at time `t` is defined as `s_t = [m_f, m_f+1,..., m_f+fps, p_f, p_f+1, ..., p_f+fps]`.
            Where we use f to denote frame in time to make the distinction from the time index `t` of the Markov
            Process.
            Then, the next state is defined as `s_t+1 = [m_f+fps,..., m_fps+fps, p_f+fps, ..., p_f+fps+fps]`.

        Args:
            sample (dict): Dictionary containing the observations of the system of shape [state_time, f].
            state_observations: Ordered list of observations names composing the state space.

        Returns:
            A dictionary containing the MDP state `s_t` and the next_state/s `[s_t+1, s_t+2, ..., s_t+pred_horizon]`.
        """
        batch_size = len(sample[f"{state_observations[0]}"])
        time_horizon = len(sample[f"{state_observations[0]}"][0])
        # Flatten observations a_t = [a_f, a_f+1, af+2, ..., a_f+F] s.t. a_t in R^{F * dim(a)}, a_f in R^{dim(a)}
        state_obs = [sample[m] for m in state_observations]
        # Define the state at time t and the states at time [t+1, t+pred_horizon]
        state_traj = np.concatenate(state_obs, axis=-1).reshape(batch_size, time_horizon, -1)
        if state_mean is not None and state_std is not None:
            state_traj = (state_traj - state_mean) / state_std
        return dict(state=state_traj[:, 0], next_state=state_traj[:, 1:])

    @staticmethod
    def map_state_action_state(sample, state_observations: List[str], action_observations: List[str]) -> dict:
        """Map composing multiple observations to single state, action, next_state samples."""
        flat_sample = DynamicsRecording.map_state_next_state(sample, state_observations)
        # Reuse the same function to for flattening action and next_action
        action_sample = DynamicsRecording.map_state_next_state(sample, action_observations)
        action_sample["action"] = action_sample.pop("state")
        action_sample["next_action"] = action_sample.pop("next_state")
        flat_sample.update(action_sample)
        return flat_sample


def estimate_dataset_size(recordings: list[DynamicsRecording], prediction_horizon: Union[int, float] = 1,
                          frames_per_step: int = 1):
    num_trajs = 0
    num_samples = 0
    steps_pred_horizon = []
    for r in recordings:
        r_num_trajs = r.info['num_traj']
        r_traj_length = r.info['trajectory_length']
        if isinstance(prediction_horizon, float):
            steps_in_pred_horizon = floor((prediction_horizon * r_traj_length) // frames_per_step)
        else:
            steps_in_pred_horizon = prediction_horizon
        steps_pred_horizon.append(steps_in_pred_horizon)
        frames_in_pred_horizon = steps_in_pred_horizon * frames_per_step
        samples = r_traj_length - frames_in_pred_horizon - (r_traj_length % frames_per_step) + 1
        num_samples += r_num_trajs * samples
        num_trajs += r_num_trajs
    steps_pred_horizon = np.mean(steps_pred_horizon)
    log.debug(f"Steps in prediction horizon {int(steps_pred_horizon)}")
    return num_trajs, num_samples


def reduce_dataset_size(recordings: Iterable[DynamicsRecording], train_ratio: float = 1.0):
    assert 0.0 < train_ratio <= 1.0, f"Invalid train ratio {train_ratio}"
    if train_ratio == 1.0:
        return recordings
    log.info(f"Reducing dataset size to {train_ratio * 100}%")
    # Ensure all training seeds use the same training data partitions
    from morpho_symm.utils.mysc import TemporaryNumpySeed
    with TemporaryNumpySeed(10):
        for r in recordings:
            # Decide to keep a ratio of the original trajectories
            num_trajs = r.info['num_traj']
            if num_trajs < 10:  # Do not discard entire trajectories, but rather parts of the trajectories
                time_horizon = r.recordings[r.state_obs[0]].shape[1]  # Take the time horizon from the first observation

                # Split the trajectory into "virtual" subtrajectories, and discard some of them
                idx = np.arange(time_horizon)
                n_partitions = math.ceil((1 / (1 - train_ratio)))
                n_partitions = n_partitions * 10 if n_partitions < 10 else n_partitions
                # Ensure partitions of the same time duration
                partitions_idx = np.split(idx[:-(time_horizon % n_partitions)], indices_or_sections=n_partitions)
                partition_length = partitions_idx[0].shape[0]
                n_partitions_to_keep = math.ceil(n_partitions * train_ratio)
                partitions_to_keep = np.random.choice(range(n_partitions),
                                                      size=n_partitions_to_keep,
                                                      replace=False)
                print(partitions_to_keep)
                partitions_to_keep = [partitions_idx[i] for i in partitions_to_keep]

                ratio_of_samples_removed = (time_horizon - (len(partitions_to_keep) * partition_length)) / time_horizon
                assert ratio_of_samples_removed - (1 - train_ratio) < 0.05, \
                    (f"Requested to remove {(1 - train_ratio) * 100}% of the samples, "
                     f"but removed {ratio_of_samples_removed * 100}%")

                new_recordings = {}
                for obs_name, obs in r.recordings.items():
                    new_obs_trajs = []
                    for part_time_idx in partitions_to_keep:
                        new_obs_trajs.append(obs[:, part_time_idx])
                    new_recordings[obs_name] = np.concatenate(new_obs_trajs, axis=0)
                r.recordings = new_recordings
                r.info['num_traj'] = len(partitions_to_keep)
                r.info['trajectory_length'] = partition_length
            else:  # Discard entire trajectories
                # Sample int(num_trajs * train_ratio) trajectories from the original recordings
                num_trajs_to_keep = math.ceil(num_trajs * train_ratio)
                idx = range(num_trajs)
                idx_to_keep = np.random.choice(idx, size=num_trajs_to_keep, replace=False)
                # Keep only the selected trajectories
                new_recordings = {k: v[idx_to_keep] for k, v in r.recordings.items()}
                r.recordings = new_recordings


def split_train_val_test(
        dyn_recording: DynamicsRecording,
        partition_sizes=(0.70, 0.15, 0.15),
        split_dimension: str = 'auto' # 'time' or 'trajectory' 
        ) -> [DynamicsRecording, DynamicsRecording, DynamicsRecording]:
    """Split the recordings into training, validation and test sets.

    Args:
        dyn_recording: (DynamicsRecording): The recordings to split.
        partition_sizes: (tuple): The sizes of the training, validation and test sets.
        split_dimension: (str): The dimension to split the data. Can be 'time' or 'trajectory' or 'auto'. In the case
         of auto, the function will decide the best way to split the data based on the shape of the data.
    Returns:
        (DynamicsRecording, DynamicsRecording, DynamicsRecording): The training, validation and test sets.
    """

    assert np.isclose(np.sum(partition_sizes), 1.0), f"Invalid partition sizes {partition_sizes}"
    partitions_names = ["train", "val", "test"]
    
    
    log.info(f"Partitioning {dyn_recording.description} into train/val/test of sizes {partition_sizes}[%]")
    # Ensure all training seeds use the same training data partitions
    from morpho_symm.utils.mysc import TemporaryNumpySeed
    with TemporaryNumpySeed(10):  # Ensure deterministic behavior
        # Decide to keep a ratio of the original trajectories
        state_traj = dyn_recording.get_state_trajs()
        assert state_traj.ndim == 3, f"Expectec (traj, time, state_dim) but got {state_traj.shape}"
        num_trajs, time_horizon, state_dim = state_traj.shape
        
        if split_dimension == 'auto':
            split_time = time_horizon > num_trajs
        else:
            split_time = split_dimension == 'time'
            
        if split_time:  # Do not discard entire trajectories, but rather parts of the trajectories
            num_samples = time_horizon
            min_idx = 0
            partitions_sample_idx = {partition: None for partition in partitions_names}
            for partition_name, ratio in zip(partitions_names, partition_sizes):
                max_idx = min_idx + int(num_samples * ratio)
                partitions_sample_idx[partition_name] = list(range(min_idx, max_idx))
                min_idx = min_idx + int(num_samples * ratio)

            # TODO: Avoid deep copying the data itself.
            partitions_recordings = {partition: copy.deepcopy(dyn_recording) for partition in partitions_names}
            for partition_name, sample_idx in partitions_sample_idx.items():
                part_num_samples = len(sample_idx)
                partitions_recordings[partition_name].info['trajectory_length'] = part_num_samples
                partitions_recordings[partition_name].recordings = dict()
                for obs_name in dyn_recording.recordings.keys():
                    if len(dyn_recording.recordings[obs_name].shape) == 3:
                        data = dyn_recording.recordings[obs_name][:, sample_idx]
                    elif len(dyn_recording.recordings[obs_name].shape) == 2:
                        data = dyn_recording.recordings[obs_name][sample_idx]
                    else:
                        raise RuntimeError(f"Invalid shape {dyn_recording.recordings[obs_name].shape} of {obs_name}")
                    partitions_recordings[partition_name].recordings[obs_name] = data

            return partitions_recordings['train'], partitions_recordings['val'], partitions_recordings['test']
        else:  # Discard entire trajectories
            raise NotImplementedError()


def get_dynamics_dataset(train_shards: list[Path],
                         test_shards: Optional[list[Path]] = None,
                         val_shards: Optional[list[Path]] = None,
                         frames_per_step: int = 1,
                         train_ratio: float = 1.0,
                         train_pred_horizon: Union[int, float] = 1,
                         eval_pred_horizon: Union[int, float] = 10,
                         test_pred_horizon: Union[int, float] = 10,
                         state_obs: Optional[tuple[str]] = None,
                         action_obs: Optional[tuple[str]] = None,
                         hard_test_augment: bool = False,
                         hard_val_augment: bool = False,
                         ) -> tuple[list[IterableDataset], DynamicsRecording]:
    """Load Markov Dynamics recordings from a list of files and return a train, test and validation dataset."""
    # TODO: ensure all shards come from the same dynamical system
    metadata: DynamicsRecording = DynamicsRecording.load_from_file(train_shards[0])
    test_shards = [] if test_shards is None else test_shards
    val_shards = [] if val_shards is None else val_shards

    if len(test_shards) > 0:
        from morpho_symm.utils.mysc import compare_dictionaries
        test_metadata = DynamicsRecording.load_from_file(test_shards[0], only_metadata=True)
        dyn_params_diff = compare_dictionaries(metadata.dynamics_parameters, test_metadata.dynamics_parameters)
        assert len(dyn_params_diff) == 0, "Different dynamical systems loaded in train/test sets"

    # Define the relevant observations of the recording to load.
    state_obs = state_obs if state_obs is not None else metadata.state_obs
    action_obs = action_obs if action_obs is not None else metadata.action_obs
    relevant_obs = set(state_obs).union(set(action_obs))
    features = {}
    assert len(relevant_obs) > 0, "Provide the names of the observations to be included in state (and action)"
    for obs_name in relevant_obs:
        assert obs_name in metadata.recordings.keys(), f"Observation {obs_name} not found in recordings"
    for obs_name in relevant_obs:
        features[obs_name] = datasets.Array2D(shape=(frames_per_step, metadata.obs_dims[obs_name]), dtype='float32')

    part_datasets = []
    for partition, partition_shards in zip(["train", "test", "val"], [train_shards, test_shards, val_shards]):
        if len(partition_shards) == 0:
            part_datasets.append(None)
            continue

        recordings = [DynamicsRecording.load_from_file(f, obs_names=relevant_obs) for f in partition_shards]
        a = recordings[0]
        mean, var = a.state_moments()
        if partition == "train":
            pred_horizon = train_pred_horizon
            if train_ratio < 1.0:
                reduce_dataset_size(recordings, train_ratio)
        elif partition == "val":
            pred_horizon = eval_pred_horizon
        else:
            pred_horizon = test_pred_horizon

        num_trajs, num_samples = estimate_dataset_size(recordings, pred_horizon, frames_per_step)
        dataset = IterableDataset.from_generator(DynamicsRecording.load_data_generator,
                                                 features=Features(features),
                                                 gen_kwargs=dict(dynamics_recordings=recordings,
                                                                 frames_per_step=frames_per_step,
                                                                 prediction_horizon=pred_horizon,
                                                                 state_obs=tuple(state_obs),
                                                                 action_obs=tuple(action_obs))
                                                 )

        log.debug(f"[Dataset {partition} - Trajs:{num_trajs} - Samples: {num_samples} - "
                  f"Frames per sample : {frames_per_step}]-----------------------------")

        dataset.info.dataset_size = num_samples
        dataset.info.dataset_name = f"[{partition}] Linear dynamics"
        dataset.info.description = metadata.description
        # dataset = partition
        part_datasets.append(dataset)

    metadata.state_obs = state_obs
    metadata.action_obs = action_obs
    return part_datasets, metadata


def get_train_test_val_file_paths(data_path: Path):
    """Search in folder for files ending in train/test/val.pkl and return a list of paths to each file."""
    train_data, test_data, val_data = [], [], []
    for file_path in data_path.iterdir():
        if file_path.name.endswith("train.pkl"):
            train_data.append(file_path)
        elif file_path.name.endswith("test.pkl"):
            test_data.append(file_path)
        elif file_path.name.endswith("val.pkl"):
            val_data.append(file_path)
    return train_data, test_data, val_data


if __name__ == "__main__":
    path_to_data = Path(__file__).parent
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    # Find all dynamic systems recordings
    path_to_data /= Path('mini_cheetah') / 'recordings' / 'grass'
    # path_to_data = Path('/home/danfoa/Projects/koopman_robotics/data/linear_system/group=C3-dim=6/n_constraints=1/')
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*test.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()
    # Obtain the training, testing and validation file paths containing distinct trajectories of motion.
    train_data, test_data, val_data = get_train_test_val_file_paths(mock_path)
    # Obtain hugging face Iterable datasets instances
    pred_horizon = .1
    frames_per_state = 10
    (train_dataset, test_dataset, val_dataset), metadata = get_dynamics_dataset(train_shards=train_data,
                                                                                test_shards=test_data,
                                                                                val_shards=val_data,
                                                                                frames_per_step=frames_per_state,
                                                                                train_pred_horizon=pred_horizon,
                                                                                eval_pred_horizon=0.5)

    # test_flat = map_state_next_state(sample, metadata.state_observations)
    # Test the map
    torch_dataset = test_dataset.with_format("torch").shuffle()
    torch_dataset = torch_dataset.map(map_state_next_state, batched=True, fn_kwargs={'state_observations': ['state']})

    # Test errors while looping.
    for i, s in enumerate(torch_dataset):
        pass
        # log.debug(s)
        # if i > 1000:
        #     break

    log.debug(i)
