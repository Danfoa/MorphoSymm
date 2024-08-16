import numpy as np
import pathlib
from morpho_symm.data.DynamicsRecording import DynamicsRecording


# HOW DID I SAVED THING??
# temp_state =  np.zeros((1, 25))
# temp_state[0, 0] = state_current["position"][0]   WORLD
# temp_state[0, 1] = state_current["position"][1]   WORLD
# temp_state[0, 2] = state_current["position"][2]   WORLD
# temp_state[0, 3:6] = state_current["linear_velocity"][0:3] WORLD
# temp_state[0, 6] = state_current["orientation"][0] WORLD
# temp_state[0, 7] = state_current["orientation"][1] WORLD
# temp_state[0, 8] = state_current["orientation"][2] WORLD
# temp_state[0, 9] = state_current["angular_velocity"][0] BASE
# temp_state[0, 10] = state_current["angular_velocity"][1] BASE
# temp_state[0, 11] = state_current["angular_velocity"][2] BASE
# temp_state[0, 12:15] = state_current["foot_FL"] WORLD
# temp_state[0, 15:18] = state_current["foot_FR"] WORLD
# temp_state[0, 18:21] = state_current["foot_RL"] WORLD
# temp_state[0, 21:24] = state_current["foot_RR"] WORLD
# temp_state[0, 24:24+12] = d.qpos[6:18]
# temp_state[0, 36:36+12] = d.qvel[6:18]
# temp_state[0, 48] = pgg.step_freq
# data_state.append(copy.deepcopy(temp_state))


# temp_ref = np.zeros((1, 22))
# temp_ref[0, 0] = reference_state["ref_position"][2]
# temp_ref[0, 1:4] = reference_state["ref_linear_velocity"][0:3] WORLD - from a base one that is always costant here
# (0.4, 0, 0), but then is rotated in world
# temp_ref[0, 4] = reference_state["ref_orientation"][0]
# temp_ref[0, 5] = reference_state["ref_orientation"][1]
# temp_ref[0, 6] = reference_state["ref_angular_velocity"][0]
# temp_ref[0, 7] = reference_state["ref_angular_velocity"][1]
# temp_ref[0, 8] = reference_state["ref_angular_velocity"][2]
# temp_ref[0, 9:12] = reference_state["ref_FL_foot"]
# temp_ref[0, 12:15] = reference_state["ref_FR_foot"]
# temp_ref[0, 15:18] = reference_state["ref_RL_foot"]
# temp_ref[0, 18:21] = reference_state["ref_RR_foot"]
# temp_ref[0, 21] = 1.3 #nominal step frequency
# data_reference.append(copy.deepcopy(temp_ref))


# temp_input = np.zeros((1, 24))
# temp_input[0, 0:3] = nmpc_GRFs[0:3] WORLD
# temp_input[0, 3:6] = nmpc_GRFs[3:6] WORLD
# temp_input[0, 6:9] = nmpc_GRFs[6:9] WORLD
# temp_input[0, 9:12] = nmpc_GRFs[9:12] WORLD
# temp_input[0, 12:15] = tau_FL
# temp_input[0, 15:18] = tau_FR
# temp_input[0, 18:21] = tau_RL
# temp_input[0, 21:24] = tau_RR
# data_input.append(copy.deepcopy(temp_input))


# temp_disturbance = np.zeros((1, 7))
# temp_disturbance[0, 0:6] = disturbance_wrench WORLD
# temp_disturbance[0, 6] = start_disturbance_boolean WORLD
# data_external_disturbance.append(copy.deepcopy(temp_disturbance))

def convert_mini_cheetah_sb_controller_recording(data_path: pathlib.Path):
    """Conversion script for the recordings of observations from the Mini-Cheetah Robot.

    This function takes recordings stored into multiple numpy arrays for state, reference, input, and external
    disturbance. Each numpy array has its specific shape and dimensions:

    1. State (numpy array of shape (time, 85)):
        - base position (3,) at indices [0:3], representation: rep_Rd
        - base linear velocity (3,) at indices [3:6], representation: rep_Rd
        - base orientation (3,) at indices [6:9], representation: rep_euler_xyz
        - base angular velocity (3,) at indices [9:12], representation: rep_euler_xyz
        - feet positions (12,) at indices [12:24], (FL, FR, RL, RR) representation: rep_Rd_on_limbs
        - joint positions (12,) at indices [24:36], representation: rep_TqQ_js
        - joint velocities (12,) at indices [36:48], representation: rep_TqQ_js
        - step frequency (1,) at index [48], representation: rep_trivial

    2. Reference (numpy array of shape (time, 22)):
        - reference base height (1,) at index [0], representation: rep_z
        - reference base linear velocity (3,) at indices [1:4], representation: rep_Rd
        - reference base orientation (2,) at indices [4:6], representation: rep_euler_xyz
        - reference base angular velocity (3,) at indices [6:9], representation: rep_euler_xyz
        - reference feet positions (12,) at indices [9:21], representation: rep_Rd_on_limbs
        - nominal step frequency (1,) at index [21], representation: UNSURE

    3. Input (numpy array of shape (time, 24)):
        - ground reaction forces (12,) at indices [0:12], (FL, FR, RL, RR) representation: rep_Rd_on_limbs
        - joint torques (12,) at indices [12:24], representation: rep_TqQ_js

    4. External Disturbance (numpy array of shape (time, 7)):
        - disturbance_force (3,) at indices [0:3], representation: rep_Rd
        - disturbance_torque (3,) at indices [3:6], representation: rep_Rd_pseudo
        - start disturbance boolean (1,) at index [6], representation: rep_trivial

    The conversion process takes these measurements and does the following:
        1. Stores them into a DynamicsRecording format for easy loading.
        2. Defines the group representation for each observation.
        3. Changes joint position to Pinocchio convention, used by MorphoSymm.
    """
    assert data_path.exists(), f"Path {data_path.absolute()} does not exist"

    with open(data_path, 'rb') as f:
        state = np.squeeze(np.load(f))
        reference = np.squeeze(np.load(f))
        input = np.squeeze(np.load(f))
        external_disturbance = np.squeeze(np.load(f))  # (n_samples, 7) : 7: 6 for the wrench, 1 for the boolean

    # Ensure the data has the expected shapes:
    assert state.shape[1] == 85, f"State shape is {state.shape}, expected (time, 85)"
    assert reference.shape[1] == 22, f"Reference shape is {reference.shape}, expected (time, 22)"
    assert input.shape[1] == 24, f"Input shape is {input.shape}, expected (time, 24)"
    assert external_disturbance.shape[1] == 7, f"External Disturbance shape is {external_disturbance.shape}, expected (time, 7)"

    # Here you can add the transformations and checks on the data as needed

    # Define the dataset.
    data_recording = DynamicsRecording(
        description=f"Mini Cheetah {data_path.parent.parent.stem}",
        info=dict(num_traj=1,
                  trajectory_length=state.shape[0]),
        dynamics_parameters=dict(dt=0.001 * dt_subsample, group=dict(group_name=G.name, group_order=G.order())),
        recordings=dict(state=state,
                        reference=reference,
                        input=input,
                        external_disturbance=external_disturbance),
        state_obs=(
            'joint_pos', 'joint_vel', 'base_z_error', 'base_ori', 'base_ori_error', 'base_vel_error',
            'base_ang_vel_error'),
        action_obs=('joint_torques',),
        obs_representations=dict(),  # Add the representations for each observation here
        obs_moments=dict()  # Add the moments for each observation here
        )

    # Compute the mean and variance of all observations considering symmetry constraints.
    for obs_name in data_recording.recordings.keys():
        if obs_name in data_recording.obs_moments:
            continue
        data_recording.compute_obs_moments(obs_name=obs_name)

    file_name = (f"n_trajs={data_recording.info['num_traj']}"
                 f"-frames={data_recording.info['trajectory_length']}.pkl")
    data_recording.save_to_file(data_path.parent.parent / file_name)
    print(f"Dynamics Recording saved to {data_path.parent.parent / file_name}")
    return data_recording

if __name__ == '__main__':
    data_path = pathlib.Path('./data_estimator.npy')
    data_recording = convert_mini_cheetah_sb_controller_recording(data_path)
    print("done")

    # Read the saved data

with open('./data_estimator.npy', 'rb') as f:
    state = np.squeeze(np.load(f))
    reference = np.squeeze(np.load(f))
    input = np.squeeze(np.load(f))
    external_disturbance = np.squeeze(np.load(f))  # (n_samples, 7) : 7: 6 for the wrench, 1 for the boolean

    # Create a scatterplot matrix of disturbances
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    not_disturbed = np.alltrue(np.isclose(external_disturbance[:, :6], 0, atol=1e-2, rtol=1e-2), axis=1)
    external_disturbance[:, -1] = np.logical_not(not_disturbed)
    df = pd.DataFrame(external_disturbance, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'is_disturbed'])
    sns.pairplot(df, hue='is_disturbed')
    plt.show()

    print(state.shape)
