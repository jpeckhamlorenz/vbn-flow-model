import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flow_predictor_analytical import flow_predictor
from constants.filepath import PROJECT_PATH


def create_dataset_from_sim(sim_filepath, save_filepath):
    """
    Create a dataset from the simulation samples in the specified directory.
    """
    # Load the simulation data
    saved_data = np.load(sim_filepath)
    Q_out_sim, Q_com_sim, P_p_sim, t_sim = saved_data['Qo_x'], saved_data['Qcom_x'], saved_data['Pp_x'], saved_data['time']

    # Define W_com (constant for now)
    W_com_sim = np.ones(np.shape(t_sim)) * 0.0029

    # Use flow_predictor to get VBN flow rates
    _, _, _, Q_vbn = flow_predictor(t_sim, Q_com_sim, W_com_sim)

    Q_res = Q_out_sim - Q_vbn  # Calculate residuals for the simulated flow rate

    # Save the dataset
    if not os.path.exists(os.path.dirname(save_filepath)):
        os.makedirs(os.path.dirname(save_filepath))

    np.savez(save_filepath,
             time=t_sim[t_sim >= 0.2],  # Only save time steps greater than 0.2s to match VBN model behavior
             Q_com=Q_com_sim[t_sim >= 0.2],
             Q_sim=Q_out_sim[t_sim >= 0.2],
             Q_vbn=Q_vbn[t_sim >= 0.2],
             Q_res=Q_res[t_sim >= 0.2]
             )

    plt.close('all')

    plt.plot(t_sim[t_sim >= 0.2], Q_com_sim[t_sim >= 0.2], label='Commanded Flow Rate', alpha=0.5)
    plt.plot(t_sim[t_sim >= 0.2], Q_out_sim[t_sim >= 0.2], label='Simulated Flow Rate', alpha=0.5)
    plt.plot(t_sim[t_sim >= 0.2], Q_vbn[t_sim >= 0.2], label='VBN Flow Rate', alpha=0.5)
    plt.plot(t_sim[t_sim >= 0.2], Q_res[t_sim >= 0.2], label='Residual (Simulated - VBN)', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Flow Rate [m3/s]')
    plt.title('Flow Rate Comparison')
    plt.legend()
    plt.grid()

    vis_filepath = os.path.join(os.path.dirname(save_filepath), 'dataset_visualization')
    vis_filename = os.path.basename(sim_filepath).rsplit('.',1)[0] + '.png'

    if not os.path.exists(vis_filepath):
        os.makedirs(vis_filepath)

    plt.savefig(os.path.join(vis_filepath, vis_filename))
    # plt.savefig(os.path.join(os.path.dirname(save_filepath), 'test2.png'))

def create_dataset_from_exp(exp_filepath, save_filepath):
    """
    Create a dataset from the experimental samples in the specified directory.
    """
    # Load the experimental data
    saved_data = pd.read_csv(exp_filepath)
    


#%%

# # Example usage to create datasets from all simulation files in a directory
# sim_folderpath = os.path.join(PROJECT_PATH, 'dataset/sim_samples')
# filepath_list = os.listdir(sim_folderpath)
#
# for filename in filepath_list:
#     if filename.endswith('.npz'):
#         sim_filepath = os.path.join(sim_folderpath, filename)
#         save_filepath = os.path.join(PROJECT_PATH, 'dataset/all_samples', filename)
#
#         print(f'Processing {filename}...')
#         create_dataset_from_sim(sim_filepath, save_filepath)
#         print(f'Saved dataset to {save_filepath}')
#         print('-'*50)  # Just for better readability in the console output

exp_folderpath = os.path.join(PROJECT_PATH, 'dataset/exp_samples')
filepath_list = os.listdir(exp_folderpath)

filename = filepath_list[2]
if filename.endswith('.csv'):
    exp_filepath = os.path.join(exp_folderpath, filename)
    save_filepath = os.path.join(PROJECT_PATH, 'dataset/all_samples', filename.rsplit('.',1)[0] + '.npz')

    print(f'Processing {filename}...')
    create_dataset_from_exp(exp_filepath, save_filepath)
    print(f'Saved dataset to {save_filepath}')
    print('-'*50)  # Just for better readability in the console output
