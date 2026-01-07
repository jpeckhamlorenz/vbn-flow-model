import os
from pathlib import Path
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

def create_dataset_from_exp(exp_file, pattern_file, save_path = Path('./dataset/all_samples'), verbose = False):
    """
    Create a dataset from the experimental samples in the specified directory.
    """

    if verbose:
        pattern_id = exp_file.name.split('_')[exp_file.name.split('_').index('pattern') + 1]
        print(pattern_id)
        print(exp_file)
        print(pattern_file)
        print('-' * 50)

    exp_data = np.load(exp_file, allow_pickle=True)
    pattern_data = pd.read_csv(pattern_file)

    t_exp = exp_data['time']
    t_exp_print = exp_data['time_print']
    Q_out_exp = np.array(exp_data['flowrates'], dtype='float64') / 1e9 # Convert from uL/s to m3/s

    t_com = pattern_data['t'].to_numpy()
    W_com = pattern_data['w'].to_numpy() / 1000 # Convert from mm to m
    W_com_pred = pattern_data['w_pred'].to_numpy() / 1000 # Convert from mm to m
    Q_com = pattern_data['q'].to_numpy() / 1e9  # Convert from uL/s to m3/s

    # Use flow_predictor to get VBN flow rates
    _, _, _, Q_vbn = flow_predictor(t_com-t_com[0], Q_com, W_com_pred)

    # interpolate Q_vbn to the experimental time points
    Q_com_upsample = np.interp(t_exp, t_com, Q_com)
    Q_vbn_upsample = np.interp(t_exp, t_com, Q_vbn)

    W_exp = np.interp(t_exp, t_com, W_com_pred)
    Q_res = Q_out_exp - Q_vbn_upsample  # Calculate residuals for the simulated flow rate

    save_file = save_path / (exp_file.name.rsplit('_', maxsplit=2)[0] + '.npz')
    save_file.parent.mkdir(exist_ok=True)

    lower_cutoff = t_com[0] + 2.5  # seconds
    upper_cutoff = t_com[-1] - 1.5  # seconds

    np.savez(save_file,
             time=t_exp[(t_exp >= lower_cutoff) & (t_exp <= upper_cutoff)],  # Only save time steps greater than 0.2s to match VBN model behavior
             Q_com=Q_com_upsample[(t_exp >= lower_cutoff) & (t_exp <= upper_cutoff)],
             Q_exp=Q_out_exp[(t_exp >= lower_cutoff) & (t_exp <= upper_cutoff)],
             Q_vbn=Q_vbn_upsample[(t_exp >= lower_cutoff) & (t_exp <= upper_cutoff)],
             Q_res=Q_res[(t_exp >= lower_cutoff) & (t_exp <= upper_cutoff)],
             W_com = W_exp[(t_exp >= lower_cutoff) & (t_exp <= upper_cutoff)]
             )

    # interpolate Q_out_exp to the commanded time points
    Q_out_exp_downsample = np.interp(t_com, t_exp, Q_out_exp)
    Q_res_downsample = Q_out_exp_downsample - Q_vbn  # Calculate residuals for the simulated flow rate

    plt.close('all')

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_com[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             1e9*Q_com[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             label='Commanded Flow Rate', alpha=0.5)
    plt.plot(t_com[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             1e9*Q_out_exp_downsample[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             label='Simulated Flow Rate', alpha=0.5)
    plt.plot(t_com[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             1e9*Q_vbn[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             label='VBN Flow Rate', alpha=0.5)
    plt.plot(t_com[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             1e9*Q_res_downsample[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             label='Residual (Simulated - VBN)', linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Flow Rate [mm3/s]')
    plt.title('Flow Rate Comparison')
    plt.ylim(-5,15)
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t_com[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             1000*W_com_pred[(t_com >= lower_cutoff) & (t_com <= upper_cutoff)],
             label='Bead Width', color='orange', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Bead Width [mm]')
    plt.title(exp_file.name)
    plt.ylim(0, 3)
    plt.legend()
    plt.grid()

    plt.subplots_adjust(hspace=0.4)





    vis_file = save_file.parent / ('dataset_visualization/' + exp_file.name.rsplit('_', maxsplit=2)[0] + '.png')
    vis_file.parent.mkdir(exist_ok=True)

    plt.savefig(vis_file)



def create_averaged_dataset(data, window_size):
    """
    Generates dataset of patterns that have been averaged across sample cycles of of the same pattern.
    """
    filelist = sorted(Path(data).glob('*.npz'))




#%%


if __name__ == '__main__':
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

    # exp_folderpath = os.path.join(PROJECT_PATH, 'dataset/exp_samples')
    # filepath_list = os.listdir(exp_folderpath)
    # list all files that end with .csv

    exp_filelist = sorted(Path('./dataset/exp_samples').glob('*.npz'))


    for exp_file in exp_filelist:

        pattern_filelist = Path('./dataset/exp_samples').glob('*.csv')

        pattern_id = exp_file.name.split('_')[exp_file.name.split('_').index('pattern') + 1]
        for pattern_file in pattern_filelist:
            if f'pattern_{pattern_id}' in pattern_file.name:
                break

        create_dataset_from_exp(exp_file, pattern_file)


    # exp_file = exp_filelist[20]
    # pattern_filelist = Path('./dataset/exp_samples').glob('*.csv')
    #
    # pattern_id = exp_file.name.split('_')[exp_file.name.split('_').index('pattern') + 1]
    # for pattern_file in pattern_filelist:
    #     if f'pattern_{pattern_id}' in pattern_file.name:
    #         break
    #
    # create_dataset_from_exp(exp_file, pattern_file)


    # if filename.endswith('.csv'):
        # exp_filepath = os.path.join(exp_folderpath, filename)
        # save_filepath = os.path.join(PROJECT_PATH, 'dataset/all_samples', filename.rsplit('.',1)[0] + '.npz')
        #
        # print(f'Processing {filename}...')
        # create_dataset_from_exp(exp_filepath, save_filepath)
        # print(f'Saved dataset to {save_filepath}')
          # Just for better readability in the console output

    print('done')
    print('-'*50)
