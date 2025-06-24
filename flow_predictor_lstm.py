# file for deploying/testing the LSTM model for predicting flow errors of the analytical model

import os
# import numpy as np
import torch
from flow_predictor_analytical import flow_predictor as flow_predictor_analytical
from constants.filepath import PROJECT_PATH


# %%

def flow_predictor_lstm(time_np, command_np, bead_np, model_type,
                   flowrate_regularization=1e9, model_filename = '', device_type = 'mps'):

    _, _, _, analytical_np = flow_predictor_analytical(time_np, command_np, bead_np)
    # _, _, _, Q_vbn = flow_predictor(test.pathgen())

    time_tensor = torch.tensor(time_np, dtype=torch.float32)
    command_tensor = torch.tensor(command_np * flowrate_regularization, dtype=torch.float32)
    bead_tensor = torch.tensor(bead_np, dtype=torch.float32)
    analytical_tensor = torch.tensor(analytical_np * flowrate_regularization, dtype=torch.float32)

    input_features = torch.stack((time_tensor, command_tensor, bead_tensor, analytical_tensor), dim=1)
    # input_features = torch.stack((time_tensor, command_tensor, analytical_tensor), dim=1)

    assert device_type in ['cpu', 'cuda', 'mps'], "Invalid device type. Choose 'cpu', 'cuda', or 'mps'."
    device = torch.device(device_type)

    if model_type == 'WALR':
        from models.traj_WALR import LightningModule, get_best_run
    elif model_type == 'WALO':
        from models.traj_WALO import LightningModule, get_best_run
    elif model_type == 'NALO':
        input_features = torch.stack((time_tensor, command_tensor, bead_tensor), dim=1)
        from models.traj_NALO import LightningModule, get_best_run
    else:
        print(f"Model type {model_type} not recognized. Defaulting to WALR.")
        from models.traj_WALR import LightningModule, get_best_run



    run_id, run_config = get_best_run()
    checkpoint_id = os.listdir(os.path.join(PROJECT_PATH, 'VBN-modeling', run_id, 'checkpoints'))[0]

    model = LightningModule.load_from_checkpoint(
        os.path.join(PROJECT_PATH, 'VBN-modeling', run_id, 'checkpoints', checkpoint_id),
        config=run_config)


    model.eval()

    print("Running LSTM flow prediction...")
    with torch.no_grad():
        output = model(input_features.unsqueeze(0).to(device), torch.tensor(len(input_features)).unsqueeze(0).to(device))

    if model_type == 'WALO' or model_type == 'NALO':
        output_flow = (output).cpu().numpy().squeeze() / 1e9
    else:
        residual = (output).cpu().numpy().squeeze() / 1e9
        output_flow = analytical_np + residual




    return output_flow, analytical_np



# %%

if __name__ == "__main__":
    from inputs.pathgen import training_twosteps
    from constants.filepath import PROJECT_PATH
    import matplotlib.pyplot as plt

    test = training_twosteps(flowrate_magnitudes=[[0.2, 0.001], [0.48, 0.001]],
                             flowrate_times=[[0.0001, 4], [7, 16]],
                             beadwidth_magnitudes=[[]],
                             beadwidth_times=[[]],
                             dt=0.001,
                             tmax=20)
    time, command, bead = test.pathgen()

    prediction, analytical = flow_predictor_lstm(time, command, bead, 'WALR')
    # plot results


    plot_skip = 0.2  # Adjust this to skip more points if needed
    plt.figure(figsize=(12, 6))
    plt.plot(time[time>plot_skip], command[time>plot_skip], label='Commanded Flow Rate', color='black', linestyle='--')
    plt.plot(time[time>plot_skip], analytical[time>plot_skip], label='Analytical Flow Rate', color='blue')
    # plt.plot(time[time>plot_skip], residual[time>plot_skip], label='Residual Prediction', color='magenta')
    plt.plot(time[time>plot_skip], prediction[time>0.2], label='Total Flow Rate', color='green')