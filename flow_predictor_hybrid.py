# file for deploying/testing the LSTM model for predicting flow errors of the analytical model

import os
# import numpy as np
import torch
import matplotlib.pyplot as plt
from constants.filepath import PROJECT_PATH
from flow_predictor import flow_predictor as flow_predictor_analytical
from pathgen import training_twosteps


#%%
class ResidualLSTM(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=3, output_size=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output

# %%

# def prep_data(time_np, command_np, bead_np, analytical_np, flowrate_regularization=1e9):
#
#     time_tensor = torch.tensor(time_np, dtype=torch.float32)
#     command_tensor = torch.tensor(command_np*flowrate_regularization, dtype=torch.float32)
#     bead_tensor = torch.tensor(bead_np, dtype=torch.float32)
#     analytical_tensor = torch.tensor(analytical_np*flowrate_regularization, dtype=torch.float32)
#
#     # input_features = torch.stack((time_tensor, command_tensor, bead_tensor, analytical_tensor), dim=1)
#     input_features = torch.stack((time_tensor, command_tensor, analytical_tensor), dim=1)
#
#     return input_features  # Return the input features and their length


# %%

def flow_predictor_lstm(time_np, command_np, bead_np,
                   flowrate_regularization=1e9, model_filename ='lstm_residual_model_v7.pth', device_type = 'mps'):

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
    model = ResidualLSTM().to(device)

    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        output = model(input_features.to(device))

    residual = (output).cpu().numpy().squeeze() / 1e9

    output_flow = analytical_np + residual

    return output_flow, residual, analytical_np



# %%

if __name__ == "__main__":
    test = training_twosteps(flowrate_magnitudes=[[0.2, 0.001], [0.48, 0.001]],
                             flowrate_times=[[0.0001, 4], [7, 16]],
                             beadwidth_magnitudes=[[]],
                             beadwidth_times=[[]],
                             dt=0.001,
                             tmax=20)
    time, command, bead = test.pathgen()


    test_result, residual, analytical = flow_predictor_lstm(time, command, bead,
                       flowrate_regularization=1e9, model_filename='lstm_residual_model_v7.pth')
    # plot results
    plot_skip = 0.2  # Adjust this to skip more points if needed
    plt.figure(figsize=(12, 6))
    plt.plot(time[time>plot_skip], command[time>plot_skip], label='Commanded Flow Rate', color='black', linestyle='--')
    plt.plot(time[time>plot_skip], analytical[time>plot_skip], label='Analytical Flow Rate', color='blue')
    plt.plot(time[time>plot_skip], residual[time>plot_skip], label='Residual Prediction', color='magenta')
    plt.plot(time[time>plot_skip], test_result[time>0.2], label='Total Flow Rate', color='green')