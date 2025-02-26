import matplotlib.pyplot as plt
from pathgen import corner_naive
import numpy as np
import torch
from glob import glob
from flow_predictor import flow_predictor
from tqdm import tqdm
from pathgen import training_twosteps

if torch.backends.mps.is_available():
    mps_device = torch.device("cpu")
    # x = torch.ones(1, device=mps_device)
    # print(x)
else:
    print("MPS device not found.")

plt.close('all')
from constants.filepath import PROJECT_PATH
import os

# %%

location = os.path.join(PROJECT_PATH, 'data', 'sim_samples')
workspace_list = glob(os.path.join(location,'*.npz'))

residuals = []
analytical_data = []
command_data = []


# print loading data in red text
print("\033[91mLoading data...\033[0m")
for file in tqdm(workspace_list):
    saved_data = np.load(file)
    Q_out_sim, Q_com_sim, P_p_sim, t_sim = saved_data['Qo_x'], saved_data['Qcom_x'], saved_data['Pp_x'], saved_data['time']
    W_com_sim = np.ones(np.shape(t_sim)) * 0.0029
    t_vbn, W_com_vbn, Q_com_vbn, Q_out_vbn = flow_predictor(t_sim, Q_com_sim, W_com_sim)
    residuals.append(np.interp(t_vbn[t_vbn > 0.2], t_sim, Q_out_sim) - Q_out_vbn[t_vbn > 0.2])
    analytical_data.append(Q_out_vbn[t_vbn > 0.2])
    command_data.append(Q_com_vbn[t_vbn > 0.2])
residuals = np.concatenate(residuals) * 1e9
analytical_data = np.concatenate(analytical_data) * 1e9
command_data = np.concatenate(command_data) * 1e9
# print finished loading data in green text
print("\033[92mFinished loading data.\033[0m")

# %%

input_data = np.stack((command_data, analytical_data), axis=1)


# input_features = torch.tensor(analytical_data, dtype=torch.float32, device=mps_device).view(-1, 1)
input_features = torch.tensor(input_data, dtype=torch.float32, device=mps_device)
target_residuals = torch.tensor(residuals, dtype=torch.float32, device=mps_device).view(-1, 1)


# %%

# Define a simple neural network model for learning the residuals
class ResidualModel(torch.nn.Module):
    def __init__(self):
        super(ResidualModel, self).__init__()
        # self.fc1 = torch.nn.Linear(2, 64)  # Input layer
        # self.fc2 = torch.nn.Linear(64, 256)  # Hidden layer
        # self.fc3 = torch.nn.Linear(256, 64)  # Output layer
        # self.fc4 = torch.nn.Linear(64, 1)  # Output layer

        self.fc1 = torch.nn.Linear(2, 64)  # Input layer
        self.fc2 = torch.nn.Linear(64, 512)  # Hidden layer
        self.fc3 = torch.nn.Linear(512, 1024)  # Hidden layer
        self.fc4 = torch.nn.Linear(1024, 512)  # Output layer
        self.fc5 = torch.nn.Linear(512, 64)  # Output layer
        self.fc6 = torch.nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)


# Instantiate the model, define loss function and optimizer
model = ResidualModel()
model.to(mps_device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 500
model.train()

print("Training model...")
for epoch in range(epochs):

    # Forward pass
    predictions = model(input_features)
    loss = criterion(predictions, target_residuals)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# Example: Run model inference (optional, based on your needs)
model.eval()


#%%

test = training_twosteps(flowrate_magnitudes = [[0.75, 0.001], [0.2, 0.001], [0.48, 0.001]],
                         flowrate_times = [[0.0001, 8], [15, 20], [22, 33]],
                         beadwidth_magnitudes = [[]],
                         beadwidth_times = [[]],
                         dt = 0.001,
                         tmax=40)
test_t_input, test_Q_input, test_W_input = test.pathgen()
test_t, test_W, test_Q_com, test_Q_out = flow_predictor(test_t_input, test_Q_input, test_W_input)

test_analytical = test_Q_out * 1e9
test_command = test_Q_com * 1e9
test_combined = np.stack((test_command, test_analytical), axis=1)
test_input = torch.tensor(test_combined, dtype=torch.float32, device=mps_device)

with torch.no_grad():
    output = model(test_input) * 1e-9
    test_result = (output).cpu().numpy().reshape(-1)

# %%
# plt.figure()
# plt.plot(test_accel.input_t, test_accel.input_Q, label='Input', color='black', linestyle='--')
# plt.plot(test_accel.ts, test_accel.sim_Q_out, label='Sim', color='red')
# plt.plot(test_accel.ts, test_accel.Q_out, label='Prediction', color='blue')
# plt.plot(test_accel.ts, test_accel.error, label='Error', color='magenta')
# plt.legend()

# %%

plt.figure()
plt.plot(test_t, test_Q_com, label='Input', color='black', linestyle='--')
# plt.plot(test_accel.ts, test_accel.sim_Q_out, label = 'Simulated Data', color = 'red')
plt.plot(test_t, test_Q_out, label = 'Analytical Data', color = 'blue')
plt.plot(test_t, test_result, label='Residual', color='magenta')
plt.plot(test_t, test_Q_out + test_result, label='Total', color='green')

plt.legend()




#%%

torch.save(model.state_dict(), 'trajectory_correction_DNN_v0.pth')
#%% testing

# location = '/Users/james/Desktop/sim_samples/'
# workspace_list = glob(location + '*.npz')
#
# for file in workspace_list:
#     plt.close('all')
#     saved_data = np.load(file)
#     Q_o, Q_com, P_p, t = saved_data['Qo_x'], saved_data['Qcom_x'], saved_data['Pp_x'], saved_data['time']
#     W_com = np.ones(np.shape(t)) * 0.0029
#     from flow_predictor import flow_predictor
#     ts, W_commanded, Q_commanded, Q_output = flow_predictor(t, Q_com, W_com)
#     fig, ax = plt.subplots()
#     ax.plot(ts[ts>0.11], Q_commanded[ts>0.11])
#     ax.plot(ts[ts>0.11], Q_output[ts>0.11])
#     ax.plot(t[t>0.11], Q_o[t>0.11])
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     print(file)
#     input("Press Enter to continue...")


#%% tmp

# location = '/Users/james/Desktop/sim data up_to_date/'
# workspace_list = glob(location + 'corner*')
#
# for file in workspace_list:
#     plt.close('all')
#     saved_data = np.load(file)
#     Q_o, Q_com, P_p, t = saved_data['Qo_x'], saved_data['Qcom_x'], saved_data['Pp_x'], saved_data['time']
#     downsampled_rate = 0.001
#     dx = 4 * 1e-5
#     dt = dx/880
#     t[::round(downsampled_rate / dt)]
#     np.savez(file.split('.')[0] + '_sim.npz',
#              Qo_x=Q_o[::round(downsampled_rate / dt)],
#              Qcom_x=Q_com[::round(downsampled_rate / dt)],
#              Pp_x=P_p[::round(downsampled_rate / dt)],
#              time=t[::round(downsampled_rate / dt)])

