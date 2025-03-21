import matplotlib.pyplot as plt
from pathgen import corner_naive
import numpy as np
import torch
from glob import glob
from flow_predictor import flow_predictor
from tqdm import tqdm
from pathgen import training_twosteps

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    # x = torch.ones(1, device=mps_device)
    # print(x)
else:
    print("MPS device not found.")

plt.close('all')

# %%

location = '/Users/james/Desktop/sim_samples/'
workspace_list = glob(location + '*.npz')

residuals = []
command_data = []

print("Loading data...")
for file in tqdm(workspace_list):
    saved_data = np.load(file)
    Q_out_sim, Q_com_sim, P_p_sim, t_sim = saved_data['Qo_x'], saved_data['Qcom_x'], saved_data['Pp_x'], saved_data['time']
    W_com_sim = np.ones(np.shape(t_sim)) * 0.0029
    residuals.append(Q_out_sim)
    command_data.append(Q_com_sim)

residuals = np.concatenate(residuals) * 1e9
command_data = np.concatenate(command_data) * 1e9

# %%



# input_features = torch.tensor(analytical_data, dtype=torch.float32, device=mps_device).view(-1, 1)
input_features = torch.tensor(command_data, dtype=torch.float32, device=mps_device).view(-1, 1)
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

        self.fc1 = torch.nn.Linear(1, 256)  # Input layer
        self.fc2 = torch.nn.Linear(256, 512)  # Hidden layer
        self.fc3 = torch.nn.Linear(512, 1024)  # Hidden layer
        self.fc4 = torch.nn.Linear(1024, 2048)  # Output layer
        self.fc5 = torch.nn.Linear(2048, 1024)  # Output layer
        self.fc6 = torch.nn.Linear(1024, 512)  # Output layer
        self.fc7 = torch.nn.Linear(512, 256)  # Output layer
        self.fc8 = torch.nn.Linear(256, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        return self.fc8(x)


# Instantiate the model, define loss function and optimizer
model = ResidualModel()
model.to(mps_device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2000
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
# test_t, test_W, test_Q_com, test_Q_out = flow_predictor(test_t_input, test_Q_input, test_W_input)

# test_analytical = test_Q_out * 1e9
test_command = test_Q_input * 1e9
# test_combined = np.stack((test_command, test_analytical), axis=1)
test_input = torch.tensor(test_command, dtype=torch.float32, device=mps_device).view(-1, 1)

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
plt.plot(test_t_input, test_Q_input, label='Input', color='black', linestyle='--')
# plt.plot(test_accel.ts, test_accel.sim_Q_out, label = 'Simulated Data', color = 'red')
# plt.plot(test_t_input, test_Q_out, label = 'Analytical Data', color = 'blue')
plt.plot(test_t_input, test_result, label='Output', color='magenta')
# plt.plot(test_t, test_Q_out + test_result, label='Total', color='green')

plt.legend()




#%%

torch.save(model.state_dict(), 'trajectory_prediction_DNN_state_v1.pth')
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

