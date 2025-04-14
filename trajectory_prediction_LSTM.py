# file for training the LSTM model for predicting flow behavior with no physical prior

import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants.filepath import PROJECT_PATH
from constants.plotting import font

plt.close('all')


# %%
device = torch.device("mps")

training_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_training')
validation_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_validation')


# plt.rcParams['figure.dpi'] = 600  # For inline display in the notebook
# plt.rcParams['savefig.dpi'] = 600  # For saving figures to files


# %%
def load_data(data_path, regularization_factor = 1e9):
    data_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')])
    sequences = []
    for file in data_files:
        data = np.load(file)
        time = torch.tensor(data['time'], dtype=torch.float32)
        command = torch.tensor(regularization_factor*data['Q_com'], dtype=torch.float32)
        bead = torch.full_like(time, 0.0029, dtype=torch.float32)

        input_features = torch.stack((time, command, bead), dim=1)
        # input_features = torch.stack((command, analytical), dim=1)
        target_output = torch.tensor(regularization_factor*data['Q_sim'], dtype=torch.float32)

        sequences.append((input_features, target_output))
    return sequences


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return input_seq, target_seq, len(input_seq)


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)  # Sort by sequence length (descending)
    inputs, targets, lengths = zip(*batch)

    padded_inputs = pad_sequence(inputs, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)

    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_inputs, padded_targets, lengths

class FlowrateLSTM(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, output_size=1):
        super(FlowrateLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.lstm(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output

# %%

train_sequences = load_data(training_folderpath)
val_sequences = load_data(validation_folderpath)


train_dataset = SequenceDataset(train_sequences)
val_dataset = SequenceDataset(val_sequences)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# %%

model = FlowrateLSTM().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100

train_epoch_losses = []
val_epoch_losses = []
pbar = tqdm(range(epochs))


#%%
model.train()
for epoch in pbar:
    train_epoch_loss = 0
    val_epoch_loss = 0
    train_batch_cnt = 0
    for batch_inputs, batch_targets, lengths in train_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()

        outputs = model(batch_inputs, lengths)
        loss = criterion(outputs[:,:,0], batch_targets)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_epoch_loss += loss
            train_batch_cnt += 1

    with torch.no_grad():
        val_batch_cnt = 0
        for val_batch_inputs, val_batch_targets, val_lengths in val_loader:
            val_batch_cnt += 1
            val_batch_inputs = val_batch_inputs.to(device)
            val_batch_targets = val_batch_targets.to(device)
            val_outputs = model(val_batch_inputs, val_lengths)
            val_loss = criterion(val_outputs[:,:,0], val_batch_targets)
            val_epoch_loss += val_loss

    train_epoch_losses.append(train_epoch_loss.to('cpu') / train_batch_cnt)
    val_epoch_losses.append(val_epoch_loss.to('cpu') / val_batch_cnt)

    pbar.set_postfix({'iter': epoch, 'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss})


    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
# %%

train_epoch_losses = np.array(train_epoch_losses)
val_epoch_losses = np.array(val_epoch_losses)


# Plotting the training and validation losses
fig_loss = plt.figure("training and validation losses")
plt.plot(np.arange(epochs), train_epoch_losses, c='r', label='train loss (MSE)')
plt.plot(np.arange(epochs), val_epoch_losses, c='g', label='val loss (MSE)')
plt.legend()


# Save the model
torch.save(model.state_dict(), 'lstm_flowrate_model_v0.pth')

print("training complete")



#%%


test_model = FlowrateLSTM().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)

state_dict = torch.load('lstm_flowrate_model_v0.pth')
test_model.load_state_dict(state_dict)


#%%

test_model.eval()

testing_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_testing')
test_sequences = load_data(testing_folderpath)
test_dataset = SequenceDataset(test_sequences)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

#%%
with torch.no_grad():
    test_batch_cnt = 0
    for test_batch_inputs, test_batch_targets, test_lengths in test_loader:
        test_batch_cnt += 1
        test_batch_inputs = test_batch_inputs.to(device)
        test_batch_targets = test_batch_targets.to(device)
        test_outputs = test_model(test_batch_inputs, test_lengths)
        test_loss = criterion(test_outputs[:, :, 0], test_batch_targets)
        print(f"Test Loss: {test_loss.item()}")

test_result = (test_outputs).cpu().numpy()
test_inputs = (test_batch_inputs).cpu().numpy()

#%%
batchID = 1

test_time = np.trim_zeros(test_inputs[batchID][:,0],'b')
test_command = test_inputs[batchID][:len(test_time),1] / 1e9
test_bead = test_inputs[batchID][:len(test_time),2]
test_output = test_result[batchID].reshape(-1)[:len(test_time)] / 1e9
test_sim_output = (test_batch_targets[batchID]).cpu().numpy().reshape(-1)[:len(test_time)] / 1e9

test_output_lstm = test_output
test_output_sim = test_sim_output
# grab the analytical output from the original npz file
batch_files = sorted([os.path.join(testing_folderpath, f) for f in os.listdir(testing_folderpath) if f.endswith('.npz')])
test_output_analytical = np.load(batch_files[batchID])['Q_vbn']





# %% figure:  flowrate prediction

fig_flows = plt.figure("flowrate prediction")
ax = fig_flows.add_subplot(1, 1, 1)

plt.xlabel('Time [s]', fontdict=font)
plt.ylabel('Output Flowrate [m3/s]', fontdict=font)

# plt.xlim(-1,31)
# plt.ylim(0,500)

plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

plt.xscale('linear')
plt.yscale('linear')

plt.plot(test_time, test_command,
         color='black', linewidth = 2, linestyle='--',
         label = 'Commanded Flowrate')
plt.plot(test_time, test_output_sim,
         color='red', linewidth = 2, linestyle='-',
         label = 'Simulation Prediction')
plt.plot(test_time, test_output_lstm,
         color='blue', linewidth = 2, linestyle='-',
         label = 'LSTM Model')
plt.plot(test_time, test_output_analytical,
         color='green', linewidth = 2, linestyle='-',
         label = 'Analytical Model')


# ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

# ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

# ax.legend()

leg_flows = plt.figure("flowrate legend")
leg_flows.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])


# %% figure:  errors in time-series

fig_residuals = plt.figure("prediction errors")
ax = fig_residuals.add_subplot(1, 1, 1)

plt.xlabel('Time [s]', fontdict=font)
plt.ylabel('Flow Prediction Error [m3/s]', fontdict=font)

# plt.xlim(-1,31)
# plt.ylim(0,500)

plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

plt.xscale('linear')
plt.yscale('linear')

plt.plot(test_time, test_output_sim - test_output_lstm,
         color='blue', linewidth = 2, linestyle='-',
         label = 'Commanded Flowrate')
plt.plot(test_time, test_output_sim - test_output_analytical,
         color='red', linewidth = 2, linestyle='-',
         label = 'Simulation Prediction')



# ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

# ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

# ax.legend()

leg_residuals = plt.figure("prediction errors legend")
leg_residuals.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])


# %% error histograms

analytical_error = np.sqrt(np.mean(((test_output_sim) - (test_output_analytical)) ** 2))
lstm_error = np.sqrt(np.mean(((test_output_sim) - (test_output_lstm)) ** 2))

print('Analytical RMSE:', analytical_error)
print("LSTM RMSE:", lstm_error)

# Plotting error histograms
fig_error = plt.figure("error histogram")
ax = fig_error.add_subplot(1, 1, 1)
plt.xlabel('Flow Prediction Error [mm3/s]', fontdict=font)
plt.ylabel('Frequency', fontdict=font)
plt.title('Error Histogram', fontdict=font)
plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)
# Plotting the histograms
ax.hist((test_output_sim - test_output_analytical)*1e9, bins=40, alpha=0.5, color='red', label='Analytical Error', density=True)
ax.hist((test_output_sim - test_output_lstm)*1e9, bins=40, alpha=0.5, color='blue', label='LSTM Error', density=True)


plt.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero Error')

# ax.legend(loc='upper right', fontsize='small')

leg_error = plt.figure("error legend")
leg_error.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])

# %% save figs

# fig_flows.savefig('flows.png',bbox_inches='tight',dpi=600)
# leg_flows.savefig('flows_leg.png',bbox_inches='tight',dpi=600)

# fig_residuals.savefig('errors.png',bbox_inches='tight',dpi=600)
# leg_residuals.savefig('errors_leg.png',bbox_inches='tight',dpi=600)

# fig_error.savefig('error_histogram.png', bbox_inches='tight', dpi=600)
# leg_error.savefig('error_histogram_leg.png', bbox_inches='tight', dpi=600)

# fig_loss.savefig('FlowrateLSTM_training_validation_losses.png', bbox_inches='tight', dpi=600)


