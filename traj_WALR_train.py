import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import wandb

import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
# from tqdm import tqdm
from constants.filepath import PROJECT_PATH
from constants.plotting import font

from traj_WALR import LightningModule, DataModule, DictToObject

plt.close('all')

# %%
# class ConfigObject:
#     def __init__(self, dictionary):
#         for key, value in dictionary.items():
#             setattr(self, key, value)


# class ResidualLSTM(torch.nn.Module):
#     def __init__(self, input_size=4, hidden_size=128, num_layers=3, output_size=1):
#         super(ResidualLSTM, self).__init__()
#         self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = torch.nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, lengths):
#         packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
#         packed_output, _ = self.lstm(packed_x)
#         output, _ = pad_packed_sequence(packed_output, batch_first=True)
#         output = self.fc(output)
#         return output


# class SequenceDataset(torch.utils.data.Dataset):
#     def __init__(self, sequences):
#         self.sequences = sequences
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, idx):
#         input_seq, target_seq = self.sequences[idx]
#         return input_seq, target_seq, len(input_seq)


# class DataModule(pl.LightningDataModule):
#     def __init__(self, config,
#                    train_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_training'),
#                    val_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_validation'),
#                    test_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_testing')):
#         super().__init__()
#         self.train_folderpath = train_folderpath
#         self.val_folderpath = val_folderpath
#         self.test_folderpath = test_folderpath
#
#
#     def _load_data(self,data_path, regularization_factor=1e9):
#         data_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')])
#         sequences = []
#         for file in data_files:
#             data = np.load(file)
#             time = torch.tensor(data['time'], dtype=torch.float32)
#             command = torch.tensor(regularization_factor * data['Q_com'], dtype=torch.float32)
#             analytical = torch.tensor(regularization_factor * data['Q_vbn'], dtype=torch.float32)
#             # todo: generate a dataset that has bead width and load from that instead of making my own bead array
#             bead = torch.full_like(time, 0.0029, dtype=torch.float32)
#
#             input_features = torch.stack((time, command, bead, analytical), dim=1)
#             # input_features = torch.stack((command, analytical), dim=1)
#             target_residuals = torch.tensor(regularization_factor * data['Q_res'], dtype=torch.float32)
#
#             sequences.append((input_features, target_residuals))
#         return sequences
#
#     def _collate_fn(self, batch):
#         batch.sort(key=lambda x: x[2], reverse=True)  # Sort by sequence length (descending)
#         inputs, targets, lengths = zip(*batch)
#
#         padded_inputs = pad_sequence(inputs, batch_first=True)
#         padded_targets = pad_sequence(targets, batch_first=True)
#
#         lengths = torch.tensor(lengths, dtype=torch.long)
#         return padded_inputs, padded_targets, lengths
#
#     def train_dataloader(self):
#         train_folderpath = self.train_folderpath
#         train_sequences = self._load_data(train_folderpath)
#         train_dataset = SequenceDataset(train_sequences)
#         return torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=self._collate_fn)
#
#
#     def val_dataloader(self):
#         val_folderpath = self.val_folderpath
#         val_sequences = self._load_data(val_folderpath)
#         val_dataset = SequenceDataset(val_sequences)
#         return torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=self._collate_fn)
#
#
#     def test_dataloader(self):
#         test_folderpath = self.test_folderpath
#         test_sequences = self._load_data(test_folderpath)
#         test_dataset = SequenceDataset(test_sequences)
#         return torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=self._collate_fn)
#
#
#     def predict_dataloader(self):
#         test_folderpath = self.test_folderpath
#         test_sequences = self._load_data(test_folderpath)
#         test_dataset = SequenceDataset(test_sequences)
#         return torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=self._collate_fn)



# class LightningModule(pl.LightningModule):
#     def __init__(self, config):  # hidden_size, num_layers, lr):
#         super().__init__()
#         self.net = ResidualLSTM(hidden_size=config.hidden_size, num_layers = config.num_layers)
#         self.lr = config.lr
#
#     #         self.save_hyperparameters()  # **wandb process fail to finish if this is uncommented**
#     def _r2_score(self, pred, target):
#         ss_res = torch.sum((target - pred) ** 2)
#         ss_tot = torch.sum((target - torch.mean(target)) ** 2)
#         return 1 - ss_res / ss_tot
#
#     def _rmse_accuracy(self, pred, target):
#         rmse = torch.sqrt(torch.mean((pred - target) ** 2))
#         scale = torch.mean(torch.abs(target))  # or max(abs(target))
#         return 1.0 - (rmse / scale)
#
#     def _percent_within(self, pred, target, tolerance=0.05):
#         error = torch.abs(pred - target) / torch.clamp(torch.abs(target), min=1e-9)
#         return torch.mean((error <= tolerance).float())
#
#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         input, target, length = batch
#         output = self.net(input, length)
#         loss = F.mse_loss(output[:,:,0], target)
#         self.log('train/loss', loss, prog_bar=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         input, target, length = batch
#         output = self.net(input, length)
#         loss = F.mse_loss(output[:,:,0], target)
#         r2_score = self._r2_score(output[:, :, 0], target)
#         rmse_accuracy = self._rmse_accuracy(output[:, :, 0], target)
#         percent_within = self._percent_within(output[:, :, 0], target)
#         self.log_dict({
#             "validate/loss": loss,
#             "validate/r2-score": r2_score,
#             "validate/rmse-accuracy": rmse_accuracy,
#             "validate/percent-within": percent_within,
#             })
#         # self.log('validation-loss', loss)
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         input, target, length = batch
#         output = self.net(input, length)
#         loss = F.mse_loss(output[:,:,0], target)
#         r2_score = self._r2_score(output[:, :, 0], target)
#         rmse_accuracy = self._rmse_accuracy(output[:, :, 0], target)
#         percent_within = self._percent_within(output[:, :, 0], target)
#         self.log_dict({
#             "test/loss": loss,
#             "test/r2-score": r2_score,
#             "test/rmse-accuracy": rmse_accuracy,
#             "test/percent-within": percent_within,
#         })
#         # self.log('test-loss', loss)
#         return loss
#
#     def configure_optimizers(self):
#         # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer
#
#     def predict_step(self, batch, batch_idx):
#         input, target, length = batch
#         output = self.net(input, length)
#         return output


def train_model(run_config):

    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    #     os.environ["WANDB_START_METHOD"] = "thread"'
    # wandb.run = None  # Reset wandb run to avoid conflicts
    # wandb.init(project="VBN-modeling",
    #            notes = 'test1',
    #            group = 'test2')
    config = DictToObject(run_config)
    # wandb_logger = WandbLogger()
    data = DataModule(config)
    module = LightningModule(config)

    # wandb_logger.watch(module.net)

    trainer = pl.Trainer(accelerator='mps', devices=1, max_epochs=50, num_sanity_val_steps=0,
                         default_root_dir="./lightning-test")
    #     wandb.require(experiment="service")
    trainer.fit(module, data)

    return trainer, module, data


if __name__ == '__main__':

    run_config = {
        'lr': 0.001,
        'num_layers': 4,
        'hidden_size': 128,
        'batch_size': 2,
    }

    trained_model, module, data = train_model(run_config)

    trained_model.test(module, datamodule=data)
    prediction = trained_model.predict(module, datamodule=data)

    # %%

    inputs = trained_model.predict_dataloaders.dataset
    outputs = prediction[0].cpu().detach().numpy()[:, :, 0] / 1e9

    for input, output in zip(inputs, outputs):
        time = input[0][:, 0].cpu().detach().numpy()
        command = input[0][:, 1].cpu().detach().numpy() / 1e9
        bead = input[0][:, 2].cpu().detach().numpy()
        analytical = input[0][:, 3].cpu().detach().numpy() / 1e9
        target = input[1].cpu().detach().numpy() / 1e9

        # Plot the results

        # plt.figure(figsize=(10, 6))
        # plt.plot(time, output[:len(time)], label='Predicted Flow Rate')
        # plt.plot(time, target[:len(time)], label='Analytical Flow Rate', linestyle='--')
        # plt.xlabel('Time (s)', fontsize=font['size'])
        # plt.ylabel('Flow Rate (mL/min)', fontsize=font['size'])
        # plt.title('WALR Flow Rate Prediction vs Analytical Model', fontsize=font['size'])
        # plt.legend(fontsize=font['size'])
        # plt.grid(True)
        # plt.show()


        fig_residuals = plt.figure(str(analytical[-1]))
        ax = fig_residuals.add_subplot(1, 1, 1)

        plt.xlabel('Time [s]', fontdict=font)
        plt.ylabel('Flow Prediction Error [m3/s]', fontdict=font)

        # plt.xlim(-1,31)
        # plt.ylim(0,500)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(time, output[:len(time)],
                 color='blue', linewidth=2, linestyle='-',
                 label='Commanded Flowrate')
        plt.plot(time, target[:len(time)],
                 color='red', linewidth=2, linestyle='-',
                 label='Simulation Prediction')

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        leg_residuals = plt.figure("residual legend")
        leg_residuals.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])