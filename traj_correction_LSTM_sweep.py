import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import wandb

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

class ResidualLSTM(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=3, output_size=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.lstm(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return input_seq, target_seq, len(input_seq)

class DataModule(pl.LightningDataModule):
    def __init__(self, config,
                   train_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_training'),
                   val_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_validation'),
                   test_folderpath = os.path.join(PROJECT_PATH, 'dataset/LSTM_testing')):
        super().__init__()
        self.train_folderpath = train_folderpath
        self.val_folderpath = val_folderpath
        self.test_folderpath = test_folderpath

    def _load_data(data_path, regularization_factor=1e9):
        data_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')])
        sequences = []
        for file in data_files:
            data = np.load(file)
            time = torch.tensor(data['time'], dtype=torch.float32)
            command = torch.tensor(regularization_factor * data['Q_com'], dtype=torch.float32)
            analytical = torch.tensor(regularization_factor * data['Q_vbn'], dtype=torch.float32)
            # todo: generate a dataset that has bead width and load from that instead of making my own bead array
            bead = torch.full_like(time, 0.0029, dtype=torch.float32)

            input_features = torch.stack((time, command, bead, analytical), dim=1)
            # input_features = torch.stack((command, analytical), dim=1)
            target_residuals = torch.tensor(regularization_factor * data['Q_res'], dtype=torch.float32)

            sequences.append((input_features, target_residuals))
        return sequences

    def _collate_fn(batch):
        batch.sort(key=lambda x: x[2], reverse=True)  # Sort by sequence length (descending)
        inputs, targets, lengths = zip(*batch)

        padded_inputs = pad_sequence(inputs, batch_first=True)
        padded_targets = pad_sequence(targets, batch_first=True)

        lengths = torch.tensor(lengths, dtype=torch.long)
        return padded_inputs, padded_targets, lengths

    def train_dataloader(self,train_folderpath):
        train_sequences = self._load_data(train_folderpath)
        train_dataset = SequenceDataset(train_sequences)
        return torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=self._collate_fn)

    def val_dataloader(self, val_folderpath):
        val_sequences = self._load_data(val_folderpath)
        val_dataset = SequenceDataset(val_sequences)
        return torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=self._collate_fn)

    def test_dataloader(self, test_folderpath):
        test_sequences = self._load_data(test_folderpath)
        test_dataset = SequenceDataset(test_sequences)
        return torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=self._collate_fn)


class LightningModule(pl.LightningModule):
    def __init__(self, config):  # hidden_size, num_layers, lr):
        super().__init__()
        self.net = ResidualLSTM(hidden_size=config.hidden_size, num_layers = config.num_layers)
        self.lr = config.lr

    #         self.save_hyperparameters()  # **wandb process fail to finish if this is uncommented**

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, x)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, x)
        self.log('validation_loss', loss)
        # return {'val_loss': loss}
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


def train_model():
    #     os.environ["WANDB_START_METHOD"] = "thread"'
    wandb.init(project="sweep")
    config = wandb.config
    wandb_logger = WandbLogger()
    data = DataModule(config)
    module = LightningModule(config)

    wandb_logger.watch(module.net)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10,
                         default_root_dir="./lightning-example", logger=wandb_logger)
    #     wandb.require(experiment="service")
    trainer.fit(module, data)


if __name__ == '__main__':
    sweep_config = {
        'method': 'random',
        'name': 'first_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'parameters': {
            'hidden_size': {'values': [2, 3, 5, 10]},
            'num_layers': {'values': [2, 3, 5, 10]},
            'lr': {'max': 1.0, 'min': 0.0001}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="test_sweep")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=5)