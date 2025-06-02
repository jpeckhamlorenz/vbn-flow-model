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

    def _load_data(self,data_path, regularization_factor=1e9):
        data_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')])
        sequences = []
        for file in data_files:
            data = np.load(file)
            time = torch.tensor(data['time'], dtype=torch.float32)
            command = torch.tensor(regularization_factor * data['Q_com'], dtype=torch.float32)

            # todo: generate a dataset that has bead width and load from that instead of making my own bead array
            bead = torch.full_like(time, 0.0029, dtype=torch.float32)

            input_features = torch.stack((time, command, bead), dim=1)
            # input_features = torch.stack((command, analytical), dim=1)
            target_output = torch.tensor(regularization_factor * data['Q_res'], dtype=torch.float32)

            sequences.append((input_features, target_output))
        return sequences

    def _collate_fn(self, batch):
        batch.sort(key=lambda x: x[2], reverse=True)  # Sort by sequence length (descending)
        inputs, targets, lengths = zip(*batch)

        padded_inputs = pad_sequence(inputs, batch_first=True)
        padded_targets = pad_sequence(targets, batch_first=True)

        lengths = torch.tensor(lengths, dtype=torch.long)
        return padded_inputs, padded_targets, lengths

    def train_dataloader(self):
        train_folderpath = self.train_folderpath
        train_sequences = self._load_data(train_folderpath)
        train_dataset = SequenceDataset(train_sequences)
        return torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=self._collate_fn)

    def val_dataloader(self):
        val_folderpath = self.val_folderpath
        val_sequences = self._load_data(val_folderpath)
        val_dataset = SequenceDataset(val_sequences)
        return torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=self._collate_fn)

    def test_dataloader(self):
        test_folderpath = self.test_folderpath
        test_sequences = self._load_data(test_folderpath)
        test_dataset = SequenceDataset(test_sequences)
        return torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=self._collate_fn)

    def predict_dataloader(self):
        test_folderpath = self.test_folderpath
        test_sequences = self._load_data(test_folderpath)
        test_dataset = SequenceDataset(test_sequences)
        return torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=self._collate_fn)


class LightningModule(pl.LightningModule):
    def __init__(self, config):  # hidden_size, num_layers, lr):
        super().__init__()
        self.net = FlowrateLSTM(hidden_size=config.hidden_size, num_layers = config.num_layers)
        self.lr = config.lr

    #         self.save_hyperparameters()  # **wandb process fail to finish if this is uncommented**
    def _r2_score(self, pred, target):
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        return 1 - ss_res / ss_tot

    def _rmse_accuracy(self, pred, target):
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        scale = torch.mean(torch.abs(target))  # or max(abs(target))
        return 1.0 - (rmse / scale)

    def _percent_within(self, pred, target, tolerance=0.05):
        error = torch.abs(pred - target) / torch.clamp(torch.abs(target), min=1e-9)
        return torch.mean((error <= tolerance).float())

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input, target, length = batch
        output = self.net(input, length)
        loss = F.mse_loss(output[:,:,0], target)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target, length = batch
        output = self.net(input, length)
        loss = F.mse_loss(output[:,:,0], target)
        r2_score = self._r2_score(output[:, :, 0], target)
        rmse_accuracy = self._rmse_accuracy(output[:, :, 0], target)
        percent_within = self._percent_within(output[:, :, 0], target)
        self.log_dict({
            "validate/loss": loss,
            "validate/r2-score": r2_score,
            "validate/rmse-accuracy": rmse_accuracy,
            "validate/percent-within": percent_within,
            })
        # self.log('validation-loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        input, target, length = batch
        output = self.net(input, length)
        loss = F.mse_loss(output[:,:,0], target)
        r2_score = self._r2_score(output[:, :, 0], target)
        rmse_accuracy = self._rmse_accuracy(output[:, :, 0], target)
        percent_within = self._percent_within(output[:, :, 0], target)
        self.log_dict({
            "test/loss": loss,
            "test/r2-score": r2_score,
            "test/rmse-accuracy": rmse_accuracy,
            "test/percent-within": percent_within,
        })
        # self.log('test-loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        input, target, length = batch
        output = self.net(input, length)
        return output


def train_model():
    #     os.environ["WANDB_START_METHOD"] = "thread"'
    wandb.init(project="VBN-modeling",
               notes = 'test1',
               group = 'test2')
    config = wandb.config
    wandb_logger = WandbLogger()
    data = DataModule(config)
    module = LightningModule(config)

    wandb_logger.watch(module.net)

    trainer = pl.Trainer(accelerator='mps', devices=1, max_epochs=70, log_every_n_steps=1,
                         default_root_dir="./lightning-test", logger=wandb_logger)
    #     wandb.require(experiment="service")
    trainer.fit(module, data)


if __name__ == '__main__':
    sweep_config = {
        'description': 'text for description',
        'method': 'bayes',
        'name': 'NALO-sweep-test',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'parameters': {
            'hidden_size': {'values': [32, 64, 96, 128, 256]},
            'num_layers': {'values': [2,3,4,5]},
            'lr': {'max': 0.01, 'min': 0.0001}
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,
            's': 2,
        },
        'run_cap': 30,
    }

    sweep_id = wandb.sweep(sweep_config, project="VBN-modeling")
    wandb.agent(sweep_id = sweep_id, count = 30,
                function = train_model)

    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    best_run = sweep.best_run()
    print("Best Run Name:", best_run.name)
    print("Best Run ID:", best_run.id)
    print(best_run.config)