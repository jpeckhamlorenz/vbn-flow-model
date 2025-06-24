import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from constants.filepath import PROJECT_PATH


class WalrLSTM(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=3, output_size=1):
        super(WalrLSTM, self).__init__()
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
        self.train_batch_size = config.batch_size

    def _load_data(self,data_path, regularization_factor=1e9):
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
        return torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=self._collate_fn,
                                           batch_size = self.train_batch_size)

    def val_dataloader(self):
        val_folderpath = self.val_folderpath
        val_sequences = self._load_data(val_folderpath)
        val_dataset = SequenceDataset(val_sequences)
        return torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=self._collate_fn,
                                           batch_size = 2)

    def test_dataloader(self):
        test_folderpath = self.test_folderpath
        test_sequences = self._load_data(test_folderpath)
        test_dataset = SequenceDataset(test_sequences)
        return torch.utils.data.DataLoader(test_dataset, shuffle=False, collate_fn=self._collate_fn,
                                           batch_size = 2)

    def predict_dataloader(self):
        test_folderpath = self.test_folderpath
        test_sequences = self._load_data(test_folderpath)
        test_dataset = SequenceDataset(test_sequences)
        return torch.utils.data.DataLoader(test_dataset, shuffle=False, collate_fn=self._collate_fn,
                                           batch_size=2)


class LightningModule(pl.LightningModule):
    def __init__(self, config):  # hidden_size, num_layers, lr):
        super().__init__()
        self.net = WalrLSTM(hidden_size=config.hidden_size, num_layers = config.num_layers)
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

    def _normalized_mse(self, pred, target):
        mse = F.mse_loss(pred, target)
        norm_factor = torch.mean(torch.abs(target))
        return mse / norm_factor

    def _percent_within(self, pred, target, tolerance=0.10):
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

    def forward(self, x, lengths):
        return self.net(x, lengths)

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def get_best_run(entity = 'jplorenz-university-of-michigan',
                 project = 'VBN-modeling',
                 sweep_id = 'e8wa5l6y'):

    api = wandb.Api()
    # best_run = api.sweep(entity + '/' + project + '/' + sweep_id).best_run()
    # best_run = api.run(entity + '/' + project + '/' + 'g0zp7x5x')
    runs = api.sweep(entity + '/' + project + '/' + sweep_id).runs

    run_losses = {}
    for run in runs:
        run_losses[run.id] = run.summary['validate/loss']

    run_losses_sorted = sorted(run_losses.items(), key=lambda item: item[1])

    best_run_id = run_losses_sorted[0][0]
    best_run = api.run(entity + '/' + project + '/' + best_run_id)

    run_config = DictToObject(best_run.config)
    run_id = best_run.id

    return run_id, run_config