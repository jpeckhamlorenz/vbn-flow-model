import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
# import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from constants.filepath import PROJECT_PATH
from collections import defaultdict
from pathlib import Path

class WalrLSTM(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, output_size=1):
        super(WalrLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout = 0.1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.lstm(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, filenames):
        self.sequences = sequences
        self.filenames = filenames

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return input_seq, target_seq, len(input_seq), self.filenames[idx]


class DataModule(pl.LightningDataModule):
    def __init__(self, config, data_folderpath: Path,
                   train_list = 'train.txt',
                   val_list = 'val.txt',
                   test_list = 'test.txt'):
        super().__init__()
        self.data_folderpath = data_folderpath
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.train_batch_size = config.batch_size
        self.huber_delta = config.huber_delta if hasattr(config, 'huber_delta') else 1.0
        self.normalize = True
        self.norm_epsilon  = 1e-8
        self.norm_stats = None
        self.regularization_factor = 1e9

    def setup(self, stage=None):
        if self.normalize and self.norm_stats is None:
            train_files = self._resolve_files(self.train_list)
            self.norm_stats = self._compute_norm_stats(train_files)


    def _resolve_files(self, data_list, split_path: Path = Path("./splits")):
        # Resolve list path
        list_path = Path(data_list)
        if not list_path.is_absolute():
            list_path = split_path / list_path
        if not list_path.exists():
            raise FileNotFoundError(f"Data list file not found: {list_path}")

        with open(list_path, "r") as f:
            rel_names = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#") and line.strip().endswith(".npz")
            ]
        if len(rel_names) == 0:
            raise RuntimeError(f"No files listed in {list_path}")

        files = []
        for name in rel_names:
            p = self.data_folderpath / name
            if not p.exists():
                raise FileNotFoundError(f"Listed file not found: {p}")
            files.append(p)

        return files

    def _compute_norm_stats(self, data_files):
        """
        Compute mean/std per channel using TRAIN FILES ONLY.
        Stats are computed on the same scaled values you feed to the network.
        """
        # We'll accumulate sums and sumsq for: command, bead, analytical, target
        sums = {k: 0.0 for k in ["command", "bead", "analytical", "target"]}
        sqs  = {k: 0.0 for k in ["command", "bead", "analytical", "target"]}
        counts = {k: 0 for k in ["command", "bead", "analytical", "target"]}

        for file in data_files:
            d = np.load(file)

            command = (self.regularization_factor * d["Q_com"]).astype(np.float64)
            analytical = (self.regularization_factor * d["Q_vbn"]).astype(np.float64)
            target = (self.regularization_factor * d["Q_res"]).astype(np.float64)

            if "W_com" in d:
                bead = (1000.0 * d["W_com"]).astype(np.float64)
            else:
                # match your default bead behavior
                bead = np.full_like(command, 1000.0 * 0.0029, dtype=np.float64)

            # Flatten (time dimension)
            for k, x in [("command", command), ("bead", bead), ("analytical", analytical), ("target", target)]:
                x = x.reshape(-1)
                sums[k] += x.sum()
                sqs[k]  += (x * x).sum()
                counts[k] += x.size

        mean = {k: sums[k] / max(counts[k], 1) for k in sums}
        std = {}
        for k in sums:
            var = sqs[k] / max(counts[k], 1) - mean[k] ** 2
            std[k] = float(np.sqrt(max(var, 0.0)) + self.norm_epsilon)

        # store as torch tensors for fast use
        stats = {
            "mean": {k: torch.tensor(mean[k], dtype=torch.float32) for k in mean},
            "std":  {k: torch.tensor(std[k], dtype=torch.float32) for k in std},
        }
        return stats

    def _load_data(self, data_list,
                   split_path: Path = Path('./splits')):

        # Resolve list path
        list_path = Path(data_list)
        if not list_path.is_absolute():
            list_path = split_path / list_path

        if not list_path.exists():
            raise FileNotFoundError(f"Data list file not found: {list_path}")

        # Read filenames from txt, must be .npz files
        with open(list_path, "r") as f:
            filenames = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#") and line.strip().endswith(".npz")
            ]

        if len(filenames) == 0:
            raise RuntimeError(f"No files listed in {list_path}")

        # Resolve full paths and sanity check
        data_files = []
        for name in filenames:
            file_path = self.data_folderpath / name
            if not file_path.exists():
                raise FileNotFoundError(f"Listed file not found: {file_path}")
            data_files.append(file_path)

        sequences = []
        filenames = []
        for file in data_files:
            data = np.load(file)
            # time = torch.tensor(data['time'], dtype=torch.float32)
            # time = torch.tensor(data['time'] - data['time'][0] + 0.5, dtype=torch.float32)
            command = torch.tensor(self.regularization_factor * data['Q_com'], dtype=torch.float32)
            analytical = torch.tensor(self.regularization_factor * data['Q_vbn'], dtype=torch.float32)

            try:
                bead = torch.tensor(1000*data['W_com'], dtype=torch.float32)
                # input('press enter to continue james lorenz')

            except KeyError:
                bead = torch.full_like(command, 1000*0.0029, dtype=torch.float32)

            target_residuals = torch.tensor(self.regularization_factor * data['Q_res'], dtype=torch.float32)

            if self.normalize:
                # make sure setup() has run (safe even if called repeatedly)
                if self.norm_stats is None:
                    self.norm_stats = self._compute_norm_stats(self._resolve_files(self.train_list),
                                                              regularization_factor=self.regularization_factor)

                m = self.norm_stats["mean"]
                s = self.norm_stats["std"]

                command = (command - m["command"]) / s["command"]
                bead = (bead - m["bead"]) / s["bead"]
                analytical = (analytical - m["analytical"]) / s["analytical"]
                target_residuals = (target_residuals - m["target"]) / s["target"]

            # input_features = torch.stack((time, command, bead, analytical), dim=1)
            input_features = torch.stack((command, bead, analytical), dim=1)
            sequences.append((input_features, target_residuals))
            filenames.append(file.name)

        return sequences, filenames

    def _collate_fn(self, batch):
        batch.sort(key=lambda x: x[2], reverse=True)  # Sort by sequence length (descending)
        inputs, targets, lengths, names = zip(*batch)

        padded_inputs = pad_sequence(inputs, batch_first=True)
        padded_targets = pad_sequence(targets, batch_first=True)
        lengths = torch.tensor(lengths, dtype=torch.long)

        return padded_inputs, padded_targets, lengths, list(names)

    def train_dataloader(self):
        train_sequences, train_filenames = self._load_data(self.train_list)
        train_dataset = SequenceDataset(train_sequences, train_filenames)
        return torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=self._collate_fn,
                                           batch_size = self.train_batch_size)

    def val_dataloader(self):
        val_sequences, val_filenames = self._load_data(self.val_list)
        val_dataset = SequenceDataset(val_sequences, val_filenames)
        return torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=self._collate_fn,
                                           batch_size = 32)

    def test_dataloader(self):
        test_sequences, test_filenames = self._load_data(self.test_list)
        test_dataset = SequenceDataset(test_sequences, test_filenames)
        return torch.utils.data.DataLoader(test_dataset, shuffle=False, collate_fn=self._collate_fn,
                                           batch_size = 32)

    def predict_dataloader(self):
        test_sequences, test_filenames = self._load_data(self.test_list)
        test_dataset = SequenceDataset(test_sequences, test_filenames)
        return torch.utils.data.DataLoader(test_dataset, shuffle=False, collate_fn=self._collate_fn,
                                           batch_size=32)


class LightningModule(pl.LightningModule):
    def __init__(self, config):  # hidden_size, num_layers, lr):
        super().__init__()
        self.net = WalrLSTM(hidden_size=config.hidden_size, num_layers = config.num_layers)
        self.lr = config.lr
        self.huber_delta = config.huber_delta if hasattr(config, 'huber_delta') else 1.0

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

    def masked_mse(self, pred, target, lengths):
        # pred, target: [B, T]
        B, T = target.shape
        mask = (torch.arange(T, device=lengths.device)[None, :] < lengths[:, None])
        diff2 = (pred - target) ** 2
        return (diff2 * mask).sum() / mask.sum().clamp_min(1)

    def masked_huber_loss(self, pred, target, lengths, delta=1.0):
        """
        pred, target: [B, T]
        lengths: [B]
        """
        B, T = pred.shape
        device = pred.device

        # mask: [B, T]
        mask = torch.arange(T, device=device)[None, :] < lengths[:, None]

        # elementwise huber
        diff = pred - target
        abs_diff = diff.abs()

        quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=device))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear

        # apply mask and normalize
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def _percent_within(self, pred, target, tolerance=0.10):
        error = torch.abs(pred - target) / torch.clamp(torch.abs(target), min=1e-9)
        return torch.mean((error <= tolerance).float())

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input, target, length, names = batch
        output = self.net(input, length)
        # loss = F.mse_loss(output[:,:,0], target)
        pred = output[:, :, 0]
        # loss = self.masked_mse(pred, target, length)
        loss = self.masked_huber_loss(pred, target, length, delta=self.huber_delta)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size = int(length.sum().item()))
        return loss

    def validation_step(self, batch, batch_idx):
        input, target, length, names = batch
        output = self.net(input, length)
        # loss = F.mse_loss(output[:,:,0], target)
        pred = output[:, :, 0]
        # loss = self.masked_mse(pred, target, length)
        loss = self.masked_huber_loss(pred, target, length, delta=self.huber_delta)
        r2_score = self._r2_score(output[:, :, 0], target)
        rmse_accuracy = self._rmse_accuracy(output[:, :, 0], target)
        percent_within = self._percent_within(output[:, :, 0], target)
        self.log_dict({
            "validate/loss": loss,
            "validate/r2-score": r2_score,
            "validate/rmse-accuracy": rmse_accuracy,
            "validate/percent-within": percent_within},
            on_step=False, on_epoch=True, batch_size = int(length.sum().item()))
        # self.log('validation-loss', loss)
        # print(f'Validation Loss: {loss.item()}')
        # import ipdb; ipdb.set_trace()

        return loss

    def test_step(self, batch, batch_idx):
        input, target, length, names = batch
        output = self.net(input, length)
        # loss = F.mse_loss(output[:,:,0], target)
        pred = output[:, :, 0]
        # loss = self.masked_mse(pred, target, length)
        loss = self.masked_huber_loss(pred, target, length, delta=self.huber_delta)
        r2_score = self._r2_score(output[:, :, 0], target)
        rmse_accuracy = self._rmse_accuracy(output[:, :, 0], target)
        percent_within = self._percent_within(output[:, :, 0], target)
        self.log_dict({
            "test/loss": loss,
            "test/r2-score": r2_score,
            "test/rmse-accuracy": rmse_accuracy,
            "test/percent-within": percent_within},
            on_step=False, on_epoch=True, batch_size = int(length.sum().item()))
        # self.log('test-loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        input, target, length, names = batch
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

