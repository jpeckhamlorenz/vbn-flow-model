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

def train_model(run_config):

    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    #     os.environ["WANDB_START_METHOD"] = "thread"'
    # wandb.run = None  # Reset wandb run to avoid conflicts

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