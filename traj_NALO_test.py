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

plt.close('all')

from traj_NALO import LightningModule, DataModule, get_best_run


#%%

def test_model(run_id, run_config,
               project: str = 'VBN-modeling',
               group_name: str = None,
               description: str = None):
    #     os.environ["WANDB_START_METHOD"] = "thread"'
    # wandb.init(project="VBN-modeling",
    #            notes = description,
    #            group = group_name)
    # config = wandb.config
    # wandb_logger = WandbLogger()

    # run_config = DictToObject(run.config)
    checkpoint_id = os.listdir(
        os.path.join(PROJECT_PATH, project, run_id, 'checkpoints')
    )[0]  # Assuming only one checkpoint file exists

    model = LightningModule.load_from_checkpoint(
        os.path.join(PROJECT_PATH, project, run_id, 'checkpoints', checkpoint_id),
        config=run_config)


    data = DataModule(run_config)
    # module = LightningModule(config)

    # model = LightningModule.load_from_checkpoint(
    #     os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints', checkpoint_id),
    #     config=DictToObject(best_run.config))

    # wandb_logger.watch(model.net)

    trainer = pl.Trainer(accelerator='mps', devices=1, max_epochs=70, log_every_n_steps=1,
                         default_root_dir="./lightning-test")
    #     wandb.require(experiment="service")
    trainer.test(model, datamodule = data)
    prediction = trainer.predict(model, datamodule = data)

    return trainer, prediction

if __name__ == '__main__':


    run_id, run_config = get_best_run()

    trainer, prediction = test_model(run_id, run_config)

    inputs = trainer.predict_dataloaders.dataset
    outputs = prediction[0].cpu().detach().numpy()[:,:,0]

    fig_flows = {}
    fig_residuals = {}
    fig_error = {}

    leg_flows = {}
    leg_residuals = {}
    leg_error = {}

    for idx, (input, output_packed) in enumerate(zip(inputs, outputs)):

        time = input[0][:,0].cpu().detach().numpy()
        command = input[0][:,1].cpu().detach().numpy()
        bead = input[0][:,2].cpu().detach().numpy()
        target = input[1].cpu().detach().numpy()
        output = output_packed[:len(time)]


        # %% figure:  flowrate prediction

        fig_flows[idx] = plt.figure('flowrate prediction' + str(idx))
        ax = fig_flows[idx].add_subplot(1, 1, 1)

        plt.xlabel('Time [s]', fontdict=font)
        plt.ylabel('Output Flowrate [m3/s]', fontdict=font)

        # plt.xlim(-1,31)
        # plt.ylim(0,500)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(time, command,
                 color='black', linewidth=2, linestyle='--',
                 label='Commanded Flowrate')
        plt.plot(time, target,
                 color='red', linewidth=2, linestyle='-',
                 label='Simulation Prediction')
        plt.plot(time, output,
                 color='blue', linewidth=2, linestyle='-',
                 label='Analytical+LSTM Model')


        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        leg_flows[idx] = plt.figure("flowrate legend")
        leg_flows[idx].legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])

    #     # %% figure:  residuals in time-series
    #
    #     fig_residuals[idx] = plt.figure('residual prediction' + str(idx))
    #     ax = fig_residuals[idx].add_subplot(1, 1, 1)
    #
    #     plt.xlabel('Time [s]', fontdict=font)
    #     plt.ylabel('Flow Prediction Error [m3/s]', fontdict=font)
    #
    #     # plt.xlim(-1,31)
    #     # plt.ylim(0,500)
    #
    #     plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)
    #
    #     plt.xscale('linear')
    #     plt.yscale('linear')
    #
    #     plt.plot(time, output,
    #              color='blue', linewidth=2, linestyle='-',
    #              label='LSTM Output')
    #     plt.plot(time, target,
    #              color='red', linewidth=2, linestyle='-',
    #              label='Simulation Output')
    #
    #     # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
    #     # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
    #
    #     # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
    #     # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)
    #
    #     # ax.legend()
    #
    #     leg_residuals[idx] = plt.figure("residual legend")
    #     leg_residuals[idx].legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])
    #
    #     # %% error histograms
    #
    #     fig_error[idx] = plt.figure('error histogram' + str(idx))
    #     ax = fig_error[idx].add_subplot(1, 1, 1)
    #
    #     plt.xlabel('Flow Prediction Error [mm3/s]', fontdict=font)
    #     plt.ylabel('Frequency', fontdict=font)
    #     plt.title('Error Histogram', fontdict=font)
    #     plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)
    #     # Plotting the histograms
    #     ax.hist((target) * 1e9, bins=40, alpha=0.5, color='red', label='Analytical Error',
    #             density=True)
    #     ax.hist((target - output) * 1e9, bins=40, alpha=0.5, color='blue',
    #             label='Analytical+LSTM Error', density=True)
    #
    #     plt.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero Error')
    #
    #     # ax.legend(loc='upper right', fontsize='small')
    #
    #     leg_error[idx] = plt.figure("error legend")
    #     leg_error[idx].legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])
    #
    # # %% save figs
    #
    # # for idx, flow in enumerate(fig_flows.values()):
    # #
    # #     fig_flows[idx].savefig('/Users/james/Desktop/flows_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
    # #     fig_residuals[idx].savefig('/Users/james/Desktop/residuals_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
    # #     fig_error[idx].savefig('/Users/james/Desktop/error_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
    # #
    # #     leg_flows[idx].savefig('/Users/james/Desktop/flows_leg_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
    # #     leg_residuals[idx].savefig('/Users/james/Desktop/residuals_leg_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
    # #     leg_error[idx].savefig('/Users/james/Desktop/error_leg_' + str(idx) + '.png',bbox_inches='tight',dpi=600)





