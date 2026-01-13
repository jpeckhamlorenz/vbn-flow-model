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

from models.traj_WALR import LightningModule, DataModule, DictToObject, get_best_run

import re
from collections import defaultdict
from pathlib import Path

# %%

# entity = 'jplorenz-university-of-michigan'
# project = 'VBN-modeling'
# sweep_id = '6ksey94e'


#%%

# sweep_list= wandb.Api().project(project).sweeps()
# sweep_id = sweep_list[1].id

# api = wandb.Api()
# sweep = api.sweep(entity + '/' + project + '/' + sweep_id)
#
# best_run = sweep.best_run()

# run_id = best_run.id
# run_name = best_run.name
#
# print("Best Run Name:", run_name)
# print("Best Run ID:", run_id)
# print(best_run.config)



# %%

# class DictToObject:
#     def __init__(self, dictionary):
#         for key, value in dictionary.items():
#             setattr(self, key, value)

# run_config = DictToObject(best_run.config)
#
# checkpoint_id = os.listdir(
#     os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints')
#     )[0] # Assuming only one checkpoint file exists
#
# model = LightningModule.load_from_checkpoint(
#     os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints', checkpoint_id),
#     config = run_config)

# trainer = pl.Trainer()
# trainer.test(model)

#%%
def test_model(run_id, run_config,
               project: str = 'VBN-modeling',
               group_name: str = None,
               description: str = None):

    checkpoint_id = os.listdir(
        os.path.join(PROJECT_PATH, project, run_id, 'checkpoints')
    )[0]  # Assuming only one checkpoint file exists

    model = LightningModule.load_from_checkpoint(
        os.path.join(PROJECT_PATH, project, run_id, 'checkpoints', checkpoint_id),
        config=run_config)


    data = DataModule(run_config)


    trainer = pl.Trainer(accelerator='mps', devices=1, max_epochs=70, log_every_n_steps=1,
                         default_root_dir="./lightning-test")
    #     wandb.require(experiment="service")
    trainer.test(model, datamodule = data)
    prediction = trainer.predict(model, datamodule = data)

    return trainer, prediction

def plot_tested_model(inputs, outputs):
    fig_flows = {}
    fig_residuals = {}
    fig_error = {}

    leg_flows = {}
    leg_residuals = {}
    leg_error = {}

    for idx, (input, output_packed) in enumerate(zip(inputs, outputs)):
        time = input[0][:, 0].cpu().detach().numpy()
        command = input[0][:, 1].cpu().detach().numpy()
        bead = input[0][:, 2].cpu().detach().numpy()
        analytical = input[0][:, 3].cpu().detach().numpy()
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
        plt.plot(time, analytical + target,
                 color='red', linewidth=2, linestyle='-',
                 label='Simulation Prediction')
        # plt.plot(time, analytical + output,
        #          color='blue', linewidth=2, linestyle='-',
        #          label='Analytical+LSTM Model')
        plt.plot(time, analytical,
                 color='green', linewidth=2, linestyle='-',
                 label='Analytical Model')

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        leg_flows[idx] = plt.figure("flowrate legend")
        leg_flows[idx].legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])

        # %% figure:  residuals in time-series

        fig_residuals[idx] = plt.figure('residual prediction' + str(idx))
        ax = fig_residuals[idx].add_subplot(1, 1, 1)

        plt.xlabel('Time [s]', fontdict=font)
        plt.ylabel('Flow Prediction Error [m3/s]', fontdict=font)

        # plt.xlim(-1,31)
        # plt.ylim(0,500)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(time, output,
                 color='blue', linewidth=2, linestyle='-',
                 label='LSTM Output')
        plt.plot(time, target,
                 color='red', linewidth=2, linestyle='-',
                 label='Simulation Output')

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        leg_residuals[idx] = plt.figure("residual legend")
        leg_residuals[idx].legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])

        # %% error histograms

        fig_error[idx] = plt.figure('error histogram' + str(idx))
        ax = fig_error[idx].add_subplot(1, 1, 1)

        plt.xlabel('Flow Prediction Error [mm3/s]', fontdict=font)
        plt.ylabel('Frequency', fontdict=font)
        plt.title('Error Histogram', fontdict=font)
        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)
        # Plotting the histograms
        ax.hist((target) * 1e9, bins=40, alpha=0.5, color='red', label='Analytical Error',
                density=True)
        ax.hist((target - output) * 1e9, bins=40, alpha=0.5, color='blue',
                label='Analytical+LSTM Error', density=True)

        plt.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero Error')

        # ax.legend(loc='upper right', fontsize='small')

        leg_error[idx] = plt.figure("error legend")
        leg_error[idx].legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])

        return fig_flows, fig_residuals, fig_error, leg_flows, leg_residuals, leg_error

def save_tested_model_plots(fig_flows, fig_residuals, fig_error, leg_flows, leg_residuals, leg_error):

    for idx, flow in enumerate(fig_flows.values()):

        fig_flows[idx].savefig('/Users/james/Desktop/WALR_flows_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
        fig_residuals[idx].savefig('/Users/james/Desktop/residuals_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
        fig_error[idx].savefig('/Users/james/Desktop/error_' + str(idx) + '.png',bbox_inches='tight',dpi=600)

        leg_flows[idx].savefig('/Users/james/Desktop/WALR_flows_leg_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
        leg_residuals[idx].savefig('/Users/james/Desktop/residuals_leg_' + str(idx) + '.png',bbox_inches='tight',dpi=600)
        leg_error[idx].savefig('/Users/james/Desktop/error_leg_' + str(idx) + '.png',bbox_inches='tight',dpi=600)

WINDOW_RE = re.compile(r"_window_(\d+)\.npz")

def parse_window_idx(name: str) -> int:
    m = WINDOW_RE.search(name)
    if m is None:
        raise ValueError(f"Could not parse window index from {name}")
    return int(m.group(1))


def stitch_parent(parent_name, data_dir, pred_map, scale=1e9):
    files = sorted(
        [f for f in pred_map if f.startswith(parent_name)],
        key=parse_window_idx
    )

    # Load first file to get window length + dt
    ex = np.load(data_dir / files[0])
    T = len(ex["time"])
    dt = np.median(np.diff(ex["time"]))

    starts = [parse_window_idx(f) for f in files]
    total_len = max(s + T for s in starts)

    acc = defaultdict(lambda: np.zeros(total_len))
    count = np.zeros(total_len)

    for fname in files:
        s = parse_window_idx(fname)
        d = np.load(data_dir / fname)

        q_vbn = d["Q_vbn"]
        q_tru = d["Q_tru"]
        q_com = d["Q_com"]
        w_com = d["W_com"]
        pred_res = pred_map[fname]

        pred_flow = q_vbn + pred_res  # residual model

        sl = slice(s, s + T)
        acc["pred"][sl] += pred_flow
        acc["tru"][sl]  += q_tru
        acc["vbn"][sl]  += q_vbn
        acc["com"][sl]  += q_com
        acc["bead"][sl] += w_com
        count[sl] += 1

    valid = count > 0
    for k in acc:
        acc[k][valid] /= count[valid]
        acc[k][~valid] = np.nan

    time = np.arange(total_len) * dt

    return time, acc

def plot_stitched(parent, data_dir, pred_map):
    t, acc = stitch_parent(parent, data_dir, pred_map)

    plt.figure(figsize=(12, 5))
    plt.plot(t, acc["tru"] / 1e0, label="Truth")
    plt.plot(t, acc["vbn"] / 1e0, "--", label="Analytical")
    plt.plot(t, acc["pred"] / 1e9 + acc["vbn"], label="Predicted")
    plt.plot(t, acc["com"] / 1e0, alpha=0.6, label="Command")
    plt.grid(True)
    plt.legend()
    plt.title(parent)
    plt.xlabel("Time [s]")
    plt.ylabel("Flow rate")
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(t, acc["bead"], label="Bead width")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.show()

if __name__ == '__main__':

    # %% test best model from sweep

    # run_id, run_config = get_best_run(sweep_id='sgbgw2hc')
    #
    # trainer, prediction = test_model(run_id, run_config)
    #
    # inputs = trainer.predict_dataloaders.dataset
    # outputs = prediction[0].cpu().detach().numpy()[:,:,0]
    #
    # fig_flows, fig_residuals, fig_error, leg_flows, leg_residuals, leg_error = plot_tested_model(inputs, outputs)
    #
    # # save_tested_model_plots(fig_flows, fig_residuals, fig_error, leg_flows, leg_residuals, leg_error)

    # %% test local model\
    ckpt_path = Path("traj_WALR-test/lightning_logs/version_1/checkpoints/epoch=159-step=4960.ckpt")

    # If you need the config explicitly
    run_config = {
        "lr": 0.001,
        "num_layers": 2,
        "hidden_size": 128,
        "batch_size": 64,
    }

    config = DictToObject(run_config)
    data = DataModule(config, data_folderpath=Path('./dataset/recursive_samples'))
    # Load model
    module = LightningModule.load_from_checkpoint(
        ckpt_path,
        config=config,  # <-- must match __init__ signature
    )

    pred_map = {}  # filename -> predicted residual [T]

    module.eval()
    with torch.no_grad():
        for batch in data.predict_dataloader():
            inputs, targets, lengths = batch[:3]
            filenames = batch[3]

            inputs = inputs.to(module.device)
            lengths = lengths.to(module.device)

            out = module.net(inputs, lengths)[:, :, 0]  # [B, T]
            out = out.cpu().numpy()

            for fname, pred in zip(filenames, out):
                pred_map[fname] = pred

    DATA_DIR = Path("./dataset/recursive_samples")

    plot_stitched(
        parent="flowrate_pattern_03_averaged_smoothed",
        data_dir=DATA_DIR,
        pred_map=pred_map,
    )


    # %%
    print("done")







