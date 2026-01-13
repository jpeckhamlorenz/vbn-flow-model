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
# from constants.plotting import font

from models.traj_WALR import DataModule, LightningModule

plt.close('all')

#%%

def train_model():

    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    #     os.environ["WANDB_START_METHOD"] = "thread"'
    wandb.run = None  # Reset wandb run to avoid conflicts
    wandb.init(project="VBN-modeling",
               notes = 'Hyperparameter sweep for WALR',
               group = 'WALR')
    config = wandb.config
    wandb_logger = WandbLogger()
    data = DataModule(config)
    module = LightningModule(config)

    wandb_logger.watch(module.net)

    trainer = pl.Trainer(accelerator='mps', devices=1, max_epochs=16, log_every_n_steps=2,
                         default_root_dir="./lightning-test", logger=wandb_logger)
    #     wandb.require(experiment="service")
    trainer.fit(module, data)


if __name__ == '__main__':
    sweep_config = {
        'description': 'With Analytical, Learn Residual',
        'method': 'bayes',
        'name': 'WALR-exp-test',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'parameters': {
            'hidden_size': {'values': [64,96,128,256,512]},
            'num_layers': {'values': [2,3,4,5,6]},
            'lr': {'max': 0.01, 'min': 0.0001},
            'batch_size': {'values': [1,2,3,4]},
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3
        },
        'run_cap': 40,
    }

    sweep_id = wandb.sweep(sweep_config, project="VBN-modeling")
    wandb.agent(sweep_id = sweep_id, count = 40,
                function = train_model)

    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    best_run = sweep.best_run()
    print("Best Run Name:", best_run.name)
    print("Best Run ID:", best_run.id)
    print(best_run.config)









