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

from traj_correction_LSTM_sweep import LightningModule

# %%

entity = 'jplorenz-university-of-michigan'
project = 'VBN-modeling'
sweep_id = '6ksey94e'


#%%

# sweep_list= wandb.Api().project(project).sweeps()
# sweep_id = sweep_list[1].id

api = wandb.Api()
sweep = api.sweep(entity + '/' + project + '/' + sweep_id)

best_run = sweep.best_run()

# run_id = best_run.id
# run_name = best_run.name
#
# print("Best Run Name:", run_name)
# print("Best Run ID:", run_id)
# print(best_run.config)



# %%

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

checkpoint_id = os.listdir(
    os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints')
    )[0] # Assuming only one checkpoint file exists

model = LightningModule.load_from_checkpoint(
    os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints', checkpoint_id),
    config = DictToObject(best_run.config))

trainer = pl.Trainer()
trainer.test(model)

#%%
#todo: just pasted this in, this is where i left off, change it to test rather than train
def train_model(group_name: str = None,
                description: str = None):
    #     os.environ["WANDB_START_METHOD"] = "thread"'
    wandb.init(project="VBN-modeling",
               notes = description,
               group = group_name)
    config = wandb.config
    wandb_logger = WandbLogger()
    data = DataModule(config)
    module = LightningModule(config)

    wandb_logger.watch(module.net)

    trainer = pl.Trainer(accelerator='mps', devices=1, max_epochs=70, log_every_n_steps=1,
                         default_root_dir="./lightning-test", logger=wandb_logger)
    #     wandb.require(experiment="service")
    trainer.fit(module, data)