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

from traj_correction_LSTM_sweep import LightningModule, DataModule

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

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

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
def test_model(run,
               group_name: str = None,
               description: str = None):
    #     os.environ["WANDB_START_METHOD"] = "thread"'
    # wandb.init(project="VBN-modeling",
    #            notes = description,
    #            group = group_name)
    # config = wandb.config
    # wandb_logger = WandbLogger()

    run_config = DictToObject(run.config)
    checkpoint_id = os.listdir(
        os.path.join(PROJECT_PATH, project, run.id, 'checkpoints')
    )[0]  # Assuming only one checkpoint file exists

    model = LightningModule.load_from_checkpoint(
        os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints', checkpoint_id),
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

    entity = 'jplorenz-university-of-michigan'
    project = 'VBN-modeling'
    sweep_id = 'xico4n8f'

    api = wandb.Api()
    best_run = api.sweep(entity + '/' + project + '/' + sweep_id).best_run()


    trainer, prediction = test_model(best_run,
               group_name = sweep_id,
               description = best_run.name)
    # run_config = DictToObject(best_run.config)
    # checkpoint_id = os.listdir(
    #     os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints')
    # )[0]  # Assuming only one checkpoint file exists
    #
    # model = LightningModule.load_from_checkpoint(
    #     os.path.join(PROJECT_PATH, project, best_run.id, 'checkpoints', checkpoint_id),
    #     config=run_config)

    # extract the original data that goes into the prediction
    inputs = trainer.predict_dataloaders.dataset
    outputs = prediction[0].cpu().detach().numpy()[:,:,0]

    for input, output in zip(inputs, outputs):
        time = input[0][:,0].cpu().detach().numpy()
        command = input[0][:,1].cpu().detach().numpy()
        bead = input[0][:,2].cpu().detach().numpy()
        analytical = input[0][:,3].cpu().detach().numpy()
        target = input[1].cpu().detach().numpy()

        # Plot the results


        plt.figure(figsize=(10, 6))
        plt.plot(time, output[:len(time)], label='Predicted Flow Rate')
        plt.plot(time, target[:len(time)], label='Analytical Flow Rate', linestyle='--')
        plt.xlabel('Time (s)', fontsize=font['size'])
        plt.ylabel('Flow Rate (mL/min)', fontsize=font['size'])
        plt.title('WALR Flow Rate Prediction vs Analytical Model', fontsize=font['size'])
        plt.legend(fontsize=font['size'])
        plt.grid(True)
        plt.show()