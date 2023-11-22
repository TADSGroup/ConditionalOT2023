import torch
import torch.optim as optim
import importlib
import training
importlib.reload(training)
from training import Trainer
import models
importlib.reload(models)

import dataloaders
importlib.reload(dataloaders)
from dataloaders import get_darcy_dataloader_transformers

from models import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np
from os import mkdir
import wandb
import argparse
import json

parser = argparse.ArgumentParser(description='Read configuration file')
parser.add_argument('config_file', type=str, help='Path to the configuration file')
args = parser.parse_args()

with open(args.config_file, 'r') as f:
    configs = json.load(f)

#We use wandb to track everything during training
run = wandb.init(
    # set the wandb project where this run will be logged
    project='darcy_monge_penalty',
    tags=configs["tags"],
    # track hyperparameters and run metadata
    config=configs
)


data_loader,pca_X,normalizer_y = get_darcy_dataloader_transformers(
    batch_size=configs['batch_size'],
    latent_dim=configs['latent_dim'],
    num_datapoints=configs['num_datapoints'],
    num_observation_grid=configs['grid_num_observed'],
    noise_level_y_observed=configs['y_noise_level'],
    path_prefix = 'data'
    )

cond,gen=data_loader.dataset.tensors
condition_dim = cond.shape[1]
sample_dim = gen.shape[1]

generator = Generator(
    condition_dim=configs['grid_num_observed']**2,
    sample_dim=configs['latent_dim'],
    hidden_layer_sizes=configs['gen_hidden_layer_sizes']
    )

discriminator = Discriminator(
    condition_dim=configs['grid_num_observed']**2,
    sample_dim=configs['latent_dim'],
    hidden_layer_sizes=configs['discriminator_hidden_layer_sizes']
    )

# Initialize optimizers

G_optimizer = optim.SGD(generator.parameters(), lr=configs['generator_learning_rate'],momentum = configs['momentum'])
D_optimizer = optim.SGD(discriminator.parameters(), lr=configs['critic_learning_rate'],momentum = configs['momentum'])

# Train model
epochs = configs['num_epochs']
trainer = Trainer(
    generator, 
    discriminator, 
    G_optimizer, 
    D_optimizer,
    device=configs['device'],
    print_every=500,
    gp_weight=configs['gp_weight'],
    wandb_run=run,
    monotone_penalty = configs['monotone_penalty'],
    penalty_type=configs['penalty_type'],
    gradient_penalty_type=configs['gradient_penalty_type'],
    full_critic_train = configs['full_critic_train'],
    )

trainer.train(data_loader, epochs)
save_dir='trained_models/'+run.name
mkdir(save_dir)

generator_path=save_dir+'/generator.pkl'
discriminator_path = save_dir+'/discriminator.pkl'
torch.save(generator,generator_path)
torch.save(discriminator,discriminator_path)

wandb.finish()
