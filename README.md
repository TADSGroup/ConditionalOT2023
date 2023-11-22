# Wasserstein Monge GAN

Pytorch implementation of [Conditional Optimal Transport on Function Spaces](https://arxiv.org/abs/2311.05672) by Hosseini, Hsu and Taghvaei.

Most of this is code is for the Wasserstein Monge GAN, but we've also included the code for pCN that we used for validation, and for the low dimensional empirical pushforward experiments.

## Conda Environments
Since pytorch and fenicsx wouldn't place nicely with each other, we have separate conda environments for solving pde's and training the WaMGANS, they can both be found in the conda_envs directory. 

## Building dataset
We build the dataset using Fenicsx. Using the conda environment defined in conda_envs/fenics_env.yml, you can use 
```
python build_dataset.py
```
after modifying configuration settings to your liking


## Usage

See the file darcy_train.py for an example of the training process. We used wandb to log everything throughout training, and have provided the conda environment defined in conda_envs/mgan_env.yml for training and using the WaMGAN.

These generative models can take a long time to converge. We ran for 100,000 epochs, but much fewer will still produce satisfactory results for this Darcy flow example.

## This code was forked from
* https://github.com/EmilienDupont/wgan-gp

with significant modifications for our use case, and is licensed as such. 
