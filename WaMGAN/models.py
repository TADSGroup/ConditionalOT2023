import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(
            self, 
            condition_dim,
            sample_dim, 
            hidden_layer_sizes = [64,128,64],
            ):
        super(Generator, self).__init__()

        self.sample_dim = sample_dim
        self.condition_dim = condition_dim

        layer_dimensions = [condition_dim+sample_dim] + hidden_layer_sizes + [sample_dim]
        layers = [self.make_gen_block(layer_dimensions[i],layer_dimensions[i+1]) for i in range(len(layer_dimensions)-2)]
        layers = layers + [self.make_gen_block(layer_dimensions[-2],layer_dimensions[-1],final_layer = True)]

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, input_data_combined):
        """Apply network to data"""
        return self.network(input_data_combined)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.sample_dim))
    
    def sample_concat_latent(self,data_cond):
        return torch.cat(
            (
                data_cond,
                self.sample_latent(data_cond.shape[0]).to(data_cond.device)
                )
                ,
                axis=1
                )
    
    def make_gen_block(self, input_dim, output_dim, final_layer=False):
        '''
        Parameters:
            input_dim: dimension of input to layer
            output_dim: dimension of output of layer
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim,bias = True),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim,bias = True),
            )

class Discriminator(nn.Module):
    def __init__(
            self, 
            condition_dim,
            sample_dim, 
            hidden_layer_sizes = [64,128,64],
            ):
        super(Discriminator, self).__init__()

        self.sample_dim = sample_dim
        self.condition_dim = condition_dim

        layer_dimensions = [condition_dim+sample_dim] + hidden_layer_sizes + [1]
        layers = [self.make_discriminator_block(layer_dimensions[i],layer_dimensions[i+1]) for i in range(len(layer_dimensions)-2)]
        layers = layers + [self.make_discriminator_block(layer_dimensions[-2],layer_dimensions[-1],final_layer = True)]

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, input_data):
        return self.network(input_data)
        
    def make_discriminator_block(self, input_dim, output_dim, final_layer=False):
        '''
        Parameters:
            input_dim: dimension of input to layer
            output_dim: dimension of output of layer
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim,bias = True),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim,bias = True),
            )