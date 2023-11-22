import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb


class Trainer():
    def __init__(
            self, 
            generator, 
            discriminator, 
            gen_optimizer, 
            dis_optimizer,
            gp_weight=10, 
            critic_iterations=5, 
            print_every=50,
            device='cuda',
            wandb_run=None,
            monotone_penalty=0.0001,
            penalty_type='hinge',
            wandb_log_every = 5,
            gradient_penalty_type = 'two_sided',
            full_critic_train = 5000,
            ):
        """
        Note that for wandb_log_every, we only log on iterations that we train the generator
        We log to wandb every wandb_log_every generator iterations.
        """
        self.wandb_run = wandb_run
        self.wandb_log_every = wandb_log_every
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.log = {
            'G': [], 
            'D': [], 
            'GP': [], 
            'gradient_norm': [],
            'real_mean_score':[],
            'generated_mean_score':[],
            'transport_penalty':[],
            'monotone_percent':[]
            }
        self.num_steps = 0
        self.device = device
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.monotone_penalty = monotone_penalty
        self.penalty_type = penalty_type
        self.gradient_penalty_type = gradient_penalty_type
        self.full_critic_train = full_critic_train

        self.G.to(self.device)
        self.D.to(self.device)

    def _critic_train_iteration(self, data_cond,data_gen,log=True):
        """
        Data is the data to condition on combined with the output of Generator
        """
        # Get generated data
        batch_size = data_cond.size()[0]
        generated_data = self.sample_generator(data_cond)
        combined=torch.cat((data_cond,data_gen),axis=1)
        generated_data_combined = torch.cat((data_cond,generated_data),axis=1)
        # Calculate scores on real and generated data
        combined = Variable(combined)

        combined = combined.to(self.device)

        d_real = self.D(combined)
        d_generated = self.D(generated_data_combined)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(combined, generated_data_combined,log=log)
        self.log['GP'].append(gradient_penalty.item())
        self.log['real_mean_score'].append(d_real.mean().item())
        self.log['generated_mean_score'].append(d_generated.mean().item())


        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = gradient_penalty + d_generated.mean() - d_real.mean()
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.log['D'].append(d_loss.item())
        if (self.wandb_run is not None) and (log is True):
            self.wandb_run.log({
                'GP':gradient_penalty.item(),
                'real_mean_score':d_real.mean().item(),
                'generated_mean_score':d_generated.mean().item(),
                'discriminator_loss':d_loss.item()
            })


    def _generator_train_iteration(self, data_cond,data_gen,log = True):
        """Trains the generator for one step of gradient descent"""
        
        self.G_opt.zero_grad()
        #We use the sample_generator method defined below
        latent_samples,generated_data = self.sample_generator(data_cond,return_input=True)
        generated_data_combined = torch.cat((data_cond,generated_data),axis=1)

        # Calculate loss and optimize
        d_generated = self.D(generated_data_combined)
        generator_monotonicity = self._generator_monotone_penalty(data_cond,data_gen)
        if self.penalty_type=='hinge':
            transport_penalty = (generator_monotonicity-torch.relu(generator_monotonicity)).mean()
        elif self.penalty_type=='mean':
            transport_penalty =  generator_monotonicity.mean()
        elif self.penalty_type=='monge':
            transport_penalty = -1*torch.mean((latent_samples - generated_data_combined)**2)

        mon_percent = (torch.gt(generator_monotonicity,0).float().mean())
        g_loss = - d_generated.mean() - transport_penalty*self.monotone_penalty
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.log['G'].append(g_loss.item())
        self.log['transport_penalty'].append(transport_penalty.item())
        self.log['monotone_percent'].append(mon_percent.item())

        if (self.wandb_run is not None) and (log is True):
            self.wandb_run.log({
                'generator_loss':g_loss.item(),
                'transport_penalty':transport_penalty.item(),
                'monotone_percent':mon_percent.item(),
            })

    def _generator_monotone_penalty(self,data_cond,data_gen):

        ##Sample two different input sets
        generator_input_1 = self.G.sample_concat_latent(data_cond)
        G_output_cat1 = torch.cat((data_cond,self.G(generator_input_1)),axis=1)

        generator_input_2 = self.G.sample_concat_latent(data_cond)
        G_output_cat2 = torch.cat((data_cond,self.G(generator_input_2)),axis=1)

        monotonicity_scores = torch.sum(
            (G_output_cat1-G_output_cat2) * (generator_input_1-generator_input_2),
            axis=1
            )
        
        return monotonicity_scores

    def _gradient_penalty(self, real_data, generated_data,log=True):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad = True
        interpolated = interpolated.to(self.device)

        # Calculate score of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of scores with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated, 
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.log['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())
        if (self.wandb_run is not None) and (log is True):
            self.wandb_run.log({
                'gradient_norm':gradients.norm(2, dim=1).mean().item()
            })


        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        if self.gradient_penalty_type == 'two_sided':
            # Return gradient penalty
            return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        elif self.gradient_penalty_type == 'one_sided':
            return self.gp_weight * (torch.relu((gradients_norm - 1)) ** 2).mean()

    def _train_epoch(self, data_loader,update_generator = True):
        """
        we only log on iterations which include a generator step
        and only log every self.wandb_log_every generator iterations
        """
        for i, (data_cond,data_gen) in enumerate(data_loader):
            whether_to_log = (self.num_steps%(self.wandb_log_every*self.critic_iterations))==0
            data_cond = data_cond.to(self.device)
            data_gen = data_gen.to(self.device)

            self._critic_train_iteration(data_cond,data_gen,log=whether_to_log)
            # Only update generator every |critic_iterations| iterations
            if (self.num_steps % self.critic_iterations == 0) and (update_generator is True):
                self._generator_train_iteration(data_cond,data_gen,log=whether_to_log)

            self.num_steps += 1

            

    def train(self, data_loader, epochs):
        for epoch in tqdm(range(epochs)):
            if (epoch+1) % self.full_critic_train==0:
                #Every full_critic_train epochs, spent a whole epoch updating critic
                self._train_epoch(data_loader,update_generator=False)
            else:
                self._train_epoch(data_loader,update_generator=True)
            if self.wandb_run is not None:
                self.wandb_run.log({
                    'epoch_number':epoch,
                })
            if epoch % self.print_every == 0 or epoch == epochs-1:
                print("\nEpoch {}".format(epoch))
                print("Discriminator loss: {}".format(self.log['D'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("Generator loss: {}".format(self.log['G'][-1]))
                print("Gradient penalty loss: {}".format(self.log['GP'][-1]))
                print("Gradient norm: {}".format(self.log['gradient_norm'][-1]))
                print(f"{self.penalty_type} score: {self.log['transport_penalty'][-1]}")

    def sample_generator(self, data_cond,return_input=False):
        generator_input = self.G.sample_concat_latent(data_cond)
        generator_input = generator_input.to(self.device)
        generated_data = self.G(generator_input)
        if return_input is True:
            return generator_input,generated_data
        else:
            return generated_data