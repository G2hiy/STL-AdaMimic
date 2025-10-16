import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class Estimator(nn.Module):
    def __init__(self,
                temporal_steps,
                 num_one_step_obs,
                 dis_hidden_dims=[128, 64],
                 latent_dim=32,
                 learning_rate=1e-4,
                 max_grad_norm=1.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(Estimator, self).__init__()

        # dis_activation = get_activation('relu')
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs

        self.num_latent = latent_dim
        self.max_grad_norm = max_grad_norm

        # Encoder
        modules = []
        activation_fn = get_activation('elu')
        encoder_input_dim = self.temporal_steps * self.num_one_step_obs
        encoder_hidden_dims = [256, 128]
        modules.extend(
            [nn.Linear(encoder_input_dim, encoder_hidden_dims[0]),
            activation_fn]
            )
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                modules.append(nn.Linear(encoder_hidden_dims[l],self.num_latent))
            else:
                modules.append(nn.Linear(encoder_hidden_dims[l],encoder_hidden_dims[l + 1]))
                modules.append(activation_fn)
        self.encoder = nn.Sequential(*modules)

        self.vel_mu = nn.Linear(self.num_latent, 7)

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.kl_weight = 1.0

    
    def encode(self,obs_history):
        encoded = self.encoder(obs_history)
        vel_mu = self.vel_mu(encoded)
        return vel_mu

    def forward(self,obs_history):
        vel_mu = self.encode(obs_history)
        return vel_mu
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def sample(self,obs_history):
        estimate_vel = self.forward(obs_history)
        return estimate_vel
    
    def inference(self,obs_history):
        estimate_vel = self.forward(obs_history)
        vel_mu = vel_params
        return vel_mu


    def update(self, obs_history, next_critic_obs, lr=None):

        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        print( self.num_one_step_obs)
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+7].detach()

        estimate_vel = self.forward(obs_history)

        vel_loss = F.mse_loss(estimate_vel, vel,reduction='none').mean(-1)
        vel_loss = torch.where(torch.isnan(vel_loss), torch.ones_like(vel_loss), vel_loss)

        if vel_loss.isnan().any():
            vel_loss = torch.ones((vel_loss.shape[0],1), device=vel.device)

        loss = vel_loss # + dis_loss
        vae_loss = torch.mean(loss)

        return 0.0, vae_loss # , classify_loss.item()


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None