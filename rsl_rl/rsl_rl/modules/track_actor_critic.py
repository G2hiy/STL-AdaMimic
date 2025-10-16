# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from rsl_rl.modules.estimator import Estimator
from torch import distributions as pyd
import math
import copy


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)

        return x

class TrackActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,
                num_actor_obs,
                num_critic_obs,
                num_one_step_obs,
                actor_history_length,
                num_actor_perception,
                num_critic_perception,
                num_critics,
                num_actions=19,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
                init_noise_std=1.0,
                infer_keyframe_time=None,
                actor_time_scale_range=None,
                fixed_dt=None,
                actor_time_init_noise_std=0.02,
                random=False,
                **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(TrackActorCritic, self).__init__()

        activation = get_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs

        self.actor_history_length = actor_history_length

        self.num_actor_perception = num_actor_perception
        self.num_critic_perception = num_critic_perception
        
        self.num_actions = num_actions

        self.dynamic_latent_dim = 32
        self.terrain_latent_dim = 32

        mlp_input_dim_a = num_one_step_obs * actor_history_length # + self.dynamic_latent_dim
        
        self.num_actor_input  = mlp_input_dim_a 

        mlp_input_dim_c = num_critic_obs

        mlp_input_dim_e = num_one_step_obs * actor_history_length

        self.infer_keyframe_time = infer_keyframe_time
        self.actor_time_scale_low = actor_time_scale_range[0]
        self.actor_time_scale_high = actor_time_scale_range[1]
        self.actor_time_scale = actor_time_scale_range[1] - actor_time_scale_range[0]

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a + 1, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                if self.infer_keyframe_time:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions - 1))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        if self.infer_keyframe_time:
            actor_layers = []
            actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], 1))
                    actor_layers.append(nn.Sigmoid())
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor_time = nn.Sequential(*actor_layers)

        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            critic_layers = []
            critic_layers.append(nn.Linear(mlp_input_dim_c + 1, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critics.append(nn.Sequential(*critic_layers))

        if self.infer_keyframe_time:
            self.critics_time = nn.ModuleList()
            for _ in range(num_critics):
                critic_layers = []
                critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                critic_layers.append(activation)
                for l in range(len(critic_hidden_dims)):
                    if l == len(critic_hidden_dims) - 1:
                        critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                    else:
                        critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                        critic_layers.append(activation)
                self.critics_time.append(nn.Sequential(*critic_layers))

        self.num_critics = num_critics

        # Action noise
        if self.infer_keyframe_time:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions-1))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        if self.infer_keyframe_time:
            # self.std_time = nn.Parameter(actor_time_init_noise_std * torch.ones(1))
            self.distribution_time = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        self.fixed_dt = fixed_dt
        self.name = 'continuous'

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return torch.cat([self.distribution.mean, self.distribution_time.mean], dim=-1)

    @property
    def action_std(self):
        return torch.cat([self.distribution.stddev, self.distribution_time.stddev], dim=-1)
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution_high(self, obs_history):
        actor_time_input = obs_history.clone()
        action_mean = self.actor_time(actor_time_input) 
        delta_time = action_mean * self.actor_time_scale + self.actor_time_scale_low 
        self.distribution_time = Normal(delta_time + self.fixed_dt, action_mean*0. + max(self.actor_time_scale / 2, 0.005))

    def update_distribution_low(self, obs_history, action_time):
        actor_input = torch.cat([obs_history, action_time.detach()], dim=-1).clone()
        action_mean_fixed = self.actor(actor_input)
        action_mean = action_mean_fixed
        self.distribution = Normal(action_mean, action_mean*0. + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution_high(obs_history)
        action_time_sample = self.distribution_time.sample().clamp(self.actor_time_scale_low + self.fixed_dt, self.actor_time_scale_high + self.fixed_dt)
        self.update_distribution_low(obs_history, action_time_sample)
        action_sample = self.distribution.sample()
        return torch.cat([action_sample, action_time_sample], dim=-1)
    
    def get_actions_log_prob(self, actions):
        return (self.distribution.log_prob(actions[:, :-1]).sum(dim=-1), self.distribution_time.log_prob(actions[:, -1:]).sum(dim=-1))

    def act_inference(self, obs_history, observations=None):
        # with torch.no_grad():
        actor_input = obs_history.clone()
        delta_time = self.actor_time(actor_input) * self.actor_time_scale + self.actor_time_scale_low
        action_time_mean = delta_time + self.fixed_dt

        actor_input = torch.cat([obs_history, action_time_mean], dim=-1)
        action_mean_fixed = self.actor(actor_input)
        action_mean = action_mean_fixed
        return torch.cat([action_mean, action_time_mean], dim=-1)

    def evaluate_high(self, critic_observations, **kwargs):
        values_high = torch.concat([critic(critic_observations) for critic in self.critics_time], dim=-1)
        return values_high

    def evaluate_low(self, critic_observations, **kwargs):
        values_low = torch.concat([critic(critic_observations) for critic in self.critics], dim=-1)
        return values_low
    

class TrackActorCriticDelta(nn.Module):
    is_recurrent = False
    def __init__(self,
                num_actor_obs,
                num_critic_obs,
                num_one_step_obs,
                actor_history_length,
                num_actor_perception,
                num_critic_perception,
                num_critics,
                num_actions=19,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
                init_noise_std=1.0,
                infer_keyframe_time=None,
                actor_time_scale_range=None,
                fixed_dt=None,
                random=False,
                resume=True,
                checkpoint_path=None,
                freeze=True,
                threshold=0.,
                **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(TrackActorCriticDelta, self).__init__()

        activation = get_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs

        self.actor_history_length = actor_history_length

        self.num_actor_perception = num_actor_perception
        self.num_critic_perception = num_critic_perception
        
        self.num_actions = num_actions

        self.dynamic_latent_dim = 32
        self.terrain_latent_dim = 32

        mlp_input_dim_a = num_one_step_obs * actor_history_length # + self.dynamic_latent_dim
        
        self.num_actor_input  = mlp_input_dim_a 

        mlp_input_dim_c = num_critic_obs

        mlp_input_dim_e = num_one_step_obs * actor_history_length

        self.infer_keyframe_time = infer_keyframe_time
        self.actor_time_scale_low = actor_time_scale_range[0]
        self.actor_time_scale_high = actor_time_scale_range[1]
        self.actor_time_scale = actor_time_scale_range[1] - actor_time_scale_range[0]

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a + 1, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                if self.infer_keyframe_time:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions - 1))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        self.actor_delta = copy.deepcopy(self.actor)

        if self.infer_keyframe_time:
            actor_layers = []
            actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], 1))
                    actor_layers.append(nn.Sigmoid())
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor_time = nn.Sequential(*actor_layers)

        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            critic_layers = []
            critic_layers.append(nn.Linear(mlp_input_dim_c + 1, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critics.append(nn.Sequential(*critic_layers))
        self.critics_delta = copy.deepcopy(self.critics)

        if self.infer_keyframe_time:
            self.critics_time = nn.ModuleList()
            for _ in range(num_critics):
                critic_layers = []
                critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                critic_layers.append(activation)
                for l in range(len(critic_hidden_dims)):
                    if l == len(critic_hidden_dims) - 1:
                        critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                    else:
                        critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                        critic_layers.append(activation)
                self.critics_time.append(nn.Sequential(*critic_layers))

        self.num_critics = num_critics

        # Action noise
        if self.infer_keyframe_time:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions-1))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        if self.infer_keyframe_time:
            # self.std_time = nn.Parameter(actor_time_init_noise_std * torch.ones(1))
            self.distribution_time = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        self.fixed_dt = fixed_dt
        self.random = random
        self.name = 'delta'
        self.threshold = threshold
        self.freeze = freeze
        if resume:
            assert checkpoint_path is not None, "Checkpoint path must be provided for resuming."
            loaded_dict = torch.load(checkpoint_path, map_location=self.std.device, weights_only=True)['model_state_dict']
            model_state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in loaded_dict.items() 
                                if (k.startswith('actor.') or k.startswith('critics.'))}
            model_state_dict.update(filtered_state_dict)
            self.load_state_dict(model_state_dict, strict=False)

            print(f"Loaded model from {checkpoint_path} with {len(filtered_state_dict)} parameters and keys {filtered_state_dict.keys()}.")

            if self.freeze:
                for param in self.actor.parameters():
                    param.requires_grad = False
                for param in self.critics.parameters():
                    param.requires_grad = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return torch.cat([self.distribution.mean, self.distribution_time.mean], dim=-1)

    @property
    def action_std(self):
        return torch.cat([self.distribution.stddev, self.distribution_time.stddev], dim=-1)
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution_high(self, obs_history):
        actor_time_input = obs_history.clone()
        action_mean = self.actor_time(actor_time_input) 
        delta_time = action_mean * self.actor_time_scale + self.actor_time_scale_low 
        self.distribution_time = Normal(delta_time + self.fixed_dt, action_mean*0. + max(self.actor_time_scale / 2, 0.005))

    def update_distribution_low(self, obs_history, action_time):
        actor_input = torch.cat([obs_history, action_time.detach()], dim=-1).clone()
        action_mean_fixed = self.actor(actor_input)
        dt = (action_time - self.fixed_dt)
        mask = (dt > -self.threshold) & (dt < self.threshold)
        dt[mask] = 0.0
        action_mean_delta = self.actor_delta(actor_input) * dt #* 1000
        action_mean = action_mean_fixed +  action_mean_delta
        self.distribution = Normal(action_mean, action_mean*0. + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution_high(obs_history)
        action_time_sample = self.distribution_time.sample().clamp(self.actor_time_scale_low + self.fixed_dt, self.actor_time_scale_high + self.fixed_dt)
        self.update_distribution_low(obs_history, action_time_sample)
        action_sample = self.distribution.sample()
        return torch.cat([action_sample, action_time_sample], dim=-1)
    
    def get_actions_log_prob(self, actions):
        return (self.distribution.log_prob(actions[:, :-1]).sum(dim=-1), self.distribution_time.log_prob(actions[:, -1:]).sum(dim=-1))

    def act_inference(self, obs_history):
        # with torch.no_grad():
        actor_input = obs_history.clone()
        delta_time = self.actor_time(actor_input) * self.actor_time_scale + self.actor_time_scale_low
        action_time_mean = delta_time + self.fixed_dt

        actor_input = torch.cat([obs_history, action_time_mean], dim=-1)
        action_mean_fixed = self.actor(actor_input)
        dt = (action_time_mean - self.fixed_dt)
        mask = (dt > -self.threshold) & (dt < self.threshold)
        dt[mask] = 0.0
        action_mean_delta = self.actor_delta(actor_input) * dt #* 1000
        action_mean = action_mean_fixed + action_mean_delta
        return torch.cat([action_mean, action_time_mean], dim=-1)

    def evaluate_high(self, critic_observations, **kwargs):
        values_high = torch.concat([critic(critic_observations) for critic in self.critics_time], dim=-1)
        return values_high

    def evaluate_low(self, critic_observations, **kwargs):
        values_low = torch.concat([critic(critic_observations) for critic in self.critics_delta], dim=-1) #* (critic_observations[:, -1:] - self.fixed_dt)
        return values_low 