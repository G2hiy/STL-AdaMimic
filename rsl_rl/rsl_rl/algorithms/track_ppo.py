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

import torch
import torch.nn as nn
import torch.optim as optim
from muon import SingleDeviceMuonWithAuxAdam

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import HIMRolloutStorage, TrackRolloutStorage

class TrackPPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 reward_group_weights,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                value_smoothness_coef=0.1,
                smoothness_upper_bound=1.0,
                smoothness_lower_bound=0.1,
                 amp=None,
                 amp_normalizer=None,
                 motion_buffer=None,
                 use_flip=True,
                 amp_critic=None,
                 infer_keyframe_time=True,
                 use_smooth=True,
                 train_high=False,
                 use_timeout=True,
                 freeze=True,
                 ):

        self.device = device
        self.use_flip = use_flip

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = torch.tensor(gamma, device=self.device)
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.reward_group_weights = reward_group_weights

        self.use_smooth = use_smooth
        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound
        self.infer_keyframe_time = infer_keyframe_time
        self.use_timeout = use_timeout

        self.motion_buffer = motion_buffer
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor'},
        ]
        self.transition = HIMRolloutStorage.Transition()
        torch.autograd.set_detect_anomaly(True)
        self.update_count = 0

        params = [
            {'params': self.actor_critic.actor.parameters(), 'name': 'actor'},
            {'params': self.actor_critic.critics.parameters(), 'name': 'critics'},
            {'params': [self.actor_critic.std], 'name': 'std'},
        ]
        if hasattr(self.actor_critic, 'actor_delta'):
            if freeze:
                params = [
                    {'params': self.actor_critic.actor_time.parameters(), 'name': 'actor'},
                    {'params': self.actor_critic.critics_time.parameters(), 'name': 'critics'},
                    {'params': self.actor_critic.actor_delta.parameters(), 'name': 'actor_delta'},
                    {'params': self.actor_critic.critics_delta.parameters(), 'name': 'critics_delta'},
                    {'params': [self.actor_critic.std], 'name': 'std'},
                ]
            else:
                params = [
                    {'params': self.actor_critic.actor.parameters(), 'name': 'actor'},
                    # {'params': self.actor_critic.critics.parameters(), 'name': 'critics'},
                    {'params': self.actor_critic.actor_time.parameters(), 'name': 'actor_time'},
                    {'params': self.actor_critic.critics_time.parameters(), 'name': 'critics_time'},
                    {'params': self.actor_critic.actor_delta.parameters(), 'name': 'actor_delta'},
                    {'params': self.actor_critic.critics_delta.parameters(), 'name': 'critics_delta'},
                    {'params': [self.actor_critic.std], 'name': 'std'},
                ]

        hidden_weights = [p for group in params for p in group['params'] if p.ndim >= 2]
        hidden_gains_biases = [p for group in params for p in group['params'] if p.ndim < 2]
        param_groups = [
            dict(params=hidden_gains_biases, use_muon=False,
                lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01),
            dict(params=hidden_weights, use_muon=True,
                lr=learning_rate, weight_decay=0.01)
        ]
        # self.optimizer = optim.Adam(params, lr=learning_rate)
        self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        self.train_high = train_high

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_critics):
        self.storage = TrackRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_critics, self.reward_group_weights, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        if obs.isnan().any():
            obs = torch.zeros((obs.shape[0],obs.shape[1]), device=obs.device)
            critic_obs = torch.zeros((critic_obs.shape[0],critic_obs.shape[1]), device=obs.device)

        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values_high = self.actor_critic.evaluate_high(critic_obs).detach()
        self.transition.values_low = self.actor_critic.evaluate_low(torch.cat([critic_obs, self.transition.actions[:, -1:]], dim=-1).clone()).detach()
        a, b = self.actor_critic.get_actions_log_prob(self.transition.actions)
        self.transition.actions_log_prob, self.transition.actions_log_prob_time = a.detach(), b.detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.clone()
        self.transition.critic_observations_high = critic_obs.clone()
        self.transition.critic_observations_low = torch.cat([critic_obs, self.transition.actions[:, -1:]], dim=-1).clone()

        return self.transition.actions
    
    def process_env_step(self, rewards_low, rewards_high, dones, infos, next_obs, next_critic_obs):
        self.transition.next_critic_observations_high = next_critic_obs.clone()
        pred_actions = self.actor_critic.act(next_obs).detach()
        self.transition.next_critic_observations_low = torch.cat([next_critic_obs, pred_actions[:, -1:]], dim=-1).clone()
        self.transition.rewards_low = rewards_low.clone()
        self.transition.rewards_high = rewards_high.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos and self.use_timeout[0]:
            #import ipdb; ipdb.set_trace()
            self.transition.rewards_low += self.gamma[0] * (self.transition.values_low * infos['time_outs'].unsqueeze(1).to(self.device))
        if 'time_outs' in infos and self.use_timeout[1]:
            self.transition.rewards_high += self.gamma[1] * (self.transition.values_high * infos['time_outs'].unsqueeze(1).to(self.device))

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_obs):
        last_values_high = self.actor_critic.evaluate_high(last_critic_obs).detach()
        pred_actions = self.actor_critic.act(last_obs).detach()
        last_values_low = self.actor_critic.evaluate_low(torch.cat([last_critic_obs, pred_actions[:, -1:]], dim=-1).clone()).detach()
        self.storage.compute_returns(last_values_high, last_values_low, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_surrogate_time_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, next_obs_batch, critic_obs_low_batch, actions_batch, next_critic_obs_low_batch, cont_batch, target_values_low_batch, advantages_low_batch, returns_low_batch, shortreturns_batch, old_actions_log_prob_low_batch, \
            old_mu_batch, old_sigma_batch, old_actions_log_prob_high_batch, advantages_time_batch,\
                critic_obs_high_batch, next_critic_obs_high_batch, advantages_high_batch, target_values_high_batch, returns_high_batch  in generator:

                self.actor_critic.act(obs_batch)
                actions_log_prob_low_batch, actions_log_prob_high_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_low_batch = self.actor_critic.evaluate_low(critic_obs_low_batch)
                if self.train_high:
                    value_high_batch = self.actor_critic.evaluate_high(critic_obs_high_batch)
                

                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(5e-4, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio_low = torch.exp(actions_log_prob_low_batch - torch.squeeze(old_actions_log_prob_low_batch))
                # print(ratio.min(), ratio.max())
                surrogate_low = -torch.squeeze(advantages_low_batch) * ratio_low
                surrogate_low_clipped = -torch.squeeze(advantages_low_batch) * torch.clamp(ratio_low, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss_low = torch.max(surrogate_low, surrogate_low_clipped).mean()
                surrogate_loss = surrogate_loss_low
                if self.train_high:
                    ratio_high = torch.exp(actions_log_prob_high_batch - torch.squeeze(old_actions_log_prob_high_batch))
                    surrogate_high = -torch.squeeze(advantages_high_batch) * ratio_high
                    surrogate_high_clipped = -torch.squeeze(advantages_high_batch) * torch.clamp(ratio_low, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    surrogate_loss_high = torch.max(surrogate_high, surrogate_high_clipped).mean()
                    surrogate_loss += surrogate_loss_high

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped_low = target_values_low_batch + (value_low_batch - target_values_low_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses_low = (value_low_batch - returns_low_batch).pow(2)
                    value_losses_clipped_low = (value_clipped_low - returns_low_batch).pow(2)
                    value_loss_low = torch.max(value_losses_low, value_losses_clipped_low).mean()
                    value_loss = value_loss_low
                    if self.train_high:
                        value_clipped_high = target_values_high_batch + (value_high_batch - target_values_high_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses_high = (value_high_batch - returns_high_batch).pow(2)
                        value_losses_clipped_high = (value_clipped_high - returns_high_batch).pow(2)
                        value_loss_high = torch.max(value_losses_high, value_losses_clipped_high).mean()
                        value_loss += value_loss_high
                else:
                    value_loss = (returns_low_batch - value_low_batch).pow(2).mean() + (returns_high_batch - value_high_batch).pow(2).mean()
                
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()#+ estvalue_loss

                # Smooth loss
                if self.use_smooth:
                    epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
                    policy_smooth_coef = self.smoothness_upper_bound * epsilon; value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

                    mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                    mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)
                    mix_critic_obs_low_batch = critic_obs_low_batch + mix_weights * (next_critic_obs_low_batch - critic_obs_low_batch)
                    policy_smooth_loss = torch.square(torch.norm(mu_batch[:, :-1] - self.actor_critic.act_inference(mix_obs_batch)[:, :-1], dim=-1)).mean()
                    value_smooth_loss_low = torch.square(torch.norm(value_low_batch - self.actor_critic.evaluate_low(mix_critic_obs_low_batch), dim=-1)).mean()
                    value_loss = value_smooth_loss_low
                    if self.train_high:
                        mix_critic_obs_high_batch = critic_obs_high_batch + mix_weights * (next_critic_obs_high_batch - critic_obs_high_batch)
                        value_smooth_loss_high = torch.square(torch.norm(value_high_batch - self.actor_critic.evaluate_high(mix_critic_obs_high_batch), dim=-1)).mean()
                        # value_loss += value_smooth_loss_high

                    smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_loss
                    loss += smooth_loss
                else:
                    smooth_loss = torch.tensor(0.0, device=self.device)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss_low.item()
                if self.train_high:
                    mean_surrogate_time_loss += surrogate_loss_high.item() if self.infer_keyframe_time else 0
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if self.train_high:
            mean_surrogate_time_loss /= num_updates
        self.storage.clear()

        self.update_count += 1

        if self.train_high:
            return value_loss_low.mean().item(), value_loss_high.mean().item(), mean_surrogate_loss, mean_surrogate_time_loss, smooth_loss.item(), entropy_batch.mean()
        else:
            return value_loss_low.mean().item(), 0, mean_surrogate_loss, 0, smooth_loss.item(), entropy_batch.mean()