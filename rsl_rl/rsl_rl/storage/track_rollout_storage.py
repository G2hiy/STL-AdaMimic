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
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class TrackRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.next_critic_observations = None
            self.action_log_prob_time = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape,num_critics, reward_group_weights,  device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_critics = num_critics
        self.reward_group_weights = torch.tensor(reward_group_weights, device=self.device).view(2, 1, -1)

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations_low = torch.zeros(num_transitions_per_env, num_envs, *[privileged_obs_shape[0]+1], device=self.device)
            self.privileged_observations_high = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
            self.next_privileged_observations_low = torch.zeros(num_transitions_per_env, num_envs, *[privileged_obs_shape[0]+1], device=self.device)
            self.next_privileged_observations_high = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None

        self.rewards_low = torch.zeros(num_transitions_per_env, num_envs,  num_critics, device=self.device)
        self.rewards_high = torch.zeros(num_transitions_per_env, num_envs,  num_critics, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions_log_prob_time = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values_low = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.values_high = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)

        self.returns_low = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.returns_high = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)

        self.shortreturns_low = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.shortreturns_high = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)

        self.advantages_low = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.advantages_high = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations_low is not None: 
            self.privileged_observations_low[self.step].copy_(transition.critic_observations_low)
            self.privileged_observations_high[self.step].copy_(transition.critic_observations_high)
        if self.next_privileged_observations_low is not None:
            self.next_privileged_observations_low[self.step].copy_(transition.next_critic_observations_low)
            self.next_privileged_observations_high[self.step].copy_(transition.next_critic_observations_high)
        self.actions[self.step].copy_(transition.actions)
        self.rewards_low[self.step].copy_(transition.rewards_low.view(-1, self.num_critics))
        self.rewards_high[self.step].copy_(transition.rewards_high.view(-1, self.num_critics))

        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values_low[self.step].copy_(transition.values_low)
        self.values_high[self.step].copy_(transition.values_high)

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.actions_log_prob_time[self.step].copy_(transition.actions_log_prob_time.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values_high, last_values_low, gamma, lam):
        advantage_low = 0
        advantage_high = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values_low = last_values_low
                next_values_high = last_values_high
            else:
                next_values_low = self.values_low[step + 1]
                next_values_high = self.values_high[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta_low = self.rewards_low[step] + next_is_not_terminal * gamma[0] * next_values_low - self.values_low[step]
            delta_high = self.rewards_high[step] + next_is_not_terminal * gamma[1] * next_values_high - self.values_high[step]
            advantage_low = delta_low + next_is_not_terminal * gamma[0] * lam * advantage_low
            advantage_high = delta_high + next_is_not_terminal * gamma[1] * lam * advantage_high
            self.returns_low[step] = advantage_low + self.values_low[step]
            self.returns_high[step] = advantage_high + self.values_high[step]
        
            for critic_idx in range(self.num_critics):
                self.advantages_low[:, :, critic_idx] = self.returns_low[:, :, critic_idx] - self.values_low[:, :, critic_idx]
                self.advantages_low[:, :, critic_idx] = (self.advantages_low[:, :, critic_idx] - self.advantages_low[:, :, critic_idx].mean()) / (self.advantages_low[:, :, critic_idx].std() + 1e-8)
                self.advantages_high[:, :, critic_idx] = self.returns_high[:, :, critic_idx] - self.values_high[:, :, critic_idx]
                self.advantages_high[:, :, critic_idx] = (self.advantages_high[:, :, critic_idx] - self.advantages_high[:, :, critic_idx].mean()) / (self.advantages_high[:, :, critic_idx].std() + 1e-8)

        self.multi_critic_advantages_low = torch.sum(self.advantages_low * self.reward_group_weights[0], dim=-1)
        self.multi_critic_advantages_high = torch.sum(self.advantages_high * self.reward_group_weights[1], dim=-1)


    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * (self.num_transitions_per_env - 1)
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations[:-1].flatten(0, 1)
        if self.privileged_observations_low is not None:
            critic_observations_low = self.privileged_observations_low[:-1].flatten(0, 1)
            critic_observations_high = self.privileged_observations_high[:-1].flatten(0, 1)
            next_critic_observations_low = self.next_privileged_observations_low[:-1].flatten(0, 1)
            next_critic_observations_high = self.next_privileged_observations_high[:-1].flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = observations


        next_observations = self.observations[1:].flatten(0, 1)
        actions = self.actions[:-1].flatten(0, 1)
        values_low = self.values_low[:-1].flatten(0, 1)
        values_high = self.values_high[:-1].flatten(0, 1)
        returns_low = self.returns_low[:-1].flatten(0, 1)
        returns_high = self.returns_high[:-1].flatten(0, 1)
        shortreturns_low = self.shortreturns_low[:-1].flatten(0, 1)
        shortreturns_high = self.shortreturns_high[:-1].flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob[:-1].flatten(0, 1)
        old_actions_log_prob_time = self.actions_log_prob_time[:-1].flatten(0, 1)
        advantages_high = self.multi_critic_advantages_high[:-1].flatten(0, 1)
        advantages_low = self.multi_critic_advantages_low[:-1].flatten(0, 1)
        old_mu = self.mu[:-1].flatten(0, 1)
        old_sigma = self.sigma[:-1].flatten(0, 1)
        not_dones = 1 - self.dones[:-1].float().flatten(0, 1)
        advantages_time = self.advantages_high[:, :, 1].flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                cont_batch = not_dones[batch_idx]
                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                next_critic_observations_low_batch = next_critic_observations_low[batch_idx]
                next_critic_observations_high_batch = next_critic_observations_high[batch_idx]
                critic_observations_low_batch = critic_observations_low[batch_idx]
                critic_observations_high_batch = critic_observations_high[batch_idx]
 
                actions_batch = actions[batch_idx]
                target_values_low_batch = values_low[batch_idx]
                target_values_high_batch = values_high[batch_idx]
                returns_low_batch = returns_low[batch_idx]
                returns_high_batch = returns_high[batch_idx]
                shortreturns_batch = shortreturns_low[batch_idx]
                old_actions_log_prob_low_batch = old_actions_log_prob[batch_idx]
                old_actions_log_prob_high_batch = old_actions_log_prob_time[batch_idx]
                advantages_low_batch = advantages_low[batch_idx]
                advantages_high_batch = advantages_high[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                advantages_time_batch = advantages_high[batch_idx]

                yield obs_batch, next_obs_batch, critic_observations_low_batch, actions_batch, next_critic_observations_low_batch, cont_batch, target_values_low_batch, advantages_low_batch, returns_low_batch, shortreturns_batch,\
                       old_actions_log_prob_low_batch, old_mu_batch, old_sigma_batch, old_actions_log_prob_high_batch, advantages_time_batch,\
                        critic_observations_high_batch, next_critic_observations_high_batch, advantages_high_batch, target_values_high_batch, returns_high_batch
                        
                        