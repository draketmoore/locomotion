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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.grid_adaptive_rough_config import GridAdaptiveRoughCfg

class GridAdaptive(LeggedRobot):
    cfg : GridAdaptiveRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if self.cfg.commands.curriculum:
            # self.lin_command_distribution = [0.] * int((self.command_ranges["lin_vel_x"][1] - self.command_ranges["lin_vel_x"][0]) / 0.5) # Range of -4 to 4 with resolution of 0.5
            # self.ang_command_distribution = [0.] * int((self.command_ranges["ang_vel_yaw"][1] - self.command_ranges["ang_vel_yaw"][0]) / 0.5) # Range from -5 to 5 with resolution of 0.5
            # self.lin_command_distribution = np.array(self.lin_command_distribution)
            # self.ang_command_distribution = np.array(self.ang_command_distribution)
            # self.lin_command_distribution[6:10] += 0.25
            # # self.lin_command_distribution[8] = 0.5
            # self.ang_command_distribution[8:12] += 0.25
            # # self.ang_command_distribution[12] = 0.5
            # print(self.lin_command_distribution, self.ang_command_distribution)

            #______________________

            lin_num = int((self.command_ranges["lin_max_x"][1] - self.command_ranges["lin_max_x"][0]) / 0.5)
            ang_num = int((self.command_ranges["ang_max"][1] - self.command_ranges["ang_max"][0]) / 0.5)


            self.env_command_bins = np.zeros(self.num_envs, dtype=np.int)
            

            raw_grid = np.meshgrid(*[np.linspace(-self.command_ranges["lin_max_x"][0], self.command_ranges["lin_max_x"][1], lin_num),
                                     np.linspace(-self.command_ranges["ang_max"][0], self.command_ranges["ang_max"][1], ang_num)], indexing='ij')
            self.grid = np.array(raw_grid).reshape([2, -1])
            self.weights = np.zeros(len(self.grid[0]))
            self.indices = np.arange(len(self.grid[0]))
            self.episode_reward_lin = np.zeros(len(self.grid[0]))
            self.episode_reward_ang = np.zeros(len(self.grid[0]))
            
            low = np.array([self.command_ranges["lin_vel_x"][0], self.command_ranges["ang_vel_yaw"][0]])
            high = np.array([self.command_ranges["lin_vel_x"][1], self.command_ranges["ang_vel_yaw"][1]])
            inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)

            self.weights[inds] = 1.





        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)


    def _prepare_reward_function(self):
        super()._prepare_reward_function()

        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in list(self.reward_scales.keys())}

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_reward(self):
        super().compute_reward()
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.command_sums[name] += rew


    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """

            
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            print("Updated Curriculum")
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_lin_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_lin_curriculum)

            # self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            # self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

        if torch.mean(self.episode_sums['tracking_ang_vel'][env_ids]) / self.max_episode_length > 0.5 * self.reward_scales['tracking_ang_vel']:
            self.command_ranges["ang_vel_yaw"][0] = np.clip(torch.mean(self.commands[env_ids, 2]).item()  - 0.5, -self.cfg.commands.max_ang_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(torch.mean(self.commands[env_ids, 2]).item()  + 0.5, 0., self.cfg.commands.max_ang_curriculum)


        # for id in env_ids:
        #     if self.episode_sums['tracking_lin_vel'][id] / self.max_episode_length > 0.8 * self.reward_scales['tracking_lin_vel']:
        #         # print("Before update:", self.lin_command_distribution, self.ang_command_distribution, self.commands[id])
        #         lin_dist_index = int(self.commands[id, 0] / 0.5)
        #         lin_dist_index = lin_dist_index + 8 if self.commands[id, 0] > 0 else lin_dist_index + 7
                
        #         self.lin_command_distribution[lin_dist_index-1:lin_dist_index+2] += 0.05
        #         self.lin_command_distribution = self.lin_command_distribution / np.sum(self.lin_command_distribution)
        #         # print("After Update:", self.lin_command_distribution, self.ang_command_distribution)

        #     if self.episode_sums['tracking_ang_vel'][id] / self.max_episode_length > 0.8 * self.reward_scales['tracking_ang_vel']:
        #         ang_dist_index = int(self.commands[id, 2] / 0.5) + 10
        #         ang_dist_index = ang_dist_index + 10 if self.commands[id, 2] > 0 else ang_dist_index + 9
        #         self.ang_command_distribution[ang_dist_index-1:ang_dist_index+2] += 0.05
        #         self.ang_command_distribution = self.ang_command_distribution / np.sum(self.ang_command_distribution)

    def _resample_commands(self, env_ids):
        # print("Resampled")
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # print('Before', self.commands[env_ids])
        # print(self.lin_command_distribution)
        # print(self.ang_command_distribution)
        if self.cfg.commands.curriculum:
            # lin_dist_index = np.random.choice(len(self.lin_command_distribution), 1, p=self.lin_command_distribution)
            # ang_dist_index = np.random.choice(len(self.ang_command_distribution), 1, p=self.ang_command_distribution)
            # min_lin = (lin_dist_index - 8) * 0.5
            # max_lin = (lin_dist_index - 7) * 0.5
            # min_ang = (ang_dist_index - 10) * 0.5
            # max_ang = (ang_dist_index - 9) * 0.5
            # print(self.lin_command_distribution, lin_dist_index, min_lin, max_lin)
            # self.commands[env_ids, 0] = torch_rand_float(min_lin, max_lin, (len(env_ids), 1), device=self.device).squeeze(1)
            # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # if self.cfg.commands.heading_command:
            #     self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # else:
            #     self.commands[env_ids, 2] = torch_rand_float(min_ang, max_ang, (len(env_ids), 1), device=self.device).squeeze(1)

            if len(env_ids) == 0: return

            # update step just uses train env performance (for now)
            self.update_curriculum_distribution(0.5, env_ids)
            
            new_bin_inds = np.random.choice(self.indices, len(env_ids), p=self.weights / self.weights.sum())
            cgf_centroid = self.grid.T[new_bin_inds]

            new_commands = []
            for v_range in cgf_centroid:
                bin_sizes = np.array([*[0.4, 0.4]])
                low, high = v_range + bin_sizes / 2, v_range - bin_sizes / 2
                new_commands.append(np.random.uniform(low, high))
            new_commands = np.stack(new_commands)

            self.env_command_bins[env_ids.cpu().numpy()] = new_bin_inds
            self.commands[env_ids, 0] = torch.Tensor(new_commands[:, 0]).to(self.device)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = torch.Tensor(new_commands[:, 1]).to(self.device)

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

            # reset command sums
            for key in self.command_sums.keys():
                self.command_sums[key][env_ids] = 0.

  
        else:
        
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # print('After', self.commands[env_ids])
        # print(self.lin_command_distribution)
        # print(self.ang_command_distribution)


    def update_curriculum_distribution(self, update_step, env_ids):
        # old_bins = self.env_command_bins[env_ids.cpu().numpy()]
        env_ids = env_ids.cpu().numpy()
        bin_ids = self.env_command_bins[env_ids]

        timesteps = int(10 / self.dt)
        ep_len = min(self.cfg.env.episode_length_s, timesteps)
        lin_vel_rewards = self.command_sums["tracking_lin_vel"][env_ids] / ep_len
        ang_vel_rewards = self.command_sums["tracking_ang_vel"][env_ids] / ep_len

        lin_vel_threshold = self.cfg.commands.lin_curriculum_threshold * self.reward_scales["tracking_lin_vel"]
        ang_vel_threshold = self.cfg.commands.ang_curriculum_threshold * self.reward_scales["tracking_ang_vel"]
        
        self.episode_reward_lin[bin_ids] = lin_vel_rewards.cpu().numpy()
        self.episode_reward_ang[bin_ids] = ang_vel_rewards.cpu().numpy()

        is_success = ((lin_vel_rewards > lin_vel_threshold) * (ang_vel_rewards > ang_vel_threshold))
        self.weights[bin_ids[is_success.cpu().numpy()]] = np.clip(self.weights[bin_ids[is_success.cpu().numpy()]] + 0.2, 0, 1)

        # print(self.grid.shape, len(bin_ids))
        adjacents = np.logical_and(
            self.grid[:, None, :].repeat(len(bin_ids), axis=1) >= self.grid[:, bin_ids, None] - update_step,
            self.grid[:, None, :].repeat(len(bin_ids), axis=1) <= self.grid[:, bin_ids, None] + update_step
        ).all(axis=0)

        for adjacent in adjacents:
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)