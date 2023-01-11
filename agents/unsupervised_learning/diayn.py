import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils

from utils.replay_buffer import MetaBatch
from .sac import SACAgent


class DIAYN(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, obs):
        skill_pred = self.skill_pred_net(obs)
        return skill_pred


class DIAYNAgent(SACAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, skill_type, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        self.skill_type = skill_type

        # create actor and critic
        super().__init__(**kwargs)

        # create diayn
        self.diayn = DIAYN(self.obs_shape - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)

        self.diayn.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        
        if self.skill_type == 'uniform':
            skill = np.random.uniform(0, 1, self.skill_dim)
            skill = skill.astype(dtype=np.float32)
            print(f'skill: {skill}')
        else:
            skill = np.zeros(self.skill_dim, dtype=np.float32)
            skill[np.random.choice(self.skill_dim)] = 1.0
        
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_diayn(self, skill, next_obs, step):
        metrics = dict()

        loss, df_accuracy = self.compute_diayn_loss(next_obs, skill)

        self.diayn_opt.zero_grad()
        loss.backward()
        self.diayn_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            metrics['diayn_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, next_obs, step):
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn(next_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)

        return reward * self.diayn_scale

    def compute_diayn_loss(self, next_state, skill):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn(next_state)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, b: MetaBatch, step):
        
        metrics = dict()

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch((b.s, b.a, b.r, b.d, b.ns, b.z), self.device)

        if self.reward_free:
            metrics.update(self.update_diayn(skill, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

       # update critic target
        with torch.no_grad():
            self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
            self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

        return metrics

    def act(self, obs: np.array, meta: np.array, step, eval_mode) -> np.array:
        meta = meta['skill']
        obs = np.concatenate((obs, meta), axis=0)
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        action, _ = self.sample_action_and_compute_log_pi(obs, use_reparametrization_trick=False)
        return action.cpu().numpy()[0]  # no need to detach first because we are not using the reparametrization trick