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

"""
Reimplementation of https://github.com/RLAgent/state-marginal-matching:
 - Removed redundant forward passes
 - No updating p_z
 - Added finetuning procedure from what's described in DIAYN
 - VAE encodes and decodes from the encoding from DDPG when n > 1
   as the paper does not make it clear how to include skills with pixel input
 - When n=1, obs_type=pixel, remove the False from line 144
    to input pixels into the vae
 - TODO: when using pixel-based vae (n=1), gpu may run out of memory.
"""


class VAE(nn.Module):
    def __init__(self, obs_dim, z_dim, code_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim

        self.make_networks(obs_dim, z_dim, code_dim)
        self.beta = vae_beta

        self.apply(utils.weight_init)
        self.device = device

    def make_networks(self, obs_dim, z_dim, code_dim):
        self.enc = nn.Sequential(nn.Linear(obs_dim + z_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU())
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(nn.Linear(code_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU(),
                                 nn.Linear(150, obs_dim + z_dim))

    def encode(self, obs_z):
        enc_features = self.enc(obs_z)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        stds = (0.5 * logvar).exp()
        return mu, logvar, stds

    def forward(self, obs_z, epsilon):
        mu, logvar, stds = self.encode(obs_z)
        code = epsilon * stds + mu
        obs_distr_params = self.dec(code)
        return obs_distr_params, (mu, logvar, stds)

    def loss(self, obs_z):
        epsilon = torch.randn([obs_z.shape[0], self.code_dim]).to(self.device)
        obs_distr_params, (mu, logvar, stds) = self(obs_z, epsilon)
        kle = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                               dim=1).mean()
        log_prob = F.mse_loss(obs_z, obs_distr_params, reduction='none')

        loss = self.beta * kle + log_prob.mean()
        return loss, log_prob.sum(list(range(1, len(log_prob.shape)))).view(
            log_prob.shape[0], 1)


class PVae(VAE):
    def make_networks(self, obs_shape, z_dim, code_dim):
        self.enc = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Flatten(),
                                 nn.Linear(32 * 35 * 35, 150), nn.ReLU())
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, 32 * 35 * 35), nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, 35, 35)),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, obs_shape[0], 4, stride=1))


class SMM(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.z_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, z_dim))
        self.vae = VAE(obs_dim=obs_dim,
                       z_dim=z_dim,
                       code_dim=128,
                       vae_beta=vae_beta,
                       device=device)
        self.apply(utils.weight_init)

        print(f"Density Model --> {self.vae}")
        print(f"Discriminator Model --> {self.z_pred_net}")

    def predict_logits(self, obs):
        z_pred_logits = self.z_pred_net(obs)
        return z_pred_logits

    def loss(self, logits, z):
        z_labels = torch.argmax(z, 1)
        return nn.CrossEntropyLoss(reduction='none')(logits, z_labels)


class PSMM(nn.Module):
    def __init__(self, obs_shape, z_dim, hidden_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.vae = PVae(obs_dim=obs_shape,
                        z_dim=z_dim,
                        code_dim=128,
                        vae_beta=vae_beta,
                        device=device)
        self.apply(utils.weight_init)

    # discriminator not needed when n=1, as z is degenerate
    def predict_logits(self, obs):
        raise NotImplementedError

    def loss(self, logits, z):
        raise NotImplementedError


class SMMAgent(SACAgent):
    def __init__(self, z_dim, sp_lr, vae_lr, vae_beta, state_ent_coef,
                 latent_ent_coef, latent_cond_ent_coef, reward_scaling, update_encoder, use_goal_reward=False,
                 **kwargs):
        self.z_dim = z_dim

        self.state_ent_coef = state_ent_coef
        self.latent_ent_coef = latent_ent_coef
        self.latent_cond_ent_coef = latent_cond_ent_coef
        self.reward_scaling = reward_scaling
        self.update_encoder = update_encoder
        self.use_goal_reward = use_goal_reward

        kwargs["meta_dim"] = self.z_dim
        #TODO: Fix this!
        self.obs_type = kwargs["obs_type"]
        super().__init__(**kwargs)
        # self.obs_shape is now the real obs_shape (or repr_dim) + z_dim
        self.smm = SMM(self.obs_shape - z_dim,
                       z_dim,
                       hidden_dim=kwargs['hidden_dim'],
                       vae_beta=vae_beta,
                       device=kwargs['device']).to(kwargs['device'])
        
        self.goal = [150, 75]

        WINDOW_WIDTH = 180
        WINDOW_HEIGHT = 150

        bounds = [[75, 55], [100, 95]]

        def _normalize(obs):
            obs[0] = (obs[0] - WINDOW_WIDTH/2) / (WINDOW_WIDTH/2)
            obs[1] = (obs[1] - WINDOW_HEIGHT/2) / (WINDOW_HEIGHT/2)
            return obs

        
        self.min_x, self.min_y = _normalize(bounds[0])
        self.max_x, self.max_y = _normalize(bounds[1])
        
        self.walls = self.obstacle
        
        self.goal = tuple(_normalize(self.goal))
        # self.goal_dist = 0.55
        self.goal_dist = 0.35

        self.pred_optimizer = torch.optim.Adam(self.smm.z_pred_net.parameters(), lr=sp_lr)
        self.vae_optimizer = torch.optim.Adam(self.smm.vae.parameters(), lr=vae_lr)

        self.smm.train()

        # fine tuning SMM agent
        self.ft_returns = np.zeros(z_dim, dtype=np.float32)
        self.ft_not_finished = [True for z in range(z_dim)]

    def get_meta_specs(self):
        return (specs.Array((self.z_dim,), np.float32, 'z'),)

    def init_meta(self):
        z = np.zeros(self.z_dim, dtype=np.float32)
        z[np.random.choice(self.z_dim)] = 1.0
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def update_meta(self, meta, global_step, time_step):
        # during fine-tuning, find the best skill and fine-tune that one only.
        if self.reward_free:
            return self.update_meta_ft(meta, global_step, time_step)
        # during training, change to randomly sampled z at the end of the episode
        if time_step.last():
            return self.init_meta()
        return meta

    def obstacle(self, state):
            if type(state) == np.ndarray:
                lower = (self.min_x, self.min_y)
                upper = (self.max_x, self.max_y)
                state = np.array(state)
                component_viol = (state > lower) * (state < upper)
                return np.product(component_viol, axis=-1)
            if type(state) == torch.Tensor:
                lower = torch.from_numpy(np.array((self.min_x, self.min_y)))
                upper = torch.from_numpy(np.array((self.max_x, self.max_y)))
                component_viol = (state > lower) * (state < upper)
                return torch.prod(component_viol, dim=-1)

    def update_meta_ft(self, meta, global_step, time_step):
        z_ind = meta['z'].argmax()
        if any(self.ft_not_finished):
            self.ft_returns[z_ind] += time_step.reward
            if time_step.last():
                if not any(self.ft_not_finished):
                    # choose the best
                    new_z_ind = self.ft_returns.argmax()
                else:
                    # or the next z to try
                    self.ft_not_finished[z_ind] = False
                    not_tried_z = sum(self.ft_not_finished)
                    # uniformly sample from the remaining unused z
                    for i in range(self.z_dim):
                        if self.ft_not_finished[i]:
                            if np.random.random() < 1 / not_tried_z:
                                new_z_ind = i
                                break
                            not_tried_z -= 1
                new_z = np.zeros(self.z_dim, dtype=np.float32)
                new_z[new_z_ind] = 1.0
                meta['z'] = new_z
        return meta

    def update_vae(self, obs_z):
        metrics = dict()
        loss, h_s_z = self.smm.vae.loss(obs_z)
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        metrics['loss_vae'] = loss.cpu().item()

        return metrics, h_s_z

    def update_pred(self, obs, z):
        metrics = dict()
        logits = self.smm.predict_logits(obs)
        h_z_s = self.smm.loss(logits, z).unsqueeze(-1)
        loss = h_z_s.mean()
        self.pred_optimizer.zero_grad()
        loss.backward()
        self.pred_optimizer.step()

        metrics['loss_pred'] = loss.cpu().item()

        return metrics, h_z_s

    def get_goal_p_star(self, agent_pos):
        x_dist = agent_pos[:, 0] - self.goal[0]
        y_dist = agent_pos[:, 1] - self.goal[1]
        x_dist = x_dist.cpu().detach().numpy()
        y_dist = y_dist.cpu().detach().numpy()
        dist = np.linalg.norm((x_dist, y_dist), axis=0)
        # def _prior_distro(dist):
        #     if dist > 1.0:
        #         p_star = 1/dist
        #     else:
        #         p_star = 1.0
        #     return p_star
        # p_star = np.array(list(map(_prior_distro, dist)), dtype=np.float32)
        p_star = -1.0 * dist
        agent_pos = agent_pos.cpu().detach().numpy()

        def add_penalty(pos, p_star):
            """Penalty for hitting wall"""
            constr = self.walls(pos)
            if constr:
                p_star -= 1
            return p_star

        def add_goal_bonus(dist, p_star):
            """Bonus for being in goal state"""
            if dist < self.goal_dist:
                p_star += 5
            return p_star

        p_map = map(add_penalty, agent_pos[:, 0:2], p_star)
        p_star = np.fromiter(p_map, dtype=np.float32)
        p_map = map(add_goal_bonus, dist, p_star)
        p_star = np.fromiter(p_map, dtype=np.float32)
        return p_star

    def update(self, b: MetaBatch, step):
        
        metrics = dict()

        obs, action, extr_reward, discount, next_obs, z = utils.to_torch((b.s, b.a, b.r, b.d, b.ns, b.z), self.device)

        obs_z = torch.cat([obs, z], dim=1)  # do not learn encoder in the VAE
        next_obs_z = torch.cat([next_obs, z], dim=1)

        # calculate intinsic reward
        if self.reward_free:

            vae_metrics, h_s_z = self.update_vae(obs_z)
            pred_metrics, h_z_s = self.update_pred(obs.detach(), z)
            h_z = np.log(self.z_dim)  # One-hot z encoding
            h_z *= torch.ones_like(extr_reward).to(self.device)

            if self.obs_type=='pixels':
                # p^*(s) is ignored, as state space dimension is inaccessible from pixel input
                pred_log_ratios = self.state_ent_coef * torch.log(h_s_z.detach())
                intr_reward = pred_log_ratios + self.latent_ent_coef * h_z + self.latent_cond_ent_coef * h_z_s.detach()
                reward = intr_reward
            else:
                # p^*(s) is based on the goal hitting time
                # TODO: Assumes obs is just (x, y) at front
                if self.use_goal_reward:
                    p_star = self.get_goal_p_star(obs)
                    p_star = torch.tensor(p_star).to(self.device)
                    pred_log_ratios = self.reward_scaling * p_star + self.state_ent_coef * h_s_z.detach()
                else:
                    pred_log_ratios = self.reward_scaling * extr_reward + self.state_ent_coef * h_s_z.detach()
                intr_reward =  pred_log_ratios + self.latent_ent_coef * h_z + self.latent_cond_ent_coef * h_z_s.detach()

                # if pred_log_ratios.mean().item() > 10000:
                #     print(f'intr_reward.mean().item(): {intr_reward.mean().item()}')
                #     print(f'pred_log_ratios: {pred_log_ratios}')
                #     print(f'h_s_z.detach(): {h_s_z.detach()}')
                #     print(f'reward: {reward}')
                #     print(f'obs: {obs}')

                reward = intr_reward
        else:
            reward = extr_reward

        if self.obs_type=='states' and self.reward_free:
            # add reward free to states motivation
            metrics['intr_reward'] = intr_reward.mean().item()
            if self.use_goal_reward:
                metrics['p_star'] = self.reward_scaling * p_star.mean().item()
            else:
                metrics['extr_reward'] = self.reward_scaling * extr_reward.mean().item()
            metrics['pred_log_ratios'] = pred_log_ratios.mean().item()
            metrics['latent_ent_coef'] = (self.latent_ent_coef * h_z).mean().item()
            metrics['latent_cond_ent_coef'] = (self.latent_cond_ent_coef * h_z_s.detach()).mean().item()
            # add loss values
            metrics['loss_vae'] = vae_metrics['loss_vae']
            metrics['loss_pred'] = pred_metrics['loss_pred']

        if self.use_tb or self.use_wandb:
            metrics.update(vae_metrics)
            metrics.update(pred_metrics)
            metrics['intr_reward'] = intr_reward.mean().item()
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs_z = obs_z.detach()
            next_obs_z = next_obs_z.detach()

        # update critic
        metrics.update(
            self.update_critic(obs_z.detach(), action, reward, discount, next_obs_z.detach(), step)
        )

        # update actor
        metrics.update(self.update_actor(obs_z.detach(), step))

        # update critic target
        with torch.no_grad():
            self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
            self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

        return metrics

    def act(self, obs: np.array, meta: np.array, step, eval_mode) -> np.array:
        # state = torch.tensor(state).unsqueeze(0).float()
        meta = meta['z']
        obs = np.concatenate((obs, meta), axis=0)
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        action, _ = self.sample_action_and_compute_log_pi(obs, use_reparametrization_trick=False)
        return action.cpu().numpy()[0]  # no need to detach first because we are not using the reparametrization trick