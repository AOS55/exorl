import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from utils.replay_buffer import Batch
import utils

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=5,
        num_neurons_per_hidden_layer:int=64
    ) -> nn.Sequential:

    layers = []
    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.ReLU(),
    ])
    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])
    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)

class NormalPolicyNet(nn.Module):

    """Outputs a distribution with parameters learnable by gradient descent."""

    def __init__(self, input_dim, action_dim, hidden_dim):
        super(NormalPolicyNet, self).__init__()
        self.shared_net   = get_net(num_in=input_dim, num_out=64, final_activation=nn.ReLU(), num_neurons_per_hidden_layer=hidden_dim)
        self.means_net    = nn.Linear(64, action_dim)
        self.log_stds_net = nn.Linear(64, action_dim)

    def forward(self, states: torch.tensor):

        out = self.shared_net(states)
        means, log_stds = self.means_net(out), self.log_stds_net(out)

        # the gradient of computing log_stds first and then using torch.exp
        # is much more well-behaved then computing stds directly using nn.Softplus()
        # ref: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L26

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        stds = torch.exp(torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX))
        return Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)

class QNet(nn.Module):

    """Has little quirks; just a wrapper so that I don't need to call concat many times"""

    def __init__(self, input_dim, action_dim, hidden_dim):
        super(QNet, self).__init__()
        self.net = get_net(num_in=input_dim+action_dim, num_out=1, final_activation=None, num_neurons_per_hidden_layer=hidden_dim)

    def forward(self, states: torch.tensor, actions: torch.tensor):
        return self.net(torch.cat([states, actions], dim=1))


class SACAgent:

    def __init__(self,
                 name,
                 reward_free,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 hidden_dim,
                 gamma,
                 tau,
                 alpha,
                 obs_type,
                 target_step_interval,
                 update_every_steps,
                 automatic_entropy_tuning,
                 use_tb,
                 use_wandb,
                 num_expl_steps,
                 batch_size,
                 init_critic,
                 nstep,
                 meta_dim = 0):
    
        self.obs_shape = obs_shape[0] + meta_dim
        self.action_shape = action_shape[0]
        self.hidden_dim = hidden_dim
        self.device = device
        self.lr = lr

        self.reward_free = reward_free
        self.obs_type = obs_type
        self.use_wandb = use_wandb
        self.use_tb = use_tb
        
        print(f'obs_shape: {self.obs_shape}, action_shape: {self.action_shape}, hidden_dim: {self.hidden_dim}')
        self.Normal = NormalPolicyNet(input_dim=self.obs_shape, action_dim=self.action_shape, hidden_dim=self.hidden_dim).to(device)
        self.Normal_optimizer = optim.Adam(self.Normal.parameters(), lr=lr)

        self.Q1 = QNet(input_dim=self.obs_shape, action_dim=self.action_shape, hidden_dim=self.hidden_dim).to(device)
        self.Q1_targ = QNet(input_dim=self.obs_shape, action_dim=self.action_shape, hidden_dim=self.hidden_dim).to(device)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)

        self.Q2 = QNet(input_dim=self.obs_shape, action_dim=self.action_shape, hidden_dim=self.hidden_dim).to(device)
        self.Q2_targ = QNet(input_dim=self.obs_shape, action_dim=self.action_shape, hidden_dim=self.hidden_dim).to(device)
        self.Q2_targ.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = 1-tau
        
        self.init_critic = init_critic

        self.train()

        print(f'Actor --> {self.Normal}')
        print(f'Critic --> Q1: {self.Q1}, Q1_targ: {self.Q1_targ}, Q2: {self.Q2}, Q2_targ: {self.Q2_targ}')
    
    def train(self, training=True):
        self.training = training
        self.Normal.train(training)
        self.Q1.train(training)
        self.Q2.train(training)

    def min_i_12(self, a: torch.tensor, b: torch.tensor) -> torch.tensor:
        return torch.min(a, b)

    def sample_action_and_compute_log_pi(self, state: torch.tensor, use_reparametrization_trick: bool) -> tuple:
        mu_given_s = self.Normal(state)
        u = mu_given_s.rsample() if use_reparametrization_trick else mu_given_s.sample()
        a = torch.tanh(u)
        log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
        return a, log_pi_a_given_s

    def clip_gradient(self, net: nn.Module) -> None:
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        
        metrics = dict()

        with torch.no_grad():
            next_action, log_pi_next_action_given_next_state = self.sample_action_and_compute_log_pi(next_obs, use_reparametrization_trick=False)
            targets = reward + self.gamma * (1-discount) * (self.min_i_12(self.Q1_targ(next_obs, next_action), self.Q2_targ(next_obs, next_action)) - self.alpha * log_pi_next_action_given_next_state)

        Q1_predictions = self.Q1(obs, action)
        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.clip_gradient(net=self.Q1)
        self.Q1_optimizer.step()

        Q2_predictions = self.Q2(obs, action)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.clip_gradient(net=self.Q2)
        self.Q2_optimizer.step()

        metrics['train/Q1_loss'] = Q1_loss.item()
        metrics['train/Q2_loss'] = Q2_loss.item()

        return metrics

    def update_actor(self, obs, step):
        
        metrics = dict()

        for param in self.Q1.parameters():
            param.requires_grad = False
        for param in self.Q2.parameters():
            param.requires_grad = False
        
        a, log_pi_a_given_s = self.sample_action_and_compute_log_pi(obs, use_reparametrization_trick=True)
        policy_loss = -torch.mean(self.min_i_12(self.Q1(obs, a), self.Q2(obs, a)) - self.alpha * log_pi_a_given_s)

        self.Normal_optimizer.zero_grad()
        policy_loss.backward()
        self.clip_gradient(net=self.Normal)
        self.Normal_optimizer.step()

        for param in self.Q1.parameters():
            param.requires_grad = True
        for param in self.Q2.parameters():
            param.requires_grad = True
        return metrics

    def update(self, b: Batch, step):

        metrics = dict()

        obs, action, reward, next_obs, discount = utils.to_torch((b.s, b.a, b.r, b.ns, b.d), self.device)

        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        metrics.update(
            self.update_actor(obs, step)
        )

        # ========================================
        # Step 15: update target networks
        # ========================================

        with torch.no_grad():
            self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
            self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

        return metrics

    def act(self, obs: np.array, step, eval_mode) -> np.array:
        # state = torch.tensor(state).unsqueeze(0).float()
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        action, _ = self.sample_action_and_compute_log_pi(obs, use_reparametrization_trick=False)
        return action.cpu().numpy()[0]  # no need to detach first because we are not using the reparametrization trick

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.Normal, self.Normal)
        if self.init_critic:
            utils.hard_update_params(other.Q1, self.Q1)
            utils.hard_update_params(other.Q2, self.Q2)