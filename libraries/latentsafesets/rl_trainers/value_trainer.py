from .trainer import Trainer
from ..utils import plot_utils as pu

import logging
from tqdm import trange
import os

log = logging.getLogger("val train")


class ValueTrainer(Trainer):
    def __init__(self, env, cfg, value, loss_plotter):
        self.cfg = cfg
        self.value = value
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = cfg.env
        self.batch_size = cfg.val_batch_size
        self.ensemble = cfg.val_ensemble
        self.n_models = cfg.val_n_models if cfg.val_ensemble else 0

    def initial_train(self, replay_buffer, update_dir):
        if self.value.trained:
            self.plot(os.path.join(update_dir, "val_start.pdf"), replay_buffer)
            return

        if len(replay_buffer.data['on_policy'].nonzero()[0]) == 0:
            # If no on_policy data available skip init train
            return

        log.info('Beginning value initial optimization')

        for i in range(2 * self.cfg.val_init_iters):
            if i < self.cfg.val_init_iters:
                out_dict = replay_buffer.sample_positive(self.batch_size, 'on_policy', self.n_models)
                obs, rtg = out_dict['obs'], out_dict['rtg']

                loss, info = self.value.update_init(obs, rtg, already_embedded=True)
            else:
                out_dict = replay_buffer.sample_positive(self.batch_size, 'on_policy', self.n_models)
                obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], \
                                           out_dict['reward'], out_dict['done']

                loss, info = self.value.update(obs, rew, next_obs, done, already_embedded=True)
            self.loss_plotter.add_data(info)

            if i % self.cfg.log_freq == 0:
                self.loss_plotter.print(i)
            if i % self.cfg.plot_freq == 0:
                log.info('Creating value function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "val%d.pdf" % i), replay_buffer)
            if i % self.cfg.checkpoint_freq == 0 and i > 0:
                self.value.save(os.path.join(update_dir, 'val_%d.pth' % i))

        self.value.save(os.path.join(update_dir, 'val.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning value update optimization')

        for _ in trange(self.cfg.val_update_iters):
            out_dict = replay_buffer.sample_positive(self.batch_size, 'on_policy', self.n_models)
            obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                       out_dict['done']

            loss, info = self.value.update(obs, rew, next_obs, done, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating value function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "val.pdf"), replay_buffer)
        self.value.save(os.path.join(update_dir, 'val.pth'))

    def plot(self, file, replay_buffer):
        obs = replay_buffer.sample(30)['obs']
        pu.visualize_value(obs, self.value, file=file, env=self.env, obs_type=self.cfg.obs_type)
