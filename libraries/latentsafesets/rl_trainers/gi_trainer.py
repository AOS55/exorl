from .trainer import Trainer
from ..utils import plot_utils as pu
from ..utils import spb_utils as spbu

import logging
from tqdm import trange
import os

log = logging.getLogger("gi train")


class GoalIndicatorTrainer(Trainer):
    def __init__(self, env, cfg, gi, loss_plotter):
        self.cfg = cfg
        self.gi = gi
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = cfg.env

    def initial_train(self, replay_buffer, update_dir):
        if self.gi.trained:
            # self.plot(os.path.join(update_dir, "gi_start.pdf"), replay_buffer)
            return

        log.info('Beginning goal indicator initial optimization')

        for i in range(self.cfg.gi_init_iters):
            out_dict = replay_buffer.sample(self.cfg.gi_batch_size)
            next_obs, rew = out_dict['next_obs'], out_dict['reward']

            loss, info = self.gi.update(next_obs, rew, already_embedded=True)
            self.loss_plotter.add_data(info)

            if i % self.cfg.log_freq == 0:
                self.loss_plotter.print(i)
            if i % self.cfg.plot_freq == 0:
                log.info('Creating goal indicator function heatmap')
                # self.loss_plotter.plot()
                # self.plot(os.path.join(update_dir, "gi%d.pdf" % i), replay_buffer)
            if i % self.cfg.checkpoint_freq == 0 and i > 0:
                self.gi.save(os.path.join(update_dir, 'gi_%d.pth' % i))

        # spbu.evaluate_constraint_func(self.gi, file=os.path.join(update_dir, "gi_init.pdf"))
        self.gi.save(os.path.join(update_dir, 'gi.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning goal indicator update optimization')

        for _ in trange(self.cfg.gi_update_iters):
            out_dict = replay_buffer.sample(self.cfg.gi_batch_size)
            next_obs, rew = out_dict['next_obs'], out_dict['reward']

            loss, info = self.gi.update(next_obs, rew, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info("Creating goal indicator function heatmap")
        self.loss_plotter.plot()
        # self.plot(os.path.join(update_dir, "gi.pdf"), replay_buffer)
        self.gi.save(os.path.join(update_dir, "gi.pth"))

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.cfg.constr_batch_size)
        next_obs = out_dict['next_obs']
        pu.visualize_onezero(next_obs, self.gi,
                             file,
                             env=self.env)
