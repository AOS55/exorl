from .trainer import Trainer
from ..utils import plot_utils as pu

import logging
from tqdm import trange
import os

log = logging.getLogger("constr train")


class ConstraintTrainer(Trainer):
    def __init__(self, env, cfg, constr, loss_plotter):
        self.cfg = cfg
        self.constr = constr
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = cfg.env

    def initial_train(self, replay_buffer, update_dir):
        if self.constr.trained:
            self.plot(os.path.join(update_dir, "constr_start.pdf"), replay_buffer)
            return

        log.info('Beginning constraint initial optimization')

        for i in range(self.cfg.constr_init_iters):
            out_dict = replay_buffer.sample(self.cfg.constr_batch_size)
            next_obs, constr = out_dict['next_obs'], out_dict['constraint']

            loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            self.loss_plotter.add_data(info)

            if i % self.cfg.log_freq == 0:
                self.loss_plotter.print(i)
            if i % self.cfg.plot_freq == 0:
                log.info('Creating constraint function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "constr%d.pdf" % i), replay_buffer)
            if i % self.cfg.checkpoint_freq == 0 and i > 0:
                self.constr.save(os.path.join(update_dir, 'constr_%d.pth' % i))

        self.constr.save(os.path.join(update_dir, 'constr.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning constraint update optimization')

        for _ in trange(self.cfg.constr_update_iters):
            out_dict = replay_buffer.sample(self.cfg.constr_batch_size)
            next_obs, constr = out_dict['next_obs'], out_dict['constraint']

            loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating constraint function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "constr.pdf"), replay_buffer)
        self.constr.save(os.path.join(update_dir, 'constr.pth'))

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.cfg.constr_batch_size)
        next_obs = out_dict['next_obs']
        pu.visualize_onezero(next_obs, self.constr,
                             file,
                             env=self.env, obs_type=self.cfg.obs_type)
