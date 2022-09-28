import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
import os
import pprint
import hydra
import torch

from pathlib import Path


from libraries.latentsafesets.rl_trainers import VAETrainer
import libraries.latentsafesets.utils as utils
from utils import set_seed_everywhere
from libraries.latentsafesets.utils import LossPlotter, EncoderDataLoader
from libraries.latentsafesets.utils import encoder_data_loader
from libraries.latentsafesets.utils.arg_parser import parse_args

from utils.logger import Logger

log = logging.getLogger("main")


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()

        if cfg.frame_stack == 1:
            cfg.d_obs = (3, 64, 64)
        else:
            cfg.d_obs = (cfg.frame_stack, 3, 64, 64)

        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        
        self.encoder_data_loader = EncoderDataLoader(cfg.env, cfg.frame_stack)
        modules = utils.make_modules(cfg)
        self.encoder = modules['enc']

        self.loss_plotter = LossPlotter(self.work_dir)

    def train(self):
        trainer = VAETrainer(self.cfg, self.encoder, self.loss_plotter)
        trainer.initial_train(self.encoder_data_loader, self.work_dir, force_train=True)


@hydra.main(config_path='configs/.', config_name='encoder')
def main(cfg):
    from train_encoder import Workspace as W
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
