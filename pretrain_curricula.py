import warnings

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import agents
import numpy as np
import torch
import wandb
from dm_env import specs

from utils.env_constructor import make, ENV_TYPES
import utils.utils as utils
from utils.logger import Logger
from utils.replay_buffer import ReplayBufferStorage, make_replay_loader
from utils.video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark

from libraries.dmc.dmc_tasks import PRIMAL_TASKS

from tqdm import tqdm
import libraries.safe.simple_point_bot as spb

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type, str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name,
                       name=exp_name)

        self.logger = Logger(self.work_dir, 
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        self.env_type = ENV_TYPES[self.cfg.domain]

        # create envs
        if self.env_type in 'open_ended':
            task = self.cfg.domain
        else:
            print(f'Only open ended tasks allowed')


        # Need to edit the make function here for custom env config
        self.train_env = make(task, cfg.obs_type, cfg.frame_stack,
                              cfg.action_repeat, cfg.seed, cfg.random_start)

        # Same for eval env
        self.eval_env = make(task, cfg.obs_type, cfg.frame_stack,
                             cfg.action_repeat, cfg.seed, cfg.random_start)

        print(f"obs_type: {cfg.obs_type}")
        print(f"obs_spec: {self.train_env.observation_spec()}")
        print(f"action_spec: {self.train_env.action_spec()}")
        print(f"num_expl_steps: {cfg.num_seed_frames // cfg.action_repeat}")
        print(f"agent: {cfg.agent}")

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir /
                                                  'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camear_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            cameara_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.gloabl_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    # evaluate training procedure
    def eval(self):
        step, episode, total_reward = 0, 0, 0


    # training procedure itself
    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        snapsh