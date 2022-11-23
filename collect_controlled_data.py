from libraries.latentsafesets.utils.arg_parser import parse_args
from libraries.latentsafesets.utils import utils
from libraries.latentsafesets.utils import plot_utils as pu

from pathlib import Path

import torch
import pprint
import hydra
import logging
import os
import numpy as np

from libraries.safe import SimplePointBot as SPB
from libraries.safe import SimpleVelocityBot as SVB
from libraries.safe import bottleneck_nav as BottleNeck
from libraries.latentsafesets.utils.teacher import ConstraintTeacher, SimplePointBotTeacher, SimpleVelocityBotTeacher, SimpleVelocityBotConstraintTeacher, BottleNeckTeacher, BottleNeckConstraintTeacher
log = logging.getLogger("collect")
from utils.env_constructor import make

ENV = {
    'SimplePointBot' : SPB,
    'SimpleVelocityBot' : SVB,
    'BottleNeck' : BottleNeck
}

ENV_TEACHERS = {
    'SimplePointBot' : [
        SimplePointBotTeacher, ConstraintTeacher
    ],
    'SimpleVelocityBot' : [
        SimpleVelocityBotTeacher, SimpleVelocityBotConstraintTeacher
    ],
    'BottleNeck' : [
        BottleNeckTeacher, BottleNeckConstraintTeacher
    ]
}

DATA_DIRS = {
    'SimplePointBot' : [
        'SimplePointBot', 'SimplePointBot'
    ],
    'SimpleVelocityBot' : [
        'SimpleVelocityBot', 'SimpleVelocityBotConstraint'
    ],
    'BottleNeck' : [
        'BottleNeck', 'BottleNeckConstraints'
    ]
}

DATA_COUNTS = {
    'SimplePointBot' : [
        150, 150
    ],
    'SimpleVeocityBot' : [
        100, 100
    ],
    'BottleNeck' : [
        100, 100
    ]
}


class Workspace:

    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.logdir = cfg.log_dir
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.env = ENV[self.cfg.env]
        if self.cfg.obs_type == 'pixels':
            self.sample_env = self.env(from_pixels=True)
        else:
            self.sample_env = self.env(from_pixels=False)

    def sample_demo_data(self):
        teachers = ENV_TEACHERS[self.cfg.env]
        data_dirs = DATA_DIRS[self.cfg.env]
        data_counts = DATA_COUNTS[self.cfg.env]

        idc = 0
        for teacher, data_dir, count in list(zip(teachers, data_dirs, data_counts)):
            self.generate_teacher_demo_data(data_dir, teacher, count, count_start=idc)
            idc += count

    def generate_teacher_demo_data(self, data_dir, teacher, count, count_start=0, noisy=False):
        demo_dir = os.path.join(self.work_dir, data_dir)
        if not os.path.exists(demo_dir):
            os.makedirs(demo_dir)
        # else:
        #     raise RuntimeError(f'Directory {demo_dir} already exists!')
        teacher = teacher(self.sample_env, noisy=noisy)
        demonstrations = []
        for idc in range(count):
            idc += count_start
            traj = teacher.generate_trajectory()
            reward = sum([frame['reward'] for frame in traj])
            print(f'Trajectory {idc}, Reward {reward}')
            demonstrations.append(traj)
            self.save_trajectory(traj, demo_dir, idc)
            # if idc < 50 and self.logdir is not None:
            #     pu.make_movie(traj, os.path.join(self.logdir, f'{data_dir}_{idc}.gif'))
        return demonstrations

    @staticmethod
    def save_trajectory(traj, demo_dir, idc):
        observation = []
        action = []
        reward = []
        safe_set = []
        constraint = []
        on_policy = []
        rtg = []
        done = []
        for trajectory in traj:
            observation.append(trajectory['obs'])
            action.append(trajectory['action'])
            reward.append(trajectory['reward'])
            safe_set.append(trajectory['safe_set'])
            on_policy.append(trajectory['on_policy'])
            constraint.append(trajectory['constraint'])
            rtg.append(trajectory['rtg'])
            done.append(trajectory['done'])
        file_name = os.path.join(demo_dir, f'episode_{idc}_100')
        np.savez_compressed(file_name, observation=observation, action=action, constraint=constraint, reward=reward, 
                            safe_set=safe_set, on_policy=on_policy, rtg=rtg, done=done)

@hydra.main(config_path='configs/.', config_name='mpc')
def main(cfg):
    from collect_controlled_data import Workspace as W
    workspace = W(cfg)
    workspace.sample_demo_data()

if __name__=='__main__':
    main()
