import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import pprint
from tqdm import trange
import numpy as np
import hydra
import torch
import wandb
from dm_env import specs

import utils.utils as utils
torch.backends.cudnn.benchmark = True
from utils.logger import Logger
from utils.env_constructor import make, ENV_TYPES
from libraries.latentsafesets.policy import CEMSafeSetPolicy
from libraries.latentsafesets.utils import utils
from libraries.latentsafesets.utils import plot_utils as pu
from libraries.latentsafesets.utils.arg_parser import parse_args
from libraries.latentsafesets.rl_trainers import MPCTrainer
from libraries.safe import SimplePointBot as SPB
from gym.wrappers import FrameStack


def make_env(cfg):
    # create env
    if cfg.obs_type=='pixels':
        env = SPB(from_pixels=cfg.obs_type)
    elif cfg.obs_type=='states':
        env = SPB(from_pixels=cfg.obs_type)
    else:
        print(f'obs_type: {cfg.obs_type} is not valid should be pixels or states')
    if cfg.frame_stack > 1:
        env = FrameStack(env, cfg.frame_stack)
    return env

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.logdir = cfg.log_dir
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print('Training safe set MPC with params...')

        # create logger 
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.env, cfg.obs_type, str(cfg.seed)
            ])
            wandb.init(project='urlb', group=cfg.agent.name, name=exp_name)
        
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create env
        self.train_env = make_env(cfg)

        modules = utils.make_modules(cfg, ss=True, val=True, dyn=True, gi=True, constr=True)

        self.encoder = modules['enc']
        self.value_func = modules['val']
        self.safe_set = modules['ss']
        self.dynamics_model = modules['dyn']
        self.goal_indicator = modules['gi']
        self.constraint_function = modules['constr']

        self.replay_buffer = utils.load_replay_buffer(cfg, self.encoder)
        self.trainer = MPCTrainer(self.train_env, cfg, modules)
        self.trainer.initial_train(self.replay_buffer)
        print('Creating Policy')

        self.policy = CEMSafeSetPolicy(self.train_env, self.encoder, self.safe_set, self.value_func, 
                                       self.dynamics_model, self.constraint_function, self.goal_indicator, cfg)
        
        self.horizon = cfg.horizon

        self.num_updates = cfg.num_updates
        self.traj_per_update = cfg.traj_per_update

    def train(self):
        losses = {}
        avg_rewards = []
        std_rewards = []
        all_rewards = []
        constr_viols = []
        task_succ = []
        n_episodes = 0

        for idx in range(self.num_updates):
            update_dir = os.path.join(self.logdir, 'update_%d' % idx)
            os.makedirs(update_dir)
            update_rewards = []

            # Collect Data
            for idy in range(self.traj_per_update):
                print(f'Collecting trajectory {idy} for update {idx}')
                transitions = []

                obs = np.array(self.train_env.reset())
                self.policy.reset()
                done = False

                # Maintain ground truth info for plotting purposes
                movie_traj = [{'obs': obs.reshape((-1, 3, 64, 64))[0]}]
                traj_rews = []
                constr_viol = False
                succ = False
                for idz in trange(self.horizon):
                    action = self.policy.act(obs / 255)
                    next_obs, reward, done, info = self.train_env.step(action)
                    next_obs = np.array(next_obs)
                    movie_traj.append({'obs': next_obs.reshape((-1, 3, 64, 64))[0]})
                    traj_rews.append(reward)

                    constr = info['constraint']
                    transition = {'obs': obs, 'action': action, 'reward': reward,
                                  'next_obs': next_obs, 'done': done, 
                                  'constraint': constr, 'safe_set': 0, 'on_policy': 1}
                    transitions.append(transition)
                    obs = next_obs
                    constr_viol = constr_viol or info.constraint
                    succ = succ or reward == 0
                    if done: 
                        break
            transitions[-1]['done'] = 1
            traj_reward = sum(traj_rews)

            # self.log.store(EpRet=traj_reward, EpLen=idz+1, EpConstr=float(constr_viol))
            all_rewards.append(traj_rews)
            constr_viols.append(constr_viol)
            task_succ.append(succ)

            pu.make_movie(movie_traj, file=os.path.join(update_dir, 'trajectory%d.gif' % idy))

            # self.log.info('Cost: %d' % traj_reward)

            in_ss = 0
            rtg = 0

            for transition in reversed(transitions):
                if transition['reward'] > -1:
                    in_ss = 1
                transition['safe_set'] = in_ss
                transition['rtg'] = rtg

                rtg = rtg + transition['reward']

            self.replay_buffer.store_transitions(transitions)
            update_rewards.append(traj_reward)

            with self.logger.log_and_dump_ctx(idx, ty='train') as log:
                log('Epoch', idx)
                log('TrainEpisodes', n_episodes)
                log('TestEpisodes', self.traj_per_update)
                log('EpRet')
                log('EpLen', average_only=True)
                log('EpConstr', average_only=True)
                log('ConstrRate', np.mean(constr_viols))
                log('SuccRate', np.mean(task_succ))
            n_episodes += self.traj_per_update

            mean_rew = float(np.mean(update_rewards))
            std_rew = float(np.std(update_rewards))
            avg_rewards.append(mean_rew)
            std_rewards.append(std_rew)

            # self.log.info('Iteration %d average reward: %.4f' % (idx, mean_rew))
            
            pu.simple_plot(avg_rewards, std=std_rewards, title='Average Rewards',
                        file=os.path.join(self.logdir, 'rewards.pdf'),
                        ylabel='Average Reward', xlabel='# Training updates')

            # Update models
            self.trainer.update(self.replay_buffer, idx)
            np.save(os.path.join(self.logdir, 'rewards.npy'), all_rewards)
            np.save(os.path.join(self.logdir, 'constr.npy'), constr_viols)
                        

@hydra.main(config_path='configs/.', config_name='mpc')
def main(cfg):
    from train_mpc import Workspace as W
    workspace = W(cfg)
    workspace.train()


if __name__=='__main__':
    main()
