import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

from utils.env_constructor import make
import utils.utils as utils
from utils.logger import Logger
from utils.replay_buffer import ReplayBufferStorage, make_replay_loader
from utils.video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.work_dir, 
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        # create env
        self.train_env = make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.sample_env = make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pre-trained
        if cfg.snapshot_ts > 0:
            print(f'snapshot is: {self.load_snapshot()}')
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        if cfg.data_type == 'unsupervised':
            constraint_spec = specs.Array((1,), bool, 'constraint')
            done_spec = specs.Array((1,), bool, 'done')
            if cfg.agent.name in ['aps', 'diayn', 'smm']:
                meta_specs = (meta_specs[0], constraint_spec, done_spec)
                self.meta_encoded = True
            else:
                meta_specs = (constraint_spec, done_spec)
                self.meta_encoded = False

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
    
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def sample(self):
        sample_until_step =  utils.Until(self.cfg.num_sample_episodes)
        step, episode, total_reward = 0, 0, 0
        meta = self.agent.init_meta()

        while sample_until_step(episode):
            meta = self.agent.init_meta()
            time_step = self.sample_env.reset()
            self.replay_storage.add(time_step, meta)
            self.video_recorder.init(self.sample_env, enabled=True)
            trajectory = []
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.sample_env.step(action)
                self.video_recorder.record(self.sample_env)
                total_reward += time_step.reward
                trajectory.append(time_step)
                step += 1
                
                if self.cfg.data_type == 'unsupervised':
                    # TODO: Provide a less hacky way of accessing info from environment
                    info = self.sample_env._env._env._env._env.get_info()
                    if self.meta_encoded:
                        unsupervised_data = {'meta': meta, 'constraint': info['constraint'], 'done': info['done']}
                    else:
                        unsupervised_data = {'constraint': info['constraint'], 'done': info['done']}
                    self.replay_storage.add(time_step, unsupervised_data)
                else:
                    self.replay_storage.add(time_step, meta)

            episode += 1
            # skill_index = str(meta['skill'])
            self.video_recorder.save(f'{episode}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        # Store data in values
        buffer_path = os.path.join(self.work_dir, 'buffer')
        os.rename(buffer_path, f'{self.cfg.agent.name}_{self.cfg.snapshot_ts}')
    
    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / f'{self.cfg.skill_dim}' / str(seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            import os
            print(f'current dir is: {os.getcwd()}')
            print(f'snapshot file location is: {snapshot}')
            print(f'snapshot exists: {snapshot.exists()}')
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                print(f'f is: {f}')
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None

@hydra.main(config_path='configs/.', config_name='sampling')
def main(cfg):
    from sampling import Workspace as W
    workspace = W(cfg)
    workspace.sample()

if __name__=='__main__':
    main()
