import warnings
from collections import OrderedDict

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import shutil
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
        self.sample_env = make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed, random_start=True)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.sample_env.observation_spec(),
                                self.sample_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pre-trained
        if cfg.snapshot_ts > 0:
            print(f'snapshot is: {self.load_snapshot()}')
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        self.prior_encoded_agents = ['aps', 'diayn', 'smm']

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        if cfg.data_type == 'unsupervised':
            constraint_spec = specs.Array((1,), bool, 'constraint')
            done_spec = specs.Array((1,), bool, 'done')
            if cfg.agent.name in self.prior_encoded_agents:
                meta_specs = (meta_specs[0], constraint_spec, done_spec)
                self.meta_encoded = True
            else:
                meta_specs = (constraint_spec, done_spec)
                self.meta_encoded = False

        # create replay buffer
        data_specs = (self.sample_env.observation_spec(),
                      self.sample_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                True, cfg.nstep, cfg.discount)

        meta_specs = self.agent.get_meta_specs()

        self._replay_iter = None

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
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        sample_until_step = utils.Until(self.cfg.num_sample_episodes)
        prioritize_sample_until_step = utils.Until(self.cfg.num_prioritize_sample_episodes)
        step, episode, total_reward = 0, 0, 0
        time_step = self.sample_env.reset()
        meta = self.agent.init_meta()

        self.replay_storage.add(time_step, meta)
        while sample_until_step(episode):

            # Random Start
            time_step = self.sample_env.reset()
            if self.cfg.agent.name not in self.prior_encoded_agents:
                # Update agent if not in the reward 
                meta = self.agent.update_meta(meta, self.global_step, time_step)
                self.sample_env._env._env._env._env.environment.reset(random_start=False)
            else:
                meta = self.agent.init_meta()
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
                self._global_step += 1
                
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
            if not seed_until_step(self.global_step):
                if self.cfg.agent.name not in self.prior_encoded_agents:
                    batch = next(self.replay_iter)
                    algo_batch = batch[0:5]
                    algo_iter = iter([algo_batch])
                    self.agent.update(algo_iter, self.global_step)
            if self.cfg.save_video:
                self.video_recorder.save(f'{episode}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        # Store data in values
        buffer_path = os.path.join(self.work_dir, 'buffer')
        os.rename(buffer_path, f'{self.cfg.agent.name}_{self.cfg.snapshot_ts}')
        domain, _ = self.cfg.task.split('_', 1)
        source_path = os.path.join(self.work_dir, f'{self.cfg.agent.name}_{self.cfg.snapshot_ts}')

        # Sample from skill
        norm_skill_reward = np.array(self.skill_reward_sum(source_path))
        norm_skill_constraint = np.array(self.skill_constraint_sum(source_path))
        print(f'reward: {norm_skill_reward}')
        print(f'constraint: {norm_skill_constraint}')

        reward_skills = np.where(norm_skill_reward > -100.0)[0]
        constraint_skills = np.where(norm_skill_constraint > 0.0)[0]
        print(f'reward_skills: {reward_skills}')
        print(f'constraint_skills: {constraint_skills}')
        
        """
        Sample from reward priors
        """
        os.makedirs(os.path.join(self.work_dir, 'buffer'))
        step, episode, total_reward = 0, 0, 0
        while prioritize_sample_until_step(episode):
            time_step = self.sample_env.reset()
            skill = np.zeros(self.cfg.skill_dim, dtype=np.float32)
            skill[np.random.choice(reward_skills)] = 1.0
            meta = OrderedDict()
            meta['skill'] = skill
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
                self._global_step += 1
                
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
            if not seed_until_step(self.global_step):
                if self.cfg.agent.name not in self.prior_encoded_agents:
                    batch = next(self.replay_iter)
                    algo_batch = batch[0:5]
                    algo_iter = iter([algo_batch])
                    self.agent.update(algo_iter, self.global_step)
            if self.cfg.save_video:
                self.video_recorder.save(f'{episode}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        # Store data in values
        buffer_path = os.path.join(self.work_dir, 'buffer')
        os.rename(buffer_path, f'reward_{self.cfg.agent.name}_{self.cfg.snapshot_ts}')
        domain, _ = self.cfg.task.split('_', 1)
        source_path = os.path.join(self.work_dir, f'{self.cfg.agent.name}_{self.cfg.snapshot_ts}')
        
        """
        Sample from Constraint Priors
        """

        os.makedirs(os.path.join(self.work_dir, 'buffer'))
        step, episode, total_reward = 0, 0, 0
        while prioritize_sample_until_step(episode):
            # for constraints randomly sample to get a better idea of how each works
            time_step = self.sample_env.reset()
            skill = np.zeros(self.cfg.skill_dim, dtype=np.float32)
            skill[np.random.choice(constraint_skills)] = 1.0
            meta = OrderedDict()
            meta['skill'] = skill
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
                self._global_step += 1
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
            if not seed_until_step(self.global_step):
                if self.cfg.agent.name not in self.prior_encoded_agents:
                    batch = next(self.replay_iter)
                    algo_batch = batch[0:5]
                    algo_iter = iter([algo_batch])
                    self.agent.update(algo_iter, self.global_step)
            if self.cfg.save_video:
                self.video_recorder.save(f'{episode}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        # Store data in values
        buffer_path = os.path.join(self.work_dir, 'buffer')
        os.rename(buffer_path, f'constraint_{self.cfg.agent.name}_{self.cfg.snapshot_ts}')
        domain, _ = self.cfg.task.split('_', 1)
        source_path = os.path.join(self.work_dir, f'{self.cfg.agent.name}_{self.cfg.snapshot_ts}')

    def skill_constraint_sum(self, source_path):
        files = os.listdir(source_path)
        skill_nums = [x for x in range(self.cfg.skill_dim)]
        skill_sum = [0 for _ in range(self.cfg.skill_dim)]
        skill_count = [0 for _ in range(self.cfg.skill_dim)]
        for file in files:
            path = os.path.join(source_path, file)
            ep = np.load(path)
            skill = np.where(ep['skill'][0] == 1)
            constraint = np.sum(ep['constraint'])
            skill_sum[skill[0][0]] += constraint
            skill_count[skill[0][0]] += 1

        def _divide(sum, count):
            try:
                return sum/count
            except ZeroDivisionError:
                return 0

        return [_divide(sum, count) for (sum, count) in zip(skill_sum, skill_count)]

    def skill_reward_sum(self, source_path):
        files = os.listdir(source_path)
        skill_nums = [x for x in range(self.cfg.skill_dim)]
        skill_sum = [0 for _ in range(self.cfg.skill_dim)]
        skill_count = [0 for _ in range(self.cfg.skill_dim)]
        for file in files:
            path = os.path.join(source_path, file)
            ep = np.load(path)
            skill = np.where(ep['skill'][0] == 1)
            reward = np.sum(ep['reward'])
            skill_sum[skill[0][0]] += reward
            skill_count[skill[0][0]] += 1

        def _divide(sum, count):
            try:
                return sum/count
            except ZeroDivisionError:
                return -100

        return [_divide(sum, count) for (sum, count) in zip(skill_sum, skill_count)]
    
    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            if self.cfg.agent.name == 'diayn':
                snapshot = snapshot_dir / f'{self.cfg.skill_dim}' / str(seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            else:
                snapshot = snapshot_dir / str(seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
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

@hydra.main(config_path='configs/.', config_name='prioritized_sampling')
def main(cfg):
    from prioritized_sampling import Workspace as W
    workspace = W(cfg)
    workspace.sample()

if __name__=='__main__':
    main()
