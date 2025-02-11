import warnings

from agents.unsupervised_learning.smm import SMMAgent
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

torch.backends.cudnn.benchmark = True

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

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        self.env_type = ENV_TYPES[self.cfg.domain]
        
        # create envs
        if self.env_type in ('gym', 'safe'):
            task = self.cfg.domain
        else:
            task = PRIMAL_TASKS[self.cfg.domain]

        self.train_env = make(task, cfg.obs_type, cfg.frame_stack,
                              cfg.action_repeat, cfg.seed, cfg.random_start)
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
                                                  self.work_dir / 'buffer')

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
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

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

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        # Make Reward heatmap's for smm
        if type(self.agent) == SMMAgent and self.cfg.plot:
            
            # p_star
            reward_dir = os.path.join(self.work_dir, str(self.global_episode))
            os.makedirs(reward_dir)
            p_reward_func = lambda x: 100.0 * np.log(self.agent.get_goal_p_star(x))
            self.plot_reward(reward_func=p_reward_func, env=self.eval_env, file=os.path.join(reward_dir, 'p_star.png'), z_prior=False)
            
            # pred_log
            pred_log_func = lambda x: self.agent.state_ent_coef * self.agent.smm.vae.loss(x)[1]
            self.plot_reward(reward_func=pred_log_func, env=self.eval_env, file=os.path.join(reward_dir, 'cond_ent.png'), z_prior=True)
            
            # h_z_s
            def h_z_s_func(x):
                obs = x[0, :2]
                z = x[0, 2:]
                logits = self.agent.smm.predict_logits(obs)
                logits = torch.unsqueeze(logits, 0)
                z = torch.unsqueeze(z, 0)
                h_z_s = self.agent.latent_cond_ent_coef * self.agent.smm.loss(logits, z).unsqueeze(-1)
                return h_z_s
            self.plot_reward(reward_func=h_z_s_func, env=self.eval_env, file=os.path.join(reward_dir, 'h_z_s.png'), z_prior=True)
            
            # h_z
            def h_z_func(z):
                z = z.to('cpu')
                h_z = np.log(self.cfg.skill_dim)  # One-hot z encoding
                # h_z *= torch.ones_like(1).to(self.device)
                h_z = torch.tensor(h_z)
                return self.agent.latent_ent_coef * h_z
                
            # intrinsic_reward
            def intrinsic_reward(x):
                x = x.to(self.device)
                obs = x[:2]
                z = x[2:]
                p_reward = p_reward_func(obs)
                p_reward = torch.tensor(p_reward)
                p_reward = p_reward.to(self.device)
                pred_log_reward = pred_log_func(x)
                h_z_s_reward = h_z_s_func(x)
                h_z_reward = h_z_func(z)
                h_z_reward = h_z_reward.to(self.device)
                # print(f'p_reward.shape: {p_reward.shape}, pred_log_reward.shape: {pred_log_reward.shape}, h_z_s_reward.shape: {h_z_s_reward.shape}, h_z_reward.shape: {h_z_reward.shape}')
                int_reward = -p_reward + pred_log_reward + h_z_s_reward + h_z_reward
                return int_reward

            self.plot_reward(reward_func=intrinsic_reward, env=self.eval_env, file=os.path.join(reward_dir, 'int_reward.png'), z_prior=True)


    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        snapshots = self.cfg.snapshots.copy()
        snapshot = snapshots[0]

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame > snapshot:
                    self.save_snapshot()
                    snapshots = snapshots[1:]
                    snapshot = snapshots[0]
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                # print(f'metrics: {metrics}')
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def plot_reward(self, reward_func, env, file, z_prior=False, plot=True, show=False):
        """ONLY FOR SPB"""
        if z_prior:
            for z in range(self.cfg.skill_dim):
                z_array = np.zeros(self.cfg.skill_dim, dtype=np.float32)
                z_array[z] = 1.0
                data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
                for y in tqdm(range(0, spb.WINDOW_HEIGHT)):
                    for x in range(0, spb.WINDOW_WIDTH):
                        obs = np.array((x, y), dtype=np.float32)
                        obs = np.concatenate((obs, z_array), axis=0)
                        obs = np.expand_dims(obs, axis=0)
                        obs = torch.tensor(obs)
                        obs = obs.to(self.cfg.device)
                        data[y, x] = reward_func(obs)
                if plot:
                    split_file = file.split('.')
                    split_file[-2] = split_file[-2] + str(z)
                    file_name = '.'.join(split_file)
                    env.draw(heatmap=data, file=file_name, show=show)
        else:
            data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
            for y in tqdm(range(0, spb.WINDOW_HEIGHT)):
                for x in range(0, spb.WINDOW_WIDTH):
                    obs = np.array([x, y])
                    obs = np.expand_dims(obs, axis=0)
                    obs = torch.tensor(obs)
                    data[y, x] = reward_func(obs)
            if plot:
                env.draw(heatmap=data, file=file, show=show)

@hydra.main(config_path='configs/.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
