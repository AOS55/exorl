import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from dm_env import specs
from pathlib import Path

from utils.env_constructor import make, ENV_TYPES
import utils.utils as utils
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer, Transition
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
        self.log_dir = cfg.log_dir
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urls", group=cfg.agent.name, name=exp_name, config=OmegaConf.to_container(cfg, resolve=True))

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

        self.goal = [150, 75]

        WINDOW_WIDTH = 180
        WINDOW_HEIGHT = 150

        walls = [[[75, 55], [100, 95]]]

        def _normalize(obs):
            obs[0] = (obs[0] - WINDOW_WIDTH/2) / (WINDOW_WIDTH/2)
            obs[1] = (obs[1] - WINDOW_HEIGHT/2) / (WINDOW_HEIGHT/2)
            return obs

        def _complex_obstacle(bounds):
            """
            Returns a function that returns true if a given state is within the
            bounds and false otherwise
            :param bounds: bounds in form [[X_min, Y_min], [X_max, Y_max]]
            :return: function described above
            """
            min_x, min_y = _normalize(bounds[0])
            max_x, max_y = _normalize(bounds[1])

            def obstacle(state):
                if type(state) == np.ndarray:
                    lower = (min_x, min_y)
                    upper = (max_x, max_y)
                    state = np.array(state)
                    component_viol = (state > lower) * (state < upper)
                    return np.product(component_viol, axis=-1)
                if type(state) == torch.Tensor:
                    lower = torch.from_numpy(np.array((min_x, min_y)))
                    upper = torch.from_numpy(np.array((max_x, max_y)))
                    component_viol = (state > lower) * (state < upper)
                    return torch.prod(component_viol, dim=-1)

            return obstacle
        
        self.walls = [_complex_obstacle(wall) for wall in walls]
        
        self.goal = tuple(_normalize(self.goal))

        self.replay_buffer = ReplayBuffer(capacity=int(cfg.replay_buffer_size))

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self._timer = utils.Timer()
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

    def get_goal_p_star(self, agent_pos):
        x_dist = agent_pos[0] - self.goal[0]
        y_dist = agent_pos[1] - self.goal[1]
        # x_dist = x_dist.cpu().detach().numpy()
        # y_dist = y_dist.cpu().detach().numpy()
        dist = np.linalg.norm((x_dist, y_dist), axis=0)
        # def _prior_distro(dist):
        #     if dist > 1.0:
        #         p_star = -np.log(dist)
        #     else:
        #         p_star = 1.0
        #     return p_star
        # p_star = _prior_distro(dist)
        p_star = -1.0 * dist
        constr = any([wall(agent_pos) for wall in self.walls])
        # add penalty for hitting wall
        if constr:
            p_star -= 5
        # p_star = np.array(list(map(_prior_distro, dist)), dtype=np.float32)
        return p_star

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode==0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, self.global_step, eval_mode=True)
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    if self.cfg.domain == 'SimplePointBot' or self.cfg.domain == "SimpleVelocityBot":
                        reward = self.get_goal_p_star(time_step.observation)
                    else:
                        reward = time_step.reward
                    total_reward += reward
                    step += 1
            
            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")
            
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / episode)
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)

            step, total_reward = 0, 0

    
    def train(self):

        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        obs = time_step.observation
        done = False
        metrics = None
        while train_until_step(self.global_step):

            if time_step.last():
                self._global_step += 1

                if metrics is not None:
                    elapsed_time, total_time = self._timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    fps = episode_frame / elapsed_time
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', fps)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer.memory))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                episode_step, episode_reward = 0, 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)

            # try to update the agent
            if self.replay_buffer.ready_for(self.cfg.batch_size):
                metrics = self.agent.update(self.replay_buffer.sample(self.cfg.batch_size), self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)

            if time_step.last():
                done = True
            else:
                done = False
            if self.cfg.domain == "SimplePointBot" or self.cfg.domain == "SimpleVelocityBot":
                reward = self.get_goal_p_star(time_step.observation)
            else:
                reward = time_step.reward
            next_obs = time_step.observation
            # store to replay buffer
            transition = Transition(obs, action, reward, next_obs, done)
            self.replay_buffer.push(transition)
            obs = next_obs
            episode_step += 1
            episode_reward += reward
            self._global_step += 1

    def run(self):
        self.train()

    @hydra.main(version_base="1.2", config_path="configs/.", config_name="sac_train")
    def main(cfg):
        from train_sac import Workspace as W
        workspace = W(cfg)
        workspace.run()

    if __name__=="__main__":
        main()
