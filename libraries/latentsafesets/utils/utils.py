import torch
import torch.nn as nn
import numpy as np

import logging
import os
import json
from datetime import datetime
import random
from tqdm import tqdm, trange

from ..utils.replay_buffer_encoded import EncodedReplayBuffer
from ..utils.replay_buffer import ReplayBuffer
from gym.wrappers import FrameStack

log = logging.getLogger("utils")


files = {
    'spb': [
        'SimplePointBot', 'SimplePointBotConstraints'
    ],
    'svb': [
        'SimpleVelocityBot', 'SimpleVelocityBotConstraints'
    ],
    'bottleneck': [
        'BottleNeck', 'BottleNeckConstraints'
    ],
    'drone': [
        'DroneGate', 'DroneGateConstraints'
    ],
    'apb': [
        'AccelerationPointBot', 'AccelerationPointBotConstraint'
    ],
    'reacher': [
        'Reacher', 'ReacherConstraints', 'ReacherInteractions'
    ]
}


def seed(seed):
    # torch.set_deterministic(True)
    if seed == -1:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_file_prefix(exper_name=None, seed=-1):
    if exper_name is not None:
        folder = os.path.join('outputs', exper_name)
    else:
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d/%H-%M-%S")
        folder = os.path.join('outputs', date_string)
    if seed != -1:
        folder = os.path.join(folder, str(seed))
    return folder


def init_logging(folder, file_level=logging.INFO, console_level=logging.DEBUG):
    # set up logging to file
    logging.basicConfig(level=file_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=os.path.join(folder, 'log.txt'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(console_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def save_trajectories(trajectories, file):
    if not os.path.exists(file):
        os.makedirs(file)
    else:
        raise RuntimeError("Directory %s already exists." % file)

    for i, traj in enumerate(trajectories):
        save_trajectory(traj, file, i)


def save_trajectory(trajectory, file, n):
    im_fields = ('obs', 'next_obs')
    for field in im_fields:
        if field in trajectory[0]:
            dat = np.array([frame[field] for frame in trajectory], dtype=np.uint8)
            np.save(os.path.join(file, "%d_%s.npy" % (n, field)), dat)
    traj_no_ims = [{key: frame[key] for key in frame if key not in im_fields}
                   for frame in trajectory]
    with open(os.path.join(file, "%d.json" % n), "w") as f:
        json.dump(traj_no_ims, f)


def load_trajectories(num_traj, file):
    log.info('Loading trajectories from %s' % file)
    if not os.path.exists(file):
        raise RuntimeError("Could not find directory %s." % file)
    trajectories = []
    iterator = range(num_traj) if num_traj <= 200 else trange(num_traj)
    for i in iterator:
        if not os.path.exists(os.path.join(file, 'episode_%d_100.npz' % i)):
            print(f"path: {os.path.join(file, 'episode_%d_100.npz' % i)}")
            log.info('Could not find %d' % i)
            continue
        trajectory = np.load(os.path.join(file, 'episode_%d_100.npz' % i))
        data = {}
        for key in trajectory.files:
            data[key] = trajectory[key]
        trajectories.append(data)
    return trajectories


def transform_dict(trajectories):
    # Covert dictionary of lists to list of dictionaries
    dict_keys = list(trajectories[0].keys())
    if 'skill' in dict_keys:
        dict_keys.remove('skill')
    new_trajectories = []
    for trajectory in trajectories:
        new_trajectory = []
        for idx in range(len(trajectory[dict_keys[0]])-1):
            new_dict = {}
            for key in dict_keys:
                if key == 'observation':
                    new_dict['obs'] = trajectory[key][idx]
                else:
                    new_dict[key] = trajectory[key][idx]
            new_dict['next_obs'] = trajectory['observation'][idx]
            if 1 in trajectory['reward']:
                new_dict['ss'] = 1
            else:
                new_dict['ss'] = 0
            new_dict['on_policy'] = 0
            new_trajectory.append(new_dict)
        new_trajectories.append(new_trajectory)
    return new_trajectories

def load_replay_buffer(cfg, encoder=None, first_only=False):
    log.info('Loading data')
    trajectories = []
    data_dirs = [cfg.data_dirs]
    data_counts = [cfg.data_counts]
    for directory, num in list(zip(data_dirs, data_counts)):
        print(f'directory is: {directory}')
        real_dir = os.path.join('../../../data', directory)
        trajectories += load_trajectories(num, file=real_dir)
        trajectories = transform_dict(trajectories)
        if first_only:
            print('wahoo')
            break

    log.info('Populating replay buffer')

    # Shuffle array so that when the replay fills up it doesn't remove one dataset before the other
    random.shuffle(trajectories)
    if encoder is not None:
        replay_buffer = EncodedReplayBuffer(encoder, cfg.buffer_size)
    else:
        replay_buffer = ReplayBuffer(cfg.buffer_size)

    for trajectory in tqdm(trajectories):
        replay_buffer.store_transitions(trajectory)

    return replay_buffer


def make_env(params, monitoring=False):
    from latentsafesets.envs import SimplePointBot, SimpleVelocityBot, SimpleVideoSaver, BottleNeck
    env_name = params['env']
    if env_name == 'spb':
        env = SimplePointBot(True)
    elif env_name == 'svb':
        env = SimpleVelocityBot(True)
    elif env_name == 'bottleneck':
        env = BottleNeck(True)
    elif env_name == 'reacher':
        import dmc2gym
        env = dmc2gym.make(domain_name='reacher', task_name='hard', seed=params['seed'],
                           from_pixels=True, visualize_reward=False, channels_first=True)
    else:
        raise NotImplementedError

    if params['frame_stack'] > 1:
        env = FrameStack(env, params['frame_stack'])

    if monitoring:
        env = SimpleVideoSaver(env, os.path.join(params['logdir'], 'videos'))

    return env


def make_modules(cfg, ss=False, val=False, dyn=False, gi=False, constr=False):
    from ..modules import VanillaVAE, ValueEnsemble, \
        ValueFunction, PETSDynamics, GoalIndicator, ConstraintEstimator, BCSafeSet, \
        BellmanSafeSet
    from ..utils import pytorch_utils as ptu

    modules = {}

    encoder = VanillaVAE(cfg)
    if cfg.enc_checkpoint:
        encoder.load(cfg.enc_checkpoint)
    modules['enc'] = encoder

    if ss:
        safe_set_type = cfg.safe_set_type
        if safe_set_type == 'bc':
            safe_set = BCSafeSet(encoder, cfg)
        elif safe_set_type == 'bellman':
            safe_set = BellmanSafeSet(encoder, cfg)
        else:
            raise NotImplementedError
        if cfg.safe_set_checkpoint:
            safe_set.load(cfg.safe_set_checkpoint)
        modules['ss'] = safe_set

    if val:
        if cfg.val_ensemble:
            value_func = ValueEnsemble(encoder, cfg).to(ptu.TORCH_DEVICE)
        else:
            value_func = ValueFunction(encoder, cfg).to(ptu.TORCH_DEVICE)
        if cfg.val_checkpoint:
            value_func.load(cfg.val_checkpoint)
        modules['val'] = value_func

    if dyn:
        dynamics = PETSDynamics(encoder, cfg)
        if cfg.dyn_checkpoint:
            dynamics.load(cfg.dyn_checkpoint)
        modules['dyn'] = dynamics

    if gi:
        goal_indicator = GoalIndicator(encoder, cfg).to(ptu.TORCH_DEVICE)
        if cfg.gi_checkpoint:
            goal_indicator.load(cfg.gi_checkpoint)
        modules['gi'] = goal_indicator

    if constr:
        constraint = ConstraintEstimator(encoder, cfg).to(ptu.TORCH_DEVICE)
        if cfg.constr_checkpoint:
            constraint.load(cfg.constr_checkpoint)
        modules['constr'] = constraint

    return modules


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        super(RunningMeanStd, self).__init__()

        from ..utils.pytorch_utils import TORCH_DEVICE

        # We store these as parameters so they'll be stored in dynamic model state dicts
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float32, device=TORCH_DEVICE),
                                 requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float32, device=TORCH_DEVICE),
                                requires_grad=False)
        self.count = nn.Parameter(torch.tensor(epsilon))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = nn.Parameter(new_mean, requires_grad=False)
        self.var = nn.Parameter(new_var, requires_grad=False)
        self.count = nn.Parameter(new_count, requires_grad=False)

def prioritized_replay_store(replay_buffer, transitions, prob_accept=0.01):
    reward_sum = 0
    for transition in transitions:
        reward_sum += transition['reward']
    
    # if no reward attained, balance acceptance case
    if reward_sum == -1.0 * len(transitions): 
        if np.random.uniform(low=0.0, high=1.0) <= prob_accept:
            replay_buffer.store_transitions(transitions)
        else:
            print(f'episode not stored did not reach goal')

