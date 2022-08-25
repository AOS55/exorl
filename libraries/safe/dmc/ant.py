"""Ant domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from libraries.safe.dmc import obstacles

from dm_control.utils import containers
from dm_control import composer
from dm_control.locomotion.tasks import go_to_target as gtt
from dm_control.locomotion.walkers import Ant

import numpy as np

SUITE = containers.TaggedTasks()

_CONTROL_TIMESTEP = .02

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env

@SUITE.add('benchmarking')
def navigate(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Navigate to a goal position"""
    walker = Ant()
    arena = obstacles.Obstacle()
    task = gtt.GoToTarget(walker=walker,
                          arena=arena,
                          target_relative_dist=5.0,
                          target_relative=False,
                          walker_spawn_position=(-7.0, 7.0),
                          target_spawn_position=(7.0, 0.0))
    task._target.rgba=(0.0, 1.0, 0.0, 1.0)
    env = composer.Environment(
        task=task,
        time_limit=time_limit,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True
    )
    return env
