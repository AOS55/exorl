import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Callable
from random_words import RandomWords
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import json
import shutil

def is_free(pos, terrain, objects, valid):
    """
    Check if the planned pose will take the aircraft into a position with valid terrain

    :param pos: Aircraft 3D position
    :param terrain: The 3D ground terrain map
    :param objects: The possible objects on the ground the aircraft could collide with
    :param valid: tuple of valid object types
    :return:
    """
    if not (0 <= pos[0] < terrain.shape[0]):
        return False
    if not (0 <= pos[1] < terrain.shape[1]):
        return False
    if not (0 <= pos[2] < terrain.shape[2]):
        return False
    if terrain[pos] not in valid:
        return False
    if not objects.free(pos):
        return False
    return True


def plot_2d(data: np.array) -> None:
    """
    Make a simple 2d plot of some data, initially made to look at noise functions

    :param data: 2d data in np.array
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot = ax.pcolormesh(data)
    fig.colorbar(plot)
    plt.show()


def point_in_hull(points, hull, tol=1e-12):
    """
    Given a point series of points calculates whether the point is contained within the hull object

    :param points: single or many points to query in convex hull
    :param hull: the convex hull itself
    :param tol: the tolerance to apply at the edge of the convex hull
    :return:
    """
    return all(
        (np.dot(eq[:-1], points) + eq[-1] <= tol)
        for eq in hull.equations)


class CircularQueue:

    def __init__(self, max_size: int):
        self.queue = list()
        self.head = 0
        self.tail = 0
        self.max_size = max_size

    def enqueue(self, data):
        if self.size() == self.max_size - 1:
            return 'Queue Full!'
        self.queue.append(data)
        self.tail = (self.tail + 1) % self.max_size

    def dequeue(self):
        if self.size() == 0:
            return 'Queue Empty!'
        data = self.queue[self.head]
        self.head = (self.head + 1) % self.max_size
        return data

    def size(self):
        if self.tail >= self.head:
            return self.tail - self.head
        return self.max_size - (self.head - self.tail)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule (from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def initialize_tensorboards(run_name: str, env_type: str = 'flyer'):
    """
    Setup the tensorboard writer and the tensorboard run directory

    :param run_name: the name of the directory to store the tensorboard
    :param env_name: the type of environment used in the tensorboard      
    :return: the tensorboard summary writer object & path to the tb file
    """

    path = os.path.dirname(__file__)
    path = os.path.join(path, 'runs')  # go to runs dir
    path = os.path.join(path, env_type)  # go to environment dir    
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, 'tensor_boards')
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, run_name)  # go to specific model dir
    ver = 0  # version number
    path_tmp = path + 'ver' + str(ver)
    while os.path.exists(path_tmp):
        ver += 1
        path_tmp = path + 'ver' + str(ver)
    path = path_tmp
    if not os.path.isdir(path):
        os.mkdir(path)
    tb_writer = SummaryWriter(path)
    return tb_writer, path


def move_tb_file(current_path: str, run_name: str, env_type: str = 'flyer'):
    """
    Move a tensorboard file to runs dir
    
    :param current_path: current path to the tb file
    :param run_name: name of the directory to store the tensorboard file
    :param env_type: the type of environment used in the tensorboard      
    :return: None
    """
    path = os.path.dirname(__file__)
    path = os.path.join(path, 'runs') 
    path = os.path.join(path, env_type)
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, 'tensor_boards')
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, run_name)
    ver = 0 # version number
    path_tmp = path + 'ver' + str(ver)
    while os.path.exists(path_tmp):
        ver += 1
        path_tmp = path + 'ver' + str(ver)
    path = path_tmp
    if not os.path.isdir(path):
        os.mkdir(path)
    shutil.move(current_path, path)


def word_gen() -> str:
    """
    Recursive implementation of word_gen to ensure a NoneType is not returned

    :return: word 
    """
    rw = RandomWords()
    word = rw.random_word()
    if word is None:
        word_gen()
    else: 
        return word


def update_policy_json(policy_path, name: str, env_type: str = 'flyer'):
    """
    Update the checkpoint in update_policy_json

    :param policy_path: the policy network
    :param name: the name of the policy to update
    :param env_type: the type of environment used (flyer is default)
    :return: None
    """

    path = os.path.dirname(__file__)
    path = os.path.join(path, 'runs')  # go to runs dir
    path = os.path.join(path, env_type)  # go to environment dir
    json_path = os.path.join(path, 'json')
    # policy_path = os.path.join(path, 'policies')

    # policy_name = name
    # ver = 0
    # while os.path.exists(os.path.join(policy_path, policy_name)):
    #     ver += 1
    # policy_name = policy_name + '-' + str(ver)
    # policy_path = os.path.join(policy_path, policy_name + '.pt')
    # torch.save(model.state_dict(), policy_path)

    json_path = os.path.join(json_path, name + '.json')
    with open(json_path, 'r+') as json_file:
        json_data = json.load(json_file)
        json_data['policy_checkpoint_path'] = policy_path
        json_file.seek(0)
        json.dump(json_data, json_file, indent=4, sort_keys=False)
        json_file.truncate()
