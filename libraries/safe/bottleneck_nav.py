"""
Gym environment box with goal state on opposite side
"""
# !/usr/bin/env python3

import numpy as np
from typing import Optional, List, Tuple

from gym import Env
from gym import utils
from gym.spaces import Box

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageDraw
from skimage.transform import resize

metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

WINDOW_WIDTH = 180
WINDOW_HEIGHT = 150

MAX_FORCE = 3


class BottleNeck(Env, utils.EzPickle):
    """
    Environment to test navigation where action is safety critical in specific scenarios
    """

    def __init__(self,
                 from_pixels: bool = True,
                 box_bounds: dict = None,
                 start_pos: tuple = None,
                 end_pos: tuple = None,
                 horizon: int = 100,
                 constr_penalty: float = -100,
                 goal_thresh: int = 6,
                 noise_scale: float = 0.125
                 ):
        utils.EzPickle.__init__(self)
        self.done = self.state = None
        self.horizon = horizon
        self.start_pos = start_pos
        self.goal_thresh = goal_thresh
        self.noise_scale = noise_scale
        self.constr_penalty = constr_penalty
        self.action_space = Box(-np.ones(2) * MAX_FORCE,
                                np.ones(2) * MAX_FORCE)
        self.scale_factor = 1.0
        if from_pixels:
            self.observation_space = Box(-1, 1, (3, 64, 64))
        else:
            self.observation_space = Box(-np.ones(2) * np.float('inf'),
                                         np.ones(2) * np.float('inf'))

        self._episode_steps = 0

        self._from_pixels = from_pixels
        self._image_cache = {}

        if box_bounds:
            self.box_bounds = box_bounds
        else:
            # State bounds (box edges)
            start_size = [12.0, 10.0]
            tunnel_size = [6.0, 2.0]
            finish_size = [6.0, 10.0]
            self.box_bounds = {'box_start': np.array(start_size),
                               'tunnel': np.array([start_size[0] + tunnel_size[0], tunnel_size[1]]),
                               'box_finish': np.array([start_size[0] + tunnel_size[0] + finish_size[0],
                                                       finish_size[1]])}

            box_width = self.box_bounds['box_finish'][0]
            box_height = max([2 * value[1] for value in self.box_bounds.values()])
            if box_height >= box_width:
                self.scale_factor = (WINDOW_HEIGHT / box_height) - 0.1
            else:
                self.scale_factor = (WINDOW_WIDTH / box_width) - 0.1
            box_bounds = {}
            last_dist = 0
            for key, value in self.box_bounds.items():
                x_val = (self.scale_factor * value[0])
                y_val = self.scale_factor/2 * value[1]
                box_bounds[key] = np.array([x_val, y_val])
            self.box_bounds = box_bounds

        self.maze_points = self._create_maze_points()

        if end_pos:
            self.goal = end_pos
        else:
            # Goal state
            self.goal = (((self.box_bounds['box_finish'][0] - self.box_bounds['tunnel'][0]) / 2) +
                         self.box_bounds['tunnel'][0], 0.0)

        if start_pos:
            self.start_pos = start_pos
        else:
            # Start position
            self.start_pos = (self.box_bounds['box_start'][0] / 2, 0.0)

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert self.action_space.contains(action), err_msg
        # assert self.state is not None, "Call reset before using step method"

        action = self._process_action(action)
        last_state = self.state.copy()
        next_state = self._next_state(self.state, action)
        cur_reward = self.step_reward(self.state, action)
        self.state = next_state
        self._episode_steps += 1
        constr = self._bounds_check(next_state)
        self.done = self._episode_steps >= self.horizon

        if self._from_pixels:
            obs = self._state_to_image(self.state)
        else:
            obs = self.state
        return obs, cur_reward, self.done, {
            "constraint": constr,
            "reward": cur_reward,
            "state": last_state,
            "next_state": next_state,
            "action": action
        }

    def step_reward(self, state, action):
        """
        Returns -1 if not in goal otherwise 0
        :param state:
        :param action:
        :return:
        """
        if self._hit_goal(state, self.goal_thresh):
            return 0
        else:
            return -1
        # return int(np.linalg.norm(np.subtract(self.goal, state)) < self.goal_thresh) - 1

    def reset(self, random_start=False):
        if random_start:
            self.state = np.random.random(2) * (WINDOW_WIDTH, WINDOW_HEIGHT)
            if self._bounds_check(self.state):  # reset if spawn outside bounds
                self.reset(True)
        else:
            self.state = self.start_pos + np.random.randn(2)
        self.done = False
        self._episode_steps = 0
        if self._from_pixels:
            obs = self._state_to_image(self.state)
        else:
            obs = self.state
        return obs

    def render(self, mode='human'):
        return self._draw_state(self.state)

    def draw(self, trajectories=None, heatmap=None, plot_starts=False, board=True, file=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if heatmap is not None:
            assert heatmap.shape == (WINDOW_HEIGHT, WINDOW_WIDTH)
            heatmap = np.flip(heatmap, axis=0)
            im = plt.imshow(heatmap, cmap='hot')
            plt.colorbar(im)

        if board:
            self.draw_board(ax)

        if trajectories is not None and type(trajectories) == list:
            if type(trajectories[0]) == list:
                self.plot_trajectories(ax, trajectories, plot_starts)
            if type(trajectories[0]) == dict:
                self.plot_trajectory(ax, trajectories, plot_starts)

        ax.set_aspect('equal')
        ax.autoscale_view()

        if file is not None:
            plt.savefig(file)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_trajectory(self, ax, trajectory, plot_start=False):
        self.plot_trajectories(ax, [trajectory], plot_start)

    @staticmethod
    def plot_trajectories(ax, trajectories, plot_start=False):
        """
        Renders a trajectory to pyplot. Assumes you already have a plot going
        :param ax:
        :param trajectories: Trajectories to impose upon the graph
        :param plot_start: whether or not to draw a circle at the start of the trajectory
        :return:
        """

        for trajectory in trajectories:
            states = np.array([frame['obs'] for frame in trajectory])
            plt.plot(states[:, 0], WINDOW_HEIGHT - states[:, 1])
            if plot_start:
                start = states[0]
                start_circle = plt.Circle((start[0], WINDOW_HEIGHT - start[1]),
                                          radius=2, color='lime')
                ax.add_patch(start_circle)

    def draw_board(self, ax):
        plt.xlim(0, WINDOW_WIDTH)
        plt.ylim(0, WINDOW_HEIGHT)

        ax.add_patch(patches.Polygon(self.maze_points, linewidth=1, color='green', fill=False))
        circle = plt.Circle((self.start_pos[0], self.start_pos[1] + (WINDOW_HEIGHT/2)), radius=3, color='k')
        ax.add_patch(circle)
        circle = plt.Circle((self.goal[0], self.goal[1] + (WINDOW_HEIGHT/2)), radius=3, color='k')
        ax.add_patch(circle)
        ax.annotate("start", xy=(self.start_pos[0], self.start_pos[1] + (WINDOW_HEIGHT/2) - 8), fontsize=10,
                    ha="center")
        ax.annotate("goal", xy=(self.goal[0], self.goal[1] + (WINDOW_HEIGHT/2) - 8), fontsize=10, ha="center")

    def _draw_state(self, state):
        BACKGROUND_COLOUR = (0, 0, 255)
        ACTOR_COLOUR = (255, 0, 0)
        OBSTACLE_COLOUR = (0, 0, 0)

        def draw_circle(draw, center, radius, colour):
            lower_bound = tuple(np.subtract(center, radius))
            upper_bound = tuple(np.add(center, radius))
            draw.ellipse([lower_bound, upper_bound], fill=colour)

        im = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), BACKGROUND_COLOUR)
        draw = ImageDraw.Draw(im)

        state = np.array([state[0], state[1] + WINDOW_HEIGHT / 2])
        draw.polygon(self.maze_points, fill=OBSTACLE_COLOUR, outline=(0, 0, 0), width=1)
        draw_circle(draw, state, 5, ACTOR_COLOUR)
        return np.array(im)

    def _next_state(self, state, action, override=False):
        if self._bounds_check(state):
            return state

        next_state = state + action + self.noise_scale * np.random.randn(len(state))
        # next_state = np.clip(next_state, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT))
        return next_state

    @staticmethod
    def _process_action(action):
        return np.clip(action, -MAX_FORCE, MAX_FORCE)

    def _state_to_image(self, state):

        def state_to_int(state):
            return int(state[0]), int(state[1])
        state = state_to_int(state)
        image = self._image_cache.get(state)
        if image is None:
            image = self._draw_state(state)
            image = image.transpose((2, 0, 1))  # Not sure why we transpose?
            image = (resize(image, (3, 64, 64)) * 255).astype(np.uint8)
            self._image_cache[state] = image
        return image

    def _bounds_check(self, state: tuple) -> bool:
        """
        Check if the bounds of the structure are exceeded
        :param state: current agent x, y position
        :return: boolean of whether the problem bounds have been breached
        """
        x, y = state
        # Fix bounds check
        x_bounds = [0.0, self.box_bounds['box_start'][0], self.box_bounds['tunnel'][0],
                    self.box_bounds['box_finish'][0]]
        idx = 0
        for _, v in self.box_bounds.items():
            if x_bounds[idx] < x < x_bounds[idx + 1] and abs(y) < v[1]:
                return False
            idx += 1
        return True

    def _hit_goal(self, state: tuple, goal_bound: float = 0.5) -> bool:
        """
        Check if the agent hit the goal state (within bound)
        :param: current agent x, y position
        :param: current agent goal position
        :return: boolean of whether we hit the goal
        """
        x_lower = self.goal[0] - goal_bound
        x_upper = self.goal[0] + goal_bound
        y_lower = self.goal[1] - goal_bound
        y_upper = self.goal[1] + goal_bound
        if x_lower < state[0] < x_upper and y_lower < state[1] < y_upper:
            return True
        else:
            return False

    def _create_maze_points(self) -> np.array:
        """
        Create a list containing the maze edge positions
        :return: maze_points as a numpy array
        """
        y_offset = WINDOW_HEIGHT / 2
        maze_points = [(0.0, self.box_bounds['box_start'][1] + y_offset),
                       (self.box_bounds['box_start'][0], self.box_bounds['box_start'][1] + y_offset),
                       (self.box_bounds['box_start'][0], self.box_bounds['tunnel'][1] + y_offset),
                       (self.box_bounds['tunnel'][0], self.box_bounds['tunnel'][1] + y_offset),
                       (self.box_bounds['tunnel'][0], self.box_bounds['box_finish'][1] + y_offset),
                       (self.box_bounds['box_finish'][0], self.box_bounds['box_finish'][1] + y_offset),
                       (self.box_bounds['box_finish'][0], -self.box_bounds['box_finish'][1] + y_offset),
                       (self.box_bounds['tunnel'][0], -self.box_bounds['box_finish'][1] + y_offset),
                       (self.box_bounds['tunnel'][0], -self.box_bounds['tunnel'][1] + y_offset),
                       (self.box_bounds['box_start'][0], -self.box_bounds['tunnel'][1] + y_offset),
                       (self.box_bounds['box_start'][0], -self.box_bounds['box_start'][1] + y_offset),
                       (0.0, -self.box_bounds['box_start'][1] + y_offset)]
        return maze_points
