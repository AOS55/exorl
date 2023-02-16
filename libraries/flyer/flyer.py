import pathlib

import gym
import imageio
import numpy as np
import opensimplex
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from scipy import misc
from PIL import Image
from utils import is_free, plot_2d, point_in_hull
import matplotlib.pyplot as plt
from hashlib import sha256
from collections import namedtuple
import os
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import contextlib
with contextlib.redirect_stdout(None):
    import pygame


TEXTURES = {
    'water': 'assets/water.png',
    'grass': 'assets/grass.png',
    'normal-grass': 'assets/normal-grass.png',
    'darker-grass': 'assets/darker-grass.png',
    'forest-grass': 'assets/forest-grass.png',
    'forest-dirt': 'assets/forest-dirt.png',
    'forest-leaves': 'assets/forest-leaves.png',
    'stone': 'assets/stone.png',
    'mud': 'assets/mud.png',
    'light-mud': 'assets/light-mud.png',
    'path': 'assets/path.png',
    'sand': 'assets/sand.png',
    'tree': 'assets/tree.png',
    '1-flower': 'assets/1-flower.png',
    '2-flowers': 'assets/2-flowers.png',
    '4-flowers': 'assets/4-flowers.png',
    '6-flowers': 'assets/6-flowers.png',
    '8-poppies': 'assets/8-poppies.png',
    'lava': 'assets/lava.png',
    'table': 'assets/table.png',
    'furnace': 'assets/furnace.png',
    'aircraft-west': 'assets/aircraft-west.png',
    'aircraft-east': 'assets/aircraft-east.png',
    'aircraft-north': 'assets/aircraft-north.png',
    'aircraft-south': 'assets/aircraft-south.png',
    'cow': 'assets/cow.png',
    'zombie': 'assets/zombie.png',
    'bush': 'assets/bush.png',
    'dead-bushel': 'assets/dead-bushel.png',
    'green-bushel': 'assets/green-bushel.png',
    'ripe-bushel': 'assets/ripe-bushel.png',
    'pink-flowers': 'assets/pink-flowers.png',
    'thistle': 'assets/thistle.png',
    'tree-stump': 'assets/tree-stump.png',
    'evergreen-fur': 'assets/evergreen-fur.png',
    'wilting-fur': 'assets/wilting-fur.png',
    'apple-tree': 'assets/apple-tree.png',
    'pruned-tree': 'assets/pruned-tree.png',
    'palm-tree': 'assets/palm-tree.png',
    'banana-tree': 'assets/banana-tree.png'
}

MATERIAL_NAMES = {
    1: 'water',
    2: 'grass',
    3: 'normal-grass',
    4: 'darker-grass',
    5: 'stone',
    6: 'mud',
    7: 'light-mud',
    8: 'path',
    9: 'sand',
    10: 'tree',
    11: '1-flower',
    12: '2-flowers',
    13: '4-flowers',
    14: '6-flowers',
    15: '8-poppies',
    16: 'lava',
    17: 'table',
    18: 'furnace',
    19: 'forest-grass',
    20: 'forest-dirt',
    21: 'forest-leaves'
}

OBS_MAPPING = {
    1: 0,
    2: 1,
    4: 2,
    6: 3,
    7: 4,
    9: 5,
    11: 6,
    12: 7,
    13: 8,
    14: 9,
    21: 10
}

LAND_TYPE_COLOR_MAP = {
    (0, 255, 0): 'grass',
    (136, 69, 19): 'forest',
    (255, 255, 0): 'crops',
    (127, 255, 212): 'orchard',
    (255, 127, 80): 'cattle'
}

MATERIAL_IDS = {
    name: id_ for id_, name in MATERIAL_NAMES.items()
}

LANDABLE = {
    MATERIAL_IDS['grass'],
    MATERIAL_IDS['normal-grass'],
    MATERIAL_IDS['darker-grass'],
    MATERIAL_IDS['mud'],
    MATERIAL_IDS['light-mud'],
    MATERIAL_IDS['path'],
    MATERIAL_IDS['sand'],
    MATERIAL_IDS['1-flower'],
    MATERIAL_IDS['2-flowers'],
    MATERIAL_IDS['4-flowers'],
    MATERIAL_IDS['6-flowers'],
    MATERIAL_IDS['8-poppies'],
    MATERIAL_IDS['forest-grass']
}

FLYABLE = {
    MATERIAL_IDS['water'],
    MATERIAL_IDS['grass'],
    MATERIAL_IDS['normal-grass'],
    MATERIAL_IDS['darker-grass'],
    MATERIAL_IDS['stone'],
    MATERIAL_IDS['mud'],
    MATERIAL_IDS['light-mud'],
    MATERIAL_IDS['path'],
    MATERIAL_IDS['sand'],
    MATERIAL_IDS['tree'],
    MATERIAL_IDS['1-flower'],
    MATERIAL_IDS['2-flowers'],
    MATERIAL_IDS['4-flowers'],
    MATERIAL_IDS['6-flowers'],
    MATERIAL_IDS['8-poppies'],
    MATERIAL_IDS['lava'],
    MATERIAL_IDS['table'],
    MATERIAL_IDS['furnace'],
    MATERIAL_IDS['forest-grass'],
    MATERIAL_IDS['forest-dirt'],
    MATERIAL_IDS['forest-leaves']
}


ENV_CONFIG = {
    'name': 'default',  # name of environment config
    'num_fields': 200,  # number of fields to decompose (less gives larger fields)
    'land_types': ['grass', 'forest', 'crops', 'orchard', 'cattle'],  # list of land types as strings
    'water_cutoff': -0.1,  # where to cutoff water based on OpenSimplex noise -0.6 < cutoff < +0.6, higher is more
    'beach_thickness': 0.04,  # amount of water to make into sand, increase for more sand
    'forest_tree_density': 0.6,  # density of trees in forest, increase for more trees
    'orchard_tree_density': 0.1,  # density of trees in orchard, increase for more trees
    'orchard_flower_density': 0.1,  # density of flowers in orchard, increase for more flowers
    'cow_density': 0.02  # density of cows in cattle field, increase for more cattle
}


class Objects:
    """
    General class inherited for objects for simulation
    """

    def __init__(self, area: tuple):
        """
        Objects constructor

        :param area: size of 3D playable area as a (x, y, z) tuple
        """
        self._map = np.zeros(area, np.uint32)
        self._objects = [None]

    def __iter__(self):
        yield from (obj for obj in self._objects if obj)

    def add(self, obj) -> None:
        """
        Add an object to the map

        :param obj: the object to be added
        :return: None
        """
        assert hasattr(obj, 'pos')
        assert self.free(obj.pos)
        self._map[obj.pos[0], obj.pos[1], obj.pos[2]] = len(self._objects)
        self._objects.append(obj)

    def remove(self, obj) -> None:
        """
        Remove an object from the map

        :param obj: the object to be removed
        :return: None
        """
        self._objects[self._map[obj.pos[0], obj.pos[1], obj.pos[2]]] = None
        self._map[obj.pos[0], obj.pos[1], obj.pos[2]] = 0

    def move(self, obj, pos) -> None:
        """
        Move an object in the playable area

        :param obj: the object to be moved
        :param pos: the position to move the object to
        :return: None
        """
        assert self.free(pos)
        self._map[pos[0], pos[1], pos[2]] = self._map[obj.pos[0], obj.pos[1], obj.pos[2]]
        self._map[obj.pos[0], obj.pos[1], obj.pos[2]] = 0
        obj.pos = pos

    def free(self, pos):
        """
        Find if there is an object at the position

        :param pos: the position to query
        :return: the object at the queried position
        """
        return self.at(pos) is None

    def at(self, pos):
        """
        Method used to check if free

        :param pos: the position to query
        :return: the objects at a given position
        """
        if not (0 <= pos[0] < self._map.shape[0]):
            return False
        if not (0 <= pos[1] < self._map.shape[1]):
            return False
        if not (0 <= pos[2] < self._map.shape[2]):
            return False
        return self._objects[self._map[pos[0], pos[1], pos[2]]]


class Aircraft:
    """
    Main agent aircraft class
    """
    def __init__(self, pos: tuple, airspeed: int, ldg_dist: int,
                 rod: int, engine_fail: bool = False, health: int = 5):
        """
        Aircraft constructor

        :param pos: the (x, y, z) position of the aircraft
        :param airspeed: the pixels moved per update
        :param ldg_dist: landing distance, the pixels moved forward in heading direction when landing
        :param rod: the pixels decreased (alt) per update
        :param engine_fail: if True rod > 0, else rod = 0
        :param health: aircraft health
        """
        self.pos = list(pos)
        self.airspeed = airspeed
        self.ldg_dist = ldg_dist
        self.state = 'aircraft-north'
        self.rod = rod
        self.engine_fail = engine_fail
        self.health = health
        self.ldg_moves = 0  # variable to track how far along the landing ground roll has moved
        self.aircraft_states = ('aircraft-north', 'aircraft-east', 'aircraft-south', 'aircraft-west')
        self.aircraft_actions = {'aircraft-north': ('aircraft-west', 'aircraft-east'),
                                 'aircraft-east': ('aircraft-north', 'aircraft-south'),
                                 'aircraft-south': ('aircraft-east', 'aircraft-west'),
                                 'aircraft-west': ('aircraft-south', 'aircraft-north')}
        self.movement = {'aircraft-north': (0, self.airspeed),
                         'aircraft-east': (self.airspeed, 0),
                         'aircraft-south': (0, -self.airspeed),
                         'aircraft-west': (-self.airspeed, 0)}
        self.gnd_movement = {'aircraft-north': (0, 1),
                             'aircraft-east': (1, 0),
                             'aircraft-south': (0, -1),
                             'aircraft-west': (-1, 0)}
        self.hdg = self.movement[self.state]

    @property
    def texture(self):
        """
        texture shown on screen based on aircraft direction

        :return: texture index for TEXTURES
        """
        return {
            (-self.airspeed, 0): 'aircraft-west',
            (self.airspeed, 0): 'aircraft-east',
            (0, -self.airspeed): 'aircraft-north',
            (0, self.airspeed): 'aircraft-south',
        }[self.hdg]

    def update(self, terrain, objects, action) -> None:
        """
        update step for the aircraft

        :param terrain: map of terrain surrounding the aircraft
        :param objects: objects present on the map
        :param action: action taken by the aircraft in the simulation
        :return: None
        """
        # Movement when at altitude
        self.pos = list(self.pos)
        if self.pos[2] > 0:
            if self.engine_fail:
                self.pos[2] -= self.rod  # glide descent (engine fail)
            if action == 0:
                pass
            if 1 <= action <= 2:
                # left turn (90 degrees)
                self.state = self.aircraft_actions[self.state][action - 1]
                self.hdg = self.movement[self.state]
            target = (self.pos[0] + self.hdg[0], self.pos[1] + self.hdg[1], self.pos[2])
            if is_free(target, terrain, objects, valid=FLYABLE):
                objects.move(self, target)
        # Ground landing movement
        if self.pos[2] == 0:
            move = self.gnd_movement[self.state]
            target = (self.pos[0] + move[0], self.pos[1] + move[1], 0)
            if is_free(target, terrain, objects, valid=LANDABLE):
                objects.move(self, target)
                # print('moved object')
            else:
                self.health -= 1  # decrement health if colliding with a non-landable object
            # print(self.health)
            self.ldg_moves += 1


class Cow:
    """
    A cow NPC (Non-Playable Character) that walks randomly around the map
    """

    def __init__(self, pos, random):
        """
        Cow constructor

        :param pos: cow starting position (x,y)
        :param random: random number for cow position and movement
        """
        self.pos = pos
        self.health = 1
        self._random = random

    @property
    def texture(self):
        """
        texture index for cow

        :return: cow texture index
        """
        return 'cow'

    def update(self, terrain, objects, action):
        """
        update step for cow

        :param terrain:
        :param objects:
        :param action:
        :return:
        """
        if self.health <= 0:
            objects.remove(self)
        if self._random.uniform() < 0.5:
            return
        direction = _random_direction(self._random)
        x = self.pos[0] + direction[0]
        y = self.pos[1] + direction[1]
        if is_free((x, y, 0), terrain, objects, valid=LANDABLE):
            objects.move(self, (x, y, 0))


class StaticObject:
    """
    A class for static objects with fixed height and position
    """

    def __init__(self,
                 pos: tuple,
                 texture_name: str):
        """
        static object constructor

        :param pos: static object (x, y) position
        :param texture_name: name of texture used for object
        """
        self.pos = pos
        self.texture_name = texture_name

    @property
    def texture(self):
        """
        apply appropriate texture property from TEXTURES

        :return:
        """
        return self.texture_name

    def update(self, terrain, objects, action):
        """
        update step for static objects, should leave untouched

        :param terrain:
        :param objects:
        :param action:
        :return:
        """
        return


class Tree:
    """
    A static tree object with a fixed height and position
    """

    def __init__(self,
                 pos: tuple):
        """
        Tree constructor

        :param pos: tree (x, y) position on the map
        """
        self.pos = pos

    @property
    def texture(self):
        """
        texture index for tree

        :return: tree texture index
        """
        return 'tree'

    def update(self, terrain, objects, action):
        """
        update step for tree

        :param terrain:
        :param objects:
        :param action:
        :return:
        """
        return


class Env(gym.Env):
    """
    Environment for training, similar to OpenAI env
    """

    def __init__(self,
                 env_config: dict = ENV_CONFIG,
                 area=(1024, 1024, 64),
                 view=4,
                 size=84,
                 length=1000,
                 rod: int = 1,
                 airspeed: int = 4,
                 ldg_dist: int = 6,
                 engine_fail: bool = True,
                 health: int = 5,
                 water_present: bool = False,
                 custom_world: str = None,
                 seed=None,
                 mode="map"):
        """
        Environment constructor

        :param env_config: environment config namedTuple to control environment
        :param area: 2D map area
        :param view: size of observed map around centered aircraft
        :param size: total size of map
        :param length: max length of each episode
        :param rod: aircraft rod, pixels decreased (alt) per update
        :param airspeed: aircraft airspeed, the pixels moved per update
        :param ldg_dist: aircraft landing distance, pixels moved on ground roll
        :param engine_fail: the state of the aircraft's engine at the beginning of the episode
        :param health: aircraft health
        :param water_present: boolean whether water is present in the world
        :param custom_world: custom world image path to use instead of default generative world, if None will use generative
        :param seed: the seed to initialize the world
        :param mode: the mode used to output from reset and step [map, image]
        """

        self._config = None
        self.set_env_config(env_config)
        self._area = area
        self._view = view
        self._size = size
        self._length = length
        self._rod = rod
        self._airspeed = airspeed
        self._ldg_dist = ldg_dist
        self._engine_fail = engine_fail
        self._health = health
        self._seed = seed
        self._episode = 0
        self._grid = self._size // (2 * self._view + 1)
        self._textures = self._load_textures()
        if type(self._area[2]) == int:
            self._terrain = np.zeros(self._area, np.uint8)
        else:
            terrain_area = list(self._area[0:2])
            terrain_area.append(self._area[2][1])
            self._terrain = np.zeros(terrain_area, np.uint8)
        self._border = (size - self._grid * (2 * self._view + 1)) // 2
        self.id_step = None
        self._random = None
        self.aircraft = None
        self._objects = None
        self._simplex = None
        self._achievements = None
        self._last_health = None
        self._water_present = water_present
        self._center = None
        self._viewer_initialized = False
        self._mode = mode
        self._custom_world = custom_world

    def set_env_config(self, env_config) -> None:
        """
        Set the config class to the loaded environment config

        :param env_config: environment configuration, usually called in constructor but this method is 'public'
        :return: None
        """
        self._config = env_config

    @property
    def observation_space(self) -> gym.spaces:
        """
        Gym observation space

        :return: the gym.spaces to be observed (an image in this case)
        """
        # shape = (self._size, self._size, 3)
        shape = self._terrain.shape
        shape_low = np.zeros((shape[0] * shape[1]) + 3, dtype=np.uint8)
        shape_high = 10 * np.ones((shape[0] * shape[1]) + 3, dtype=np.uint8)
        shape_low[-3:] = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        shape_high[-3:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # shape = (shape[0] * shape[1]) + 3
        return gym.spaces.Box(low=shape_low, high=shape_high)
        # return gym.spaces.Box(0, 255, shape, np.uint8)

    @property
    def action_space(self) -> gym.spaces:
        """
        Gym action space

        :return: the gym.spaces actions can be taken in
        """
        return gym.spaces.Discrete(3)

    @property
    def action_names(self) -> list:
        """
        Gym action names, the names of each possible gym action

        :return: list of possible action names
        """
        return [
            'noop', 'right', 'left'
        ]

    def _noise(self, x: float, y: float, z, sizes, normalize: bool = True):
        """
        3D simplex noise function

        :param x: x_noise co-ord
        :param y: y_noise co-ord
        :param z: z_noise co-ord
        :param sizes: range of sizes and weights that is randomly sampled between with simplex noise
        :param normalize: whether to normalize the value so it is between 0-1
        :return: noise value for the given co-ord
        """
        if not isinstance(sizes, dict):
            sizes = {sizes: 1}
        value = 0
        for size, weight in sizes.items():
            value += weight * self._simplex.noise3(x / size, y / size, z)
        if normalize:
            value /= sum(sizes.values())
        return value

    def reset(self):
        """
        reset and setup environment, objects and agent

        :return: _obs initial observation vars
        """
        self.id_step = 0
        self._episode += 1
        self._terrain[:] = 0
        self._simplex = opensimplex.OpenSimplex(
            seed=hash((self._seed, self._episode)))
        # TODO: Find a way to seed to self._random
        self._random = np.random.RandomState(self._generate_random_seed())
        # seed=np.uint32(hash((self._seed, self._episode))))

        if type(self._area[2]) == tuple:
            playable_area = self._area[0], self._area[1], self._random.randint(self._area[2][0], self._area[2][1])
        else:
            playable_area = self._area
        self._center = playable_area[0] // 2, playable_area[1] // 2, playable_area[2] - 1

        simplex = self._noise
        uniform = self._random.uniform
        self._last_health = self._health
        self._objects = Objects(playable_area)
        simplex_array = np.zeros((self._area[0], self._area[1]))

        if self._custom_world:
            # Read world image file and generate world
            world_map = imageio.imread(self._custom_world)
            world_map = world_map[:, :, :3]
            # id_array = np.zeros((self._area[0], self._area[1]))
            # TODO: this approach is slow, can this conversion be done inplace
            # for idx in range(world_map.shape[0]):
            #     for idy in range(world_map.shape[1]):
            #         id_array[idx][idy] = LAND_TYPE_COLOR_MAP[tuple(world_map[idx][idy])]
        else:
            # Randomly generate world
            num_land_types = len(self._config['land_types'])
            kd = self._random_kd_tree_clustering(self._config['num_fields'])
            field_categories = {}
            for id_point in range(self._config['num_fields']):
                last_value = 0
                field_value = uniform(0, num_land_types)
                # Assign a category value to each cluster id, currently uniform
                # TODO: Create a method to control the density of each region (non-uniform)
                for value in range(0, num_land_types+1):
                    if last_value <= field_value <= value:
                        field_categories[id_point] = self._config['land_types'][value-1]
                        break
                    last_value = value

        for x in range(self._area[0]):
            for y in range(self._area[1]):
                macro_noise = simplex(x, y, 3, {15: 1, 25: 1}, True)
                simplex_array[x][y] = macro_noise
                # Divide into land and water region, (can add mountains here too)
                if self._config['water_cutoff'] > macro_noise and self._water_present:
                    self._terrain[x, y] = MATERIAL_IDS['water']
                    if self._config['water_cutoff'] - self._config['beach_thickness'] < macro_noise:
                        self._terrain[x, y] = MATERIAL_IDS['sand']
                else:
                    if self._custom_world:
                        land_id = LAND_TYPE_COLOR_MAP[tuple(world_map[x][y])]
                    else:
                        _, pos_id = kd.query([x, y])
                        land_id = field_categories[pos_id]
                    if land_id == 'grass':
                        # Plain grass field
                        self._terrain[x, y] = MATERIAL_IDS['grass']

                    elif land_id == 'forest':
                        # Forest
                        tree_probability = uniform(0, 1)
                        height = int(3 * (simplex(x, y, 6, {5: 1, 10: 1}, True) + 1.0))
                        object_placement = simplex(x, y, 6, {5: 1, 10: 1}, True)
                        if object_placement < 0.2:
                            self._terrain[x, y] = MATERIAL_IDS['forest-leaves']
                            if tree_probability < self._config['forest_tree_density']:
                                self._add_3d_static_object(x, y, height, 'evergreen-fur')
                        elif object_placement < 0.4:
                            self._terrain[x, y] = MATERIAL_IDS['forest-leaves']
                            if tree_probability < self._config['forest_tree_density']:
                                self._add_3d_static_object(x, y, height, 'wilting-fur')
                        else:
                            self._terrain[x, y] = MATERIAL_IDS['forest-leaves']
                    elif land_id == 'crops':
                        # Mud fields with crops (ploughed)
                        gnd_tile = simplex(x, y, 5, {5: 1, 10: 1}, True)
                        if gnd_tile < -0.2:
                            self._terrain[x, y] = MATERIAL_IDS['light-mud']
                        else:
                            self._terrain[x, y] = MATERIAL_IDS['mud']
                        object_placement = simplex(x, y, 10, {5: 1, 10: 1})
                        if object_placement < -0.25:
                            green_bushel = StaticObject((x, y, 0), 'green-bushel')
                            self._objects.add(green_bushel)
                        elif object_placement < 0.00:
                            ripe_bushel = StaticObject((x, y, 0), 'ripe-bushel')
                            self._objects.add(ripe_bushel)
                        elif object_placement < 0.10:
                            dead_bushel = StaticObject((x, y, 0), 'dead-bushel')
                            self._objects.add(dead_bushel)
                    elif land_id == 'orchard':
                        # Orchard
                        self._terrain[x, y] = MATERIAL_IDS['darker-grass']
                        obj_probability = uniform(0, 1)
                        if obj_probability < self._config['orchard_tree_density']:
                            tree_type = uniform(0, 1)
                            if tree_type < 0.75:
                                apple_tree = StaticObject((x, y, 0), 'apple-tree')
                                self._objects.add(apple_tree)
                            else:
                                empty_tree = StaticObject((x, y, 0), 'pruned-tree')
                                self._objects.add(empty_tree)
                        elif obj_probability < self._config['orchard_flower_density'] +\
                                self._config['orchard_tree_density']:
                            flower_type = uniform(0, 1)
                            if flower_type < 0.25:
                                self._terrain[x, y] = MATERIAL_IDS['1-flower']
                            elif flower_type < 0.5:
                                self._terrain[x, y] = MATERIAL_IDS['2-flowers']
                            elif flower_type < 0.75:
                                self._terrain[x, y] = MATERIAL_IDS['4-flowers']
                            else:
                                self._terrain[x, y] = MATERIAL_IDS['6-flowers']
                    elif land_id == 'cattle':
                        # Cow field
                        obj_probability = uniform(0, 1)
                        obstacle_density = 0.015
                        self._terrain[x, y] = MATERIAL_IDS['grass']
                        if obj_probability < self._config['cow_density']:
                            self._objects.add(Cow((x, y, 0), self._random))
                        elif obj_probability - self._config['cow_density'] < obstacle_density:
                            obstacle_type = uniform(0, 1)
                            if obstacle_type < 0.5:
                                obstacle = StaticObject((x, y, 0), 'tree-stump')
                            else:
                                obstacle = StaticObject((x, y, 0), 'pink-flowers')
                            self._objects.add(obstacle)

                    else:
                        self._terrain[x, y] = MATERIAL_IDS['grass']
                        print(f'unexpected land id detected: {land_id}, setting grass for now')
        # plot_2d(simplex_array)
        # vor = self._random_voronoi_decomposition(50)
        # voronoi_plot_2d(vor)
        # plt.show()
        self.aircraft = Aircraft(self._center,
                                 airspeed=self._airspeed,
                                 ldg_dist=self._ldg_dist,
                                 rod=self._rod,
                                 engine_fail=self._engine_fail,
                                 health=self._health)
        self._objects.add(self.aircraft)
        # self._add_ground_objects()
        if self._mode == "map":
            obs = self._terrain[:, :, 0]
            obs = np.array([[OBS_MAPPING[y]for y in x] for x in obs])
            return obs
        elif self._mode == "image":
            return self._obs()
        else:
            print(f"mode not recognized: {self._mode}")

    def step(self, action):
        """
        advance simulation by one step

        :param action: action for each agent to take, only aircraft at the moment
        :return: observation, reward, done state, info
        """
        self.id_step += 1
        self._view = int(self.aircraft.pos[2] * 0.1875) + 4
        self._update_view_size()
        for obj in self._objects:
            obj.update(self._terrain, self._objects, action)
        if self._mode == "image":
            obs = self._obs()
        elif self._mode == "map":
            obs = self._terrain[:, :, 0].flatten()
            obs = np.array([OBS_MAPPING[x] for x in obs])
            obs = np.concatenate((obs, self.aircraft.pos))
        else:
            print(f"mode: {self._mode}")
        reward = 0.0
        if self.aircraft.health < self._last_health:
            # If aircraft loses health reduce reward
            self._last_health = self.aircraft.health
            reward -= 0.1
        elif self.aircraft.health > self._last_health:
            # If aircraft gains health give reward (this shouldn't be possible)
            self._last_health = self.aircraft.health
            reward += 0.1
        crash = self.aircraft.health <= 0
        # if crash:
        #     # If aircraft crashes get -1 reward
        #     reward -= 1
        time_out = self._length and self.id_step >= self._length
        landed = self.aircraft.ldg_moves >= self.aircraft.ldg_dist and self.aircraft.health > 0
        if landed:
            # If aircraft lands get +1 reward
            reward += 1
        done = time_out or crash or landed
        info = {
        }
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        """
        base render class to call appropriate args
        """
        return self._render(*args, **kwargs)

    def _render(self, mode='rgb_array') -> np.array:
        """
        render mode human using pygame for an env viewer

        :param mode: mode for displaying images, 'human' is for viewer, 'rgb_array' is just to get an rgb array no
         screen started
        :return: numpy array of rgb image
        """
        img = self._generate_image()
        # TODO: allow 'human' mode to be called alone
        if mode == 'human':
            screen = None
            if not self._viewer_initialized:
                pygame.init()
                screen = pygame.display.set_mode([self._size, self._size])
            surface = pygame.surfarray.make_surface(img.transpose((1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        return img

    def _generate_image(self) -> np.array:
        """
        generate an image of the whole environment as an np.array

        :return: canvas of render, an np.array [H, W, C]
        """
        z = 0
        canvas = np.zeros((self._size, self._size, 3), np.uint8) + 127
        for i in range(2 * self._view + 2):
            for j in range(2 * self._view + 2):
                x = self.aircraft.pos[0] + i - self._view
                y = self.aircraft.pos[1] + j - self._view
                if not (0 <= x < self._area[0] and 0 <= y < self._area[1]):
                    continue
                texture = self._textures[MATERIAL_NAMES[self._terrain[x, y, z]]]
                self._draw(canvas, (x, y), texture)
        for obj in self._objects:
            texture = self._textures[obj.texture]
            self._draw(canvas, obj.pos, texture)
        return canvas.transpose((1, 0, 2))

    def _obs(self):
        """
        observation for roll-out

        :return: image observation from rgb_array render mode
        """
        obs = self.render(mode='rgb_array')
        # obs = {'image': self.render(), 'health': _uint8(self.aircraft.health)}
        # obs.update({k: _uint8(v) for k, v in self._player.inventory.items()})
        # for key, value in self._player.inventory.items():
        #   obs[key] = np.clip(value, 0, 255).astype(np.uint8)
        return obs

    def _draw(self, canvas, pos, texture):
        """
        draw the aircraft's position at each point

        :param canvas: the np array of materials
        :param pos: position of aircraft
        :param texture: textures to use on canvas
        :return: image of aircraft on the grid
        """
        # TODO: This function is slow.
        x = self._grid * (pos[0] + self._view - self.aircraft.pos[0]) + self._border
        y = self._grid * (pos[1] + self._view - self.aircraft.pos[1]) + self._border
        w, h = texture.shape[:2]
        if not (0 <= x and x + w <= canvas.shape[0]):
            return
        if not (0 <= y and y + h <= canvas.shape[1]):
            return
        if texture.shape[-1] == 4:
            alpha = texture[..., 3:].astype(np.float32) / 255
            texture = texture[..., :3].astype(np.float32) / 255
            current = canvas[x: x + w, y: y + h].astype(np.float32) / 255
            blended = alpha * texture + (1 - alpha) * current
            result = (255 * blended).astype(np.uint8)
        else:
            result = texture
        canvas[x: x + w, y: y + h] = result

    def _load_textures(self):
        """
        load in textures from asset dir

        :return: textures dictionary for each image
        """
        textures = {}
        for name, filename in TEXTURES.items():
            filename = pathlib.Path(__file__).parent / filename
            image = imageio.imread(filename)
            image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
            image = np.array(Image.fromarray(image).resize(
                (self._grid, self._grid), resample=Image.NEAREST))
            textures[name] = image
        return textures

    def _update_view_size(self):
        """
        update the observable size of the surroudings as the aircraft descends

        :return:
        """
        # TODO: speed up method of loading textures, can we preinstall or just maintain an np.array
        self._grid = self._size // (2 * self._view + 1)
        self._textures = self._load_textures()
        self._border = (self._size - self._grid * (2 * self._view + 1)) // 2

    # def _add_ground_objects(self):
    #     z = 0
    #     uniform = self._random.uniform
    #     for x in range(self._area[0]):
    #         for y in range(self._area[1]):
    #             dist = np.sqrt((x - self._center[0]) ** 2 + (y - self._center[1]) ** 2)
    #             if self._terrain[x, y, z] in LANDABLE:
    #                 grass = self._terrain[x, y, z] == MATERIAL_IDS['grass']
    #                 if dist > 6 and grass and uniform() > 0.99:
    #                     self._objects.add(Cow((x, y, 0), self._random))

    def _add_3d_static_object(self, x: int, y: int, obj_height: int, obj_name: str) -> None:
        """
        add a 3d static object

        :param x: object x position
        :param y: object y position
        :param obj_height: object height
        :param obj_name: name of object
        :return: None
        """
        # TODO: Find a better method for having a 3D object than laying objects over each other
        for height in range(0, obj_height):
            self._objects.add(StaticObject((x, y, height), obj_name))

    def _random_voronoi_decomposition(self, num_points: int = 50) -> Voronoi:
        """
        Generates voronoi partitions from a selection of random points

        :param num_points: number of points to use in cell decomposition, high gives lots of fields, small fewer fields
        :return: Voronoi object from scipy.spatial
        """
        points_x = np.random.randint(0, self._area[0], num_points)
        points_y = np.random.randint(0, self._area[1], num_points)
        points = np.vstack((points_x, points_y)).T
        vor = Voronoi(points)
        return vor

    def _random_kd_tree_clustering(self, num_points: int = 50) -> KDTree:
        """
        Generates a KDTree for a series of points that can be queried on the grid
        :param num_points: number of point to use, similar to number of land types used throughout
        :return: KDTree object from scipy.spatial
        """
        points_x = np.random.randint(0, self._area[0], num_points)
        points_y = np.random.randint(0, self._area[1], num_points)
        points = np.vstack((points_x, points_y)).T
        kd_tree = KDTree(points)
        return kd_tree

    @staticmethod
    def _generate_random_seed():
        data = np.random.rand(1000)
        hash_func = sha256(data)
        seed = np.frombuffer(hash_func.digest(), dtype='uint32')
        return seed


def _random_direction(random):
    """
    move an agent in a random direction at each time step

    :param random:
    :return:
    """
    if random.uniform() > 0.5:
        return 0, random.randint(-1, 2)
    else:
        return random.randint(-1, 2), 0


def _uint8(value):
    """
    return a unsigned integer array from a given np input
    :param value:
    :return:
    """
    # return np.clip(value, 0, 255).astype(np.uint8)
    return np.array(max(0, min(value, 255)), dtype=np.uint8)
