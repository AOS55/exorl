from dataclasses import dataclass, replace, fields
from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
from collections import namedtuple

Env_Config = namedtuple('Env_Config', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

def name_bwc_config(ground_roughness,
                	pit_gap,
                    stump_width, stump_height, stump_float,
                    stair_width, stair_height, stair_steps):
	"""
	Make a name for an environment based on the contents of the environment

	:param ground_roughness:
	:param pit_gap:
	:param stump_width:
	:param stump_height:
	:param stump_float:
	:param stair_width:
	:param stair_height:
	:param stair_steps:
	:return:
	"""
	env_name = 'r' + str(ground_roughness)
	if pit_gap:
		env_name += '.p' + str(pit_gap[0]) + '_' + str(pit_gap[1])
	if stump_width:
		env_name += '.b' + str(stump_width[0]) + '_' + str(stump_height[0]) + '_' + str(stump_height[1])
	if stair_steps:
		env_name += '.s' + str(stair_steps[0]) + '_' + str(stair_height[1])
	return env_name

def init_flat_bwc():
	roughness = 1
	pit_gap = (0, 0.8)
	stump_width = (1, 2)
	stump_height = (0.1, 0.4)
	stump_float = (0.1, 1)
	stair_width = (1, 2)
	stair_height = (0.1, 0.4)
	stair_steps = (1, 2)
	name = name_bwc_config(roughness, pit_gap, stump_width, stump_height, stump_float, stair_height, stair_width, stair_steps)
	config = Env_Config(name, roughness, pit_gap, stump_width, stump_height, stump_float, stair_height, stair_width, stair_steps)
	return config

@dataclass
class EnvParams:
    name: str
    value: tuple or float
    default_value: tuple or float
    increment: tuple or float
    limits: tuple or float

class EnvMaker(ABC):

    def __init__(self, params: EnvParams, master_seed: int, Populate=None):
        self.params = params
        self.rs = np.random.RandomState(master_seed)
        if Populate:
            populate_array = Populate(self.params, self.rs)
            self.populate_array = populate_array.populate_array
        else:
            self.populate_array = self._populate_array

    def new_value(self, arr, interval=0, enforce=False):
        arr = self.populate_array(arr, interval, enforce)
        return arr

    def _populate_array(self, arr, interval=0, enforce=False):
        """
        Method to populate array based on current values
        """
        assert isinstance(arr, list)
        if len(arr) == 0 or enforce:
            arr = list(self.params.default_value)
        elif len(self.params.limits) == 2:
            choices = []
        for change0 in [self.params.increment[0], 0.0, self.params.increment[1]]:
            arr0 = np.round(arr[0] + change0, 1)
            if arr0 > self.params.limits[1] or arr0 < self.params.default_value[0]:
                continue
            for change1 in [self.params.increment[0], 0.0, self.params.increment[1]]:
                arr1 = np.round(arr[1] + change1, 1)
            if arr1 > self.params.limits[1] or arr1 < self.params.default_value[1]:
                continue
            if change0 == 0.0 and change1 == 0.0:
                continue
            if arr0 + interval > arr1:
                continue
        
            choices.append([arr0, arr1])

        num_choices = len(choices)
        if num_choices > 0:
            idx = self.rs.randint(num_choices)
            arr[0] = choices[idx][0]
            arr[1] = choices[idx][1]
        return arr

class DR:

    def __init__(self, params, rs):
        self.params = params
        self.rs = rs

    def populate_array(self, arr, interval=0, enforce=False):
        """
        Domain Randomized array, occurs over entire design space
        """
        arr = [self.params.limits[0], self.params.limits[1]]
        return arr

class Roughness(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None):
        if params:
            self.params = params
        else:
            init_value = 1.0
            self.params = EnvParams(name='roughness', value=init_value, default_value=init_value, increment=(-0.6, 0.6), limits=(0.0, 10.0))
        super().__init__(self.params, master_seed, Populate)

    def new_value(self, _):
        self.params.value = np.round(self.params.value + self.rs.uniform(self.params.increment[0], self.params.increment[1]), 1)
        if self.params.value > self.params.limits[1]:
            self.params.value = self.params.limits[1]
        if self.params.value < self.params.limits[0]:
            self.params = 0.0
        return self.params


class PitGap(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None): 
        if params:
            self.params = params
        else:
            init_value = [0, 0.8]
            self.params = EnvParams(name='pit_gap', value=init_value, default_value=init_value, increment=(-0.4, 0.4), limits=(-8.0, 8.0))
        super().__init__(self.params, master_seed, Populate)


class StumpWidth(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None):
        if params:
            self.params = params
        else:
            init_value = [1, 2]
            self.params = EnvParams(name='stump_width', value=init_value, default_value=init_value, increment=(0, 0), limits=(0, 0))
        super().__init__(self.params, master_seed, Populate)

    def new_value(self, arr, enforce=False):
        enforce = (len(self.params.value) == 0)
        arr = self.populate_array(arr, (0, 0), enforce)
        return arr


class StumpHeight(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None):
        if params:
            self.params = params
        else:
            init_value = [0.1, 0.4]
            self.params = EnvParams(name='stump_height', value=init_value, default_value=init_value, increment=(-0.2, 0.2), limits=(-5.0, 5.0))
        super().__init__(self.params, master_seed, Populate)


class StumpFloat(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None):
        if params:
            self.params = params
        else:
            init_value = [0.1, 1]
            self.params = EnvParams(name='stump_float', value=init_value, default_value=init_value, increment=(0, 0), limits=(0, 0))
        super().__init__(self.params, master_seed, Populate)


class StairWidth(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None):
        if params:
            self.params = params
        else:
            init_value = [4, 5]
            self.params = EnvParams(name='stair_width', value=init_value, default_value=init_value, increment=(0, 0), limits=(0, 0))
        super().__init__(self.params, master_seed, Populate)


class StairHeight(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None):
        if params:
            self.params = params
        else:
            init_value = [0.1, 0.4]
            self.params = EnvParams(name='stair_height', value=init_value, default_value=init_value, increment=(-0.2, 0.2), limits=(-5.0, 5.0))
        super().__init__(self.params, master_seed, Populate)


class StairSteps(EnvMaker):

    def __init__(self, master_seed: int, params: EnvParams = None, Populate=None):
        if params:
            self.params = params
        else:
            init_value = [1, 2]
            self.params = EnvParams(name='stair_steps', value=init_value, default_value=init_value, increment=(-1.0, 1.0), limits=(-9.0, 9.0))
        super().__init__(self.params, master_seed, Populate)


def init_env_params(master_seed: int, Populate=None):

    env_params = {
        'roughness': Roughness(master_seed, Populate=Populate),
        'pit_gap': PitGap(master_seed, Populate=Populate),
        'stump_width': StumpWidth(master_seed, Populate=Populate),
        'stump_height': StumpHeight(master_seed, Populate=Populate),
        'stump_float': StumpFloat(master_seed, Populate=Populate),
        'stair_width': StairWidth(master_seed, Populate=Populate),
        'stair_height': StairHeight(master_seed, Populate=Populate),
        'stair_steps': StairSteps(master_seed, Populate=Populate)
        }
    return env_params

def make_env_config(name, env_params):
    env_config = Env_Config(
        name=name,
        ground_roughness=env_params['roughness'].params.value,
        pit_gap=env_params['pit_gap'].params.value,
        stump_width=env_params['stump_width'].params.value,
        stump_height=env_params['stump_height'].params.value,
        stump_float=env_params['stump_float'].params.value,
        stair_width=env_params['stair_width'].params.value,
        stair_height=env_params['stair_height'].params.value,
        stair_steps=env_params['stair_steps'].params.value,
    )
    return env_config

def make_new_params(env_params):
    print(f'env_params: {env_params}')
    for key, value in env_params.items():
        new_value = value.new_value(value.params.value)
        env_params.update({key: replace(env_params[key].params, value=new_value)})
    print(f'updated env_params: {env_params}')
    return env_params
