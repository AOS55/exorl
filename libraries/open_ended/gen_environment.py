from abc import ABCMeta, abstractmethod
import numpy as np


class Reproducer(metaclass=ABCMeta):
    """
    Abstract Base Class to create a list of levels/env paramters based on the env parameters
    """
    def __init__(self,
                 env_categories, 
                 master_seed, 
                 init_level_params):
        self.rs = np.random.RandomState(master_seed)
        self.env_categories = env_categories
        self.levels = [self.init_level(init_level_params)]

    @abstractmethod
    def init_level(self, init_level_params):
        """
        Initialize the level based on parameters
        """
        init_level = None
        return init_level

    def populate_array(self, arr, default_value, interval=0, increment=0, enforce=False, max_value=[]):
        """
        Method to populate array based on current value
        """
        assert isinstance(arr, list)
        if len(arr) == 0 or enforce:
            arr = list(default_value)
        elif len(max_value) == 2:
            choices = []
        for change0 in [increment, 0.0, -increment]:
            arr0 = np.round(arr[0] + change0, 1)
            if arr0 > max_value[0] or arr0 < default_value[0]:
                continue
            for change1 in [increment, 0.0, -increment]:
                arr1 = np.round(arr[1] + change1, 1)
            if arr1 > max_value[1] or arr1 < default_value[1]:
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

    @abstractmethod
    def add_level(self):
        """
        Abstract method to add a level to levels
        """
        return self.levels