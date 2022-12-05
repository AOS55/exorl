from libraries.open_ended import Reproducer
import numpy as np


class Random(Reproducer):

    def __init__(self,
                 env,
                 init_level_params,
                 level_limits, 
                 master_seed):
        super().__init__(env, master_seed, init_level_params)

    def init_level(self, init_level_params):
        level = self.env(init_level_params)
        return level

    def add_level(self):
        
