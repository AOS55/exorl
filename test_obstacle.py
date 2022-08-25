from libraries.safe.dmc import obstacles
from dm_control import composer
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import go_to_target as gtt
from dm_control.locomotion.walkers import Ant

import os
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def main():
    walker = Ant()
    arena = obstacles.Obstacle()
    task = gtt.GoToTarget(walker=walker,
                          arena=arena,
                          target_relative_dist=5.0,
                          target_relative=True)
    env = composer.Environment(
        task=task,
        time_limit=10,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True
    )
    env.reset()
    pixels = []
    for camera_id in range(4):
        pixels.append(env.physics.render(camera_id=camera_id, width=240))
    pixel_image = PIL.Image.fromarray(np.hstack(pixels))
    print(f"Pixel Image: {pixel_image}")
    pixel_image.save("obstacle_image.jpg")

if __name__=='__main__':
    main()
