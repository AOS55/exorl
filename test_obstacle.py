from libraries.safe.dmc import ant

import os
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt


def cameras(env):
    pixels = []
    for camera_id in range(4):
        pixels.append(env.physics.render(camera_id=camera_id, width=240))
    pixel_image = PIL.Image.fromarray(np.hstack(pixels))
    return pixel_image


def main():

    env = ant.make(task='navigate')

    out = env.reset()
    while out.step_type != 2:
        action = np.random.random(8)
        action = np.zeros(8)
        out = env.step(action)
        print(out)
    pixel_image = cameras(env)
    pixel_image.save("obstacle_image.jpg")


if __name__=='__main__':
    main()
