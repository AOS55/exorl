# import libraries.latentsafesets.utils as utils
# from libraries.latentsafesets.utils.arg_parser import parse_args
from libraries.latentsafesets.utils import *

import hydra
import os
from PIL import Image
from tqdm import tqdm


@hydra.main(config_path='configs/.', config_name='data_to_images')
def main(cfg):
    env_name = cfg.env
    frame_stack = cfg.frame_stack
    demo_trajectories = []
    for count, data_dir in list(zip([cfg.data_counts,], [cfg.data_dirs,])):
        demo_trajectories += utils.load_trajectories(count, file='./../../../data/' + data_dir)

    i = 0
    save_folder = os.path.join('./../../../data/data_images', env_name)
    os.makedirs(save_folder, exist_ok=True)
    for trajectory in tqdm(demo_trajectories):
        for frame in trajectory['observation']:
            if frame_stack == 1:
                im = Image.fromarray(frame.transpose((1, 2, 0)))
                im.save(os.path.join(save_folder, '%d.png' % i))
            else:
                for j in range(frame_stack):
                    im = Image.fromarray(frame[3*j:(3*j + 3), :, :].transpose(1, 2, 0))
                    im.save(os.path.join(save_folder, '%d_%d.png' % (i, j)))
            i += 1


if __name__ == '__main__':
    main()
