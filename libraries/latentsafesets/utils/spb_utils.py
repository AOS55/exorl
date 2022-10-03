import libraries.safe.simple_point_bot as spb
import matplotlib.pyplot as plt
import os

import numpy as np
from tqdm import tqdm


def evaluate_safe_set(s_set,
                      env,
                      file=None,
                      plot=True,
                      show=False,
                      skip=2,
                      obs_type='pixels'):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            if obs_type == 'pixels':
                state = env._state_to_image((x, y)) / 255
            else:
                state = (x/255, y/255)
            row_states.append(state)
        if obs_type == 'pixels':
            vals = s_set.safe_set_probability_np(np.array(row_states)).squeeze()
        else:
            vals = s_set.safe_set_probability_np(np.array(row_states), already_embedded=True).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y+1, ::2], data[y+1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show)

    return data


def evaluate_value_func(value_func,
                        env,
                        file=None,
                        plot=True,
                        show=False,
                        skip=2,
                        obs_type='pixels'):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            if obs_type == 'pixels':
                state = env._state_to_image((x, y)) / 255
            else:
                state = (x/255, y/255)
            row_states.append(state)
        if obs_type == 'pixels':
            vals = value_func.get_value_np(np.array(row_states)).squeeze()
        else:
            vals = value_func.get_value_np(np.array(row_states), already_embedded=True).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show)

    return data


def evaluate_constraint_func(constraint,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=2,
                             obs_type='pixels'):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            if obs_type == 'pixels':
                state = env._state_to_image((x, y)) / 255
            else:
                state = (x/255, y/255)
            row_states.append(state)
        if obs_type == 'pixels':
            vals = constraint.prob(np.array(row_states), already_embedded=False).squeeze()
        else:
            vals = constraint.prob(np.array(row_states), already_embedded=True).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def _centroids(file):
    ep = np.load(os.path.join('data/datasets/states/SimplePointBot/diayn200_500000/buffer', file))
    frames = ep['observation']
    centroids = [_get_red_centroid(frame) for frame in frames]
    centroids = frames
    return centroids

def multi_track():
    files = os.listdir('data/datasets/states/SimplePointBot/diayn200_500000/buffer')
    print(f'files: {len(files)}')
    tracks = []
    idx = 0
    for file in files:
        track = np.array(_centroids(file))
        tracks.append(track)
        idx += 1
        plt.plot(track[:, 0], track[:, 1])
        if idx > 10000: break
    plt.savefig('priors.png')
    return tracks

def _get_red_centroid(frame):
    frame = np.moveaxis(frame, 0, -1)
    red = (255, 0, 0)
    red_pixels = np.where(np.all(frame==red, axis=-1))
    centroid = [np.sum(x) / len(x) for x in red_pixels]
    return centroid
