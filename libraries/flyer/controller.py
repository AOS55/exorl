import os
import argparse
import imageio
from PIL import Image as im
from PIL import ImageFilter
import matplotlib.pyplot as plt 
from scipy import ndimage
import cv2
import numpy as np
import pandas as pd
from flyer import Env
from scipy import ndimage
from ga import make_ga, TrackCalc, make_fitness_func
import pygame
from PIL import Image


def get_longest_region(array, val):
    region_array = np.zeros(array.shape)
    for idx, x in enumerate(array):
        for idy, y in enumerate(x):
            if y == val:
                region_array[idx, idy] = 1
    position = (0, 0)
    max_length = 0
    direction = "NS"
    longest_region = {"position": position, "direction": direction, "length": max_length}
    for idx, x in enumerate(region_array):
        for idy, y in enumerate(x):
            if y==1:
                length = 1
                position = (idx, idy)
                while y == 1 and idy < region_array.shape[1] - 1:
                    length += 1
                    idy += 1
                    y = region_array[idx, idy]
                if length > max_length:
                    max_length = length
                    longest_region = {"position": position, "direction": "NS", "length": length}
    
    rot_array = np.rot90(region_array)
    for idy, y in enumerate(rot_array):
        for idx, x in enumerate(y):
            if x==1:
                length = 1
                position = (idx, idy)
                while x == 1 and idx < region_array.shape[0] - 1:
                    length += 1
                    idx += 1
                    x = region_array[idx, idy]
                if length > max_length:
                    max_length = length
                    longest_region = {"position": position, "direction": "EW", "length": length}
    return longest_region

def get_grass_centers(obs):

    # Set Grass to 1 and NOT Grass to 0
    grass_array = np.zeros(obs.shape)
    grass_value = 1
    for idx, x in enumerate(obs):
        for idy, y in enumerate(x):
            if y == grass_value:
                grass_array[idx, idy] = 1
    grass_array = grass_array.astype(int)

    longest_region = get_longest_region(grass_array, 1)
    print(f"longest_region: {longest_region}")
    if longest_region["direction"] == "EW":
        goal_pos = (longest_region["position"][0] + (longest_region["length"] // 2), longest_region["position"][1])
    if longest_region["direction"] == "NS":
        goal_pos = (longest_region["position"][0], longest_region["position"][1] + (longest_region["length"] // 2))

    return goal_pos


def make_grass_image(obs, goal_pos):
    grass_array = np.zeros(obs.shape + (3,))
    grass_value = 1
    for idx, x in enumerate(obs):
        for idy, y in enumerate(x):
            if y == grass_value:
                grass_array[idx, idy, :] = np.array((0, 255, 0))
            else:
                grass_array[idx, idy, :] = np.array((0, 0, 0))
    grass_array[goal_pos] = (255, 0, 0)
    grass_array = grass_array.astype('uint8')
    img = Image.fromarray(grass_array)
    img.show()

def get_action_traj(ga_params, goal_pos):
    start_pos = np.array([128, 128, 64])
    goal_pos = list(goal_pos)
    goal_pos.append(0)
    airspeed = 5
    fitness_func = make_fitness_func(start_pos=start_pos, goal_pos=np.array(goal_pos), airspeed=airspeed)
    GA = make_ga(ga_params, fitness_func)
    GA.run()
    best_action, best_fitness, best_id = GA.best_solution()
    tc = TrackCalc(start_pos, goal_pos, airspeed)
    pos = tc.generate_trajectory(best_action)
    print(f"best_final_pos: {pos[-1]}")
    
    return best_action

def make_dataset(env: Env, n_episodes: int, ga_params: dict, save_dir: str = "data"):
    for ide in range(n_episodes):
        ida = 0
        states = []
        action_list = []
        rewards = []
        dones = []
        env._mode = "map"
        obs = env.reset()
        done = False
        goal_pos = get_grass_centers(obs)
        actions = get_action_traj(ga_params, goal_pos)
        actions = np.append(actions, [0, 0, 0])
        env._mode = "image"
        dir_path = os.path.join(save_dir, f'{ide}.npz')
        while done == False:
            action = actions[ida]
            obs, reward, done, _ = env.step(action)
            states.append(obs)
            action_list.append(action)
            rewards.append(reward)
            dones.append(done)
            ida += 1
        np.savez(dir_path, obs=np.array(states), actions=np.array(action_list), rewards=np.array(rewards), dones=np.array(dones))
    return None

def generate_videos(env, actions):
    pygame.init()
    screen = pygame.display.set_mode([800, 800])
    clock = pygame.time.Clock()
    ida = 0
    done = False
    while done == False:
        ida += 1
        pygame.event.pump()
        obs, reward, done, _ = env.step(actions[ida])
        surface = pygame.surfarray.make_surface(
            obs.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(10)
    pygame.quit()

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(256, 256, 64))
    parser.add_argument('--view', type=int, default=10)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--rod', type=int, default=1)
    parser.add_argument('--airspeed', type=int, default=5)
    parser.add_argument('--ldg_dist', type=int, default=4)
    parser.add_argument('--alt', type=int, default=128)
    parser.add_argument('--engine_fail', type=bool, default=True)
    parser.add_argument('--health', type=int, default=1)
    parser.add_argument('--window', type=int, default=800)
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--water_present', type=bool, default=True)
    parser.add_argument('--custom_world', type=str, default=None)
    parser.add_argument('--mode', type=str, default="map")
    args = parser.parse_args()

    env = Env(area=args.area, view=args.view, size=args.window, length=args.length, rod=args.rod, airspeed=args.airspeed, ldg_dist=args.ldg_dist, engine_fail=args.engine_fail, health=args.health, water_present=args.water_present, seed=args.seed, custom_world=args.custom_world, mode=args.mode)

    # obs = env.reset()
    # print(f'obs.shape: {obs.shape}')
    # print(f'obs: {obs}')
    # goal_pos = get_grass_centers(obs)
    # print(f"goal_pos: {goal_pos}")

    ga_params = {"num_generations": 20000,
                 "num_parents_mating": 4,
                 "sol_per_pop": 8,
                 "num_genes": 64,  # action size
                 "init_range_low": 0,
                 "init_range_high": 3,
                 "parent_selection_type": "sss",
                 "keep_parents": 1,
                 "crossover_type": "single_point",
                 "mutation_type": "random",
                 "mutation_percent_genes": 10,
                 "gene_type": int}

    # make_grass_image(obs, goal_pos)

    # actions = get_action_traj(ga_params, goal_pos)
    # env._mode = "image"
    # generate_videos(env, actions)

    # actions = get_action_traj(ga_params)
    # env._mode = "image"
    # ida = 0
    # done = False
    # while done == False:
    #     obs, rewards, done, info = env.step(actions[ida])
    #     env.render()
    #     ida += 1

    make_dataset(env, 100, ga_params)

    return None

if __name__=="__main__":
    main()
