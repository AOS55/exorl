import argparse
import numpy as np
import wandb
from flyer import Env

from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(256, 256, (55, 64)))
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
    args = parser.parse_args()

    print(f'vars(args): {vars(args)}')

    env = Env(area=args.area, view=args.view, size=args.window, length=args.length, rod=args.rod, airspeed=args.airspeed, ldg_dist=args.ldg_dist, engine_fail=args.engine_fail, health=args.health, water_present=args.water_present, seed=args.seed, custom_world=args.custom_world)

    config = vars(args)

    run = wandb.init(
        project="flyer-sb3",
        config=config,
        sync_tensorboard=True,
        save_code=True
    )

    obs = env.reset()
    print(f'obs.shape: {obs.shape}')
    train(env)
    run.finish()

def train(env):
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_flyer_tensorboard/", device='cuda')
    print(f'callbacks: {model._init_callback}')
    model.learn(
        total_timesteps=2500000,
        tb_log_name="mac_run",
        callback=WandbCallback(
            gradient_save_freq=10,
            verbose=2
        )
    )

    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


if __name__=="__main__":
    main()