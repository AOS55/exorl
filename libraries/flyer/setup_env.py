import argparse
from flyer import Env


def setup_env():
    """
    Setup flyer environment based on parser arguments, uses default environment_config

    :return: environment object, record boolean, seed integer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(256, 256, 128))
    parser.add_argument('--view', type=int, default=10)
    parser.add_argument('--length', type=int, default=1000)
    parser.add_argument('--rod', type=int, default=1)
    parser.add_argument('--airspeed', type=int, default=5)
    parser.add_argument('--ldg_dist', type=int, default=4)
    parser.add_argument('--alt', type=int, default=128)
    parser.add_argument('--engine_fail', type=bool, default=True)
    parser.add_argument('--health', type=int, default=4)
    parser.add_argument('--window', type=int, default=800)
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--water_present', type=bool, default=True)
    args = parser.parse_args()

    env = Env(area=args.area, view=args.view, size=args.window, length=args.length, rod=args.rod,
              airspeed=args.airspeed, ldg_dist=args.ldg_dist, engine_fail=args.engine_fail, health=args.health,
              water_present=args.water_present, seed=args.seed)
    return env, args.record, args.seed


def setup_custom_env(env_config: dict):
    """
    Setup an environment with a custom config file

    :param env_config: a custom named tuple defining the specific environment attributes
    :return: environment object, record boolean, seed integer
    """
    env, record, seed = setup_env()
    env.set_env_config(env_config)
    return env, record, seed


def setup_env_no_parser(env_config: dict):
    """
    Setup an environment without the use of a parser

    :param env_config:
    :return: flyer environment
    """
    env = Env(env_config=env_config, area=(256, 256, 128), view=10, size=512, length=1000, rod=1, airspeed=5,
              ldg_dist=4, engine_fail=True, health=4, water_present=True, seed=None)
    return env

def setup_env_fully_custom(config: dict):
    """
    Setup environment in its entirety with custom configuration

    :param config: configuration dictionary with all the information to construct the gym Env
    :return:
    """
    env = Env(env_config=config['env_config'], area=['area'], view=['view'], size=['size'],
             length=['length'], rod=['rod'], airspeed=['airspeed'], ldg_dist=['ldg_dist'],
             engine_fail=['engine_fail'], health=['health'], water_present=['water_present'], seed=['seed'])
    return env