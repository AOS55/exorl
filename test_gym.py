import gym
from utils import GymWrapper
import utils.env_constructor as constructor


def main():
    env = constructor.make('BipedalWalker-v3',
                           obs_type='states',
                           frame_stack=3,
                           action_repeat=1,
                           seed=42)
    print(f'action_spec is: {env.action_spec()}')
    time_step = env.reset()
    last_episode = False
    ide = 0
    while not last_episode:
        ide += 1
        # print(f'time_step: {time_step}')
        print(f'observation: {time_step.observation.shape}')
        action = env.action_space.sample()
        # print(f'action: {action}')
        time_step = env.step(action)
        if time_step.last():
            last_episode = True
            print(f'time_step: {time_step}')
            print(f'iterated through: {ide}')
            time_step = env.reset()
    env.close()


if __name__=='__main__':
    main()
