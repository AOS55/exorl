import gym
from utils.wrappers import GymWrapper
from utils.env_constructor import make


def main():
    env = make('SimplePointBot',
                obs_type='states',
                frame_stack=3,
                action_repeat=1,
                seed=42)
    print(f'action_spec is: {env.action_spec()}')
    time_step = env.reset()
    print(f'render: {env.render()}')
    last_episode = False
    ide = 0
    while not last_episode:
        ide += 1
        # print(f'time_step: {time_step}')
        # print(f'observation: {time_step.observation.shape}')
        action = env.action_space.sample()
        # print(f'action: {action}')
        time_step = env.step(action)
        print(env._env._env._env._env.get_info())
        if time_step.last():
            last_episode = True
            # print(f'time_step: {time_step}')
            # print(f'iterated through: {ide}')
            time_step = env.reset()

            print(f'dir(time_step): {dir(time_step)}')
            print(env._env._env._env._env.get_info())
            # print(f'info is: {time_step.get_info()}')
    env.close()


if __name__=='__main__':
    main()
