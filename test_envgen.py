from libraries.open_ended.bipedal_walker_custom import BipedalWalkerCustom
from libraries.open_ended.utils import init_env_params, make_env_config, DR, make_new_params
from utils.env_constructor import make

def main():
    env = make('BipedalCustom', 'states', 1, 1, 1, False)
    print(f'env: {env}')
    init_params = init_env_params(42, DR)
    # print(init_params)
    init_config = make_env_config('test', init_params)
    new_params = make_new_params(init_params)
    # print(new_params)
    new_config = make_env_config('new_config', new_params)
    # env = BipedalWalkerCustom(init_config)
    observation = env.reset()
    # print(f'init_config: {init_config}')
    # print(f'new_config: {new_config}')
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)
        print(f'env.render: {env.render()}')
        print(f'obs: {observation}')
        print(f'reward: {reward}')
        print(f'terminated: {terminated}')
        print(f'info: {info}')
        if terminated:
            observation = env.reset()

    env.close()

    return None


if __name__=='__main__':
    main()