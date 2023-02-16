import argparse
import imageio
try:
    import pygame
except ImportError:
    print('Please install pygame package to use GUI')
    raise
from flyer import Env


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

    keymap = {
        pygame.K_a: 'left',
        pygame.K_d: 'right',
    }

    print('Actions:')
    for key, action in keymap.items():
        print(f' {pygame.key.name(key)}: {action}')

    env = Env(area=args.area, view=args.view, size=args.window, length=args.length, rod=args.rod,
              airspeed=args.airspeed, ldg_dist=args.ldg_dist, engine_fail=args.engine_fail, health=args.health,
              water_present=args.water_present, seed=args.seed, custom_world=args.custom_world)
    env.reset()
    health = None
    return_ = 0
    if args.record:
        frames = []

    pygame.init()
    screen = pygame.display.set_mode([args.window, args.window])
    clock = pygame.time.Clock()
    running = True
    while running:
        action = None
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
                action = keymap[event.key]
        if action is None:
            pressed = pygame.key.get_pressed()
            for key, action in keymap.items():
                if pressed[key]:
                    break
            else:
                action = 'noop'

        messages = []
        obs, reward, done, _ = env.step(env.action_names.index(action))
        if env.id_step > 0 and env.id_step % 100 == 0:
            messages.append(f'Time step: {env.id_step}')
        if not health or health != env.aircraft.health:
            health = env.aircraft.health
            messages.append(f'Health: {health}/{args.health}')
        if reward:
            messages.append(f'Reward: {reward}')
            return_ += reward
        if done:
            messages.append(f'Episode end: {done}')
            env.reset()
        if messages:
            print('\n', '\n'.join(messages), sep='')

        if args.record:
            frames.append(obs['image'])
        surface = pygame.surfarray.make_surface(
            obs.transpose((1, 0, 2))
        )
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print('Return:', return_)
    if args.record:
        imageio.mimsave(args.record, frames)
        print('Saved', args.record)


if __name__ == '__main__':
    main()
