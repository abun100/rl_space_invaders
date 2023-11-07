import argparse
import gymnasium as gym

from space_invaders import environment

def run(args):
    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        repeat_action_probability=.25,
        full_action_space=True,
        obs_type=args.obs_type
    )
    environment.run_game(env)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='rgb_array')
    args.add_argument('--obs_type', type=str, choices=['rgb', 'grayscale', 'ram'], default='grayscale')

    return args.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
