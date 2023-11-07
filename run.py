import argparse
import os
import gymnasium as gym

from space_invaders import environment, model

def run(args):
    model = load_model(args.model, args.weights, args.init_weights)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        repeat_action_probability=.25,
        full_action_space=True,
        obs_type=args.obs_type
    )
    environment.run_game(env)

def load_model(model_type: str, weights_file: str, init_weights: bool) -> model.Model:
    return model.DQNGrayScale()

def parse_args():
    args = argparse.ArgumentParser()
    
    # Env configuration
    args.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='rgb_array')
    args.add_argument('--obs_type', type=str, choices=['rgb', 'grayscale', 'ram'], default='grayscale')

    # Model configuration
    args.add_argument('--init_weights', type=bool, default=False)
    args.add_argument('--weights', type=str, default=os.path.join('data', 'weights.npy'))
    args.add_argument('--model', type=str, choices=['dqn-grayscale'], default='dqn-grayscale')

    return args.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
