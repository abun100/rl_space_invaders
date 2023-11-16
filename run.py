import argparse
import os
import gymnasium as gym
import numpy as np

from space_invaders import model, gameState
from space_invaders.model import real_reward


def run(args):
    q_func = load_q_func(args.model, args.weights)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        repeat_action_probability=.25,
        full_action_space=True,
        obs_type=args.obs_type
    )

    Lambda = 0.3

    run_game(env, q_func, Lambda)

def run_game(env, q_func, Lambda):
    score = 0
    start = env.reset() # represents first state (very beginning of game)
    buff = []

    # Creates our array of observations and preprocess them 
    # we use start[0] to represent the observation image
    state = gameState.State(start[0])
    s = state.to_numpy()
    action_vector = q_func.predict(s)

    while True:
        action = np.argmax(action_vector)
        obs, reward, ended, _, _ = env.step(action)
        score += reward

        state.add_observation(obs)
        sprime = state.to_numpy()
        action_vector = q_func.predict(sprime)
        
        y = real_reward(reward, action_vector, ended, Lambda)
        buff.append((s, y))

        s = sprime

        env.render()

        if ended:
            env.close()
            print(f'Score:{score}')
            break

def load_q_func(model_type: str, weights_file: str) -> model.Model:
    m = model.DQNBasic()
    m.load(weights_file)
    
    return m

def parse_args():
    args = argparse.ArgumentParser()
    
    # Env configuration
    args.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='rgb_array')
    args.add_argument('--obs_type', type=str, choices=['rgb', 'grayscale', 'ram'], default='grayscale')

    # Model configuration
    args.add_argument('--weights', type=str, default=os.path.join('data', 'weights.h5'))
    args.add_argument('--model', type=str, choices=['dqn-basic'], default='dqn-basic')

    return args.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
