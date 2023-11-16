import argparse
import os
import gymnasium as gym
import numpy as np

from space_invaders import environment, model, gameState


def run(args):
    q_func = load_q_func(args.model, args.weights)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        repeat_action_probability=.25,
        full_action_space=True,
        obs_type=args.obs_type
    )
    run_game(env, q_func)

def run_game(env, q_func):
    score = 0
    start = env.reset() #represents first state (very begining of game)

    #creates our array of observations and preprocess them 
    #we use start[0] to represent the observation image
    state_frames = gameState.State(start[0]) 

    #set our state to the stacked_frames 
    state = state_frames.obs

    

    while True:
        # action_vector = q_func.predict(state_transformer(state))
        action_vector = q_func.predict(state)

        # Map action_vector to action
        action = np.argmax(action_vector)

        obs, reward, done, truncate, info = env.step(action)
        score += reward

        # With observation update state
        # let assume that newob -> is a pre-processed obs from env
        
        state = np.concatenate((state[:,:,1:], newob), axis=2)

        env.render()

        if done:
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
