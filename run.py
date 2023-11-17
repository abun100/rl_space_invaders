import argparse
import os
import gymnasium as gym
import numpy as np

from space_invaders import gameState, model
from space_invaders.model import Model, real_reward, back_prop


def run(args):
    q_func = load_q_func(args.model, args.weights)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        repeat_action_probability=.25,
        full_action_space=True,
        obs_type=args.obs_type
    )

    epsilon = 0.03 # With probability epsilon a random action will be selected
    gamma = 0.3 # Discount factor on the computation of future rewards

    # During training, we will maintain a dataset of size buff_capacity in memory
    buff_capacity = 500

    run_game(env, q_func, epsilon, gamma, buff_capacity, args.train)

def run_game(env, q_func, epsilon, gamma, buff_capacity, train: bool = False):
    score = 0
    start = env.reset() # represents first state (very beginning of game)
    buff = [] # generated data to train the model over time

    # Creates our array of observations and preprocess them 
    # we use start[0] to represent one observation image
    state = gameState.State(start[0])
    s = state.to_numpy()
    action_vector = q_func.predict(s)

    while True:
        action = None
        if np.random.uniform() <= epsilon:
            action = np.random.randint(0, 5)
        else:
            action = np.argmax(action_vector)

        obs, reward, ended, _, _ = env.step(action)
        score += reward

        state.add_observation(obs)
        sprime = state.to_numpy()
        # we predict the future reward here to avoid doing this computation twice.
        # (once for the playing loop and one for the dataset generation in real_reward function)
        action_vector = q_func.predict(sprime)
        
        # We want our model to learn to predict the future reward given a state s,
        # for this we create a running dataset where our predictor is the 
        # State at time t (s) and the response is a discounted sum of all future reward (y).
        # Here we create a running dataset that we use to improve our model over time.
        y = real_reward(reward, action_vector, ended, gamma)
        buff.append((s, y))

        # keep the data buffer size under control
        if len(buff) > buff_capacity:
            buff.pop()

        # update weights and flush buffer
        if train and len(buff) == buff_capacity:
            back_prop(q_func, buff)
            buff = []

        s = sprime

        env.render()

        if ended:
            env.close()
            print(f'Score:{score}')
            break

def load_q_func(model_type: str, weights_file: str) -> Model:
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
    args.add_argument('--train', type=bool, default=False)

    return args.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
