import argparse
import os
import gymnasium as gym
import numpy as np

from space_invaders import model
from space_invaders.gameState import StateFrames, State
from space_invaders.model import DiscountFactor, Model, back_prop, ReplayBuff
from typing import Tuple


def run(args):
    q_func = load_q_func(args.model, args.weights)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        repeat_action_probability=.25,
        full_action_space=False,
        obs_type=args.obs_type
    )

    epsilon = 0.03 # With probability epsilon a random action will be selected
    gamma: DiscountFactor = 0.3

    # During training, we will maintain a dataset of size buff_capacity in memory
    buff_capacity = 500

    run_game(env, q_func, epsilon, gamma, buff_capacity, train=args.train)

def run_game(
        env: gym.Env,
        q_func: Model,
        epsilon: float,
        gamma: float,
        buff_capacity: int,
        episodes: int = 1,
        train: bool = False
    ) -> None:
    buff: ReplayBuff = ([], [], [], [], []) # generated data to train the model over time

    for _ in range(episodes):
        score = 0
        state, s = reset_env(env)
        while True:
            action_vector = q_func.predict(s)
            
            action = None
            if np.random.uniform() <= epsilon:
                action = int(np.random.randint(0, 5))
            else:
                action = int(np.argmax(action_vector))

            obs, reward, ended, _, _ = env.step(action)
            score += reward

            state.add_observation(obs)
            sprime = state.to_numpy()
            update_replay_buffer(buff, buff_capacity, s, action, reward, ended, sprime)
            s = sprime # !important this needs to occur after buff is updated

            # update weights
            if train and len(buff) >= buff_capacity:
                back_prop(q_func, buff, gamma)

            env.render()

            if ended:
                print(f'Score:{score}')
                break
    
    env.close()
            


def update_replay_buffer(buff, cap, s, action, reward, ended, sprime):
    # keep the data buffer size under control
    if len(buff[0]) > cap:
        for i in range(len(buff)):
            buff[i].pop()
    
    # add new observation
    buff[0].append(s)
    buff[1].append(sprime)
    buff[2].append(action)
    buff[3].append(reward)
    buff[4].append(ended)


def reset_env(env: gym.Env) -> Tuple[State, StateFrames]:
    # Create array of observations and preprocess them 
    # we use start[0] to represent one observation image
    
    start = env.reset()
    state = State(start[0])
    s = state.to_numpy()
    
    return [state, s]


def load_q_func(model_type: str, weights_file: str) -> Model:
    m = model.DQNBasic()
    m.load(weights_file)
    
    return m


def parse_args():
    args = argparse.ArgumentParser()
    
    # Env configuration
    args.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='human')
    args.add_argument('--obs_type', type=str, choices=['rgb', 'grayscale', 'ram'], default='grayscale')

    # Model configuration
    args.add_argument('--weights', type=str, default=os.path.join('data', 'weights.h5'))
    args.add_argument('--model', type=str, choices=['dqn-basic'], default='dqn-basic')
    args.add_argument('--train', type=bool, default=False)

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
