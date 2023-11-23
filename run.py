import argparse
import logging
import os
import gymnasium as gym
import numpy as np

from space_invaders import model
from space_invaders.gameState import StateFrames, State
from space_invaders.model import DiscountFactor, Model, back_prop, ReplayBuff
from tensorflow import keras
from typing import Tuple


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run(args):
    q_func = load_q_func(args.model, args.weights)
    if args.train:
        q_func = compile_model(q_func, args.learning_rate)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        repeat_action_probability=.25,
        full_action_space=False,
        obs_type=args.obs_type
    )

    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    gamma: DiscountFactor = 0.3

    # During training, we will maintain a dataset of size buff_capacity in memory
    buff_capacity = args.buff_capacity
    batch_size = args.batch_size
    epochs = args.epochs

    try:
        run_game(env, q_func, epsilon, epsilon_decay,gamma,
             buff_capacity, epochs, batch_size, episodes=args.episodes, train=args.train)
    except KeyboardInterrupt:
        print('shutting down program, please wait...')
    except Exception as e:
        log.error(e, exc_info=True)
    finally:
        shut_down(args, q_func)
        env.close()


def run_game(
        env: gym.Env,
        q_func: Model,
        epsilon: float,
        epsilon_decay: float,
        gamma: float,
        buff_capacity: int,
        epochs: int,
        batch_size: int,
        episodes: int = 1,
        train: bool = False,
        k = 3 # frame skipping parameter
    ) -> None:
    buff: ReplayBuff = init_buffer(env, buff_capacity, k) # generated data to train the model over time

    for e in range(episodes):
        print(f'Playing episode {e} of {episodes}')

        score = 0
        state, s = reset_env(env)
        
        while True:
            action = compute_action(q_func, epsilon, train, s)

            reward, ended = step(env, k, state, action)
            score += reward

            sprime = state.to_numpy()
            update_replay_buffer(buff, buff_capacity, s, action, reward, ended, sprime)
            s = sprime # !important this needs to occur after buff is updated

            # update weights
            if train and len(buff[0]) >= buff_capacity:
                back_prop(q_func, buff, gamma, batch_size, epochs)

            if ended:
                print(f'Score:{score}')
                break

            if epsilon > .1:
                epsilon -= 1 / epsilon_decay

            env.render()

def init_buffer(env, buff_capacity, k) -> ReplayBuff:
    """
    init_buffer fills up the buffer up to its max capacity by selecting random actions
    """
    buff = ([], [], [], [], [])
    state, s = reset_env(env)

    while len(buff[0]) < buff_capacity:
        action = int(np.random.randint(0, 5))
        reward, ended = step(env, k, state, action)
        sprime = state.to_numpy()
        update_replay_buffer(buff, buff_capacity, s, action, reward, ended, sprime)
        s = sprime

        if ended:
            state, s = reset_env(env)

    return buff

def step(env, k, state, action):
    score = 0
    for _ in range(k):
        obs, reward, ended, _, _ = env.step(action)
        score += reward
        state.add_observation(obs)

        if ended:
            break
    
    return score, ended

def compute_action(q_func, epsilon, train, s):
    action_vector = q_func.predict(s)

    if train and np.random.uniform() <= epsilon:
        action = int(np.random.randint(0, 5))
    else:
        action = int(np.argmax(action_vector))

    return action           


def shut_down(args, model: Model) -> None:
    if not args.train or not args.save_on_cancel:
        return

    model.save(args.weights)


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


def compile_model(model: Model, learning_rate=0.001) -> Model:
    return model.compile(
        keras.optimizers.RMSprop(learning_rate),
        ['accuracy', 'mse']
    )


def parse_args():
    args = argparse.ArgumentParser()
    
    # Env configuration
    args.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='rgb_array')
    args.add_argument('--obs_type', type=str, choices=['rgb', 'grayscale', 'ram'], default='grayscale')

    # Model configuration
    args.add_argument('--weights', type=str, default=os.path.join('data', 'weights.h5'))
    args.add_argument('--model', type=str, choices=['dqn-basic'], default='dqn-basic')

    # Training Configuration
    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--save_on_cancel', type=bool, default=True)
    args.add_argument('--buff_capacity', type=int, default=6_000) # size of available data set
    args.add_argument('--batch_size', type=int, default=32) # when training how many samples to take
    args.add_argument('--epochs', type=int, default=1) # how many steps of gradient descent to perform ea time
    args.add_argument('--epsilon', type=float, default=1) # with probability epsilon choose random action
    args.add_argument('--epsilon_decay', type=int, default=100_000)
    args.add_argument('--learning_rate', type=float, default=0.00025)

    # Game configuration
    args.add_argument('--episodes', type=int, default=1)

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
