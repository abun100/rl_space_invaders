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
    gamma: DiscountFactor = 0.3

    # During training, we will maintain a dataset of size buff_capacity in memory
    buff_capacity = args.buff_capacity
    batch_size = args.batch_size
    epochs = args.epochs

    try:
        run_game(env, q_func, epsilon, gamma,
             buff_capacity, epochs, batch_size, episodes=args.episodes, train=args.train)
    except Exception as e:
        log.error(e, exc_info=True)
    finally:
        shut_down(args, q_func)
        env.close()


def run_game(
        env: gym.Env,
        q_func: Model,
        epsilon: float,
        gamma: float,
        buff_capacity: int,
        epochs: int,
        batch_size: int,
        episodes: int = 1,
        train: bool = False
    ) -> None:
    buff: ReplayBuff = ([], [], [], [], []) # generated data to train the model over time
    
    for _ in range(episodes):
        score = 0
        state, s = reset_env(env)
        lives = env.ale.lives()
        
        while True:
            action_vector = q_func.predict(s)
            
            action = None
            if np.random.uniform() <= epsilon:
                action = int(np.random.randint(0, 5))
            else:
                action = int(np.argmax(action_vector))

            obs, reward, ended, _, _ = env.step(action)
            score += reward

            was_life_lost = False
            curr_lives = env.ale.lives()
            if lives != curr_lives:
                was_life_lost = True
                lives = curr_lives

            state.add_observation(obs)
            sprime = state.to_numpy()
            update_replay_buffer(buff, buff_capacity, s, action, reward, was_life_lost, sprime)
            s = sprime # !important this needs to occur after buff is updated

            # update weights
            if train and len(buff[0]) >= buff_capacity:
                back_prop(q_func, buff, gamma, batch_size, epochs)

            if ended:
                print(f'Score:{score}')
                break

            env.render()            


def shut_down(args, model: Model) -> None:
    print('shutting down program, please wait...')
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
        keras.optimizers.SGD(learning_rate),
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
    args.add_argument('--buff_capacity', type=int, default=500) # size of available data set
    args.add_argument('--batch_size', type=int, default=16) # when training how many samples to take
    args.add_argument('--epochs', type=int, default=1) # how many steps of gradient descent to perform ea time
    args.add_argument('--epsilon', type=float, default=0.03) # with probability epsilon choose random action
    args.add_argument('--learning_rate', type=float, default=.01)

    # Game configuration
    args.add_argument('--episodes', type=int, default=1)

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
