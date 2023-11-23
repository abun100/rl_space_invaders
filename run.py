import argparse
import logging
import os
import gymnasium as gym
import numpy as np

from space_invaders import model
from space_invaders.environment import reset_env, step
from space_invaders.model import Model, compile_model, compute_action
from space_invaders.training import DiscountFactor, back_prop, init_buffer, update_replay_buffer


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run(args):
    q_func = load_q_func(args.model, args.weights)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        full_action_space=False,
        obs_type=args.obs_type,
        frameskip=3
    )

    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    gamma: DiscountFactor = 0.3

    # During training, we will maintain a dataset of size buff_capacity in memory
    if args.train:
        buff_capacity = args.buff_capacity
    else:
        buff_capacity = 0
    batch_size = args.batch_size
    epochs = args.epochs

    try:
        play_game(env, q_func, epsilon, epsilon_decay,gamma,
             buff_capacity, epochs, batch_size, episodes=args.episodes, train=args.train)
    except KeyboardInterrupt:
        log.info('shutting down program, please wait...')
    except Exception as e:
        log.error(e, exc_info=True)
    finally:
        shut_down(args, q_func)
        env.close()


def play_game(
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
    ) -> None:
    buff = init_buffer(env, buff_capacity) # generated data to train the model over time

    for e in range(episodes):
        log.info(f'Playing episode {e} of {episodes}')

        score = 0
        state, s = reset_env(env)
        
        while True:
            action = compute_action(q_func, epsilon, train, s)
            sprime, reward, ended, died = step(env, state, action)
            score += reward

            update_replay_buffer(buff, s, action, reward, died, sprime)
            s = sprime # !important this needs to occur after buff is updated

            # update weights
            if train:
                back_prop(q_func, buff, gamma, batch_size, epochs)

            if ended:
                log.info(f'Score:{score}')
                break

            if epsilon > .1:
                epsilon -= 1 / epsilon_decay

            env.render()


def shut_down(args, model: Model) -> None:
    if not args.train or not args.save_on_cancel:
        return

    model.save(args.weights)


def load_q_func(model_type: str, weights_file: str) -> Model:
    m = model.DQNBasic()
    m.load(weights_file)
    m = compile_model(m, args.learning_rate)
    return m


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
    args.add_argument('--epsilon', type=float, default=.25) # with probability epsilon choose random action
    args.add_argument('--epsilon_decay', type=int, default=1_000)
    args.add_argument('--learning_rate', type=float, default=0.00025)

    # Game configuration
    args.add_argument('--episodes', type=int, default=1)

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
