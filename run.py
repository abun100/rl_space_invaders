import argparse
import logging
import os
import gymnasium as gym

from space_invaders import model
from space_invaders.environment import reset_env, step, ReplayBuff, init_buffer
from space_invaders.model import Model, compile_model, compute_action
from space_invaders.training import DiscountFactor, back_prop


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run(args):
    q_func = load_q_func(args)

    env = gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode=args.render_mode,
        full_action_space=False,
        obs_type=args.obs_type,
        frameskip=3
    )

    epsilon = args.epsilon
    epsilon_decay = 1 / args.epsilon_decay
    gamma: DiscountFactor = args.discount_factor

    # During training, we will maintain a dataset of size buff_capacity in memory
    if args.train:
        replay_buffer = init_buffer(env, args.buff_capacity)
    else:
        replay_buffer = None
    batch_size = args.batch_size
    epochs = args.epochs

    try:
        play_game(
            env, q_func, epsilon, epsilon_decay, gamma,
            epochs, batch_size,
            episodes=args.episodes,
            train=args.train,
            buff=replay_buffer
        )
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
        epochs: int,
        batch_size: int,
        episodes: int = 1,
        train: bool = False,
        buff: ReplayBuff | None = None,
    ) -> None:
    for e in range(episodes):
        log.info(f'Playing episode {e} of {episodes}')

        score = 0
        state = reset_env(env)
        
        while True:
            action = compute_action(q_func, epsilon, train, state)
            state, reward, ended = step(env, state, action, buff)
            score += reward

            # update weights
            if train:
                back_prop(q_func, buff, gamma, batch_size, epochs)

            if ended:
                log.info(f'Score:{score}')
                break

            if epsilon > .1:
                epsilon -= epsilon_decay

            env.render()


def shut_down(args, model: Model) -> None:
    if not args.train or not args.save_on_cancel:
        return

    model.save(args.weights)


def load_q_func(args) -> Model:
    m = model.DQNBasic()
    m.load(args.weights)
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
    args.add_argument('--buff_capacity', type=int, default=3_000) # size of available data set
    args.add_argument('--batch_size', type=int, default=32) # when training how many samples to take
    args.add_argument('--epochs', type=int, default=1) # how many steps of gradient descent to perform ea time
    args.add_argument('--epsilon', type=float, default=.25) # with probability epsilon choose a random action
    args.add_argument('--epsilon_decay', type=int, default=1_000)
    args.add_argument('--learning_rate', type=float, default=0.000015)
    args.add_argument('--discount_factor', type=float, default=0.99)

    # Game configuration
    args.add_argument('--episodes', type=int, default=1)

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run(args)
