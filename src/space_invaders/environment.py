import gymnasium as gym

from typing import Tuple
from space_invaders.gameState import State, StateFrames


Action = int

Reward = float

Terminated = bool

Died = bool


def step(
        env: gym.Env, 
        state: StateFrames,
        action: Action
    ) -> Tuple[StateFrames, Reward, Terminated, Died]:
    """
    Takes an step in the environment. It will repeat the action for k steps.
    """
    prev_lives = env.ale.lives()
    obs, reward, ended, _, _ = env.step(action)
    curr_lives = env.ale.lives()
    state.add_observation(obs)

    return (state.to_numpy(), reward, ended, prev_lives < curr_lives)


def reset_env(env: gym.Env) -> Tuple[State, StateFrames]:
    # Create array of observations and preprocess them 
    # we use start[0] to represent one observation image
    
    start = env.reset()
    state = State(start[0])
    s = state.to_numpy()
    
    return [state, s]
