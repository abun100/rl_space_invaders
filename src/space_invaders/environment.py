import gymnasium as gym
import numpy as np

from typing import List, Tuple
from typing import Tuple
from space_invaders.gameState import Action, State, Reward, Terminated, StateFrames

ReplayBuff = Tuple[List[StateFrames], 
                   List[StateFrames], 
                   List[Reward], 
                   List[Terminated], 
                   List[Action]]


def init_buffer(env, buff_capacity) -> ReplayBuff:
    """
    init_buffer fills up the buffer up to its max capacity by selecting random actions
    """
    buff = ([], [], [], [], [])
    state = reset_env(env)

    while len(buff[0]) < buff_capacity:
        action = int(np.random.randint(0, 5))
        state, _, ended = step(env, state, action, buff, True)

        if ended:
            state = reset_env(env)

    return buff


def update_replay_buffer(
        buff: ReplayBuff, 
        s: StateFrames,
        action: Action,
        reward: Reward,
        ended: Terminated,
        sprime: StateFrames,
        initializing: bool = False
    ):
    # keep the data buffer size under control
    if not initializing and len(buff[0]) > 0:
        for i in range(len(buff)):
            buff[i].pop()
    
    # add new observation
    buff[0].append(s)
    buff[1].append(sprime)
    buff[2].append(action)
    buff[3].append(reward)
    buff[4].append(ended)


def unstack_buff(buff: ReplayBuff) -> Tuple[
        np.ndarray,
        np.ndarray, 
        np.ndarray, 
        np.ndarray, 
        np.ndarray,
    ]:
    return (
        np.stack(buff[0], axis=0),
        np.stack(buff[1], axis=0),
        np.array(buff[2]),
        np.array(buff[3]),
        np.array(buff[4])
    )


def step(
        env: gym.Env, 
        state: State,
        action: Action,
        buff: ReplayBuff = None,
        init_buff: bool = False,
    ) -> Tuple[State, Reward, Terminated]:
    """
    Takes an step in the environment. It will repeat the action for k steps.
    """
    s = state.to_numpy()
    prev_lives = env.ale.lives()
    
    obs, reward, ended, _, _ = env.step(action)
    state.add_observation(obs)
    died = env.ale.lives() < prev_lives

    if not buff is None:
        sprime = state.to_numpy()
        update_replay_buffer(buff, s, action, reward, died, sprime, init_buff)

    return (state, reward, ended)


def reset_env(env: gym.Env) -> State:
    # Create array of observations and preprocess them 
    # we use start[0] to represent one observation image
    
    start = env.reset()
    state = State(start[0])
    
    return state
