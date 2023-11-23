from typing import List, Tuple
import numpy as np
import tensorflow as tf
from space_invaders.environment import Action, Reward, Terminated, reset_env, step
from space_invaders.gameState import StateFrames
from space_invaders.model import Model


ReplayBuff = Tuple[List[StateFrames], 
                   List[StateFrames], 
                   List[Reward], 
                   List[Terminated], 
                   List[Action]]

DiscountFactor = float # Discount factor on the computation of future rewards


def init_buffer(env, buff_capacity) -> ReplayBuff:
    """
    init_buffer fills up the buffer up to its max capacity by selecting random actions
    """
    buff = ([], [], [], [], [])
    state, s = reset_env(env)

    while len(buff[0]) < buff_capacity:
        action = int(np.random.randint(0, 5))
        sprime, reward, ended = step(env, state, action)
        update_replay_buffer(buff, s, action, reward, ended, sprime, True)
        s = sprime

        if ended:
            state, s = reset_env(env)

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
    if not initializing:
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
        np.ndarray
    ]:
    return (
        np.stack(buff[0], axis=0),
        np.stack(buff[1], axis=0),
        np.array(buff[2]),
        np.array(buff[3]),
        np.array(buff[4])
    )


def expected_reward(
        model: Model,
        y_hat: np.ndarray,
        sprime: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        isTerminalState: np.ndarray,
        gamma: DiscountFactor
    ) -> np.ndarray:
    """
    Computes the expected reward on a batch of samples by leveraging 
    vectorized operations.
    """
    y = np.copy(y_hat)
    
    
    rprime = model.predict(sprime).numpy()

    i = np.arange(y.shape[0]) # Index hack to access all rows in the predictions
    a = action # The actions we are updating (columns of the predictions we will modify)

    t = isTerminalState == False

    r = reward + gamma * rprime[i, a] * t
    y[i, a] = r # We only update the y of those actions we know exactly what the future looks like

    return y


def back_prop(model: Model, buff: ReplayBuff, gamma: DiscountFactor,
    batch_size, epochs) -> None:
    s, sprime, action, reward, ended = unstack_buff(buff)

    total_observations = s.shape[0] # how many states do we have
    rng = np.random.default_rng()

    for _ in range(epochs):
        sample = rng.choice(total_observations, batch_size) # random states we are training on
        
        state_sample, sprime_sample, action_sample, reward_sample, ended_sample = (
            s[sample,:,:,:], 
            sprime[sample,:,:,:],
            action[sample],
            reward[sample],
            ended[sample]
        )

        with tf.GradientTape() as tape:
            y_hat = model.predict(state_sample, training=True)

            # Stop watching expected_rewards 
            y = tf.stop_gradient(expected_reward(model, y_hat.numpy(), 
                                                 sprime_sample, action_sample, reward_sample,
                                                 ended_sample, gamma))

            loss = model.loss(y, y_hat)

        grads = tape.gradient(loss, model._model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model._model.trainable_weights))
