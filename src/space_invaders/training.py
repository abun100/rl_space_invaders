import numpy as np
import tensorflow as tf

from space_invaders.model import Model
from space_invaders.environment import ReplayBuff, unstack_buff


DiscountFactor = float # Discount factor on the computation of future rewards


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
    a = action # The actions we are updating (specific item of the predictions we will modify)

    t = isTerminalState == False

    r = reward + gamma * rprime[i, a] * t
    y[i, a] = r # We only update the y of those actions we know what the future looks like

    return y


def back_prop(model: Model, buff: ReplayBuff, gamma: DiscountFactor,
    batch_size: int, epochs: int) -> None:
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
            y = tf.stop_gradient(
                expected_reward(
                    model, y_hat.numpy(), 
                    sprime_sample,
                    action_sample,
                    reward_sample,
                    ended_sample,
                    gamma
                )
            )

            loss = model.loss(y, y_hat)

        grads = tape.gradient(loss, model._model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model._model.trainable_weights))
