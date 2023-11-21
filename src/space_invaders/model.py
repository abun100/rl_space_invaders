import numpy as np
import tensorflow as tf

from tensorflow import keras
from typing import List, Tuple
from space_invaders.environment import Action, Reward, Terminated
from space_invaders.gameState import StateFrames


ACTIONS_SPACE = 6

ReplayBuff = Tuple[List[StateFrames], List[StateFrames], List[Reward], List[Terminated], List[Action]]

DiscountFactor = float # Discount factor on the computation of future rewards


class Model:
    _model: keras.Model = None

    def save(self, filepath: str) -> None:
        if self._model == None:
            raise Exception('_model property should be initialized before saving weights')
        
        self._model.save_weights(filepath, save_format='h5')
    
    def load(self, filepath: str) -> None:
        if self._model == None:
            raise Exception('_model property should be initialized before loading weights')
        
        self._model.load_weights(filepath, by_name=True)

    def predict(self, state: StateFrames) -> np.ndarray:
        raise NotImplementedError()
    
    def compile(self, optimizer: keras.optimizers.Optimizer, metrics: List[str]) -> type["Model"]:
        assert isinstance(optimizer, keras.optimizers.Optimizer)
        assert len(metrics) > 0
        for m in metrics:
            assert isinstance(m, str)

        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = keras.losses.mse
        
        return self


class DQNBasic(Model):
    """
    DQNBasic is a keras implementation of the model described 
    in DeepMind's paper "Playing Atari with Deep Reinforcement Learning".

    The input for the model is a numpy array of 4 images (in gray scale)
    cropped to be 80 x 80 pixels. The output is a numpy array of length 6, where
    each action's value is represented by one of the items in the array.
    """
    
    def __init__(self):
        self._model = self.__build_model()

    def __build_model(self) -> keras.Model:
        inputs = keras.Input(shape=(84,84,4))
        
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=(8,8),
            strides=4,
            activation='relu')(inputs)
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(4,4),
            strides=2,
            activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(ACTIONS_SPACE, activation='linear')(x)

        return keras.Model(inputs=inputs, outputs=outputs)

    def predict(self, state: np.ndarray, training=False) -> np.ndarray:
        x = state
        if len(x.shape) == 3:
            x = x.reshape((1, 84, 84, 4))
        
        return self._model(x, training=training)


def expected_reward(
        model: Model,
        y_hat: np.ndarray,
        sprime: List[StateFrames],
        action: List[Action],
        reward: List[Reward],
        isTerminalState: List[Terminated],
        gamma: DiscountFactor
    ) -> np.ndarray:
    """
    Computes the expected reward on a batch of samples by leveraging 
    vectorized operations.
    """
    y = np.copy(y_hat)
    
    sprime = np.stack(sprime, axis=0)
    rprime = model.predict(sprime).numpy()

    i = np.arange(y.shape[0]) # Index hack to access all rows in the predictions
    a = np.array(action) # The actions we are updating (columns of the predictions we will modify)

    t = np.array(isTerminalState) == False

    r = np.array(reward) + gamma * rprime[i, a] * t
    y[i, a] = r # We only update the y of those actions we know exactly what the future looks like

    return y


def back_prop(model: Model, buff: ReplayBuff, gamma: DiscountFactor) -> None:
    s, sprime, action, reward, ended = buff
    s = np.stack(s, axis=0)

    with tf.GradientTape() as tape:
        y_hat = model.predict(s, training=True)

        # Stop watching expected_rewards 
        y = tf.stop_gradient(expected_reward(model, y_hat.numpy(), sprime, action, reward, ended, gamma))
        
        loss = model.loss(y, y_hat)

    grads = tape.gradient(loss, model._model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model._model.trainable_weights))
