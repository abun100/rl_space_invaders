from tensorflow import keras
from typing import List

import numpy as np

from space_invaders.gameState import State


ACTIONS_SPACE = 6


class Model():
    _model: keras.Model = None

    def save(self, filepath: str) -> None:
        if self._model == None:
            raise Exception('_model property should be initialized before saving weights')
        
        self._model.save_weights(filepath, save_format='h5')
    
    def load(self, filepath: str) -> None:
        if self._model == None:
            raise Exception('_model property should be initialized before loading weights')
        
        self._model.load_weights(filepath, by_name=True)

    def predict(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def fit(self, state: List, y: List) -> None:
        raise NotImplementedError()


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

    def predict(self, state: np.ndarray) -> np.ndarray:
        shape_error = Exception("expected input size is (None, 84, 84, 4) where None represents any batch size")
        shape = state.shape
        expected_shape = (84, 84, 4)

        if len(shape) < 3:
            raise shape_error
        elif len(shape) == 3 and shape != expected_shape:
            raise shape_error
        elif len(shape) == 4 and shape[1:] != expected_shape:
            raise shape_error
        elif len(shape) > 4: raise shape_error

        # Keras documentation recommends only using model.predict method when
        # the predicting a batch of size > 1. Otherwise, model(x) is the best
        # way of computing y.
        x = state
        if len(shape) == 3 or shape[0] == 1:
            x = x.reshape((1, 84, 84, 4))
            return self._model(x)
        
        return self._model.predict(x)
    
def real_reward(reward:int, prediction, ended:bool, Lambda = float) -> np.ndarray:
    if ended:
        return reward + Lambda * prediction
    else:
        return reward
