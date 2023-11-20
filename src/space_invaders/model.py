import numpy as np

from tensorflow import keras
from typing import List, Tuple
from space_invaders.environment import Action, Reward, Terminated
from space_invaders.gameState import StateFrames


ACTIONS_SPACE = 6

ReplayBuff = Tuple[List[StateFrames], List[StateFrames], List[Reward], List[Terminated], List[Action]]

DiscountFactor = float # Discount factor on the computation of future rewards


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

    def predict(self, state: StateFrames) -> np.ndarray:
        raise NotImplementedError()
    
    def fit(self, state: StateFrames, y: List[Reward]) -> None:
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


class Compiler:
    def __init__(self, model: Model):
        self.__model = model

    def withOptimizer(self, optimizer: keras.optimizers.Optimizer) -> type["Compiler"]:
        assert isinstance(optimizer, keras.optimizers.Optimizer)
        self.__optimizer = optimizer
        return self
    
    def withMetrics(self, metrics: List[str]) -> type["Compiler"]:
        assert len(metrics) > 0
        for m in metrics:
            assert isinstance(m, str)
        
        self.__metrics = metrics
        
        return self
    
    def compile(self) -> Model:
        assert self.__optimizer != None
        assert self.__metrics != None

        self.__model._model.compile(
            optimizer=self.__optimizer,
            loss=keras.losses.mean_squared_error,
            metrics=self.__metrics
        )
        self.__model._model.summary()

        return self.__model


def expected_reward(
        q_func: Model,
        s: List[StateFrames],
        sprime: List[StateFrames],
        action: List[Action],
        reward: List[Reward],
        isTerminalState: List[Terminated],
        gamma: DiscountFactor
    ) -> np.ndarray:
    """
    Computes the expected reward on a batch of samples by leveraging 
    vectorized operations
    """
    s = np.stack(s, axis=0)
    y_hat = q_func.predict(s)
    
    sprime = np.stack(sprime, axis=0)
    rprime = q_func.predict(sprime)

    i = np.arange(y_hat.shape[0]) # Index hack to access all rows in the predictions
    a = np.array(action) # The actions we are updating (columns of the predictions we will modify)

    t = np.array(isTerminalState) == False

    r = np.array(reward) + gamma * rprime[i, a] * t
    y_hat[i, a] = r # We only update the y_hat of those actions we know exactly what the future looks like

    return y_hat



def back_prop(model: Model, buff: ReplayBuff, gamma: DiscountFactor) -> None:
    pass
