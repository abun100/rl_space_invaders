from typing import List

import numpy as np

class Weights():
    def __init__(self, w: np.ndarray | None):
        self.__w = w

    def to_numpy(self) -> np.ndarray:
        return self.__w


class Model():
    __weights: Weights = None

    def save(self, file: str) -> None:
        w = self.__weights.to_numpy()
        np.save(file, w)
    
    def load(self, file: str) -> None:
        w = np.load(file)
        self.__weights = Weights(w)
    
    def weights(self) -> Weights:
        if self.__weights == None:
            raise Exception("model weights have not been loaded. Try random_weights to initialize.")
        
        return self.__weights
    
    def random_weights(self) -> None:
        raise NotImplementedError()

    def predict(self, state: List) -> List:
        raise NotImplementedError()
    
    def fit(self, state: List, y: List) -> None:
        raise NotImplementedError()

class DQNGrayScale(Model):
    pass