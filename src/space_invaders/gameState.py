import numpy as np
from skimage import transform
from PIL import Image

# we need to initialize a state (4 gray scale consecutive observations
# rescaled to 84x84 pixels)
#
# class State:
#    def __init__(self, env):
#      # obs should be rbg images from the game
#      self.obs = sample.4.times(env)
#
#    def add_observation(self, obs: np.ndarray) -> self
#
#
# def state_transformer(s: State) -> np.ndarray
#
# implement state_transformer(s: State) -> np.random.rand(84, 84, 4)

class State:
    def __init__(self, ob):
        self.obs = [process_frame(ob) for _ in range(4)]

    def add_observation(self, newobs : np.ndarray):
        self.obs.pop(0)
        self.obs.append(process_frame(newobs))

    def to_numpy(self) -> np.ndarray:
        return np.stack(self.obs, axis=2)

def process_frame(obs):
    # crop the screen to get rid not needed screen parts (ex. the area below player)
    cropped_frame = obs[24:-12, 16:-28]

    # Normalize pixel values
    normalized = cropped_frame / 255.0

    # Resize to desired frame size
    preprocessed_frame = transform.resize(normalized, [84,84])

    # return 84 x 84 x 1 frame
    return preprocessed_frame


