import numpy as np
from skimage import transform
from skimage.color import rgb2gray

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
    def __init__(self, env):
        self.obs = [process_frame(env) for i in range(4)]

    def add_observation(self, newobs : np.ndarray):
        self.obs.pop(0)
        self.obs.append(newobs)


def state_transformer(state : State):
    return np.stack(state.obs, axis=2)

def process_frame(obs):
    #turn rgb image -> gray scaled 
    grayScale = rgb2gray(obs)

    #crop the screen to get rid not needed screen parts (ex. the area below player)
    cropped_frame = grayScale[8:-12, 4:-12]

    #Normalize pixel values
    normalized = cropped_frame / 255.0

    #Resize to desired frame size
    preprocessed_frame = transform.resize(normalized, [84,84])

    #return 84 x 84 x 1 frame
    return preprocessed_frame

