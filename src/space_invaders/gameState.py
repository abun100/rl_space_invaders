from skimage import transform

import numpy as np


StateFrames = np.ndarray

Action = int

Reward = float

Terminated = bool

Died = bool


class State:
    def __init__(self, ob) -> None:
        self.obs = [process_frame(ob) for _ in range(4)]

    def add_observation(self, newobs : np.ndarray) -> None:
        self.obs.pop(0)
        self.obs.append(process_frame(newobs))

    def to_numpy(self) -> StateFrames:
        return np.stack(self.obs, axis=2)


def process_frame(obs: np.ndarray) -> np.ndarray:
    # crop the screen to get rid not needed screen parts (ex. the area below player)
    cropped_frame = obs[24:-12, 16:-28]

    # Normalize pixel values
    normalized = cropped_frame / 255.0

    # Resize to desired frame size
    preprocessed_frame = transform.resize(normalized, [84,84])

    # return 84 x 84 x 1 frame
    return preprocessed_frame
