import unittest
from space_invaders import gameState
import gymnasium as gym
from PIL import Image

env = gym.make('ALE/SpaceInvaders-v5')

class TestgameState(unittest.TestCase):
    def transform_image_test(self):
        start = env.reset()
        state_frames = gameState.State(start[0])
        image = Image.fromarray(state_frames.obs[0])
        image.save("original_state.png")
        
if __name__ == '__main__':
    unittest.main()