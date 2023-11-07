import gymnasium

env=gymnasium.make("ALE/SpaceInvaders-v5", render_mode='human',
                   repeat_action_probability=.25,full_action_space=False, obs_type='rgb')

def run_game():
    score = 0
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, truncate, info = env.step(action)
        score+=reward
        env.render()
        if done:
            env.close()
            print(f'Score:{score}')
            break

def get_end_obs():
    """
    @return (array): ending observation image
    """
    return env.unwrapped.ale.getScreenRGB()

def get_end_reward():
    """
    @return (float): ending reward number
    """
    return env.unwrapped.ale.act(0)

def get_episode_frame():
    """
    @return (int): ending frame number
    """
    return env.unwrapped.ale.getEpisodeFrameNumber()

def get_data(step):
    """
    Get the data at specific step
    @return (list): observation, reward, and info : {lives, ep. frame #, and 
    total frame #} 
    """
    obs, reward, info = None, None, None
    env.reset()
    for i in range(step):
        action = env.action_space.sample()
        obs, reward, done, truncate , info = env.step(action)
        if done:
            break
    #info returns [lives, episode frame #, total frame #]
    return obs, reward, info

def perform_action(action):
    """
    Perform a specific action.
    @param action (int): The action to perform.
    @return (list) : obs, reward, done, info
    """
    obs, reward, done, truncate , info = env.step(action)
    return obs, reward, done, info

def play_episodes(num_episodes):
    """
    Play multiple episodes and return the total scores achieved.
    @param num_episodes (int): The number of episodes to play.
    @return total_scores [float]: Total scores for each episode.
    """
    total_scores = []
    for _ in range(num_episodes):
        score = run_game()
        total_scores.append(score)
    return total_scores

if __name__ == '__main__':
    run_game()