def get_end_obs(env):
    """
    @return (array): ending observation image
    """
    
    return env.unwrapped.ale.getScreenRGB()

def get_end_reward(env):
    """
    @return (float): ending reward number
    """

    return env.unwrapped.ale.act(0)

def get_episode_frame(env):
    """
    @return (int): ending frame number
    """

    return env.unwrapped.ale.getEpisodeFrameNumber()

def get_data(step, env):
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
    
    # info returns [lives, episode frame #, total frame #]
    return obs, reward, info

def perform_action(action, env):
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
