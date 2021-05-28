import gym
from toy_games.games.snake import Action
import numpy as np

env_name = 'toy_games:Snake-v0'

def test_make_env():
    gym.make(env_name)

def test_observation():
    snake = gym.make(env_name, render_shape=(128,128))
    prev_obs = snake.reset()
    for i in range(3):
        obs, reward, done, _ = snake.step(Action.UP)
        assert reward <= 0
        assert not done 
        # the difference should be a single pixel 
        if reward == -1:
            assert (obs - prev_obs).sum() == 255*3*2
