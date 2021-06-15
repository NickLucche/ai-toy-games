import gym
from toy_games.games.snake import Action
import numpy as np

env_name = 'toy_games:Snake-v0'

def test_make_env():
    gym.make(env_name)

def test_run_env():
    import time
    snake = gym.make(env_name, render_shape=(30,30), body_len=4)
    n_episodes = 10000
    fps = 0
    
    s = time.time()
    for i in range(n_episodes):
        score = 0
        body_len = 4
        prev_obs = snake.reset()
        while True:
            fps += 1
            obs, reward, done, info = snake.step(snake.action_space.sample())
            if done:
                print("episode over")
                break
            if reward>0:
                body_len += 1
                score += 1

            assert score == snake._env.score
            assert body_len == len(info['body_position']) >= 4, f'reward: {reward}'

    print("FPS", fps/(time.time()-s))
