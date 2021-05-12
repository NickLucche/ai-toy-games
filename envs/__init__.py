from gym.envs.registration import register
from envs.snake_env import SnakeEnv

register(
    id='Snake-v0',
    entry_point='envs:SnakeEnv',
)
