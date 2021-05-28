"""
    OpenAI Gym Env wrapper of the simple snake game. 
"""
import gym
from gym.spaces.space import Space
from toy_games.games.snake import Snake
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import numpy as np


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ai']}
    reward_range = (-1, 1)
    # clockwise UP RIGHT DOWN LEFT #TODO: NOOP
    action_space = Discrete(4)
    observation_space: Space
    ep_counter: int = -1
    _current_ep_score: int = 0
    _env: Snake

    def __init__(self,
                 grid_size: int=15,
                 walls: bool = True,
                 render_shape=(800, 800),
                 render_mode: str = 'human') -> None:
        super().__init__()
        self.grid_size = grid_size
        self.walls = walls
        self.render_shape = render_shape
        self.observation_space = Box(low=0.,
                                     high=1.,
                                     shape=(*render_shape, 3),
                                     dtype=np.float32)
        self.render_mode = render_mode

    def reset(self):
        self._env = Snake(self.grid_size,
                          walls=self.walls,
                          render_width=self.render_shape[0],
                          render_height=self.render_shape[1])  # 🐍
        self.ep_counter += 1
        self._current_ep_score = 0
        return self._env.render(normalize=True)

    def step(self, action):
        if self._env.done:
            raise Exception(
                "Must call `reset` to restart game, this one's done!")
        if self.ep_counter < 0:
            raise Exception("Must call `reset` before starting game!")

        body, done = self._env.step(action)
        obs = body
        if self.render_mode == 'human':
            obs = self._env.render(normalize=True)
        reward = self._get_reward()
        return obs, reward, done, {
            'body_position': body,
            'ep_counter': self.ep_counter
        }

    def _get_reward(self):
        # -1 for each step taken (important to have a penalty)
        # 0 when food is reached, 1 for winning, -1 for losing
        if self._env.done:
            # TODO: tune
            return 1 if self._env.won else -1
        score = self._env.score
        r = score - self._current_ep_score - 1
        self._current_ep_score = score
        return r

    def render(self, mode='human'):
        # TODO: mode
        return self._env.render(normalize=False)


if __name__ == '__main__':
    import cv2
    cv2.namedWindow('Snake')
    # env = SnakeEnv(20, walls=False)
    env = gym.make('toy_games:Snake-v0', grid_size=30, walls=False)#, render_shape=(84,84))
    cv2.imshow('Snake', env.reset())
    for i in range(10):
        cv2.imshow('Snake', env.render())
        k = cv2.waitKey(1000)
        if k == ord('q'):
            break
        obs, reward, done, _ = env.step(env.action_space.sample())
        print("Reward:", reward, "Done:", done)
        if done:
            print('dead')
            break
    cv2.destroyAllWindows()
    env.close()
