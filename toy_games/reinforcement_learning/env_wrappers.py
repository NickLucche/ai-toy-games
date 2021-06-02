from gym import Wrapper, ObservationWrapper
import numpy as np
import gym
import torch

class PreprocessingWrapper(ObservationWrapper):
    def observation(self, obs):
        """preprocess the obs"""
        return PreprocessingWrapper.process(obs)

    @staticmethod
    def process(img: np.ndarray):
        # single channel
        return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114



class BufferWrapper(ObservationWrapper):
    """"Wrapper for image stacking"""

    def __init__(self, env, n_steps=4, resolution=(84, 84), dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_steps, *resolution),
            dtype=dtype,
        )

    def reset(self):
        """reset env"""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """convert observation"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class Array2Tensor(ObservationWrapper):

    def __init__(self, env):
        super(Array2Tensor, self).__init__(env)

    def observation(self, observation):
        """convert observation"""
        return torch.from_numpy(observation)