import pytorch_lightning as pl
from pl_bolts.models.rl import DQN, DoubleDQN
import torch
import gym
import numpy as np
import cv2
from gym import Wrapper, ObservationWrapper
from pl_bolts.models.rl.common.gym_wrappers import MaxAndSkipEnv

# class SnakeDQN(pl.LightningModule):

#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)

# default implementation is really dumb in handling custom envs with unnecessary wrappers
class PreprocessingWrapper(ObservationWrapper):

    def observation(self, obs):
        """preprocess the obs"""
        return PreprocessingWrapper.process(obs)

    @staticmethod
    def process(img: np.ndarray):
        # single channel
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # to torch tensor
        return img
        # return torch.from_numpy(img)

class BufferWrapper(ObservationWrapper):
    """"Wrapper for image stacking"""

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            # old_space.low.repeat(n_steps, axis=0),
            # old_space.high.repeat(n_steps, axis=0),
            low=0., high=1.,
            shape=(4, 84, 84),
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



    

if __name__ == "__main__":
    test = True
    fuck_u_poor_interface = "PongNoFrameskip-v4"
    env_name = "toy_games:Snake-v0"
    def make_env():
        env = gym.make(env_name, grid_size=30, walls=False, render_shape=(84, 84))
        # env = MaxAndSkipEnv(env) no way we can skip frames in snake
        env = PreprocessingWrapper(env)
        env = BufferWrapper(env, 4)
        return env
    env = make_env()
    #TODO: check it works, hstack obs and plot them first 
    if not test:
        dqn = DQN(make_env, warm_start_size=100)
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        trainer = pl.Trainer(max_epochs=30, gpus=1)
        trainer.fit(dqn)
    else:
        model = DQN.load_from_checkpoint('lightning_logs/version_5/checkpoints/epoch=29-step=29999.ckpt')
        # env = gym.make(env_name)
        # they're setting up the preprocessing pipeline here with Env wrappers, cool
        # it's stacking 4 frames by default
        # env = DQN.make_environment(env_name)
        
        obs = env.reset()
        print('observation', obs.shape, obs.max(), obs.dtype, type(obs))
        # obs = torch.from_numpy(obs).unsqueeze(0)
        # x = torch.from_numpy(obs).permute(2, 0, 1)[0, ...]
        # x = x.float()/255.
        # print('q', qvals.shape, qvals)
        done = False
        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0)
            # obs = obs.unsqueeze(0)
            qvals = model(obs)
            action = qvals.argmax(dim=1)
            obs, reward, done, _ = env.step(action.item())
            # obs, reward, done, _ = env.step(env.action_space.sample())
            # print('observation', obs.shape, obs.max(), obs.dtype, type(obs))
            x = cv2.resize(obs[-1], (900, 900), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('', x)
            k = cv2.waitKey(500)
            if k == ord('q'):
                break

        cv2.destroyAllWindows() 
        env.close()