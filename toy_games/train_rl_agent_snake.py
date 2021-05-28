import pytorch_lightning as pl
from pl_bolts.models.rl import DoubleDQN
import torch
import gym
import numpy as np
import cv2
from gym import Wrapper, ObservationWrapper
from pl_bolts.models.rl.common.gym_wrappers import MaxAndSkipEnv
from pl_bolts.models.rl.dqn_model import ValueAgent
from typing import OrderedDict, Tuple

# default implementation is really dumb in handling custom envs with unnecessary wrappers
class SnakeDDQN(DoubleDQN):
    def __init__(self, env: gym.Env, test_env: gym.Env, *args, **kwargs):
        fuck_u_poor_interface = "PongNoFrameskip-v4"
        super().__init__(fuck_u_poor_interface, *args, **kwargs)
        print('pre env obs space', self.env.observation_space)
        # self.env = env
        # self.test_env = test_env

        # this already takes into account obs space transformation applied by env wrappers
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        print('obs space', self.obs_shape)

        # from defaults
        eps_start = 1.0
        eps_end = 0.02
        eps_last_frame = 150000
        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )
        self.build_networks()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        d = super().training_step(batch, _)
        for name, value in d.items():
            self.log(name, value, logger=True, prog_bar=True)
        return d
        


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

    def __init__(self, env, n_steps=4, resolution=(84, 84), dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            # old_space.low.repeat(n_steps, axis=0),
            # old_space.high.repeat(n_steps, axis=0),
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


if __name__ == "__main__":
    test = False

    def make_snake_env(render_shape=(84, 84)):
        env = gym.make(
            "toy_games:Snake-v0", grid_size=30, walls=True, render_shape=render_shape
        )
        # env = MaxAndSkipEnv(env) no way we can skip frames in snake
        env = PreprocessingWrapper(env)
        env = BufferWrapper(env, 4, render_shape)
        env = Array2Tensor(env)
        return env

    # env = make_snake_env((512, 512))
    # import matplotlib.pyplot as plt
    # obs = env.reset()
    # for i in range(3):
    #     obs, reward, done, _ = env.step(env.action_space.sample())
    #     print(obs.shape, obs.dtype)
    #     fig, ax = plt.subplots(len(obs))
    #     for i, o in enumerate(obs):
    #         ax[i].imshow(o, cmap='gray')
    #     plt.show()
    #     print("Reward:", reward, "Done:", done)
    #     if done:
    #         print('dead')
    #         break
    # exit()

    env = make_snake_env()
    # TODO: check it works, hstack obs and plot them first
    if not test:
        test_env = make_snake_env()
        dqn = SnakeDDQN(env, test_env, warm_start_size=100)
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        trainer = pl.Trainer(max_epochs=30, gpus=1)
        # obs = env.reset()
        # q_values = dqn.net(obs.unsqueeze(0))
        trainer.fit(dqn)
    else:
        model = DoubleDQN.load_from_checkpoint(
            "lightning_logs/version_5/checkpoints/epoch=29-step=29999.ckpt"
        )
        # env = gym.make(env_name)
        # they're setting up the preprocessing pipeline here with Env wrappers, cool
        # it's stacking 4 frames by default
        # env = DQN.make_environment(env_name)

        obs = env.reset()
        print("observation", obs.shape, obs.max(), obs.dtype, type(obs))
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
            cv2.imshow("", x)
            k = cv2.waitKey(500)
            if k == ord("q"):
                break

        cv2.destroyAllWindows()
        env.close()
