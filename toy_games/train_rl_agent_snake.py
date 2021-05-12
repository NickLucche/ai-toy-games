import pytorch_lightning as pl
from pl_bolts.models.rl import DQN
import torch
import gym
import numpy as np
import cv2

if __name__ == "__main__":
    test = False
    # env_name = "PongNoFrameskip-v4"
    env_name = "toy_games:Snake-v0"
    if not test:
        dqn = DQN(env_name)
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        trainer = pl.Trainer(max_epochs=3, gpus=1, num_workers=16)
        trainer.fit(dqn)
    else:
        model = DQN.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=17-step=17999.ckpt')
        # env = gym.make(env_name)
        # they're setting up the preprocessing pipeline here with Env wrappers, cool
        # it's stacking 4 frames by default
        env = DQN.make_environment(env_name)
        
        obs = env.reset()
        print('observation', obs.shape, obs.max(), obs.dtype, type(obs))
        # obs = torch.from_numpy(obs).unsqueeze(0)
        # x = torch.from_numpy(obs).permute(2, 0, 1)[0, ...]
        # x = x.float()/255.
        # print('q', qvals.shape, qvals)
        done = False
        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0)
            qvals = model(obs)
            action = qvals.argmax(dim=1)
            obs, reward, done, _ = env.step(action.item())
            # print('observation', obs.shape, obs.max(), obs.dtype, type(obs))
            x = cv2.resize(obs[-1], (900, 900), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('', x)
            k = cv2.waitKey(500)
            if k == ord('q'):
                break

        cv2.destroyAllWindows() 
        env.close()