from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import vec_frame_stack
import gym
from toy_games.reinforcement_learning.env_wrappers import *
import cv2


def make_snake_env(render_shape=(84, 84)):
    env = gym.make(
        "toy_games:Snake-v0", grid_size=30, walls=True, render_shape=render_shape
    )
    # env = MaxAndSkipEnv(env) no way we can skip frames in snake
    env = PreprocessingWrapper(env)
    env = BufferWrapper(env, 4, render_shape)
    env = Array2Tensor(env)
    return env


if __name__ == "__main__":
    train = False
    # check env 
    _check_env=True
    # env = gym.make(
    #         "toy_games:Snake-v0", grid_size=30, walls=True, render_shape=(90, 90)
    # )
    env = make_snake_env((90,90))
    if _check_env:
        check_env(env)

    # env = gym.make('CartPole-v1')
    if train:
        model = DQN('CnnPolicy', env, verbose=1, buffer_size=10000)
        model.learn(total_timesteps=1000000)
        # Save the agent
        model.save("snake")

    # Load the trained agent
    model = DQN.load("snake", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)

    for ep in range(2):
        obs = env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cv2.imshow('snake', env.render((512, 512)))
            key = cv2.waitKey(500)
            if key==ord('q'):
                break

            if done:
                break
