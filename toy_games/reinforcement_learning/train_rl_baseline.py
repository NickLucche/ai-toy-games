from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import vec_frame_stack
import gym
from toy_games.reinforcement_learning.env_wrappers import *
import cv2
from stable_baselines3.common.atari_wrappers import *
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env


def make_snake_env(render_shape=(84, 84), frame_stack=4):
    env = gym.make(
        "toy_games:Snake-v0", grid_size=30, walls=True, render_shape=render_shape
    )
    # env = MaxAndSkipEnv(env) no way we can skip frames in snake
    env = PreprocessingWrapper(env)

    # env = Array2Tensor(env)
    # env = WarpFrame(env, *render_shape)
    env = BufferWrapper(env, frame_stack, render_shape, dtype=np.uint8)
    return env
    env = make_vec_env(
        "toy_games:Snake-v0",
        1,
        wrapper_class=PreprocessingWrapper,
        env_kwargs=dict(grid_size=30, walls=True, render_shape=render_shape),
    )
    env = VecFrameStack(env, 4)
    return env


if __name__ == "__main__":
    train = True
    env = make_snake_env((90, 90), 2)
    # import matplotlib.pyplot as plt
    # obs = env.reset()
    # obs, r, d, info = env.step(env.action_space.sample())
    # fig, ax = plt.subplots(4)
    # for i in range(4):
    #     ax[i].imshow(obs[i])
    # plt.show()
    # print(obs.shape, obs.dtype, obs.max(), r, d, info)
    # exit()

    # check env
    check_env(env)

    # env = gym.make('CartPole-v1')
    if train:
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=1e-2,
            buffer_size=100000,
            learning_starts=100000,
            batch_size=192,
            exploration_fraction=0.5,
            max_grad_norm=1e6, # no-clipping?
            tensorboard_log="./toy_games/reinforcement_learning/logs",
        )
        model.learn(total_timesteps=int(1e6), tb_log_name="snake")
        # Save the agent
        model.save("snake")

    # Load the trained agent
    model = DQN.load("snake", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print("Eval:", mean_reward, std_reward)

    for ep in range(2):
        obs = env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cv2.imshow("snake", env.render((512, 512)))
            key = cv2.waitKey(500)
            if key == ord("q"):
                break

            if done:
                break
