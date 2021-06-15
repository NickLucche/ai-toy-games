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


def make_snake_env(render_shape=(84, 84), frame_stack=4, grid_size=30):
    env = gym.make(
        "toy_games:Snake-v0", grid_size=grid_size, walls=True, render_shape=render_shape
    )
    # env = MaxAndSkipEnv(env) no way we can skip frames in snake
    env = PreprocessingWrapper(env)

    # env = Array2Tensor(env)
    # env = WarpFrame(env, *render_shape)
    env = BufferWrapper(env, frame_stack, render_shape, dtype=np.uint8)
    return env
    # env = make_vec_env(
    #     "toy_games:Snake-v0",
    #     1,
    #     wrapper_class=PreprocessingWrapper,
    #     env_kwargs=dict(grid_size=30, walls=True, render_shape=render_shape),
    # )
    # env = VecFrameStack(env, 4)
    # return env


if __name__ == "__main__":
    train = False
    make_gif = True
    # make sure this coincides with testing env
    env = make_snake_env((90, 90), 4, grid_size=10)
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
            learning_rate=1e-4,
            buffer_size=int(1e6),
            learning_starts=int(5e4),
            batch_size=64,
            exploration_fraction=0.4,
            # max_grad_norm=1e6, # no-clipping?
            tensorboard_log="./toy_games/reinforcement_learning/logs",
        )
        model.learn(total_timesteps=int(5e6), tb_log_name="snake")
        # Save the agent
        model.save("snake")

    # Load the trained agent, avoid instatiating useless buffer
    model = DQN.load("snake", env=env, custom_objects={'buffer_size': 1})

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=25)
    print("Evaluation results:", mean_reward, std_reward)
    if make_gif:
        from PIL import Image
        frames = []


    for ep in range(2):
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            frame = env.render((512, 512))
            cv2.imshow("snake", frame)
            if make_gif:
                frames.append(frame)
            key = cv2.waitKey(400)
            if key == ord("q"):
                break
    cv2.destroyAllWindows()

    if make_gif:
        frames = list(map(lambda img: Image.fromarray(img[..., ::-1]), frames))
        print("Writing GIF..")
        img, *imgs = frames
        img.save(
            fp="assets/snake.gif",
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )
