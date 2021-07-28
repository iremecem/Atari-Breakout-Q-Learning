from Agent import Agent
from ImageModel import Preprocessor
from argparse import ArgumentParser
from Wrapper import make_atari, wrap_deepmind
from Logger import Logger

import gym
import tensorflow as tf
import numpy as np


def train(mode: str, strategy: str):
    __NUM_EPISODES__: int = 1000
    __MAX_STEPS__: int = 10000
    __BATCH_SIZE__: int = 8 # Prev 64, 32, 16
    __TARGET_UPDATE__: int = 1000 # Prev 10000
    __STRATEGY__: str = strategy

    time_spent: int = 0
    rewards: int = 0
    loss: float = np.nan
    done: bool = False

    environment = gym.make('BreakoutNoFrameskip-v4')
    preprocessor = Preprocessor()
    logger = Logger("log_" + strategy)

    input_shape = (84, 84, 4,)
    num_actions = environment.action_space.n

    agent = Agent(input_shape, num_actions)
    if mode == "train_cont":
        agent.load(f"./saves/{strategy}/", "train")

    for episode in range(__NUM_EPISODES__):
        starting_state = environment.reset()
        state = preprocessor.preprocess(starting_state)
        for _ in range(3):
            observation = environment.reset()
            observation = preprocessor.preprocess(observation)
            state = np.concatenate([state, observation], axis=-1)

        rewards_cur = 0
        loss_cur = np.nan

        for step in range(__MAX_STEPS__):
            environment.render()
            time_spent += 1

            action = agent.step(state, True)
            next_state, reward, done, _ = environment.step(action)
            next_state_processed = preprocessor.preprocess(next_state)
            state_next = state[:, :, 1:]
            state_next = np.concatenate(
                [state_next, next_state_processed], axis=-1)
            agent.addToReplayMemory(
                state, action, reward, state_next, done)
            state = state_next

            rewards_cur += reward

            if time_spent % __TARGET_UPDATE__ == 0:
                agent.update()
            if done:
                rewards += rewards_cur
                if loss == np.nan:
                    loss = loss_cur / step
                else:
                    loss += loss_cur / step
                logger.add_to_log(episode, rewards_cur,
                                  rewards/episode, loss/episode)
                agent.save(f"./saves/{strategy}/")
                break

            if agent.getReplayMemorySize() > __BATCH_SIZE__:
                res = agent.replay(__BATCH_SIZE__, __STRATEGY__)
                if loss_cur == np.nan:
                    loss_cur = res
                else:
                    loss_cur += res


def test(strategy):
    env = make_atari("BreakoutNoFrameskip-v4")

    input_shape = (84, 84, 4,)
    num_actions = env.action_space.n

    agent = Agent(input_shape, num_actions)
    agent.load(f"./saves/{strategy}/", "test")

    env = wrap_deepmind(env, frame_stack=True, scale=True)
    for i in range(100):
        state = np.array(env.reset())
        done = False
        while not done:
            env.render()
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_values = agent.predict(state_tensor)
            action = np.argmax(action_values)
            state_next, _, done, _ = env.step(action)
            state = np.array(state_next)
    env.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-m", choices=["train", "train_cont", "test"], required=True,
                        help="Mode to run script", default="train", type=str)
    parser.add_argument("-str", choices=["dqn", "ddqn"], required=True,
                        help="Strategy to use while training", default="ddqn", type=str)

    args = parser.parse_args()

    mode = args.m
    strategy = args.str

    if mode == "train":
        train(mode, strategy)
    elif mode == "train_cont":
        train(mode, strategy)
    elif mode == "test":
        test(strategy)
    else:
        print("Unknown mode")
