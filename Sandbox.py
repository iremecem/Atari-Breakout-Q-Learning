import gym
from gym.utils import play
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", help="Game Type", required=True, type=int)
    args = parser.parse_args()
    game = args.m

    if game == 1:
        environment = gym.make('Breakout-v0')
    elif game == 2:
        environment = gym.make('BreakoutNoFrameskip-v4')

    play.play(environment, zoom=3)
