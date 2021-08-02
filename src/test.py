import argparse

import torch

from cherry_rl.test import play_n_episodes
from src.train import make_mario_env, make_ac_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', '-w', type=int, default=1)
    parser.add_argument('--level', '-l', type=int, default=1)
    parser.add_argument('--random', '-r', action='store_true')
    parser.add_argument('--n_episodes', '-n', type=int, default=10)
    parser.add_argument('--pause_first', '-p', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_mario_env(args.world, args.level)
    ac = make_ac_model(torch.device('cpu'))
    checkpoint = torch.load(
        f'checkpoints/world_{args.world}_level_{args.level}.pth',
        map_location='cpu'
    )
    ac.load_state_dict(checkpoint['ac_model'])
    play_n_episodes(
        env, ac, not args.random, args.n_episodes,
        silent=False, reward_threshold=None, save_demo=None, pause_first=args.pause_first
    )


if __name__ == '__main__':
    main()
