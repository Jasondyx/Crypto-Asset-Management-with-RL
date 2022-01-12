import numpy as np
import pandas as pd
import os
import copy
import torch
from torch import nn
from torch.optim import Adam
from DNN import DNN, FeatureExtractor


import argparse
import os

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  # NOQA:E402
from torch import nn  # NOQA:E402

import pfrl  # NOQA:E402
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import a3c, a2c  # NOQA:E402
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt  # NOQA:E402
from pfrl.policies import SoftmaxCategoricalHead, GaussianHeadWithDiagonalCovariance  # NOQA:E402
from pfrl.wrappers import atari_wrappers  # NOQA:E402
import logging
from TradeEnv import TradeEnv


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=8 * 10 ** 7)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument("--load", type=str, default="")
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--fixed_feature_extractor",
        action="store_true",
        default=False,
        help=(
            "Fix the pretrained feature extractor."
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=20)

    # Set a random seed used in PFRL.
    utils.set_random_seed(0)

    # Set different random seeds for different subprocesses.
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    asset_list = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    n_actions = len(asset_list)

    # pretrained feature extractor
    file_list = os.listdir(f'./result/representation_learning/')
    file_pretrained = 'model_state_time_2021-12-17 13_04_i_batch_last_batch'
    feature_extractor = FeatureExtractor(n_asset=len(asset_list), filepath=f'./result/representation_learning/{file_pretrained}', device=device)

    class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.n_actions = args[0]

        def forward(self, x):
            return x.reshape(-1, 2*self.n_actions)

    fixed_feature_extractor = args.fixed_feature_extractor
    print(f'Fixed_feature_extractor: {fixed_feature_extractor}.')
    model = nn.Sequential(
        feature_extractor.requires_grad_(not fixed_feature_extractor),
        nn.Linear(256, 256),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(256, 2*n_actions),
                Reshape(n_actions),
                GaussianHeadWithDiagonalCovariance(),
            ),
            nn.Linear(256, 1),
        ),
    )
    model.to(device)
    print(f'Using device: {device}.')

    # SharedRMSprop is same as torch.optim.RMSprop except that it initializes
    opt = SharedRMSpropEpsInsideSqrt(model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99)
    assert opt.state_dict()["state"], (
        "To share optimizer state across processes, the state must be"
        " initialized before training."
    )

    agent = a2c.A2C(
        model,
        opt,
        num_processes=1,
        gpu=0 if torch.cuda.is_available() else -1,
        gamma=0.99,
        # phi=phi,
        max_grad_norm=40.0,
    )

    if args.load:
        agent.load(args.load)
        # agent.load('./results/20211217T222256.078788/best')

    args.demo = True
    if args.demo:
        env = TradeEnv(0, True, outdir=args.outdir)
        eval_stats = experiments.eval_performance(
            env=env, agent=agent, n_steps=None, n_episodes=1
        )
        print(
            "n_steps: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_steps,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        env = TradeEnv(datafile='./data/train.csv', outdir=args.outdir)
        pfrl.experiments.train_agent_with_evaluation(
            agent,
            env,
            steps=2000000,  # Train the agent for 2000 steps
            eval_n_steps=None,  # We evaluate for episodes, not time
            eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
            eval_max_episode_len=500,
            train_max_episode_len=2000,  # Maximum length of each episode
            eval_interval=20000,  # Evaluate the agent after every 1000 steps
            outdir=args.outdir,  # Save everything to 'result' directory
        )


if __name__ == "__main__":
    main()