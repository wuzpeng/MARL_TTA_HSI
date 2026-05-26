#!/usr/bin/env python
import sys
import os
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.share_harfang_runner import ShareHarfangRunner
from envs.harfang.Harfang2DEnv_Box import HarfangEnv
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = HarfangEnv()
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="HarFang", help="Which scenario to run on")
    parser.add_argument("--num_agents", type=int, default=6, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    curr_run = 'render'
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "eval_envs": None,
        "envs": envs,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = ShareHarfangRunner(config)

    runner.render()

    # post process
    envs.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    main(sys.argv[1:])
