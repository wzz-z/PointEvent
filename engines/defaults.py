import os
import sys
import argparse
import multiprocessing as mp

import utils.comm as comm
from utils.env import get_random_seed, set_seed
from utils.config import Config, DictAction


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Example:
        $ {sys.argv[0]} --config-file configs/EVUAV/evuav.py --options data_root=/path/to/dataset weight=/path/to/model_best.pth save_path=exp/test
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--options", default=None, nargs="+", action=DictAction, help="custom options"
    )
    return parser


def default_config_parser(file_path, options):
    option_keys = set(options or {})
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    if (
        "data_root" in cfg
        and "data" in cfg
        and "test" in cfg.data
        and "data.test.data_root" not in option_keys
    ):
        cfg.data.test.data_root = cfg.data_root

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    set_seed(seed)
    return cfg
