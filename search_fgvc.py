"""
tune lr, wd for fgvc datasets and other datasets with train / val / test splits
"""
import os
import warnings

from time import sleep
from random import randint

from src.configs.config import get_cfg
from src.utils.file_io import PathManager

from train import train as train_main
from launch import default_argument_parser
warnings.filterwarnings("ignore")


def setup(args, pt, check_runtime=True):
    """
    Create configs and perform basic setups.
    overwrite the 2 parameters in cfg and args
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:4000".format(os.environ["SLURMD_NODENAME"])

    # overwrite percentile parameter
    cfg.MODEL.SUBSET.PERCENTILE = pt

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"pt{pt}"
    )
    # output_folder = os.path.splitext(os.path.basename(args.config_file))[0]

    # train cfg.RUN_N_TIMES times
    if check_runtime:
        count = 1
        while count <= cfg.RUN_N_TIMES:
            output_path = os.path.join(output_dir, output_folder, f"run{count}")
            # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
            sleep(randint(1, 5))
            if not PathManager.exists(output_path):
                PathManager.mkdirs(output_path)
                cfg.OUTPUT_DIR = output_path
                break
            else:
                count += 1
        if count > cfg.RUN_N_TIMES:
            raise ValueError(
                f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")
    else:
        # only used for dummy config file
        output_path = os.path.join(output_dir, output_folder, f"run1")
        cfg.OUTPUT_DIR = output_path

    cfg.freeze()
    return cfg


def subset_main(args):
    # pt_range = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 
    #     0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15]
    pt_range = [0.03, 0.04, 0.06, 0.07]
    # pt_range = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # pt_range = [0.05, 0.06, 0.07, 0.08, 0.09, 0.125, 0.15]
    for pt in pt_range:
        try:
            cfg = setup(args, pt)
        except ValueError:
            continue
        train_main(cfg, args)
        sleep(randint(1, 10))


def main(args):
    """main function to call from workflow"""
    if args.train_type == "subset":
        subset_main(args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
