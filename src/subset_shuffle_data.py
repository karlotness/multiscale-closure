# Take subsets of trajectories and pre-shuffle them for use in limited
# data tests

import argparse
import re
import pathlib
import logging
import numpy as np
import h5py
import utils

parser = argparse.ArgumentParser(description="Produce a subset dataset from an original")
parser.add_argument("in_file", type=str, help="Original file to subset")
parser.add_argument("out_file", type=str, help="Output file")
parser.add_argument("num_traj", type=int, help="Number of trajectories to write")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])


def main():
    args = parser.parse_args()
    in_path = pathlib.Path(args.in_file)
    out_path = pathlib.Path(args.out_file)
    if not in_path.is_file():
        raise ValueError(f"input file {in_path} must exist")
    utils.set_up_logging(level=args.log_level, out_file=in_path.parent / f"subset-{args.num_traj}.log")
    logger = logging.getLogger("subset_shuffle")
    logger.info("Arguments %s", vars(args))
    git_info = utils.get_git_info(base_logger=logger)
    with h5py.File(in_path, "r") as in_file, h5py.File(out_path, "w", libver="latest") as out_file:
        # Copy parameters and stats
        logger.info("Copying stats")
        in_file.copy(source=in_file["/stats"], dest=out_file["/"], shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
        logger.info("Copying parameters")
        in_file.copy(source=in_file["/params"], dest=out_file["/"], shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
        # Copy subset of trajectories
        traj_group = out_file.create_group("trajs")
        traj_re = re.compile(r"traj(?P<num>\d+)")
        logger.info("Copying special data fields")
        for name in ("t", "tc", "ablevel"):
            in_file.copy(source=in_file["trajs"][name], dest=traj_group, shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
        logger.info("Copying subset of trajectories")
        for name in in_file["trajs"].keys():
            if m := traj_re.match(name):
                traj_num = int(m.group("num"))
                if traj_num < args.num_traj:
                    logger.info("Copying value %s", name)
                    in_file.copy(source=in_file["trajs"][name], dest=traj_group, shallow=False, expand_soft=True, expand_external=True, expand_refs=True, without_attrs=False)
    logger.info("Finished producing data subset")



if __name__ == "__main__":
    main()
