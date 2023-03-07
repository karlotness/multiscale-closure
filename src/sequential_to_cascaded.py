import itertools
import json
import argparse
import pathlib
import logging
import equinox as eqx
from train import determine_channel_size
from eval import load_network
import utils

parser = argparse.ArgumentParser(description="Join sequentially trained networks into a cascaded network")
parser.add_argument("out_dir", type=str, help="Directory containing sequential nets")
parser.add_argument("--log_level", type=str, help="Level for logger", default="info", choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--net_load_types", type=str, nargs="+", default=["best_loss", "interval"])


def join_network_info(out_dir, num_nets):
    out_dir = pathlib.Path(out_dir)
    network_infos = []
    for i in range(num_nets):
        with (out_dir / f"net{i}" / "weights" / "network_info.json").open("r", encoding="utf8") as net_info_file:
            network_infos.append(json.load(net_info_file))
    # Make sure things match
    coarse_op_name = network_infos[0]["coarse_op_name"]
    train_path = network_infos[0]["train_path"]
    processing_scales = {
        determine_channel_size(c)
        for c in itertools.chain.from_iterable(
            itertools.chain(ni["input_channels"], ni["output_channels"])
            for ni in network_infos
        )
    }
    # Combine
    return {
        "train_path": train_path,
        "coarse_op_name": coarse_op_name,
        "processing_scales": sorted(processing_scales),
        "networks": [
            {
                "arch": ni["arch"],
                "args": ni["args"],
                "input_channels": ni["input_channels"],
                "output_channels": ni["output_channels"],
                "processing_size": ni["processing_size"],
            }
            for ni in network_infos
        ]
    }


def join_network_weights(out_dir, num_nets, net_load_type):
    out_dir = pathlib.Path(out_dir)
    nets = []
    for i in range(num_nets):
        net, _net_info = load_network(out_dir / f"net{i}" / "weights" / f"{net_load_type}.eqx")
        nets.append(net)
    return tuple(nets)


def main():
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {args.out_dir}")
    utils.set_up_logging(level=args.log_level, out_file=out_dir/"run.log")
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(args))
    git_info = utils.get_git_info(base_logger=logger)
    if git_info is not None:
        logger.info(
            "Running on commit %s (%s worktree)",
            git_info.hash,
            "clean" if git_info.clean_worktree else "dirty"
        )
    if not utils.check_environment_variables(base_logger=logger):
        sys.exit(1)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    # Determine how many networks we have to load
    num_nets = sum(1 for net_dir in out_dir.glob("net*") if net_dir.is_dir())
    logger.info("Combining %d networks", num_nets)
    # Construct network info file
    logger.info("Combining network info")
    with utils.rename_save_file(weights_dir / "network_info.json", "w", encoding="utf8") as net_info_file:
        json.dump(
            join_network_info(
                out_dir=out_dir,
                num_nets=num_nets,
            ),
            net_info_file,
        )
    logger.info("Finished combining network info")
    # Store network weights
    for net_load_type in args.net_load_types:
        logger.info("Combining network %s", net_load_type)
        with utils.rename_save_file(weights_dir / f"{net_load_type}.eqx", "wb") as net_file:
            eqx.tree_serialise_leaves(
                net_file,
                join_network_weights(out_dir, num_nets, net_load_type),
            )
        logger.info("Finished combining network %s", net_load_type)
    logger.info("Finished conversion run")


if __name__ == "__main__":
    main()
