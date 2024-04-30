import sys
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="Internal worker script for data loader")
parser.add_argument("file_path", type=str, help="Path to hdf5 data file")
parser.add_argument("steps", type=int, help="Number of steps to gather")
parser.add_argument("size", type=int, help="Size of data snapshots to load")
parser.add_argument("--fields", type=str, nargs="+", default=["u", "v", "u_corr", "v_corr"], help="Which fields to load")

def main(args):
    rollout_steps = args.steps
    size = args.size
    fields = args.fields
    with h5py.File(args.file_path, "r") as h5_file:
        trajs_group = h5_file[f"sz{size}"]["trajs"]
        # Do work here
        while line := sys.stdin.readline():
            line_parts = line.split()
            traj = int(line_parts[0])
            step = int(line_parts[1])
            end_step = step + rollout_steps
            field_prefix = f"traj{traj:05d}_"
            sys.stdout.buffer.write(
                b"".join(
                    (
                        trajs_group[f"{field_prefix}{field}"][step:end_step].tobytes()
                    )
                    for field in fields
                )
            )
            sys.stdout.buffer.flush()
        else:
            # Orderly exit
            sys.exit(0)


if __name__ == "__main__":
    main(parser.parse_args())
