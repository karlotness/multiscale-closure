import sys
import argparse
import h5py

parser = argparse.ArgumentParser(description="Internal worker script for data loader")
parser.add_argument("file_path", type=str, help="Path to hdf5 data file")
parser.add_argument("rollout_steps", type=int, help="Number of rollout steps to gather")

def main(args):
    rollout_steps = args.rollout_steps
    with h5py.File(args.file_path, "r") as h5_file:
        trajs_group = h5_file["trajs"]
        # Do work here
        while line := sys.stdin.readline():
            line_parts = line.split()
            traj = int(line_parts[0])
            step = int(line_parts[1])
            sys.stdout.buffer.write(trajs_group[f"traj{traj:05d}_q"][step:step + rollout_steps].tobytes())
            sys.stdout.buffer.write(trajs_group[f"traj{traj:05d}_dqhdt"][step:step + rollout_steps + 2].tobytes())
            sys.stdout.buffer.flush()
        else:
            # Orderly exit
            sys.exit(0)


if __name__ == "__main__":
    main(parser.parse_args())
