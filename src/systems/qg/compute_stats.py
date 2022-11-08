import argparse
import dataclasses
import math
import contextlib
import h5py
import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser(description="Compute min/max/mean/var of a large QG data set")
parser.add_argument("out_file", type=str, help="Output file to save statistics (.npz format)")
parser.add_argument("in_file", type=str, help="HDF5 file containing the trajectories")


def traj_chunk_iter(h5_path, *, chunk_size=1000, include_remainders=True):
    with h5py.File(h5_path, "r") as data_file:
        trajs_group = data_file["trajs"]
        for k in trajs_group.keys():
            if not (k.startswith("traj") and k.endswith("_q")):
                # This isn't a q component
                continue
            # This is a q component of the trajectory, load slices and yield them
            record = trajs_group[k]
            num_steps = record.shape[0]
            num_batches = num_steps // chunk_size
            remainder = num_steps % chunk_size
            for i in range(num_batches):
                start = i * chunk_size
                end = start + chunk_size
                yield record[start:end]
            # Yield the remainder
            if include_remainders and remainder != 0:
                yield record[-remainder:]


@dataclasses.dataclass
class QGStats:
    mean: npt.NDArray[np.float32]
    var: npt.NDArray[np.float32]
    min: npt.NDArray[np.float32]
    max: npt.NDArray[np.float32]


class QGStatAccumulator:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.m2 = 0
        self.min = np.inf
        self.max = -np.inf

    def add_batch(self, batch):
        # Aggregate min/max
        self.min = np.minimum(self.min, np.min(batch, axis=(0, -1, -2)))
        self.max = np.maximum(self.max, np.max(batch, axis=(0, -1, -2)))
        # Aggregate remaining statistics
        num_elems = batch.shape[0] * math.prod(batch.shape[2:])
        self.count += num_elems
        delta = batch - np.expand_dims(self.mean, axis=(0, -1, -2))
        self.mean += np.sum(delta / self.count, axis=(0, -1, -2))
        delta2 = batch - np.expand_dims(self.mean, axis=(0, -1, -2))
        self.m2 += np.sum(delta * delta2, axis=(0, -1, -2))

    def finalize(self):
        if self.count < 2:
            raise ValueError("not enough samples, need at least two")
        return QGStats(
            mean=self.mean,
            var=self.m2 / self.count,
            min=self.min,
            max=self.max,
        )


def compute_stats(batch_iter):
    stat_accum = QGStatAccumulator()
    for batch in batch_iter:
        stat_accum.add_batch(batch.astype(np.float64))
    return stat_accum.finalize()


def main():
    args = parser.parse_args()
    with contextlib.closing(traj_chunk_iter(args.in_file)) as batch_iter:
        stats = compute_stats(batch_iter)
    np.savez(
        args.out_file,
        mean=stats.mean.astype(np.float32),
        var=stats.var.astype(np.float32),
        min=stats.min.astype(np.float32),
        max=stats.max.astype(np.float32),
    )
    print("Finished computing stats")


if __name__ == "__main__":
    main()
