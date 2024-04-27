import math
import dataclasses
import numpy as np


@dataclasses.dataclass
class NSStats:
    mean: np.float64
    var: np.float64
    min: np.float64
    max: np.float64


class NSStatAccumulator:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.m2 = 0
        self.min = np.inf
        self.max = -np.inf

    def add_batch(self, batch):
        batch = np.asarray(batch).astype(np.float64)
        # Aggregate min/max
        self.min = np.minimum(self.min, np.min(batch))
        self.max = np.maximum(self.max, np.max(batch))
        # Aggregate remaining statistics
        num_elems = math.prod(batch.shape)
        self.count += num_elems
        delta = batch - self.mean
        self.mean += np.sum(delta / self.count)
        delta2 = batch - self.mean
        self.m2 += np.sum(delta * delta2)

    def finalize(self):
        if self.count < 2:
            raise ValueError("not enough samples, need at least two")
        return NSStats(
            mean=self.mean,
            var=self.m2 / self.count,
            min=self.min,
            max=self.max,
        )
