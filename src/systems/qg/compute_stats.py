import sys
import h5py
import numpy as np

def stats_component(trajs_group, component):
    # First compute means
    means = []
    for k in trajs_group.keys():
        if k.startswith("traj"):
            means.append(trajs_group[k][component].mean())
    mean_val = np.mean(means)
    # Next compute the deviations
    sq_devs = []
    for k in trajs_group.keys():
        if k.startswith("traj"):
            sq_devs.append((trajs_group[k][component] - mean_val)**2)
    std_val = np.sqrt(np.mean(sq_devs))
    return mean_val, std_val

with h5py.File(sys.argv[1], "r") as h5_file:
    traj_group = h5_file["trajs"]
    for component in ["u", "v", "q"]:
        mean, std = stats_component(traj_group, component)
        print(f"{component} mean = {repr(mean)}")
        print(f"{component} std  = {repr(std)}")
