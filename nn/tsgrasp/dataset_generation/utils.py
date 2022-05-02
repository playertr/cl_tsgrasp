"""utils.py
Module for functions that help load and examine data.
"""

import h5py
import os
from . import params

def load_h5s(dir_=params.TRAJDIR):
    """Load a set of h5py Files from a directory.

    h5py Files remain open until the objects associated with them are deleted.

    Args:
        dir (string, optional): Directory to search for h5 files. Defaults to params.TRAJDIR.

    Returns:
        dict: Dictionary of {path : h5py.File} key-value pairs.
    """

    h5_paths = [os.path.join(dir_, p) for p in os.listdir(dir_) if '.h5' in p]

    dsets = {}
    for p in h5_paths:
        dsets[p] = h5py.File(p)

    return dsets

class TrajectoryLoader():
    """Class for torch-style dataloader that accesses h5py File objects.
    """
    def __init__(self, dir_=params.TRAJDIR):
        """Load a dictionary of h5py.File objects and instantiate the list of (name, DataSet) tuples.

        Args:
            dir_ (string, optional): Directory to search for h5 files. Defaults to params.TRAJDIR.
        """
        self.dsets = load_h5s(dir_)

        self.trajectories = []
        for path, ds in self.dsets.items():
            keys = {k for k in ds.keys() if k.startswith('pitch')} # a Set
            self.trajectories += [(k, ds) for k in keys]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """Get string name for this trajectory, as well as the h5py.File object in which to find the trajectory.
        """
        k, ds = self.trajectories[idx]
        return k, ds