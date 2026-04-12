from __future__ import annotations

import h5py


class HDF5Dataset:
    def __init__(self, path: str):
        self.path = path
        self.handle = h5py.File(path, "r")
        self.obs = self.handle["obs"]
        self.actions = self.handle["actions"]
        self.next_obs = self.handle["next_obs"]

    def __len__(self) -> int:
        return int(self.obs.shape[0])
