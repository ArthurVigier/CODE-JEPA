from __future__ import annotations

from .hdf5_dataset import HDF5Dataset


class Qwen3ReasoningDataset(HDF5Dataset):
    def __getitem__(self, idx: int) -> dict:
        return {
            "observation": self.obs[idx],
            "action": self.actions[idx],
            "next_observation": self.next_obs[idx],
        }
