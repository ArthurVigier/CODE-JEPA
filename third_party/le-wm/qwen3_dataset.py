from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class Qwen3ReasoningSequenceDataset:
    """Expose canonical Activation-as-View triplets as LeWM sequences.

    Each canonical `(obs_t, action_t, obs_t1)` record becomes a two-frame LeWM
    sequence with `pixels=[obs_t, obs_t1]` and a single action at `t`.
    """

    def __init__(self, path: str | Path, indices: np.ndarray | None = None) -> None:
        self.path = str(path)
        self.indices = None if indices is None else indices.astype(np.int64)
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.length = int(handle["obs"].shape[0])
            self.obs_shape = tuple(int(v) for v in handle["obs"].shape[1:])
            self.action_dim = int(handle["actions"].shape[1])
            self.attrs = dict(handle.attrs)

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return int(self.indices.shape[0]) if self.indices is not None else self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = int(self.indices[idx]) if self.indices is not None else int(idx)
        handle = self._h5()
        obs = handle["obs"][real_idx].astype(np.float32)
        next_obs = handle["next_obs"][real_idx].astype(np.float32)
        action = handle["actions"][real_idx].astype(np.float32)
        item: dict[str, Any] = {
            "pixels": np.stack([obs, next_obs], axis=0),
            "action": action[None, :],
            "index": real_idx,
        }
        if "categories" in handle:
            category = handle["categories"][real_idx]
            item["category"] = category.decode("utf-8") if isinstance(category, bytes) else str(category)
        return item

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def make_split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_n = max(1, int(n * val_fraction))
    return indices[val_n:], indices[:val_n]


def collate_qwen3_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "pixels": torch.from_numpy(np.stack([item["pixels"] for item in batch], axis=0)).float(),
        "action": torch.from_numpy(np.stack([item["action"] for item in batch], axis=0)).float(),
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }
    if "category" in batch[0]:
        out["category"] = [str(item["category"]) for item in batch]
    return out
