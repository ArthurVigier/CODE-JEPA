from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .contracts import TripletRecord


HDF5_KEYS = ("obs", "actions", "next_obs", "categories", "source_ids", "prompt_ids", "step_ids")


def write_triplets_hdf5(path: str | Path, triplets: list[TripletRecord], split: str, git_commit: str = "unknown") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for triplet in triplets:
        triplet.validate()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("obs", data=np.stack([t.obs_t.image for t in triplets], axis=0))
        handle.create_dataset("actions", data=np.stack([t.action.vector for t in triplets], axis=0))
        handle.create_dataset("next_obs", data=np.stack([t.obs_t1.image for t in triplets], axis=0))
        handle.create_dataset("categories", data=np.asarray([t.category.encode("utf-8") for t in triplets], dtype="S64"))
        handle.create_dataset("source_ids", data=np.asarray([t.source.encode("utf-8") for t in triplets], dtype="S64"))
        handle.create_dataset("prompt_ids", data=np.asarray([t.prompt_id.encode("utf-8") for t in triplets], dtype="S64"))
        handle.create_dataset("step_ids", data=np.asarray([t.step_id for t in triplets], dtype=np.int32))
        handle.attrs["model"] = triplets[0].model_name if triplets else "unknown"
        handle.attrs["layers"] = str(triplets[0].obs_t.layers if triplets else [])
        handle.attrs["obs_shape"] = str(list(triplets[0].obs_t.image.shape) if triplets else [])
        handle.attrs["encoding"] = triplets[0].encoding if triplets else "unknown"
        handle.attrs["action_encoding"] = triplets[0].action.encoding if triplets else "unknown"
        handle.attrs["split"] = split
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["git_commit"] = git_commit


def validate_hdf5(path: str | Path) -> dict[str, str]:
    with h5py.File(path, "r") as handle:
        for key in HDF5_KEYS:
            if key not in handle:
                raise ValueError(f"missing key {key}")
        obs = handle["obs"]
        actions = handle["actions"]
        next_obs = handle["next_obs"]
        if len(obs.shape) != 4 or obs.shape[1] < 1 or obs.shape[2] != obs.shape[3]:
            raise ValueError("obs shape mismatch")
        if next_obs.shape != obs.shape:
            raise ValueError("next_obs shape mismatch")
        if actions.shape[1] != 256:
            raise ValueError("actions shape mismatch")
        return {
            "n_records": str(obs.shape[0]),
            "encoding": str(handle.attrs.get("encoding", "")),
            "split": str(handle.attrs.get("split", "")),
        }
