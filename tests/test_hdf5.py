from pathlib import Path

import numpy as np

from activation_views.contracts import ActionVector, ObservationImage, TripletRecord
from activation_views.hdf5_io import validate_hdf5, write_triplets_hdf5


def test_write_and_validate_hdf5(tmp_path: Path) -> None:
    obs = ObservationImage(np.ones((3, 64, 64), dtype=np.float32), "hsv_pca3", "model", [20, 40, 60])
    action = ActionVector(np.ones((256,), dtype=np.float32))
    triplet = TripletRecord(obs, action, obs, "source", "category", "prompt-1", 0, "model", "hsv_pca3")
    path = tmp_path / "demo.h5"
    write_triplets_hdf5(path, [triplet], split="debug")
    meta = validate_hdf5(path)
    assert meta["n_records"] == "1"
