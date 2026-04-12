from pathlib import Path

import numpy as np

from activation_views.contracts import ActionVector, ObservationImage, TripletRecord
from activation_views.dataset_validation import validate_dataset
from activation_views.hdf5_io import write_triplets_hdf5


def test_validate_dataset(tmp_path: Path) -> None:
    obs = ObservationImage(np.random.rand(3, 64, 64).astype(np.float32), "token_similarity_v1", "model", [10, 20, 30])
    action = ActionVector(np.random.randn(256).astype(np.float32))
    triplets = [TripletRecord(obs, action, obs, "src", "cat", f"p{i}", i, "model", "token_similarity_v1") for i in range(4)]
    path = tmp_path / "dataset.h5"
    write_triplets_hdf5(path, triplets, split="test")
    report = validate_dataset(path, tmp_path / "validation")
    assert report["obs_shape"] == [4, 3, 64, 64]
    assert Path(report["report_path"]).exists()
    assert Path(report["preview_path"]).exists()
