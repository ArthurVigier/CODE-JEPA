from pathlib import Path

import numpy as np

from activation_views.contracts import ActionVector, ObservationImage, TripletRecord
from activation_views.dynamics_baselines import compute_dynamics_baselines
from activation_views.hdf5_io import write_triplets_hdf5


def test_compute_dynamics_baselines(tmp_path: Path) -> None:
    triplets = []
    for idx in range(4):
        obs = ObservationImage(np.random.rand(3, 64, 64).astype(np.float32), "token_similarity_v1", "model", [10, 20, 30])
        next_obs = ObservationImage(np.clip(obs.image + 0.01, 0, 1).astype(np.float32), "token_similarity_v1", "model", [10, 20, 30])
        action = ActionVector(np.random.randn(256).astype(np.float32))
        triplets.append(TripletRecord(obs, action, next_obs, "src", "cat", f"p{idx}", idx, "model", "token_similarity_v1"))
    path = tmp_path / "dataset.h5"
    write_triplets_hdf5(path, triplets, split="test")
    report = compute_dynamics_baselines(path, tmp_path / "baselines.json")
    assert report["sample_count"] == 4
    assert report["identity_mse"]["mean"] > 0
    assert Path(tmp_path / "baselines.json").exists()
