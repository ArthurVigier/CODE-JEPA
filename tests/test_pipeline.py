import json
from pathlib import Path

from activation_views.hdf5_io import validate_hdf5
from activation_views.pipeline import build_demo_triplets, run_phase0


def test_phase0_outputs(tmp_path: Path) -> None:
    result = run_phase0(tmp_path / "phase0", layers=[10, 20, 30], profile="debug")
    report = json.loads((tmp_path / "phase0" / "phase0_report.json").read_text())
    assert result["verdict"] in {"kill", "fallback", "go"}
    assert "auc_ovr" in report


def test_demo_triplets_outputs(tmp_path: Path) -> None:
    path = build_demo_triplets(tmp_path / "triplets.h5", profile="debug")
    meta = validate_hdf5(path)
    assert int(meta["n_records"]) > 0
