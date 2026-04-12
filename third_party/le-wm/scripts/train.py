from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_ROOT = ROOT.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from activation_views.pipeline import build_demo_triplets
from activation_views.logging_utils import LocalMetricLogger, write_json_summary
from stable_worldmodel.data import Qwen3ReasoningDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.exists():
        build_demo_triplets(dataset_path, profile="debug")
    dataset = Qwen3ReasoningDataset(str(dataset_path))
    logging_cfg = cfg.get("logging", {})
    logger = LocalMetricLogger(logging_cfg.get("tensorboard_dir", "artifacts/tensorboard/qwen3_reasoning"))
    logger.log_scalars({"train/n_examples": float(len(dataset)), "train/probe_every_n_epochs": float(cfg.get("probe_every_n_epochs", 5))}, 0)
    report = {
        "status": "stub",
        "task": cfg["task"],
        "n_examples": len(dataset),
        "probe_every_n_epochs": cfg.get("probe_every_n_epochs", 5),
        "logging_backend": logging_cfg.get("backend", "tensorboard"),
        "tensorboard_dir": logging_cfg.get("tensorboard_dir", "artifacts/tensorboard/qwen3_reasoning"),
        "message": "Replace this stub with the native LeWM training loop in the integrated fork.",
    }
    out = Path(logging_cfg.get("summary_path", "artifacts/lewm_train_summary.json"))
    write_json_summary(out, report)
    logger.close()
    print(out)


if __name__ == "__main__":
    main()
