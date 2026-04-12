from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch


LEWM_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "le-wm"
if str(LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(LEWM_ROOT))

from module import SIGReg
from qwen3_dataset import Qwen3ReasoningSequenceDataset, collate_qwen3_sequences
from train_qwen3_reasoning import build_lewm_model, evaluate_lewm, lewm_forward


def _write_triplets(path: Path, n: int = 8) -> None:
    rng = np.random.default_rng(7)
    obs = rng.random((n, 3, 64, 64), dtype=np.float32)
    next_obs = np.clip(obs + rng.normal(0.0, 0.01, obs.shape).astype(np.float32), 0.0, 1.0)
    actions = rng.normal(0.0, 1.0, (n, 256)).astype(np.float32)
    categories = np.array([b"code", b"math"] * (n // 2))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("obs", data=obs)
        handle.create_dataset("next_obs", data=next_obs)
        handle.create_dataset("actions", data=actions)
        handle.create_dataset("categories", data=categories)


def test_qwen3_dataset_exposes_lewm_sequence(tmp_path: Path) -> None:
    path = tmp_path / "triplets.h5"
    _write_triplets(path)
    dataset = Qwen3ReasoningSequenceDataset(path)
    item = dataset[0]
    assert item["pixels"].shape == (2, 3, 64, 64)
    assert item["pixels"].dtype == np.float32
    assert item["action"].shape == (1, 256)
    assert item["category"] == "code"

    batch = collate_qwen3_sequences([dataset[0], dataset[1]])
    assert batch["pixels"].shape == (2, 2, 3, 64, 64)
    assert batch["action"].shape == (2, 1, 256)
    assert batch["category"] == ["code", "math"]


def test_lewm_forward_and_eval_on_qwen_batch(tmp_path: Path) -> None:
    path = tmp_path / "triplets.h5"
    _write_triplets(path)
    cfg = {
        "dataset_path": str(path),
        "wm": {"history_size": 1, "num_preds": 1, "embed_dim": 32, "hidden_dim": 32, "projector_hidden_dim": 64},
        "predictor": {"depth": 1, "heads": 2, "mlp_dim": 64, "dim_head": 16},
        "loss": {"sigreg": {"weight": 0.01, "kwargs": {"knots": 5, "num_proj": 8}}},
    }
    dataset = Qwen3ReasoningSequenceDataset(path)
    batch = collate_qwen3_sequences([dataset[0], dataset[1], dataset[2], dataset[3]])
    model = build_lewm_model(cfg, obs_channels=3, action_dim=256)
    sigreg = SIGReg(knots=5, num_proj=8)
    out = lewm_forward(model, sigreg, batch, cfg)
    assert out["pred_emb"].shape == (4, 1, 32)
    assert out["target_emb"].shape == (4, 1, 32)
    assert torch.isfinite(out["loss"])

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_qwen3_sequences)
    metrics = evaluate_lewm(model, sigreg, loader, torch.device("cpu"), cfg)
    assert set(metrics) == {"latent_model_mse", "latent_identity_mse", "latent_improvement_ratio", "image_identity_mse"}
    assert metrics["latent_model_mse"] >= 0.0
