from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2, axis=tuple(range(1, a.ndim)))


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    flat_a = a.reshape(a.shape[0], -1).astype(np.float32)
    flat_b = b.reshape(b.shape[0], -1).astype(np.float32)
    flat_a = flat_a / (np.linalg.norm(flat_a, axis=1, keepdims=True) + 1e-8)
    flat_b = flat_b / (np.linalg.norm(flat_b, axis=1, keepdims=True) + 1e-8)
    return np.sum(flat_a * flat_b, axis=1)


def _aggregate(values: list[np.ndarray]) -> dict[str, float]:
    merged = np.concatenate(values).astype(np.float64)
    return {
        "mean": float(np.mean(merged)),
        "std": float(np.std(merged)),
        "p05": float(np.percentile(merged, 5)),
        "p50": float(np.percentile(merged, 50)),
        "p95": float(np.percentile(merged, 95)),
    }


def compute_dynamics_baselines(
    path: str | Path,
    output_path: str | Path = "artifacts/dataset_validation/dynamics_baselines.json",
    sample_limit: int | None = None,
    batch_size: int = 512,
) -> dict[str, Any]:
    path = Path(path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identity_mse = []
    obs_next_cosine = []
    delta_l1 = []
    delta_l2 = []
    action_norm = []
    action_delta_cosine = []
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        next_obs = handle["next_obs"]
        actions = handle["actions"]
        n = int(obs.shape[0])
        if sample_limit is not None:
            n = min(n, sample_limit)
        mean_next = np.mean(next_obs[: min(n, batch_size)].astype(np.float32), axis=0, keepdims=True)
        mean_mse = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            obs_batch = obs[start:end].astype(np.float32)
            next_batch = next_obs[start:end].astype(np.float32)
            action_batch = actions[start:end].astype(np.float32)
            delta = next_batch - obs_batch
            identity_mse.append(_mse(obs_batch, next_batch))
            mean_mse.append(_mse(np.broadcast_to(mean_next, next_batch.shape), next_batch))
            obs_next_cosine.append(_cosine(obs_batch, next_batch))
            delta_l1.append(np.mean(np.abs(delta), axis=(1, 2, 3)))
            delta_l2.append(np.sqrt(np.mean(delta * delta, axis=(1, 2, 3))))
            action_norm.append(np.linalg.norm(action_batch, axis=1))
            delta_flat = delta.reshape(delta.shape[0], -1)
            delta_energy = np.linalg.norm(delta_flat, axis=1, keepdims=True)
            action_energy = np.linalg.norm(action_batch, axis=1, keepdims=True)
            usable = min(delta_flat.shape[1], action_batch.shape[1])
            corr = np.sum(
                delta_flat[:, :usable] / (delta_energy + 1e-8)
                * action_batch[:, :usable] / (action_energy + 1e-8),
                axis=1,
            )
            action_delta_cosine.append(corr)
    report: dict[str, Any] = {
        "path": str(path),
        "sample_count": n,
        "identity_mse": _aggregate(identity_mse),
        "mean_next_mse": _aggregate(mean_mse),
        "obs_next_cosine": _aggregate(obs_next_cosine),
        "delta_l1": _aggregate(delta_l1),
        "delta_l2": _aggregate(delta_l2),
        "action_norm": _aggregate(action_norm),
        "action_delta_prefix_cosine": _aggregate(action_delta_cosine),
    }
    report["identity_vs_mean_mse_ratio"] = report["identity_mse"]["mean"] / (report["mean_next_mse"]["mean"] + 1e-12)
    output_path.write_text(json.dumps(report, indent=2))
    return report
