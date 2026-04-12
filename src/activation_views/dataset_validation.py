from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _finite_stats(dataset, batch_size: int = 512) -> dict[str, float]:
    n = int(dataset.shape[0])
    total = 0
    total_sq = 0.0
    total_sum = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    nonfinite = 0
    for start in range(0, n, batch_size):
        batch = dataset[start : min(start + batch_size, n)].astype(np.float32)
        finite = np.isfinite(batch)
        nonfinite += int((~finite).sum())
        clean = np.where(finite, batch, 0.0)
        total += clean.size
        total_sum += float(clean.sum())
        total_sq += float((clean * clean).sum())
        min_value = min(min_value, float(np.min(clean)))
        max_value = max(max_value, float(np.max(clean)))
    mean = total_sum / max(total, 1)
    variance = total_sq / max(total, 1) - mean * mean
    return {
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "min": min_value,
        "max": max_value,
        "nonfinite": float(nonfinite),
    }


def _preview_image(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] == 3:
        return np.transpose(image_chw, (1, 2, 0))
    if image_chw.shape[0] == 1:
        return image_chw[0]
    idx = np.linspace(0, image_chw.shape[0] - 1, 3).round().astype(int)
    return np.transpose(image_chw[idx], (1, 2, 0))


def save_triplet_preview(path: str | Path, output_path: str | Path, max_triplets: int = 8) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        next_obs = handle["next_obs"]
        n = min(max_triplets, int(obs.shape[0]))
        fig, axes = plt.subplots(n, 2, figsize=(5, 2.5 * n))
        axes = np.atleast_2d(axes)
        for idx in range(n):
            axes[idx, 0].imshow(_preview_image(obs[idx]))
            axes[idx, 0].set_title(f"obs_t #{idx}")
            axes[idx, 1].imshow(_preview_image(next_obs[idx]))
            axes[idx, 1].set_title(f"obs_t1 #{idx}")
            axes[idx, 0].axis("off")
            axes[idx, 1].axis("off")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def validate_dataset(path: str | Path, output_dir: str | Path = "artifacts/dataset_validation") -> dict[str, Any]:
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as handle:
        required = ["obs", "actions", "next_obs", "categories", "source_ids", "prompt_ids", "step_ids"]
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"missing keys: {missing}")
        obs = handle["obs"]
        actions = handle["actions"]
        next_obs = handle["next_obs"]
        if obs.shape != next_obs.shape:
            raise ValueError("obs and next_obs shapes differ")
        if len(obs.shape) != 4 or obs.shape[1] < 1 or obs.shape[2] != obs.shape[3]:
            raise ValueError(f"bad obs shape: {obs.shape}")
        if len(actions.shape) != 2 or actions.shape[1] != 256:
            raise ValueError(f"bad action shape: {actions.shape}")
        sample_n = min(1024, int(obs.shape[0]))
        delta = next_obs[:sample_n].astype(np.float32) - obs[:sample_n].astype(np.float32)
        action_sample = actions[:sample_n].astype(np.float32)
        categories = handle["categories"][:sample_n]
        unique_categories = sorted({item.decode("utf-8") for item in categories})
        report: dict[str, Any] = {
            "path": str(path),
            "attrs": {key: str(value) for key, value in handle.attrs.items()},
            "obs_shape": list(obs.shape),
            "actions_shape": list(actions.shape),
            "next_obs_shape": list(next_obs.shape),
            "obs_stats": _finite_stats(obs),
            "next_obs_stats": _finite_stats(next_obs),
            "actions_stats": _finite_stats(actions),
            "temporal_coherence_sample": float(np.mean(np.abs(delta))),
            "action_norm_mean_sample": float(np.mean(np.linalg.norm(action_sample, axis=1))),
            "unique_categories_sample": unique_categories,
        }
    report_path = output_dir / f"{path.stem}.validation.json"
    report_path.write_text(json.dumps(report, indent=2))
    preview_path = output_dir / f"{path.stem}.preview.png"
    save_triplet_preview(path, preview_path)
    report["report_path"] = str(report_path)
    report["preview_path"] = str(preview_path)
    return report
