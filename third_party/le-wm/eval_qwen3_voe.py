from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module import SIGReg
from qwen3_dataset import Qwen3ReasoningSequenceDataset, collate_qwen3_sequences, make_split_indices
from train_qwen3_reasoning import build_lewm_model, lewm_forward


def _category_permutation(categories: list[str], *, same_domain: bool, rng: np.random.Generator) -> np.ndarray:
    """Return a per-row permutation constrained by category when possible."""

    cats = np.array(categories)
    perm = np.arange(cats.shape[0])
    for i, category in enumerate(cats):
        if same_domain:
            candidates = np.flatnonzero((cats == category) & (np.arange(cats.shape[0]) != i))
        else:
            candidates = np.flatnonzero(cats != category)
        if candidates.size:
            perm[i] = int(rng.choice(candidates))
        else:
            fallback = np.delete(np.arange(cats.shape[0]), i)
            if fallback.size:
                perm[i] = int(rng.choice(fallback))
    return perm


def _latent_mse(out: dict[str, torch.Tensor]) -> list[float]:
    return (out["pred_emb"] - out["target_emb"]).pow(2).mean(dim=(1, 2)).detach().cpu().tolist()


def _image_mse(pixels: torch.Tensor, perm: torch.Tensor | None = None) -> list[float]:
    target = pixels[:, 1] if perm is None else pixels[perm, 1]
    return (pixels[:, 0] - target).pow(2).mean(dim=(1, 2, 3)).detach().cpu().tolist()


def evaluate_voe(cfg: dict[str, Any], checkpoint_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(cfg["dataset_path"])
    full = Qwen3ReasoningSequenceDataset(dataset_path)
    _, val_idx = make_split_indices(full.length, float(cfg.get("val_fraction", 0.05)), int(cfg.get("seed", 7)))
    val_idx = val_idx[: int(cfg.get("voe_max_examples", 1024))]
    dataset = Qwen3ReasoningSequenceDataset(dataset_path, val_idx)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("voe_batch_size", cfg.get("loader", {}).get("batch_size", 128))),
        shuffle=False,
        collate_fn=collate_qwen3_sequences,
    )
    device = torch.device("cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else cfg.get("device", "cpu"))
    model = build_lewm_model(cfg, obs_channels=dataset.obs_shape[0], action_dim=dataset.action_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sigreg = SIGReg(**cfg.get("loss", {}).get("sigreg", {}).get("kwargs", {})).to(device)
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    coherent_latent: list[float] = []
    intra_domain_latent: list[float] = []
    inter_domain_latent: list[float] = []
    coherent_image: list[float] = []
    intra_domain_image: list[float] = []
    inter_domain_image: list[float] = []
    with torch.no_grad():
        for batch in loader:
            categories = [str(v) for v in batch.get("category", [])]
            if not categories:
                categories = ["unknown"] * int(batch["pixels"].shape[0])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            good = lewm_forward(model, sigreg, batch, cfg)
            coherent_latent.extend(_latent_mse(good))
            coherent_image.extend(_image_mse(batch["pixels"]))

            intra_perm = torch.as_tensor(_category_permutation(categories, same_domain=True, rng=rng), device=device)
            intra_batch = dict(batch)
            intra_batch["pixels"] = batch["pixels"].clone()
            intra_batch["pixels"][:, 1] = batch["pixels"][intra_perm, 1]
            intra = lewm_forward(model, sigreg, intra_batch, cfg)
            intra_domain_latent.extend(_latent_mse(intra))
            intra_domain_image.extend(_image_mse(batch["pixels"], intra_perm))

            inter_perm = torch.as_tensor(_category_permutation(categories, same_domain=False, rng=rng), device=device)
            inter_batch = dict(batch)
            inter_batch["pixels"] = batch["pixels"].clone()
            inter_batch["pixels"][:, 1] = batch["pixels"][inter_perm, 1]
            inter = lewm_forward(model, sigreg, inter_batch, cfg)
            inter_domain_latent.extend(_latent_mse(inter))
            inter_domain_image.extend(_image_mse(batch["pixels"], inter_perm))

    coherent_latent_arr = np.array(coherent_latent, dtype=np.float64)
    intra_domain_latent_arr = np.array(intra_domain_latent, dtype=np.float64)
    inter_domain_latent_arr = np.array(inter_domain_latent, dtype=np.float64)
    coherent_image_arr = np.array(coherent_image, dtype=np.float64)
    intra_domain_image_arr = np.array(intra_domain_image, dtype=np.float64)
    inter_domain_image_arr = np.array(inter_domain_image, dtype=np.float64)
    coherent_latent_mean = float(coherent_latent_arr.mean())
    intra_domain_latent_mean = float(intra_domain_latent_arr.mean())
    inter_domain_latent_mean = float(inter_domain_latent_arr.mean())
    coherent_image_mean = float(coherent_image_arr.mean())
    intra_domain_image_mean = float(intra_domain_image_arr.mean())
    inter_domain_image_mean = float(inter_domain_image_arr.mean())
    report = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "n_examples": int(coherent_latent_arr.shape[0]),
        "coherent_mse_mean": coherent_latent_mean,
        "incoherent_mse_mean": inter_domain_latent_mean,
        "voe_ratio": float(inter_domain_latent_mean / max(coherent_latent_mean, 1e-12)),
        "coherent_latent_mse_mean": coherent_latent_mean,
        "intra_domain_latent_mse_mean": intra_domain_latent_mean,
        "inter_domain_latent_mse_mean": inter_domain_latent_mean,
        "intra_domain_voe_ratio": float(intra_domain_latent_mean / max(coherent_latent_mean, 1e-12)),
        "inter_domain_voe_ratio": float(inter_domain_latent_mean / max(coherent_latent_mean, 1e-12)),
        "coherent_image_mse_mean": coherent_image_mean,
        "intra_domain_image_mse_mean": intra_domain_image_mean,
        "inter_domain_image_mse_mean": inter_domain_image_mean,
        "intra_domain_image_ratio": float(intra_domain_image_mean / max(coherent_image_mean, 1e-12)),
        "inter_domain_image_ratio": float(inter_domain_image_mean / max(coherent_image_mean, 1e-12)),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(evaluate_voe(cfg, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
