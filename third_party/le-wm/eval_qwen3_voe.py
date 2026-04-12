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
    coherent: list[float] = []
    incoherent: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            good = lewm_forward(model, sigreg, batch, cfg)
            coherent.extend((good["pred_emb"] - good["target_emb"]).pow(2).mean(dim=(1, 2)).detach().cpu().tolist())
            perm = torch.as_tensor(rng.permutation(batch["pixels"].shape[0]), device=device)
            bad_batch = dict(batch)
            bad_batch["pixels"] = batch["pixels"].clone()
            bad_batch["pixels"][:, 1] = batch["pixels"][perm, 1]
            bad = lewm_forward(model, sigreg, bad_batch, cfg)
            incoherent.extend((bad["pred_emb"] - bad["target_emb"]).pow(2).mean(dim=(1, 2)).detach().cpu().tolist())
    coherent_arr = np.array(coherent, dtype=np.float64)
    incoherent_arr = np.array(incoherent, dtype=np.float64)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "n_examples": int(coherent_arr.shape[0]),
        "coherent_mse_mean": float(coherent_arr.mean()),
        "incoherent_mse_mean": float(incoherent_arr.mean()),
        "voe_ratio": float(incoherent_arr.mean() / max(coherent_arr.mean(), 1e-12)),
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
