from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from qwen3_dataset import Qwen3ReasoningSequenceDataset, collate_qwen3_sequences, make_split_indices


class JsonlFallbackWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({"tag": tag, "step": global_step, "value": float(scalar_value)}) + "\n")

    def close(self) -> None:
        return None


def make_writer(output_dir: Path):
    try:
        os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(output_dir / "tensorboard"))
    except Exception as exc:
        fallback = output_dir / "metrics.jsonl"
        print(f"TensorBoard unavailable ({exc!r}); writing scalar metrics to {fallback}", flush=True)
        return JsonlFallbackWriter(fallback)


class ConvSequenceEncoder(nn.Module):
    """Small pixel encoder exposing the HuggingFace-style output used by LeWM JEPA."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, pixels: torch.Tensor, interpolate_pos_encoding: bool = True) -> SimpleNamespace:
        emb = self.net(pixels.float())
        return SimpleNamespace(last_hidden_state=emb[:, None, :])


@dataclass(slots=True)
class LeWMTrainDecision:
    epoch: int
    decision: str
    reason: str
    improvement_ratio: float


def build_lewm_model(cfg: dict[str, Any], obs_channels: int, action_dim: int) -> JEPA:
    wm_cfg = cfg.get("wm", {})
    pred_cfg = cfg.get("predictor", {})
    embed_dim = int(wm_cfg.get("embed_dim", 192))
    hidden_dim = int(wm_cfg.get("hidden_dim", embed_dim))
    history_size = int(wm_cfg.get("history_size", 1))
    encoder = ConvSequenceEncoder(obs_channels, embed_dim)
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=int(pred_cfg.get("depth", 4)),
        heads=int(pred_cfg.get("heads", 6)),
        mlp_dim=int(pred_cfg.get("mlp_dim", 768)),
        dim_head=int(pred_cfg.get("dim_head", 32)),
        dropout=float(pred_cfg.get("dropout", 0.1)),
        emb_dropout=float(pred_cfg.get("emb_dropout", 0.0)),
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    projector_hidden = int(wm_cfg.get("projector_hidden_dim", 512))
    projector = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=projector_hidden)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=projector_hidden)
    return JEPA(encoder, predictor, action_encoder, projector=projector, pred_proj=predictor_proj)


def lewm_forward(model: JEPA, sigreg: SIGReg, batch: dict[str, Any], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    ctx_len = int(cfg.get("wm", {}).get("history_size", 1))
    n_preds = int(cfg.get("wm", {}).get("num_preds", 1))
    lambd = float(cfg.get("loss", {}).get("sigreg", {}).get("weight", 0.09))
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    target_emb = emb[:, n_preds:]
    pred_emb = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    pred_loss = (pred_emb - target_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    return {
        "loss": pred_loss + lambd * sigreg_loss,
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "emb": emb,
        "pred_emb": pred_emb,
        "target_emb": target_emb,
    }


def compute_probe_auc(model: JEPA, loader: DataLoader, device: torch.device, max_examples: int) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    model.eval()
    xs: list[np.ndarray] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"][:, :1].to(device)
            action = batch["action"][:, :1].to(device)
            out = model.encode({"pixels": pixels, "action": action})
            xs.append(out["emb"][:, 0].detach().cpu().numpy())
            labels.extend([str(v) for v in batch.get("category", [])])
            if sum(x.shape[0] for x in xs) >= max_examples:
                break
    if not xs:
        return None
    x = np.concatenate(xs, axis=0)[:max_examples]
    y_names = np.array(labels[: x.shape[0]])
    unique = np.unique(y_names)
    if x.shape[0] < 20 or unique.shape[0] < 2:
        return None
    y = np.searchsorted(unique, y_names)
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return None
    scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    for train_idx, test_idx in cv.split(x, y):
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        clf.fit(x[train_idx], y[train_idx])
        scores.append(roc_auc_score(y[test_idx], clf.predict_proba(x[test_idx]), multi_class="ovr", labels=np.arange(unique.shape[0])))
    return float(np.mean(scores))


def evaluate_lewm(model: JEPA, sigreg: SIGReg, loader: DataLoader, device: torch.device, cfg: dict[str, Any]) -> dict[str, float]:
    model.eval()
    pred_sum = 0.0
    identity_latent_sum = 0.0
    image_identity_sum = 0.0
    n_latent = 0
    n_pixels = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = lewm_forward(model, sigreg, batch, cfg)
            pred_sum += float(F.mse_loss(out["pred_emb"], out["target_emb"], reduction="sum").detach().cpu())
            identity_latent_sum += float(F.mse_loss(out["emb"][:, :1], out["target_emb"], reduction="sum").detach().cpu())
            image_identity_sum += float(F.mse_loss(batch["pixels"][:, 0], batch["pixels"][:, 1], reduction="sum").detach().cpu())
            n_latent += int(np.prod(out["target_emb"].shape))
            n_pixels += int(np.prod(batch["pixels"][:, 1].shape))
    model_mse = pred_sum / max(n_latent, 1)
    identity_latent_mse = identity_latent_sum / max(n_latent, 1)
    return {
        "latent_model_mse": model_mse,
        "latent_identity_mse": identity_latent_mse,
        "latent_improvement_ratio": identity_latent_mse / max(model_mse, 1e-12),
        "image_identity_mse": image_identity_sum / max(n_pixels, 1),
    }


def train_qwen3_reasoning(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(cfg["dataset_path"])
    output_dir = Path(cfg.get("output_dir", "artifacts/lewm_qwen3_reasoning"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 7))
    torch.manual_seed(seed)
    np.random.seed(seed)
    with h5py.File(dataset_path, "r") as handle:
        n_examples = int(handle["obs"].shape[0])
        obs_channels = int(handle["obs"].shape[1])
        action_dim = int(handle["actions"].shape[1])
    train_idx, val_idx = make_split_indices(n_examples, float(cfg.get("val_fraction", 0.05)), seed)
    train_ds = Qwen3ReasoningSequenceDataset(dataset_path, train_idx)
    val_ds = Qwen3ReasoningSequenceDataset(dataset_path, val_idx)
    loader_cfg = cfg.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", cfg.get("num_workers", 0)))
    common = {
        "batch_size": int(loader_cfg.get("batch_size", cfg.get("batch_size", 128))),
        "num_workers": num_workers,
        "collate_fn": collate_qwen3_sequences,
        "pin_memory": bool(loader_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
        common["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    device = torch.device("cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else cfg.get("device", "cpu"))
    model = build_lewm_model(cfg, obs_channels, action_dim).to(device)
    sigreg = SIGReg(**cfg.get("loss", {}).get("sigreg", {}).get("kwargs", {})).to(device)
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg.get("lr", 5e-5)), weight_decay=float(opt_cfg.get("weight_decay", 1e-3)))
    writer = make_writer(output_dir)
    max_epochs = int(cfg.get("trainer", {}).get("max_epochs", cfg.get("max_epochs", 20)))
    probe_every = int(cfg.get("probe_every_n_epochs", 5))
    check_epoch = int(cfg.get("check_epoch", 5))
    stop_below = float(cfg.get("stop_if_improvement_below", 1.2))
    continue_above = float(cfg.get("continue_if_improvement_above", 1.5))
    history = []
    decision = LeWMTrainDecision(0, "running", "training started", 0.0)
    start_time = time.time()
    for key, value in evaluate_lewm(model, sigreg, val_loader, device, cfg).items():
        writer.add_scalar(f"val/{key}", value, 0)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_pred_sum = 0.0
        train_sigreg_sum = 0.0
        train_n = 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = lewm_forward(model, sigreg, batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_clip = float(cfg.get("trainer", {}).get("gradient_clip_val", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_n = int(batch["pixels"].shape[0])
            train_pred_sum += float(out["pred_loss"].detach().cpu()) * batch_n
            train_sigreg_sum += float(out["sigreg_loss"].detach().cpu()) * batch_n
            train_n += batch_n
        val = evaluate_lewm(model, sigreg, val_loader, device, cfg)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_pred_loss": train_pred_sum / max(train_n, 1),
            "train_sigreg_loss": train_sigreg_sum / max(train_n, 1),
            **val,
        }
        if probe_every > 0 and epoch % probe_every == 0:
            row["probe_auc_ovr"] = compute_probe_auc(model, val_loader, device, int(cfg.get("probe_max_examples", 1000)))
        history.append(row)
        writer.add_scalar("train/pred_loss", row["train_pred_loss"], epoch)
        writer.add_scalar("train/sigreg_loss", row["train_sigreg_loss"], epoch)
        for key, value in val.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        if row.get("probe_auc_ovr") is not None:
            writer.add_scalar("probe/auc_ovr", float(row["probe_auc_ovr"]), epoch)
        print(json.dumps(row), flush=True)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg, "metrics": row}, output_dir / "latest.pt")
        torch.save(model, output_dir / f"lewm_qwen3_epoch_{epoch}_object.ckpt")
        if epoch == check_epoch:
            ratio = float(val["latent_improvement_ratio"])
            if ratio < stop_below:
                decision = LeWMTrainDecision(epoch, "stop_and_generate_horizon_10", f"latent_improvement_ratio {ratio:.3f} < {stop_below}", ratio)
                break
            if ratio > continue_above:
                decision = LeWMTrainDecision(epoch, "continue_to_epoch_20", f"latent_improvement_ratio {ratio:.3f} > {continue_above}", ratio)
            else:
                decision = LeWMTrainDecision(epoch, "continue_cautiously", f"latent_improvement_ratio {ratio:.3f} in gray zone", ratio)
    writer.close()
    final = {
        "status": "complete",
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "n_examples": n_examples,
        "train_examples": int(train_idx.shape[0]),
        "val_examples": int(val_idx.shape[0]),
        "decision": asdict(decision),
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(train_qwen3_reasoning(cfg), indent=2))


if __name__ == "__main__":
    main()
