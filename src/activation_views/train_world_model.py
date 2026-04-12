from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(slots=True)
class TrainDecision:
    epoch: int
    decision: str
    reason: str
    improvement_ratio: float


class HDF5TripletTorchDataset:
    def __init__(self, path: str | Path, indices: np.ndarray):
        self.path = str(path)
        self.indices = indices.astype(np.int64)
        self._handle = None

    def _h5(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        import torch

        real_idx = int(self.indices[idx])
        handle = self._h5()
        obs = torch.from_numpy(handle["obs"][real_idx].astype(np.float32))
        action = torch.from_numpy(handle["actions"][real_idx].astype(np.float32))
        next_obs = torch.from_numpy(handle["next_obs"][real_idx].astype(np.float32))
        return obs, action, next_obs


def make_split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_n = max(1, int(n * val_fraction))
    return indices[val_n:], indices[:val_n]


def build_model(obs_channels: int, action_dim: int, hidden_dim: int = 256):
    import torch
    from torch import nn

    class ResidualWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(obs_channels, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
            )
            self.action = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 128),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv2d(64, obs_channels, kernel_size=3, padding=1),
            )

        def forward(self, obs, action):
            encoded = self.obs_encoder(obs)
            action_features = self.action(action)[:, :, None, None].expand(-1, -1, encoded.shape[-2], encoded.shape[-1])
            delta = self.decoder(torch.cat([encoded, action_features], dim=1))
            return torch.clamp(obs + delta, 0.0, 1.0)

    return ResidualWorldModel()


def evaluate(model, loader, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    model_mse_sum = 0.0
    identity_mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for obs, action, next_obs in loader:
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)
            pred = model(obs, action)
            batch_n = obs.shape[0]
            model_mse_sum += float(F.mse_loss(pred, next_obs, reduction="sum").detach().cpu())
            identity_mse_sum += float(F.mse_loss(obs, next_obs, reduction="sum").detach().cpu())
            n += int(np.prod(next_obs.shape))
    model_mse = model_mse_sum / max(n, 1)
    identity_mse = identity_mse_sum / max(n, 1)
    return {
        "model_mse": model_mse,
        "identity_mse": identity_mse,
        "improvement_ratio": identity_mse / max(model_mse, 1e-12),
    }


def train_world_model(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config.get("output_dir", "artifacts/world_model_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 7))
    torch.manual_seed(seed)
    np.random.seed(seed)

    with h5py.File(dataset_path, "r") as handle:
        n = int(handle["obs"].shape[0])
        obs_channels = int(handle["obs"].shape[1])
        action_dim = int(handle["actions"].shape[1])
    train_idx, val_idx = make_split_indices(n, float(config.get("val_fraction", 0.05)), seed)
    train_ds = HDF5TripletTorchDataset(dataset_path, train_idx)
    val_ds = HDF5TripletTorchDataset(dataset_path, val_idx)
    batch_size = int(config.get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(config.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(config.get("num_workers", 0)))

    if config.get("device", "auto") == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config["device"])

    model = build_model(obs_channels=obs_channels, action_dim=action_dim, hidden_dim=int(config.get("hidden_dim", 256))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    max_epochs = int(config.get("max_epochs", 20))
    check_epoch = int(config.get("check_epoch", 5))
    stop_below = float(config.get("stop_if_improvement_below", 1.2))
    continue_above = float(config.get("continue_if_improvement_above", 1.5))
    history = []
    decision = TrainDecision(epoch=0, decision="running", reason="training started", improvement_ratio=0.0)
    start_time = time.time()

    initial_eval = evaluate(model, val_loader, device)
    writer.add_scalar("val/model_mse", initial_eval["model_mse"], 0)
    writer.add_scalar("val/identity_mse", initial_eval["identity_mse"], 0)
    writer.add_scalar("val/improvement_ratio", initial_eval["improvement_ratio"], 0)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_pixels = 0
        for obs, action, next_obs in train_loader:
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)
            pred = model(obs, action)
            loss = F.mse_loss(pred, next_obs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * int(np.prod(next_obs.shape))
            train_pixels += int(np.prod(next_obs.shape))
        val_metrics = evaluate(model, val_loader, device)
        train_mse = train_loss_sum / max(train_pixels, 1)
        row = {"epoch": epoch, "train_mse": train_mse, **val_metrics}
        history.append(row)
        writer.add_scalar("train/mse", train_mse, epoch)
        writer.add_scalar("val/model_mse", val_metrics["model_mse"], epoch)
        writer.add_scalar("val/identity_mse", val_metrics["identity_mse"], epoch)
        writer.add_scalar("val/improvement_ratio", val_metrics["improvement_ratio"], epoch)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": row,
        }
        torch.save(checkpoint, output_dir / "latest.pt")

        if epoch == check_epoch:
            ratio = val_metrics["improvement_ratio"]
            if ratio < stop_below:
                decision = TrainDecision(epoch=epoch, decision="stop_and_generate_horizon_10", reason=f"improvement_ratio {ratio:.3f} < {stop_below}", improvement_ratio=ratio)
                break
            if ratio > continue_above:
                decision = TrainDecision(epoch=epoch, decision="continue_to_epoch_20", reason=f"improvement_ratio {ratio:.3f} > {continue_above}", improvement_ratio=ratio)
            else:
                decision = TrainDecision(epoch=epoch, decision="continue_cautiously", reason=f"improvement_ratio {ratio:.3f} in gray zone", improvement_ratio=ratio)

    writer.close()
    final = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "n_examples": n,
        "train_examples": int(train_idx.shape[0]),
        "val_examples": int(val_idx.shape[0]),
        "decision": asdict(decision),
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(final, indent=2))
    return final
