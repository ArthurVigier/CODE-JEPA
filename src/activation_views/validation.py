from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .contracts import ObservationImage, TripletRecord


@dataclass(slots=True)
class Phase0Report:
    auc_ovr: float
    inter_intra_ratio: float
    saturation_mean: float
    n_images: int
    verdict: str
    pixel_std_mean: float = 0.0
    edge_energy_mean: float = 0.0
    near_constant_image_rate: float = 0.0
    edge_energy_mean_viz: float | None = None
    near_constant_image_rate_viz: float | None = None


def compute_probe_auc(images: np.ndarray, labels: np.ndarray) -> float:
    min_class = int(np.min(np.bincount(labels)))
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    scores = []
    logistic_kwargs = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
    }
    for train_idx, test_idx in cv.split(images, labels):
        model = LogisticRegression(**logistic_kwargs)
        model.fit(images[train_idx], labels[train_idx])
        proba = model.predict_proba(images[test_idx])
        scores.append(
            roc_auc_score(labels[test_idx], proba, multi_class="ovr", average="macro", labels=np.unique(labels))
        )
    return float(np.mean(scores))


def compute_inter_intra_ratio(images: np.ndarray, labels: np.ndarray) -> float:
    class_means = []
    intra = []
    for label in np.unique(labels):
        class_images = images[labels == label]
        class_mean = class_images.mean(axis=0)
        class_means.append(class_mean)
        intra.append(np.mean(np.linalg.norm(class_images - class_mean, axis=1)))
    class_means = np.stack(class_means, axis=0)
    inter = np.mean(np.linalg.norm(class_means[:, None, :] - class_means[None, :, :], axis=-1))
    intra_mean = float(np.mean(intra)) if intra else 1.0
    return float(inter / max(intra_mean, 1e-8))


def compute_saturation_mean(images_chw: np.ndarray) -> float:
    return float(np.mean(np.clip(images_chw, 0.0, 1.0)))


def compute_pixel_std_mean(images_chw: np.ndarray) -> float:
    flattened = images_chw.reshape(images_chw.shape[0], -1)
    return float(np.mean(np.std(flattened, axis=1)))


def compute_near_constant_image_rate(images_chw: np.ndarray, std_threshold: float = 0.03) -> float:
    flattened = images_chw.reshape(images_chw.shape[0], -1)
    return float(np.mean(np.std(flattened, axis=1) < std_threshold))


def compute_edge_energy_mean(images_chw: np.ndarray) -> float:
    images = np.clip(images_chw, 0.0, 1.0)
    dx = np.diff(images, axis=-1)
    dy = np.diff(images, axis=-2)
    return float((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2.0)


def kill_go_verdict(auc_ovr: float, inter_intra_ratio: float, near_constant_image_rate: float = 0.0) -> str:
    if auc_ovr < 0.60:
        return "kill"
    if auc_ovr >= 0.65 and inter_intra_ratio >= 1.2 and near_constant_image_rate < 0.05:
        return "go"
    return "fallback"


def save_image_grid(observations: list[ObservationImage], output_path: Path, max_images: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[ObservationImage]] = {}
    for obs in observations:
        category = str(obs.metadata.get("category", "unknown"))
        by_category.setdefault(category, []).append(obs)
    subset: list[ObservationImage] = []
    categories = sorted(by_category)
    per_category = max(1, max_images // max(len(categories), 1))
    for category in categories:
        subset.extend(by_category[category][:per_category])
    subset = subset[:max_images]
    cols = 5
    rows = max(1, int(np.ceil(len(subset) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, obs in zip(axes.flat, subset):
        ax.imshow(_image_for_display(obs.image))
        ax.set_title(str(obs.metadata.get("category", ""))[:12])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_encoding_compare(
    train_observations: list[ObservationImage],
    viz_observations: list[ObservationImage],
    output_path: Path,
    max_pairs: int = 8,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(train_observations, viz_observations))[:max_pairs]
    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, len(pairs) * 2.5))
    axes = np.atleast_2d(axes)
    for row, (train_obs, viz_obs) in enumerate(pairs):
        axes[row, 0].imshow(_image_for_display(train_obs.image))
        axes[row, 0].set_title(f"{train_obs.metadata.get('category', '')} train")
        axes[row, 1].imshow(_image_for_display(viz_obs.image))
        axes[row, 1].set_title(f"{viz_obs.metadata.get('category', '')} viz")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _image_for_display(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] == 1:
        return image_chw[0]
    if image_chw.shape[0] == 2:
        third = np.zeros_like(image_chw[0])
        return np.transpose(np.stack([image_chw[0], image_chw[1], third], axis=0), (1, 2, 0))
    if image_chw.shape[0] == 3:
        return np.transpose(image_chw, (1, 2, 0))
    indices = np.linspace(0, image_chw.shape[0] - 1, 3).round().astype(int)
    return np.transpose(image_chw[indices], (1, 2, 0))


def build_phase0_report(
    observations: list[ObservationImage],
    labels: list[int],
    report_path: Path,
    viz_observations: list[ObservationImage] | None = None,
) -> Phase0Report:
    images = np.stack([obs.image.reshape(-1) for obs in observations], axis=0)
    labels_arr = np.asarray(labels)
    obs_images = np.stack([obs.image for obs in observations], axis=0)
    auc = compute_probe_auc(images, labels_arr)
    ratio = compute_inter_intra_ratio(images, labels_arr)
    saturation = compute_saturation_mean(obs_images)
    near_constant = compute_near_constant_image_rate(obs_images)
    edge_energy = compute_edge_energy_mean(obs_images)
    pixel_std = compute_pixel_std_mean(obs_images)
    edge_energy_viz = None
    near_constant_viz = None
    if viz_observations is not None:
        viz_images = np.stack([obs.image for obs in viz_observations], axis=0)
        edge_energy_viz = compute_edge_energy_mean(viz_images)
        near_constant_viz = compute_near_constant_image_rate(viz_images)
    report = Phase0Report(
        auc_ovr=auc,
        inter_intra_ratio=ratio,
        saturation_mean=saturation,
        n_images=len(observations),
        verdict=kill_go_verdict(auc, ratio, near_constant),
        pixel_std_mean=pixel_std,
        edge_energy_mean=edge_energy,
        near_constant_image_rate=near_constant,
        edge_energy_mean_viz=edge_energy_viz,
        near_constant_image_rate_viz=near_constant_viz,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2))
    return report


def temporal_coherence_score(triplets: list[TripletRecord]) -> float:
    distances = []
    for triplet in triplets:
        delta = triplet.obs_t1.image - triplet.obs_t.image
        distances.append(float(np.mean(np.abs(delta))))
    return float(np.mean(distances)) if distances else 0.0
