from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ALLOWED_ENCODINGS = {
    "hsv_pca3",
    "lab_pca3",
    "cosine_matrix",
    "jab8",
    "pca256_projection",
    "residual_svd64_v1",
    "residual_svd128_v1",
    "thermal_svd64_v1",
    "thermal_svd128_v1",
    "thermal_dynamics_v1",
    "token_similarity_v1",
    "flow_svd_v1",
    "flow_particles_v1",
    "token_similarity_depth_v1",
}


@dataclass(slots=True)
class ActivationSnapshot:
    residuals_by_layer: dict[int, np.ndarray]
    prompt_id: str
    source: str
    category: str
    token_step: int
    model_name: str
    layers: list[int]
    seq_len: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.layers != sorted(self.layers):
            raise ValueError("layers must be sorted")
        if set(self.layers) != set(self.residuals_by_layer):
            raise ValueError("layers must match residuals_by_layer keys")
        for layer, residual in self.residuals_by_layer.items():
            if residual.ndim != 2:
                raise ValueError(f"layer {layer} residual must be [seq, hidden]")
            if residual.shape[0] != self.seq_len:
                raise ValueError("seq_len must match residual first dim")


@dataclass(slots=True)
class ObservationImage:
    image: np.ndarray
    encoding: str
    model_name: str
    layers: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    image_shape: tuple[int, int, int] | None = None

    def validate(self) -> None:
        if self.encoding not in ALLOWED_ENCODINGS:
            raise ValueError(f"unknown encoding: {self.encoding}")
        expected_shape = self.image_shape or self.image.shape
        if len(expected_shape) != 3 or expected_shape[0] < 1 or expected_shape[1] != expected_shape[2]:
            raise ValueError("image shape must be [channels, resolution, resolution]")
        if self.image.shape != expected_shape:
            raise ValueError(f"image must be {expected_shape}")
        if self.image.dtype != np.float32:
            raise ValueError("image must be float32")


@dataclass(slots=True)
class ActionVector:
    vector: np.ndarray
    encoding: str = "embedding_pca256"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.vector.shape != (256,):
            raise ValueError("action vector must be [256]")
        if self.vector.dtype != np.float32:
            raise ValueError("action vector must be float32")


@dataclass(slots=True)
class TripletRecord:
    obs_t: ObservationImage
    action: ActionVector
    obs_t1: ObservationImage
    source: str
    category: str
    prompt_id: str
    step_id: int
    model_name: str
    encoding: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.obs_t.validate()
        self.obs_t1.validate()
        self.action.validate()
        if self.obs_t.encoding != self.encoding or self.obs_t1.encoding != self.encoding:
            raise ValueError("triplet encoding mismatch")
