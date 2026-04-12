from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from .contracts import ActivationSnapshot, ActionVector, ObservationImage


def _normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    low = float(array.min())
    high = float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - low) / (high - low)


def _resize_to_rgb(array: np.ndarray, mode: str, resolution: int = 64) -> np.ndarray:
    pil = Image.fromarray((array * 255).astype("uint8"), mode=mode).convert("RGB")
    pil = pil.resize((resolution, resolution), resample=Image.Resampling.BILINEAR)
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return np.transpose(out, (2, 0, 1)).astype(np.float32)


def _sequence_to_square(array: np.ndarray, target: int = 64) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("expected [seq, 3] array")
    seq_len = array.shape[0]
    needed = target * target
    if seq_len >= needed:
        padded = array[:needed]
    else:
        repeats = int(np.ceil(needed / max(seq_len, 1)))
        padded = np.tile(array, (repeats, 1))[:needed]
    return padded.reshape(target, target, 3)


def _pca_to_hsv(projected: np.ndarray) -> np.ndarray:
    hsv = np.empty_like(projected, dtype=np.float32)
    hsv[:, 0] = projected[:, 0]
    hsv[:, 1] = 0.35 + 0.65 * projected[:, 1]
    hsv[:, 2] = 0.55 + 0.45 * projected[:, 2]
    return np.clip(hsv, 0.0, 1.0)


def _robust_scale(array: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    array = array.astype(np.float32)
    low, high = np.percentile(array, [p_low, p_high])
    if float(high - low) < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _resize_matrix(matrix: np.ndarray, resolution: int) -> np.ndarray:
    pil = Image.fromarray((_robust_scale(matrix) * 255).astype("uint8"), mode="L")
    pil = pil.resize((resolution, resolution), resample=Image.Resampling.BILINEAR)
    return (np.asarray(pil, dtype=np.float32) / 255.0).astype(np.float32)


def _canonicalize_svd_signs(u: np.ndarray, vt: np.ndarray) -> np.ndarray:
    u = u.copy()
    for idx in range(min(u.shape[1], vt.shape[0])):
        anchor = int(np.argmax(np.abs(vt[idx])))
        if vt[idx, anchor] < 0:
            u[:, idx] *= -1
    return u


def _residual_to_svd_heatmap(residual: np.ndarray, components: int, resolution: int) -> np.ndarray:
    centered = residual.astype(np.float32) - residual.astype(np.float32).mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(components, u.shape[1], singular_values.shape[0])
    u = _canonicalize_svd_signs(u[:, :n_components], vt[:n_components])
    heatmap = u * singular_values[:n_components][None, :]
    if n_components < components:
        padded = np.zeros((heatmap.shape[0], components), dtype=np.float32)
        padded[:, :n_components] = heatmap
        heatmap = padded
    return _resize_matrix(heatmap, resolution)


def _gaussian_diffuse(field: np.ndarray, steps: int = 5, source_strength: float = 0.15) -> np.ndarray:
    temperature = field.astype(np.float32).copy()
    source = np.square(np.clip(field, 0.0, 1.0)).astype(np.float32)
    kernel = np.asarray(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    for _ in range(steps):
        padded = np.pad(temperature, 1, mode="edge")
        blurred = (
            kernel[0, 0] * padded[:-2, :-2]
            + kernel[0, 1] * padded[:-2, 1:-1]
            + kernel[0, 2] * padded[:-2, 2:]
            + kernel[1, 0] * padded[1:-1, :-2]
            + kernel[1, 1] * padded[1:-1, 1:-1]
            + kernel[1, 2] * padded[1:-1, 2:]
            + kernel[2, 0] * padded[2:, :-2]
            + kernel[2, 1] * padded[2:, 1:-1]
            + kernel[2, 2] * padded[2:, 2:]
        )
        temperature = blurred + source_strength * source
    return _robust_scale(temperature)


def _edge_magnitude(field: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(field, dtype=np.float32)
    dy = np.zeros_like(field, dtype=np.float32)
    dx[:, 1:] = field[:, 1:] - field[:, :-1]
    dy[1:, :] = field[1:, :] - field[:-1, :]
    return _robust_scale(np.sqrt(dx * dx + dy * dy))


def _residual_to_thermal_heatmap(
    residual: np.ndarray,
    components: int,
    resolution: int,
    diffusion_steps: int = 5,
) -> np.ndarray:
    source_field = _residual_to_svd_heatmap(residual, components=components, resolution=resolution)
    return _gaussian_diffuse(source_field, steps=diffusion_steps)


def _residual_to_thermal_dynamics(residual: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _residual_to_svd_heatmap(residual, components=64, resolution=resolution)
    temperature = _gaussian_diffuse(source, steps=5, source_strength=0.08)
    gradient = _edge_magnitude(temperature)
    return source, temperature, gradient


def _residual_to_token_similarity(residual: np.ndarray, resolution: int) -> np.ndarray:
    z = residual.astype(np.float32)
    z = z - z.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-6)
    similarity = z @ z.T
    similarity = (similarity + 1.0) / 2.0
    return _resize_matrix(similarity, resolution)


def _residual_to_flow_components(residual: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = residual.astype(np.float32) - residual.astype(np.float32).mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, u.shape[1], singular_values.shape[0])
    if n_components < 2:
        return (
            np.zeros((resolution, resolution), dtype=np.float32),
            np.zeros((resolution, resolution), dtype=np.float32),
            np.zeros((resolution, resolution), dtype=np.float32),
        )
    u = _canonicalize_svd_signs(u[:, :2], vt[:2])
    velocity = u * singular_values[:2][None, :]
    vx = _resize_matrix(velocity[:, 0:1], resolution)
    vy = _resize_matrix(velocity[:, 1:2], resolution)
    vx_signed = (vx - 0.5) * 2.0
    vy_signed = (vy - 0.5) * 2.0
    magnitude = _robust_scale(np.sqrt(vx_signed * vx_signed + vy_signed * vy_signed))
    dvx_dx = np.gradient(vx_signed, axis=1)
    dvy_dy = np.gradient(vy_signed, axis=0)
    dvy_dx = np.gradient(vy_signed, axis=1)
    dvx_dy = np.gradient(vx_signed, axis=0)
    divergence = _robust_scale(dvx_dx + dvy_dy)
    curl = _robust_scale(dvy_dx - dvx_dy)
    return magnitude, divergence, curl


def _residual_to_particle_flow_components(residual: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = residual.astype(np.float32) - residual.astype(np.float32).mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(4, u.shape[1], singular_values.shape[0])
    if n_components < 4:
        zeros = np.zeros((resolution, resolution), dtype=np.float32)
        return zeros, zeros, zeros
    u = _canonicalize_svd_signs(u[:, :4], vt[:4])
    coords_raw = u[:, :2] * singular_values[:2][None, :]
    velocity_raw = u[:, 2:4] * singular_values[2:4][None, :]
    coords = np.stack([_robust_scale(coords_raw[:, 0]), _robust_scale(coords_raw[:, 1])], axis=1)
    velocity = velocity_raw / (np.percentile(np.abs(velocity_raw), 95) + 1e-6)
    field_x = np.zeros((resolution, resolution), dtype=np.float32)
    field_y = np.zeros((resolution, resolution), dtype=np.float32)
    weights = np.zeros((resolution, resolution), dtype=np.float32)
    sigma = max(resolution / 24.0, 1.0)
    radius = max(2, int(3 * sigma))
    for (x_norm, y_norm), (vx, vy) in zip(coords, velocity):
        cx = int(np.clip(round(x_norm * (resolution - 1)), 0, resolution - 1))
        cy = int(np.clip(round(y_norm * (resolution - 1)), 0, resolution - 1))
        x0, x1 = max(0, cx - radius), min(resolution, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(resolution, cy + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
        field_x[y0:y1, x0:x1] += kernel * float(vx)
        field_y[y0:y1, x0:x1] += kernel * float(vy)
        weights[y0:y1, x0:x1] += kernel
    mask = weights > 1e-6
    field_x[mask] /= weights[mask]
    field_y[mask] /= weights[mask]
    magnitude = _robust_scale(np.sqrt(field_x * field_x + field_y * field_y))
    dfx_dx = np.gradient(field_x, axis=1)
    dfy_dy = np.gradient(field_y, axis=0)
    dfy_dx = np.gradient(field_y, axis=1)
    dfx_dy = np.gradient(field_x, axis=0)
    divergence = _robust_scale(dfx_dx + dfy_dy)
    curl = _robust_scale(dfy_dx - dfx_dy)
    return magnitude, divergence, curl


def _project_three_channels(residual: np.ndarray) -> np.ndarray:
    n_components = min(3, residual.shape[0], residual.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(residual)
    if n_components < 3:
        padded = np.zeros((projected.shape[0], 3), dtype=np.float32)
        padded[:, :n_components] = projected
        projected = padded
    return _normalize(projected)


def snapshot_to_image(snapshot: ActivationSnapshot, encoding: str = "hsv_pca3", resolution: int = 64) -> ObservationImage:
    snapshot.validate()
    layer_images: list[np.ndarray] = []
    for layer in snapshot.layers:
        projected = _project_three_channels(snapshot.residuals_by_layer[layer])
        if encoding == "hsv_pca3":
            layer_images.append(_resize_to_rgb(_sequence_to_square(_pca_to_hsv(projected), target=resolution), "HSV", resolution=resolution))
        elif encoding == "lab_pca3":
            pseudo_lab = np.clip(projected, 0.0, 1.0)
            layer_images.append(_resize_to_rgb(_sequence_to_square(pseudo_lab, target=resolution), "RGB", resolution=resolution))
        elif encoding == "pca256_projection":
            flat = snapshot.residuals_by_layer[layer].reshape(-1)
            target = np.zeros(resolution * resolution * 3, dtype=np.float32)
            target[: min(target.size, flat.size)] = _normalize(flat[: min(target.size, flat.size)])
            layer_images.append(target.reshape(3, resolution, resolution))
        elif encoding in {"residual_svd64_v1", "residual_svd128_v1"}:
            components = 64 if encoding == "residual_svd64_v1" else 128
            layer_images.append(_residual_to_svd_heatmap(snapshot.residuals_by_layer[layer], components, resolution))
        elif encoding in {"thermal_svd64_v1", "thermal_svd128_v1"}:
            components = 64 if encoding == "thermal_svd64_v1" else 128
            layer_images.append(_residual_to_thermal_heatmap(snapshot.residuals_by_layer[layer], components, resolution))
        elif encoding == "thermal_dynamics_v1":
            layer_images.append(_residual_to_thermal_dynamics(snapshot.residuals_by_layer[layer], resolution))
        elif encoding == "token_similarity_v1":
            layer_images.append(_residual_to_token_similarity(snapshot.residuals_by_layer[layer], resolution))
        elif encoding == "flow_svd_v1":
            layer_images.append(_residual_to_flow_components(snapshot.residuals_by_layer[layer], resolution))
        elif encoding == "flow_particles_v1":
            layer_images.append(_residual_to_particle_flow_components(snapshot.residuals_by_layer[layer], resolution))
        elif encoding == "token_similarity_depth_v1":
            layer_images.append(_residual_to_token_similarity(snapshot.residuals_by_layer[layer], resolution))
        else:
            raise ValueError(f"encoding not implemented in core path: {encoding}")
    if encoding in {"residual_svd64_v1", "residual_svd128_v1", "thermal_svd64_v1", "thermal_svd128_v1"}:
        if len(layer_images) != 3:
            raise ValueError("svd-derived encodings require exactly three layers")
        image = np.stack(layer_images, axis=0).astype(np.float32)
    elif encoding == "thermal_dynamics_v1":
        if len(layer_images) != 3:
            raise ValueError("thermal dynamics encoding requires exactly three layers")
        sources = np.stack([item[0] for item in layer_images], axis=0)
        temperatures = np.stack([item[1] for item in layer_images], axis=0)
        gradients = np.stack([item[2] for item in layer_images], axis=0)
        image = np.stack(
            [
                np.mean(sources, axis=0),
                np.mean(temperatures, axis=0),
                np.mean(gradients, axis=0),
            ],
            axis=0,
        ).astype(np.float32)
    elif encoding == "token_similarity_v1":
        if len(layer_images) != 3:
            raise ValueError("token similarity encoding requires exactly three layers")
        image = np.stack(layer_images, axis=0).astype(np.float32)
    elif encoding == "token_similarity_depth_v1":
        image = np.stack(layer_images, axis=0).astype(np.float32)
    elif encoding == "flow_svd_v1":
        if len(layer_images) != 3:
            raise ValueError("flow encoding requires exactly three layers")
        magnitudes = np.stack([item[0] for item in layer_images], axis=0)
        divergences = np.stack([item[1] for item in layer_images], axis=0)
        curls = np.stack([item[2] for item in layer_images], axis=0)
        image = np.stack(
            [
                np.mean(magnitudes, axis=0),
                np.mean(divergences, axis=0),
                np.mean(curls, axis=0),
            ],
            axis=0,
        ).astype(np.float32)
    elif encoding == "flow_particles_v1":
        if len(layer_images) != 3:
            raise ValueError("particle flow encoding requires exactly three layers")
        magnitudes = np.stack([item[0] for item in layer_images], axis=0)
        divergences = np.stack([item[1] for item in layer_images], axis=0)
        curls = np.stack([item[2] for item in layer_images], axis=0)
        image = np.stack(
            [
                np.mean(magnitudes, axis=0),
                np.mean(divergences, axis=0),
                np.mean(curls, axis=0),
            ],
            axis=0,
        ).astype(np.float32)
    else:
        image = np.mean(np.stack(layer_images, axis=0), axis=0).astype(np.float32)
    obs = ObservationImage(
        image=image,
        encoding=encoding,
        model_name=snapshot.model_name,
        layers=snapshot.layers,
        metadata={
            "prompt_id": snapshot.prompt_id,
            "category": snapshot.category,
            "source": snapshot.source,
            "resolution": resolution,
        },
        image_shape=tuple(image.shape),
    )
    obs.validate()
    return obs


@dataclass(slots=True)
class ActionProjector:
    pca_components: np.ndarray
    mean: np.ndarray

    @classmethod
    def fit_from_embeddings(cls, embeddings: np.ndarray, n_components: int = 256) -> "ActionProjector":
        pca = PCA(n_components=n_components)
        pca.fit(embeddings.astype(np.float32))
        return cls(pca_components=pca.components_.astype(np.float32), mean=pca.mean_.astype(np.float32))

    def transform(self, embedding: np.ndarray) -> ActionVector:
        centered = embedding.astype(np.float32) - self.mean
        vector = centered @ self.pca_components.T
        action = ActionVector(vector=vector.astype(np.float32))
        action.validate()
        return action
