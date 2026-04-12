from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dataclasses import asdict

import numpy as np

from .contracts import ActivationSnapshot, TripletRecord
from .encoding import ActionProjector, snapshot_to_image
from .env import load_local_env, require_env
from .extractors import TransformerActivationExtractor
from .hdf5_io import write_triplets_hdf5
from .prompts import stratified_phase0_prompts
from .validation import build_phase0_report, save_encoding_compare, save_image_grid


@dataclass(slots=True)
class RunProfile:
    name: str
    prompt_limit: int
    max_steps: int


RUN_PROFILES = {
    "debug": RunProfile(name="debug", prompt_limit=25, max_steps=2),
    "pilot": RunProfile(name="pilot", prompt_limit=250, max_steps=4),
    "full": RunProfile(name="full", prompt_limit=500, max_steps=8),
}


def fake_snapshot(prompt_id: str, source: str, category: str, layers: list[int], token_step: int, model_name: str) -> ActivationSnapshot:
    rng = np.random.default_rng(abs(hash((prompt_id, token_step, model_name))) % (2**32))
    residuals_by_layer = {
        layer: rng.normal(size=(64, 128)).astype(np.float32) + (layer / 100.0) for layer in layers
    }
    return ActivationSnapshot(
        residuals_by_layer=residuals_by_layer,
        prompt_id=prompt_id,
        source=source,
        category=category,
        token_step=token_step,
        model_name=model_name,
        layers=layers,
        seq_len=64,
    )


def run_phase0(output_dir: str | Path, layers: list[int], encoding: str = "hsv_pca3", profile: str = "debug") -> dict[str, str]:
    out = Path(output_dir)
    all_prompts = stratified_phase0_prompts()
    categories = sorted({prompt.category for prompt in all_prompts})
    per_category = max(1, RUN_PROFILES[profile].prompt_limit // max(len(categories), 1))
    prompts = []
    for category in categories:
        prompts.extend([prompt for prompt in all_prompts if prompt.category == category][:per_category])
    observations = []
    viz_observations = []
    labels = []
    label_map: dict[str, int] = {}
    for prompt in prompts:
        snapshot = fake_snapshot(prompt.prompt_id, prompt.source, prompt.category, layers, token_step=0, model_name="Qwen/Qwen3-32B")
        observation = snapshot_to_image(snapshot, encoding=encoding, resolution=64)
        viz_encoding = encoding
        viz_observation = snapshot_to_image(snapshot, encoding=viz_encoding, resolution=128)
        observations.append(observation)
        viz_observations.append(viz_observation)
        label_map.setdefault(prompt.category, len(label_map))
        labels.append(label_map[prompt.category])
    save_image_grid(observations, out / "phase0_grid.png")
    save_image_grid(observations, out / "phase0_grid_train_64.png")
    save_image_grid(viz_observations, out / "phase0_grid_viz_128.png")
    save_image_grid(viz_observations, out / "phase0_contact_sheet.png", max_images=25)
    save_encoding_compare(observations, viz_observations, out / "phase0_encoding_compare.png")
    report = build_phase0_report(observations, labels, out / "phase0_report.json", viz_observations=viz_observations)
    return {"verdict": report.verdict, "report_path": str(out / "phase0_report.json")}


def run_phase0_live(
    output_dir: str | Path,
    model_name: str,
    layers: list[int],
    encoding: str = "hsv_pca3",
    profile: str = "debug",
    hf_token_env: str = "HF_TOKEN",
    train_resolution: int = 64,
    viz_resolution: int = 128,
) -> dict[str, Any]:
    load_local_env()
    _ = require_env(hf_token_env)
    out = Path(output_dir)
    all_prompts = stratified_phase0_prompts()
    categories = sorted({prompt.category for prompt in all_prompts})
    per_category = max(1, RUN_PROFILES[profile].prompt_limit // max(len(categories), 1))
    prompts = []
    for category in categories:
        prompts.extend([prompt for prompt in all_prompts if prompt.category == category][:per_category])

    extractor = TransformerActivationExtractor(model_name=model_name, layer_ids=layers)
    observations = []
    viz_observations = []
    labels = []
    label_map: dict[str, int] = {}
    try:
        for prompt in prompts:
            snapshot = extractor.extract_snapshot(
                prompt_id=prompt.prompt_id,
                source=prompt.source,
                category=prompt.category,
                text=prompt.text,
                token_step=0,
            )
            observation = snapshot_to_image(snapshot, encoding=encoding, resolution=train_resolution)
            viz_encoding = encoding
            viz_observation = snapshot_to_image(snapshot, encoding=viz_encoding, resolution=viz_resolution)
            observations.append(observation)
            viz_observations.append(viz_observation)
            label_map.setdefault(prompt.category, len(label_map))
            labels.append(label_map[prompt.category])
    finally:
        extractor.close()

    save_image_grid(observations, out / "phase0_grid.png")
    save_image_grid(observations, out / "phase0_grid_train_64.png")
    save_image_grid(viz_observations, out / "phase0_grid_viz_128.png")
    save_image_grid(viz_observations, out / "phase0_contact_sheet.png", max_images=25)
    save_encoding_compare(observations, viz_observations, out / "phase0_encoding_compare.png")
    report = build_phase0_report(observations, labels, out / "phase0_report.json", viz_observations=viz_observations)
    return {
        "verdict": report.verdict,
        "report_path": str(out / "phase0_report.json"),
        "report": asdict(report),
        "grid_path": str(out / "phase0_grid.png"),
        "grid_train_path": str(out / "phase0_grid_train_64.png"),
        "grid_viz_path": str(out / "phase0_grid_viz_128.png"),
        "contact_sheet_path": str(out / "phase0_contact_sheet.png"),
        "encoding_compare_path": str(out / "phase0_encoding_compare.png"),
        "n_prompts": len(prompts),
        "model_name": model_name,
        "layers": layers,
        "train_resolution": train_resolution,
        "viz_resolution": viz_resolution,
    }


def build_demo_triplets(output_path: str | Path, profile: str = "debug") -> str:
    run_profile = RUN_PROFILES[profile]
    prompts = stratified_phase0_prompts()[: run_profile.prompt_limit]
    embeddings = np.random.default_rng(123).normal(size=(512, 300)).astype(np.float32)
    projector = ActionProjector.fit_from_embeddings(embeddings, n_components=256)
    triplets: list[TripletRecord] = []
    for prompt in prompts:
        for step in range(run_profile.max_steps):
            snap_t = fake_snapshot(prompt.prompt_id, prompt.source, prompt.category, [20, 40, 60], step, "Qwen/Qwen3-72B")
            snap_t1 = fake_snapshot(prompt.prompt_id, prompt.source, prompt.category, [20, 40, 60], step + 1, "Qwen/Qwen3-72B")
            triplets.append(
                TripletRecord(
                    obs_t=snapshot_to_image(snap_t),
                    action=projector.transform(embeddings[(step + len(prompt.prompt_id)) % len(embeddings)]),
                    obs_t1=snapshot_to_image(snap_t1),
                    source=prompt.source,
                    category=prompt.category,
                    prompt_id=prompt.prompt_id,
                    step_id=step,
                    model_name="Qwen/Qwen3-72B",
                    encoding="hsv_pca3",
                )
            )
    write_triplets_hdf5(output_path, triplets, split=profile, git_commit="local-dev")
    return str(output_path)
