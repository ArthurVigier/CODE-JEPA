from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ActionVector, ActivationSnapshot, TripletRecord
from .encoding import ActionProjector, snapshot_to_image
from .env import load_local_env, require_env
from .extractors import ResidualHookCollector
from .hdf5_io import write_triplets_hdf5
from .prompts import PromptRecord, stratified_phase0_prompts
from .validation import temporal_coherence_score


@dataclass(slots=True)
class DatasetGenerationReport:
    model_name: str
    layers: list[int]
    encoding: str
    profile: str
    n_prompts: int
    n_triplets: int
    output_path: str
    checkpoint_path: str
    temporal_coherence: float


def _select_stratified_prompts(limit: int) -> list[PromptRecord]:
    all_prompts = stratified_phase0_prompts()
    categories = sorted({prompt.category for prompt in all_prompts})
    per_category = max(1, limit // max(len(categories), 1))
    selected = []
    for category in categories:
        selected.extend([prompt for prompt in all_prompts if prompt.category == category][:per_category])
    return selected[:limit]


def _load_model_and_tokenizer(model_name: str, torch_dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _model_input_device(model):
    try:
        return model.model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device


def _snapshot_from_collector(
    collector: ResidualHookCollector,
    prompt: PromptRecord,
    token_step: int,
    model_name: str,
) -> ActivationSnapshot:
    return collector.snapshot(
        prompt_id=prompt.prompt_id,
        source=prompt.source,
        category=prompt.category,
        token_step=token_step,
        model_name=model_name,
    )


def fit_action_projector_from_model(model, sample_size: int | None = None) -> ActionProjector:
    embeddings = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    if sample_size is not None and sample_size < embeddings.shape[0]:
        rng = np.random.default_rng(17)
        indices = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        embeddings = embeddings[indices]
    return ActionProjector.fit_from_embeddings(embeddings, n_components=256)


def generate_qwen_triplets_live(
    output_path: str | Path,
    checkpoint_path: str | Path,
    model_name: str,
    layers: list[int],
    encoding: str,
    profile: str,
    prompt_limit: int,
    max_new_tokens: int,
    torch_dtype: str = "bfloat16",
    hf_token_env: str = "HF_TOKEN",
    action_pca_sample_size: int | None = None,
) -> dict[str, Any]:
    load_local_env()
    _ = require_env(hf_token_env)
    import torch

    model, tokenizer = _load_model_and_tokenizer(model_name, torch_dtype=torch_dtype)
    collector = ResidualHookCollector(layers)
    collector.attach(model)
    projector = fit_action_projector_from_model(model, sample_size=action_pca_sample_size)
    prompts = _select_stratified_prompts(prompt_limit)
    device = _model_input_device(model)
    triplets: list[TripletRecord] = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for prompt in prompts:
            input_ids = tokenizer(prompt.text, return_tensors="pt")["input_ids"].to(device)
            for step in range(max_new_tokens):
                collector.clear()
                with torch.no_grad():
                    out = model(input_ids=input_ids)
                snapshot_t = _snapshot_from_collector(collector, prompt, step, model_name)
                next_token = out.logits[:, -1, :].argmax(dim=-1)
                token_embedding = model.model.embed_tokens(next_token).detach().cpu().float().squeeze(0).numpy()
                action = projector.transform(token_embedding)
                if not isinstance(action, ActionVector):
                    raise TypeError("action projector did not return ActionVector")
                input_ids = torch.cat([input_ids, next_token[:, None].to(device)], dim=-1)
                collector.clear()
                with torch.no_grad():
                    model(input_ids=input_ids)
                snapshot_t1 = _snapshot_from_collector(collector, prompt, step + 1, model_name)
                triplets.append(
                    TripletRecord(
                        obs_t=snapshot_to_image(snapshot_t, encoding=encoding),
                        action=action,
                        obs_t1=snapshot_to_image(snapshot_t1, encoding=encoding),
                        source=prompt.source,
                        category=prompt.category,
                        prompt_id=prompt.prompt_id,
                        step_id=step,
                        model_name=model_name,
                        encoding=encoding,
                    )
                )
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "last_prompt_id": prompt.prompt_id,
                        "n_triplets": len(triplets),
                        "profile": profile,
                    },
                    indent=2,
                )
            )
    finally:
        collector.detach()

    write_triplets_hdf5(output_path, triplets, split=profile, git_commit="modal")
    report = DatasetGenerationReport(
        model_name=model_name,
        layers=layers,
        encoding=encoding,
        profile=profile,
        n_prompts=len(prompts),
        n_triplets=len(triplets),
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        temporal_coherence=temporal_coherence_score(triplets),
    )
    report_path = Path(output_path).with_suffix(".report.json")
    report_path.write_text(json.dumps(asdict(report), indent=2))
    return asdict(report) | {"report_path": str(report_path)}
