from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LEWM_ROOT = ROOT / "third_party" / "le-wm"
for path in (SRC, LEWM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from activation_views.contracts import ActivationSnapshot
from activation_views.encoding import snapshot_to_image
from activation_views.prompts import PromptRecord, stratified_phase0_prompts
from train_qwen3_reasoning import build_lewm_model


@dataclass(slots=True)
class CapturedPair:
    prompt: PromptRecord
    obs: np.ndarray
    next_obs: np.ndarray
    continuation: str


class GenericResidualCollector:
    def __init__(self, layer_ids: list[int]) -> None:
        self.layer_ids = sorted(layer_ids)
        self.residuals: dict[int, np.ndarray] = {}
        self.handles = []

    def attach(self, layers) -> None:
        for layer_id in self.layer_ids:
            handle = layers[layer_id].register_forward_hook(self._make_hook(layer_id))
            self.handles.append(handle)

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hasattr(hidden, "last_hidden_state"):
                hidden = hidden.last_hidden_state
            if not torch.is_tensor(hidden):
                raise TypeError(f"layer {layer_id} hook returned unsupported output type {type(hidden)!r}")
            hidden = hidden.detach().float().cpu()
            if hidden.ndim == 3:
                hidden = hidden[0]
            self.residuals[layer_id] = hidden.numpy().astype(np.float32, copy=False)

        return hook

    def clear(self) -> None:
        self.residuals.clear()

    def snapshot(self, prompt: PromptRecord, token_step: int, model_name: str) -> ActivationSnapshot:
        if set(self.layer_ids) != set(self.residuals):
            missing = sorted(set(self.layer_ids) - set(self.residuals))
            raise RuntimeError(f"missing residuals for layers {missing}")
        seq_len = int(self.residuals[self.layer_ids[0]].shape[0])
        snapshot = ActivationSnapshot(
            residuals_by_layer={layer: self.residuals[layer].copy() for layer in self.layer_ids},
            prompt_id=prompt.prompt_id,
            source=prompt.source,
            category=prompt.category,
            token_step=token_step,
            model_name=model_name,
            layers=self.layer_ids,
            seq_len=seq_len,
            metadata={"cross_arch": True},
        )
        snapshot.validate()
        return snapshot

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _find_transformer_layers(model):
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("layers", lambda m: m.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
        except Exception:
            continue
        if hasattr(layers, "__len__") and len(layers) > 0:
            return name, layers
    raise RuntimeError("could not locate transformer layers for hooks")


def _relative_layers(num_layers: int, relative: str) -> list[int]:
    layers = []
    for item in relative.split(","):
        frac = float(item)
        idx = int(round(num_layers * frac))
        layers.append(min(max(idx, 0), num_layers - 1))
    return sorted(dict.fromkeys(layers))


def _select_layers(model, explicit_layers: str | None, relative_layers: str) -> tuple[str, list[Any], list[int]]:
    layer_path, layers = _find_transformer_layers(model)
    if explicit_layers:
        layer_ids = [int(x) for x in explicit_layers.split(",")]
    else:
        layer_ids = _relative_layers(len(layers), relative_layers)
    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= len(layers):
            raise ValueError(f"layer {layer_id} out of range for {len(layers)} layers")
    return layer_path, layers, layer_ids


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _format_prompt(tokenizer, text: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def _forward_snapshot(
    *,
    model,
    tokenizer,
    collector: GenericResidualCollector,
    prompt: PromptRecord,
    text: str,
    max_input_tokens: int,
    model_name: str,
    token_step: int,
) -> ActivationSnapshot:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    device = _input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    collector.clear()
    with torch.no_grad():
        try:
            model(**inputs, use_cache=False)
        except TypeError:
            model(**inputs)
    return collector.snapshot(prompt, token_step=token_step, model_name=model_name)


def _generate_continuation(
    *,
    model,
    tokenizer,
    text: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    if max_new_tokens <= 0:
        return ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    device = _input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _load_prompts(n_per_category: int, seed: int) -> list[PromptRecord]:
    rng = np.random.default_rng(seed)
    by_category: dict[str, list[PromptRecord]] = {}
    for prompt in stratified_phase0_prompts():
        by_category.setdefault(prompt.category, []).append(prompt)
    selected: list[PromptRecord] = []
    for category in sorted(by_category):
        prompts = by_category[category]
        order = rng.permutation(len(prompts))[:n_per_category]
        selected.extend([prompts[int(i)] for i in order])
    return selected


def _load_lewm(checkpoint_path: str | Path, cfg: dict[str, Any], device: torch.device):
    model = build_lewm_model(cfg, obs_channels=3, action_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _resolve_lewm_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _encode_obs(lewm, obs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], batch_size):
            batch = torch.from_numpy(obs[start : start + batch_size].astype(np.float32)).to(device)
            pixels = batch[:, None]
            action = torch.zeros((pixels.shape[0], 1, 256), dtype=torch.float32, device=device)
            out = lewm.encode({"pixels": pixels, "action": action})
            embeddings.append(out["emb"][:, 0].detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _predict_next_zero_action(lewm, obs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], batch_size):
            batch = torch.from_numpy(obs[start : start + batch_size].astype(np.float32)).to(device)
            pixels = batch[:, None]
            action = torch.zeros((pixels.shape[0], 1, 256), dtype=torch.float32, device=device)
            encoded = lewm.encode({"pixels": pixels, "action": action})
            pred = lewm.predict(encoded["emb"], encoded["act_emb"])
            predictions.append(pred[:, 0].detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _category_permutation(categories: list[str], *, same_domain: bool, rng: np.random.Generator) -> np.ndarray:
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


def _mean_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def _probe_auc(embeddings: np.ndarray, categories: list[str], seed: int) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    labels = np.array(categories)
    unique = np.unique(labels)
    if unique.shape[0] < 2:
        return None
    y = np.searchsorted(unique, labels)
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return None
    scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(embeddings, y):
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        clf.fit(embeddings[train_idx], y[train_idx])
        scores.append(
            roc_auc_score(
                y[test_idx],
                clf.predict_proba(embeddings[test_idx]),
                multi_class="ovr",
                labels=np.arange(unique.shape[0]),
            )
        )
    return float(np.mean(scores))


def _write_progress(path: str | Path, row: dict[str, Any]) -> None:
    progress = Path(path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def run_cross_arch_glm(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text())
    prompts = _load_prompts(args.n_per_category, args.seed)
    started = time.time()
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
    model.eval()
    layer_path, layers, layer_ids = _select_layers(model, args.layers, args.relative_layers)
    collector = GenericResidualCollector(layer_ids)
    collector.attach(layers)

    pairs: list[CapturedPair] = []
    progress_path = Path(args.progress_output or str(Path(args.output).with_suffix(".progress.jsonl")))
    try:
        for idx, prompt in enumerate(prompts):
            formatted = _format_prompt(tokenizer, prompt.text, args.use_chat_template)
            snap_t = _forward_snapshot(
                model=model,
                tokenizer=tokenizer,
                collector=collector,
                prompt=prompt,
                text=formatted,
                max_input_tokens=args.max_input_tokens,
                model_name=args.model,
                token_step=0,
            )
            continuation = _generate_continuation(
                model=model,
                tokenizer=tokenizer,
                text=formatted,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.temporal_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            snap_t1 = _forward_snapshot(
                model=model,
                tokenizer=tokenizer,
                collector=collector,
                prompt=prompt,
                text=f"{formatted}{continuation}",
                max_input_tokens=args.max_input_tokens,
                model_name=args.model,
                token_step=args.temporal_new_tokens,
            )
            obs = snapshot_to_image(snap_t, encoding=args.encoding, resolution=args.resolution).image
            next_obs = snapshot_to_image(snap_t1, encoding=args.encoding, resolution=args.resolution).image
            pairs.append(CapturedPair(prompt=prompt, obs=obs, next_obs=next_obs, continuation=continuation))
            _write_progress(
                progress_path,
                {
                    "idx": idx,
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category,
                    "seq_len_t": snap_t.seq_len,
                    "seq_len_t1": snap_t1.seq_len,
                    "continuation_chars": len(continuation),
                },
            )
            print(json.dumps({"idx": idx, "prompt_id": prompt.prompt_id, "category": prompt.category}), flush=True)
    finally:
        collector.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    obs = np.stack([pair.obs for pair in pairs], axis=0).astype(np.float32)
    next_obs = np.stack([pair.next_obs for pair in pairs], axis=0).astype(np.float32)
    categories = [pair.prompt.category for pair in pairs]
    prompt_ids = [pair.prompt.prompt_id for pair in pairs]
    lewm_device = _resolve_lewm_device(args.lewm_device)
    lewm = _load_lewm(args.checkpoint, cfg, lewm_device)

    emb_t = _encode_obs(lewm, obs, lewm_device, args.batch_size)
    emb_t1 = _encode_obs(lewm, next_obs, lewm_device, args.batch_size)
    pred_t1 = _predict_next_zero_action(lewm, obs, lewm_device, args.batch_size)

    rng = np.random.default_rng(args.seed)
    intra_perm = _category_permutation(categories, same_domain=True, rng=rng)
    inter_perm = _category_permutation(categories, same_domain=False, rng=rng)

    coherent_latent_mse = _mean_mse(pred_t1, emb_t1)
    intra_latent_mse = _mean_mse(pred_t1, emb_t1[intra_perm])
    inter_latent_mse = _mean_mse(pred_t1, emb_t1[inter_perm])
    coherent_image_mse = _mean_mse(obs, next_obs)
    intra_image_mse = _mean_mse(obs, next_obs[intra_perm])
    inter_image_mse = _mean_mse(obs, next_obs[inter_perm])
    temporal_distance_mse = _mean_mse(emb_t, emb_t1)
    intra_distance_mse = _mean_mse(emb_t, emb_t1[intra_perm])
    inter_distance_mse = _mean_mse(emb_t, emb_t1[inter_perm])
    probe_auc = _probe_auc(emb_t, categories, args.seed)

    embeddings_path = Path(args.embeddings_output)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embeddings_path,
        obs=obs,
        next_obs=next_obs,
        emb_t=emb_t,
        emb_t1=emb_t1,
        pred_t1_zero_action=pred_t1,
        categories=np.array(categories),
        prompt_ids=np.array(prompt_ids),
        intra_perm=intra_perm,
        inter_perm=inter_perm,
    )

    report = {
        "status": "complete",
        "target_model": args.model,
        "teacher_checkpoint": str(args.checkpoint),
        "encoding": args.encoding,
        "resolution": args.resolution,
        "layer_path": layer_path,
        "layers": layer_ids,
        "relative_layers": args.relative_layers,
        "n_prompts": len(pairs),
        "n_per_category": args.n_per_category,
        "categories": sorted(set(categories)),
        "temporal_new_tokens": args.temporal_new_tokens,
        "action_mode": "zero_action_256",
        "probe_auc_ovr": probe_auc,
        "coherent_latent_mse_mean": coherent_latent_mse,
        "intra_domain_latent_mse_mean": intra_latent_mse,
        "inter_domain_latent_mse_mean": inter_latent_mse,
        "intra_domain_voe_ratio": float(intra_latent_mse / max(coherent_latent_mse, 1e-12)),
        "inter_domain_voe_ratio": float(inter_latent_mse / max(coherent_latent_mse, 1e-12)),
        "coherent_image_mse_mean": coherent_image_mse,
        "intra_domain_image_mse_mean": intra_image_mse,
        "inter_domain_image_mse_mean": inter_image_mse,
        "intra_domain_image_ratio": float(intra_image_mse / max(coherent_image_mse, 1e-12)),
        "inter_domain_image_ratio": float(inter_image_mse / max(coherent_image_mse, 1e-12)),
        "temporal_embedding_distance_mse": temporal_distance_mse,
        "intra_domain_embedding_distance_mse": intra_distance_mse,
        "inter_domain_embedding_distance_mse": inter_distance_mse,
        "embeddings_output": str(embeddings_path),
        "progress_output": str(progress_path),
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="zai-org/GLM-5.1-FP8")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/evals/cross_arch_glm_5_1_fp8.json")
    parser.add_argument("--embeddings-output", default="artifacts/evals/cross_arch_glm_5_1_fp8_embeddings.npz")
    parser.add_argument("--progress-output")
    parser.add_argument("--layers")
    parser.add_argument("--relative-layers", default="0.2,0.5,0.75")
    parser.add_argument("--encoding", default="token_similarity_v1")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n-per-category", type=int, default=100)
    parser.add_argument("--temporal-new-tokens", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--lewm-device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run_cross_arch_glm(args)


if __name__ == "__main__":
    main()
