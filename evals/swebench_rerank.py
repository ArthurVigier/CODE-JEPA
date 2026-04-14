from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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


@dataclass(slots=True)
class SwebenchCandidate:
    instance_id: str
    candidate_id: str
    model_patch: str
    problem_statement: str | None
    metadata: dict[str, Any]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad JSONL at {path}:{line_no}") from exc
    return records


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.flush()


def _candidate_patch(record: dict[str, Any]) -> str:
    for key in ("model_patch", "patch", "prediction", "diff"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"candidate missing model patch field: keys={sorted(record)}")


def _problem_text(record: dict[str, Any]) -> str | None:
    for key in ("problem_statement", "issue", "text", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _candidate_id(record: dict[str, Any], fallback: int) -> str:
    for key in ("candidate_id", "sample_id", "completion_id", "idx"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return str(fallback)


def _expand_candidate_records(records: list[dict[str, Any]]) -> list[SwebenchCandidate]:
    candidates: list[SwebenchCandidate] = []
    for row_idx, record in enumerate(records):
        instance_id = str(record.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError(f"candidate row {row_idx} is missing instance_id")
        problem = _problem_text(record)
        nested = record.get("candidates")
        if isinstance(nested, list):
            for nested_idx, item in enumerate(nested):
                if isinstance(item, str):
                    child = {"model_patch": item, "candidate_id": nested_idx}
                elif isinstance(item, dict):
                    child = {**record, **item}
                else:
                    raise ValueError(f"candidate row {row_idx} has unsupported nested candidate type {type(item)!r}")
                candidates.append(
                    SwebenchCandidate(
                        instance_id=instance_id,
                        candidate_id=_candidate_id(child, nested_idx),
                        model_patch=_candidate_patch(child),
                        problem_statement=_problem_text(child) or problem,
                        metadata={k: v for k, v in child.items() if k not in {"model_patch", "patch", "prediction", "diff", "problem_statement", "issue", "text", "prompt"}},
                    )
                )
            continue
        candidates.append(
            SwebenchCandidate(
                instance_id=instance_id,
                candidate_id=_candidate_id(record, row_idx),
                model_patch=_candidate_patch(record),
                problem_statement=problem,
                metadata={k: v for k, v in record.items() if k not in {"model_patch", "patch", "prediction", "diff", "problem_statement", "issue", "text", "prompt"}},
            )
        )
    return candidates


def _group_candidates(candidates: list[SwebenchCandidate], min_candidates: int) -> dict[str, list[SwebenchCandidate]]:
    by_instance: dict[str, list[SwebenchCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_instance[candidate.instance_id].append(candidate)
    too_small = {instance_id: len(group) for instance_id, group in by_instance.items() if len(group) < min_candidates}
    if too_small:
        sample = dict(list(too_small.items())[:5])
        raise ValueError(f"instances with fewer than {min_candidates} candidates: {sample}")
    return dict(by_instance)


def _load_swebench_problems(dataset_name: str, split: str, instance_ids: set[str]) -> dict[str, str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Some candidates are missing problem_statement. Install `datasets` or include problem text in the candidates JSONL."
        ) from exc
    dataset = load_dataset(dataset_name, split=split)
    problems: dict[str, str] = {}
    for row in dataset:
        instance_id = str(row.get("instance_id", ""))
        if instance_id in instance_ids:
            problem = row.get("problem_statement")
            if isinstance(problem, str) and problem.strip():
                problems[instance_id] = problem
        if len(problems) == len(instance_ids):
            break
    missing = sorted(instance_ids - set(problems))
    if missing:
        raise ValueError(f"could not resolve problem_statement for {len(missing)} instances; examples={missing[:5]}")
    return problems


def validate_candidates_file(
    path: str | Path,
    min_candidates: int,
    swebench_dataset_name: str | None,
    swebench_split: str,
) -> tuple[dict[str, list[SwebenchCandidate]], dict[str, str]]:
    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"empty candidates file: {path}")
    candidates = _expand_candidate_records(records)
    by_instance = _group_candidates(candidates, min_candidates=min_candidates)
    problems = {
        instance_id: group[0].problem_statement
        for instance_id, group in by_instance.items()
        if group[0].problem_statement
    }
    missing = set(by_instance) - set(problems)
    if missing:
        if not swebench_dataset_name:
            raise ValueError(f"missing problem_statement for {len(missing)} instances; provide --swebench-dataset-name or include problem text")
        problems.update(_load_swebench_problems(swebench_dataset_name, swebench_split, missing))
    return by_instance, problems


def _load_lewm(checkpoint_path: str | Path, cfg: dict[str, Any], device: torch.device):
    from train_qwen3_reasoning import build_lewm_model

    model = build_lewm_model(cfg, obs_channels=3, action_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _encode_images(model, images: list[np.ndarray], device: torch.device, batch_size: int = 64) -> np.ndarray:
    embeddings = []
    zero_action = None
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = np.stack(images[start : start + batch_size], axis=0).astype(np.float32)
            pixels = torch.from_numpy(batch[:, None]).to(device)
            if zero_action is None or zero_action.shape[0] != pixels.shape[0]:
                zero_action = torch.zeros((pixels.shape[0], 1, 256), dtype=torch.float32, device=device)
            out = model.encode({"pixels": pixels, "action": zero_action[: pixels.shape[0]]})
            embeddings.append(out["emb"][:, 0].detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=1)


def _load_processed_predictions(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows = _read_jsonl(path)
    processed = {str(row["instance_id"]) for row in rows if row.get("instance_id")}
    return rows, processed


def _write_progress(
    path: Path,
    *,
    processed: int,
    total: int,
    instance_id: str | None,
    started_at: float,
    status: str,
) -> None:
    elapsed = time.time() - started_at
    remaining = total - processed
    rate = processed / elapsed if elapsed > 0 else 0.0
    progress = {
        "status": status,
        "processed_instances": processed,
        "total_instances": total,
        "remaining_instances": remaining,
        "last_instance_id": instance_id,
        "elapsed_seconds": elapsed,
        "instances_per_hour": rate * 3600.0,
        "eta_seconds": remaining / rate if rate > 0 and remaining > 0 else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2))


def rerank_swebench(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"missing candidates file: {candidates_path}. "
            "Generate N candidate patches per SWE-bench instance before reranking."
        )
    by_instance, problems = validate_candidates_file(
        candidates_path,
        min_candidates=int(args.min_candidates),
        swebench_dataset_name=args.swebench_dataset_name,
        swebench_split=args.swebench_split,
    )
    n_candidates = sum(len(group) for group in by_instance.values())
    if args.validate_only:
        summary = {
            "status": "validated",
            "candidates_path": str(candidates_path),
            "n_instances": len(by_instance),
            "n_candidates": n_candidates,
            "min_candidates": int(args.min_candidates),
            "swebench_dataset_name": args.swebench_dataset_name,
            "swebench_split": args.swebench_split,
        }
        print(json.dumps(summary, indent=2))
        return summary

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing LeWM checkpoint: {checkpoint_path}")

    from activation_views.encoding import snapshot_to_image
    from activation_views.extractors import TransformerActivationExtractor

    model_name = args.teacher_model or cfg.get("teacher_model", "Qwen/Qwen3-32B")
    layer_value = args.layers or cfg.get("layers", "10,20,30")
    layers = [int(x) for x in layer_value.split(",")] if isinstance(layer_value, str) else [int(x) for x in layer_value]
    encoding = args.encoding or cfg.get("encoding", "token_similarity_v1")
    resolution = int(args.resolution or cfg.get("resolution", 64))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lewm = _load_lewm(checkpoint_path, cfg, device)
    extractor = TransformerActivationExtractor(model_name=model_name, layer_ids=layers)

    output_path = Path(args.output)
    score_path = Path(args.scores_output or str(output_path.with_suffix(".scores.jsonl")))
    summary_path = Path(args.summary_output or str(output_path.with_suffix(".summary.json")))
    progress_path = Path(args.progress_output or str(output_path.with_suffix(".progress.json")))

    if args.resume:
        predictions, processed_instances = _load_processed_predictions(output_path)
        score_rows = _read_jsonl(score_path) if score_path.exists() else []
    else:
        predictions = []
        score_rows = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")
        score_path.write_text("")
        processed_instances = set()

    started_at = time.time()
    total_instances = len(by_instance)
    _write_progress(
        progress_path,
        processed=len(processed_instances),
        total=total_instances,
        instance_id=None,
        started_at=started_at,
        status="running",
    )
    try:
        for instance_id, group in by_instance.items():
            if instance_id in processed_instances:
                continue
            problem = problems[instance_id]
            problem_snapshot = extractor.extract_snapshot(
                prompt_id=f"{instance_id}:problem",
                source="swebench_verified",
                category="problem",
                text=problem,
            )
            problem_img = snapshot_to_image(problem_snapshot, encoding=encoding, resolution=resolution).image
            candidate_images = []
            for idx, record in enumerate(group):
                text = (
                    "Problem:\n"
                    f"{problem}\n\n"
                    "Candidate patch:\n"
                    f"{record.model_patch}"
                )
                snapshot = extractor.extract_snapshot(
                    prompt_id=f"{instance_id}:candidate:{idx}",
                    source="swebench_verified",
                    category="candidate_patch",
                    text=text,
                )
                candidate_images.append(snapshot_to_image(snapshot, encoding=encoding, resolution=resolution).image)

            problem_emb = _encode_images(lewm, [problem_img], device=device)
            candidate_emb = _encode_images(lewm, candidate_images, device=device)
            scores = _cosine(np.repeat(problem_emb, candidate_emb.shape[0], axis=0), candidate_emb)
            best_idx = int(np.argmax(scores))
            best = group[best_idx]
            prediction = {
                "instance_id": instance_id,
                "model_patch": best.model_patch,
                "model_name_or_path": args.model_name_or_path,
            }
            predictions.append(prediction)
            instance_score_rows = []
            for idx, (record, score) in enumerate(zip(group, scores)):
                instance_score_rows.append(
                    {
                        "instance_id": instance_id,
                        "candidate_id": record.candidate_id,
                        "score": float(score),
                        "selected": idx == best_idx,
                    }
                )
            score_rows.extend(instance_score_rows)
            processed_instances.add(instance_id)
            _append_jsonl(output_path, [prediction])
            _append_jsonl(score_path, instance_score_rows)
            _write_progress(
                progress_path,
                processed=len(processed_instances),
                total=total_instances,
                instance_id=instance_id,
                started_at=started_at,
                status="running",
            )
    finally:
        extractor.close()

    summary = {
        "status": "complete",
        "candidates_path": str(candidates_path),
        "output": str(output_path),
        "scores_output": str(score_path),
        "n_instances": len(predictions),
        "n_candidates": n_candidates,
        "min_candidates": int(args.min_candidates),
        "teacher_model": model_name,
        "layers": layers,
        "encoding": encoding,
        "resolution": resolution,
        "checkpoint": str(checkpoint_path),
        "model_name_or_path": args.model_name_or_path,
        "swebench_dataset_name": args.swebench_dataset_name,
        "swebench_split": args.swebench_split,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_progress(
        progress_path,
        processed=len(processed_instances),
        total=total_instances,
        instance_id=None,
        started_at=started_at,
        status="complete",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--progress-output")
    parser.add_argument("--teacher-model")
    parser.add_argument("--swebench-dataset-name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--swebench-split", default="test")
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip instance_ids already present in --output and append new results.")
    parser.add_argument("--layers")
    parser.add_argument("--encoding")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--model-name-or-path", default="code-jepa-lewm-qwen32-rerank")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rerank_swebench(cfg, args)


if __name__ == "__main__":
    main()
