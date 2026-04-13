from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch


SYSTEM_PROMPT = """You are an expert software engineer solving SWE-bench tasks.
Return only a unified git diff patch. Do not include explanations, markdown, or tests output."""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad JSONL at {path}:{line_no}") from exc
    return records


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def _completed_counts(path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_jsonl(path):
        instance_id = str(row.get("instance_id", "")).strip()
        if instance_id:
            counts[instance_id] = counts.get(instance_id, 0) + 1
    return counts


def _extract_patch(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    marker = text.find("diff --git ")
    if marker >= 0:
        return text[marker:].strip() + "\n"
    return text.strip() + "\n"


def _build_prompt(row: dict[str, Any]) -> str:
    repo = row.get("repo", "")
    base_commit = row.get("base_commit", "")
    instance_id = row.get("instance_id", "")
    problem = row.get("problem_statement", "")
    hints = row.get("hints_text", "")
    created_at = row.get("created_at", "")
    return (
        "Solve this SWE-bench issue by producing a minimal unified git diff patch.\n\n"
        f"Instance: {instance_id}\n"
        f"Repository: {repo}\n"
        f"Base commit: {base_commit}\n"
        f"Created at: {created_at}\n\n"
        "Problem statement:\n"
        f"{problem}\n\n"
        "Hints:\n"
        f"{hints}\n\n"
        "Return only the patch, starting with diff --git when possible."
    )


def _format_chat(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{prompt}\n\nPatch:\n"


def _load_swebench_rows(dataset_name: str, split: str, max_instances: int | None, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = list(load_dataset(dataset_name, split=split))
    rows = [dict(row) for row in rows]
    if max_instances is not None:
        rng = random.Random(seed)
        rows = rows.copy()
        rng.shuffle(rows)
        rows = rows[:max_instances]
    return rows


def _candidate_seed(base_seed: int, instance_id: str, candidate_id: int) -> int:
    return base_seed + candidate_id + 1009 * len(instance_id)


def _candidate_record(
    *,
    row: dict[str, Any],
    instance_id: str,
    candidate_id: int,
    generated_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "candidate_id": str(candidate_id),
        "model_patch": _extract_patch(generated_text),
        "raw_generation": generated_text,
        "problem_statement": row.get("problem_statement", ""),
        "repo": row.get("repo", ""),
        "base_commit": row.get("base_commit", ""),
        "model_name_or_path": args.model_name_or_path,
        "teacher_model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": _candidate_seed(args.seed, instance_id, candidate_id),
        "generation_batch_size": args.generation_batch_size,
    }


def generate_candidates(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_swebench_rows(args.dataset_name, args.split, args.max_instances, args.seed)
    completed = _completed_counts(args.output) if args.resume else {}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    generated = 0
    skipped_instances = 0
    generation_batch_size = max(1, int(args.generation_batch_size))
    for row in rows:
        instance_id = str(row["instance_id"])
        already = completed.get(instance_id, 0)
        if already >= args.num_candidates:
            skipped_instances += 1
            continue
        prompt = _format_chat(tokenizer, _build_prompt(row))
        for start_candidate in range(already, args.num_candidates, generation_batch_size):
            candidate_ids = list(range(start_candidate, min(args.num_candidates, start_candidate + generation_batch_size)))
            prompts = [prompt] * len(candidate_ids)
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
                padding=True,
            )
            input_device = next(model.parameters()).device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            chunk_seed = _candidate_seed(args.seed, instance_id, start_candidate)
            torch.manual_seed(chunk_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(chunk_seed)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for row_idx, candidate_id in enumerate(candidate_ids):
                generated_text = tokenizer.decode(output_ids[row_idx, prompt_len:], skip_special_tokens=True)
                _append_jsonl(
                    args.output,
                    _candidate_record(
                        row=row,
                        instance_id=instance_id,
                        candidate_id=candidate_id,
                        generated_text=generated_text,
                        args=args,
                    ),
                )
                generated += 1
                print(
                    json.dumps(
                        {
                            "instance_id": instance_id,
                            "candidate_id": candidate_id,
                            "generated": generated,
                            "generation_batch_size": len(candidate_ids),
                        }
                    ),
                    flush=True,
                )

    summary = {
        "status": "complete",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model": args.model,
        "output": str(args.output),
        "num_candidates": args.num_candidates,
        "n_instances": len(rows),
        "generated_candidates": generated,
        "skipped_instances": skipped_instances,
        "generation_batch_size": generation_batch_size,
        "resume": bool(args.resume),
    }
    summary_path = Path(args.summary_output or str(Path(args.output).with_suffix(".summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--model-name-or-path", default="qwen3-32b-n8")
    parser.add_argument("--output", default="artifacts/swebench/qwen_candidates_n8.jsonl")
    parser.add_argument("--summary-output")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    generate_candidates(args)


if __name__ == "__main__":
    main()
