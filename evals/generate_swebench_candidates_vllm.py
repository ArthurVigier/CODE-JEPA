from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.generate_swebench_candidates import (
    SYSTEM_PROMPT,
    _append_jsonl,
    _build_prompt,
    _candidate_record,
    _completed_counts,
)


def _load_swebench_rows(dataset_name: str, split: str, max_instances: int | None, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = [dict(row) for row in load_dataset(dataset_name, split=split)]
    if max_instances is not None:
        rng = random.Random(seed)
        rows = rows.copy()
        rng.shuffle(rows)
        rows = rows[:max_instances]
    return rows


def _format_chat(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{prompt}\n\nPatch:\n"


def _pending_requests(
    rows: list[dict[str, Any]],
    completed: dict[str, int],
    num_candidates: int,
) -> list[tuple[dict[str, Any], str, int]]:
    pending: list[tuple[dict[str, Any], str, int]] = []
    for row in rows:
        instance_id = str(row["instance_id"])
        already = completed.get(instance_id, 0)
        for candidate_id in range(already, num_candidates):
            pending.append((row, instance_id, candidate_id))
    return pending


def generate_candidates_vllm(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = _load_swebench_rows(args.dataset_name, args.split, args.max_instances, args.seed)
    completed = _completed_counts(args.output) if args.resume else {}
    pending = _pending_requests(rows, completed, args.num_candidates)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompts = [_format_chat(tokenizer, _build_prompt(row)) for row, _instance_id, _candidate_id in pending]

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "download_dir": args.download_dir,
        "seed": args.seed,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**{key: value for key, value in llm_kwargs.items() if value is not None})
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    generated = 0
    skipped = sum(max(0, min(args.num_candidates, count)) for count in completed.values())
    chunk_size = max(1, int(args.request_batch_size))
    for start in range(0, len(prompts), chunk_size):
        prompt_chunk = prompts[start : start + chunk_size]
        meta_chunk = pending[start : start + chunk_size]
        outputs = llm.generate(prompt_chunk, sampling, use_tqdm=True)
        if len(outputs) != len(meta_chunk):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(meta_chunk)} prompts")
        for output, (row, instance_id, candidate_id) in zip(outputs, meta_chunk):
            text = output.outputs[0].text if output.outputs else ""
            record = _candidate_record(
                row=row,
                instance_id=instance_id,
                candidate_id=candidate_id,
                generated_text=text,
                args=args,
            )
            record["engine"] = "vllm"
            record["request_batch_size"] = chunk_size
            _append_jsonl(args.output, record)
            generated += 1
            print(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "candidate_id": candidate_id,
                        "generated": generated,
                        "engine": "vllm",
                    }
                ),
                flush=True,
            )

    summary = {
        "status": "complete",
        "engine": "vllm",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model": args.model,
        "output": str(args.output),
        "num_candidates": args.num_candidates,
        "n_instances": len(rows),
        "generated_candidates": generated,
        "skipped_candidates": skipped,
        "request_batch_size": chunk_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "resume": bool(args.resume),
    }
    summary_path = Path(args.summary_output or str(Path(args.output).with_suffix(".vllm.summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--model-name-or-path", default="qwen3-32b-n8-vllm")
    parser.add_argument("--output", default="artifacts/swebench/qwen_candidates_n8.jsonl")
    parser.add_argument("--summary-output")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--max-model-len", type=int, default=14336)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--request-batch-size", type=int, default=64)
    parser.add_argument("--download-dir")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    generate_candidates_vllm(args)


if __name__ == "__main__":
    main()
