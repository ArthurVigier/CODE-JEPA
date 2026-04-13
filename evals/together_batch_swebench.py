from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
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
    _candidate_seed,
    _extract_patch,
    _read_jsonl,
)


FINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


def _load_swebench_rows(dataset_name: str, split: str, max_instances: int | None, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = [dict(row) for row in load_dataset(dataset_name, split=split)]
    if max_instances is not None:
        rng = random.Random(seed)
        rows = rows.copy()
        rng.shuffle(rows)
        rows = rows[:max_instances]
    return rows


def _completed_candidate_ids(path: str | Path) -> dict[str, set[str]]:
    completed: dict[str, set[str]] = {}
    for row in _read_jsonl(path):
        instance_id = str(row.get("instance_id", "")).strip()
        candidate_id = row.get("candidate_id")
        if instance_id and candidate_id is not None:
            completed.setdefault(instance_id, set()).add(str(candidate_id))
    return completed


def _messages_for_row(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(row)},
    ]


def _custom_id(row_index: int, candidate_id: int, instance_id: str) -> str:
    stable = hashlib.sha1(instance_id.encode("utf-8")).hexdigest()[:8]
    return f"swb-{row_index:04d}-c{candidate_id:02d}-{stable}"


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_swebench_rows(args.dataset_name, args.split, args.max_instances, args.seed)
    completed = _completed_candidate_ids(args.existing_candidates) if args.resume and args.existing_candidates else {}

    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        instance_id = str(row["instance_id"])
        done = completed.get(instance_id, set())
        for candidate_id in range(args.num_candidates):
            if str(candidate_id) in done:
                continue
            custom_id = _custom_id(row_index, candidate_id, instance_id)
            seed = _candidate_seed(args.seed, instance_id, candidate_id)
            body = {
                "model": args.model,
                "messages": _messages_for_row(row),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": seed,
                "context_length_exceeded_behavior": args.context_length_exceeded_behavior,
            }
            requests.append({"custom_id": custom_id, "body": body})
            manifest.append(
                {
                    "custom_id": custom_id,
                    "instance_id": instance_id,
                    "candidate_id": str(candidate_id),
                    "row_index": row_index,
                    "model": args.model,
                    "seed": seed,
                    "problem_statement": row.get("problem_statement", ""),
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                }
            )

    _write_jsonl(args.batch_input, requests)
    _write_jsonl(args.manifest, manifest)
    summary = {
        "status": "prepared",
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model": args.model,
        "batch_input": str(args.batch_input),
        "manifest": str(args.manifest),
        "n_instances": len(rows),
        "num_candidates": args.num_candidates,
        "pending_requests": len(requests),
        "resume": bool(args.resume),
        "existing_candidates": str(args.existing_candidates) if args.existing_candidates else None,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _together_client():
    try:
        from together import Together
    except ImportError as exc:
        raise RuntimeError("Install the Together SDK first: python -m pip install together") from exc
    return Together(api_key=os.environ.get("TOGETHER_API_KEY"))


def preflight(args: argparse.Namespace) -> None:
    client = _together_client()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Return a minimal empty unified diff patch."},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length_exceeded_behavior=args.context_length_exceeded_behavior,
        seed=args.seed,
    )
    content = _response_text(response)
    print(json.dumps({"status": "ok", "model": args.model, "sample": content[:500]}, indent=2))


def submit(args: argparse.Namespace) -> dict[str, Any]:
    client = _together_client()
    file_resp = client.files.upload(file=args.batch_input, purpose="batch-api", check=False)
    batch = client.batches.create_batch(file_resp.id, endpoint="/v1/chat/completions")
    state = _object_to_dict(batch)
    state["input_upload"] = _object_to_dict(file_resp)
    Path(args.state_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.state_output).write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    return state


def status(args: argparse.Namespace) -> dict[str, Any]:
    client = _together_client()
    batch = client.batches.get_batch(args.batch_id)
    state = _object_to_dict(batch)
    if args.state_output:
        Path(args.state_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.state_output).write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    return state


def wait(args: argparse.Namespace) -> dict[str, Any]:
    deadline = None if args.timeout_seconds <= 0 else time.time() + args.timeout_seconds
    while True:
        state = status(args)
        if str(state.get("status", "")).upper() in FINAL_STATUSES:
            return state
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"batch {args.batch_id} did not finish within {args.timeout_seconds} seconds")
        time.sleep(args.poll_seconds)


def download(args: argparse.Namespace) -> dict[str, Any]:
    client = _together_client()
    batch = client.batches.get_batch(args.batch_id)
    state = _object_to_dict(batch)
    output_file_id = state.get("output_file_id")
    error_file_id = state.get("error_file_id")
    if not output_file_id:
        raise ValueError(f"batch {args.batch_id} has no output_file_id yet; status={state.get('status')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    client.files.retrieve_content(id=output_file_id, output=args.output)
    if error_file_id and args.error_output:
        Path(args.error_output).parent.mkdir(parents=True, exist_ok=True)
        client.files.retrieve_content(id=error_file_id, output=args.error_output)
    state["downloaded_output"] = str(args.output)
    if error_file_id and args.error_output:
        state["downloaded_errors"] = str(args.error_output)
    print(json.dumps(state, indent=2))
    return state


def convert(args: argparse.Namespace) -> dict[str, Any]:
    manifest = {row["custom_id"]: row for row in _read_jsonl(args.manifest)}
    rows = _read_jsonl(args.batch_output)
    seen = set()
    written = 0
    failures = 0
    for row in rows:
        custom_id = row.get("custom_id")
        meta = manifest.get(custom_id)
        if not meta:
            failures += 1
            continue
        try:
            text = _response_text(row)
        except ValueError:
            failures += 1
            continue
        record_args = argparse.Namespace(
            model_name_or_path=args.model_name_or_path,
            model=meta.get("model", args.model),
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            generation_batch_size=1,
        )
        record = _candidate_record(
            row=meta,
            instance_id=str(meta["instance_id"]),
            candidate_id=int(meta["candidate_id"]),
            generated_text=text,
            args=record_args,
        )
        record["engine"] = "together-batch"
        record["custom_id"] = custom_id
        record["raw_batch_response"] = row
        _append_jsonl(args.output, record)
        seen.add(custom_id)
        written += 1

    missing = sorted(set(manifest) - seen)
    summary = {
        "status": "converted",
        "batch_output": str(args.batch_output),
        "manifest": str(args.manifest),
        "output": str(args.output),
        "written_candidates": written,
        "failed_rows": failures,
        "missing_responses": len(missing),
        "missing_examples": missing[:10],
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"cannot convert {type(value)!r} to dict")


def _response_text(value: Any) -> str:
    data = _object_to_dict(value) if not isinstance(value, dict) else value
    body = data
    if isinstance(data.get("response"), dict):
        response = data["response"]
        body = response.get("body", response)
    if isinstance(body.get("body"), dict):
        body = body["body"]
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    if isinstance(body.get("output"), str):
        return body["output"]
    if isinstance(data.get("error"), dict):
        raise ValueError(f"batch row error: {data['error']}")
    raise ValueError(f"could not extract response text from keys={sorted(data)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, submit, download, and convert Together Batch SWE-bench candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    prep.add_argument("--split", default="test")
    prep.add_argument("--model", default="Qwen/Qwen3-32B")
    prep.add_argument("--num-candidates", type=int, default=8)
    prep.add_argument("--max-instances", type=int)
    prep.add_argument("--max-tokens", type=int, default=2048)
    prep.add_argument("--temperature", type=float, default=0.8)
    prep.add_argument("--top-p", type=float, default=0.95)
    prep.add_argument("--seed", type=int, default=7)
    prep.add_argument("--context-length-exceeded-behavior", choices=["truncate", "error"], default="truncate")
    prep.add_argument("--existing-candidates", default="artifacts/swebench/qwen_candidates_n8.jsonl")
    prep.add_argument("--batch-input", default="artifacts/swebench/together_batch_input.jsonl")
    prep.add_argument("--manifest", default="artifacts/swebench/together_batch_manifest.jsonl")
    prep.add_argument("--summary-output", default="artifacts/swebench/together_batch_prepare.json")
    prep.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    prep.set_defaults(func=prepare)

    pf = subparsers.add_parser("preflight")
    pf.add_argument("--model", default="Qwen/Qwen3-32B")
    pf.add_argument("--max-tokens", type=int, default=64)
    pf.add_argument("--temperature", type=float, default=0.1)
    pf.add_argument("--top-p", type=float, default=0.95)
    pf.add_argument("--seed", type=int, default=7)
    pf.add_argument("--context-length-exceeded-behavior", choices=["truncate", "error"], default="truncate")
    pf.set_defaults(func=preflight)

    sub = subparsers.add_parser("submit")
    sub.add_argument("--batch-input", default="artifacts/swebench/together_batch_input.jsonl")
    sub.add_argument("--state-output", default="artifacts/swebench/together_batch_state.json")
    sub.set_defaults(func=submit)

    stat = subparsers.add_parser("status")
    stat.add_argument("--batch-id", required=True)
    stat.add_argument("--state-output")
    stat.set_defaults(func=status)

    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--batch-id", required=True)
    wait_parser.add_argument("--poll-seconds", type=int, default=60)
    wait_parser.add_argument("--timeout-seconds", type=int, default=0)
    wait_parser.add_argument("--state-output")
    wait_parser.set_defaults(func=wait)

    dl = subparsers.add_parser("download")
    dl.add_argument("--batch-id", required=True)
    dl.add_argument("--output", default="artifacts/swebench/together_batch_output.jsonl")
    dl.add_argument("--error-output", default="artifacts/swebench/together_batch_errors.jsonl")
    dl.set_defaults(func=download)

    conv = subparsers.add_parser("convert")
    conv.add_argument("--batch-output", default="artifacts/swebench/together_batch_output.jsonl")
    conv.add_argument("--manifest", default="artifacts/swebench/together_batch_manifest.jsonl")
    conv.add_argument("--output", default="artifacts/swebench/qwen_candidates_n8.jsonl")
    conv.add_argument("--summary-output", default="artifacts/swebench/together_batch_convert.json")
    conv.add_argument("--model", default="Qwen/Qwen3-32B")
    conv.add_argument("--model-name-or-path", default="qwen3-32b-n8-together-batch")
    conv.add_argument("--temperature", type=float, default=0.8)
    conv.add_argument("--top-p", type=float, default=0.95)
    conv.add_argument("--seed", type=int, default=7)
    conv.set_defaults(func=convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
