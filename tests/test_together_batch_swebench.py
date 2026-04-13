from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from evals import together_batch_swebench as together_batch


def test_prepare_writes_batch_input_and_manifest(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "problem_statement": "Fix the units bug.",
            "hints_text": "",
        }
    ]
    monkeypatch.setattr(together_batch, "_load_swebench_rows", lambda *_args, **_kwargs: rows)
    existing = tmp_path / "existing.jsonl"
    existing.write_text(json.dumps({"instance_id": "astropy__astropy-12907", "candidate_id": "0"}) + "\n")
    batch_input = tmp_path / "batch.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    summary = tmp_path / "summary.json"

    together_batch.prepare(
        Namespace(
            dataset_name="dataset",
            split="test",
            model="Qwen/Qwen3-32B",
            num_candidates=3,
            max_instances=None,
            max_tokens=2048,
            temperature=0.8,
            top_p=0.95,
            seed=7,
            context_length_exceeded_behavior="truncate",
            existing_candidates=existing,
            batch_input=batch_input,
            manifest=manifest,
            summary_output=summary,
            resume=True,
        )
    )

    requests = [json.loads(line) for line in batch_input.read_text().splitlines()]
    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [row["body"]["model"] for row in requests] == ["Qwen/Qwen3-32B", "Qwen/Qwen3-32B"]
    assert [row["candidate_id"] for row in manifest_rows] == ["1", "2"]
    assert all(len(row["custom_id"]) <= 64 for row in requests)


def test_convert_together_output_to_candidate_jsonl(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    batch_output = tmp_path / "batch_output.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    summary = tmp_path / "summary.json"
    manifest.write_text(
        json.dumps(
            {
                "custom_id": "swb-0000-c00-deadbeef",
                "instance_id": "repo__project-1",
                "candidate_id": "0",
                "model": "Qwen/Qwen3-32B",
                "problem_statement": "Fix it.",
                "repo": "repo/project",
                "base_commit": "abc123",
            }
        )
        + "\n"
    )
    batch_output.write_text(
        json.dumps(
            {
                "custom_id": "swb-0000-c00-deadbeef",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": "Sure\n```diff\ndiff --git a/a.py b/a.py\n+ok = True\n```"
                                }
                            }
                        ]
                    }
                },
            }
        )
        + "\n"
    )

    together_batch.convert(
        Namespace(
            batch_output=batch_output,
            manifest=manifest,
            output=candidates,
            summary_output=summary,
            model="Qwen/Qwen3-32B",
            model_name_or_path="qwen3-32b-n8-together-batch",
            temperature=0.8,
            top_p=0.95,
            seed=7,
        )
    )

    record = json.loads(candidates.read_text().strip())
    assert record["instance_id"] == "repo__project-1"
    assert record["candidate_id"] == "0"
    assert record["engine"] == "together-batch"
    assert record["model_patch"] == "diff --git a/a.py b/a.py\n+ok = True\n"
