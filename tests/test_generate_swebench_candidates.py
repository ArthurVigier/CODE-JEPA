from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.generate_swebench_candidates import _build_prompt, _candidate_record, _candidate_seed, _completed_counts, _extract_patch


def test_extract_patch_prefers_diff_inside_fence() -> None:
    text = "Here is the fix:\n```diff\ndiff --git a/a.py b/a.py\n+ok = True\n```\nthanks"
    assert _extract_patch(text) == "diff --git a/a.py b/a.py\n+ok = True\n"


def test_extract_patch_drops_preamble_before_diff_marker() -> None:
    text = "Sure.\n\ndiff --git a/b.py b/b.py\n-print(1)\n+print(2)"
    assert _extract_patch(text) == "diff --git a/b.py b/b.py\n-print(1)\n+print(2)\n"


def test_completed_counts_supports_resume(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    with path.open("w") as handle:
        handle.write(json.dumps({"instance_id": "a", "candidate_id": "0"}) + "\n")
        handle.write(json.dumps({"instance_id": "a", "candidate_id": "1"}) + "\n")
        handle.write(json.dumps({"instance_id": "b", "candidate_id": "0"}) + "\n")
    assert _completed_counts(path) == {"a": 2, "b": 1}


def test_candidate_record_tracks_batch_size_and_stable_seed() -> None:
    args = Namespace(
        model_name_or_path="qwen3-32b-n8",
        model="Qwen/Qwen3-32B",
        temperature=0.8,
        top_p=0.95,
        seed=7,
        generation_batch_size=4,
    )
    row = {"problem_statement": "Fix it.", "repo": "repo/project", "base_commit": "abc123"}
    record = _candidate_record(
        row=row,
        instance_id="repo__project-1",
        candidate_id=3,
        generated_text="diff --git a/a.py b/a.py\n+ok = True",
        args=args,
    )
    assert record["candidate_id"] == "3"
    assert record["model_patch"].startswith("diff --git")
    assert record["generation_batch_size"] == 4
    assert record["seed"] == _candidate_seed(7, "repo__project-1", 3)


def test_build_prompt_contains_swebench_context() -> None:
    prompt = _build_prompt(
        {
            "instance_id": "repo__project-1",
            "repo": "repo/project",
            "base_commit": "abc123",
            "problem_statement": "The parser crashes.",
            "hints_text": "Look at parser.py",
        }
    )
    assert "repo__project-1" in prompt
    assert "repo/project" in prompt
    assert "abc123" in prompt
    assert "The parser crashes." in prompt
    assert "Return only the patch" in prompt
