from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.generate_swebench_candidates import _build_prompt, _completed_counts, _extract_patch


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
