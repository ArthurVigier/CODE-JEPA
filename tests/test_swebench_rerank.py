from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.swebench_rerank import _read_jsonl, _write_jsonl, validate_candidates_file


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_validate_candidates_accepts_one_row_per_candidate(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    _write_rows(
        path,
        [
            {
                "instance_id": "repo__project-1",
                "candidate_id": "a",
                "problem_statement": "Fix the parser.",
                "model_patch": "diff --git a/a.py b/a.py\n+pass\n",
            },
            {
                "instance_id": "repo__project-1",
                "candidate_id": "b",
                "problem_statement": "Fix the parser.",
                "model_patch": "diff --git a/a.py b/a.py\n+return 1\n",
            },
        ],
    )

    by_instance, problems = validate_candidates_file(
        path,
        min_candidates=2,
        swebench_dataset_name=None,
        swebench_split="test",
    )

    assert sorted(by_instance) == ["repo__project-1"]
    assert [candidate.candidate_id for candidate in by_instance["repo__project-1"]] == ["a", "b"]
    assert problems["repo__project-1"] == "Fix the parser."


def test_validate_candidates_accepts_nested_candidate_list(tmp_path: Path) -> None:
    path = tmp_path / "nested.jsonl"
    _write_rows(
        path,
        [
            {
                "instance_id": "repo__project-2",
                "problem_statement": "Fix the renderer.",
                "candidates": [
                    {"candidate_id": "0", "patch": "diff --git a/r.py b/r.py\n+ok = True\n"},
                    {"candidate_id": "1", "patch": "diff --git a/r.py b/r.py\n+ok = False\n"},
                ],
            }
        ],
    )

    by_instance, _problems = validate_candidates_file(
        path,
        min_candidates=2,
        swebench_dataset_name=None,
        swebench_split="test",
    )

    patches = [candidate.model_patch for candidate in by_instance["repo__project-2"]]
    assert patches == [
        "diff --git a/r.py b/r.py\n+ok = True\n",
        "diff --git a/r.py b/r.py\n+ok = False\n",
    ]


def test_validate_candidates_rejects_empty_patches_and_singletons(tmp_path: Path) -> None:
    empty_patch = tmp_path / "empty.jsonl"
    _write_rows(
        empty_patch,
        [
            {
                "instance_id": "repo__project-3",
                "problem_statement": "Fix the solver.",
                "model_patch": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="model patch"):
        validate_candidates_file(empty_patch, min_candidates=2, swebench_dataset_name=None, swebench_split="test")

    singleton = tmp_path / "singleton.jsonl"
    _write_rows(
        singleton,
        [
            {
                "instance_id": "repo__project-4",
                "problem_statement": "Fix the solver.",
                "model_patch": "diff --git a/s.py b/s.py\n+ok = True\n",
            }
        ],
    )
    with pytest.raises(ValueError, match="fewer than 2 candidates"):
        validate_candidates_file(singleton, min_candidates=2, swebench_dataset_name=None, swebench_split="test")


def test_jsonl_writer_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "out" / "predictions.jsonl"
    records = [
        {"instance_id": "x", "model_patch": "diff", "model_name_or_path": "m"},
        {"instance_id": "y", "model_patch": "diff2", "model_name_or_path": "m"},
    ]
    _write_jsonl(path, records)
    assert _read_jsonl(path) == records
