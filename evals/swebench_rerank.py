from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out = Path("artifacts/evals/swebench_rerank.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "stub", "baseline": "majority_vote", "candidate_count": 8}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
