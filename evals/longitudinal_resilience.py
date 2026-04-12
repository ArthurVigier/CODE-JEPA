from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out = Path("artifacts/evals/longitudinal_resilience.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "stub", "window_tokens": 512, "trajectories": [5000, 15000, 30000]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
