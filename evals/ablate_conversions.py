from __future__ import annotations

import json
from pathlib import Path

import yaml


def main(config_path: str = "configs/evals/ablation.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    result = {"status": "stub", "variants": cfg["variants"], "metrics": cfg["metrics"]}
    out = Path("artifacts/evals/ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(out)


if __name__ == "__main__":
    main()
