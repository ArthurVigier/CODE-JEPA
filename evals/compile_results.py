from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    payload = {
        "json_artifacts": sorted(str(path) for path in runs_dir.rglob("*.json")),
        "png_artifacts": sorted(str(path) for path in runs_dir.rglob("*.png")),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(out)


if __name__ == "__main__":
    main()
