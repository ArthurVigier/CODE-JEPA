from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from activation_views.train_world_model import train_world_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(train_world_model(config), indent=2))


if __name__ == "__main__":
    main()
