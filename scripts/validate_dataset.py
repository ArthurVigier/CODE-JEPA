from __future__ import annotations

import argparse
import json

from activation_views.dataset_validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output-dir", default="artifacts/dataset_validation")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.path, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
