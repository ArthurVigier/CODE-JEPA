from __future__ import annotations

import argparse
import json

from activation_views.dynamics_baselines import compute_dynamics_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output-path", default="artifacts/dataset_validation/dynamics_baselines.json")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    report = compute_dynamics_baselines(args.path, args.output_path, args.sample_limit, args.batch_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
