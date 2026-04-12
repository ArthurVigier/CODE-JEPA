#!/usr/bin/env bash
set -euo pipefail
python third_party/le-wm/scripts/train.py --config third_party/le-wm/configs/train/qwen3_reasoning.yaml
