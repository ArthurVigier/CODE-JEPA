#!/usr/bin/env bash
set -euo pipefail
python third_party/le-wm/train_qwen3_reasoning.py --config third_party/le-wm/config/train/qwen3_reasoning.yaml
