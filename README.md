# Activation-as-View

Research monorepo for the "Activation-as-View: LeWM x Qwen3" roadmap.

The current experimental teacher is `Qwen/Qwen3-32B`. The strongest Phase 0 representation so far is `token_similarity_v1`, which encodes token-token relational fields from selected transformer layers as image-like observations.

## Layout

- `src/activation_views/`: core contracts, encoding, validation, HDF5 IO
- `jobs/`: launchers for Modal, RunPod, Hugging Face Endpoint
- `configs/`: versioned configs for models, datasets, runs, train, eval
- `third_party/le-wm/`: vendored LeWM fork plus Qwen activation adapters
- `evals/`: standalone evaluation entrypoints
- `paper/`: figures, tables, notes, compiled reports

Large generated datasets, checkpoints, and experiment artifacts are intentionally excluded from git. Keep them under `artifacts/` or external storage.

## Logging

This repo uses local-first experiment logging:

- `TensorBoard` event files for training curves and scalar metrics
- `artifacts/*.json` for run summaries, ablations, kill/go decisions, and eval outputs
- `paper/figures/*.png` and `paper/results_manifest.json` for paper-facing exports

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Main commands

```bash
python -m activation_views.cli phase0 --config configs/runs/phase0_debug.yaml
python -m activation_views.cli phase0-live --config configs/runs/phase0_modal_qwen32.yaml
python -m activation_views.cli validate-hdf5 --path artifacts/qwen3_reasoning_triplets.h5
PYTHONPATH=src python scripts/validate_dataset.py artifacts/qwen3_32b_train_token_similarity.h5
PYTHONPATH=src python scripts/dynamics_baselines.py artifacts/qwen3_32b_train_token_similarity.h5
PYTHONPATH=src python scripts/train_world_model.py --config configs/train/world_model_qwen32_token_similarity.yaml
python third_party/le-wm/train_qwen3_reasoning.py --config third_party/le-wm/config/train/qwen3_reasoning.yaml
python third_party/le-wm/eval_qwen3_voe.py --config third_party/le-wm/config/train/qwen3_reasoning.yaml --checkpoint artifacts/lewm_qwen3_token_similarity/latest.pt --output artifacts/lewm_qwen3_token_similarity/voe.json
python evals/compile_results.py --runs-dir artifacts --output paper/results_manifest.json
# then inspect local training logs with:
# tensorboard --logdir artifacts/tensorboard
```

For Modal Phase 0 runs:

```bash
modal run jobs/modal/extract_phase0.py
```
