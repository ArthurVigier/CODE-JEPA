# Activation-as-View

Research monorepo for the "Activation-as-View: LeWM x Qwen3" roadmap.

The current experimental teacher is `Qwen/Qwen3-32B`. The strongest Phase 0 representation so far is `token_similarity_v1`, which encodes token-token relational fields from selected transformer layers as image-like observations.

## Results So Far

Current status: Phase 0 and the first real LeWM training pass are positive on `Qwen/Qwen3-32B` with `token_similarity_v1`. The result is not yet a SWE-bench or GLM generalization claim; it is a strong falsification pass showing that the activation-view dataset is learnable by the vendored LeWM pipeline and produces a large VoE signal.

Published Hugging Face artifacts:

- Dataset: [`Artvv/qwen3-32b-token-similarity-activation-triplets`](https://huggingface.co/datasets/Artvv/qwen3-32b-token-similarity-activation-triplets)
- LeWM model: [`Artvv/lewm-qwen3-32b-token-similarity`](https://huggingface.co/Artvv/lewm-qwen3-32b-token-similarity)

Main run:

- Teacher: `Qwen/Qwen3-32B`
- Encoding: `token_similarity_v1`
- Layers: `[10, 20, 30]`
- Dataset: `50K` triplets, observations `[3, 64, 64]`, actions `[256]`
- Real LeWM checkpoint: `artifacts/lewm_qwen3_token_similarity/latest.pt`, mirrored on Hugging Face as `checkpoints/latest.pt`

Phase 0 visual/probe result:

- Probe AUC: `~0.996-1.000` across successful 500-prompt runs
- Best working representation so far: `token_similarity_v1`
- Earlier thermal/flow attempts were useful ablations but visually less reliable than token similarity.

Real LeWM 50K result:

- Epoch 20 latent improvement ratio: `6.125`
- Probe AUC after LeWM encoding: `0.9994`
- Calibrated VoE ratio, temporal coherent vs inter-domain incoherent: `71.43x`
- Calibrated VoE ratio, temporal coherent vs intra-domain non-transition: `59.90x`
- Interpretation: the model learned strong local trajectory compatibility. The intra-domain non-transition score is also high, so the current result is evidence for temporal activation dynamics, not yet evidence for broad semantic invariance.

## Artifact Names

There are two different training paths in this repo. They are intentionally named differently:

- `artifacts/world_model_qwen32_token_similarity/` is the early proxy world-model baseline. It is useful only as a sanity check and is not the main LeWM result.
- `artifacts/lewm_qwen3_token_similarity/` is the real vendored LeWM training run on the 50K Qwen3-32B activation triplets.

Do not treat root-level `latest.pt` files as canonical. The canonical trained LeWM checkpoint is `artifacts/lewm_qwen3_token_similarity/latest.pt`, mirrored on Hugging Face as `checkpoints/latest.pt`.

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
python evals/generate_swebench_candidates.py --dataset-name princeton-nlp/SWE-bench_Verified --split test --model Qwen/Qwen3-32B --num-candidates 8 --output artifacts/swebench/qwen_candidates_n8.jsonl
python evals/swebench_rerank.py --candidates artifacts/swebench/qwen_candidates_n8.jsonl --checkpoint artifacts/lewm_qwen3_token_similarity/latest.pt --config third_party/le-wm/config/train/qwen3_reasoning.yaml --output artifacts/evals/swebench_reranked_predictions.jsonl
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --split test --predictions_path artifacts/evals/swebench_reranked_predictions.jsonl --max_workers 8 --run_id code_jepa_lewm_qwen32_rerank
python evals/compile_results.py --runs-dir artifacts --output paper/results_manifest.json
# then inspect local training logs with:
# tensorboard --logdir artifacts/tensorboard
```

For Modal Phase 0 runs:

```bash
modal run jobs/modal/extract_phase0.py
```
