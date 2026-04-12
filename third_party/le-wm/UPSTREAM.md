# LeWM Fork Notes

This directory vendors the upstream `lucas-maes/le-wm` codebase and keeps the
original `train.py`, `jepa.py`, `module.py`, `eval.py`, and config layout intact.

Activation-as-View additions live beside the upstream files:

- `qwen3_dataset.py`: canonical HDF5 triplets as LeWM pixel/action sequences.
- `train_qwen3_reasoning.py`: LeWM training loop for Qwen activation images,
  reusing upstream `JEPA`, `ARPredictor`, `Embedder`, `MLP`, and `SIGReg`.
- `eval_qwen3_voe.py`: latent VoE evaluation for coherent vs mismatched next states.
- `config/train/qwen3_reasoning.yaml`: default 50K Qwen3-32B token-similarity run.

The Qwen path intentionally avoids WandB and logs to TensorBoard plus JSON
artifacts under `artifacts/`.
