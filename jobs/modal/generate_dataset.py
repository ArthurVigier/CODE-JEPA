from __future__ import annotations

"""Modal Phase 1 dataset generation entrypoint."""

import os
from pathlib import Path
import sys

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)
    for candidate in candidates:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    return Path.cwd()


ROOT = _find_project_root()
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from activation_views.env import load_local_env, require_env

load_local_env()

try:
    import modal
except Exception:  # pragma: no cover
    modal = None


if modal is not None:
    app = modal.App("activation-as-view-phase1")
    artifacts = modal.Volume.from_name("activation-as-view-artifacts", create_if_missing=True)
    image = (
        modal.Image.debian_slim()
        .pip_install(
            "h5py>=3.11",
            "numpy>=1.26",
            "scikit-learn>=1.4",
            "pillow>=10.0",
            "matplotlib>=3.8",
            "pyyaml>=6.0",
            "torch>=2.2",
            "transformers>=4.45",
            "accelerate>=0.33",
        )
        .add_local_python_source("activation_views", copy=True)
    )
    secret = modal.Secret.from_dict({"HF_TOKEN": require_env("HF_TOKEN")})

    @app.function(gpu="H100:2", timeout=14400, image=image, secrets=[secret], volumes={"/outputs": artifacts})
    def generate_triplets(config: dict) -> dict:
        from activation_views.dataset_generation import generate_qwen_triplets_live

        os.environ.setdefault("HF_TOKEN", require_env("HF_TOKEN"))
        output_path = Path(config["output_path"])
        checkpoint_path = Path(config["checkpoint_path"])
        if not output_path.is_absolute():
            output_path = Path("/outputs") / output_path.name
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path("/outputs/checkpoints") / checkpoint_path.name
        return generate_qwen_triplets_live(
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            model_name=config["model_name"],
            layers=config["layers"],
            encoding=config.get("encoding", "hsv_pca3"),
            profile=config.get("profile", "pilot"),
            prompt_limit=int(config.get("prompt_limit", 10)),
            max_new_tokens=int(config.get("max_new_tokens", 2)),
            torch_dtype=config.get("torch_dtype", "bfloat16"),
            hf_token_env=config.get("hf_token_env", "HF_TOKEN"),
            action_pca_sample_size=config.get("action_pca_sample_size"),
        )

    @app.local_entrypoint()
    def main(config_path: str = "configs/runs/phase1_modal_qwen72_pilot.yaml") -> None:
        import yaml

        config = yaml.safe_load((ROOT / config_path).read_text())
        result = generate_triplets.remote(config)
        print(result)
        print("Dataset artifacts persisted to Modal Volume 'activation-as-view-artifacts'.")
else:
    def main(config_path: str = "configs/runs/phase1_modal_qwen72_pilot.yaml") -> dict:
        import yaml
        from activation_views.dataset_generation import generate_qwen_triplets_live

        config = yaml.safe_load((ROOT / config_path).read_text())
        return generate_qwen_triplets_live(**config)


if __name__ == "__main__":
    print(main())
