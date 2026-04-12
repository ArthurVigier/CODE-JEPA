from __future__ import annotations

"""Modal Phase 0 extraction entrypoint."""

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
except Exception:  # pragma: no cover - local fallback when modal is not installed
    modal = None


if modal is not None:
    app = modal.App("activation-as-view-phase0")
    volume = modal.Volume.from_name("activation-as-view-artifacts", create_if_missing=True)
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

    @app.function(gpu="H100", timeout=3600, image=image, secrets=[secret], volumes={"/outputs": volume})
    def extract_and_probe(config: dict) -> dict:
        from activation_views.pipeline import run_phase0_live

        os.environ.setdefault("HF_TOKEN", require_env("HF_TOKEN"))
        output_dir = config["output_dir"]
        relative_output = Path(output_dir)
        if relative_output.is_absolute():
            remote_output_dir = relative_output
        else:
            remote_output_dir = Path("/outputs") / relative_output.name
        return run_phase0_live(
            output_dir=remote_output_dir,
            model_name=config["model_name"],
            layers=config["layers"],
            encoding=config.get("encoding", "hsv_pca3"),
            profile=config.get("profile", "debug"),
            hf_token_env=config.get("hf_token_env", "HF_TOKEN"),
            train_resolution=int(config.get("train_resolution", 64)),
            viz_resolution=int(config.get("viz_resolution", 128)),
        )

    @app.local_entrypoint()
    def main(config_path: str = "configs/runs/phase0_modal_qwen32.yaml") -> None:
        import yaml

        config = yaml.safe_load((ROOT / config_path).read_text())
        result = extract_and_probe.remote(config)
        print(result)
        print("Artifacts persisted to Modal Volume 'activation-as-view-artifacts'.")
        print(f"Remote report path: {result['report_path']}")
        print(f"Remote train grid path: {result['grid_train_path']}")
        print(f"Remote viz grid path: {result['grid_viz_path']}")
else:
    def main(config_path: str = "configs/runs/phase0_modal_qwen32.yaml") -> dict:
        import yaml
        from activation_views.pipeline import run_phase0_live

        config = yaml.safe_load((ROOT / config_path).read_text())
        return run_phase0_live(
            output_dir=config["output_dir"],
            model_name=config["model_name"],
            layers=config["layers"],
            encoding=config.get("encoding", "hsv_pca3"),
            profile=config.get("profile", "debug"),
            hf_token_env=config.get("hf_token_env", "HF_TOKEN"),
            train_resolution=int(config.get("train_resolution", 64)),
            viz_resolution=int(config.get("viz_resolution", 128)),
        )


if __name__ == "__main__":
    print(main())
