from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .hdf5_io import validate_hdf5
from .pipeline import build_demo_triplets, run_phase0, run_phase0_live

app = typer.Typer(no_args_is_help=True)


def _read_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


@app.command()
def phase0(config: str = typer.Option(..., "--config")) -> None:
    cfg = _read_config(config)
    result = run_phase0(
        output_dir=cfg["output_dir"],
        layers=cfg["layers"],
        encoding=cfg.get("encoding", "hsv_pca3"),
        profile=cfg.get("profile", "debug"),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command(name="phase0-live")
def phase0_live(config: str = typer.Option(..., "--config")) -> None:
    cfg = _read_config(config)
    result = run_phase0_live(
        output_dir=cfg["output_dir"],
        model_name=cfg["model_name"],
        layers=cfg["layers"],
        encoding=cfg.get("encoding", "hsv_pca3"),
        profile=cfg.get("profile", "debug"),
        hf_token_env=cfg.get("hf_token_env", "HF_TOKEN"),
        train_resolution=cfg.get("train_resolution", 64),
        viz_resolution=cfg.get("viz_resolution", 128),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def build_triplets(config: str = typer.Option(..., "--config")) -> None:
    cfg = _read_config(config)
    output = build_demo_triplets(cfg["output_path"], profile=cfg.get("profile", "debug"))
    typer.echo(output)


@app.command(name="validate-hdf5")
def validate_hdf5_cmd(path: str = typer.Option(..., "--path")) -> None:
    typer.echo(json.dumps(validate_hdf5(path), indent=2))


if __name__ == "__main__":
    app()
