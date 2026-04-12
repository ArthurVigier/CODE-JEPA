from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalMetricLogger:
    def __init__(self, tensorboard_dir: str | Path):
        self.tensorboard_dir = Path(tensorboard_dir)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        except Exception:
            self.writer = None

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def write_json_summary(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
