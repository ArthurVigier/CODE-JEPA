from pathlib import Path

import numpy as np

from activation_views.contracts import ObservationImage
from activation_views.validation import save_image_grid


def test_save_image_grid_stratifies_categories(tmp_path: Path) -> None:
    observations = []
    for category_idx in range(5):
        for item_idx in range(5):
            observations.append(
                ObservationImage(
                    image=np.ones((3, 64, 64), dtype=np.float32) * (category_idx + 1) / 5,
                    encoding="hsv_pca3",
                    model_name="model",
                    layers=[10],
                    metadata={"category": f"cat-{category_idx}", "item": item_idx},
                )
            )
    output = tmp_path / "grid.png"
    save_image_grid(observations, output, max_images=10)
    assert output.exists()
    assert output.stat().st_size > 0
