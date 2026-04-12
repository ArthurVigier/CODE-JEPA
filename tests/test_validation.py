import inspect
import math

import numpy as np

from activation_views.validation import compute_near_constant_image_rate, compute_probe_auc
from sklearn.linear_model import LogisticRegression


def test_compute_probe_auc_handles_multiclass() -> None:
    rng = np.random.default_rng(7)
    images = rng.normal(size=(30, 512)).astype(np.float32)
    labels = np.repeat(np.arange(5), 6)
    auc = compute_probe_auc(images, labels)
    assert math.isfinite(auc)
    assert 0.0 <= auc <= 1.0


def test_logistic_regression_signature_compat() -> None:
    params = inspect.signature(LogisticRegression).parameters
    assert "solver" in params
    assert "max_iter" in params
    assert "C" in params


def test_near_constant_image_rate() -> None:
    flat = np.ones((2, 3, 64, 64), dtype=np.float32) * 0.5
    structured = flat.copy()
    structured[1, :, 10:20, 10:20] = 1.0
    assert compute_near_constant_image_rate(flat) == 1.0
    assert compute_near_constant_image_rate(structured) == 0.5
