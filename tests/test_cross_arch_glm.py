from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.cross_arch_glm import _category_permutation, _mean_mse, _relative_layers


def test_relative_layers_match_glm_5_1_defaults() -> None:
    assert _relative_layers(78, "0.2,0.5,0.75") == [16, 39, 58]


def test_category_permutation_separates_same_and_different_domains() -> None:
    rng = np.random.default_rng(7)
    categories = ["code", "code", "math", "math", "sql", "sql"]
    same = _category_permutation(categories, same_domain=True, rng=rng)
    assert all(categories[i] == categories[j] and i != j for i, j in enumerate(same))

    rng = np.random.default_rng(7)
    different = _category_permutation(categories, same_domain=False, rng=rng)
    assert all(categories[i] != categories[j] for i, j in enumerate(different))


def test_mean_mse() -> None:
    a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    b = np.array([[0.0, 2.0], [4.0, 3.0]], dtype=np.float32)
    assert _mean_mse(a, b) == 1.25
