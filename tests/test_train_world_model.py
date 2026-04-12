import numpy as np

from activation_views.train_world_model import make_split_indices


def test_make_split_indices() -> None:
    train, val = make_split_indices(100, 0.1, 7)
    assert len(train) == 90
    assert len(val) == 10
    assert len(set(train).intersection(set(val))) == 0
