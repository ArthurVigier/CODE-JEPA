import numpy as np

from activation_views.dataset_generation import _select_stratified_prompts, fit_action_projector_from_model


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.weight = self

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return np.random.default_rng(3).normal(size=(512, 300)).astype(np.float32)


class _FakeModelInner:
    def __init__(self) -> None:
        self.embed_tokens = _FakeEmbeddings()


class _FakeModel:
    def __init__(self) -> None:
        self.model = _FakeModelInner()


def test_select_stratified_prompts_has_multiple_categories() -> None:
    prompts = _select_stratified_prompts(10)
    assert len(prompts) == 10
    assert len({prompt.category for prompt in prompts}) > 1


def test_fit_action_projector_from_model() -> None:
    projector = fit_action_projector_from_model(_FakeModel(), sample_size=300)
    action = projector.transform(np.ones((300,), dtype=np.float32))
    assert action.vector.shape == (256,)
