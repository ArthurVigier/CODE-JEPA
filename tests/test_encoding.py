import numpy as np

from activation_views.contracts import ActivationSnapshot
from activation_views.encoding import ActionProjector, snapshot_to_image


def test_snapshot_to_image_shape() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={10: np.random.randn(32, 64).astype(np.float32)},
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10],
        seq_len=32,
    )
    obs = snapshot_to_image(snapshot, encoding="hsv_pca3", resolution=64)
    assert obs.image.shape == (3, 64, 64)
    assert obs.image.dtype == np.float32
    assert float(obs.image.mean()) > 0.1


def test_snapshot_to_image_svd_dual_resolution() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={
            10: np.random.randn(32, 96).astype(np.float32),
            20: np.random.randn(32, 96).astype(np.float32),
            30: np.random.randn(32, 96).astype(np.float32),
        },
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10, 20, 30],
        seq_len=32,
    )
    obs64 = snapshot_to_image(snapshot, encoding="residual_svd64_v1", resolution=64)
    obs128 = snapshot_to_image(snapshot, encoding="residual_svd128_v1", resolution=128)
    assert obs64.image.shape == (3, 64, 64)
    assert obs128.image.shape == (3, 128, 128)
    assert not np.allclose(obs64.image[0], obs64.image[1])


def test_snapshot_to_image_thermal_encoding() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={
            10: np.random.randn(48, 96).astype(np.float32),
            20: np.random.randn(48, 96).astype(np.float32),
            30: np.random.randn(48, 96).astype(np.float32),
        },
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10, 20, 30],
        seq_len=48,
    )
    obs = snapshot_to_image(snapshot, encoding="thermal_svd64_v1", resolution=64)
    assert obs.image.shape == (3, 64, 64)
    assert obs.image.dtype == np.float32
    assert 0.0 <= float(obs.image.min()) <= float(obs.image.max()) <= 1.0


def test_snapshot_to_image_thermal_dynamics_encoding() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={
            10: np.random.randn(48, 96).astype(np.float32),
            20: np.random.randn(48, 96).astype(np.float32),
            30: np.random.randn(48, 96).astype(np.float32),
        },
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10, 20, 30],
        seq_len=48,
    )
    obs = snapshot_to_image(snapshot, encoding="thermal_dynamics_v1", resolution=64)
    assert obs.image.shape == (3, 64, 64)
    assert obs.image.dtype == np.float32
    assert not np.allclose(obs.image[0], obs.image[1])
    assert not np.allclose(obs.image[1], obs.image[2])


def test_snapshot_to_image_token_similarity_encoding() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={
            10: np.random.randn(48, 96).astype(np.float32),
            20: np.random.randn(48, 96).astype(np.float32),
            30: np.random.randn(48, 96).astype(np.float32),
        },
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10, 20, 30],
        seq_len=48,
    )
    obs = snapshot_to_image(snapshot, encoding="token_similarity_v1", resolution=64)
    assert obs.image.shape == (3, 64, 64)
    assert obs.image.dtype == np.float32
    assert 0.0 <= float(obs.image.min()) <= float(obs.image.max()) <= 1.0


def test_snapshot_to_image_token_similarity_depth_encoding() -> None:
    residuals = {layer: np.random.randn(48, 96).astype(np.float32) for layer in [4, 8, 12, 16, 20, 24, 28, 32]}
    snapshot = ActivationSnapshot(
        residuals_by_layer=residuals,
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[4, 8, 12, 16, 20, 24, 28, 32],
        seq_len=48,
    )
    obs = snapshot_to_image(snapshot, encoding="token_similarity_depth_v1", resolution=64)
    assert obs.image.shape == (8, 64, 64)
    assert obs.image.dtype == np.float32
    assert 0.0 <= float(obs.image.min()) <= float(obs.image.max()) <= 1.0


def test_snapshot_to_image_flow_encoding() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={
            10: np.random.randn(48, 96).astype(np.float32),
            20: np.random.randn(48, 96).astype(np.float32),
            30: np.random.randn(48, 96).astype(np.float32),
        },
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10, 20, 30],
        seq_len=48,
    )
    obs = snapshot_to_image(snapshot, encoding="flow_svd_v1", resolution=64)
    assert obs.image.shape == (3, 64, 64)
    assert obs.image.dtype == np.float32
    assert 0.0 <= float(obs.image.min()) <= float(obs.image.max()) <= 1.0
    assert not np.allclose(obs.image[0], obs.image[1])


def test_snapshot_to_image_particle_flow_encoding() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={
            10: np.random.randn(64, 128).astype(np.float32),
            20: np.random.randn(64, 128).astype(np.float32),
            30: np.random.randn(64, 128).astype(np.float32),
        },
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10, 20, 30],
        seq_len=64,
    )
    obs = snapshot_to_image(snapshot, encoding="flow_particles_v1", resolution=64)
    assert obs.image.shape == (3, 64, 64)
    assert obs.image.dtype == np.float32
    assert 0.0 <= float(obs.image.min()) <= float(obs.image.max()) <= 1.0
    assert float(obs.image.std()) > 0.01


def test_action_projector_shape() -> None:
    embeddings = np.random.randn(300, 300).astype(np.float32)
    projector = ActionProjector.fit_from_embeddings(embeddings, n_components=256)
    action = projector.transform(embeddings[0])
    assert action.vector.shape == (256,)
