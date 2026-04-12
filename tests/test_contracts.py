import numpy as np

from activation_views.contracts import ActionVector, ActivationSnapshot, ObservationImage, TripletRecord


def test_contract_validation_roundtrip() -> None:
    snapshot = ActivationSnapshot(
        residuals_by_layer={10: np.ones((8, 16), dtype=np.float32)},
        prompt_id="p0",
        source="src",
        category="cat",
        token_step=0,
        model_name="model",
        layers=[10],
        seq_len=8,
    )
    snapshot.validate()
    obs = ObservationImage(np.ones((3, 64, 64), dtype=np.float32), "hsv_pca3", "model", [10])
    action = ActionVector(np.ones((256,), dtype=np.float32))
    triplet = TripletRecord(obs, action, obs, "src", "cat", "p0", 0, "model", "hsv_pca3")
    triplet.validate()
