import os

from activation_views.env import load_local_env


def test_load_local_env(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=test-token\n")
    os.environ.pop("HF_TOKEN", None)
    load_local_env(env_path)
    assert os.environ["HF_TOKEN"] == "test-token"
