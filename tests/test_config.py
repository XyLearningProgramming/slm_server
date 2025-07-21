import os

from slm_server.config import get_settings


def test_get_settings_with_env_vars():
    """
    Tests if the get_settings function correctly reads environment variables.
    """
    # Clear the instance to ensure a fresh read
    if hasattr(get_settings, "_instance"):
        delattr(get_settings, "_instance")

    # Set environment variables
    os.environ["SLM_MODEL_PATH"] = "/test/path"
    os.environ["SLM_N_CTX"] = "2048"

    settings = get_settings()

    assert settings.model_path == "/test/path"
    assert settings.n_ctx == 2048

    # Clean up environment variables
    del os.environ["SLM_MODEL_PATH"]
    del os.environ["SLM_N_CTX"]

    # Clear the instance again to avoid affecting other tests
    if hasattr(get_settings, "_instance"):
        delattr(get_settings, "_instance")
