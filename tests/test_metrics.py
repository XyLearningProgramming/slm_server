from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from slm_server.config import MetricsSettings
from slm_server.metrics import setup_metrics


def test_setup_metrics_disabled():
    """When metrics are disabled, no /metrics endpoint is added."""
    app = FastAPI()
    setup_metrics(app, MetricsSettings(enabled=False))
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 404


def test_setup_metrics_enabled_does_not_raise():
    """When metrics are enabled, setup_metrics instruments the app without error."""
    app = FastAPI()
    with (
        patch("slm_server.metrics.Instrumentator") as mock_inst,
        patch("slm_server.metrics.system_cpu_usage", return_value=lambda info: None),
        patch("slm_server.metrics.system_memory_usage", return_value=lambda info: None),
    ):
        mock_instance = MagicMock()
        mock_inst.return_value = mock_instance
        mock_instance.instrument.return_value = mock_instance

        setup_metrics(app, MetricsSettings(enabled=True, endpoint="/metrics"))

        mock_inst.assert_called_once()
        mock_instance.add.assert_called()
        mock_instance.instrument.assert_called_once_with(app)
        mock_instance.expose.assert_called_once_with(app, endpoint="/metrics")
