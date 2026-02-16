import logging
from unittest.mock import MagicMock, patch

from fastapi import FastAPI

from slm_server.config import TraceSettings
from slm_server.trace import setup_tracing


def test_setup_tracing_disabled():
    """When tracing is disabled, nothing is set up."""
    app = FastAPI()
    settings = TraceSettings(
        enabled=False,
        endpoint="http://tempo:4318",
        username="user",
        password="pass",
    )
    with patch("slm_server.trace.trace.set_tracer_provider") as mock_set_tp:
        setup_tracing(app, settings)
        mock_set_tp.assert_not_called()


def test_setup_tracing_missing_endpoint(caplog):
    """When enabled but endpoint is empty, logs warning and skips setup."""
    app = FastAPI()
    settings = TraceSettings(
        enabled=True,
        endpoint="",
        username="user",
        password="pass",
    )
    with (
        patch("slm_server.trace.trace.set_tracer_provider") as mock_set_tp,
        caplog.at_level(logging.WARNING, logger="slm_server.trace"),
    ):
        setup_tracing(app, settings)
        mock_set_tp.assert_not_called()
        assert "not configured" in caplog.text


def test_setup_tracing_missing_username(caplog):
    """When enabled but username is empty, logs warning and skips setup."""
    app = FastAPI()
    settings = TraceSettings(
        enabled=True,
        endpoint="http://tempo:4318",
        username="",
        password="pass",
    )
    with (
        patch("slm_server.trace.trace.set_tracer_provider") as mock_set_tp,
        caplog.at_level(logging.WARNING, logger="slm_server.trace"),
    ):
        setup_tracing(app, settings)
        mock_set_tp.assert_not_called()
        assert "not configured" in caplog.text


def test_setup_tracing_missing_password(caplog):
    """When enabled but password is empty, logs warning and skips setup."""
    app = FastAPI()
    settings = TraceSettings(
        enabled=True,
        endpoint="http://tempo:4318",
        username="user",
        password="",
    )
    with (
        patch("slm_server.trace.trace.set_tracer_provider") as mock_set_tp,
        caplog.at_level(logging.WARNING, logger="slm_server.trace"),
    ):
        setup_tracing(app, settings)
        mock_set_tp.assert_not_called()
        assert "not configured" in caplog.text


def test_setup_tracing_full_setup():
    """When fully configured, sets up tracer provider, processors, and instruments app."""
    app = FastAPI()
    settings = TraceSettings(
        enabled=True,
        service_name="test-service",
        endpoint="http://tempo:4318/v1/traces",
        username="user",
        password="pass",
        sample_rate=1.0,
        excluded_urls=["/health"],
    )

    mock_provider = MagicMock()

    with (
        patch("slm_server.trace.trace.set_tracer_provider") as mock_set_tp,
        patch("slm_server.trace.trace.get_tracer_provider", return_value=mock_provider),
        patch("slm_server.trace.OTLPSpanExporter") as mock_otlp,
        patch("slm_server.trace.BatchSpanProcessor") as mock_batch,
        patch("slm_server.trace.FastAPIInstrumentor") as mock_instrumentor,
    ):
        setup_tracing(app, settings)

        # Tracer provider was set
        mock_set_tp.assert_called_once()

        # OTLP exporter created with endpoint and auth header
        mock_otlp.assert_called_once()
        call_kwargs = mock_otlp.call_args
        assert call_kwargs[1]["endpoint"] == "http://tempo:4318/v1/traces"
        assert "Authorization" in call_kwargs[1]["headers"]
        assert call_kwargs[1]["headers"]["Authorization"].startswith("Basic ")

        # BatchSpanProcessor created with the OTLP exporter
        mock_batch.assert_called_once_with(mock_otlp.return_value)

        # Three span processors added: OTLP batch + logging + metrics
        assert mock_provider.add_span_processor.call_count == 3

        # FastAPI instrumented
        mock_instrumentor.instrument_app.assert_called_once()
        instr_kwargs = mock_instrumentor.instrument_app.call_args
        assert instr_kwargs[1]["excluded_urls"] == "/health"
