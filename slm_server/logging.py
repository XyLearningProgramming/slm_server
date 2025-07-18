import logging

from opentelemetry.instrumentation.logging import LoggingInstrumentor

from slm_server.config import LoggingSettings


def setup_logging(settings: LoggingSettings):
    """Setup all logging configurations based on settings."""
    # List of all uvicorn loggers that need configuration
    uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access"]

    # Initialize OpenTelemetry logging instrumentation with default settings
    log_level = logging.getLevelNamesMapping().get(settings.level, logging.INFO)
    LoggingInstrumentor().instrument(set_logging_format=True, log_level=log_level)

    # Configure all uvicorn loggers to inherit from root logger defaults
    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.handlers.clear()
        logger.propagate = True
