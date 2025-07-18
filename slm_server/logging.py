import logging

from opentelemetry.instrumentation.logging import LoggingInstrumentor

from slm_server.config import LoggingSettings



def setup_logging(settings: LoggingSettings):
    """Setup all logging configurations based on settings."""
    # List of all uvicorn loggers that need configuration
    uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access"]
    known_loggers = uvicorn_loggers

    if not settings.verbose:
        # Suppress all logging output when not verbose
        logging.getLogger().setLevel(logging.CRITICAL)
        for logger_name in known_loggers:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        return

    # Initialize OpenTelemetry logging instrumentation with default settings
    LoggingInstrumentor().instrument(set_logging_format=True)

    # Set root logger level to INFO - LoggingInstrumentor handles the format
    logging.getLogger().setLevel(logging.INFO)

    # Configure all uvicorn loggers to inherit from root logger defaults
    for logger_name in known_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = True

