import base64

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from slm_server.config import TraceSettings


def setup_tracing(app: FastAPI, settings: TraceSettings) -> None:
    """Initialize OpenTelemetry tracing with optional Grafana Tempo export."""
    if not settings.enabled:
        return

    # Define your service name in a Resource
    resource = Resource.create(
        attributes={
            "service.name": settings.service_name,
            "service.version": "1.0.0",  # TODO: make this more accurate?
        }
    )
    from slm_server.utils import ErrorAwareSampler

    # Set up tracer provider with error-aware sampler
    ratio_based_sampler = TraceIdRatioBased(settings.sample_rate)
    error_aware_sampler = ErrorAwareSampler(ratio_based_sampler)

    trace.set_tracer_provider(
        TracerProvider(
            resource=resource,
            sampler=ParentBased(root=error_aware_sampler),
        )
    )
    tracer_provider = trace.get_tracer_provider()

    # Configure OTLP exporter for Grafana Tempo if endpoint is provided
    if settings.endpoint:
        headers = {}

        # Add basic auth if credentials are provided
        if settings.username and settings.password:
            credentials = f"{settings.username}:{settings.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"

        otlp_exporter = OTLPSpanExporter(endpoint=settings.endpoint, headers=headers)
        # Only sampled spans go to OTLP endpoint
        otlp_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(otlp_processor)

    # Add span processors - process ALL spans (sampled + unsampled)
    from slm_server.utils import SLMLoggingSpanProcessor, SLMMetricsSpanProcessor

    # Add logging processor for SLM-specific logging
    logging_processor = SLMLoggingSpanProcessor()
    tracer_provider.add_span_processor(logging_processor)

    # Add metrics processor for SLM-specific metrics
    metrics_processor = SLMMetricsSpanProcessor()
    tracer_provider.add_span_processor(metrics_processor)

    # Instrument FastAPI with tracing
    excluded_urls_str = (
        ",".join(settings.excluded_urls) if settings.excluded_urls else ""
    )
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=tracer_provider, excluded_urls=excluded_urls_str
    )
