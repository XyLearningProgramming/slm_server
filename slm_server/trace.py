import base64

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from slm_server.config import TraceSettings


def setup_tracing(settings: TraceSettings) -> None:
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
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
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

        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
