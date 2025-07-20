import logging
from contextlib import contextmanager

from llama_cpp import ChatCompletionStreamResponse
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import Counter, Histogram

from slm_server.model import ChatCompletionRequest, ChatCompletionResponse

# Constants for span naming and attributes
MODEL_NAME = "llama-cpp"
SPAN_PREFIX = "slm"

# Span names
SPAN_CHAT_COMPLETION = f"{SPAN_PREFIX}.chat_completion"
SPAN_CHUNK_GENERATION = f"{SPAN_PREFIX}.chunk_generation"

# Attribute names
ATTR_MODEL = f"{SPAN_PREFIX}.model"
ATTR_STREAMING = f"{SPAN_PREFIX}.streaming"
ATTR_MAX_TOKENS = f"{SPAN_PREFIX}.max_tokens"
ATTR_TEMPERATURE = f"{SPAN_PREFIX}.temperature"
ATTR_INPUT_MESSAGES = f"{SPAN_PREFIX}.input.messages"
ATTR_INPUT_CONTENT_LENGTH = f"{SPAN_PREFIX}.input.content_length"
ATTR_OUTPUT_CONTENT_LENGTH = f"{SPAN_PREFIX}.output.content_length"
ATTR_CHUNK_COUNT = f"{SPAN_PREFIX}.output.chunk_count"
ATTR_CHUNK_SIZE = f"{SPAN_PREFIX}.chunk.size"
ATTR_PROMPT_TOKENS = f"{SPAN_PREFIX}.usage.prompt_tokens"
ATTR_COMPLETION_TOKENS = f"{SPAN_PREFIX}.usage.completion_tokens"
ATTR_TOTAL_TOKENS = f"{SPAN_PREFIX}.usage.total_tokens"
ATTR_FORCE_SAMPLE = f"{SPAN_PREFIX}.force_sample"

# Performance timing attributes
ATTR_FIRST_TOKEN_DELAY = f"{SPAN_PREFIX}.timing.first_token_delay_ms"
ATTR_TOKENS_PER_SECOND = f"{SPAN_PREFIX}.timing.completion_tokens_per_second"
ATTR_TOTAL_TOKENS_PER_SECOND = f"{SPAN_PREFIX}.timing.total_tokens_per_second"
ATTR_CHUNK_DELAY = f"{SPAN_PREFIX}.timing.chunk_delay_ms"
ATTR_CHUNK_DURATION = f"{SPAN_PREFIX}.timing.chunk_duration_ms"
ATTR_TOTAL_DURATION = f"{SPAN_PREFIX}.timing.total_duration_ms"
ATTR_FIRST_CHUNK_TIME = f"{SPAN_PREFIX}.timing.first_chunk_time"
ATTR_CHUNK_CONTENT_SIZE = f"{SPAN_PREFIX}.chunk.content_size"


# Get tracer
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def set_atrribute_response(span: Span, response: ChatCompletionResponse):
    """Set response attributes automatically."""
    # Non-streaming response
    if response.usage:
        span.set_attribute(ATTR_PROMPT_TOKENS, response.usage.prompt_tokens)
        span.set_attribute(ATTR_COMPLETION_TOKENS, response.usage.completion_tokens)
        span.set_attribute(ATTR_TOTAL_TOKENS, response.usage.total_tokens)

    if response.choices and response.choices[0].message:
        content = response.choices[0].message.content or ""
        span.set_attribute(ATTR_OUTPUT_CONTENT_LENGTH, len(content))


def set_atrribute_response_stream(span: Span, response: ChatCompletionStreamResponse):
    """Set streaming chunk response attributes automatically."""
    chunk_content = ""
    if (
        response.choices
        and response.choices[0].delta
        and response.choices[0].delta.content
    ):
        chunk_content = response.choices[0].delta.content

    chunk_json = response.model_dump_json()
    span.set_attribute(ATTR_CHUNK_SIZE, len(chunk_json))
    span.set_attribute(ATTR_CHUNK_CONTENT_SIZE, len(chunk_content))


@contextmanager
def slm_span(req: ChatCompletionRequest, is_streaming: bool):
    """Create SLM span with automatic timing and error handling."""
    span_name = (
        f"{SPAN_CHAT_COMPLETION}.{'streaming' if is_streaming else 'non_streaming'}"
    )

    with tracer.start_as_current_span(span_name) as span:
        try:
            # Set initial attributes
            messages_for_llm = [msg.model_dump() for msg in req.messages]
            input_content_length = sum(
                len(msg.get("content", "")) for msg in messages_for_llm
            )

            span.set_attribute(ATTR_MODEL, MODEL_NAME)
            span.set_attribute(ATTR_STREAMING, is_streaming)
            span.set_attribute(ATTR_MAX_TOKENS, req.max_tokens or 0)
            span.set_attribute(ATTR_TEMPERATURE, req.temperature)
            span.set_attribute(ATTR_INPUT_MESSAGES, len(messages_for_llm))
            span.set_attribute(ATTR_INPUT_CONTENT_LENGTH, input_content_length)

            yield span, messages_for_llm

        except Exception as e:
            # Use native error handling
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute(ATTR_FORCE_SAMPLE, True)
            raise


@contextmanager
def slm_chunk_span(parent_span):
    """Create chunk span that measures the actual next() call timing."""
    # Get current chunk count from parent span attributes
    current_chunk_count = parent_span.attributes.get(ATTR_CHUNK_COUNT, 0)
    chunk_number = current_chunk_count + 1

    # Update parent span chunk count
    parent_span.set_attribute(ATTR_CHUNK_COUNT, chunk_number)

    with tracer.start_as_current_span(
        f"{SPAN_CHUNK_GENERATION}.{chunk_number}"
    ) as span:
        # Record first chunk time on parent span for first token delay calculation
        if chunk_number == 1:
            parent_span.set_attribute(ATTR_FIRST_CHUNK_TIME, span.start_time)

        yield span


def calculate_performance_metrics(span):
    """Calculate performance metrics for a span after it has ended."""
    if not (span.end_time and span.start_time):
        return {}

    attrs = span.attributes or {}
    duration_ms = (span.end_time - span.start_time) / 1_000_000

    # Get token counts
    total_tokens = attrs.get(ATTR_TOTAL_TOKENS, 0)
    completion_tokens = attrs.get(ATTR_COMPLETION_TOKENS, 0)

    metrics = {
        ATTR_TOTAL_DURATION: duration_ms,
        ATTR_TOKENS_PER_SECOND: 0,
        ATTR_TOTAL_TOKENS_PER_SECOND: 0,
    }

    # Calculate tokens per second
    if duration_ms > 0:
        duration_s = duration_ms / 1000
        if completion_tokens > 0:
            metrics[ATTR_TOKENS_PER_SECOND] = completion_tokens / duration_s
        if total_tokens > 0:
            metrics[ATTR_TOTAL_TOKENS_PER_SECOND] = total_tokens / duration_s

    # Calculate streaming-specific metrics
    is_streaming = attrs.get(ATTR_STREAMING, False)
    if is_streaming:
        chunk_count = attrs.get(ATTR_CHUNK_COUNT, 0)
        if chunk_count > 0 and duration_ms > 0:
            metrics[ATTR_CHUNK_DELAY] = duration_ms / chunk_count

        # First token delay
        first_chunk_time = attrs.get(ATTR_FIRST_CHUNK_TIME)
        if first_chunk_time:
            first_token_delay = first_chunk_time - span.start_time
            metrics[ATTR_FIRST_TOKEN_DELAY] = first_token_delay / 1_000_000

    return metrics


def calculate_chunk_metrics(span):
    """Calculate chunk-specific metrics for a chunk span after it has ended."""
    if not (span.end_time and span.start_time):
        return {}

    chunk_duration_ms = (span.end_time - span.start_time) / 1_000_000
    return {ATTR_CHUNK_DURATION: chunk_duration_ms}


class SLMLoggingSpanProcessor(SpanProcessor):
    """Span processor for SLM logging using constants."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_start(self, span, parent_context=None):
        """Log span start."""
        if not span.name.startswith(SPAN_CHAT_COMPLETION):
            return

        attrs = span.attributes or {}
        is_streaming = attrs.get(ATTR_STREAMING, False)
        log_data = {
            "max_tokens": attrs.get(ATTR_MAX_TOKENS, 0),
            "temperature": attrs.get(ATTR_TEMPERATURE, 0.0),
            "input_messages": attrs.get(ATTR_INPUT_MESSAGES, 0),
            "input_content_length": attrs.get(ATTR_INPUT_CONTENT_LENGTH, 0),
        }
        mode = "streaming" if is_streaming else "non-streaming"
        self.logger.info(f"Starting {mode} SLM call: {log_data}")

    def on_end(self, span):
        """Log span completion or error."""
        if not span.name.startswith(SPAN_PREFIX):
            return

        attrs = span.attributes or {}

        # Handle chunk spans
        if span.name.startswith(SPAN_CHUNK_GENERATION):
            # Calculate and set chunk metrics
            chunk_metrics = calculate_chunk_metrics(span)
            for attr_name, value in chunk_metrics.items():
                span.set_attribute(attr_name, value)
            return

        # Use native error status
        if span.status.status_code == StatusCode.ERROR:
            self.logger.error(f"SLM call failed: {span.status.description}")
            return

        # Log success for main spans only
        if not span.name.startswith(SPAN_CHAT_COMPLETION):
            return

        # Calculate and set performance metrics
        performance_metrics = calculate_performance_metrics(span)
        for attr_name, value in performance_metrics.items():
            span.set_attribute(attr_name, value)

        # Refresh attrs after setting new attributes
        attrs = span.attributes or {}
        is_streaming = attrs.get(ATTR_STREAMING, False)
        mode = "streaming" if is_streaming else "non-streaming"

        log_data = {
            "duration_ms": round(attrs.get(ATTR_TOTAL_DURATION, 0), 2),
            "output_content_length": attrs.get(ATTR_OUTPUT_CONTENT_LENGTH, 0),
            "total_tokens": attrs.get(ATTR_TOTAL_TOKENS, 0),
            "completion_tokens": attrs.get(ATTR_COMPLETION_TOKENS, 0),
            "completion_tokens_per_second": round(
                attrs.get(ATTR_TOKENS_PER_SECOND, 0), 2
            ),
            "total_tokens_per_second": round(
                attrs.get(ATTR_TOTAL_TOKENS_PER_SECOND, 0), 2
            ),
        }

        # Add streaming-specific metrics
        if is_streaming:
            log_data.update(
                {
                    "chunk_count": attrs.get(ATTR_CHUNK_COUNT, 0),
                    "avg_chunk_delay_ms": round(attrs.get(ATTR_CHUNK_DELAY, 0), 2),
                    "first_token_delay_ms": round(
                        attrs.get(ATTR_FIRST_TOKEN_DELAY, 0), 2
                    ),
                }
            )

        self.logger.info(f"Completed {mode} SLM call: {log_data}")

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True


class SLMMetricsSpanProcessor(SpanProcessor):
    """Span processor for SLM metrics using constants."""

    def __init__(self):
        # Duration metrics
        self.completion_duration = Histogram(
            "slm_completion_duration_seconds",
            "SLM completion duration in seconds",
            labelnames=["model", "streaming", "status"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        )

        # Token metrics
        self.token_count = Histogram(
            "slm_tokens_total",
            "Total tokens processed",
            labelnames=["model", "streaming", "token_type"],
            buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000],
        )

        # Throughput metrics - completion tokens (generation rate)
        self.completion_tokens_per_second = Histogram(
            "slm_completion_tokens_per_second",
            "Completion token generation rate (tokens/sec)",
            labelnames=["model", "streaming"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # Throughput metrics - total tokens (including prompt processing)
        self.total_tokens_per_second = Histogram(
            "slm_total_tokens_per_second",
            "Total token throughput including prompt processing (tokens/sec)",
            labelnames=["model", "streaming"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # First token delay (streaming only)
        self.first_token_delay = Histogram(
            "slm_first_token_delay_ms",
            "Time to first token in milliseconds (streaming)",
            labelnames=["model"],
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
        )

        # Chunk delay metrics (streaming only)
        self.chunk_delay = Histogram(
            "slm_chunk_delay_ms",
            "Average chunk delay in milliseconds (streaming)",
            labelnames=["model"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # Chunk duration metrics
        self.chunk_duration = Histogram(
            "slm_chunk_duration_ms",
            "Individual chunk processing duration in milliseconds",
            labelnames=["model"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # Error rate
        self.error_total = Counter(
            "slm_errors_total",
            "Total SLM errors",
            labelnames=["model", "streaming", "error_type"],
        )

        # Chunk count for streaming
        self.chunk_count = Histogram(
            "slm_chunks_total",
            "Number of chunks in streaming response",
            labelnames=["model"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        """Record metrics on span end."""
        if not span.name.startswith(SPAN_PREFIX):
            return

        attrs = span.attributes or {}
        model = attrs.get(ATTR_MODEL, "unknown")

        # Handle chunk spans
        if span.name.startswith(SPAN_CHUNK_GENERATION):
            chunk_duration_ms = attrs.get(ATTR_CHUNK_DURATION, 0)
            if chunk_duration_ms > 0:
                self.chunk_duration.labels(model=model).observe(chunk_duration_ms)
            return

        # Record metrics for main completion spans
        if not span.name.startswith(SPAN_CHAT_COMPLETION):
            return

        is_streaming = attrs.get(ATTR_STREAMING, False)
        streaming_label = "streaming" if is_streaming else "non_streaming"

        # Duration using pre-calculated attribute
        duration_ms = attrs.get(ATTR_TOTAL_DURATION, 0)
        duration_s = duration_ms / 1000 if duration_ms > 0 else 0
        status = "success" if span.status.status_code == StatusCode.OK else "error"

        self.completion_duration.labels(
            model=model, streaming=streaming_label, status=status
        ).observe(duration_s)

        # Error tracking
        if span.status.status_code == StatusCode.ERROR:
            error_type = (
                type(span.status.description).__name__
                if span.status.description
                else "unknown"
            )
            self.error_total.labels(
                model=model, streaming=streaming_label, error_type=error_type
            ).inc()
            return

        # Token metrics
        prompt_tokens = attrs.get(ATTR_PROMPT_TOKENS, 0)
        completion_tokens = attrs.get(ATTR_COMPLETION_TOKENS, 0)
        # total_tokens = attrs.get(ATTR_TOTAL_TOKENS, 0)

        if prompt_tokens > 0:
            self.token_count.labels(
                model=model, streaming=streaming_label, token_type="prompt"
            ).observe(prompt_tokens)

        if completion_tokens > 0:
            self.token_count.labels(
                model=model, streaming=streaming_label, token_type="completion"
            ).observe(completion_tokens)

        # Throughput metrics using pre-calculated attributes
        completion_tps = attrs.get(ATTR_TOKENS_PER_SECOND, 0)
        if completion_tps > 0:
            self.completion_tokens_per_second.labels(
                model=model, streaming=streaming_label
            ).observe(completion_tps)

        total_tps = attrs.get(ATTR_TOTAL_TOKENS_PER_SECOND, 0)
        if total_tps > 0:
            self.total_tokens_per_second.labels(
                model=model, streaming=streaming_label
            ).observe(total_tps)

        # Streaming-specific metrics
        if is_streaming:
            # Chunk count
            chunk_count = attrs.get(ATTR_CHUNK_COUNT, 0)
            if chunk_count > 0:
                self.chunk_count.labels(model=model).observe(chunk_count)

            # First token delay
            first_token_delay_ms = attrs.get(ATTR_FIRST_TOKEN_DELAY, 0)
            if first_token_delay_ms > 0:
                self.first_token_delay.labels(model=model).observe(first_token_delay_ms)

            # Average chunk delay
            chunk_delay_ms = attrs.get(ATTR_CHUNK_DELAY, 0)
            if chunk_delay_ms > 0:
                self.chunk_delay.labels(model=model).observe(chunk_delay_ms)

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True


class ErrorAwareSampler(Sampler):
    """Sampler that forces sampling on errors."""

    attr_force_sample = ATTR_FORCE_SAMPLE

    def __init__(self, base_sampler: Sampler):
        self.base_sampler = base_sampler

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None,
    ):
        # Force sample if error attribute is set
        if attributes and attributes.get(self.attr_force_sample):
            return SamplingResult(
                decision=Decision.RECORD_AND_SAMPLE,
                attributes=attributes,
                trace_state=trace_state,
            )

        # Use base sampler otherwise
        return self.base_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self):
        return f"ErrorAwareSampler(base={self.base_sampler})"
