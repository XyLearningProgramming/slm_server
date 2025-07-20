import logging
import traceback
from contextlib import contextmanager

from llama_cpp import ChatCompletionStreamResponse
from opentelemetry import trace
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from slm_server.model import ChatCompletionRequest, ChatCompletionResponse

# Constants for span naming and attributes
MODEL_NAME = "llama-cpp"
SPAN_PREFIX = "slm"

# Span names
SPAN_CHAT_COMPLETION = f"{SPAN_PREFIX}.chat_completion"

# Event names
EVENT_CHUNK_GENERATED = f"{SPAN_PREFIX}.chunk_generated"

# Event attribute names
EVENT_ATTR_CHUNK_SIZE = f"{SPAN_PREFIX}.chunk_size"
EVENT_ATTR_CHUNK_CONTENT_SIZE = f"{SPAN_PREFIX}.chunk_content_size"
# EVENT_ATTR_CHUNK_CONTENT = f"{SPAN_PREFIX}.chunk_content"
# EVENT_ATTR_FINISH_REASON = f"{SPAN_PREFIX}.finish_reason"

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
ATTR_CHUNK_CONTENT_SIZE = f"{SPAN_PREFIX}.chunk.content_size"

# Calculated metric names (used as keys in calculate_performance_metrics)
METRIC_TOTAL_DURATION = ATTR_TOTAL_DURATION
METRIC_TOKENS_PER_SECOND = ATTR_TOKENS_PER_SECOND
METRIC_TOTAL_TOKENS_PER_SECOND = ATTR_TOTAL_TOKENS_PER_SECOND
METRIC_CHUNK_DELAY = ATTR_CHUNK_DELAY
METRIC_FIRST_TOKEN_DELAY = ATTR_FIRST_TOKEN_DELAY
METRIC_AVG_CHUNK_SIZE = f"{SPAN_PREFIX}.metrics.avg_chunk_size"
METRIC_AVG_CHUNK_CONTENT_SIZE = f"{SPAN_PREFIX}.metrics.avg_chunk_content_size"
METRIC_MAX_CHUNK_SIZE = f"{SPAN_PREFIX}.metrics.max_chunk_size"
METRIC_MIN_CHUNK_SIZE = f"{SPAN_PREFIX}.metrics.min_chunk_size"
METRIC_CHUNKS_WITH_CONTENT = f"{SPAN_PREFIX}.metrics.chunks_with_content"
METRIC_EMPTY_CHUNKS = f"{SPAN_PREFIX}.metrics.empty_chunks"

# Log data keys (for consistent logging format)
LOG_KEY_MAX_TOKENS = "max_tokens"
LOG_KEY_TEMPERATURE = "temperature"
LOG_KEY_INPUT_MESSAGES = "input_messages"
LOG_KEY_INPUT_CONTENT_LENGTH = "input_content_length"
LOG_KEY_DURATION_MS = "duration_ms"
LOG_KEY_OUTPUT_CONTENT_LENGTH = "output_content_length"
LOG_KEY_TOTAL_TOKENS = "total_tokens"
LOG_KEY_COMPLETION_TOKENS = "completion_tokens"
LOG_KEY_COMPLETION_TOKENS_PER_SECOND = "completion_tokens_per_second"
LOG_KEY_TOTAL_TOKENS_PER_SECOND = "total_tokens_per_second"
LOG_KEY_CHUNK_COUNT = "chunk_count"
LOG_KEY_AVG_CHUNK_DELAY_MS = "avg_chunk_delay_ms"
LOG_KEY_FIRST_TOKEN_DELAY_MS = "first_token_delay_ms"
LOG_KEY_AVG_CHUNK_SIZE = "avg_chunk_size"
LOG_KEY_AVG_CHUNK_CONTENT_SIZE = "avg_chunk_content_size"
LOG_KEY_CHUNKS_WITH_CONTENT = "chunks_with_content"
LOG_KEY_EMPTY_CHUNKS = "empty_chunks"

# Prometheus metric names and descriptions
PROMETHEUS_COMPLETION_DURATION = "slm_completion_duration_seconds"
PROMETHEUS_COMPLETION_DURATION_DESC = "SLM completion duration in seconds"
PROMETHEUS_TOKEN_COUNT = "slm_tokens_total"
PROMETHEUS_TOKEN_COUNT_DESC = "Total tokens processed"
PROMETHEUS_COMPLETION_TOKENS_PER_SECOND = "slm_completion_tokens_per_second"
PROMETHEUS_COMPLETION_TOKENS_PER_SECOND_DESC = (
    "Completion token generation rate (tokens/sec)"
)
PROMETHEUS_TOTAL_TOKENS_PER_SECOND = "slm_total_tokens_per_second"
PROMETHEUS_TOTAL_TOKENS_PER_SECOND_DESC = (
    "Total token throughput including prompt processing (tokens/sec)"
)
PROMETHEUS_FIRST_TOKEN_DELAY = "slm_first_token_delay_ms"
PROMETHEUS_FIRST_TOKEN_DELAY_DESC = "Time to first token in milliseconds (streaming)"
PROMETHEUS_CHUNK_DELAY = "slm_chunk_delay_ms"
PROMETHEUS_CHUNK_DELAY_DESC = "Average chunk delay in milliseconds (streaming)"
PROMETHEUS_CHUNK_DURATION = "slm_chunk_duration_ms"
PROMETHEUS_CHUNK_DURATION_DESC = "Individual chunk processing duration in milliseconds"
PROMETHEUS_ERROR_TOTAL = "slm_errors_total"
PROMETHEUS_ERROR_TOTAL_DESC = "Total SLM errors"
PROMETHEUS_CHUNK_COUNT = "slm_chunks_total"
PROMETHEUS_CHUNK_COUNT_DESC = "Number of chunks in streaming response"

# Log message templates
LOG_MSG_STARTING_CALL = "[SLM] starting {}: {}"
LOG_MSG_COMPLETED_CALL = "[SLM] completed {}: {}"
LOG_MSG_FAILED_CALL = "[SLM] failed: {}"


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
    """Record streaming chunk as an event and accumulate tokens."""
    chunk_content = ""
    if (
        response.choices
        and response.choices[0].delta
        and response.choices[0].delta.content
    ):
        chunk_content = response.choices[0].delta.content

    chunk_json = response.model_dump_json()

    # Record chunk as an event
    chunk_event = {
        EVENT_ATTR_CHUNK_SIZE: len(chunk_json),
        EVENT_ATTR_CHUNK_CONTENT_SIZE: len(chunk_content),
        # EVENT_ATTR_CHUNK_CONTENT: chunk_content,
        # EVENT_ATTR_FINISH_REASON: response.choices[0].finish_reason or 0
        # if response.choices
        # else None,
    }
    span.add_event(EVENT_CHUNK_GENERATED, chunk_event)

    # Only count chunks with actual content
    if not chunk_content:
        return

    # Accumulate tokens directly on the span
    current_completion_tokens = span.attributes.get(ATTR_COMPLETION_TOKENS, 0)
    span.set_attribute(ATTR_COMPLETION_TOKENS, current_completion_tokens + 1)

    # Update total content length
    current_output_length = span.attributes.get(ATTR_OUTPUT_CONTENT_LENGTH, 0)
    span.set_attribute(
        ATTR_OUTPUT_CONTENT_LENGTH, current_output_length + len(chunk_content)
    )

    # Update total tokens (assuming we have prompt tokens from initial setup)
    prompt_tokens = span.attributes.get(ATTR_PROMPT_TOKENS, 0)
    total_tokens = prompt_tokens + current_completion_tokens + 1
    span.set_attribute(ATTR_TOTAL_TOKENS, total_tokens)

    # Update chunk count
    current_chunk_count = span.attributes.get(ATTR_CHUNK_COUNT, 0)
    span.set_attribute(ATTR_CHUNK_COUNT, current_chunk_count + 1)


@contextmanager
def slm_span(req: ChatCompletionRequest, is_streaming: bool):
    """Create SLM span with automatic timing and error handling."""
    span_name = (
        f"{SPAN_CHAT_COMPLETION}.{'streaming' if is_streaming else 'non_streaming'}"
    )

    # Pre-calculate attributes before starting span
    messages_for_llm = [msg.model_dump() for msg in req.messages]
    input_content_length = sum(len(msg.get("content", "")) for msg in messages_for_llm)

    # Set initial attributes that will be available in on_start
    initial_attributes = {
        ATTR_MODEL: MODEL_NAME,
        ATTR_STREAMING: is_streaming,
        ATTR_MAX_TOKENS: req.max_tokens or 0,
        ATTR_TEMPERATURE: req.temperature,
        ATTR_INPUT_MESSAGES: len(messages_for_llm),
        ATTR_INPUT_CONTENT_LENGTH: input_content_length,
    }

    # Add prompt tokens estimate for streaming
    if is_streaming:
        # Estimate prompt tokens for streaming (rough approximation: 1 token per 4 chars)
        estimated_prompt_tokens = (
            max(1, input_content_length // 4) if is_streaming else 0
        )
        initial_attributes[ATTR_PROMPT_TOKENS] = estimated_prompt_tokens

    with tracer.start_as_current_span(span_name, attributes=initial_attributes) as span:
        try:
            yield span, messages_for_llm

        except Exception:
            # Use native error handling
            error_str = traceback.format_exc()
            span.set_status(Status(StatusCode.ERROR, error_str))
            span.set_attribute(ATTR_FORCE_SAMPLE, True)
            raise


def calculate_performance_metrics(span: Span):
    """Calculate performance metrics for a span after it has ended."""
    if not (span.end_time and span.start_time):
        return {}

    attrs = span.attributes or {}
    duration_ms = (span.end_time - span.start_time) / 1_000_000

    # Get token counts
    total_tokens = attrs.get(ATTR_TOTAL_TOKENS, 0)
    completion_tokens = attrs.get(ATTR_COMPLETION_TOKENS, 0)

    metrics = {
        METRIC_TOTAL_DURATION: duration_ms,
        METRIC_TOKENS_PER_SECOND: 0,
        METRIC_TOTAL_TOKENS_PER_SECOND: 0,
    }

    # Calculate tokens per second
    if duration_ms > 0:
        duration_s = duration_ms / 1000
        if completion_tokens > 0:
            metrics[METRIC_TOKENS_PER_SECOND] = completion_tokens / duration_s
        if total_tokens > 0:
            metrics[METRIC_TOTAL_TOKENS_PER_SECOND] = total_tokens / duration_s

    # Calculate streaming-specific metrics
    is_streaming = attrs.get(ATTR_STREAMING, False)
    if is_streaming:
        chunk_count = attrs.get(ATTR_CHUNK_COUNT, 0)
        if chunk_count > 0 and duration_ms > 0:
            metrics[METRIC_CHUNK_DELAY] = duration_ms / chunk_count

        # Calculate chunk metrics from events
        chunk_metrics = _calculate_chunk_metrics_from_events(span.events)
        metrics.update(chunk_metrics)

        # First token delay - find first chunk with content
        first_content_event = None
        for event in span.events:
            if event.name == EVENT_CHUNK_GENERATED:
                first_content_event = event
                break

        if first_content_event:
            first_token_delay = first_content_event.timestamp - span.start_time
            metrics[METRIC_FIRST_TOKEN_DELAY] = first_token_delay / 1_000_000

    return metrics


def _calculate_chunk_metrics_from_events(events):
    """Calculate chunk-related metrics from span events."""
    chunk_events = [e for e in events if e.name == EVENT_CHUNK_GENERATED]

    if not chunk_events:
        return {}

    chunk_sizes = []
    chunk_content_sizes = []
    chunks_with_content = 0
    empty_chunks = 0

    for event in chunk_events:
        attrs = event.attributes or {}

        chunk_size = attrs.get(EVENT_ATTR_CHUNK_SIZE, 0)
        chunk_content_size = attrs.get(EVENT_ATTR_CHUNK_CONTENT_SIZE, 0)
        # chunk_content = attrs.get(EVENT_ATTR_CHUNK_CONTENT, "")

        chunk_sizes.append(chunk_size)
        chunk_content_sizes.append(chunk_content_size)

        if chunk_content_size:
            chunks_with_content += 1
        else:
            empty_chunks += 1

    metrics = {}

    if chunk_sizes:
        metrics[METRIC_AVG_CHUNK_SIZE] = sum(chunk_sizes) / len(chunk_sizes)
        metrics[METRIC_MAX_CHUNK_SIZE] = max(chunk_sizes)
        metrics[METRIC_MIN_CHUNK_SIZE] = min(chunk_sizes)

    if chunk_content_sizes:
        metrics[METRIC_AVG_CHUNK_CONTENT_SIZE] = sum(chunk_content_sizes) / len(
            chunk_content_sizes
        )

    metrics[METRIC_CHUNKS_WITH_CONTENT] = chunks_with_content
    metrics[METRIC_EMPTY_CHUNKS] = empty_chunks

    return metrics


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
            LOG_KEY_MAX_TOKENS: attrs.get(ATTR_MAX_TOKENS, 0),
            LOG_KEY_TEMPERATURE: attrs.get(ATTR_TEMPERATURE, 0.0),
            LOG_KEY_INPUT_MESSAGES: attrs.get(ATTR_INPUT_MESSAGES, 0),
            LOG_KEY_INPUT_CONTENT_LENGTH: attrs.get(ATTR_INPUT_CONTENT_LENGTH, 0),
        }
        mode = "streaming" if is_streaming else "non-streaming"
        self.logger.info(LOG_MSG_STARTING_CALL.format(mode, log_data))

    def on_end(self, span):
        """Log span completion or error."""
        if not span.name.startswith(SPAN_PREFIX):
            return

        attrs = span.attributes or {}

        # Skip non-main spans (we no longer use chunk spans)
        if not span.name.startswith(SPAN_CHAT_COMPLETION):
            return

        # Use native error status
        if span.status.status_code == StatusCode.ERROR:
            self.logger.error(LOG_MSG_FAILED_CALL.format(span.status.description))
            return

        # Calculate performance metrics (but don't try to set them on ended span)
        performance_metrics = calculate_performance_metrics(span)
        # Merge calculated metrics with existing attributes for logging
        attrs = dict(attrs)
        attrs.update(performance_metrics)
        is_streaming = attrs.get(ATTR_STREAMING, False)
        mode = "streaming" if is_streaming else "non-streaming"

        log_data = {
            LOG_KEY_DURATION_MS: round(attrs.get(METRIC_TOTAL_DURATION, 0), 2),
            LOG_KEY_OUTPUT_CONTENT_LENGTH: attrs.get(ATTR_OUTPUT_CONTENT_LENGTH, 0),
            LOG_KEY_TOTAL_TOKENS: attrs.get(ATTR_TOTAL_TOKENS, 0),
            LOG_KEY_COMPLETION_TOKENS: attrs.get(ATTR_COMPLETION_TOKENS, 0),
            LOG_KEY_COMPLETION_TOKENS_PER_SECOND: round(
                attrs.get(METRIC_TOKENS_PER_SECOND, 0), 2
            ),
            LOG_KEY_TOTAL_TOKENS_PER_SECOND: round(
                attrs.get(METRIC_TOTAL_TOKENS_PER_SECOND, 0), 2
            ),
        }

        # Add streaming-specific metrics
        if is_streaming:
            log_data.update(
                {
                    LOG_KEY_CHUNK_COUNT: attrs.get(ATTR_CHUNK_COUNT, 0),
                    LOG_KEY_AVG_CHUNK_DELAY_MS: round(
                        attrs.get(METRIC_CHUNK_DELAY, 0), 2
                    ),
                    LOG_KEY_FIRST_TOKEN_DELAY_MS: round(
                        attrs.get(METRIC_FIRST_TOKEN_DELAY, 0), 2
                    ),
                    LOG_KEY_AVG_CHUNK_SIZE: round(
                        attrs.get(METRIC_AVG_CHUNK_SIZE, 0), 2
                    ),
                    LOG_KEY_AVG_CHUNK_CONTENT_SIZE: round(
                        attrs.get(METRIC_AVG_CHUNK_CONTENT_SIZE, 0), 2
                    ),
                    LOG_KEY_CHUNKS_WITH_CONTENT: attrs.get(
                        METRIC_CHUNKS_WITH_CONTENT, 0
                    ),
                    LOG_KEY_EMPTY_CHUNKS: attrs.get(METRIC_EMPTY_CHUNKS, 0),
                }
            )

        self.logger.info(LOG_MSG_COMPLETED_CALL.format(mode, log_data))

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True


class SLMMetricsSpanProcessor(SpanProcessor):
    """Span processor for SLM metrics using constants."""

    def __init__(self):
        # Duration metrics
        self.completion_duration = Histogram(
            PROMETHEUS_COMPLETION_DURATION,
            PROMETHEUS_COMPLETION_DURATION_DESC,
            labelnames=["model", "streaming", "status"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        )

        # Token metrics
        self.token_count = Histogram(
            PROMETHEUS_TOKEN_COUNT,
            PROMETHEUS_TOKEN_COUNT_DESC,
            labelnames=["model", "streaming", "token_type"],
            buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000],
        )

        # Throughput metrics - completion tokens (generation rate)
        self.completion_tokens_per_second = Histogram(
            PROMETHEUS_COMPLETION_TOKENS_PER_SECOND,
            PROMETHEUS_COMPLETION_TOKENS_PER_SECOND_DESC,
            labelnames=["model", "streaming"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # Throughput metrics - total tokens (including prompt processing)
        self.total_tokens_per_second = Histogram(
            PROMETHEUS_TOTAL_TOKENS_PER_SECOND,
            PROMETHEUS_TOTAL_TOKENS_PER_SECOND_DESC,
            labelnames=["model", "streaming"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # First token delay (streaming only)
        self.first_token_delay = Histogram(
            PROMETHEUS_FIRST_TOKEN_DELAY,
            PROMETHEUS_FIRST_TOKEN_DELAY_DESC,
            labelnames=["model"],
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
        )

        # Chunk delay metrics (streaming only)
        self.chunk_delay = Histogram(
            PROMETHEUS_CHUNK_DELAY,
            PROMETHEUS_CHUNK_DELAY_DESC,
            labelnames=["model"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # Chunk duration metrics
        self.chunk_duration = Histogram(
            PROMETHEUS_CHUNK_DURATION,
            PROMETHEUS_CHUNK_DURATION_DESC,
            labelnames=["model"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # Error rate
        self.error_total = Counter(
            PROMETHEUS_ERROR_TOTAL,
            PROMETHEUS_ERROR_TOTAL_DESC,
            labelnames=["model", "streaming", "error_type"],
        )

        # Chunk count for streaming
        self.chunk_count = Histogram(
            PROMETHEUS_CHUNK_COUNT,
            PROMETHEUS_CHUNK_COUNT_DESC,
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

        # Skip non-main spans (we no longer use chunk spans)
        if not span.name.startswith(SPAN_CHAT_COMPLETION):
            return

        is_streaming = attrs.get(ATTR_STREAMING, False)
        streaming_label = "streaming" if is_streaming else "non_streaming"

        # Calculate performance metrics first
        performance_metrics = calculate_performance_metrics(span)
        # Merge calculated metrics with existing attributes
        all_attrs = dict(attrs)
        all_attrs.update(performance_metrics)

        # Duration using calculated metric
        duration_ms = all_attrs.get(METRIC_TOTAL_DURATION, 0)
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
        prompt_tokens = all_attrs.get(ATTR_PROMPT_TOKENS, 0)
        completion_tokens = all_attrs.get(ATTR_COMPLETION_TOKENS, 0)

        if prompt_tokens > 0:
            self.token_count.labels(
                model=model, streaming=streaming_label, token_type="prompt"
            ).observe(prompt_tokens)

        if completion_tokens > 0:
            self.token_count.labels(
                model=model, streaming=streaming_label, token_type="completion"
            ).observe(completion_tokens)

        # Throughput metrics using calculated metrics
        completion_tps = all_attrs.get(METRIC_TOKENS_PER_SECOND, 0)
        if completion_tps > 0:
            self.completion_tokens_per_second.labels(
                model=model, streaming=streaming_label
            ).observe(completion_tps)

        total_tps = all_attrs.get(METRIC_TOTAL_TOKENS_PER_SECOND, 0)
        if total_tps > 0:
            self.total_tokens_per_second.labels(
                model=model, streaming=streaming_label
            ).observe(total_tps)

        # Streaming-specific metrics
        if is_streaming:
            # Chunk count
            chunk_count = all_attrs.get(ATTR_CHUNK_COUNT, 0)
            if chunk_count > 0:
                self.chunk_count.labels(model=model).observe(chunk_count)

            # First token delay
            first_token_delay_ms = all_attrs.get(METRIC_FIRST_TOKEN_DELAY, 0)
            if first_token_delay_ms > 0:
                self.first_token_delay.labels(model=model).observe(first_token_delay_ms)

            # Average chunk delay
            chunk_delay_ms = all_attrs.get(METRIC_CHUNK_DELAY, 0)
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
