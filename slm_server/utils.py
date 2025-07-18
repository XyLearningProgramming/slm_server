import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

from llama_cpp import ChatCompletionStreamResponse
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Histogram

from slm_server.model import ChatCompletionRequest, ChatCompletionResponse

# LLM model constants
LLM_MODEL_NAME = "llama-cpp"
LLM_SPAN_PREFIX = "llm.chat_completion"

# Get tracer and logger
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


@dataclass
class LLMStats:
    """Statistics and metrics for LLM calls."""

    input_content_length: int
    start_time: float
    chunk_count: int = 0
    total_output_length: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Chunk-level performance metrics
    first_chunk_time: float | None = None
    last_chunk_time: float | None = None
    chunk_times: list[float] = None
    chunk_sizes: list[int] = None
    chunk_intervals: list[float] = None

    def __post_init__(self):
        if self.chunk_times is None:
            self.chunk_times = []
        if self.chunk_sizes is None:
            self.chunk_sizes = []
        if self.chunk_intervals is None:
            self.chunk_intervals = []

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        duration = self.duration
        if duration <= 0:
            return 0
        return (
            self.total_tokens / duration
            if self.total_tokens > 0
            else self.chunk_count / duration
        )

    @property
    def time_to_first_token(self) -> float | None:
        """Time to first token in seconds."""
        if self.first_chunk_time is None:
            return None
        return self.first_chunk_time - self.start_time

    @property
    def average_chunk_interval(self) -> float | None:
        """Average time between chunks in seconds."""
        if not self.chunk_intervals:
            return None
        return sum(self.chunk_intervals) / len(self.chunk_intervals)

    @property
    def average_chunk_size(self) -> float | None:
        """Average chunk size in characters."""
        if not self.chunk_sizes:
            return None
        return sum(self.chunk_sizes) / len(self.chunk_sizes)

    @staticmethod
    def create_llm_stats(messages_for_llm: list[dict]) -> "LLMStats":
        """Create LLM stats object."""
        input_content_length = sum(
            len(msg.get("content", "")) for msg in messages_for_llm
        )
        return LLMStats(
            input_content_length=input_content_length, start_time=time.time()
        )

    def update_with_usage(self, response_model: ChatCompletionResponse):
        """Update stats with usage information from completion result."""
        usage_stats = _extract_usage_stats(response_model)
        self.prompt_tokens, self.completion_tokens, self.total_tokens = usage_stats
        # Extract metrics and update stats
        if response_model.choices and response_model.choices[0].message:
            content = response_model.choices[0].message.content or ""
            self.total_output_length = len(content)

    def update_with_chunk(self, response_model: ChatCompletionStreamResponse):
        """Update stats with usage information from streaming."""
        current_time = time.time()

        # Record first chunk time
        if self.first_chunk_time is None:
            self.first_chunk_time = current_time

        # Calculate interval from last chunk
        if self.last_chunk_time is not None:
            interval = current_time - self.last_chunk_time
            self.chunk_intervals.append(interval)

        self.chunk_count += 1
        self.last_chunk_time = current_time
        self.chunk_times.append(current_time)

        chunk_size = 0
        if (
            response_model.choices
            and response_model.choices[0].delta
            and response_model.choices[0].delta.content
        ):
            chunk_content = response_model.choices[0].delta.content
            chunk_size = len(chunk_content)
            self.total_output_length += chunk_size

        self.chunk_sizes.append(chunk_size)


@contextmanager
def start_as_custom_span(
    req: ChatCompletionRequest,
    messages_for_llm: list,
    stats: LLMStats,
    is_streaming: bool,
):
    """
    Context manager for LLM span with automatic setup, finalization, and error handling.
    ."""
    span_name = f"{LLM_SPAN_PREFIX}.{'streaming' if is_streaming else 'non_streaming'}"

    with tracer.start_as_current_span(span_name) as span:
        _setup_llm_span(span, req, messages_for_llm, stats, is_streaming)
        _log_llm_start(req, messages_for_llm, stats, is_streaming)

        try:
            yield span
            _finalize_llm_span(span, stats, is_streaming)
            _log_llm_completion(stats, is_streaming)
        except Exception as e:
            _handle_llm_error(span, stats, e, is_streaming)
            raise


def _setup_llm_span(
    span,
    req: ChatCompletionRequest,
    messages_for_llm: list,
    stats: LLMStats,
    is_streaming: bool,
):
    """Set up common span attributes for LLM calls."""
    span.set_attribute("llm.model", LLM_MODEL_NAME)
    span.set_attribute("llm.streaming", is_streaming)
    span.set_attribute("llm.max_tokens", req.max_tokens or 0)
    span.set_attribute("llm.temperature", req.temperature)
    span.set_attribute("llm.input.message_count", len(messages_for_llm))
    span.set_attribute("llm.input.content_length", stats.input_content_length)


def _log_llm_start(
    req: ChatCompletionRequest,
    messages_for_llm: list,
    stats: LLMStats,
    is_streaming: bool,
):
    """Log LLM call start."""
    mode = "streaming" if is_streaming else "non-streaming"
    log_data = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "input_messages": len(messages_for_llm),
        "input_content_length": stats.input_content_length,
    }
    logger.info(f"Starting {mode} LLM call: {log_data}")


def _extract_usage_stats(
    completion_result: ChatCompletionResponse,
) -> tuple[int, int, int]:
    """Extract usage statistics from completion result."""
    usage = completion_result.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _finalize_llm_span(span, stats: LLMStats, is_streaming: bool):
    """Finalize span with completion metrics."""
    span.set_attribute("llm.output.content_length", stats.total_output_length)
    span.set_attribute("llm.duration_ms", stats.duration * 1000)

    if is_streaming:
        span.set_attribute("llm.output.chunk_count", stats.chunk_count)

        # Add streaming-specific performance metrics
        if stats.time_to_first_token is not None:
            span.set_attribute(
                "llm.time_to_first_token_ms", stats.time_to_first_token * 1000
            )

        if stats.average_chunk_interval is not None:
            span.set_attribute(
                "llm.average_chunk_interval_ms", stats.average_chunk_interval * 1000
            )

        if stats.average_chunk_size is not None:
            span.set_attribute("llm.average_chunk_size", stats.average_chunk_size)

        if stats.chunk_sizes:
            span.set_attribute("llm.chunk_size_min", min(stats.chunk_sizes))
            span.set_attribute("llm.chunk_size_max", max(stats.chunk_sizes))

    # Add token usage attributes for both streaming and non-streaming
    if stats.prompt_tokens > 0:
        span.set_attribute("llm.usage.prompt_tokens", stats.prompt_tokens)
    if stats.completion_tokens > 0:
        span.set_attribute("llm.usage.completion_tokens", stats.completion_tokens)
    if stats.total_tokens > 0:
        span.set_attribute("llm.usage.total_tokens", stats.total_tokens)

    span.set_status(Status(StatusCode.OK))


def _log_llm_completion(stats: LLMStats, is_streaming: bool):
    """Log LLM call completion."""
    mode = "streaming" if is_streaming else "non-streaming"
    log_data = {
        "duration_ms": stats.duration * 1000,
        "tokens_per_second": stats.tokens_per_second,
    }

    if is_streaming:
        streaming_data = {
            "chunk_count": stats.chunk_count,
            "output_content_length": stats.total_output_length,
        }

        # Add chunk-level performance metrics
        if stats.time_to_first_token is not None:
            streaming_data["time_to_first_token_ms"] = stats.time_to_first_token * 1000

        if stats.average_chunk_interval is not None:
            streaming_data["avg_chunk_interval_ms"] = (
                stats.average_chunk_interval * 1000
            )

        if stats.average_chunk_size is not None:
            streaming_data["avg_chunk_size"] = stats.average_chunk_size

        if stats.chunk_sizes:
            streaming_data["chunk_size_min"] = min(stats.chunk_sizes)
            streaming_data["chunk_size_max"] = max(stats.chunk_sizes)

        log_data.update(streaming_data)
    else:
        log_data.update(
            {
                "output_content_length": stats.total_output_length,
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.total_tokens,
            }
        )

    logger.info(f"Completed {mode} LLM call: {log_data}")


def _handle_llm_error(span, stats: LLMStats, error: Exception, is_streaming: bool):
    """Handle LLM call errors with span and logging."""
    span.set_status(Status(StatusCode.ERROR, type(error)))

    mode = "streaming" if is_streaming else "non-streaming"
    logger.exception(
        f"Error in {mode} LLM call: duration_ms: {stats.duration * 1000}",
    )


class PrometheusSpanProcessor(SpanProcessor):
    """Custom span processor that generates Prometheus metrics from span data."""

    def __init__(self):
        # LLM-specific metrics
        self.llm_span_latency = Histogram(
            "llm_span_latency_seconds",
            "LLM span duration in seconds",
            labelnames=["model", "operation", "streaming"],
            buckets=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120],
        )

        # self.llm_span_total = Counter(
        #     "llm_span_total",
        #     "Total number of LLM spans",
        #     labelnames=["model", "operation", "streaming", "status"],
        # )

        self.llm_input_tokens = Histogram(
            "llm_input_tokens",
            "Number of input tokens processed",
            labelnames=["model", "streaming"],
            buckets=[50, 100, 500, 1000, 2000, 5000, 10000],
        )

        self.llm_output_tokens = Histogram(
            "llm_output_tokens",
            "Number of output tokens generated",
            labelnames=["model", "streaming"],
            buckets=[50, 100, 500, 1000, 2000, 5000, 10000],
        )

        self.llm_time_to_first_token = Histogram(
            "llm_time_to_first_token_seconds",
            "Time to first token in seconds (streaming only)",
            labelnames=["model"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # General span metrics: ignored for now
        #
        # self.span_latency = Histogram(
        #     "span_latency_seconds",
        #     "Span duration in seconds",
        #     labelnames=["span_name", "status"],
        #     buckets=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120],
        # )

    def on_start(self, span, parent_context=None):
        """Called when a span starts."""
        pass

    def on_end(self, span):
        """Called when a span ends - record metrics."""
        if not span.end_time or not span.start_time:
            return

        # Calculate duration in seconds
        duration = (span.end_time - span.start_time) / 1_000_000_000

        # Determine span status
        status = "ok" if span.status.status_code == StatusCode.OK else "error"

        # Record general span metrics
        # self.span_latency.labels(span_name=span.name, status=status).observe(duration)

        # Process LLM-specific spans
        if span.name.startswith("llm."):
            self._record_llm_metrics(span, duration, status)

    def _record_llm_metrics(self, span, duration, status):
        """Record LLM-specific metrics from span attributes."""
        attrs = span.attributes or {}

        # Extract key attributes
        model = attrs.get("llm.model", "unknown")
        streaming = str(attrs.get("llm.streaming", False)).lower()
        operation = "streaming" if streaming == "true" else "non_streaming"

        # Record basic LLM metrics
        self.llm_span_latency.labels(
            model=model, operation=operation, streaming=streaming
        ).observe(duration)

        # self.llm_span_total.labels(
        #     model=model, operation=operation, streaming=streaming, status=status
        # ).inc()

        # Record token metrics if available
        input_tokens = attrs.get("llm.usage.prompt_tokens")
        if input_tokens is not None:
            self.llm_input_tokens.labels(model=model, streaming=streaming).observe(
                input_tokens
            )

        output_tokens = attrs.get("llm.usage.completion_tokens")
        if output_tokens is not None:
            self.llm_output_tokens.labels(model=model, streaming=streaming).observe(
                output_tokens
            )

        # Record time to first token for streaming
        if streaming == "true":
            ttft_ms = attrs.get("llm.time_to_first_token_ms")
            if ttft_ms is not None:
                self.llm_time_to_first_token.labels(model=model).observe(ttft_ms / 1000)

    def shutdown(self):
        """Cleanup when processor shuts down."""
        pass

    def force_flush(self, timeout_millis: int = 30000):
        """Force flush - no-op for Prometheus metrics."""
        return True
