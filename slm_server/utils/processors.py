import logging

from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace import StatusCode
from prometheus_client import Counter, Histogram

from .constants import (
    ATTR_CHUNK_COUNT,
    ATTR_COMPLETION_TOKENS,
    ATTR_INPUT_CONTENT_LENGTH,
    ATTR_INPUT_COUNT,
    ATTR_INPUT_MESSAGES,
    ATTR_MAX_TOKENS,
    ATTR_MODEL,
    ATTR_OUTPUT_CONTENT_LENGTH,
    ATTR_OUTPUT_COUNT,
    ATTR_PROMPT_TOKENS,
    ATTR_STREAMING,
    ATTR_TEMPERATURE,
    ATTR_TOTAL_TOKENS,
    LOG_KEY_AVG_CHUNK_CONTENT_SIZE,
    LOG_KEY_AVG_CHUNK_DELAY_MS,
    LOG_KEY_AVG_CHUNK_SIZE,
    LOG_KEY_CHUNK_COUNT,
    LOG_KEY_CHUNKS_WITH_CONTENT,
    LOG_KEY_COMPLETION_TOKENS,
    LOG_KEY_COMPLETION_TOKENS_PER_SECOND,
    LOG_KEY_DURATION_MS,
    LOG_KEY_EMBEDDINGS_PER_SECOND,
    LOG_KEY_EMPTY_CHUNKS,
    LOG_KEY_FIRST_TOKEN_DELAY_MS,
    LOG_KEY_INPUT_CONTENT_LENGTH,
    LOG_KEY_INPUT_COUNT,
    LOG_KEY_INPUT_MESSAGES,
    LOG_KEY_MAX_TOKENS,
    LOG_KEY_OUTPUT_CONTENT_LENGTH,
    LOG_KEY_OUTPUT_COUNT,
    LOG_KEY_TEMPERATURE,
    LOG_KEY_TOTAL_TOKENS,
    LOG_KEY_TOTAL_TOKENS_PER_SECOND,
    LOG_MSG_COMPLETED_CALL,
    LOG_MSG_FAILED_CALL,
    LOG_MSG_STARTING_CALL,
    METRIC_AVG_CHUNK_CONTENT_SIZE,
    METRIC_AVG_CHUNK_SIZE,
    METRIC_CHUNK_DELAY,
    METRIC_CHUNKS_WITH_CONTENT,
    METRIC_EMBEDDINGS_PER_SECOND,
    METRIC_EMPTY_CHUNKS,
    METRIC_FIRST_TOKEN_DELAY,
    METRIC_TOKENS_PER_SECOND,
    METRIC_TOTAL_DURATION,
    METRIC_TOTAL_TOKENS_PER_SECOND,
    PROMETHEUS_CHUNK_COUNT,
    PROMETHEUS_CHUNK_COUNT_DESC,
    PROMETHEUS_CHUNK_DELAY,
    PROMETHEUS_CHUNK_DELAY_DESC,
    PROMETHEUS_CHUNK_DURATION,
    PROMETHEUS_CHUNK_DURATION_DESC,
    PROMETHEUS_COMPLETION_DURATION,
    PROMETHEUS_COMPLETION_DURATION_DESC,
    PROMETHEUS_COMPLETION_TOKENS_PER_SECOND,
    PROMETHEUS_COMPLETION_TOKENS_PER_SECOND_DESC,
    PROMETHEUS_EMBEDDINGS_PER_SECOND,
    PROMETHEUS_EMBEDDINGS_PER_SECOND_DESC,
    PROMETHEUS_ERROR_TOTAL,
    PROMETHEUS_ERROR_TOTAL_DESC,
    PROMETHEUS_FIRST_TOKEN_DELAY,
    PROMETHEUS_FIRST_TOKEN_DELAY_DESC,
    PROMETHEUS_TOKEN_COUNT,
    PROMETHEUS_TOKEN_COUNT_DESC,
    PROMETHEUS_TOTAL_TOKENS_PER_SECOND,
    PROMETHEUS_TOTAL_TOKENS_PER_SECOND_DESC,
    SPAN_CHAT_COMPLETION,
    SPAN_EMBEDDING,
    SPAN_PREFIX,
)
from .metrics import calculate_performance_metrics


class SLMLoggingSpanProcessor(SpanProcessor):
    """Span processor for SLM logging using constants."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_start(self, span, parent_context=None):
        """Log span start."""
        if not span.name.startswith(SPAN_PREFIX):
            return

        attrs = span.attributes or {}
        log_data = {}
        mode = "unknown"

        if span.name.startswith(SPAN_CHAT_COMPLETION):
            is_streaming = attrs.get(ATTR_STREAMING, False)
            log_data = {
                LOG_KEY_MAX_TOKENS: attrs.get(ATTR_MAX_TOKENS, 0),
                LOG_KEY_TEMPERATURE: attrs.get(ATTR_TEMPERATURE, 0.0),
                LOG_KEY_INPUT_MESSAGES: attrs.get(ATTR_INPUT_MESSAGES, 0),
                LOG_KEY_INPUT_CONTENT_LENGTH: attrs.get(ATTR_INPUT_CONTENT_LENGTH, 0),
            }
            mode = "streaming" if is_streaming else "non-streaming"

        elif span.name == SPAN_EMBEDDING:
            log_data = {
                "input_count": attrs.get(ATTR_INPUT_COUNT, 0),
                LOG_KEY_INPUT_CONTENT_LENGTH: attrs.get(ATTR_INPUT_CONTENT_LENGTH, 0),
            }
            mode = "embedding"

        self.logger.info(LOG_MSG_STARTING_CALL.format(mode, log_data))

    def on_end(self, span):
        """Log span completion or error."""
        if not span.name.startswith(SPAN_PREFIX):
            return

        attrs = span.attributes or {}

        # Use native error status
        if span.status.status_code == StatusCode.ERROR:
            self.logger.error(LOG_MSG_FAILED_CALL.format(span.status.description))
            return

        # Calculate performance metrics (but don't try to set them on ended span)
        performance_metrics = calculate_performance_metrics(span)
        # Merge calculated metrics with existing attributes for logging
        attrs = dict(attrs)
        attrs.update(performance_metrics)

        log_data = {
            LOG_KEY_DURATION_MS: round(attrs.get(METRIC_TOTAL_DURATION, 0), 2),
            LOG_KEY_TOTAL_TOKENS: attrs.get(ATTR_TOTAL_TOKENS, 0),
            LOG_KEY_TOTAL_TOKENS_PER_SECOND: round(
                attrs.get(METRIC_TOTAL_TOKENS_PER_SECOND, 0), 2
            ),
        }

        mode = "unknown"

        if span.name.startswith(SPAN_CHAT_COMPLETION):
            is_streaming = attrs.get(ATTR_STREAMING, False)
            mode = "streaming" if is_streaming else "non-streaming"
            log_data.update(
                {
                    LOG_KEY_OUTPUT_CONTENT_LENGTH: attrs.get(
                        ATTR_OUTPUT_CONTENT_LENGTH, 0
                    ),
                    LOG_KEY_COMPLETION_TOKENS: attrs.get(ATTR_COMPLETION_TOKENS, 0),
                    LOG_KEY_COMPLETION_TOKENS_PER_SECOND: round(
                        attrs.get(METRIC_TOKENS_PER_SECOND, 0), 2
                    ),
                }
            )
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

        elif span.name == SPAN_EMBEDDING:
            mode = "embedding"
            log_data.update(
                {
                    LOG_KEY_INPUT_COUNT: attrs.get(ATTR_INPUT_COUNT, 0),
                    LOG_KEY_OUTPUT_COUNT: attrs.get(ATTR_OUTPUT_COUNT, 0),
                    LOG_KEY_EMBEDDINGS_PER_SECOND: round(
                        attrs.get(METRIC_EMBEDDINGS_PER_SECOND, 0), 2
                    ),
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
            labelnames=["model", "streaming"],
        )

        # Chunk count for streaming
        self.chunk_count = Histogram(
            PROMETHEUS_CHUNK_COUNT,
            PROMETHEUS_CHUNK_COUNT_DESC,
            labelnames=["model"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

        # Embedding metrics
        self.embeddings_per_second = Histogram(
            PROMETHEUS_EMBEDDINGS_PER_SECOND,
            PROMETHEUS_EMBEDDINGS_PER_SECOND_DESC,
            labelnames=["model"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):  # noqa: C901
        """Record metrics on span end."""
        if not span.name.startswith(SPAN_PREFIX):
            return

        attrs = span.attributes or {}
        model = attrs.get(ATTR_MODEL, "unknown")
        status = "success" if span.status.status_code == StatusCode.OK else "error"

        # Calculate performance metrics first
        performance_metrics = calculate_performance_metrics(span)
        # Merge calculated metrics with existing attributes
        all_attrs = dict(attrs)
        all_attrs.update(performance_metrics)

        duration_ms = all_attrs.get(METRIC_TOTAL_DURATION, 0)
        duration_s = duration_ms / 1000 if duration_ms > 0 else 0

        if span.name.startswith(SPAN_CHAT_COMPLETION):
            is_streaming = attrs.get(ATTR_STREAMING, False)
            streaming_label = "streaming" if is_streaming else "non_streaming"

            self.completion_duration.labels(
                model=model, streaming=streaming_label, status=status
            ).observe(duration_s)

            if span.status.status_code == StatusCode.ERROR:
                self.error_total.labels(
                    model=model,
                    streaming=streaming_label,
                ).inc()
                return

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

            if is_streaming:
                chunk_count = all_attrs.get(ATTR_CHUNK_COUNT, 0)
                if chunk_count > 0:
                    self.chunk_count.labels(model=model).observe(chunk_count)

                first_token_delay_ms = all_attrs.get(METRIC_FIRST_TOKEN_DELAY, 0)
                if first_token_delay_ms > 0:
                    self.first_token_delay.labels(model=model).observe(
                        first_token_delay_ms
                    )

                chunk_delay_ms = all_attrs.get(METRIC_CHUNK_DELAY, 0)
                if chunk_delay_ms > 0:
                    self.chunk_delay.labels(model=model).observe(chunk_delay_ms)

        elif span.name == SPAN_EMBEDDING:
            self.completion_duration.labels(
                model=model, streaming="embedding", status=status
            ).observe(duration_s)

            if span.status.status_code == StatusCode.ERROR:
                self.error_total.labels(model=model, streaming="embedding").inc()
                return

            prompt_tokens = all_attrs.get(ATTR_PROMPT_TOKENS, 0)
            if prompt_tokens > 0:
                self.token_count.labels(
                    model=model, streaming="embedding", token_type="prompt"
                ).observe(prompt_tokens)

            total_tps = all_attrs.get(METRIC_TOTAL_TOKENS_PER_SECOND, 0)
            if total_tps > 0:
                self.total_tokens_per_second.labels(
                    model=model, streaming="embedding"
                ).observe(total_tps)

            embeddings_per_second = all_attrs.get(METRIC_EMBEDDINGS_PER_SECOND, 0)
            if embeddings_per_second > 0:
                self.embeddings_per_second.labels(model=model).observe(
                    embeddings_per_second
                )

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True
