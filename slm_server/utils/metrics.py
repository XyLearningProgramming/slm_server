from opentelemetry.sdk.trace import Span

from .constants import (
    ATTR_CHUNK_COUNT,
    ATTR_COMPLETION_TOKENS,
    ATTR_OUTPUT_COUNT,
    ATTR_STREAMING,
    ATTR_TOTAL_TOKENS,
    EVENT_ATTR_CHUNK_CONTENT_SIZE,
    EVENT_ATTR_CHUNK_SIZE,
    EVENT_CHUNK_GENERATED,
    METRIC_AVG_CHUNK_CONTENT_SIZE,
    METRIC_AVG_CHUNK_SIZE,
    METRIC_CHUNK_DELAY,
    METRIC_CHUNKS_WITH_CONTENT,
    METRIC_EMBEDDINGS_PER_SECOND,
    METRIC_EMPTY_CHUNKS,
    METRIC_FIRST_TOKEN_DELAY,
    METRIC_MAX_CHUNK_SIZE,
    METRIC_MIN_CHUNK_SIZE,
    METRIC_TOKENS_PER_SECOND,
    METRIC_TOTAL_DURATION,
    METRIC_TOTAL_TOKENS_PER_SECOND,
    SPAN_EMBEDDING,
)


def calculate_performance_metrics(span: Span):  # noqa: C901
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

    elif span.name == SPAN_EMBEDDING:
        output_count = attrs.get(ATTR_OUTPUT_COUNT, 0)
        if output_count > 0 and duration_ms > 0:
            metrics[METRIC_EMBEDDINGS_PER_SECOND] = output_count / (duration_ms / 1000)

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
