# Constants for span naming and attributes
MODEL_NAME = "llama-cpp"
SPAN_PREFIX = "slm"

# Span names
SPAN_CHAT_COMPLETION = f"{SPAN_PREFIX}.chat_completion"
SPAN_EMBEDDING = f"{SPAN_PREFIX}.embedding"

# Event names
EVENT_CHUNK_GENERATED = f"{SPAN_PREFIX}.chunk_generated"

# Event attribute names
EVENT_ATTR_CHUNK_SIZE = f"{SPAN_PREFIX}.chunk_size"
EVENT_ATTR_CHUNK_CONTENT_SIZE = f"{SPAN_PREFIX}.chunk_content_size"

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

# Embedding attributes
ATTR_INPUT_COUNT = f"{SPAN_PREFIX}.input.count"
ATTR_OUTPUT_COUNT = f"{SPAN_PREFIX}.output.count"

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

# Embedding metrics
METRIC_EMBEDDINGS_PER_SECOND = f"{SPAN_PREFIX}.metrics.embeddings_per_second"

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

# Embedding log keys
LOG_KEY_INPUT_COUNT = "input_count"
LOG_KEY_OUTPUT_COUNT = "output_count"
LOG_KEY_EMBEDDINGS_PER_SECOND = "embeddings_per_second"

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

# Embedding metrics
PROMETHEUS_EMBEDDINGS_PER_SECOND = "slm_embeddings_per_second"
PROMETHEUS_EMBEDDINGS_PER_SECOND_DESC = "Embeddings generated per second"

# Log message templates
LOG_MSG_STARTING_CALL = "[SLM] starting {}: {}"
LOG_MSG_COMPLETED_CALL = "[SLM] completed {}: {}"
LOG_MSG_FAILED_CALL = "[SLM] failed: {}"