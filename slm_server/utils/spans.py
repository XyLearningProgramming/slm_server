import logging
import traceback
from contextlib import contextmanager

from llama_cpp import ChatCompletionStreamResponse
from opentelemetry import trace
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import Status, StatusCode

from llama_cpp.llama_types import (
    CreateChatCompletionResponse as ChatCompletionResponse,
    CreateEmbeddingResponse as EmbeddingResponse,
)
from slm_server.model import (
    ChatCompletionRequest,
    EmbeddingRequest,
)

from .constants import (
    ATTR_CHUNK_COUNT,
    ATTR_COMPLETION_TOKENS,
    ATTR_FORCE_SAMPLE,
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
    EVENT_ATTR_CHUNK_CONTENT_SIZE,
    EVENT_ATTR_CHUNK_SIZE,
    EVENT_CHUNK_GENERATED,
    MODEL_NAME,
    SPAN_CHAT_COMPLETION,
    SPAN_EMBEDDING,
)

# Get tracer
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def set_atrribute_response(span: Span, response: ChatCompletionResponse | dict):
    """Set response attributes automatically."""
    # Non-streaming response - handle both dict and object responses
    if isinstance(response, dict):
        # Handle dict response
        usage = response.get("usage")
        if usage:
            span.set_attribute(ATTR_PROMPT_TOKENS, usage.get("prompt_tokens", 0))
            span.set_attribute(
                ATTR_COMPLETION_TOKENS, usage.get("completion_tokens", 0)
            )
            span.set_attribute(ATTR_TOTAL_TOKENS, usage.get("total_tokens", 0))

        choices = response.get("choices", [])
        if choices and choices[0].get("message"):
            content = choices[0]["message"].get("content") or ""
            span.set_attribute(ATTR_OUTPUT_CONTENT_LENGTH, len(content))
    else:
        # Handle object response (original code)
        if response.usage:
            span.set_attribute(ATTR_PROMPT_TOKENS, response.usage.prompt_tokens)
            span.set_attribute(ATTR_COMPLETION_TOKENS, response.usage.completion_tokens)
            span.set_attribute(ATTR_TOTAL_TOKENS, response.usage.total_tokens)

        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""
            span.set_attribute(ATTR_OUTPUT_CONTENT_LENGTH, len(content))


def set_atrribute_response_stream(
    span: Span, response: ChatCompletionStreamResponse | dict
):
    """Record streaming chunk as an event and accumulate tokens."""
    chunk_content = ""
    if isinstance(response, dict):
        # Handle dict response
        choices = response.get("choices", [])
        if choices and choices[0].get("delta") and choices[0]["delta"].get("content"):
            chunk_content = choices[0]["delta"]["content"]
        chunk_json = str(response)  # Simple string representation for dict
    else:
        # Handle object response (original code)
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

    # Accumulate tokens directly on the span (only for recording spans)
    if span.is_recording():
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


def set_attribute_response_embedding(span: Span, response: EmbeddingResponse | dict):
    """Set embedding response attributes automatically."""
    if isinstance(response, dict):
        # Handle dict response
        usage = response.get("usage")
        if usage:
            span.set_attribute(ATTR_PROMPT_TOKENS, usage.get("prompt_tokens", 0))
            span.set_attribute(ATTR_TOTAL_TOKENS, usage.get("total_tokens", 0))
        data = response.get("data")
        if data:
            span.set_attribute(ATTR_OUTPUT_COUNT, len(data))
    else:
        # Handle object response (original code)
        if response.usage:
            span.set_attribute(ATTR_PROMPT_TOKENS, response.usage.prompt_tokens)
            span.set_attribute(ATTR_TOTAL_TOKENS, response.usage.total_tokens)
        if response.data:
            span.set_attribute(ATTR_OUTPUT_COUNT, len(response.data))


def set_attribute_cancelled(span: Span, reason: str = "client disconnected"):
    """Set span status to error for cancellation."""
    span.set_status(Status(StatusCode.ERROR, description=reason))


@contextmanager
def slm_span(req: ChatCompletionRequest, is_streaming: bool):
    """Create SLM span with automatic timing and error handling."""
    span_name = (
        f"{SPAN_CHAT_COMPLETION}.{'streaming' if is_streaming else 'non_streaming'}"
    )

    # Pre-calculate attributes before starting span
    messages_for_llm = req.messages
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
        # Estimate prompt tokens for streaming
        # (rough approximation: 1 token per 4 chars)
        estimated_prompt_tokens = (
            max(1, input_content_length // 4) if is_streaming else 0
        )
        initial_attributes[ATTR_PROMPT_TOKENS] = estimated_prompt_tokens

    with tracer.start_as_current_span(span_name, attributes=initial_attributes) as span:
        try:
            yield span

        except Exception:
            # Use native error handling
            error_str = traceback.format_exc()
            span.set_status(Status(StatusCode.ERROR, error_str))
            span.set_attribute(ATTR_FORCE_SAMPLE, True)
            raise


@contextmanager
def slm_embedding_span(req: EmbeddingRequest):
    """Create SLM span for embedding requests."""
    span_name = SPAN_EMBEDDING

    if isinstance(req.input, list):
        input_count = len(req.input)
        input_content_length = sum(len(text) for text in req.input)
    else:
        input_count = 1
        input_content_length = len(req.input)

    initial_attributes = {
        ATTR_MODEL: MODEL_NAME,
        ATTR_INPUT_COUNT: input_count,
        ATTR_INPUT_CONTENT_LENGTH: input_content_length,
    }

    with tracer.start_as_current_span(span_name, attributes=initial_attributes) as span:
        try:
            yield span

        except Exception:
            error_str = traceback.format_exc()
            span.set_status(Status(StatusCode.ERROR, error_str))
            span.set_attribute(ATTR_FORCE_SAMPLE, True)
            raise
