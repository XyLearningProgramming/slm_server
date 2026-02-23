from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from fastapi import FastAPI

from llama_cpp.llama_types import (
    ChatCompletionFunction,
    ChatCompletionRequestFunctionCall,
    ChatCompletionRequestMessage,
    ChatCompletionRequestResponseFormat,
    ChatCompletionTool,
    ChatCompletionToolChoiceOption,
)
from pydantic import BaseModel, ConfigDict, Field, conlist, field_validator, model_validator


# ---------------------------------------------------------------------------
# Chat completion request
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionRequestMessage] = Field(
        description="List of chat completion messages in the conversation"
    )
    functions: list[ChatCompletionFunction] | None = Field(
        default=None, description="List of functions available for the model to call"
    )
    function_call: ChatCompletionRequestFunctionCall | None = Field(
        default=None, description="Controls which function the model should call"
    )
    tools: list[ChatCompletionTool] | None = Field(
        default=None, description="List of tools available for the model to use"
    )
    tool_choice: ChatCompletionToolChoiceOption | None = Field(
        default=None, description="Controls which tool the model should use"
    )
    temperature: float = Field(
        default=0.2, description="Sampling temperature (0.0 to 2.0)"
    )
    top_p: float = Field(default=0.95, description="Nucleus sampling parameter")
    top_k: int = Field(default=40, description="Top-k sampling parameter")
    min_p: float = Field(
        default=0.05, description="Minimum probability threshold for sampling"
    )
    typical_p: float = Field(default=1.0, description="Typical sampling parameter")
    stream: bool = Field(default=False, description="Whether to stream the response")
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequences to end generation"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducible generation"
    )
    response_format: ChatCompletionRequestResponseFormat | None = Field(
        default=None, description="Response format specification"
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    presence_penalty: float = Field(
        default=0.0, description="Presence penalty (-2.0 to 2.0)"
    )
    frequency_penalty: float = Field(
        default=0.0, description="Frequency penalty (-2.0 to 2.0)"
    )
    repeat_penalty: float = Field(
        default=1.0, description="Repetition penalty (1.0 = no penalty)"
    )
    tfs_z: float = Field(default=1.0, description="Tail free sampling parameter")
    mirostat_mode: int = Field(
        default=0, description="Mirostat sampling mode (0=disabled, 1=v1, 2=v2)"
    )
    mirostat_tau: float = Field(default=5.0, description="Mirostat target entropy")
    mirostat_eta: float = Field(default=0.1, description="Mirostat learning rate")
    model: str | None = Field(default=None, description="Model identifier")
    # Cannot be properly serialized with pydantic, so we ignore it for now.
    #
    # logits_processor: LogitsProcessorList | None = Field(
    #     default=None, description="List of logits processors to apply"
    # )
    # grammar: LlamaGrammar | None = Field(
    #     default=None, description="Grammar constraints for generation"
    # )
    logit_bias: dict[int, float] | None = Field(
        default=None, description="Logit bias adjustments for specific tokens"
    )
    logprobs: bool | None = Field(
        default=None, description="Whether to return log probabilities"
    )
    top_logprobs: int | None = Field(
        default=None, description="Number of top log probabilities to return"
    )

    @field_validator("messages", mode="before")
    @classmethod
    def _normalize_assistant_content(cls, v: Any) -> Any:
        """Allow ``content: null`` on assistant messages (OpenAI spec compat).

        llama-cpp's TypedDict defines ``content: NotRequired[str]`` which
        rejects ``None`` when the key is present.  The OpenAI spec allows
        ``null`` on assistant messages that carry ``tool_calls``, and
        langchain-openai sends it that way, so we normalise here.
        """
        if isinstance(v, list):
            for msg in v:
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and "content" in msg
                    and msg["content"] is None
                ):
                    msg["content"] = ""
        return v

    @model_validator(mode="after")
    def _default_tool_choice_auto(self) -> Self:
        """Match OpenAI: default to "auto" when tools are present."""
        if self.tools and self.tool_choice is None:
            self.tool_choice = "auto"
        return self


# ---------------------------------------------------------------------------
# Shared field types — aligned with llama_cpp.llama_types
# ---------------------------------------------------------------------------

# ChatCompletionResponseMessage.role (non-streaming response)
MessageRole = Literal["assistant", "function"]
# ChatCompletionStreamResponseDelta.role (streaming delta)
DeltaRole = Literal["system", "user", "assistant", "tool"]
# ChatCompletionStreamResponseChoice.finish_reason (both paths; our
# postprocessor can also set "tool_calls")
FinishReason = Literal["stop", "length", "tool_calls", "function_call"]


# ---------------------------------------------------------------------------
# Chat completion response (non-streaming)
# ---------------------------------------------------------------------------


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: MessageRole
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Any | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "chatcmpl_MInMfQVLOPebdbjF",
                    "object": "chat.completion",
                    "created": 1771828648,
                    "model": "Qwen3-0.6B-Q4_K_M",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hi there!",
                                "reasoning_content": "The user wants a greeting.",
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 14,
                        "completion_tokens": 149,
                        "total_tokens": 163,
                    },
                },
                {
                    "id": "chatcmpl_6HEgD7zMer6W3ob8",
                    "object": "chat.completion",
                    "created": 1771828660,
                    "model": "Qwen3-0.6B-Q4_K_M",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_2HWeVAmARMukopsB",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": (
                                                '{"location": "San Francisco"}'
                                            ),
                                        },
                                    }
                                ],
                            },
                            "logprobs": None,
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 166,
                        "completion_tokens": 24,
                        "total_tokens": 190,
                    },
                },
            ]
        }
    )

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


# ---------------------------------------------------------------------------
# Chat completion response (streaming chunks)
# ---------------------------------------------------------------------------


class DeltaToolCall(BaseModel):
    """Streaming tool call emitted by ``StreamPostProcessor``.

    Our postprocessor accumulates the full ``<tool_call>`` block before
    emitting, so every field is always present (never partial/incremental).
    ``index`` is the ordinal position of this tool call in the response.
    """

    index: int
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatDelta(BaseModel):
    role: DeltaRole | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatDelta
    logprobs: Any | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionChunkResponse(BaseModel):
    """Schema for each SSE ``data:`` payload in a streaming chat completion.

    Not used for runtime validation (chunks are pre-serialised dicts),
    but exposed here so OpenAPI / Swagger documents the streaming format.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "chatcmpl_vGxUUAi7KYLIEaNGb",
                    "object": "chat.completion.chunk",
                    "created": 1771828652,
                    "model": "Qwen3-0.6B-Q4_K_M",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_vGxUUAi7KYLIEaNGb",
                    "object": "chat.completion.chunk",
                    "created": 1771828652,
                    "model": "Qwen3-0.6B-Q4_K_M",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "Let me think."},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_vGxUUAi7KYLIEaNGb",
                    "object": "chat.completion.chunk",
                    "created": 1771828652,
                    "model": "Qwen3-0.6B-Q4_K_M",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "2 + 2 equals **4**."},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_uXmbuefsXElrLqSkb",
                    "object": "chat.completion.chunk",
                    "created": 1771828662,
                    "model": "Qwen3-0.6B-Q4_K_M",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1MSB4PoE3eerSEJu",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": (
                                                '{"location": "San Francisco"}'
                                            ),
                                        },
                                    }
                                ]
                            },
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_vGxUUAi7KYLIEaNGb",
                    "object": "chat.completion.chunk",
                    "created": 1771828652,
                    "model": "Qwen3-0.6B-Q4_K_M",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                },
            ]
        }
    )

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# ---------------------------------------------------------------------------
# Embeddings API
# ---------------------------------------------------------------------------

MAX_EMBEDDING_INPUTS = 2048


class EmbeddingRequest(BaseModel):
    input: str | conlist(str, max_length=MAX_EMBEDDING_INPUTS)
    model: str | None = Field(default=None, description="Model identifier")


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [
                                -0.046052,
                                0.028006,
                                0.014284,
                                0.025734,
                                -0.034211,
                            ],
                            "index": 0,
                        }
                    ],
                    "model": "all-MiniLM-L6-v2",
                }
            ]
        }
    )

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str


# ---------------------------------------------------------------------------
# List models API
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """Single model entry for GET /api/v1/models."""

    id: str = Field(description="Model identifier for use in API endpoints")
    object: Literal["model"] = "model"
    created: int = Field(description="Unix timestamp when the model was created")
    owned_by: str = Field(description="Organization that owns the model")


class ModelListResponse(BaseModel):
    """Response for GET /api/v1/models."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen3-0.6B-Q4_K_M",
                            "object": "model",
                            "created": 1771811606,
                            "owned_by": "second-state",
                        },
                        {
                            "id": "all-MiniLM-L6-v2",
                            "object": "model",
                            "created": 1771811614,
                            "owned_by": "sentence-transformers",
                        },
                    ],
                }
            ]
        }
    )

    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(description="List of available models")


# ---------------------------------------------------------------------------
# OpenAPI helpers
# ---------------------------------------------------------------------------


def register_streaming_schema(app: FastAPI) -> None:
    """Register ``ChatCompletionChunkResponse`` in OpenAPI ``components.schemas``.

    Pydantic's ``model_json_schema()`` puts nested models under a local
    ``$defs`` key, but Swagger UI resolves ``$ref`` from the document root.
    This patches ``app.openapi`` to hoist the chunk schema and its
    dependencies into ``components.schemas`` with correct ``$ref`` paths.
    """
    _original = app.openapi

    def patched_openapi():  # type: ignore[no-untyped-def]
        if app.openapi_schema:
            return app.openapi_schema

        schema = _original()
        chunk_schema = ChatCompletionChunkResponse.model_json_schema(
            ref_template="#/components/schemas/{model}",
        )
        defs = chunk_schema.pop("$defs", {})
        schemas = schema.setdefault("components", {}).setdefault("schemas", {})
        schemas["ChatCompletionChunkResponse"] = chunk_schema
        for name, defn in defs.items():
            schemas.setdefault(name, defn)

        app.openapi_schema = schema
        return schema

    app.openapi = patched_openapi  # type: ignore[method-assign]
