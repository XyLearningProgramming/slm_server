from llama_cpp.llama_types import (
    ChatCompletionFunction,
    ChatCompletionRequestFunctionCall,
    ChatCompletionRequestMessage,
    ChatCompletionRequestResponseFormat,
    ChatCompletionTool,
    ChatCompletionToolChoiceOption,
)
from pydantic import BaseModel, Field


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


# Embeddings API Models
class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = Field(
        default=None, description="Model name, not important for our server"
    )


# OpenAI-compatible list models API
class ModelInfo(BaseModel):
    """Single model entry for GET /api/v1/models."""

    id: str = Field(description="Model identifier for use in API endpoints")
    object: str = Field(default="model", description="Object type")
    created: int = Field(description="Unix timestamp when the model was created")
    owned_by: str = Field(description="Organization that owns the model")


class ModelListResponse(BaseModel):
    """Response for GET /api/v1/models."""

    object: str = Field(default="list", description="Object type")
    data: list[ModelInfo] = Field(description="List of available models")
