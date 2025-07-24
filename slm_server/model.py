import time
import uuid

from pydantic import BaseModel, Field


def generate_chat_id():
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_embedding_id():
    return f"embedding-{uuid.uuid4().hex}"


def generate_timestamp():
    return int(time.time())


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = Field(
        "Qwen3-0.6B-GGUF", description="Model name used, not important."
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)
    stream: bool = Field(False)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=generate_chat_id)
    object: str = "chat.completion"
    created: int = Field(default_factory=generate_timestamp)
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=generate_chat_id)
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=generate_timestamp)
    model: str
    choices: list[ChatCompletionStreamChoice]


EmbeddingInput = str | list[str] | list[int] | list[list[int]]


# Embeddings API Models
class EmbeddingRequest(BaseModel):
    input: EmbeddingInput
    model: str | None = Field(
        "text-embedding-ada-002", description="Model name, not important for our server"
    )
    encoding_format: str | None = Field(
        None, description="Encoding format for embeddings"
    )


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float] | list[list[float]]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
