import time
import uuid
from typing import List, Optional, Union

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
    messages: List[ChatMessage]
    model: Optional[str] = Field(
        "Qwen3-0.6B-GGUF", description="Model name used, not important."
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)
    stream: bool = Field(False)
    wait_timeout: Optional[float] = Field(
        0, description="Max wait timeout to request sem. Default to server settings."
    )


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=generate_chat_id)
    object: str = "chat.completion"
    created: int = Field(default_factory=generate_timestamp)
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=generate_chat_id)
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=generate_timestamp)
    model: str
    choices: List[ChatCompletionStreamChoice]


# Embeddings API Models
class EmbeddingRequest(BaseModel):
    model_config = {"extra": "ignore"}
    input: Union[str, List[str]]
    model: Optional[str] = Field(
        "text-embedding-ada-002", description="Model name, not important for our server"
    )
    # encoding_format: Optional[str] = Field(
    #     "float",
    #     description="NOT IN USE FOR NOW. The format to return the embeddings in",
    # )
    # dimensions: Optional[int] = Field(
    #     None,
    #     description="NOT IN USE FOR NOW. Number of dimensions the \
    #         resulting output embeddings should have",
    # )
    # user: Optional[str] = Field(
    #     None,
    #     description="NOT IN USE FOR NOW. A unique identifier representing \
    #         your end-user",
    # )


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
