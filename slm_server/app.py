# model-server/app.py
import time
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama

from slm_server.config import settings

# --- Global Objects ---

llm = Llama(
    model_path=settings.model_path,
    n_ctx=settings.n_ctx,
    n_threads=settings.n_threads,
    seed=settings.seed,
    logits_all=False,
    embedding=False,
)

llm_semaphore = asyncio.Semaphore(1)

app = FastAPI(
    title="OpenAI-compatible SLM Server",
    description="A simple API server for serving a Small Language Model, compatible with the OpenAI Chat Completions format.",
)

# --- Pydantic Models (Unchanged) ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "Qwen3-0.6B-GGUF"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(256, gt=0)
    stream: bool = Field(False)

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
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
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


# --- Centralized Concurrency Control ---

@asynccontextmanager
async def locked_llm_session() -> AsyncGenerator[None, None]:
    """Context manager to acquire and release the LLM semaphore with a timeout."""
    try:
        await asyncio.wait_for(llm_semaphore.acquire(), timeout=settings.s_timeout)
        yield
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy, please try again later.")
    finally:
        if llm_semaphore.locked():
            llm_semaphore.release()


# --- API Endpoint and Streaming Logic ---

async def run_llm_streaming(req: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generator that runs the LLM and yields SSE chunks under lock."""
    async with locked_llm_session():
        messages_for_llm = [msg.dict() for msg in req.messages]
        completion_stream = await asyncio.to_thread(
            llm.create_chat_completion,
            messages=messages_for_llm,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=True,
        )
        for chunk in completion_stream:
            response_model = ChatCompletionStreamResponse.parse_obj(chunk)
            yield f"data: {response_model.json()}\n\n"
        yield "data: [DONE]\n\n"

async def run_llm_non_streaming(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """Runs the LLM for a non-streaming request under lock."""
    async with locked_llm_session():
        messages_for_llm = [msg.dict() for msg in req.messages]
        completion_result = await asyncio.to_thread(
            llm.create_chat_completion,
            messages=messages_for_llm,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=False,
        )
        return ChatCompletionResponse.parse_obj(completion_result)


@app.post("/api/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest):
    """
    Generates a chat completion, handling both streaming and non-streaming cases.
    Concurrency is managed by the `locked_llm_session` context manager.
    """
    try:
        if req.stream:
            return StreamingResponse(
                run_llm_streaming(req),
                media_type="text/event-stream"
            )
        else:
            return await run_llm_non_streaming(req)
    except HTTPException as e:
        # Re-raise HTTP exceptions directly
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=str(e))