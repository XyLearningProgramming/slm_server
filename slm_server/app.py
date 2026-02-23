import asyncio
import json
import traceback
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, AsyncGenerator, Generator, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import CreateChatCompletionStreamResponse, Llama

from slm_server.config import Settings, get_settings
from slm_server.embedding import OnnxEmbeddingModel
from slm_server.logging import setup_logging
from slm_server.metrics import setup_metrics
from slm_server.model import (
    ChatCompletionRequest,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ModelListResponse,
)
from slm_server.trace import setup_tracing
from slm_server.utils import (
    set_atrribute_response,
    set_atrribute_response_stream,
    set_attribute_response_embedding,
    slm_embedding_span,
    slm_span,
)

# MAX_CONCURRENCY decides how many threads are calling model.
# Default to 1 since llama cpp is designed to be at most efficiency
# for single thread. Meanwhile, value larger than 1 allows
# threads to compete for same resources.
MAX_CONCURRENCY = 1
# Keeps function calling and also compatible with ReAct agents.
CHAT_FORMAT = "chatml-function-calling"
# Default timeout message in detail field.
DETAIL_SEM_TIMEOUT = "Server is busy, please try again later."
# Status code for semaphore timeout.
STATUS_CODE_SEM_TIMEOUT = HTTPStatus.REQUEST_TIMEOUT
# Status code for unexpected errors.
# This is used when the server encounters an error that is not handled
STATUS_CODE_EXCEPTION = HTTPStatus.INTERNAL_SERVER_ERROR
# Media type for streaming responses.
STREAM_RESPONSE_MEDIA_TYPE = "text/event-stream"


def get_llm_semaphor() -> asyncio.Semaphore:
    if not hasattr(get_llm_semaphor, "_instance"):
        get_llm_semaphor._instance = asyncio.Semaphore(MAX_CONCURRENCY)
    return get_llm_semaphor._instance


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> Llama:
    if not hasattr(get_llm, "_instance"):
        get_llm._instance = Llama(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            n_batch=settings.n_batch,
            verbose=settings.logging.verbose,
            seed=settings.seed,
            chat_format=CHAT_FORMAT,
            logits_all=False,
            embedding=False,
            use_mlock=True,
            use_mmap=True,
        )
    return get_llm._instance


def get_embedding_model(
    settings: Annotated[Settings, Depends(get_settings)],
) -> OnnxEmbeddingModel:
    if not hasattr(get_embedding_model, "_instance"):
        get_embedding_model._instance = OnnxEmbeddingModel(settings.embedding)
    return get_embedding_model._instance


def get_app() -> FastAPI:
    # Get settings when creating app.
    settings = get_settings()
    # Set up trace and logging if enabled.
    setup_logging(settings.logging)
    app = FastAPI(
        title="OpenAI-compatible SLM Server",
        description="A simple API server for serving a Small Language Model, "
        + "compatible with the OpenAI Chat Completions format.",
    )
    # Setup metrics if enabled
    setup_metrics(app, settings.metrics)

    # Setup trace and OTel metrics (this will also instrument FastAPI)
    setup_tracing(app, settings.tracing)

    return app


# Default app.
app = get_app()


async def lock_llm_semaphor(
    sem: Annotated[asyncio.Semaphore, Depends(get_llm_semaphor)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[Literal[True], None]:
    """Context manager to acquire and release the LLM semaphore with a timeout."""
    try:
        await asyncio.wait_for(sem.acquire(), settings.s_timeout)
        yield True
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=STATUS_CODE_SEM_TIMEOUT, detail=DETAIL_SEM_TIMEOUT
        )
    finally:
        if sem.locked():
            sem.release()


def raise_as_http_exception() -> Generator[Literal[True], None, None]:
    """Capture exception with stack trace in details."""
    try:
        yield True
    except Exception:
        error_str = traceback.format_exc()
        raise HTTPException(status_code=STATUS_CODE_EXCEPTION, detail=error_str)


async def run_llm_streaming(
    llm: Llama, req: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generator that runs the LLM and yields SSE chunks under lock."""
    with slm_span(req, is_streaming=True) as span:
        completion_stream = await asyncio.to_thread(
            llm.create_chat_completion,
            **req.model_dump(),
        )

        # Use traced iterator that automatically handles chunk spans
        # and parent span updates
        chunk: CreateChatCompletionStreamResponse
        for chunk in completion_stream:
            set_atrribute_response_stream(span, chunk)
            yield f"data: {json.dumps(chunk)}\n\n"
            # NOTE: This is a workaround to yield control back to the event loop
            # to allow checking for socket after yield and pop in CancelledError.
            # Ref: https://github.com/encode/starlette/discussions/1776#discussioncomment-3207518
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"


async def run_llm_non_streaming(llm: Llama, req: ChatCompletionRequest):
    """Runs the LLM for a non-streaming request under lock."""
    with slm_span(req, is_streaming=False) as span:
        completion_result = await asyncio.to_thread(
            llm.create_chat_completion,
            **req.model_dump(),
        )
        set_atrribute_response(span, completion_result)

        return completion_result


@app.post("/api/v1/chat/completions")
async def create_chat_completion(
    req: ChatCompletionRequest,
    llm: Annotated[Llama, Depends(get_llm)],
    _: Annotated[None, Depends(lock_llm_semaphor)],
    __: Annotated[None, Depends(raise_as_http_exception)],
):
    """
    Generates a chat completion, handling both streaming and non-streaming cases.
    Concurrency is managed by the `locked_llm_session` context manager.
    """
    if req.stream:
        return StreamingResponse(
            run_llm_streaming(llm, req), media_type=STREAM_RESPONSE_MEDIA_TYPE
        )
    else:
        return await run_llm_non_streaming(llm, req)


@app.post("/api/v1/embeddings")
async def create_embeddings(
    req: EmbeddingRequest,
    emb_model: Annotated[OnnxEmbeddingModel, Depends(get_embedding_model)],
    _: Annotated[None, Depends(lock_llm_semaphor)],
    __: Annotated[None, Depends(raise_as_http_exception)],
):
    """Create embeddings using the dedicated ONNX embedding model."""
    with slm_embedding_span(req) as span:
        inputs = req.input if isinstance(req.input, list) else [req.input]
        vectors = await asyncio.to_thread(emb_model.encode, inputs, True)
        result = EmbeddingResponse(
            data=[
                EmbeddingData(embedding=vec.tolist(), index=i)
                for i, vec in enumerate(vectors)
            ],
            model=emb_model.model_id,
        )
        set_attribute_response_embedding(span, result)
        return result


@app.get("/api/v1/models", response_model=ModelListResponse)
async def list_models(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelListResponse:
    """List available models (OpenAI-compatible)."""
    chat_model_id = Path(settings.model_path).stem
    try:
        chat_created = int(Path(settings.model_path).stat().st_mtime)
    except (OSError, ValueError):
        chat_created = 0

    try:
        emb_created = int(Path(settings.embedding.onnx_path).stat().st_mtime)
    except (OSError, ValueError):
        emb_created = 0

    return ModelListResponse(
        data=[
            ModelInfo(
                id=chat_model_id,
                created=chat_created,
                owned_by=settings.model_owner,
            ),
            ModelInfo(
                id=settings.embedding.model_id,
                created=emb_created,
                owned_by="sentence-transformers",
            ),
        ],
    )


@app.get("/health")
async def health():
    return "ok"
