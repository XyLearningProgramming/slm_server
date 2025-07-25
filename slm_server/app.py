import asyncio
import json
import traceback
from http import HTTPStatus
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import CreateChatCompletionStreamResponse, Llama

from slm_server.config import Settings, get_settings
from slm_server.logging import setup_logging
from slm_server.metrics import setup_metrics
from slm_server.model import (
    ChatCompletionRequest,
    EmbeddingRequest,
)
from slm_server.trace import setup_tracing
from slm_server.utils import (
    set_atrribute_response,
    set_atrribute_response_stream,
    set_attribute_cancelled,
    set_attribute_response_embedding,
    slm_embedding_span,
    slm_span,
)

# MAX_CONCURRENCY decides how many threads are calling model.
# Default to 1 since llama cpp is designed to be at most efficiency
# for single thread. Meanwhile, value larger than 1 allows
# threads to compete for same resources.
MAX_CONCURRENCY = 1
# Default timeout message in detail field.
DETAIL_SEM_TIMEOUT = "Server is busy, please try again later."
# Status code for semaphore timeout.
STATUS_CODE_SEM_TIMEOUT = HTTPStatus.REQUEST_TIMEOUT
# Status code for unexpected errors.
# This is used when the server encounters an error that is not handled
STATUS_CODE_EXCEPTION = HTTPStatus.INTERNAL_SERVER_ERROR


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
            logits_all=False,
            embedding=True,
            use_mlock=True,  # Use mlock to prevent memory swapping
            use_mmap=True,  # Use memory-mapped files for faster access
            chat_format="chatml-function-calling",
        )
    return get_llm._instance


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
) -> AsyncGenerator[None, None]:
    """Context manager to acquire and release the LLM semaphore with a timeout."""
    try:
        await asyncio.wait_for(sem.acquire(), settings.s_timeout)
        yield None
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=STATUS_CODE_SEM_TIMEOUT, detail=DETAIL_SEM_TIMEOUT
        )
    finally:
        if sem.locked():
            sem.release()


async def run_llm_streaming(
    llm: Llama, req: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generator that runs the LLM and yields SSE chunks under lock."""
    with slm_span(req, is_streaming=True) as span:
        try:
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

            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            # Handle cancellation gracefully during sse.
            set_attribute_cancelled(span)


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
):
    """
    Generates a chat completion, handling both streaming and non-streaming cases.
    Concurrency is managed by the `locked_llm_session` context manager.
    """
    try:
        if req.stream:
            return StreamingResponse(
                run_llm_streaming(llm, req), media_type="text/event-stream"
            )
        else:
            return await run_llm_non_streaming(llm, req)
    except Exception:
        # Catch any other unexpected errors
        error_str = traceback.format_exc()
        raise HTTPException(status_code=STATUS_CODE_EXCEPTION, detail=error_str)


@app.post("/api/v1/embeddings")
async def create_embeddings(
    req: EmbeddingRequest,
    llm: Annotated[Llama, Depends(get_llm)],
    _: Annotated[None, Depends(lock_llm_semaphor)],
):
    """Create embeddings for the given input text(s)."""
    try:
        with slm_embedding_span(req) as span:
            # Use llama-cpp-python's create_embedding method directly
            embedding_result = await asyncio.to_thread(
                llm.create_embedding,
                **req.model_dump(),
            )
            # Convert llama-cpp response using model_validate like chat completion
            set_attribute_response_embedding(span, embedding_result)
            return embedding_result
    except Exception:
        error_str = traceback.format_exc()
        raise HTTPException(status_code=STATUS_CODE_EXCEPTION, detail=error_str)


@app.get("/health")
async def health():
    return "ok"
