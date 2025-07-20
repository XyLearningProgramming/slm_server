import asyncio
import traceback
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

from slm_server.config import Settings, get_settings
from slm_server.logging import setup_logging
from slm_server.metrics import setup_metrics
from slm_server.model import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)
from slm_server.trace import setup_tracing
from slm_server.utils import (
    set_atrribute_response,
    set_atrribute_response_stream,
    slm_span,
)

# MAX_CONCURRENCY decides how many threads are calling model.
# Default to 1 since llama cpp is designed to be at most efficiency
# for single thread. Meanwhile, value larger than 1 allows
# threads to compete for same resources.
MAX_CONCURRENCY = 1
# Default timeout message in detail field.
DETAIL_SEM_TIMEOUT = "Server is busy, please try again later."


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
            seed=settings.seed,
            logits_all=False,
            embedding=False,
            verbose=settings.logging.verbose,
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
    req: ChatCompletionRequest,
    sem: Annotated[asyncio.Semaphore, Depends(get_llm_semaphor)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[None, None]:
    """Context manager to acquire and release the LLM semaphore with a timeout."""
    try:
        await asyncio.wait_for(
            sem.acquire(), timeout=req.wait_timeout or settings.s_timeout
        )
        yield None
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail=DETAIL_SEM_TIMEOUT)
    finally:
        if sem.locked():
            sem.release()


async def run_llm_streaming(
    llm: Llama, req: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generator that runs the LLM and yields SSE chunks under lock."""
    with slm_span(req, is_streaming=True) as (span, messages_for_llm):
        completion_stream = await asyncio.to_thread(
            llm.create_chat_completion,
            messages=messages_for_llm,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=True,
        )

        # Use traced iterator that automatically handles chunk spans and parent span updates
        for chunk in completion_stream:
            response_model = ChatCompletionStreamResponse.model_validate(chunk)
            set_atrribute_response_stream(span, response_model)
            yield f"data: {response_model.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"


async def run_llm_non_streaming(
    llm: Llama, req: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Runs the LLM for a non-streaming request under lock."""
    with slm_span(req, is_streaming=False) as (span, messages_for_llm):
        completion_result = await asyncio.to_thread(
            llm.create_chat_completion,
            messages=messages_for_llm,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=False,
        )

        response_model = ChatCompletionResponse.model_validate(completion_result)
        set_atrribute_response(span, response_model)

        return response_model


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
        raise HTTPException(status_code=500, detail=error_str)


@app.get("/health")
async def health():
    return "ok"
