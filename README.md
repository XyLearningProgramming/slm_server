# SLM Server

A FastAPI-based model server that serves small language models (by default `Qwen3-0.6B-GGUF`) via API with simple configuration and full performance statistics without user content records.

## Features

- **OpenAI-compatible API** - Chat completions endpoint with streaming support
- **Llama.cpp integration** - Efficient inference using llama-cpp-python for speed under limited cpu and mem resources
- **Observability for Prod** - Built-in logging, metrics (Prometheus), and tracing (OpenTelemetry), all toggle-able.
- **CI&CD** - Includes unittest, end-2-end test, Helm chart and Docker support

## Quick Start

1. Place your GGUF model in the `models/` directory, or use `./scripts/download.sh`; note that `.from_pretrained` provided by HuggingFace is not used because I think under no circumstances model can be pulled at place in prod env.

2. Configure via environment variables (prefix: `SLM_`) or via `.env`, see `./slm_server/config.py` for details.

3. Run: `./scripts/start.sh`
