# Small Language Model Server

[![CI Pipeline](https://github.com/XyLearningProgramming/slm_server/actions/workflows/ci.yml/badge.svg)](https://github.com/XyLearningProgramming/slm_server/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/XyLearningProgramming/slm_server/branch/main/graph/badge.svg)](https://codecov.io/gh/XyLearningProgramming/slm_server)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/x3huang/slm_server)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A lightweight model server that serves small language models (default: Qwen3-0.6B-GGUF) as a thin wrapper around llama-cpp with OpenAI-compatible `/chat/completions` API. Core logic is <100 lines in `./slm_server/app.py`.

## Features

- **OpenAI-compatible API** - Drop-in replacement with `/chat/completions` endpoint and streaming support
- **Llama.cpp integration** - High-performance inference optimized for limited CPU and memory resources
- **Production observability** - Built-in logging, Prometheus metrics, and OpenTelemetry tracing
- **Enterprise deployment** - Complete CI/CD pipeline with unit tests, e2e tests, Helm charts, and Docker support
- **Simple configuration** - Environment-based config with sensible defaults

## Use Cases

- **Self-hosting** - Deploy small models under resource constraints
- **Privacy-first inference** - No user content logging, complete data control
- **Development environments** - Local LLM testing and prototyping
- **Edge deployments** - Lightweight inference in constrained environments
- **API standardization** - Unified OpenAI-compatible interface for small models

## Quick Start

### Local Development

```bash
# Download model
./scripts/download.sh  # Downloads default Qwen3-0.6B-GGUF

# Install and start
uv sync
./scripts/start.sh
```

### Docker

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models x3huang/slm_server/general
```

### Test API

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Observability

All observability components are configurable and enabled by default:

- **Structured Logging** - Request lifecycle logging with trace correlation
- **Prometheus Metrics** - Available at `/metrics` (latency, throughput, token rates, memory usage)
- **OpenTelemetry Tracing** - Distributed tracing with request flow visualization

## Model Choice

Default model: **Qwen3-0.6B-Q4_K_M** (484 MB) from [`second-state/Qwen3-0.6B-GGUF`](https://huggingface.co/second-state/Qwen3-0.6B-GGUF).

Previously the default was Qwen3-0.6B-Q8_0 (805 MB) from the [official Qwen repo](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF). The switch to Q4_K_M was made to better fit deployment on resource-constrained VPS nodes (1 CPU / 1 GB RAM each).

### Why Qwen3-0.6B

0.6B parameters is the largest Qwen3 tier that fits on a 1 GB node. The next step up (Qwen3-1.7B) requires ~1 GB+ for model weights alone at even aggressive quantization, leaving nothing for the OS, kubelet, or KV cache.

### Why Q4_K_M over Q8_0

| | Q8_0 | Q4_K_M |
|---|---|---|
| File size | 805 MB | 484 MB |
| Est. RAM (with `use_mlock`, 4096 ctx) | ~750 MB | ~550 MB |
| Quality vs F16 | ~99.9% | ~99% |
| Inference speed (CPU) | Slower (more data through cache) | **~40-50% faster** |

For a 0.6B model the quality bottleneck is parameter count, not quantization precision -- the difference between Q4 and Q8 is negligible in practice. Q4_K_M ("K_M" = mixed precision on important layers) is the community-recommended sweet spot for balanced quality and performance.

The RAM savings (~200 MB) are significant on a 1 GB node: the pod's memory request drops from ~750 Mi to ~600 Mi, leaving headroom for the OS and co-located workloads.

### Resource estimates

Current Helm resource settings (`deploy/helm/values.yaml`):

| Setting | Value | Rationale |
|---|---|---|
| Memory request | 600 Mi | Steady-state with model locked in RAM via `use_mlock` |
| Memory limit | 700 Mi | ~100 Mi headroom over steady-state |
| CPU request | 200 m | Meaningful reservation for inference on 1-core VPS |
| CPU limit | 1 | Matches physical core count |

### Switching models

To use a different quantization, update `scripts/download.sh` and set `SLM_MODEL_PATH`:

```bash
# In .env or as environment variable
SLM_MODEL_PATH=/app/models/Qwen3-0.6B-Q8_0.gguf
```

Available quantizations at [`second-state/Qwen3-0.6B-GGUF`](https://huggingface.co/second-state/Qwen3-0.6B-GGUF): Q2_K (347 MB) through F16 (1.51 GB).

## Configuration

Configure via environment variables (prefix: `SLM_`) or `.env` file. See [`./slm_server/config.py`](./slm_server/config.py) for all options.

## Deployment

### Kubernetes with Helm

```bash
helm upgrade --install slm-server ./deploy/helm \
  --namespace backend \
  --values ./deploy/helm/values.yaml
```

### Docker Compose

```yaml
version: '3.8'
services:
  slm-server:
    image: x3huang/slm_server:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - slm_server_PATH=/app/models/your-model.gguf
```

## Development

### Testing

```bash
# Unit tests
uv run pytest tests/ --ignore=tests/e2e/

# End-to-end tests
uv run python ./tests/e2e/main.py

# With coverage
uv run pytest tests/ --ignore=tests/e2e/ --cov=slm_server --cov-report=html
```

### Code Quality

```bash
uv run ruff check .
uv run ruff format .
```

## API Documentation

- **Interactive docs**: http://localhost:8000/docs
- **OpenAPI spec**: http://localhost:8000/openapi.json
- **Health check**: http://localhost:8000/health

## License

MIT License - see [LICENSE](LICENSE) file for details.