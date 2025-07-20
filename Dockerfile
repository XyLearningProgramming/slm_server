# Stage 1: Dependency Installation
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS deps-builder

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --find-links https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/expanded_assets/textgen-webui

# Stage 2: Final image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Set workdir to run.
WORKDIR /app

# Copy only the virtual environment from the deps-builder stage
COPY --from=deps-builder /app/.venv /app/.venv

# Copy application source code
COPY slm_server/ /app/slm_server/
COPY scripts/ /app/scripts/

# Set PATH for python
ENV PATH="/app/.venv/bin:$PATH"

# Default port number
ENV PORT=8000

# Use start script as entrypoint
ENTRYPOINT ["/app/scripts/start.sh"]
