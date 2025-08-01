#!/bin/bash

set -ex

# Set default port to 8000 if not provided
PORT=${PORT:-8000}

exec uvicorn slm_server.app:app --host 0.0.0.0 --port $PORT --workers 1
