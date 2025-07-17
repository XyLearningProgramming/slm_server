#!/bin/bash

exec uv run -- uvicorn slm_server.app:app --host 0.0.0.0 --port 8000
