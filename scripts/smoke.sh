#!/bin/bash

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=== Health check ==="
curl -sf "$BASE_URL/health"
echo

echo "=== List models ==="
curl -sf "$BASE_URL/api/v1/models" | python3 -m json.tool
echo

echo "=== Chat completion ==="
curl -sf "$BASE_URL/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }' | python3 -m json.tool
echo

echo "=== Chat completion (streaming) ==="
curl -sf "$BASE_URL/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
    "stream": true
  }'
echo

echo "=== Embeddings (single) ==="
curl -sf "$BASE_URL/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world"
  }' | python3 -m json.tool
echo

echo "=== Embeddings (batch) ==="
curl -sf "$BASE_URL/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["The cat sat on the mat.", "A dog played in the park."]
  }' | python3 -m json.tool
echo

echo "All smoke tests passed."
