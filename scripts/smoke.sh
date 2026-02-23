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
    "max_tokens": 512
  }' | python3 -m json.tool
echo

echo "=== Chat completion (streaming) ==="
curl -sf "$BASE_URL/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 512,
    "stream": true
  }'
echo

echo "=== Tool call (no tool_choice, defaults to auto) ==="
TOOL_RESP=$(curl -sf "$BASE_URL/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in San Francisco? /no_think"}],
    "max_tokens": 256,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "City name"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }')
echo "$TOOL_RESP" | python3 -m json.tool

# Verify response has structured tool_calls (not raw <tool_call> in content)
echo "$TOOL_RESP" | python3 -c "
import sys, json
resp = json.load(sys.stdin)
choice = resp['choices'][0]
msg = choice['message']
has_tool = 'tool_calls' in msg and msg['tool_calls']
has_content = 'content' in msg and msg['content']
if not has_tool and not has_content:
    print('FAIL: no tool_calls and no content'); sys.exit(1)
if has_tool:
    tc = msg['tool_calls'][0]
    assert tc['type'] == 'function', f'bad type: {tc[\"type\"]}'
    assert 'name' in tc['function'], 'missing function name'
    assert 'arguments' in tc['function'], 'missing arguments'
    assert '<tool_call>' not in (msg.get('content') or ''), 'raw <tool_call> leaked into content'
    assert choice['finish_reason'] == 'tool_calls', f'bad finish_reason: {choice[\"finish_reason\"]}'
    print(f'tool_calls: {tc[\"function\"][\"name\"]}({tc[\"function\"][\"arguments\"]})')
else:
    print(f'content_only: {msg[\"content\"][:80]}...')
"
echo

echo "=== Tool call streaming (no tool_choice, defaults to auto) ==="
STREAM_RESP=$(curl -sf "$BASE_URL/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in San Francisco? /no_think"}],
    "max_tokens": 256,
    "stream": true,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "City name"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }')
echo "$STREAM_RESP"

# Verify no raw <tool_call> tags leaked through as content
echo "$STREAM_RESP" | python3 -c "
import sys
raw = sys.stdin.read()
assert '<tool_call>' not in raw, 'raw <tool_call> tag leaked into stream'
assert '</tool_call>' not in raw, 'raw </tool_call> tag leaked into stream'
# Check for structured tool_calls in at least one chunk
has_tool_calls = '\"tool_calls\"' in raw
has_content = '\"content\"' in raw
if has_tool_calls:
    print('streaming tool_calls: structured delta found')
elif has_content:
    print('streaming tool_calls: content_only (model chose not to call tool)')
else:
    print('FAIL: no tool_calls and no content in stream'); sys.exit(1)
"
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
