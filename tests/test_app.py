from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from slm_server.app import DETAIL_SEM_TIMEOUT, app, get_llm

# Create a mock Llama instance
mock_llama = MagicMock()


# Override the get_llm dependency with the mock
def override_get_llm():
    return mock_llama


app.dependency_overrides[get_llm] = override_get_llm

# Use TestClient with lifespan context to ensure metrics endpoint is created
client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_mock():
    """Reset the mock before each test."""
    mock_llama.reset_mock()
    mock_llama.create_chat_completion.side_effect = None  # Clear any side effects


def test_chat_completion_non_streaming():
    """Tests the non-streaming chat completion endpoint."""
    mock_llama.create_chat_completion.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    response = client.post(
        "/api/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello there!"
    mock_llama.create_chat_completion.assert_called_once()


def test_chat_completion_streaming():
    """Tests the streaming chat completion endpoint."""
    mock_chunks = [
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " there!"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    mock_llama.create_chat_completion.return_value = iter(mock_chunks)

    response = client.post(
        "/api/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )

    assert response.status_code == 200
    # Check that the response is a streaming response
    assert "text/event-stream" in response.headers["content-type"]

    # Collect the streaming content
    content = response.text
    assert "Hello" in content
    assert "there!" in content
    assert "data: [DONE]" in content
    mock_llama.create_chat_completion.assert_called_once()


def test_server_busy_exception():
    """Tests the 503 Server Busy exception."""
    # Simulate a timeout when acquiring the semaphore
    with patch("asyncio.wait_for", side_effect=TimeoutError):
        response = client.post(
            "/api/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )
        assert response.status_code == 503
        assert response.json()["detail"] == DETAIL_SEM_TIMEOUT


def test_generic_exception():
    """Tests the 500 Internal Server Error for unexpected exceptions."""
    mock_llama.create_chat_completion.side_effect = Exception("Something went wrong")

    response = client.post(
        "/api/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
    )

    assert response.status_code == 500
    assert "Something went wrong" in response.json()["detail"]


def test_health_endpoint():
    """Tests the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == "ok"


def test_metrics_endpoint_integration():
    """Tests that /metrics endpoint integrates both OpenTelemetry and prometheus-fastapi-instrumentator metrics."""
    # Make a request to generate metrics
    health_response = client.get("/health")
    assert health_response.status_code == 200

    # Check metrics endpoint
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200

    content = metrics_response.text

    # Verify prometheus-fastapi-instrumentator custom metrics are present
    assert "process_cpu_usage_percent" in content
    assert "process_memory_usage_bytes" in content

    # Verify standard Python metrics are present
    assert "python_gc_objects_collected_total" in content
    assert "python_info" in content
    assert "process_virtual_memory_bytes" in content

    # Verify custom SLM metrics are present (even if empty)
    assert "slm_completion_duration_seconds" in content
    assert "slm_tokens_total" in content
    assert "slm_completion_tokens_per_second" in content
    assert "slm_first_token_delay_ms" in content


def test_streaming_call_with_tracing_integration():
    """Integration test for streaming call with complete tracing flow."""
    from unittest.mock import patch
    
    # Mock chunks with realistic content
    mock_chunks = [
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-123", 
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " there!"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk", 
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " How are you?"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    mock_llama.create_chat_completion.return_value = iter(mock_chunks)
    
    # Make streaming request
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello world! This is a test message."}],
            "stream": True,
            "max_tokens": 100,
            "temperature": 0.8
        },
    )
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    # Verify streaming content
    content = response.text
    assert "Hello" in content
    assert "there!" in content
    assert "How are you?" in content
    assert "data: [DONE]" in content
    
    # Verify the LLM was called with correct parameters
    mock_llama.create_chat_completion.assert_called_once()
    call_args = mock_llama.create_chat_completion.call_args
    
    assert call_args[1]["max_tokens"] == 100
    assert call_args[1]["temperature"] == 0.8
    assert call_args[1]["stream"] is True
    assert len(call_args[1]["messages"]) == 1
    assert call_args[1]["messages"][0]["content"] == "Hello world! This is a test message."


def test_non_streaming_call_with_tracing_integration():
    """Integration test for non-streaming call with complete tracing flow."""
    from unittest.mock import patch
    
    # Mock complete response with usage metrics
    mock_llama.create_chat_completion.return_value = {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 20,
            "total_tokens": 35
        },
    }
    
    # Make non-streaming request
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
                {"role": "user", "content": "Great! Can you help me with something?"}
            ],
            "stream": False,
            "max_tokens": 150,
            "temperature": 0.3
        },
    )
    
    assert response.status_code == 200
    
    # Verify response structure
    response_data = response.json()
    assert response_data["object"] == "chat.completion"
    assert response_data["choices"][0]["message"]["content"] == "Hello! I'm doing well, thank you for asking. How can I help you today?"
    assert response_data["usage"]["prompt_tokens"] == 15
    assert response_data["usage"]["completion_tokens"] == 20
    assert response_data["usage"]["total_tokens"] == 35
    
    # Verify the LLM was called with correct parameters  
    mock_llama.create_chat_completion.assert_called_once()
    call_args = mock_llama.create_chat_completion.call_args
    
    assert call_args[1]["max_tokens"] == 150
    assert call_args[1]["temperature"] == 0.3
    assert call_args[1]["stream"] is False
    assert len(call_args[1]["messages"]) == 3
    
    # Verify message content
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "Hello, how are you?"
    assert messages[1]["content"] == "I'm doing well!"
    assert messages[2]["content"] == "Great! Can you help me with something?"


def test_streaming_call_with_empty_chunks():
    """Test streaming call handling empty chunks correctly."""
    # Mock chunks including empty ones
    mock_chunks = [
        {
            "id": "chatcmpl-789",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Start"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-789",
            "object": "chat.completion.chunk", 
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},  # Empty chunk
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-789",
            "object": "chat.completion.chunk",
            "created": 1234567890, 
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " End"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    mock_llama.create_chat_completion.return_value = iter(mock_chunks)
    
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test empty chunks"}],
            "stream": True
        },
    )
    
    assert response.status_code == 200
    
    # Verify content handling
    content = response.text
    assert "Start" in content
    assert " End" in content
    assert "data: [DONE]" in content
    
    # Should handle empty chunks gracefully without errors
    mock_llama.create_chat_completion.assert_called_once()


def test_request_validation_and_defaults():
    """Test request validation and default parameter handling."""
    # Test minimal request
    mock_llama.create_chat_completion.return_value = {
        "id": "chatcmpl-minimal",
        "object": "chat.completion", 
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    
    response = client.post(
        "/api/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Test"}]},
    )
    
    assert response.status_code == 200
    
    # Verify defaults were applied
    call_args = mock_llama.create_chat_completion.call_args
    assert call_args[1]["max_tokens"] == 2048  # Default value
    assert call_args[1]["temperature"] == 0.7  # Default value
    assert call_args[1]["stream"] is False     # Default value


