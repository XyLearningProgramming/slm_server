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

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_mock():
    """Reset the mock before each test."""
    mock_llama.reset_mock()


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
    assert response.json()["detail"] == "Something went wrong"
