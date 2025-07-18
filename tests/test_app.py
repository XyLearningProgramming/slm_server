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

    # Verify custom LLM metrics are present (even if empty)
    assert "llm_span_latency_seconds" in content
    assert "llm_input_tokens" in content
    assert "llm_output_tokens" in content
    assert "llm_time_to_first_token_seconds" in content


def test_llm_span_setup():
    """Test the span setup logic for both streaming and non-streaming modes."""
    from unittest.mock import Mock, patch

    from slm_server.model import ChatCompletionRequest, ChatCompletionStreamResponse
    from slm_server.utils import LLMStats, _finalize_llm_span, _setup_llm_span

    # Create a mock span
    mock_span = Mock()

    # Create a test request
    req = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        max_tokens=100,
        temperature=0.7,
    )

    messages_for_llm = [msg.model_dump() for msg in req.messages]

    def test_streaming_span_setup():
        """Test streaming mode span setup."""
        # Mock the start time to a known value
        with patch("time.time", return_value=1000.0):
            stats = LLMStats.create_llm_stats(messages_for_llm)

        # Test initial span setup
        _setup_llm_span(mock_span, req, messages_for_llm, stats, True)

        # Verify span attributes were set
        expected_calls = [
            ("llm.model", "llama-cpp"),
            ("llm.streaming", True),
            ("llm.max_tokens", 100),
            ("llm.temperature", 0.7),
            ("llm.input.message_count", 2),
            ("llm.input.content_length", stats.input_content_length),
        ]

        for attr_name, attr_value in expected_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

        # Simulate chunk processing
        mock_response = Mock(spec=ChatCompletionStreamResponse)
        mock_response.choices = [Mock()]
        mock_response.choices[0].delta = Mock()
        mock_response.choices[0].delta.content = "Hello"

        # Process multiple chunks with delays
        with patch("time.time", side_effect=[1000.1, 1000.2, 1000.3]):
            stats.update_with_chunk(mock_response)

            mock_response.choices[0].delta.content = " there!"
            stats.update_with_chunk(mock_response)

            mock_response.choices[0].delta.content = " How are you?"
            stats.update_with_chunk(mock_response)

        # Test span finalization with streaming metrics
        mock_span.reset_mock()
        _finalize_llm_span(mock_span, stats, True)

        # Verify streaming-specific attributes were set
        streaming_calls = [
            ("llm.output.content_length", stats.total_output_length),
            ("llm.output.chunk_count", 3),
            ("llm.time_to_first_token_ms", stats.time_to_first_token * 1000),
            ("llm.average_chunk_interval_ms", stats.average_chunk_interval * 1000),
            ("llm.average_chunk_size", stats.average_chunk_size),
            ("llm.chunk_size_min", min(stats.chunk_sizes)),
            ("llm.chunk_size_max", max(stats.chunk_sizes)),
        ]

        for attr_name, attr_value in streaming_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

        # Verify metrics calculations
        assert stats.chunk_count == 3
        expected_length = len("Hello") + len(" there!") + len(" How are you?")
        assert stats.total_output_length == expected_length
        assert stats.time_to_first_token == pytest.approx(0.1)  # 1000.1 - 1000.0
        assert len(stats.chunk_intervals) == 2  # intervals between chunks
        assert stats.chunk_sizes == [5, 7, 13]  # lengths of chunk contents
        # Check that intervals are approximately 0.1 seconds
        assert all(interval == pytest.approx(0.1) for interval in stats.chunk_intervals)

    def test_non_streaming_span_setup():
        """Test non-streaming mode span setup."""
        mock_span.reset_mock()
        stats = LLMStats.create_llm_stats(messages_for_llm)

        # Test initial span setup
        _setup_llm_span(mock_span, req, messages_for_llm, stats, False)

        # Verify span attributes were set
        expected_calls = [
            ("llm.model", "llama-cpp"),
            ("llm.streaming", False),
            ("llm.max_tokens", 100),
            ("llm.temperature", 0.7),
            ("llm.input.message_count", 2),
            ("llm.input.content_length", stats.input_content_length),
        ]

        for attr_name, attr_value in expected_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

        # Simulate completion result processing
        from slm_server.model import ChatCompletionResponse

        mock_response = Mock(spec=ChatCompletionResponse)
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello there! How are you?"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        stats.update_with_usage(mock_response)

        # Test span finalization with non-streaming metrics
        mock_span.reset_mock()
        _finalize_llm_span(mock_span, stats, False)

        # Verify non-streaming-specific attributes were set
        non_streaming_calls = [
            ("llm.output.content_length", stats.total_output_length),
            ("llm.usage.prompt_tokens", 10),
            ("llm.usage.completion_tokens", 15),
            ("llm.usage.total_tokens", 25),
        ]

        for attr_name, attr_value in non_streaming_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

        # Verify metrics
        assert stats.total_output_length == len("Hello there! How are you?")
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 15
        assert stats.total_tokens == 25

    # Run both tests
    test_streaming_span_setup()
    test_non_streaming_span_setup()
