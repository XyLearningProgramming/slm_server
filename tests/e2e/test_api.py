
import json
import pytest
import httpx

@pytest.mark.api
@pytest.mark.api_non_streaming
def test_chat_completion_non_streaming(server):
    """Test non-streaming chat completion API."""
    with httpx.Client() as client:
        response = client.post(
            "http://localhost:8000/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello /no think"}],
                "stream": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        response_data = response.json()
        assert "choices" in response_data
        assert len(response_data["choices"]) > 0
        assert "message" in response_data["choices"][0]
        assert "content" in response_data["choices"][0]["message"]

@pytest.mark.api
def test_chat_completion_streaming(server):
    """Test streaming chat completion API."""
    with httpx.Client() as client:
        with client.stream(
            "POST",
            "http://localhost:8000/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello /no think"}],
                "stream": True,
            },
            timeout=30,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if chunk.strip():
                    data_str = chunk.decode("utf-8").replace("data: ", "").strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        response_data = json.loads(data_str)
                        assert "choices" in response_data
                        assert len(response_data["choices"]) > 0
                        assert "delta" in response_data["choices"][0]
                    except json.JSONDecodeError:
                        pytest.fail(f"Error decoding JSON: {data_str}")

@pytest.mark.api
@pytest.mark.api_non_streaming
def test_embeddings(server):
    """Test embeddings API."""
    with httpx.Client() as client:
        response = client.post(
            "http://localhost:8000/api/v1/embeddings",
            json={
                "input": "Hello world"
            },
        )
        response.raise_for_status()
        response_data = response.json()
        assert response_data["object"] == "list"
        assert len(response_data["data"]) == 1
        assert "embedding" in response_data["data"][0]
        assert len(response_data["data"][0]["embedding"]) > 0

        # Test with multiple inputs
        response = client.post(
            "http://localhost:8000/api/v1/embeddings",
            json={
                "input": ["Hello", "World"],
                "model": "Qwen3-0.6B-GGUF"
            },
        )
        response.raise_for_status()
        response_data = response.json()
        assert len(response_data["data"]) == 2


@pytest.mark.api
@pytest.mark.api_non_streaming
def test_embeddings_multiple(server):
    """Test embeddings API."""
    with httpx.Client() as client:
        response = client.post(
            "http://localhost:8000/api/v1/embeddings",
            json={
                "input": ["Hello, world"]
            },
        )
        response.raise_for_status()
        response_data = response.json()
        assert response_data["object"] == "list"
        assert len(response_data["data"]) == 1
        assert "embedding" in response_data["data"][0]
        assert len(response_data["data"][0]["embedding"]) > 0

        # Test with multiple inputs
        response = client.post(
            "http://localhost:8000/api/v1/embeddings",
            json={
                "input": ["Hello", "World"],
                "model": "Qwen3-0.6B-GGUF"
            },
            timeout=30,
        )
        response.raise_for_status()
        response_data = response.json()
        assert len(response_data["data"]) == 2

