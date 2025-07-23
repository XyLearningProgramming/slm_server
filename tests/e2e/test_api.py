import asyncio
import json

import httpx


async def test_chat_completion_non_streaming():
    """Test non-streaming chat completion API."""
    print("Testing non-streaming chat completion...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello /no think"}],
                "stream": False,
            },
            timeout=30,
        )
        assert response.status_code == 200
        response_data = response.json()
        print(f"Non-streaming response: {response_data}")
        assert "choices" in response_data
        assert len(response_data["choices"]) > 0
        assert "message" in response_data["choices"][0]
        assert "content" in response_data["choices"][0]["message"]


async def test_chat_completion_streaming():
    """Test streaming chat completion API."""
    print("\nTesting streaming chat completion...")
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello /no think"}],
                "stream": True,
            },
            timeout=30,
        ) as response:
            assert response.status_code == 200
            print("Streaming response:")
            async for chunk in response.aiter_bytes():
                if chunk.strip():
                    # Decode bytes to string and remove the 'data: ' prefix
                    data_str = chunk.decode("utf-8").replace("data: ", "").strip()
                    if data_str == "[DONE]":
                        print("\nStream finished.")
                        break
                    try:
                        # Parse the JSON data
                        response_data = json.loads(data_str)
                        print(response_data, end="", flush=True)
                        assert "choices" in response_data
                        assert len(response_data["choices"]) > 0
                        assert "delta" in response_data["choices"][0]
                    except json.JSONDecodeError:
                        print(f"\nError decoding JSON: {data_str}")


async def test_embeddings():
    """Test embeddings API."""
    print("Testing embeddings API...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/embeddings",
            json={
                "input": "Hello world",
                "model": "text-embedding-ada-002"
            },
            timeout=30,
        )
        assert response.status_code == 200
        response_data = response.json()
        print(f"Embeddings response: {response_data}")
        
        # Validate response structure
        assert "object" in response_data
        assert response_data["object"] == "list"
        assert "data" in response_data
        assert len(response_data["data"]) == 1
        assert "embedding" in response_data["data"][0]
        assert "index" in response_data["data"][0]
        assert len(response_data["data"][0]["embedding"]) == 1536
        assert "usage" in response_data
        
        # Test with multiple inputs
        response = await client.post(
            "http://localhost:8000/api/v1/embeddings",
            json={
                "input": ["Hello", "World"],
                "model": "text-embedding-ada-002"
            },
            timeout=30,
        )
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["data"]) == 2
        print("Multiple inputs test passed!")


async def run_api_tests():
    """Run all API tests."""
    print("=== API Tests ===\n")
    try:
        await test_chat_completion_non_streaming()
        print("\n" + "="*50 + "\n")
        
        await test_chat_completion_streaming()
        print("\n" + "="*50 + "\n")
        
        await test_embeddings()
        print("\n" + "="*50 + "\n")
        
        print("✅ All API tests completed successfully!")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(run_api_tests())