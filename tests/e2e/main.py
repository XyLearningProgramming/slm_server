import asyncio
import json

import httpx


async def test_chat_completion_non_streaming():
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


async def main():
    await test_chat_completion_non_streaming()
    await test_chat_completion_streaming()


if __name__ == "__main__":
    asyncio.run(main())
