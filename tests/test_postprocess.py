"""Tests for response post-processing: reasoning extraction and tool call parsing."""

import json

from slm_server.utils.ids import gen_id
from slm_server.utils.postprocess import (
    StreamPostProcessor,
    extract_reasoning,
    parse_tool_calls,
    postprocess_completion,
)


class TestExtractReasoning:

    def test_extracts_reasoning(self):
        content = "<think>\nStep 1: analyze.\nStep 2: respond.\n</think>\n\nHello!"
        reasoning, cleaned = extract_reasoning(content)

        assert reasoning == "Step 1: analyze.\nStep 2: respond."
        assert cleaned == "Hello!"

    def test_empty_think_block(self):
        content = "<think>\n\n</think>\n\nHello!"
        reasoning, cleaned = extract_reasoning(content)

        assert reasoning is None
        assert cleaned == "Hello!"

    def test_no_think_block(self):
        content = "Just a regular response."
        reasoning, cleaned = extract_reasoning(content)

        assert reasoning is None
        assert cleaned == content

    def test_multiple_think_blocks(self):
        content = (
            "<think>First thought.</think>\n"
            "Middle text.\n"
            "<think>Second thought.</think>\n"
            "Final answer."
        )
        reasoning, cleaned = extract_reasoning(content)

        assert reasoning == "First thought.\nSecond thought."
        assert "Middle text." in cleaned
        assert "Final answer." in cleaned

    def test_think_only_no_content(self):
        content = "<think>Just thinking...</think>"
        reasoning, cleaned = extract_reasoning(content)

        assert reasoning == "Just thinking..."
        assert cleaned == ""


class TestParseToolCalls:

    def test_single_tool_call(self):
        content = (
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"location": "San Francisco"}}\n'
            '</tool_call>'
        )
        tool_calls, remaining = parse_tool_calls(content)

        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {
            "location": "San Francisco"
        }
        assert tc["id"].startswith("call_")
        assert remaining is None

    def test_multiple_tool_calls(self):
        content = (
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"location": "SF"}}\n'
            '</tool_call>\n'
            '<tool_call>\n'
            '{"name": "get_time", "arguments": {"timezone": "PST"}}\n'
            '</tool_call>'
        )
        tool_calls, remaining = parse_tool_calls(content)

        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_time"
        assert tool_calls[0]["id"] != tool_calls[1]["id"]

    def test_no_tool_calls(self):
        content = "The weather in San Francisco is sunny today."
        tool_calls, remaining = parse_tool_calls(content)

        assert tool_calls == []
        assert remaining == content

    def test_tool_call_with_remaining_content(self):
        content = (
            "Here is the result:\n"
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": "test"}}\n'
            '</tool_call>\n'
            "Let me know if you need more."
        )
        tool_calls, remaining = parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert remaining == "Here is the result:\n\nLet me know if you need more."

    def test_arguments_as_string(self):
        content = (
            '<tool_call>\n'
            '{"name": "run", "arguments": "{\\"key\\": \\"val\\"}"}\n'
            '</tool_call>'
        )
        tool_calls, remaining = parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == '{"key": "val"}'

    def test_empty_arguments(self):
        content = (
            '<tool_call>\n'
            '{"name": "ping", "arguments": {}}\n'
            '</tool_call>'
        )
        tool_calls, remaining = parse_tool_calls(content)

        assert len(tool_calls) == 1
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {}


class TestPostprocessCompletion:

    def _make_response(self, content, finish_reason="stop"):
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    def test_extracts_reasoning_from_plain_response(self):
        resp = self._make_response(
            "<think>\nAnalyzing the question.\n</think>\n\nThe answer is 42."
        )
        result = postprocess_completion(resp)
        msg = result["choices"][0]["message"]

        assert result["id"].startswith("chatcmpl_")
        assert msg["content"] == "The answer is 42."
        assert msg["reasoning_content"] == "Analyzing the question."
        assert "tool_calls" not in msg
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_extracts_reasoning_and_tool_calls(self):
        resp = self._make_response(
            '<think>\nI should call the weather API.\n</think>\n\n'
            '<tool_call>\n'
            '{"name": "get_weather", "arguments": {"location": "SF"}}\n'
            '</tool_call>'
        )
        result = postprocess_completion(resp)
        msg = result["choices"][0]["message"]

        assert result["id"].startswith("chatcmpl_")
        assert msg["content"] is None
        assert msg["reasoning_content"] == "I should call the weather API."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_empty_think_block_no_reasoning_content(self):
        resp = self._make_response(
            "<think>\n\n</think>\n\nHello!"
        )
        result = postprocess_completion(resp)
        msg = result["choices"][0]["message"]

        assert msg["content"] == "Hello!"
        assert "reasoning_content" not in msg

    def test_leaves_plain_text_unchanged(self):
        resp = self._make_response("Hello, how can I help?")
        result = postprocess_completion(resp)
        msg = result["choices"][0]["message"]

        assert msg["content"] == "Hello, how can I help?"
        assert "reasoning_content" not in msg
        assert "tool_calls" not in msg
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_leaves_empty_content_unchanged(self):
        resp = self._make_response("")
        result = postprocess_completion(resp)
        msg = result["choices"][0]["message"]

        assert msg["content"] == ""
        assert "reasoning_content" not in msg

    def test_rewrites_id_and_preserves_usage(self):
        resp = self._make_response(
            '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        )
        result = postprocess_completion(resp)

        assert result["id"].startswith("chatcmpl_")
        assert result["id"] != "chatcmpl-123"
        assert result["usage"]["total_tokens"] == 150

    def test_rewrites_model_when_provided(self):
        resp = self._make_response("Hello!")
        result = postprocess_completion(resp, model_id="Qwen3-0.6B")

        assert result["model"] == "Qwen3-0.6B"

    def test_preserves_model_when_not_provided(self):
        resp = self._make_response("Hello!")
        result = postprocess_completion(resp)

        assert result["model"] == "test-model"

    def test_no_raw_tags_leak_into_content(self):
        resp = self._make_response(
            '<think>\nThinking...\n</think>\n\n'
            '<tool_call>\n'
            '{"name": "f", "arguments": {}}\n'
            '</tool_call>'
        )
        result = postprocess_completion(resp)
        msg = result["choices"][0]["message"]
        content = msg["content"] or ""

        assert "<think>" not in content
        assert "</think>" not in content
        assert "<tool_call>" not in content
        assert "</tool_call>" not in content


# ---------------------------------------------------------------------------
# Streaming post-processing
# ---------------------------------------------------------------------------


def _make_chunk(content=None, finish_reason=None, **extra_delta):
    """Helper to build a minimal streaming chunk dict."""
    delta = {}
    if content is not None:
        delta["content"] = content
    delta.update(extra_delta)
    return {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _feed(processor, tokens):
    """Feed a list of token strings through the processor, return all output chunks."""
    out = []
    for tok in tokens:
        chunk = _make_chunk(content=tok)
        out.extend(processor.process_chunk(chunk))
    # Process a final chunk with finish_reason
    final = _make_chunk(finish_reason="stop")
    out.extend(processor.process_chunk(final))
    out.extend(processor.flush())
    return out


def _collect_field(chunks, field):
    """Concatenate a delta field across all output chunks."""
    parts = []
    for c in chunks:
        delta = c["choices"][0]["delta"]
        val = delta.get(field)
        if val:
            parts.append(val)
    return "".join(parts)


class TestStreamPostProcessor:

    def test_plain_content_passthrough(self):
        proc = StreamPostProcessor()
        chunks = _feed(proc, ["Hello", " world", "!"])

        assert _collect_field(chunks, "content") == "Hello world!"
        assert _collect_field(chunks, "reasoning_content") == ""

    def test_think_block_to_reasoning_content(self):
        proc = StreamPostProcessor()
        tokens = ["<think>", "I need to think", "</think>", "Answer here"]
        chunks = _feed(proc, tokens)

        assert _collect_field(chunks, "reasoning_content") == "I need to think"
        assert _collect_field(chunks, "content") == "Answer here"

    def test_think_tags_split_across_tokens(self):
        proc = StreamPostProcessor()
        tokens = ["<", "think", ">", "reasoning", "<", "/think", ">", "visible"]
        chunks = _feed(proc, tokens)

        assert _collect_field(chunks, "reasoning_content") == "reasoning"
        assert _collect_field(chunks, "content") == "visible"

    def test_tool_call_emits_tool_calls_delta(self):
        proc = StreamPostProcessor()
        tokens = [
            "<tool_call>",
            '{"name": "get_weather",',
            ' "arguments": {"location": "SF"}}',
            "</tool_call>",
        ]
        chunks = _feed(proc, tokens)

        tc_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("tool_calls")
        ]
        assert len(tc_chunks) == 1
        tc = tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "SF"}
        assert tc["id"].startswith("call_")

    def test_tool_call_sets_finish_reason(self):
        proc = StreamPostProcessor()
        tokens = [
            "<tool_call>",
            '{"name": "f", "arguments": {}}',
            "</tool_call>",
        ]
        chunks = _feed(proc, tokens)

        final_chunks = [
            c for c in chunks if c["choices"][0].get("finish_reason") is not None
        ]
        assert any(
            c["choices"][0]["finish_reason"] == "tool_calls" for c in final_chunks
        )

    def test_think_then_tool_call(self):
        proc = StreamPostProcessor()
        tokens = [
            "<think>",
            "Let me check",
            "</think>",
            "<tool_call>",
            '{"name": "search", "arguments": {"q": "test"}}',
            "</tool_call>",
        ]
        chunks = _feed(proc, tokens)

        assert _collect_field(chunks, "reasoning_content") == "Let me check"

        tc_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("tool_calls")
        ]
        assert len(tc_chunks) == 1
        assert tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0][
            "function"
        ]["name"] == "search"

    def test_multiple_tool_calls(self):
        proc = StreamPostProcessor()
        tokens = [
            "<tool_call>",
            '{"name": "a", "arguments": {}}',
            "</tool_call>",
            "<tool_call>",
            '{"name": "b", "arguments": {}}',
            "</tool_call>",
        ]
        chunks = _feed(proc, tokens)

        tc_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("tool_calls")
        ]
        assert len(tc_chunks) == 2
        assert tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
        assert tc_chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1

    def test_partial_tag_not_matching_flushed_as_content(self):
        proc = StreamPostProcessor()
        tokens = ["Hello <b>world</b>"]
        chunks = _feed(proc, tokens)

        assert _collect_field(chunks, "content") == "Hello <b>world</b>"

    def test_no_content_chunk_passthrough(self):
        proc = StreamPostProcessor()
        role_chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        result = proc.process_chunk(role_chunk)
        assert len(result) == 1
        assert result[0]["choices"][0]["delta"]["role"] == "assistant"

    def test_empty_think_block_suppressed(self):
        proc = StreamPostProcessor()
        tokens = ["<think>", "\n\n", "</think>", "\n\nHello!"]
        chunks = _feed(proc, tokens)

        content = _collect_field(chunks, "content")
        assert "Hello!" in content
        assert "<think>" not in content
        assert "</think>" not in content

    def test_leading_whitespace_stripped_after_tags(self):
        """Align with non-streaming .strip(): no leading \\n after tag transitions."""
        proc = StreamPostProcessor()
        tokens = ["<think>", "\n\nI think.", "\n</think>", "\n\nAnswer here"]
        chunks = _feed(proc, tokens)

        reasoning = _collect_field(chunks, "reasoning_content")
        content = _collect_field(chunks, "content")
        assert reasoning == "I think.\n"
        assert content == "Answer here"

    def test_whitespace_only_after_close_think_suppressed(self):
        proc = StreamPostProcessor()
        tokens = ["<think>", "ok", "</think>", "\n\n", "Hello"]
        chunks = _feed(proc, tokens)

        content = _collect_field(chunks, "content")
        assert content == "Hello"

    def test_stream_id_consistent_across_chunks(self):
        proc = StreamPostProcessor()
        chunks = _feed(proc, ["Hello", " world"])
        ids = {c["id"] for c in chunks}
        assert len(ids) == 1
        assert ids.pop().startswith("chatcmpl_")

    def test_stream_model_rewritten_when_provided(self):
        proc = StreamPostProcessor(model_id="Qwen3-0.6B")
        chunks = _feed(proc, ["Hello"])
        for c in chunks:
            assert c["model"] == "Qwen3-0.6B"

    def test_stream_model_preserved_when_not_provided(self):
        proc = StreamPostProcessor()
        chunks = _feed(proc, ["Hello"])
        for c in chunks:
            if "model" in c:
                assert c["model"] == "test-model"


class TestGenId:

    def test_chatcmpl_prefix(self):
        cid = gen_id("chatcmpl")
        assert cid.startswith("chatcmpl_")

    def test_call_prefix_uses_underscore(self):
        cid = gen_id("call")
        assert cid.startswith("call_")

    def test_unique(self):
        ids = {gen_id("chatcmpl") for _ in range(100)}
        assert len(ids) == 100

    def test_compact_length(self):
        cid = gen_id("chatcmpl")
        assert len(cid) < 30
