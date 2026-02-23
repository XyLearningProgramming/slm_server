"""Post-process raw model output into OpenAI-compatible response structures.

Handles two concerns for both non-streaming and streaming responses:
1. **Reasoning** – extracts ``<think>`` blocks into ``reasoning_content``
   (de facto standard from DeepSeek, supported by LangChain / LiteLLM / OpenRouter).
2. **Tool calls** – extracts ``<tool_call>`` blocks into structured ``tool_calls``.
"""

from __future__ import annotations

import copy
import json
import re
from enum import Enum, auto
from typing import Any

from slm_server.utils.ids import gen_id

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def extract_reasoning(content: str) -> tuple[str | None, str]:
    """Split ``<think>`` blocks from visible content.

    Returns (reasoning_content, cleaned_content):
      * *reasoning_content* – concatenated think-block text, or ``None``.
      * *cleaned_content* – original text with think blocks removed.
    """
    matches = _THINK_RE.findall(content)
    if not matches:
        return None, content

    reasoning = "\n".join(m.strip() for m in matches)
    cleaned = _THINK_RE.sub("", content).strip()
    return reasoning or None, cleaned


def parse_tool_calls(
    content: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Extract ``<tool_call>`` blocks from raw model output.

    Returns (tool_calls, remaining_content):
      * *tool_calls* – list of OpenAI-style tool-call dicts, empty when none found.
      * *remaining_content* – text left after stripping tool_call blocks,
        or ``None`` when nothing meaningful remains.
    """
    matches = _TOOL_CALL_RE.findall(content)
    if not matches:
        return [], content

    tool_calls: list[dict[str, Any]] = []
    for raw_json in matches:
        parsed = json.loads(raw_json)
        arguments = parsed.get("arguments", {})
        tool_calls.append(
            {
                "id": gen_id("call"),
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(arguments)
                    if not isinstance(arguments, str)
                    else arguments,
                },
            }
        )

    remaining = _TOOL_CALL_RE.sub("", content).strip()
    return tool_calls, remaining or None


def postprocess_completion(
    response: dict[str, Any],
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Rewrite a chat-completion response in place:

    1. Normalise ``id`` and ``model`` fields.
    2. Extract ``<think>`` blocks → ``message.reasoning_content``
    3. Extract ``<tool_call>`` blocks → ``message.tool_calls``
    """
    response["id"] = gen_id("chatcmpl")
    if model_id:
        response["model"] = model_id

    for choice in response.get("choices", []):
        message = choice.get("message")
        if message is None or not message.get("content"):
            continue

        content = message["content"]

        reasoning, content = extract_reasoning(content)
        if reasoning is not None:
            message["reasoning_content"] = reasoning

        tool_calls, content = parse_tool_calls(content)
        if tool_calls:
            message["tool_calls"] = tool_calls
            choice["finish_reason"] = "tool_calls"

        message["content"] = content

    return response


# ---------------------------------------------------------------------------
# Streaming post-processing
# ---------------------------------------------------------------------------

_KNOWN_TAGS = frozenset({"<think>", "</think>", "<tool_call>", "</tool_call>"})
_MAX_TAG_LEN = max(len(t) for t in _KNOWN_TAGS)


class _State(Enum):
    CONTENT = auto()
    THINKING = auto()
    TOOL_CALL = auto()


class StreamPostProcessor:
    """State machine that rewrites streaming chunks in-flight.

    Raw ``delta.content`` tokens containing ``<think>``/``<tool_call>`` tags
    are intercepted and re-emitted as ``delta.reasoning_content`` or
    ``delta.tool_calls`` respectively.
    """

    _TAG_TRANSITIONS: dict[str, _State] = {
        "<think>": _State.THINKING,
        "</think>": _State.CONTENT,
        "<tool_call>": _State.TOOL_CALL,
        "</tool_call>": _State.CONTENT,
    }

    def __init__(self, *, model_id: str | None = None) -> None:
        self._state = _State.CONTENT
        self._tag_buf = ""
        self._tool_call_buf = ""
        self._had_tool_calls = False
        self._tool_call_index = 0
        self._last_chunk_template: dict[str, Any] | None = None
        self._strip_leading = False
        self._model_id = model_id
        self._stream_id = gen_id("chatcmpl")

    # -- public API ----------------------------------------------------------

    def process_chunk(
        self, chunk: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process one streaming chunk, returning zero or more output chunks."""
        self._last_chunk_template = chunk

        delta = _get_delta(chunk)
        if delta is None:
            return [self._stamp(chunk)]

        text = delta.get("content")
        if text is None:
            finish = _get_finish_reason(chunk)
            if finish is not None and self._had_tool_calls:
                return [self._rewrite_finish(chunk)]
            return [self._stamp(chunk)]

        result = self._consume_text(text, chunk)

        # Propagate finish_reason from the original chunk onto the last output
        orig_finish = _get_finish_reason(chunk)
        if orig_finish is not None:
            if result:
                result[-1]["choices"][0]["finish_reason"] = (
                    "tool_calls" if self._had_tool_calls else orig_finish
                )
            else:
                result.append(self._rewrite_finish(chunk) if self._had_tool_calls
                              else self._stamp(copy.deepcopy(chunk)))

        return result

    def flush(self) -> list[dict[str, Any]]:
        """Flush any buffered content at end-of-stream."""
        if not self._tag_buf or self._last_chunk_template is None:
            return []
        text = self._tag_buf
        self._tag_buf = ""
        return self._emit_batch(text)

    # -- internals -----------------------------------------------------------

    def _consume_text(
        self, text: str, chunk: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Feed *text* through the tag-detection state machine.

        Batches consecutive characters for the same state into single chunks
        to avoid per-character emission.
        """
        output: list[dict[str, Any]] = []
        batch = ""
        pos = 0

        while pos < len(text):
            ch = text[pos]

            if ch == "<" and not self._tag_buf:
                if batch:
                    output.extend(self._emit_batch(batch))
                    batch = ""
                self._tag_buf = ch
                pos += 1
                continue

            if self._tag_buf:
                self._tag_buf += ch
                pos += 1

                matched_tag = self._try_match_tag()
                if matched_tag is not None:
                    new_state = self._TAG_TRANSITIONS[matched_tag]
                    if (
                        matched_tag == "</tool_call>"
                        and self._state == _State.TOOL_CALL
                    ):
                        output.extend(self._finish_tool_call(chunk))
                    self._state = new_state
                    self._tag_buf = ""
                    self._strip_leading = True
                elif not self._could_be_tag():
                    batch += self._tag_buf
                    self._tag_buf = ""
                continue

            batch += ch
            pos += 1

        if batch:
            output.extend(self._emit_batch(batch))

        return output

    def _try_match_tag(self) -> str | None:
        """Return the tag string if ``_tag_buf`` exactly matches one."""
        if self._tag_buf in _KNOWN_TAGS:
            return self._tag_buf
        return None

    def _could_be_tag(self) -> bool:
        """Check if ``_tag_buf`` is a valid prefix of any known tag."""
        return any(t.startswith(self._tag_buf) for t in _KNOWN_TAGS)

    def _emit_batch(self, text: str) -> list[dict[str, Any]]:
        """Emit *text* as a single chunk for the current state."""
        if not text or self._last_chunk_template is None:
            return []

        if self._strip_leading:
            text = text.lstrip()
            if not text:
                return []
            self._strip_leading = False

        if self._state == _State.CONTENT:
            return [self._make_chunk(content=text)]
        elif self._state == _State.THINKING:
            return [self._make_chunk(reasoning_content=text)]
        else:
            self._tool_call_buf += text
            return []

    def _finish_tool_call(
        self, chunk: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Parse the accumulated JSON and emit a ``tool_calls`` delta."""
        raw = self._tool_call_buf.strip()
        self._tool_call_buf = ""

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [self._make_chunk(content=raw)]

        arguments = parsed.get("arguments", {})
        tool_id = gen_id("call")
        idx = self._tool_call_index
        self._tool_call_index += 1
        self._had_tool_calls = True

        return [
            self._make_chunk(
                tool_calls=[
                    {
                        "index": idx,
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(arguments)
                            if not isinstance(arguments, str)
                            else arguments,
                        },
                    }
                ]
            )
        ]

    def _stamp(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Apply consistent id/model to a passthrough chunk (mutates)."""
        chunk["id"] = self._stream_id
        if self._model_id:
            chunk["model"] = self._model_id
        return chunk

    def _rewrite_finish(self, chunk: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(chunk)
        out["id"] = self._stream_id
        if self._model_id:
            out["model"] = self._model_id
        choice = out["choices"][0]
        choice["finish_reason"] = "tool_calls"
        return out

    def _make_chunk(self, **delta_fields: Any) -> dict[str, Any]:
        """Build an output chunk from the last seen template."""
        out = copy.deepcopy(self._last_chunk_template)
        out["id"] = self._stream_id
        if self._model_id:
            out["model"] = self._model_id
        choice = out["choices"][0]
        # Preserve role from original delta if present
        orig_delta = choice.get("delta", {})
        new_delta: dict[str, Any] = {}
        if "role" in orig_delta and not delta_fields.get("tool_calls"):
            new_delta["role"] = orig_delta["role"]
        new_delta.update(delta_fields)
        choice["delta"] = new_delta
        choice["finish_reason"] = None
        return out


def _get_delta(chunk: dict[str, Any]) -> dict[str, Any] | None:
    choices = chunk.get("choices")
    if not choices:
        return None
    return choices[0].get("delta")


def _get_finish_reason(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices")
    if not choices:
        return None
    return choices[0].get("finish_reason")
