"""Compact ID generation for OpenAI-compatible responses."""

from __future__ import annotations

import os
import string

_LENGTH = 12
_ALPHABET = string.ascii_letters + string.digits
_BASE = len(_ALPHABET)


def gen_id(prefix: str) -> str:
    """Return ``prefix_<base62 random>``, e.g. ``chatcmpl_BxK9mQ7r2pNw``."""
    raw = int.from_bytes(os.urandom(_LENGTH))
    chars: list[str] = []
    while raw:
        raw, idx = divmod(raw, _BASE)
        chars.append(_ALPHABET[idx])
    return f"{prefix}_{''.join(chars)}"
