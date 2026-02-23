from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

if TYPE_CHECKING:
    from slm_server.config import EmbeddingSettings

logger = logging.getLogger(__name__)

ONNX_PROVIDERS: list[str] = ["CPUExecutionProvider"]
KEY_INPUT_IDS: str = "input_ids"
KEY_ATTENTION_MASK: str = "attention_mask"
KEY_TOKEN_TYPE_IDS: str = "token_type_ids"


class OnnxEmbeddingModel:
    """Lightweight ONNX-based sentence embedding model.

    Replicates the sentence-transformers all-MiniLM-L6-v2 pipeline:
    tokenize -> BERT forward pass -> mean pooling -> L2 normalize.
    Uses onnxruntime + tokenizers directly (no PyTorch dependency).
    """

    def __init__(self, settings: EmbeddingSettings):
        start = time.monotonic()

        self.model_id = settings.model_id

        self.tokenizer = Tokenizer.from_file(settings.tokenizer_path)
        self.tokenizer.enable_truncation(max_length=settings.max_length)
        self.tokenizer.enable_padding(length=None)

        self.session = InferenceSession(settings.onnx_path, providers=ONNX_PROVIDERS)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Loaded embedding model %s in %.1fms", settings.onnx_path, elapsed_ms
        )

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into dense vectors (384-dim for MiniLM-L6-v2)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        encodings = self.tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self.session.run(
            None,
            {
                KEY_INPUT_IDS: input_ids,
                KEY_ATTENTION_MASK: attention_mask,
                KEY_TOKEN_TYPE_IDS: token_type_ids,
            },
        )

        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

        # Mean pooling: average token embeddings weighted by attention mask
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings
