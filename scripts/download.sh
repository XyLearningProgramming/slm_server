#!/bin/sh
#
# Download model files for slm-server.
#
# When run inside the init container, MODEL_DIR is set by the caller
# (the Helm-rendered configmap). For local use it defaults to
# ../models relative to this script.

set -e

MODEL_DIR="${MODEL_DIR:-$(cd -- "$(dirname "$0")" && pwd)/../models}"
mkdir -p "$MODEL_DIR"

# --- Chat LLM: Qwen3-0.6B (Q4_K_M quantisation from second-state) ---
GGUF_REPO="https://huggingface.co/second-state/Qwen3-0.6B-GGUF"
GGUF_FILE="Qwen3-0.6B-Q4_K_M.gguf"

if [ -f "$MODEL_DIR/$GGUF_FILE" ]; then
  echo "$GGUF_FILE already exists, skipping."
else
  echo "Downloading $GGUF_FILE ..."
  curl -fSL -o "$MODEL_DIR/$GGUF_FILE" "$GGUF_REPO/resolve/main/$GGUF_FILE"
fi

# --- Embedding: all-MiniLM-L6-v2 (ONNX, quantized UINT8 for AVX2) ---
EMBED_REPO="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIR="$MODEL_DIR/all-MiniLM-L6-v2"
mkdir -p "$EMBED_DIR/onnx"

if [ -f "$EMBED_DIR/tokenizer.json" ]; then
  echo "tokenizer.json already exists, skipping."
else
  echo "Downloading tokenizer.json ..."
  curl -fSL -o "$EMBED_DIR/tokenizer.json" "$EMBED_REPO/resolve/main/tokenizer.json"
fi

ONNX_FILE="model_quint8_avx2.onnx"
if [ -f "$EMBED_DIR/onnx/$ONNX_FILE" ]; then
  echo "$ONNX_FILE already exists, skipping."
else
  echo "Downloading $ONNX_FILE ..."
  curl -fSL -o "$EMBED_DIR/onnx/$ONNX_FILE" "$EMBED_REPO/resolve/main/onnx/$ONNX_FILE"
fi

echo "Download complete. Files are in $MODEL_DIR"
