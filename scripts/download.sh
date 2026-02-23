#!/bin/bash

set -ex

# Get the absolute path of the directory where the script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Original (official Qwen repo, Q8_0 only):
#   https://huggingface.co/Qwen/Qwen3-0.6B-GGUF  ->  Qwen3-0.6B-Q8_0.gguf
# Switched to second-state community repo for Q4_K_M quantization.
# See README.md "Model Choice" section for rationale.
REPO_URL="https://huggingface.co/second-state/Qwen3-0.6B-GGUF"
# Set model directory relative to the script's location
MODEL_DIR="$SCRIPT_DIR/../models"

# Create the directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# --- Files to download ---
FILES_TO_DOWNLOAD=(
    "Qwen3-0.6B-Q4_K_M.gguf"
    # Previous default: "Qwen3-0.6B-Q8_0.gguf" (805 MB, from Qwen/Qwen3-0.6B-GGUF)
)

echo "Downloading Qwen3-0.6B-GGUF model and params files..."

for file in "${FILES_TO_DOWNLOAD[@]}"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        echo "$file already exists, skipping download."
    else
        echo "Downloading $file..."
        wget -P "$MODEL_DIR" "$REPO_URL/resolve/main/$file" || {
            echo "Failed to download $file with wget, trying curl..."
            curl -L -o "$MODEL_DIR/$file" "$REPO_URL/resolve/main/$file"
        }
    fi
done

# --- Embedding model: all-MiniLM-L6-v2 (ONNX, quantized UINT8 for AVX2) ---
EMBEDDING_REPO_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_DIR="$MODEL_DIR/all-MiniLM-L6-v2"

mkdir -p "$EMBEDDING_MODEL_DIR/onnx"

EMBEDDING_FILES=(
    "onnx/model_quint8_avx2.onnx"
    "tokenizer.json"
)

echo "Downloading all-MiniLM-L6-v2 ONNX embedding model..."

for file in "${EMBEDDING_FILES[@]}"; do
    dest="$EMBEDDING_MODEL_DIR/$file"
    if [ -f "$dest" ]; then
        echo "$file already exists, skipping download."
    else
        echo "Downloading $file..."
        wget -O "$dest" "$EMBEDDING_REPO_URL/resolve/main/$file" || {
            echo "Failed to download $file with wget, trying curl..."
            curl -L -o "$dest" "$EMBEDDING_REPO_URL/resolve/main/$file"
        }
    fi
done

echo "Download process complete! Files are in $MODEL_DIR"
