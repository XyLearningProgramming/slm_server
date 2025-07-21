#!/bin/bash

set -ex

# Get the absolute path of the directory where the script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

REPO_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF"
# Set model directory relative to the script's location
MODEL_DIR="$SCRIPT_DIR/../models"

# Create the directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# --- Files to download ---
FILES_TO_DOWNLOAD=(
    "Qwen3-0.6B-Q8_0.gguf"
    # "params"
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

echo "Download process complete! Files are in $MODEL_DIR"
