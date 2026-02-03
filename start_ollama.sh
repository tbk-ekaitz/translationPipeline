#!/bin/bash
# Start Ollama with OLLAMA_NUM_PARALLEL from config.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# Extract max_workers from config.yaml
MAX_WORKERS=$(grep -E '^\s*max_workers:' "$CONFIG_FILE" | head -1 | awk '{print $2}')

if [ -z "$MAX_WORKERS" ]; then
    MAX_WORKERS=16
    echo "Warning: max_workers not found in config.yaml, using default: $MAX_WORKERS"
fi

echo "Starting Ollama with OLLAMA_NUM_PARALLEL=$MAX_WORKERS"
echo "---"

# Stop existing Ollama if running
if pgrep -x "ollama" > /dev/null; then
    echo "Stopping existing Ollama process..."
    pkill ollama
    sleep 2
fi

# Start Ollama with the configured parallelism
export OLLAMA_NUM_PARALLEL=$MAX_WORKERS
exec ollama serve
