#!/bin/bash
# Start Ollama with OLLAMA_NUM_PARALLEL from config.yaml
# Uses the maximum max_workers from all models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# Extract all max_workers values and find the maximum
# (Each model can have different max_workers for optimal parallelism)
ALL_WORKERS=$(grep -E '^\s*max_workers:' "$CONFIG_FILE" | awk '{print $2}')

MAX_WORKERS=16  # Default
for w in $ALL_WORKERS; do
    if [ "$w" -gt "$MAX_WORKERS" ] 2>/dev/null; then
        MAX_WORKERS=$w
    fi
done

if [ -z "$ALL_WORKERS" ]; then
    echo "Warning: max_workers not found in config.yaml, using default: $MAX_WORKERS"
fi

echo "Starting Ollama with OLLAMA_NUM_PARALLEL=$MAX_WORKERS"
echo "---"

# Stop existing Ollama if running
if pgrep -x "ollama" > /dev/null; then
    echo "Stopping existing Ollama process..."
    pkill ollama
    sleep 2
else
    echo "No existing Ollama process found."
fi

# Start Ollama with the configured parallelism
export OLLAMA_NUM_PARALLEL=$MAX_WORKERS
echo "Launching Ollama server..."
ollama serve &

# Wait for server to be ready
echo "Waiting for Ollama to start..."
for i in {1..10}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "---"
        echo "✓ Ollama started successfully with OLLAMA_NUM_PARALLEL=$MAX_WORKERS"
        echo "✓ Server ready at http://localhost:11434"
        exit 0
    fi
    sleep 1
done

echo "Warning: Ollama may not have started correctly. Check logs."
