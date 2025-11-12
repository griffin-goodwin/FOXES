#!/bin/bash

# Helper script to build and run the FOXES Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FOXES Docker Container Setup ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! docker info | grep -q nvidia; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU support may not work."
    echo "Install nvidia-docker2 for GPU support."
    echo ""
fi

# Build the container
echo "Building FOXES Docker container..."
docker build -t foxes:latest .

echo ""
echo "=== Container built successfully! ==="
echo ""
echo "To start the container:"
echo "  docker run -d --name FOXES --gpus all --runtime nvidia \\"
echo "    -e NVIDIA_VISIBLE_DEVICES=all \\"
echo "    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\"
echo "    -v \$(pwd):/workspace/FOXES \\"
echo "    -v /data/FOXES_Data:/data/FOXES_Data \\"
echo "    --shm-size=8gb --restart unless-stopped -it foxes:latest"
echo ""
echo "To access the container:"
echo "  docker exec -it FOXES bash"
echo ""
echo "To stop the container:"
echo "  docker stop FOXES"
echo ""
echo "For more information, see DOCKER_SETUP.md"

