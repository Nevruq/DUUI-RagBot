#!/bin/bash
# Minimal Docker build script for DUUI Hate Detection Component

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

IMAGE_NAME="duui-hate-detection"
VERSION="0.1.0"

echo "Building Docker image: ${IMAGE_NAME}:${VERSION}"
echo "Build context: ${SCRIPT_DIR}"

# Change to script directory to ensure correct context
cd "${SCRIPT_DIR}"

docker build \
  -t ${IMAGE_NAME}:${VERSION} \
  -t ${IMAGE_NAME}:latest \
  -f Dockerfile \
  .

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Run with: docker run -p 9714:9714 ${IMAGE_NAME}:latest"
else
    echo "✗ Build failed!"
    exit 1
fi