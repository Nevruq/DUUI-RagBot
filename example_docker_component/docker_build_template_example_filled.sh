#!/bin/bash
# DUUI Docker Build Script
# ============================================
# Generated from Hugging Face Model JSON:
#   model_id: debajyotimaz/codemix_hate
#   model_type: bert
#   architectures: BertForSequenceClassification
#   inferred_task: text-classification
#   num_labels: 2
#   id2label: {"0": "Non-hateful", "1": "Hateful"}
#
# COMPONENT_NAME: hate
# ============================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# COMPONENT_NAME: hate -> duui-hate
IMAGE_NAME="duui-hate"

# VERSION: 0.1.0 (default)
VERSION="0.1.0"

# PORT: 9714 (default, matches Dockerfile)
PORT=9714

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
    echo "Build successful!"
    echo "Run with: docker run -p ${PORT}:${PORT} ${IMAGE_NAME}:latest"
else
    echo "Build failed!"
    exit 1
fi
