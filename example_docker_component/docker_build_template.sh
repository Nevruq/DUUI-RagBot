#!/bin/bash
# DUUI Docker Build Script Template
# ============================================
# LLM INSTRUCTIONS:
# Generate a complete build script by replacing all {PLACEHOLDERS}
# using the provided Hugging Face model JSON.
#
# NAMING CONVENTION (CRITICAL - must match Dockerfile):
# All files use the same {COMPONENT_NAME} derived from model purpose:
#   - Docker image:  duui-{COMPONENT_NAME}
#   - Python file:   duui_{COMPONENT_NAME}.py
#   - Lua file:      duui_{COMPONENT_NAME}.lua
#
# Examples:
#   hate detection  -> COMPONENT_NAME = "hate"
#   sentiment       -> COMPONENT_NAME = "sentiment"
#   sarcasm         -> COMPONENT_NAME = "sarcasm"
#   NER             -> COMPONENT_NAME = "ner"
# ============================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================
# PLACEHOLDER: {COMPONENT_NAME}
#
# GENERATION RULE:
# Derive from model_id or task:
#   - hate detection  -> "hate"
#   - sentiment       -> "sentiment"
#   - sarcasm         -> "sarcasm"
#
# Docker image name format: duui-{COMPONENT_NAME}
#
# INPUT USED: model_id (to derive component name)
# ============================================
IMAGE_NAME="duui-{COMPONENT_NAME}"

# ============================================
# PLACEHOLDER: {VERSION}
#
# GENERATION RULE:
# Default: "0.1.0"
# Can be derived from model revision if available
#
# INPUT USED: revision (optional)
# ============================================
VERSION="{VERSION}"

# ============================================
# PLACEHOLDER: {PORT}
#
# GENERATION RULE:
# Default: 9714
# Must match the port in Dockerfile
# ============================================
PORT={PORT}

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
