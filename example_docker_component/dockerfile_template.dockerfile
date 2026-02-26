# DUUI Dockerfile Template
# ============================================
# LLM INSTRUCTIONS:
# Generate a complete Dockerfile by replacing all {PLACEHOLDERS}
# using the provided Hugging Face model JSON.
#
# INPUT JSON FIELDS TO USE:
# - model_id           -> For pre-downloading the model
# - inferred_task      -> Determines which model class to use
# - architectures      -> Helps determine AutoModel class
#
# NAMING CONVENTION (CRITICAL - must be consistent):
# All files use the same {COMPONENT_NAME} derived from model purpose:
#   - Python file:   duui_{COMPONENT_NAME}.py
#   - Lua file:      duui_{COMPONENT_NAME}.lua
#   - Docker image:  duui-{COMPONENT_NAME}
#
# Examples:
#   hate detection  -> COMPONENT_NAME = "hate"
#   sentiment       -> COMPONENT_NAME = "sentiment"
#   sarcasm         -> COMPONENT_NAME = "sarcasm"
#   NER             -> COMPONENT_NAME = "ner"
# ============================================

FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# PLACEHOLDER: {MODEL_DOWNLOAD_COMMAND}
#
# GENERATION RULE:
# Pre-download the HuggingFace model based on inferred_task:
#
# For text-classification:
#   RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
#       AutoModelForSequenceClassification.from_pretrained('{MODEL_ID}'); \
#       AutoTokenizer.from_pretrained('{MODEL_ID}')"
#
# For token-classification (NER):
#   RUN python -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; \
#       AutoModelForTokenClassification.from_pretrained('{MODEL_ID}'); \
#       AutoTokenizer.from_pretrained('{MODEL_ID}')"
#
# For text-generation:
#   RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#       AutoModelForCausalLM.from_pretrained('{MODEL_ID}'); \
#       AutoTokenizer.from_pretrained('{MODEL_ID}')"
#
# INPUT USED: model_id, inferred_task
# ============================================
{MODEL_DOWNLOAD_COMMAND}

# ============================================
# Copy application files
# PLACEHOLDER: {COMPONENT_NAME}
#
# GENERATION RULE:
# Derive from model_id or task:
#   - hate      -> duui_hate.py, duui_hate.lua
#   - sentiment -> duui_sentiment.py, duui_sentiment.lua
#   - sarcasm   -> duui_sarcasm.py, duui_sarcasm.lua
#
# INPUT USED: model_id (to derive component name)
# ============================================
COPY TypeSystem.xml .
COPY duui_{COMPONENT_NAME}.lua .
COPY duui_{COMPONENT_NAME}.py .

# ============================================
# PLACEHOLDER: {PORT}
#
# GENERATION RULE:
# Default port: 9714
# Can be customized if needed
# ============================================
EXPOSE {PORT}

# ============================================
# ENTRYPOINT with Python module name
# PLACEHOLDER: {COMPONENT_NAME}
#
# GENERATION RULE:
# Use the same {COMPONENT_NAME} for module reference:
#   duui_{COMPONENT_NAME}:app
# ============================================
ENTRYPOINT ["uvicorn", "duui_{COMPONENT_NAME}:app", "--host", "0.0.0.0", "--port", "{PORT}"]
CMD ["--workers", "1"]
