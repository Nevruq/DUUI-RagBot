# DUUI Dockerfile
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

FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MODEL_DOWNLOAD_COMMAND: For text-classification using model_id
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    AutoModelForSequenceClassification.from_pretrained('debajyotimaz/codemix_hate'); \
    AutoTokenizer.from_pretrained('debajyotimaz/codemix_hate')"

# Copy application files (COMPONENT_NAME: hate)
COPY TypeSystem.xml .
COPY duui_hate.lua .
COPY duui_hate.py .

# PORT: 9714 (default)
EXPOSE 9714

# ENTRYPOINT with module duui_hate:app
ENTRYPOINT ["uvicorn", "duui_hate:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
