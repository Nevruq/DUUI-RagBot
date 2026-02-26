FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model to speed up container startup
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('debajyotimaz/codemix_hate'); AutoTokenizer.from_pretrained('debajyotimaz/codemix_hate')"

# Copy application files
COPY TypeSystem.xml .
COPY duui_model.lua .
COPY duui_hate.py .

EXPOSE 9714

ENTRYPOINT ["uvicorn", "duui_hate:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]

