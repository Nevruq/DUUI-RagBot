# DUUI Python Component
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

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List
# MODEL_IMPORTS: For text-classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from cassis import load_typesystem

# FastAPI App
# APP_TITLE: Derived from "hate" task
# APP_DESCRIPTION: Based on model purpose
app = FastAPI(
    title="DUUI Hate Detection",
    description="Hate speech detection with HuggingFace transformer",
    version="1.0.0"
)

# Load model and tokenizer
# MODEL_ID: debajyotimaz/codemix_hate
# MODEL_CLASS: AutoModelForSequenceClassification (for text-classification)
MODEL_NAME = "debajyotimaz/codemix_hate"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on {device}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# Load TypeSystem
with open("TypeSystem.xml", 'rb') as f:
    typesystem = load_typesystem(f)

# Load Lua script
# COMPONENT_NAME: hate -> duui_hate.lua
with open("duui_hate.lua", 'r') as f:
    lua_script = f.read()

# Request/Response Models (STATIC)
class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int

class UimaSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]

class DUUIRequest(BaseModel):
    selections: List[UimaSelection]
    lang: str
    doc_len: int

# RESPONSE_MODEL: For text-classification
class DUUIResponse(BaseModel):
    begins: List[int]
    ends: List[int]
    labels: List[str]
    scores: List[float]
    model_name: str

# Endpoints
@app.get("/v1/typesystem")
def get_typesystem():
    xml = typesystem.to_xml().encode("utf-8")
    return Response(content=xml, media_type="application/xml")

@app.get("/v1/communication_layer")
def get_communication_layer():
    return Response(content=lua_script, media_type="text/plain")

# DOCUMENTATION_DESCRIPTION: Based on task
@app.get("/v1/documentation")
def get_documentation():
    return {"description": "Hate speech detection using HuggingFace transformer"}

# PROCESS_LOGIC: For text-classification with softmax + argmax
@app.post("/v1/process")
def process(request: DUUIRequest):
    begins = []
    ends = []
    labels = []
    scores = []

    for selection in request.selections:
        for sentence in selection.sentences:
            text = sentence.text

            # Tokenize and predict
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_label_idx = torch.argmax(probs).item()
                pred_score = probs[0][pred_label_idx].item()

            # Get label name from id2label
            label_name = model.config.id2label.get(pred_label_idx, f"LABEL_{pred_label_idx}")

            begins.append(sentence.begin)
            ends.append(sentence.end)
            labels.append(label_name)
            scores.append(pred_score)

    return DUUIResponse(
        begins=begins,
        ends=ends,
        labels=labels,
        scores=scores,
        model_name=MODEL_NAME
    )

# Main entry point
# PORT: 9714 (default)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9714)
