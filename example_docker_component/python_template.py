# DUUI Python Component Template
# ============================================
# LLM INSTRUCTIONS:
# Generate a complete Python DUUI component by replacing all {PLACEHOLDERS}
# using the provided Hugging Face model JSON.
#
# INPUT JSON FIELDS TO USE:
# - model_id           -> MODEL_NAME constant
# - inferred_task      -> Determines model class and processing logic
# - architectures      -> Helps determine AutoModel class
# - id2label           -> Label mapping for output
# - num_labels         -> Number of classification labels
#
# NAMING CONVENTION (CRITICAL - must be consistent):
# All files use the same {COMPONENT_NAME}:
#   - Python file:   duui_{COMPONENT_NAME}.py
#   - Lua file:      duui_{COMPONENT_NAME}.lua
#   - Docker image:  duui-{COMPONENT_NAME}
# ============================================

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List
# ============================================
# PLACEHOLDER: {MODEL_IMPORTS}
#
# GENERATION RULE:
# Based on inferred_task:
#   text-classification:
#     from transformers import AutoModelForSequenceClassification, AutoTokenizer
#   token-classification:
#     from transformers import AutoModelForTokenClassification, AutoTokenizer
#   text-generation:
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#
# INPUT USED: inferred_task
# ============================================
{MODEL_IMPORTS}
import torch
from cassis import load_typesystem

# ============================================
# FastAPI App
# PLACEHOLDER: {APP_TITLE}, {APP_DESCRIPTION}
#
# GENERATION RULE:
# Derive from model_id and inferred_task:
#   - APP_TITLE: "DUUI {Task} Detection" (e.g., "DUUI Hate Detection")
#   - APP_DESCRIPTION: "Minimal {task} detection with HuggingFace"
#
# INPUT USED: model_id, inferred_task
# ============================================
app = FastAPI(
    title="{APP_TITLE}",
    description="{APP_DESCRIPTION}",
    version="1.0.0"
)

# ============================================
# Load model and tokenizer
# PLACEHOLDER: {MODEL_ID}, {MODEL_CLASS}
#
# GENERATION RULE:
# MODEL_ID: Use model_id directly from JSON
# MODEL_CLASS: Based on inferred_task:
#   text-classification  -> AutoModelForSequenceClassification
#   token-classification -> AutoModelForTokenClassification
#   text-generation      -> AutoModelForCausalLM
#
# INPUT USED: model_id, inferred_task
# ============================================
MODEL_NAME = "{MODEL_ID}"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on {device}")
model = {MODEL_CLASS}.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# Load TypeSystem
with open("TypeSystem.xml", 'rb') as f:
    typesystem = load_typesystem(f)

# ============================================
# Load Lua script
# PLACEHOLDER: {COMPONENT_NAME}
#
# GENERATION RULE:
# Use component_name for Lua file reference:
#   duui_{COMPONENT_NAME}.lua
#
# INPUT USED: component_name (derived from model_id)
# ============================================
with open("duui_{COMPONENT_NAME}.lua", 'r') as f:
    lua_script = f.read()

# ============================================
# Request/Response Models (STATIC)
# These models are standard for DUUI components
# ============================================
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

# ============================================
# PLACEHOLDER: {RESPONSE_MODEL}
#
# GENERATION RULE:
# For text-classification:
#   class DUUIResponse(BaseModel):
#       begins: List[int]
#       ends: List[int]
#       labels: List[str]
#       scores: List[float]
#       model_name: str
#
# For token-classification (NER):
#   class DUUIResponse(BaseModel):
#       begins: List[int]
#       ends: List[int]
#       labels: List[str]
#       scores: List[float]
#       model_name: str
#
# INPUT USED: inferred_task
# ============================================
{RESPONSE_MODEL}

# ============================================
# Endpoints (STATIC structure)
# ============================================
@app.get("/v1/typesystem")
def get_typesystem():
    xml = typesystem.to_xml().encode("utf-8")
    return Response(content=xml, media_type="application/xml")

@app.get("/v1/communication_layer")
def get_communication_layer():
    return Response(content=lua_script, media_type="text/plain")

# ============================================
# PLACEHOLDER: {DOCUMENTATION_DESCRIPTION}
#
# GENERATION RULE:
# Derive from model_id and inferred_task:
#   "{Task} detection using HuggingFace transformer"
#
# INPUT USED: model_id, inferred_task
# ============================================
@app.get("/v1/documentation")
def get_documentation():
    return {"description": "{DOCUMENTATION_DESCRIPTION}"}

# ============================================
# PLACEHOLDER: {PROCESS_LOGIC}
#
# GENERATION RULE:
# Based on inferred_task, generate appropriate processing logic.
#
# For text-classification:
#   - Tokenize input text
#   - Run model inference
#   - Apply softmax to logits
#   - Get argmax for predicted label
#   - Use id2label mapping for label name
#   - Return begins, ends, labels, scores
#
# For token-classification (NER):
#   - Tokenize with return_offsets_mapping=True
#   - Run model inference
#   - Process each token's prediction
#   - Map token offsets to character offsets
#   - Return entity spans with labels and scores
#
# INPUT USED: inferred_task, id2label, recommended_postprocess
# ============================================
@app.post("/v1/process")
def process(request: DUUIRequest):
{PROCESS_LOGIC}

# ============================================
# Main entry point
# PLACEHOLDER: {PORT}
#
# GENERATION RULE:
# Default: 9714
# Must be consistent with Dockerfile and docker_build.sh
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port={PORT})
