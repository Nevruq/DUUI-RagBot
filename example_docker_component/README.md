# Minimal DUUI Hate Detection Component

This is a minimal example of a DUUI component using the HuggingFace model `debajyotimaz/codemix_hate`.

## Component Files

### Core Files (Required)

1. **[duui_hate.py](duui_hate.py)** - FastAPI service that:
   - Loads the HuggingFace hate detection model
   - Exposes DUUI-required endpoints: `/v1/typesystem`, `/v1/communication_layer`, `/v1/process`
   - Processes text and returns hate speech predictions

2. **[duui_model.lua](duui_model.lua)** - Lua communication script that:
   - `serialize()`: Extracts text from UIMA CAS and sends it to Python service
   - `deserialize()`: Receives predictions and creates annotations in CAS
   - **Uses CAS API** (not JCas) to avoid needing Java class files

3. **[TypeSystem.xml](TypeSystem.xml)** - UIMA type definitions:
   - Defines `org.example.HateSpeech` annotation type
   - Features: `label` (String) and `score` (Double)

4. **[requirements.txt](requirements.txt)** - Minimal Python dependencies:
   - transformers, torch, fastapi, uvicorn, dkpro-cassis, pydantic

5. **[Dockerfile](Dockerfile)** - Container definition:
   - Pre-downloads model for faster startup
   - Exposes port 9714
   - Runs uvicorn server

## Building and Running

```bash
# Build Docker image
docker build -t duui-hate-detection .

# Run container
docker run -p 9714:9714 duui-hate-detection
```

## Testing

### Automated Test Suite

Run all component tests:

```bash
# Run complete test suite
./run_tests.sh
```

### Individual Component Tests

**1. Test TypeSystem.xml:**
```bash
python3 test_typesystem.py
# Validates TypeSystem structure and features
```

**2. Test Lua Script:**
```bash
./test_lua.sh
# Checks syntax and required functions
```

**3. Test Python Service (requires running service):**
```bash
# First, start the service:
python duui_hate.py
# or
docker run -p 9714:9714 duui-hate-detection:latest

# Then in another terminal:
python3 test_service.py
# Tests all DUUI endpoints
```

### Manual Testing with curl

```bash
# Get TypeSystem
curl http://localhost:9714/v1/typesystem

# Get Lua communication layer
curl http://localhost:9714/v1/communication_layer

# Test hate detection
curl -X POST http://localhost:9714/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "selections": [{
      "selection": "text",
      "sentences": [{
        "text": "I hate you and everyone like you",
        "begin": 0,
        "end": 32
      }]
    }],
    "lang": "en",
    "doc_len": 32
  }'
```

### Test Files

- [test_typesystem.py](test_typesystem.py) - Validates TypeSystem.xml with dkpro-cassis
- [test_lua.sh](test_lua.sh) - Checks Lua script syntax and structure
- [test_service.py](test_service.py) - Tests all FastAPI endpoints
- [run_tests.sh](run_tests.sh) - Master test runner

## Files to Ignore

- `hatechecker.py` - Old complex implementation with hardcoded model mappings (not used)
- `reqiurements.txt` - Typo version of requirements file (use `requirements.txt` instead)
- `ablauf.txt` - Notes file
- `docker_build.sh` - Optional build script

## Key Design Decisions

1. **CAS vs JCas**: Uses CAS API in Lua to avoid requiring Java class files for the custom type
2. **Minimal dependencies**: Only includes strictly necessary packages
3. **Direct model loading**: No complex abstraction layers or caching mechanisms
4. **Single model focus**: Tailored specifically for `debajyotimaz/codemix_hate`
