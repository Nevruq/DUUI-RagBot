# DUUI-RagBot

Was ist der Use-Case?
Definiere den Aufbau, wie man DUUI setup baut
  - Alle komponenten die dazu gehören
Defniere 

# TODO
    - Implementiere optimizierte Embedding Methode 
      - Überprüfe welche Sematnische Zusammensetzung (exp. descrip. + code usw.)
    - Optimierte RAG-Einträge (siehe unten)

# Embedding
  - Ollama-modell
  - Nutze Code-Description
  - todo: keyword search, top-k search, weitere filter


# Ablauf RAG-Abfrage-Verbesserungen
    - Filestruktur
    - Generiere Codebeschreibungen für jeden Eintrag für bessere Semantische zuordnung
    
# Mögliche queryoptionen
    - Key-word searching (konkrete query filter)
    - Frage rewriten
    - Hybrid retrieval
    - reranker : https://jasonkang14.github.io/llm/how-to-use-llm-as-a-reranker


# Mögliche RAG-Struktur
    Quellen: https://www.reddit.com/r/Rag/comments/1jdmszc/best_embedding_model_for_code_text_documents_in/

    {
  "id": "myrepo:9f3a1c2:src/rag/chunking.py::chunk_python_code@120-210",
  "repo": "myrepo",
  "version": "9f3a1c2",
  "file_path": "src/rag/chunking.py",
  "start_line": 120,
  "end_line": 210,
  "language": "python",
  "type": "function",

  "name": "chunk_python_code",
  "qualified_name": "rag.chunking.chunk_python_code",
  "signature": "chunk_python_code(code: str, max_tokens: int, overlap: int = 50) -> list[str]",
  "visibility": "public",
  "decorators": [],

  "llm_summary": "Splits Python source code into token-limited chunks while preserving semantic boundaries (e.g., functions/classes) for indexing in a RAG pipeline.",
  "llm_keywords": ["chunking", "python", "code splitting", "RAG", "ChromaDB", "token limit", "AST", "overlap"],
  "tags": ["rag", "chunking", "code-indexing", "python"],
  "intent": "chunk python source for vector indexing",
  "usage_examples_nl": [
    "Wie kann ich Python-Code sinnvoll chunken für RAG?",
    "Welche Funktion macht chunking nach Funktionsgrenzen?",
    "Wie erzeuge ich token-limitierte Code-Chunks für ChromaDB?"
  ],

  "docstring": "Chunks Python code for retrieval by splitting along semantic boundaries.",
  "imports_used": ["ast", "tiktoken"],
  "dependencies_internal": ["count_tokens", "split_by_ast_nodes"],
  "raises": ["ValueError if max_tokens <= 0"],
  "side_effects": "none",

  "code": "def chunk_python_code(code: str, max_tokens: int, overlap: int = 50) -> list[str]:\n    ...",

  "chunk_strategy": "function-level",
  "chunk_hash": "sha256:....",
  "embedding_model": "bge-m3",
  "created_at": "2026-01-20T12:00:00Z",
  "source": "parser+llm_augmented",

  "related_symbols": ["chunk_file", "chunk_by_tokens"],
  "tests": ["tests/test_chunking.py::test_chunk_python_code_basic"],
  "examples_code": ["src/rag/indexer.py@55-80"]
}
