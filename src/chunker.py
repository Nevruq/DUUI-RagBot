"""
Unified chunker API.
"""

from __future__ import annotations

from typing import Any, Dict

from src.chunker.rag_chunk import RAGChunk
from src.chunker.chunk_python import (
    _build_chunk_fields,
    _gen_code_description,
    _get_module_header,
    _node_span,
    _parse_description_response,
    _safe_read,
    _slice_lines,
    _split_lines,
    chunk_python_code,
    chunk_python_file,
)
from src.chunker.chunk_java import (
    chunk_java_code,
    chunk_java_file,
)

# Backwards-compatible alias
CodeChunk = RAGChunk


def chunk_to_chroma_item(
    chunk: RAGChunk,
    *,
    id_mode: str = "stable_hash",
    id_prefix: str = "code",
) -> Dict[str, Any]:
    return chunk.to_chroma_item(id_mode=id_mode, id_prefix=id_prefix)
