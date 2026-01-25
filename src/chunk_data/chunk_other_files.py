"""
Chunker for non-code files (README, XML, config, etc.).

- Default: one chunk per file
- Optional: deferred LLM enrichment
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import llm_wrapper
import utils
from chunk_data.rag_chunk import RAGChunk, make_repo_id


def _safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _infer_language(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".md", ".markdown"}:
        return "markdown"
    if ext in {".xml"}:
        return "xml"
    if ext in {".json"}:
        return "json"
    if ext in {".yaml", ".yml"}:
        return "yaml"
    if ext in {".toml"}:
        return "toml"
    if ext in {".ini", ".cfg"}:
        return "ini"
    if ext in {".txt"}:
        return "text"
    return ext.lstrip(".") or "text"

def infer_chunk_type(path: str) -> str:
    path_lower = path.lower()
    if "readme" in path_lower or path_lower.endswith(".md"):
        return "docs"
    if path_lower.endswith((".yml", ".yaml", ".json", ".toml", ".ini", ".cfg")):
        return "config"
    if path_lower.endswith(".xml"):
        if "typesystem" in path_lower:
            return "typesystem"
        return "schema"
    if path_lower.endswith("Dockerfile"):
        return "dockerfile"
    if path_lower.endswith(".py"):
        return "python"
    if path_lower.endswith(".java"):
        return "java"
    if path_lower.endswith((".csv", ".tsv", ".parquet", ".txt")):
        return "data"
    return "other"


def _parse_description_response(raw: str) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "description" in parsed and "keywords" in parsed:
                return parsed
            if "codeDescription" in parsed and "keywords" in parsed:
                return {"description": parsed["codeDescription"], "keywords": parsed["keywords"]}
    except json.JSONDecodeError:
        pass

    return {"description": "N.A", "keywords": ["N.A"]}


def _gen_file_description(text: str) -> dict:
    llm = llm_wrapper.LLMWrapper()
    response = llm.llm_other_file_description(text)
    return _parse_description_response(response)


def _build_chunk_fields(
    *,
    file_path: str,
    symbol_type: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
    language: str,
    llm_data: Optional[Dict[str, object]] = None,
    chunk_type: str = "other",
    repo_id: str = "repo::unknown",
) -> Dict[str, object]:
    description = "N.A"
    keywords = ["N.A"]
    if llm_data:
        description = str(llm_data.get("description", "N.A"))
        keywords = llm_data.get("keywords", ["N.A"])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        elif not isinstance(keywords, list):
            keywords = [str(keywords)]

    return {
        "file": file_path,
        "language": language,
        "symbol_type": symbol_type,
        "symbol_name": symbol_name,
        "start_line": start_line,
        "end_line": end_line,
        "description": description,
        "keywords": keywords,
        "chunk_type": chunk_type,
        "repo_id": repo_id,
    }


def chunk_other_file(
    path: str,
    *,
    deferred_llm: bool = False,
    repo_root: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> List[RAGChunk]:
    text = _safe_read(path)
    lines = text.splitlines()
    language = _infer_language(path)
    chunk_type = infer_chunk_type(path)
    effective_repo_id = make_repo_id(os.path.abspath(repo_root)) if repo_root else "repo::unknown"
    llm_data = None if deferred_llm else _gen_file_description(text)
    chunk = RAGChunk(
        text=text,
        **_build_chunk_fields(
            file_path=path,
            symbol_type="file",
            symbol_name=os.path.basename(path),
            start_line=1,
            end_line=max(len(lines), 1),
            language=language,
            llm_data=llm_data,
            chunk_type=chunk_type,
            repo_id=effective_repo_id,
        ),
    )

    return [chunk]
