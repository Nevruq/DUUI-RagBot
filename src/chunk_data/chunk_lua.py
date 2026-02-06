"""
Lua code chunker (simple function split)

- Chunks by: serialize and deserialize functions only
- Includes optional header context (top-of-file lines before first function)
- Adds metadata: file, symbol_type, symbol_name, start_line, end_line
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import llm_wrapper
import utils
from chunk_data.rag_chunk import RAGChunk, make_repo_id


_FUNC_RE = re.compile(r"^\s*function\s+(\w+)\b")


def _safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True)


def _find_function_line(lines: List[str], name: str) -> Optional[int]:
    for i, line in enumerate(lines):
        m = _FUNC_RE.match(line)
        if m and m.group(1) == name:
            return i + 1
    return None


def _slice_lines(lines: List[str], start_line: int, end_line: int) -> str:
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "".join(lines[start_idx:end_idx])


def _gen_code_description(code: str) -> dict:
    llm = llm_wrapper.LLMWrapper()
    response = llm.llm_code_description(code)
    return response


def _build_chunk_fields(
    *,
    file_path: str,
    symbol_type: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
    language: str = "lua",
    llm_data: Optional[Dict[str, object]] = None,
    chunk_type: str = "code",
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


def _get_lua_header(lines: List[str], first_func_line: int, max_header_lines: int = 80) -> str:
    if first_func_line <= 1:
        return ""
    header_lines = lines[: first_func_line - 1]
    if len(header_lines) > max_header_lines:
        header_lines = header_lines[:max_header_lines]
    return "".join(header_lines).strip() + ("\n\n" if header_lines else "")


def chunk_lua_code(
    code: str,
    file_path: str = "<memory>",
    include_header: bool = True,
    header_max_lines: int = 80,
    deferred_llm: bool = False,
    repo_root: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> List[RAGChunk]:
    lines = _split_lines(code)
    if repo_id:
        effective_repo_id = repo_id
    else:
        if repo_root is None:
            repo_root = utils.find_repo_root(file_path)
        effective_repo_id = make_repo_id(os.path.abspath(repo_root)) if repo_root else "repo::unknown"

    serialize_line = _find_function_line(lines, "serialize")
    deserialize_line = _find_function_line(lines, "deserialize")

    if serialize_line is None or deserialize_line is None:
        llm_data = None if deferred_llm else _gen_code_description(code)
        return [
            RAGChunk(
                text=code,
                **_build_chunk_fields(
                    file_path=file_path,
                    symbol_type="file",
                    symbol_name=os.path.basename(file_path),
                    start_line=1,
                    end_line=len(lines),
                    language="lua",
                    llm_data=llm_data,
                    chunk_type="lua",
                    repo_id=effective_repo_id,
                ),
            )
        ]

    first_func_line = min(serialize_line, deserialize_line)
    header = _get_lua_header(lines, first_func_line, max_header_lines=header_max_lines) if include_header else ""

    chunks: List[RAGChunk] = []

    # Serialize chunk
    serialize_end = max(deserialize_line - 1, serialize_line)
    serialize_text = _slice_lines(lines, serialize_line, serialize_end)
    serialize_text = (header + serialize_text) if include_header else serialize_text
    serialize_llm = None if deferred_llm else _gen_code_description(serialize_text)
    chunks.append(
        RAGChunk(
            text=serialize_text,
            **_build_chunk_fields(
                file_path=file_path,
                symbol_type="function",
                symbol_name="serialize",
                start_line=serialize_line,
                end_line=serialize_end,
                language="lua",
                llm_data=serialize_llm,
                chunk_type="lua",
                repo_id=effective_repo_id,
            ),
        )
    )

    # Deserialize chunk
    deserialize_end = len(lines)
    deserialize_text = _slice_lines(lines, deserialize_line, deserialize_end)
    deserialize_text = (header + deserialize_text) if include_header else deserialize_text
    deserialize_llm = None if deferred_llm else _gen_code_description(deserialize_text)
    chunks.append(
        RAGChunk(
            text=deserialize_text,
            **_build_chunk_fields(
                file_path=file_path,
                symbol_type="function",
                symbol_name="deserialize",
                start_line=deserialize_line,
                end_line=deserialize_end,
                language="lua",
                llm_data=deserialize_llm,
                chunk_type="lua",
                repo_id=effective_repo_id,
            ),
        )
    )

    return chunks


def chunk_lua_file(
    path: str,
    include_header: bool = True,
    header_max_lines: int = 80,
    deferred_llm: bool = False,
    repo_root: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> List[RAGChunk]:
    code = _safe_read(path)
    return chunk_lua_code(
        code,
        file_path=path,
        include_header=include_header,
        header_max_lines=header_max_lines,
        deferred_llm=deferred_llm,
        repo_root=repo_root,
        repo_id=repo_id,
    )
