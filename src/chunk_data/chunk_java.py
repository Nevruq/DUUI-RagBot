"""
Java code chunker (regex/brace-based)

- Chunks by: classes/interfaces/enums and (optionally) methods
- Includes optional header context (package + imports)
- Adds metadata: file, symbol_type, symbol_name, start_line, end_line
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import llm_wrapper
import utils
from chunk_data.rag_chunk import RAGChunk, make_repo_id


_CLASS_RE = re.compile(
    r"^\s*(public|protected|private)?\s*(abstract|final|static)?\s*(class|interface|enum)\s+(\w+)"
)
_METHOD_RE = re.compile(
    r"^\s*(public|protected|private)?\s*(static|final|abstract|synchronized|native|strictfp|default)?\s*"
    r"[\w\<\>\[\], ?]+\s+(\w+)\s*\([^;]*\)\s*(throws [^{]+)?\{"
)


def _safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True)


def _get_java_header(lines: List[str], max_header_lines: int = 80) -> str:
    header_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("package ") or stripped.startswith("import "):
            header_lines.append(line)
        elif stripped == "" or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            if header_lines:
                continue
        else:
            break
        if len(header_lines) >= max_header_lines:
            break
    return "".join(header_lines).strip() + ("\n\n" if header_lines else "")


def _find_block_end(lines: List[str], start_idx: int) -> Optional[int]:
    brace_count = 0
    started = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                brace_count += 1
                started = True
            elif ch == "}":
                brace_count -= 1
        if started and brace_count == 0:
            return i + 1
    return None


def _slice_lines(lines: List[str], start_line: int, end_line: int) -> str:
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "".join(lines[start_idx:end_idx])


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


def _gen_code_description(code: str) -> dict:
    llm = llm_wrapper.LLMWrapper()
    response = llm.llm_code_description(code)
    return _parse_description_response(response)


def _build_chunk_fields(
    *,
    file_path: str,
    symbol_type: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
    language: str = "java",
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


def chunk_java_code(
    code: str,
    file_path: str = "<memory>",
    include_header: bool = True,
    header_max_lines: int = 80,
    include_methods: bool = True,
    deferred_llm: bool = False,
    repo_root: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> List[RAGChunk]:
    lines = _split_lines(code)
    if repo_root is None:
        repo_root = utils.find_repo_root(file_path)
    effective_repo_id = make_repo_id(os.path.abspath(repo_root)) if repo_root else "repo::unknown"
    header = _get_java_header(lines, max_header_lines=header_max_lines) if include_header else ""

    chunks: List[RAGChunk] = []
    for i, line in enumerate(lines):
        class_match = _CLASS_RE.match(line)
        if not class_match:
            continue
        class_type = class_match.group(3)
        class_name = class_match.group(4)
        c_start = i + 1
        c_end = _find_block_end(lines, i)
        if not c_end:
            continue
        class_text = _slice_lines(lines, c_start, c_end)
        text = (header + class_text) if include_header else class_text
        llm_data = None if deferred_llm else _gen_code_description(text)

        chunks.append(
            RAGChunk(
                text=text,
                **_build_chunk_fields(
                    file_path=file_path,
                    symbol_type=class_type,
                    symbol_name=class_name,
                    start_line=c_start,
                    end_line=c_end,
                    language="java",
                    llm_data=llm_data,
                    chunk_type="java",
                    repo_id=effective_repo_id,
                ),
            )
        )

        if include_methods:
            for j in range(i, c_end):
                method_match = _METHOD_RE.match(lines[j])
                if not method_match:
                    continue
                method_name = method_match.group(3)
                m_start = j + 1
                m_end = _find_block_end(lines, j)
                if not m_end:
                    continue
                method_text = _slice_lines(lines, m_start, m_end)
                text = (header + method_text) if include_header else method_text
                llm_data = None if deferred_llm else _gen_code_description(text)
                symbol_type = "constructor" if method_name == class_name else "method"
                chunks.append(
                    RAGChunk(
                        text=text,
                        **_build_chunk_fields(
                            file_path=file_path,
                            symbol_type=symbol_type,
                            symbol_name=f"{class_name}.{method_name}",
                            start_line=m_start,
                            end_line=m_end,
                            language="java",
                            llm_data=llm_data,
                            chunk_type="java",
                            repo_id=effective_repo_id,
                        ),
                    )
                )

    if not chunks:
        chunks.append(
            RAGChunk(
                text=code,
                **_build_chunk_fields(
                    file_path=file_path,
                    symbol_type="file",
                    symbol_name=file_path.split("/")[-1],
                    start_line=1,
                    end_line=len(lines),
                    language="java",
                    chunk_type="java",
                    repo_id=effective_repo_id,
                ),
            )
        )

    return chunks


def chunk_java_file(
    path: str,
    include_header: bool = True,
    header_max_lines: int = 80,
    include_methods: bool = True,
    deferred_llm: bool = False,
    repo_root: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> List[RAGChunk]:
    code = _safe_read(path)
    return chunk_java_code(
        code,
        file_path=path,
        include_header=include_header,
        header_max_lines=header_max_lines,
        include_methods=include_methods,
        deferred_llm=deferred_llm,
        repo_root=repo_root,
        repo_id=repo_id,
    )
