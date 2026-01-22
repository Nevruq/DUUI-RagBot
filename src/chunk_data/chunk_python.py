"""
Python code chunker (AST-based)

- Chunks by: module-level functions, classes, and class methods
- Includes optional "header context" (imports + module docstring) for each chunk
- Adds metadata: file, symbol_type, symbol_name, start_line, end_line
- Works on Python 3.8+ (uses end_lineno when available)
"""

from __future__ import annotations

import ast
import json
import os
from typing import Dict, List, Optional, Tuple

import llm_wrapper
from rag_chunk import RAGChunk


def _safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True)


def _get_module_header(tree: ast.AST, lines: List[str], max_header_lines: int = 80) -> str:
    """
    Build a small header context:
    - module docstring (if any)
    - import statements (top-level)
    Limited by max_header_lines to avoid huge duplication.
    """
    header_nodes: List[ast.AST] = []

    doc = ast.get_docstring(tree, clean=False)
    if doc:
        header_nodes.append(("__DOCSTRING__", doc))

    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header_nodes.append(node)

    header_lines: List[str] = []
    for node in header_nodes:
        if isinstance(node, tuple) and node[0] == "__DOCSTRING__":
            header_lines.append('"""' + str(node[1]).strip() + '"""\n\n')
        else:
            try:
                seg = ast.get_source_segment("".join(lines), node)
            except Exception:
                seg = None
            if seg:
                header_lines.append(seg.strip() + "\n")
            else:
                if isinstance(node, ast.Import):
                    names = ", ".join(alias.name for alias in node.names)
                    header_lines.append(f"import {names}\n")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(alias.name for alias in node.names)
                    header_lines.append(f"from {module} import {names}\n")

    joined = "".join(header_lines)
    limited = _split_lines(joined)[:max_header_lines]
    return "".join(limited).strip() + ("\n\n" if limited else "")


def _node_span(node: ast.AST) -> Optional[Tuple[int, int]]:
    """
    Returns (start_line, end_line) 1-indexed, inclusive.
    Requires Python 3.8+ for end_lineno; otherwise returns None.
    """
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return None
    return int(start), int(end)


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
    """
    Generates a short description of a code section for the Metadata of the RAG properties.
    """
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
    language: str = "python",
    llm_data: Optional[Dict[str, object]] = None,
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
        "code_description": description,
        "keywords": keywords,
    }


def chunk_python_code(
    code: str,
    file_path: str = "<memory>",
    include_header: bool = True,
    header_max_lines: int = 80,
    include_methods: bool = True,
    disable_llm: bool = False,
) -> List[RAGChunk]:
    """
    Chunk python code into logical units: functions, classes, (optionally) methods.

    Returns list of RAGChunk objects with explicit fields.
    """
    lines = _split_lines(code)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        llm_code_description = _gen_code_description(code)
        return [
            RAGChunk(
                text=code,
                **_build_chunk_fields(
                    file_path=file_path,
                    symbol_type="file_fallback",
                    symbol_name=os.path.basename(file_path),
                    start_line=1,
                    end_line=len(lines),
                    llm_data=llm_code_description,
                ),
            )
        ]

    header = _get_module_header(tree, lines, max_header_lines=header_max_lines) if include_header else ""

    chunks: List[RAGChunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = _node_span(node)
            if not span:
                continue
            start, end = span
            body_text = _slice_lines(lines, start, end)

            text = (header + body_text) if include_header else body_text
            llm_code_description = _gen_code_description(text)

            chunks.append(
                RAGChunk(
                    text=text,
                    **_build_chunk_fields(
                        file_path=file_path,
                        symbol_type="function" if isinstance(node, ast.FunctionDef) else "async_function",
                        symbol_name=node.name,
                        start_line=start,
                        end_line=end,
                        llm_data=llm_code_description,
                    ),
                )
            )

        elif isinstance(node, ast.ClassDef):
            span = _node_span(node)
            if not span:
                continue
            c_start, c_end = span
            class_text = _slice_lines(lines, c_start, c_end)

            text = (header + class_text) if include_header else class_text
            llm_code_description = _gen_code_description(text)
            chunks.append(
                RAGChunk(
                    text=text,
                    **_build_chunk_fields(
                        file_path=file_path,
                        symbol_type="class",
                        symbol_name=node.name,
                        start_line=c_start,
                        end_line=c_end,
                        llm_data=llm_code_description,
                    ),
                )
            )

            if include_methods:
                for inner in node.body:
                    if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_span = _node_span(inner)
                        if not m_span:
                            continue
                        m_start, m_end = m_span
                        method_text = _slice_lines(lines, m_start, m_end)
                        text = (header + method_text) if include_header else method_text
                        llm_code_description = _gen_code_description(text)
                        chunks.append(
                            RAGChunk(
                                text=text,
                                **_build_chunk_fields(
                                    file_path=file_path,
                                    symbol_type="method" if isinstance(inner, ast.FunctionDef) else "async_method",
                                    symbol_name=f"{node.name}.{inner.name}",
                                    start_line=m_start,
                                    end_line=m_end,
                                    llm_data=llm_code_description,
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
                    symbol_name=os.path.basename(file_path),
                    start_line=1,
                    end_line=len(lines),
                ),
            )
        )

    return chunks


def chunk_python_file(
    path: str,
    include_header: bool = True,
    header_max_lines: int = 80,
    include_methods: bool = True,
) -> List[RAGChunk]:
    code = _safe_read(path)
    return chunk_python_code(
        code,
        file_path=path,
        include_header=include_header,
        header_max_lines=header_max_lines,
        include_methods=include_methods,
    )
