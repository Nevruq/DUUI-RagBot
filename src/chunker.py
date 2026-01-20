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
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import hashlib
import ollama


@dataclass
class CodeChunk:
    text: str
    meta: Dict[str, object]


def _safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_lines(text: str) -> List[str]:
    # keep line endings so slicing preserves formatting
    return text.splitlines(keepends=True)


def _get_module_header(tree: ast.AST, lines: List[str], max_header_lines: int = 80) -> str:
    """
    Build a small header context:
    - module docstring (if any)
    - import statements (top-level)
    Limited by max_header_lines to avoid huge duplication.
    """
    header_nodes: List[ast.AST] = []

    # module docstring
    doc = ast.get_docstring(tree, clean=False)
    if doc:
        # best-effort: include the docstring as text (not exact original quoting)
        header_nodes.append(("__DOCSTRING__", doc))  # sentinel tuple

    # top-level imports
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header_nodes.append(node)

    header_lines: List[str] = []
    for node in header_nodes:
        if isinstance(node, tuple) and node[0] == "__DOCSTRING__":
            header_lines.append('"""' + str(node[1]).strip() + '"""\n\n')
        else:
            # try to reconstruct import line(s)
            try:
                seg = ast.get_source_segment("".join(lines), node)
            except Exception:
                seg = None
            if seg:
                header_lines.append(seg.strip() + "\n")
            else:
                # fallback rough import text
                if isinstance(node, ast.Import):
                    names = ", ".join(alias.name for alias in node.names)
                    header_lines.append(f"import {names}\n")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(alias.name for alias in node.names)
                    header_lines.append(f"from {module} import {names}\n")

    # limit header size
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
    # start_line/end_line are 1-indexed inclusive
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "".join(lines[start_idx:end_idx])

from typing import Any

def chunk_to_chroma_item(
    chunk_text: str,
    meta: Dict[str, Any],
    *,
    id_mode: str = "stable_hash",   # "stable_hash" or "symbol_lines"
    id_prefix: str = "code"
) -> Dict[str, Any]:
    """
    Convert one code chunk (+metadata) into a Chroma-ready item:
      - id: str
      - document: str
      - metadata: dict

    Returns a dict you can later batch into:
      collection.add(ids=[...], documents=[...], metadatas=[...])

    Parameters
    ----------
    chunk_text : str
        The chunk content.
    meta : dict
        Chunk metadata, ideally including: file, symbol_type, symbol_name, start_line, end_line, language
    id_mode : str
        - "symbol_lines": deterministic id based on file/symbol/lines
        - "stable_hash": deterministic hash of key fields + chunk text
    id_prefix : str
        Prefix for the id (useful when mixing sources).
    """
    file_path = str(meta.get("file", "unknown_file"))
    symbol_type = str(meta.get("symbol_type", "unknown_type"))
    symbol_name = str(meta.get("symbol_name", "unknown_symbol"))
    start_line = int(meta.get("start_line", 0))
    end_line = int(meta.get("end_line", 0))
    language = str(meta.get("language", "python"))

    # Build deterministic ID
    if id_mode == "symbol_lines":
        # Good when file+symbol+line spans are stable across runs
        raw_id = f"{id_prefix}::{file_path}::{symbol_type}::{symbol_name}::{start_line}-{end_line}"
        # sanitize a bit for Chroma
        chunk_id = raw_id.replace("\\", "/")
    elif id_mode == "stable_hash":
        # More robust when line numbers might shift; still deterministic for identical inputs
        base = f"{file_path}|{symbol_type}|{symbol_name}|{start_line}|{end_line}|{language}|{chunk_text}"
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]
        chunk_id = f"{id_prefix}::{h}"
    else:
        raise ValueError("id_mode must be 'stable_hash' or 'symbol_lines'")

    # Ensure metadata is JSON-serializable and compact
    chroma_meta = {
        "file": file_path.replace("\\", "/"),
        "language": language,
        "symbol_type": symbol_type,
        "symbol_name": symbol_name,
        "start_line": start_line,
        "end_line": end_line,
    }

    # Optionally carry through extra metadata keys
    for k, v in meta.items():
        if k in chroma_meta:
            continue
        # keep only simple JSON-safe values
        if isinstance(v, (str, int, float, bool)) or v is None:
            chroma_meta[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
            chroma_meta[k] = list(v)
        # else: skip complex objects

    return {
        "id": chunk_id,
        "document": chunk_text,
        "metadata": chroma_meta,
    }

import llm_wrapper
import utils
from pydantic import BaseModel


def _parse_description_response(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "description" in parsed and "keywords" in parsed:
                return parsed
            if "codeDescription" in parsed and "keywords" in parsed:
                return {"description": parsed["codeDescription"], "keywords": parsed["keywords"]}
    except json.JSONDecodeError:
        pass

    description = ""
    keywords = []
    for line in raw.splitlines():
        if line.lower().startswith("description:"):
            description = line.split(":", 1)[1].strip()
        elif line.lower().startswith("codedescription:"):
            description = line.split(":", 1)[1].strip()
        elif line.lower().startswith("keywords:"):
            kw_text = line.split(":", 1)[1].strip()
            try:
                keywords = json.loads(kw_text)
            except json.JSONDecodeError:
                kw_text = kw_text.strip("[]")
                keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

    if not description:
        description = raw.strip()
    if not isinstance(keywords, list):
        keywords = []
    if not keywords:
        keywords = ["file: unknown", "code", "summary"]

    return {"description": description, "keywords": keywords}


def _gen_code_description(code: str) -> dict:
    """
    Generates a short description of a code section for the Metadata of the RAG properties.
    """
    class metadatasRag(BaseModel):
        codeDescription: str
        keywords: list[str]

    # Angepasste Prompt für LLM aufruf
    llm = llm_wrapper.LLMWrapper()

    # load prompt 
    prompt_code_description = utils.load_prompt_template("src/prompts/code_section_summary.txt")

    response = llm.gen_response_formatted(code, metadatasRag, prompt_code_description)

    return _parse_description_response(response)


def chunk_python_code(
    code: str,
    file_path: str = "<memory>",
    include_header: bool = True,
    header_max_lines: int = 80,
    include_methods: bool = True,
) -> List[CodeChunk]:
    """
    Chunk python code into logical units: functions, classes, (optionally) methods.

    Returns list of CodeChunk(text, meta).
    """
    lines = _split_lines(code)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # fallback: one big chunk if code can't be parsed
        llm_code_description = _gen_code_description(code)

        return [
            CodeChunk(
                text=code,
                meta={
                    "file": file_path,
                    "language": "python",
                    "symbol_type": "file_fallback",
                    "symbol_name": os.path.basename(file_path),
                    "start_line": 1,
                    "end_line": len(lines),
                    "code_description": llm_code_description["description"],
                    "keywords": llm_code_description["keywords"]
                },
            )
        ]

    header = _get_module_header(tree, lines, max_header_lines=header_max_lines) if include_header else ""

    chunks: List[CodeChunk] = []

    for node in tree.body:
        # Top-level functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = _node_span(node)
            if not span:
                continue
            start, end = span
            body_text = _slice_lines(lines, start, end)

            text=(header + body_text) if include_header else body_text
            # Generate Additional Metainformation
            llm_code_description = _gen_code_description(text)

            chunks.append(
                CodeChunk(
                    text=text,
                    meta={
                        "file": file_path,
                        "language": "python",
                        "symbol_type": "function" if isinstance(node, ast.FunctionDef) else "async_function",
                        "symbol_name": node.name,
                        "start_line": start,
                        "end_line": end,
                        "code_description": llm_code_description["description"],
                        "keywords": llm_code_description["keywords"]
                    },
                )
            )

        # Top-level classes (+ methods)
        elif isinstance(node, ast.ClassDef):
            span = _node_span(node)
            if not span:
                continue
            c_start, c_end = span
            class_text = _slice_lines(lines, c_start, c_end)

            # Chunk for the whole class
            chunks.append(
                CodeChunk(
                    text=(header + class_text) if include_header else class_text,
                    meta={
                        "file": file_path,
                        "language": "python",
                        "symbol_type": "class",
                        "symbol_name": node.name,
                        "start_line": c_start,
                        "end_line": c_end,
                    },
                )
            )

            # Optional: chunk each method separately (often best for RAG)
            if include_methods:
                for inner in node.body:
                    if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_span = _node_span(inner)
                        if not m_span:
                            continue
                        m_start, m_end = m_span
                        method_text = _slice_lines(lines, m_start, m_end)
                        text=(header + method_text) if include_header else method_text
                        llm_code_description = _gen_code_description(text)
                        chunks.append(
                            CodeChunk(
                                text=text,
                                meta={
                                    "file": file_path,
                                    "language": "python",
                                    "symbol_type": "method" if isinstance(inner, ast.FunctionDef) else "async_method",
                                    "symbol_name": f"{node.name}.{inner.name}",
                                    "start_line": m_start,
                                    "end_line": m_end,
                                    "code_description": llm_code_description["description"],
                                    "keywords": llm_code_description["keywords"]
                                },
                            )
                        )

    # If nothing found, fallback to full file
    if not chunks:
        chunks.append(
            CodeChunk(
                text=code,
                meta={
                    "file": file_path,
                    "language": "python",
                    "symbol_type": "file",
                    "symbol_name": os.path.basename(file_path),
                    "start_line": 1,
                    "end_line": len(lines),
                },
            )
        )

    return chunks


def chunk_python_file(
    path: str,
    include_header: bool = True,
    header_max_lines: int = 80,
    include_methods: bool = True,
) -> List[CodeChunk]:
    code = _safe_read(path)
    return chunk_python_code(
        code,
        file_path=path,
        include_header=include_header,
        header_max_lines=header_max_lines,
        include_methods=include_methods,
    )



if __name__ == "__main__":
    import ollama

    response = ollama.embed(
        model='mxbai-embed-large',
        input='The sky is blue because of Rayleigh scattering',
    )
    print(response.embeddings)
    p = "src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"

    #chunks = chunk_python_file(p, include_header=True, include_methods=True)
    #print(chunks)

