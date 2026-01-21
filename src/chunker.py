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

LLM_DISABLED = os.getenv("LLM_DISABLE", "").lower() in {"1", "true", "yes"}

@dataclass
class CodeChunk:
    text: str
    file: str
    language: str
    symbol_type: str
    symbol_name: str
    start_line: int
    end_line: int
    code_description: str
    keywords: List[str]

    @property
    def meta(self) -> Dict[str, object]:
        return {
            "code": self.text,
            "file": self.file,
            "language": self.language,
            "symbol_type": self.symbol_type,
            "symbol_name": self.symbol_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code_description": self.code_description,
            "keywords": ", ".join(self.keywords),
            #TODO implementiere Intent attribut
        }

    def gen_embedding_meta(self):
        """
        First approach for Embedding: Meta(language, code_description, keywords) #todo vergleiche wie mit code
        """
        embedding_string = f"""
            name: {self.file}
            summary: {self.code_description}
            keywords: {self.keywords}
        """
        # Embedding using ollama model
        return ollama.embed(
                    model='mxbai-embed-large',
                    input=embedding_string
                    ).embeddings[0]


    def gen_embedding_code(self):
        """
        First approach for Embedding: Meta(language, code_description, keywords) #todo vergleiche wie mit code
        """
        #TODO
        embedding_string = f"""
        
        """

    def to_chroma_item(
        self,
        *,
        id_mode: str = "stable_hash",
        id_prefix: str = "code",
    ) -> Dict[str, object]:
        """
        Convert one code chunk (+metadata) into a Chroma-ready item:
          - id: str
          - document: str
          - metadata: dict
        """
        file_path = str(self.file)
        symbol_type = str(self.symbol_type)
        symbol_name = str(self.symbol_name)
        start_line = int(self.start_line)
        end_line = int(self.end_line)
        language = str(self.language)
        code_description = str(self.code_description)
        keywords = str(", ".join(self.keywords))

        # Build deterministic ID
        if id_mode == "symbol_lines":
            raw_id = f"{id_prefix}::{file_path}::{symbol_type}::{symbol_name}::{start_line}-{end_line}"
            chunk_id = raw_id.replace("\\", "/")
        elif id_mode == "stable_hash":
            base = f"{file_path}|{symbol_type}|{symbol_name}|{start_line}|{end_line}|{language}|{self.text}"
            h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]
            chunk_id = f"{id_prefix}::{h}"
        else:
            raise ValueError("id_mode must be 'stable_hash' or 'symbol_lines'")

        chroma_meta = {
            "file": file_path.replace("\\", "/"),
            "language": language,
            "symbol_type": symbol_type,
            "symbol_name": symbol_name,
            "start_line": start_line,
            "end_line": end_line,
            "code_description": code_description,
            "keywords": keywords,
        }

        return {
            "id": chunk_id,
            "embedding": self.gen_embedding_meta(),
            "document": self.text,
            "metadata": chroma_meta
        }


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
    chunk: CodeChunk,
    *,
    id_mode: str = "stable_hash",
    id_prefix: str = "code"
) -> Dict[str, Any]:
    return chunk.to_chroma_item(id_mode=id_mode, id_prefix=id_prefix)

import llm_wrapper
import utils
from pydantic import BaseModel


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

    description = "N.A"
    keywords = ["N.A"]

    return {"description": description, "keywords": keywords}


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
    disable_llm: bool = False
) -> List[CodeChunk]:
    """
    Chunk python code into logical units: functions, classes, (optionally) methods.

    Returns list of CodeChunk objects with explicit fields.
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

        # Top-level classes (+ methods)
        elif isinstance(node, ast.ClassDef):
            span = _node_span(node)
            if not span:
                continue
            c_start, c_end = span
            class_text = _slice_lines(lines, c_start, c_end)

            text=(header + class_text) if include_header else class_text
            llm_code_description = _gen_code_description(text)
            # Chunk for the whole class
            chunks.append(
                CodeChunk(
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

    # If nothing found, fallback to full file
    if not chunks:
        chunks.append(
            CodeChunk(
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
    print(response.embeddings[0])
    p = "src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"

    #chunks = chunk_python_file(p, include_header=True, include_methods=True)
    #print(chunks)
