from __future__ import annotations
# Eigentlich irrelvant für Python 3.13

from dataclasses import dataclass
from typing import Dict, List
import hashlib

import ollama


@dataclass
class RAGChunk:
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
        }

    def gen_embedding_meta(self):
        embedding_string = f"""
            name: {self.file}
            summary: {self.code_description}
            keywords: {self.keywords}
        """
        return ollama.embed(
            model="mxbai-embed-large",
            input=embedding_string,
        ).embeddings[0]

    def to_chroma_item(
        self,
        *,
        id_mode: str = "stable_hash",
        id_prefix: str = "code",
    ) -> Dict[str, object]:
        file_path = str(self.file)
        symbol_type = str(self.symbol_type)
        symbol_name = str(self.symbol_name)
        start_line = int(self.start_line)
        end_line = int(self.end_line)
        language = str(self.language)
        code_description = str(self.code_description)
        keywords = str(", ".join(self.keywords))

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
            "metadata": chroma_meta,
        }
    
    
