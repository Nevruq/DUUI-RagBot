from __future__ import annotations
# Eigentlich irrelvant für Python 3.13

from dataclasses import dataclass
from typing import Dict, List
import hashlib
import json

import ollama


def make_repo_id(repo_root: str) -> str:
    normalized = repo_root.replace("\\", "/").rstrip("/")
    h = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"repo::{h}"


@dataclass
class RAGChunk:
    text: str
    file: str
    language: str
    symbol_type: str
    symbol_name: str
    start_line: int
    end_line: int
    description: str
    keywords: str
    chunk_type: str
    repo_id: str

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
            "description": self.description,
            "keywords": self.keywords,
            "chunk_type": self.chunk_type,
            "repo_id": self.repo_id,
        }

    def gen_embedding_meta(self):
        embedding_string = f"""
            name: {self.file}
            summary: {self.description}
            keywords: {self.keywords}
        """
        return ollama.embed(
            model="mxbai-embed-large",
            input=embedding_string,
        ).embeddings[0]
    
    def append_llm_data(self, llm_data: str) -> None:
        """
        Appends LLM data afterwards for better threading
        """
        if isinstance(llm_data, str):
            try:
                llm_data = json.loads(llm_data)
            except json.JSONDecodeError:
                llm_data = {}
        # gen_code_descritpion exp. {description: "..", keywords: ["a","b"]}
        desc = llm_data.get("description", "N.A") if isinstance(llm_data, dict) else "N.A"
        kw = llm_data.get("keywords", ["N.A"]) if isinstance(llm_data, dict) else ["N.A"]
        if isinstance(kw, str):
            kw = [k.strip() for k in kw.split(",") if k.strip()]
        elif not isinstance(kw, list):
            kw = [str(kw)]
        self.description = desc
        self.keywords = kw

    def to_chroma_item(
        self,
        *,
        id_mode: str = "stable_hash",
        id_prefix: str = "id",
    ) -> Dict[str, object]:
        file_path = str(self.file)
        symbol_type = str(self.symbol_type)
        symbol_name = str(self.symbol_name)
        start_line = int(self.start_line)
        end_line = int(self.end_line)
        language = str(self.language)
        description = str(self.description)
        keywords = str(", ".join(self.keywords))

        #TODO überlege eine bessere hashmethode für die IDS, file_path komisch
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
            "description": description,
            "keywords": keywords,
            "chunk_type": self.chunk_type,
            "repo_id": self.repo_id,
        }

        return {
            "id": chunk_id,
            "embedding": self.gen_embedding_meta(),
            "document": self.text,
            "metadata": chroma_meta,
        }
    
    def to_json_item(self,
        *,
        id_mode: str = "stable_hash",
        id_prefix: str = "id",
    ) -> Dict[str, object]:
        """
        Converts the RAGChunk object to a JSON
        """
        return self.to_chroma_item(id_mode=id_mode, id_prefix=id_prefix)


def ragchunks_from_json_items(items: List[Dict[str, object]]) -> List[RAGChunk]:
    """
    Convert a list of JSONL-loaded dicts into RAGChunk objects.
    """
    chunks: List[RAGChunk] = []
    for item in items:
        document = str(item.get("document", ""))
        metadata = item.get("metadata", {}) or {}
        keywords = metadata.get("keywords", "")
        if isinstance(keywords, str):
            keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        elif isinstance(keywords, list):
            keywords_list = [str(k).strip() for k in keywords if str(k).strip()]
        else:
            keywords_list = []

        chunks.append(
            RAGChunk(
                text=document,
                file=str(metadata.get("file", "")),
                language=str(metadata.get("language", "")),
                symbol_type=str(metadata.get("symbol_type", "")),
                symbol_name=str(metadata.get("symbol_name", "")),
                start_line=int(metadata.get("start_line", 0) or 0),
                end_line=int(metadata.get("end_line", 0) or 0),
                description=str(metadata.get("description", "N.A")),
                keywords=keywords_list,
                chunk_type=str(metadata.get("chunk_type", "code")),
                repo_id=str(metadata.get("repo_id", "repo::unknown")),
            )
        )
    return chunks
