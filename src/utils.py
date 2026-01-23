import ollama
import os

def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    

def short(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + "\n... [truncated] ...\n" + s[-300:]


def write_ragchunks_jsonl(chunks, path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            item = chunk.to_json_item()
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

def embed_ollama(input: str):
    return ollama.embed(
                    model='mxbai-embed-large',
                    input=input
                    ).embeddings[0]

def _safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_lines(text: str) -> list[str]:
    # keep line endings so slicing preserves formatting
    return text.splitlines(keepends=True)

def filter_files(path: str, filters: set = {".python"}):
    LIST_XML_FILES = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            current_file = os.path.join(root, file)
            # filter files and ignore pom.xml
            if (file.endswith(tuple(filters)) and (file != "pom.xml")):
                LIST_XML_FILES.append(current_file)
    return LIST_XML_FILES

import chunk_data.rag_chunk as rc
import json


def load_jsonl_ragChunk(path: str) -> list[rc.RAGChunk]:
    """
    Function loads JsonL and converts it to RAGChunk objects.
    """
    with open(path) as f:
        data = [json.loads(line) for line in f]
        return rc.ragchunks_from_json_items(data)


