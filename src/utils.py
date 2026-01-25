import ollama
import os
import chunk_data.rag_chunk as rc
import json


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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


def filter_files(path: str, filters: set = None):
    """
    Filterse a path and returns all file with that set filter. If no filter is given all files are returned.
    """
    LIST_XML_FILES = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            current_file = os.path.join(root, file)
            # filter files and ignore pom.xml
            if not filters:
                LIST_XML_FILES.append(current_file)
            else:
                if file.endswith(tuple(filters)):
                    LIST_XML_FILES.append(current_file)
    return LIST_XML_FILES


def infer_file_type(path: str) -> str:
    path_lower = path.lower()
    if "/test/" in path_lower or path_lower.endswith("_test.py") or path_lower.endswith("test.py"):
        return "tests"
    if "readme" in path_lower or path_lower.endswith(".md"):
        return "docs"
    if path_lower.endswith((".yml", ".yaml", ".json", ".toml", ".ini", ".cfg")):
        return "config"
    if path_lower.endswith(".xml"):
        if "typesystem" in path_lower:
            return "typesystem"
        return "schema"
    if path_lower.endswith((".py", ".java", ".js", ".ts", ".rb", ".go", ".rs", ".cpp", ".c", ".h", ".hpp")):
        return "code"
    if path_lower.endswith((".csv", ".tsv", ".parquet", ".txt")):
        return "data"
    return "other"


def find_repo_root(file_path: str, markers: tuple[str, ...] = (".git", "pyproject.toml", "pom.xml", "package.json")) -> str | None:
    path = os.path.abspath(file_path)
    cur = os.path.dirname(path)
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent




"""
def load_jsonl_ragChunk(path: str) -> list[rc.RAGChunk]:
    
    Function loads JsonL and converts it to RAGChunk objects.
    
    with open(path) as f:
        data = [json.loads(line) for line in f]
        return rc.ragchunks_from_json_items(data)
"""
