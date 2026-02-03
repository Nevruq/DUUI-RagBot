import ollama
import os
import chunk_data.rag_chunk as rc
import json
from dotenv import load_dotenv


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


def get_rag_path(default: str = "chroma") -> str:
    load_dotenv()
    return os.getenv("RAG_PATH", default)


def load_jsonl_ragChunk(path: str) -> list[rc.RAGChunk]:
    with open(path) as f:
        data = [json.loads(line) for line in f]
        return rc.ragchunks_from_json_items(data)
    
def calc_token_length(context: str) -> int:
    """
    A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).
    """
    #TODO wenn erforderlich implementiere mit richtigem Tokenizer
    return len(context) / 4


from lxml import etree

ALLOWED_RANGES = {
    "uima.cas.String", "uima.cas.Integer", "uima.cas.Float", "uima.cas.Boolean",
    "uima.cas.Double", "uima.cas.Long", "uima.cas.Short", "uima.cas.Byte",
    "uima.cas.FSArray", "uima.cas.IntegerArray", "uima.cas.FloatArray",
    "uima.tcas.Annotation", "uima.cas.TOP"
}

from cassis import load_typesystem
from lxml import etree

def validate_typesystem(xml_text: str) -> list[str]:
    issues = []
    # 2) UIMA/Cassis load

    try:
        load_typesystem(xml_text.encode("utf-8"))
    except Exception as e:
        issues.append(f"Cassis load error: {e}")

    # 3) Simple duplicate checks (optional, fast)
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
        types = root.findall(".//typeDescription")
        type_names = [t.findtext("name") for t in types]
        dup_types = {t for t in type_names if type_names.count(t) > 1}
        if dup_types:
            issues.append(f"Duplicate types: {sorted(dup_types)}")
    except Exception:
        pass

    return issues

def validate_labels(labels: list[str]) -> list[str]:
    """
    Checks if all Labels in a List are valid and exisist in the DUUIRAG dictonary.
    """
    valid_labels = load_prompt_template("src/DUUIDictonary.txt")
    filtered_labels = []
    for label in labels:
        if label in valid_labels:
            filtered_labels.append(label)
    return filtered_labels