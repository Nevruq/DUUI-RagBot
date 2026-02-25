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
    Filtere einen Pfad rekursiv und gib alle Dateien zurück, die den Filter erfüllen.
    - `path` kann eine Datei oder ein Verzeichnis sein (rekursiv durchsucht).
    - `filters` kann `None` (alle Dateien), eine Menge/Iterable von Erweiterungen
      wie `{'.py', '.java'}` oder auch eine fehlerhafte Eingabe wie `set('.py')`
      enthalten. In letzteren Fällen wird versucht, die korrekte Erweiterung zu
      rekonstruieren.
    """
    matched = []

    # Normalize filters: accept None or any iterable of strings
    proc_filters = None
    if filters:
        # If user passed a single string like '.py', allow that too
        if isinstance(filters, str):
            proc_filters = {filters}
        else:
            # coerce to set of strings
            proc_filters = {str(f) for f in filters}

        # Heuristic: if filters are single characters (e.g. set('.py'))
        # reconstruct probable extension
        if all(len(f) == 1 for f in proc_filters):
            joined = "".join(sorted(proc_filters))
            if not joined.startswith('.'):
                joined = '.' + joined
            proc_filters = {joined}

        # ensure extensions start with a dot (treat values as suffixes)
        proc_filters = {f if f.startswith('.') else '.' + f for f in proc_filters}

    # If path is a file, check it directly
    if os.path.isfile(path):
        if not proc_filters:
            return [os.path.abspath(path)]
        else:
            fname = os.path.basename(path).lower()
            if fname.endswith(tuple(f.lower() for f in proc_filters)):
                return [os.path.abspath(path)]
            return []

    # If path does not exist or is not a directory, return empty list
    if not os.path.exists(path):
        return []

    for root, _subdirs, files in os.walk(path):
        for file in files:
            current_file = os.path.join(root, file)
            if not proc_filters:
                matched.append(current_file)
            else:
                if file.lower().endswith(tuple(f.lower() for f in proc_filters)):
                    matched.append(current_file)

    return matched


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


# Global RAG_PATH constant - loaded once at module import
load_dotenv()
RAG_PATH = os.getenv("RAG_PATH", "chroma")


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



from cassis import load_typesystem
from lxml import etree

def validate_typesystem(xml_text: str) -> list[str]:
    issues = []
    # 2) UIMA/Cassis load

    # Accept either raw XML text or a path to an XML file
    text = xml_text
    if os.path.exists(xml_text) and os.path.isfile(xml_text):
        try:
            with open(xml_text, 'r', encoding='utf-8') as fh:
                text = fh.read()
        except Exception as e:
            issues.append(f"Failed to read typesystem file: {e}")
            return issues

    try:
        load_typesystem(text.encode("utf-8"))
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
    valid_labels = load_prompt_template("src/data/DUUIDictonary.txt")
    filtered_labels = []
    for label in labels:
        if label in valid_labels:
            filtered_labels.append(label)
    return filtered_labels

