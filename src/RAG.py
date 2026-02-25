# This file contains import functions for the connection to the RAG Databank with valuable information of the DUUI System

import chromadb as cdb
import ollama
from utils import embed_ollama, RAG_PATH


DATABASE_RAG = "DUUI_RAG_PYTHON"
def init_run_db():
    client = cdb.PersistentClient(RAG_PATH)
    collection =client.get_or_create_collection(name="test_OLLAMA")
    collection.add()

def _get_collection():
    pass


def query_results(query_input: str, collection_name: str, n_results: int = 5, distinct_file: bool = False, ollama_embedding: bool = True):
    client = cdb.PersistentClient(RAG_PATH)
    collection = client.get_or_create_collection(name=collection_name)
    if collection.count() == 0:
        raise Exception("Collection is Empty.")
    if distinct_file:
        # use proper embedding ollama
        return get_distinct_file(query_input, collection_name, n_results)

    # Always use Ollama embeddings to match the stored embeddings (1024 dim)
    embedding_input = embed_ollama(query_input)
    return collection.query(query_embeddings=embedding_input, n_results=n_results)


def get_distinct_file(query_input: str, collection_name: str, n_results: int = 5):
    client = cdb.PersistentClient(RAG_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    embedding_input = embed_ollama(query_input)
    pri_results = collection.query(query_embeddings=embedding_input, n_results=n_results)

    metadatas = pri_results.get("metadatas", [[]])[0] or []
    if not metadatas:
        return pri_results

    seen_chunk_types = set()
    keep_indices = []
    for idx, meta in enumerate(metadatas):
        chunk_type = (meta or {}).get("chunk_type", "Other")
        if chunk_type in seen_chunk_types:
            continue
        seen_chunk_types.add(chunk_type)
        keep_indices.append(idx)

    filtered = {}
    for key, value in pri_results.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            filtered[key] = [[value[0][i] for i in keep_indices]]
        else:
            filtered[key] = value

    return filtered


def get_few_shot_examples(
    query_input: str,
    collection_name: str,
    chunk_type: str = "lua",
    n_examples: int = 2,
    diversity: bool = True
) -> list[dict]:
    """
    Retrieves complete code examples from the RAG database for few-shot learning.

    Args:
        query_input: The user's query to find relevant examples
        collection_name: Name of the ChromaDB collection
        chunk_type: Chunk type filter (default: "lua" for Lua scripts)
        n_examples: Number of examples to retrieve (default: 2)
        diversity: If True, ensures examples are from different files (default: True)

    Returns:
        List of dicts with 'code' and 'metadata' keys
    """
    client = cdb.PersistentClient(RAG_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    if collection.count() == 0:
        return []

    # Filter by chunk_type only
    where_filter = {"chunk_type": chunk_type}

    # Get more results than needed for diversity filtering
    search_n = n_examples * 5 if diversity else n_examples

    # Always use Ollama embeddings to match the stored embeddings (1024 dim)
    embedding_input = embed_ollama(query_input)
    results = collection.query(
        query_embeddings=embedding_input,
        n_results=search_n,
        where=where_filter
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []

    if not documents:
        return []

    examples = []
    seen_files = set()

    for doc, meta in zip(documents, metadatas):
        if len(examples) >= n_examples:
            break

        file_path = (meta or {}).get("file", "")

        # If diversity is enabled, skip if we already have an example from this file
        if diversity and file_path in seen_files:
            continue

        examples.append({
            "code": doc,
            "metadata": meta or {}
        })

        if diversity:
            seen_files.add(file_path)

    return examples


if __name__ == "__main__":
    print(get_distinct_file("how do i implement a typesystem", "all_data_v1"))
