# This file contains import functions for the connection to the RAG Databank with valuable information of the DUUI System

import chromadb as cdb
import ollama
from utils import embed_ollama, get_rag_path


DATABASE_RAG = "DUUI_RAG_PYTHON"
RAG_PATH = get_rag_path()
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
    if ollama_embedding:
        embedding_input = embed_ollama(query_input)
        return collection.query(query_embeddings=embedding_input, n_results=n_results)
    else:
        return collection.query(query_texts=query_input, n_results=n_results)


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


if __name__ == "__main__":
    print(get_distinct_file("how do i implement a typesystem", "all_data_v1"))
