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


def query_results(query_input: str, collection_name: str, n_results: int = 5):
    client = cdb.PersistentClient(RAG_PATH)
    collection = client.get_or_create_collection(name=collection_name)
    if collection.count() == 0:
        raise Exception("Collection is Empty.")
    # use proper embedding ollama
    embedding_input = embed_ollama(query_input)
    return collection.query(query_embeddings=embedding_input, n_results=n_results)




if __name__ == "__main__":
    print(query_results("test"))
