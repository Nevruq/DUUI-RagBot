# This file contains import functions for the connection to the RAG Databank with valuable information of the DUUI System

import chromadb as cdb

DATABASE_RAG = "DUUI_RAG_PYTHON"
RAG_PATH = "src/chroma"
def init_run_db():
    client = cdb.PersistentClient("chroma")
    client.get_or_create_collection(name="DUUI_RAG_PYTHON")

def _get_collection():
    pass

def query_results(query_input: str, n_results: int = 5):
    client = cdb.PersistentClient(RAG_PATH)
    collection = client.get_or_create_collection(name=DATABASE_RAG)
    result = collection.query(query_texts="test", n_results=n_results)
    documents = result.get("documents", [[]])[0] or []
    metadatas = result.get("metadatas", [[]])[0] or []
    return documents, metadatas




if __name__ == "__main__":
    print(query_results("test"))