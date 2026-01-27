
if __name__ == "__main__":
    import ollama
    import os   
    from utils import embed_ollama
    from tqdm import tqdm
    import chromadb as cdb
    import traceback

    PATH_DUUI = "src/data/duui-uima"
    FILTER_FILES = {".py", ".ipynb"}
    TEST_FILE = ["src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"]

    import utils
    client = cdb.PersistentClient(utils.get_rag_path())
    collection = client.get_collection("java_v2")
    emb = embed_ollama("create a pipeline for cas objects")
    print(collection.query(query_embeddings=emb)["metadatas"])
