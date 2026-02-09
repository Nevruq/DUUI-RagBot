
if __name__ == "__main__":
    import ollama
    import os   
    from utils import embed_ollama
    from tqdm import tqdm
    import chromadb as cdb
    import traceback
    from utils import filter_files
    from import_data import chunk_file
    from chunk_data.rag_chunk import RAGChunk


    files = filter_files("src/data/",  filters={".lua"})[:10]
    all_chunks = []
    for file in tqdm(files):
        cur_chunks = chunk_file(file, deferred_llm=True)
        all_chunks.extend(cur_chunks)
    

    import utils

    client = cdb.PersistentClient(utils.get_rag_path())
    collection = client.get_or_create_collection(name="lua_test")
    formatted_chunks = []
    for chunk in all_chunks:
        collection.add(
            ids=[chunk.to_chroma_item()["id"]],
            documents=[chunk.to_chroma_item()["document"]],
            metadatas=[chunk.to_chroma_item()["metadata"]],
        )
