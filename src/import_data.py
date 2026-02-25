import llm_wrapper
import chunk_data.rag_chunk as rg
import os
from chunk_data.chunk_java import chunk_java_file
from chunk_data.chunk_other_files import chunk_other_file
from chunk_data.chunk_python import chunk_python_file
from chunk_data.chunk_lua import chunk_lua_file

import chromadb
import tqdm
from utils import filter_files, RAG_PATH, load_jsonl_ragChunk
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def describe_chunk(chunk):
    llm = llm_wrapper.LLMWrapper()
    data = llm.llm_code_description(chunk.text)
    return chunk, data



def chunk_file(path: str,
    include_header: bool = True,
    header_max_lines: int = 80,
    include_methods: bool = True,
    deferred_llm: bool = False,
    repo_root: str | None = None,
    repo_id: str | None = None ) -> list[rg.RAGChunk]:
    """
    This function takes any files and properly chunks it return a list of RAGChunk chunks.
    Proper formatting and chunking only available for python and java files for now.
    """
    if path.endswith(".py"):
        return chunk_python_file(path=path, deferred_llm=deferred_llm, repo_root=repo_root, repo_id=repo_id)
    if path.endswith(".java"):
        return chunk_java_file(path=path, deferred_llm=deferred_llm, repo_root=repo_root, repo_id=repo_id)
    if path.endswith(".lua"):
        return chunk_lua_file(path=path, deferred_llm=deferred_llm, repo_root=repo_root, repo_id=repo_id)
    else:
        return chunk_other_file(path=path, deferred_llm=deferred_llm, repo_root=repo_root, repo_id=repo_id)

def load_data(LIST_FILES: list[str], output_jsonl: str, repo_root: str | None = None, repo_id: str | None = None):
    """
    The function takes a list of all files that should be chunked and then processed. Chunking is done beforehand. Afterwards LLM calls to
    generate descriptions and keywords for better retrieval results are made. Finally the enriched chunks are stored in a jsonl file for later use.
    
    :param LIST_FILES: Description
    :type LIST_FILES: list[str]
    """
    import chunk_data.rag_chunk

    all_chunks = []
    for file in tqdm.tqdm(LIST_FILES):
        cur_chunks = chunk_file(file, deferred_llm=True, repo_root=repo_root, repo_id=repo_id)
        all_chunks.extend(cur_chunks)

    print("ALL CHUNKS LOADED.")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(describe_chunk, c) for c in all_chunks]
            for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
                rchunk, data = fut.result()
                rchunk.append_llm_description(data)
                f.write(json.dumps(rchunk.to_json_item()) + "\n")
                f.flush()
    
def insert_data_chroma(chunks: list[rg.RAGChunk], collection_name: str):
    items = [chunk.to_chroma_item() for chunk in chunks]
    ids = [item["id"] for item in items]
    embs = [item["embedding"] for item in items]
    docs = [item["document"] for item in items]
    metas = [item["metadata"] for item in items]
    client = chromadb.PersistentClient(RAG_PATH)
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

        


if __name__ == "__main__":
    from chunk_data.rag_chunk import jsonl_to_RagChunks
    client = chromadb.PersistentClient("chroma").get_collection("all_data_v2").count()
    print(client)
    #file_path = "src/data/DockerUnifiedUIMAInterface"
    #files = filter_files(file_path, filters={".lua", ".py", ".java", ".md", ".xml"})
    #print(f"Total files to process: {len(files)}")
    #load_data(LIST_FILES=files, output_jsonl="src/data/DUUI_v1.jsonl", repo_root=file_path)
    list_ragChunks = jsonl_to_RagChunks("src/data/DUUI_v1.jsonl")
    #insert_data_chroma(list_ragChunks, "all_data_v2")

    


    
        
    
