import llm_wrapper
import chunk_data.rag_chunk as rg
import os
from chunk_data.chunk_java import chunk_java_file
from chunk_data.chunk_other_files import chunk_other_file
from chunk_data.chunk_python import chunk_python_file

import chromadb
import tqdm
from utils import filter_files
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
        return chunk_python_file(path=path, deferred_llm=True)
    if path.endswith(".java"):
        return chunk_java_file(path=path, deferred_llm=True)
    else:
        return chunk_other_file(path=path, deferred_llm=True)


if __name__ == "__main__":

    PATH_DUUI = "src/data/duui-uima/duui-Hate"
    PATH_DUUI_2 = "src/data/duui-uima/duui-entailment"

    LIST_FILES_1 = filter_files(PATH_DUUI)
    LIST_FILES_1.extend(filter_files(PATH_DUUI_2))
    
    client = chromadb.PersistentClient("src/chroma")

    collection = client.get_or_create_collection("all_data_v1")

    all_chunks = []
    for file in tqdm.tqdm(LIST_FILES_1):
        cur_chunks = chunk_file(file, deferred_llm=True)
        all_chunks.extend(cur_chunks)

    print("ALL CHUNKS LOADED.")

    with open("src/data/chunks_all_v1.jsonl", "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(describe_chunk, c) for c in all_chunks]
            for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
                rchunk, data = fut.result()
                rchunk.append_llm_data(data)
                f.write(json.dumps(rchunk.to_json_item()) + "\n")
                f.flush()
        
    
    """
        # add collection
        ids, embs, docs, metas = [], [], [], []
        for chunk in chunks:
            chunk_data = chunk.to_chroma_item()
            ids.append(chunk_data["id"])
            embs.append(chunk_data["embedding"])
            docs.append(chunk_data["document"])
            metas.append(chunk_data["metadata"])

            # dump json formatted file in jsonl
            chunk.to_json_item()
        collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    """


