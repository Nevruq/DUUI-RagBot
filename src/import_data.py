import llm_wrapper
import chunk_data.rag_chunk as rg
import os
from chunk_data.chunk_java import chunk_java_file
import chromadb
import tqdm
from utils import filter_files
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import llm_wrapper


def describe_chunk(chunk):
    llm = llm_wrapper.LLMWrapper()
    data = llm.llm_code_description(chunk.text)
    return chunk, data


if __name__ == "__main__":

    PATH_DUUI = "src/data/duui-uima"
    LIST_FILES = filter_files(PATH_DUUI, {".java"})

    client = chromadb.PersistentClient("src/chroma")
    client.delete_collection("Java_all_v1")
    collection = client.get_or_create_collection("Java_all_v1")

    all_chunks = []
    for file in tqdm.tqdm(LIST_FILES):
        cur_chunks = chunk_java_file(file, deferred_llm=True)
        all_chunks.extend(cur_chunks)

    with open("src/data/chunks_java.jsonl", "w", encoding="utf-8") as f:
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


