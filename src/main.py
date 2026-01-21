
if __name__ == "__main__":
    import ollama
    import chunker
    import os   
    from tqdm import tqdm
    import chunker
    import chromadb as cdb
    import traceback

    PATH_DUUI = "src/data/duui-uima"
    FILTER_FILES = {".py", ".ipynb"}
    TEST_FILE = ["src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"]

    count = 0
    LIST_XML_FILES = []
    for root, subdirs, files in os.walk(PATH_DUUI):
        for file in files:
            current_file = os.path.join(root, file)
            # filter files and ignore pom.xml
            if (file.endswith(tuple(FILTER_FILES)) and (file != "pom.xml")):
                LIST_XML_FILES.append(current_file)

    # chunk the files with chunker
    print(len(LIST_XML_FILES))


    client = cdb.PersistentClient("src/chroma")
    client.delete_collection(name="test_OLLAMA")
    collection = client.get_or_create_collection(name="test_OLLAMA")
    print("Collection created!")

    # cast files to Chroma format

    count = 0
    for file in tqdm(TEST_FILE):
        chunks = chunker.chunk_python_file(file, include_header=True, include_methods=True)
        ids, emb, metas, docs = [], [], [], []

        for i, chunk in enumerate(chunks):
            count += 1
            chroma_format = chunker.chunk_to_chroma_item(chunk)
            # some empty documents exist
            document = chroma_format["document"]
            if document:
                # generate manual Ollama embedding: mxbai-embed-large
          
                emb.append(chroma_format["embedding"])
                ids.append(chroma_format["id"])
                metas.append(chroma_format["metadata"])
                docs.append(chroma_format["document"])
              
            else:
                print("leerer eintrag")
        if ids:
            collection.add(ids=ids, embeddings=emb, metadatas=metas, documents=docs)
            print("chunks added")
            
    print(count)
