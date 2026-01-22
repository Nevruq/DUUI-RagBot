# temp file für das einfügen der Datein in Db


if __name__ == "__main__":
    import os
    from chunk_data.chunk_java import chunk_java_file
    import chromadb
    import tqdm

    PATH_DUUI = "src/data/duui-uima"
    FILTER_FILES = {".java"}
    count = 0
    LIST_XML_FILES = []
    for root, subdirs, files in os.walk(PATH_DUUI):
        for file in files:
            current_file = os.path.join(root, file)
            # filter files and ignore pom.xml
            if (file.endswith(tuple(FILTER_FILES)) and (file != "pom.xml")):
                LIST_XML_FILES.append(current_file)
    shorten_list = LIST_XML_FILES[:10]
    for file in tqdm.tqdm(shorten_list):
        chunks = chunk_java_file(file)

        # add collection
        ids, embs, docs, metas = [], [], [], []
        for chunk in chunks:
            chunk_data = chunk.to_chroma_item()
            ids.append(chunk_data["id"])
            embs.append(chunk_data["embedding"])
            docs.append(chunk_data["document"])
            metas.append(chunk_data["metadata"])

        client = chromadb.PersistentClient("src/chroma")
        collection = client.get_or_create_collection("JAVA_Test")
        collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)



