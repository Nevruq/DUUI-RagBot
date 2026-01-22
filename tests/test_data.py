import chromadb
import ollama


client = chromadb.PersistentClient("src/chroma")
collection = client.get_collection("JAVA_Test")

test_query = "how can I create a CAS Object from a List of Strings"

response = ollama.embed(
        model='mxbai-embed-large',
        input=test_query,
    ).embeddings
print(collection.query(query_embeddings=response, n_results=30)["metadatas"])
print(collection.count())


