import chromadb
import ollama


client = chromadb.PersistentClient("src/chroma")
collection = client.get_collection("test_OLLAMA")

test_query = "how can I add annotaitons to the hate modell"

response = ollama.embed(
        model='mxbai-embed-large',
        input=test_query,
    ).embeddings
print(collection.query(query_embeddings=response, n_results=30)["ids"])


