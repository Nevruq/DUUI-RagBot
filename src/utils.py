import ollama

def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    

def short(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + "\n... [truncated] ...\n" + s[-300:]

def embed_ollama(input: str):
    return ollama.embed(
                    model='mxbai-embed-large',
                    input=input
                    ).embeddings[0]