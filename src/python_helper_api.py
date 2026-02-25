# This File generates with a given User Input a help / fix to a DUUI-question implemented in python?
import llm_wrapper
from RAG import query_results
import utils

MODEL_NAME_2 = "gpt-5-nano-2025-08-07" 
PROMPT_TEMPLATE_PATH = "src/prompts/gen_python_code.txt"
COLLECTION_NAME = "all_data_v1"


def python_fix_code(prompt: str, stream: bool = False):
    
    llmObject = llm_wrapper.LLMWrapper()
    response = llmObject.llm_code_assistant(
        input_user=prompt,
        collection_name=COLLECTION_NAME,
        rag_context=True,
        stream=stream,
    )
    return response


if __name__ == "__main__":
    for chunk in python_fix_code(
        "Schreibe mir ein UIMA TypeSystem für mein neues Project, ich analysiere reden, es die RedeID den Text, sowie auf eine Hate annotation",
        stream=True,
    ):
        print(chunk, end="", flush=True)
