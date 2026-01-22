# This File generates with a given User Input a help / fix to a DUUI-question implemented in python?
import llm_wrapper
from RAG import query_results
import utils

MODEL_NAME_2 = "gpt-5-nano-2025-08-07" 
PROMPT_TEMPLATE_PATH = "src/prompts/gen_python_code.txt"
COLLECTION_NAME = "JAVA_Test"


def python_fix_code(prompt:str):
    
    llmObject = llm_wrapper.LLMWrapper()
    response = llmObject.llm_code_assistant(input_user=prompt, collection_name=COLLECTION_NAME, coding_lg="java", rag_context=True)
    return response

print(python_fix_code("Schreibe mir eine Beispiel code in java wie zb das Sarkasmusmodel in DUUI benutzte und wie kann ich damit Sachen analysizen?"))