# This File generates with a given User Input a help / fix to a DUUI-question implemented in python?
import llm_wrapper
from RAG import query_results
import utils

MODEL_NAME_2 = "gpt-5-nano-2025-08-07" 
PROMPT_TEMPLATE_PATH = "src/prompts/gen_python_code.txt"
COLLECTION_NAME = "all_data_v1"


def python_fix_code(prompt:str):
    
    llmObject = llm_wrapper.LLMWrapper()
    response = llmObject.llm_code_assistant(input_user=prompt, collection_name=COLLECTION_NAME, coding_lg="java", rag_context=True)
    return response

print(python_fix_code("Schreibe mir den code in java wie ich ein Hate Model mit DUUI implementieren kann. Ich brauche den vollen code zum ausfüh"))