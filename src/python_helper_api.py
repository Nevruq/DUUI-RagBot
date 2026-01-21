# This File generates with a given User Input a help / fix to a DUUI-question implemented in python?
import llm_wrapper
from RAG import query_results
import utils

MODEL_NAME_2 = "gpt-5-nano-2025-08-07" 
PROMPT_TEMPLATE_PATH = "src/prompts/gen_python_code.txt"


def python_fix_code(prompt:str):
    
    llmObject = llm_wrapper.LLMWrapper()
    response = llmObject.llm_python_code_assistant(input_user=prompt)

    return response


print(python_fix_code("Welches andere Modelle bzw. sourcemodelle wurde benutzt für das hate DUUI modell neben tomh/toxigen_hatebert?"))