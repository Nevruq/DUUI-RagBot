# This File generates with a given User Input a help / fix to a DUUI-question implemented in python?
import llm_wrapper
from RAG import query_results
import utils

MODEL_NAME_2 = "gpt-5-nano-2025-08-07" 
PROMPT_TEMPLATE_PATH = "src/prompts/gen_python_code.txt"


def python_fix_code(prompt:str):
    
    llmWrapper = llm_wrapper.LLMWrapper(model=MODEL_NAME_2)

    # Get Query results
    query_results_output = query_results(query_input=prompt)[0]
    print(query_results_output)

    # load prompt
    question_prompt = utils.load_prompt_template("src/prompts/context_questions.txt")
    # make llm call
    response = llmWrapper.gen_reponse(input_user=prompt + str(query_results_output), instructions_user=question_prompt)
    
    return response


print(python_fix_code("Welches andere Modelle bzw. sourcemodelle wurde benutzt für das hate DUUI modell neben tomh/toxigen_hatebert?"))