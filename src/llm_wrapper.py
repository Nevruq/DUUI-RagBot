from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb as cbd
from pydantic import BaseModel
import utils
from RAG import query_results

MODEL_NAME_2 = "gpt-5-nano-2025-08-07"  

class LLMWrapper():
    def __init__(self, model: str = None):
        self.model = MODEL_NAME_2
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_disabled = os.getenv("LLM_DISABLE", "").lower() in {"1", "true", "yes"}

    def add_model(self, model:str):
        self.model = model
    
    def llm_python_code_assistant(self, input_user: str)-> str:
        """
        This function call instucts the Model in a certain way to assist with coding Question for DUUI and in particular python.
        """
        prompt_code_assistant = utils.load_prompt_template("src/prompts/gen_python_code.txt")

        # format query response
        # Todo eventuell schlauer in der query_reponse funktion zu formatieren
        query_response = query_results(input_user)["documents"]  
        print(query_response)    

        concat_prompt = prompt_code_assistant.replace("{{user_input}}", input_user).replace("{{rag_context}}", *query_response)

        return self.client.responses.parse(
            model=self.model,
            instructions="You are a DUUI coding assistant",
            input=concat_prompt
        ).output_text
    

    def llm_code_description(self, code: str)-> str:
        """
        Generates the Output for the code descirption in the proper Json format
        """
        class metadatasRag(BaseModel):
            codeDescription: str
            keywords: list[str]

        # Load Prompt 
        prompt_code_description = utils.load_prompt_template("src/prompts/code_section_summary.txt")

        if self.llm_disabled:
            return str({"description": "N.A", "keywords": ["file:unknown", "code", "summary"]})
        print("LLM aufruf.")
        return self.client.responses.parse(
            model=self.model,
            instructions=prompt_code_description,
            input=code,
            text_format=metadatasRag
        ).output_text

