from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb as cbd

MODEL_NAME_2 = "gpt-5-nano-2025-08-07"  

class LLMWrapper():
    def __init__(self, model: str = None):
        self.model = MODEL_NAME_2
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def add_model(self, model:str):
        self.model = model
    
    def gen_response(self, input_user: str, instructions_user: str = None)-> str:
        return self.client.responses.parse(
            model=self.model,
            instructions=instructions_user,
            input=input_user
        ).output_text
    

    def gen_response_formatted(self, input_user: str, text_format, instructions_user: str = None)-> str:
        """
        Allows for formatted Outputs
        """
        return self.client.responses.parse(
            model=self.model,
            instructions=instructions_user,
            input=input_user,
            text_format=text_format
        ).output_text

