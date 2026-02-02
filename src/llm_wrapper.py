from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb as cbd
from pydantic import BaseModel
import utils
import json
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

    import json
    def llm_rewrite_query(self, input_str) -> dict:
        """
        In order for better query_results the prompt from the user needs to be rewritten by
        Returns dict with the str:description and List[str]: keywords
        """
        class queryDescription(BaseModel):
            description: str
            keywords: list[str]

        template = utils.load_prompt_template("src/prompts/rewrite_query.txt")
        response = self.client.responses.parse(
            model=self.model,
            instructions="Your Task is to summarize. Given a Text ",
            text_format=queryDescription,
            input=input_str
        )
        formatted_dict = {"description": response.output_parsed.description, "keywords": response.output_parsed.keywords}
        return formatted_dict

        # calc prompt length in tokens


        return response.output_text

    
    def llm_code_assistant(self, input_user: str, collection_name: str, coding_lg: str = "python", rag_context: bool = True)-> str:
        """
        This function call instucts the Model in a certain way to assist with coding Question for DUUI and in particular python.
        """
        prompt_code_assistant = ""
        match coding_lg.lower():
            case "python":
                prompt_code_assistant = utils.load_prompt_template("src/prompts/gen_python_code.txt")
            case "java":
                prompt_code_assistant = utils.load_prompt_template("src/prompts/gen_java_code.txt")

        # format query response
        query_response = {}
        if rag_context:
            # TODO eventuell schlauer in der query_reponse funktion zu formatieren
            # Anpassen,dass die collection ausgewählt wreden kann
            query_response = query_results(input_user, collection_name=collection_name)

        documents = query_response.get("documents", [[]])[0] if query_response else []
        metadatas = query_response.get("metadatas", [[]])[0] if query_response else []
        context_parts = []
        for i, doc in enumerate(documents or []):
            meta = metadatas[i] if i < len(metadatas) else {}
            context_parts.append(f"[{i + 1}] document:\n{doc}\nmetadata:\n{meta}")
        rag_context_text = "\n\n".join(context_parts) if context_parts else "No RAG context."
        print(rag_context_text)

        concat_prompt = (
            prompt_code_assistant
            .replace("{{user_input}}", input_user)
            .replace("{{rag_context}}", rag_context_text)
        )

        print(utils.calc_token_length(concat_prompt))
        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI assitant and answer question about.",
            input=concat_prompt
        )

        # calc prompt length in tokens


        return response.output_text


    

    def llm_code_description(self, code: str)-> dict:
        """
        Generates the Output for the code descirption in the proper Json format
        """
        class metadatasRag(BaseModel):
            description: str
            keywords: list[str]
            

        # Load Prompt 
        prompt_code_description = utils.load_prompt_template("src/prompts/code_section_summary.txt")

        if self.llm_disabled:
            return {"description": "N.A", "keywords": ["file:unknown", "code", "summary"]}
        print("LLM aufruf.")
        response = self.client.responses.parse(
            model=self.model,
            instructions=prompt_code_description,
            input=code,
            text_format=metadatasRag
        ).output_parsed
        return {"description": response.description, "keywords": response.keywords}
    

    def llm_other_file_description(self, text: str) -> str:
        class otherFileMetadata(BaseModel):
            description: str
            keywords: list[str]

        if self.llm_disabled:
            return {"description": "N.A", "keywords": ["file:unknown", "noncode", "summary"]}

        prompt_other_file = utils.load_prompt_template("src/prompts/other_file_summary.txt")

        response = self.client.responses.parse(
            model=self.model,
            instructions=prompt_other_file,
            input=text,
            text_format=otherFileMetadata
        ).output_parsed
        return {"description": response.description, "keywords": response.keywords}
    
    


    
