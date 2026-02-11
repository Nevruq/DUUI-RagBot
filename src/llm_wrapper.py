from openai import OpenAI
import os
import ast
from dotenv import load_dotenv
import chromadb as cbd
from pydantic import BaseModel
import utils
import json
from RAG import query_results
from typing import Optional, Dict, List, Literal

MODEL_NAME_2 = "gpt-5-nano-2025-08-07"  

class LLMWrapper():
    def __init__(self, model: str = None):
        self.model = MODEL_NAME_2
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_disabled = os.getenv("LLM_DISABLE", "").lower() in {"1", "true", "yes"}

    def add_model(self, model:str):
        self.model = model


    def llm_rewrite_query(self, input_str) -> dict:
        """
        In order for better query_results the prompt from the user needs to be rewritten by
        Returns dict with the str:description and List[str]: keywords
        """
        class queryDescription(BaseModel):
            description: str
            keywords: list[str]
            subqueries: list[str]

        template = utils.load_prompt_template("src/prompts/rewrite_query.txt")
        response = self.client.responses.parse(
            model=self.model,
            instructions="Your Task is to summarize. Given a Text ",
            text_format=queryDescription,
            input=input_str
        )
        formatted_dict = {"description": response.output_parsed.description, 
                          "keywords": response.output_parsed.keywords, 
                          "subqueries": response.output_parsed.subqueries}
        return formatted_dict


    
    def llm_code_assistant(self, input_user: str, collection_name: str, coding_lg: str = "", rewrite_query: bool = True, rag_context: bool = True)-> str:
        """
        This function call instucts the Model in a certain way to assist with coding Question for DUUI and in particular python.
        """
        prompt_code_assistant = ""
        match coding_lg.lower():
            case "python":
                prompt_code_assistant = utils.load_prompt_template("src/prompts/gen_python_code.txt")
            case "java":
                prompt_code_assistant = utils.load_prompt_template("src/prompts/gen_java_code.txt")
            case "typesytem":
                prompt_code_assistant = utils.load_prompt_template("src/prompts/typesystem_builder.txt")
            case _:
                prompt_code_assistant = utils.load_prompt_template("src/prompts/answer_from_context.txt")

        # format query response
        query_response = {}

        # When rewrite_query is True the Prompt of the user is rewritten for better RAG
        if rewrite_query:
            input_user = self.llm_rewrite_query(input_str=input_user)["description"]

        if rag_context:
            # TODO eventuell schlauer in der query_reponse funktion zu formatieren
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

        return response.output_text



    def llm_lua_code_builder(self, input_user: str, collection_name: str, rewrite_query: bool = True, rag_context: bool = True, ollama_embedding: bool = True) -> str:
        """
        This function call instucts the Model in a certain way to assist with coding Question for DUUI and in particular python.
        """
        prompt_lua_code_builder = utils.load_prompt_template("src/prompts/lua_code_builder.txt")

        blue_print = 

        # format query response
        query_response = {}

        # When rewrite_query is True the Prompt of the user is rewritten for better RAG
        if rewrite_query:
            input_user = self.llm_rewrite_query(input_str=input_user)["description"]

        if rag_context:
            # TODO eventuell schlauer in der query_reponse funktion zu formatieren
            query_response = query_results(input_user, collection_name=collection_name, ollama_embedding=ollama_embedding)

        documents = query_response.get("documents", [[]])[0] if query_response else []
        metadatas = query_response.get("metadatas", [[]])[0] if query_response else []
        context_parts = []
        for i, doc in enumerate(documents or []):
            meta = metadatas[i] if i < len(metadatas) else {}
            context_parts.append(f"[{i + 1}] document:\n{doc}\nmetadata:\n{meta}")
        rag_context_text = "\n\n".join(context_parts) if context_parts else "No RAG context."

        concat_prompt = (
            prompt_lua_code_builder
            .replace("{{blue_print}}", blue_print)
            .replace("{{user_input}}", input_user)
            .replace("{{rag_context}}", rag_context_text)
        )

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI Lua code builder.",
            input=concat_prompt
        )

        return response.output_text


    def llm_typesystem_builder(
        self,
        input_user: str,
        collection_name: str,
        rewrite_query: bool = True,
        rag_context: bool = True
    ) -> str:
        """
        Builds a UIMA TypeSystem based on user requirements and RAG context.
        """
        if self.llm_disabled:
            return "typesystem_xml:\n```xml\n<!-- LLM disabled -->\n```\nassumptions: none\nvalidation:\n- LLM disabled\nsuggestions:\n- Enable LLM to generate a TypeSystem."

        prompt_typesystem = utils.load_prompt_template("src/prompts/typesystem_builder.txt")

        if rewrite_query:
            input_user = self.llm_rewrite_query(input_str=input_user)["description"]

        query_response = {}
        if rag_context:
            query_response = query_results(input_user, collection_name=collection_name)

        documents = query_response.get("documents", [[]])[0] if query_response else []
        metadatas = query_response.get("metadatas", [[]])[0] if query_response else []
        context_parts = []
        for i, doc in enumerate(documents or []):
            meta = metadatas[i] if i < len(metadatas) else {}
            context_parts.append(f"[{i + 1}] document:\n{doc}\nmetadata:\n{meta}")
        rag_context_text = "\n\n".join(context_parts) if context_parts else "No RAG context."

        concat_prompt = (
            prompt_typesystem
            .replace("{{user_input}}", input_user)
            .replace("{{rag_context}}", rag_context_text)
        )

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI TypeSystem builder. Only return the Typesytem.",
            input=concat_prompt
        ).output_text

        try:
            utils.validate_typesystem(response)
        except:
            raise "Reponse has invalid Formatting."
        return response
    
    def llm_generate_hf_context(self, input_user: str) -> dict:
        """
        Generates a context dictionary for HuggingFace model information.
        TODO maybe add more infomration from the model
        """
        prompt_hf_context = utils.load_prompt_template("src/prompts/hf_context_generator.txt")

        class HFModelInformation(BaseModel):
            # --- Identity ---
            model_id: str                 
            revision: Optional[str] = None    
            # --- Task & Architecture ---
            task: Literal[
                "text-classification",
                "token-classification",
                "question-answering",
                "text-generation",
                "text2text-generation",
                "unknown"
            ]
            architectures: List[str]           
            model_class_hint: str         

            input_type: Literal[
                "text",
                "text_pair",
                "tokens",
                "text_with_offsets"
            ]
            tokenizer_class: str      
            model_max_length: Optional[int]
            required_model_inputs: List[str]   # e.g. ["input_ids", "attention_mask"]
            # --- Outputs (what comes out) ---
            output_keys: List[str]             # e.g. ["logits"] or ["start_logits", "end_logits"]

            num_labels: Optional[int] = None
            id2label: Optional[Dict[int, str]] = None

            output_semantics: Literal[
                "single_label_classification",
                "multi_label_classification",
                "regression",
                "span_extraction",
                "generation"
            ]
            recommended_postprocess: Literal[
                "softmax",
                "sigmoid",
                "none",
                "argmax_span",
                "generate"
            ]
            # --- Safety flags for generation / DUUI ---
            labels_are_semantic: bool           # False if LABEL_0, LABEL_1, ...
            manual_mapping_required: bool       # True → LLM must NOT guess labels

            # --- DUUI relevance ---
            produces_spans: bool                # begin/end needed?
            produces_text: bool                 # generated text?


        response = self.client.responses.parse(
            model=self.model,
            instructions="You are a DUUI HuggingFace context generator.",
            input=prompt_hf_context.replace("{{user_input}}", input_user),
            text_format=HFModelInformation
        ).output_parsed 

        return response.dict()



    # ===========================================================================================================================0
    # This part is for pre-Data annotations.

    def llm_code_description(self, code: str)-> dict:
        # TODO gucke inwiefern die diese Funkltion nötig ist oder ob die andere File code als auch andere Files gut abdeckt
        """
        Generates the Output for the code descirption in the proper Json format
        """
        class metadatasRag(BaseModel):
            description: str
            keywords: list[str]
            
        # Load Prompt 
        prompt_code_description = utils.load_prompt_template("src/prompts/code_section_summary.txt")
        labels = utils.load_prompt_template("src/data/DUUIDictonary.txt")

        try:
            prompt_code_description = prompt_code_description.replace("{{labels}}", labels)
        except Exception:
            print("Error replacing labels in prompt. Check if the placeholder {{labels}} exists in the prompt template.")
            raise

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

        labels = utils.load_prompt_template("src/data/DUUIDictonary.txt")

        try:
            prompt_other_file = prompt_other_file.replace("{{labels}}", labels)
        except Exception:
            print("Error replacing labels in prompt. Check if the placeholder {{labels}} exists in the prompt template.")
            raise

        response = self.client.responses.parse(
            model=self.model,
            instructions=prompt_other_file,
            input=text,
            text_format=otherFileMetadata
        ).output_parsed
        return {"description": response.description, "keywords": response.keywords}
    