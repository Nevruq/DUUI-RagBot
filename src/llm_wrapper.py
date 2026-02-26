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


    
    def llm_code_assistant(
        self,
        input_user: str,
        collection_name: str,
        coding_lg: str = "",
        rewrite_query: bool = True,
        rag_context: bool = True,
        stream: bool = False,
    ):
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
        if stream:
            stream_resp = self.client.responses.create(
                model=self.model,
                instructions="You are a DUUI assitant and answer question about.",
                input=concat_prompt,
                stream=True,
            )

            def _iter_text_deltas():
                for event in stream_resp:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type == "response.refusal.delta":
                        yield event.delta

            return _iter_text_deltas()

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI assitant and answer question about.",
            input=concat_prompt
        )

        return response.output_text



    def llm_lua_code_builder(
        self,
        hf_model_json: dict,
        typesystem_xml: str,
        lua_template_path: str = "example_docker_component/lua_template.lua"
    ) -> str:
        """
        Generates Lua code for DUUI communication scripts based on Hugging Face model information.
        Uses CAS API exclusively (no JCas classes).

        Args:
            hf_model_json: Dictionary containing Hugging Face model information
                Required fields:
                - model_id: str (e.g., "debajyotimaz/codemix_hate")
                - inferred_task: str (e.g., "text-classification")
                - id2label: dict (e.g., {"0": "Non-hateful", "1": "Hateful"})
                - num_labels: int
                - model_type: str (e.g., "bert")
                - architectures: list[str]
            typesystem_xml: The TypeSystem XML string that defines the annotation type and features
            lua_template_path: Path to the Lua template file

        Returns:
            Generated Lua code as string (uses CAS API, not JCas)
        """
        if self.llm_disabled:
            return "-- LLM disabled, cannot generate Lua script"

        # Load prompt template
        prompt_lua_code_builder = utils.load_prompt_template("src/prompts/lua_code_builder.txt")

        # Load Lua template
        lua_template = utils.load_prompt_template(lua_template_path)

        # Format HF model JSON for the prompt
        hf_model_json_str = json.dumps(hf_model_json, indent=2, ensure_ascii=False)

        # Build final prompt
        concat_prompt = (
            prompt_lua_code_builder
            .replace("{{hf_model_json}}", hf_model_json_str)
            .replace("{{lua_template}}", lua_template)
            .replace("{{typesystem_info}}", typesystem_xml)
        )

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI Lua code generator. Generate a complete Lua script using CAS API (not JCas). Output ONLY the Lua code, no explanations.",
            input=concat_prompt
        )

        return response.output_text


    def llm_typesystem_builder(
        self,
        hf_model_json: dict,
        typesystem_template_path: str = "example_docker_component/typesystem_template.xml"
    ) -> str:
        """
        Generates a UIMA TypeSystem XML based on Hugging Face model information.

        Args:
            hf_model_json: Dictionary containing Hugging Face model information
                Required fields:
                - model_id: str (e.g., "debajyotimaz/codemix_hate")
                - inferred_task: str (e.g., "text-classification")
                - id2label: dict (e.g., {"0": "Non-hateful", "1": "Hateful"})
                - num_labels: int
                - model_type: str (e.g., "bert")
                - architectures: list[str]
            typesystem_template_path: Path to the TypeSystem template file

        Returns:
            Generated TypeSystem XML as string
        """
        if self.llm_disabled:
            return "<!-- LLM disabled, cannot generate TypeSystem -->"

        # Load prompt template
        prompt_typesystem = utils.load_prompt_template("src/prompts/typesystem_builder.txt")

        # Load TypeSystem template
        typesystem_template = utils.load_prompt_template(typesystem_template_path)

        # Format HF model JSON for the prompt
        hf_model_json_str = json.dumps(hf_model_json, indent=2, ensure_ascii=False)

        # Build final prompt
        concat_prompt = (
            prompt_typesystem
            .replace("{{hf_model_json}}", hf_model_json_str)
            .replace("{{typesystem_template}}", typesystem_template)
        )

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI TypeSystem generator. Generate a complete TypeSystem XML. Output ONLY the XML, no explanations.",
            input=concat_prompt
        )

        return response.output_text


    def llm_dockerfile_builder(
        self,
        hf_model_json: dict,
        component_name: str,
        dockerfile_template_path: str = "example_docker_component/dockerfile_template.dockerfile"
    ) -> str:
        """
        Generates a Dockerfile based on Hugging Face model information.

        Args:
            hf_model_json: Dictionary containing Hugging Face model information
                Required fields:
                - model_id: str (e.g., "debajyotimaz/codemix_hate")
                - inferred_task: str (e.g., "text-classification")
                - architectures: list[str]
            component_name: Unified component name for consistent file naming
                (e.g., "hate" -> duui_hate.py, duui_hate.lua, duui-hate image)
            dockerfile_template_path: Path to the Dockerfile template

        Returns:
            Generated Dockerfile as string
        """
        if self.llm_disabled:
            return "# LLM disabled, cannot generate Dockerfile"

        # Load prompt template
        prompt_dockerfile = utils.load_prompt_template("src/prompts/dockerfile_builder.txt")

        # Load Dockerfile template
        dockerfile_template = utils.load_prompt_template(dockerfile_template_path)

        # Format HF model JSON for the prompt
        hf_model_json_str = json.dumps(hf_model_json, indent=2, ensure_ascii=False)

        # Build final prompt
        concat_prompt = (
            prompt_dockerfile
            .replace("{{hf_model_json}}", hf_model_json_str)
            .replace("{{dockerfile_template}}", dockerfile_template)
            .replace("{{component_name}}", component_name)
        )

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI Dockerfile generator. Generate a complete Dockerfile. Output ONLY the Dockerfile, no explanations.",
            input=concat_prompt
        )

        response.output_text


    def llm_docker_build_builder(
        self,
        hf_model_json: dict,
        component_name: str,
        docker_build_template_path: str = "example_docker_component/docker_build_template.sh"
    ) -> str:
        """
        Generates a docker_build.sh script based on Hugging Face model information.

        Args:
            hf_model_json: Dictionary containing Hugging Face model information
                Required fields:
                - model_id: str (e.g., "debajyotimaz/codemix_hate")
                - revision: str (optional, for version)
            component_name: Unified component name for consistent file naming
                (e.g., "hate" -> duui-hate image name)
            docker_build_template_path: Path to the docker_build.sh template

        Returns:
            Generated docker_build.sh script as string
        """
        if self.llm_disabled:
            return "#!/bin/bash\n# LLM disabled, cannot generate docker_build.sh"

        # Load prompt template
        prompt_docker_build = utils.load_prompt_template("src/prompts/docker_build_builder.txt")

        # Load docker_build template
        docker_build_template = utils.load_prompt_template(docker_build_template_path)

        # Format HF model JSON for the prompt
        hf_model_json_str = json.dumps(hf_model_json, indent=2, ensure_ascii=False)

        # Build final prompt
        concat_prompt = (
            prompt_docker_build
            .replace("{{hf_model_json}}", hf_model_json_str)
            .replace("{{docker_build_template}}", docker_build_template)
            .replace("{{component_name}}", component_name)
        )

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI docker build script generator. Generate a complete shell script. Output ONLY the script, no explanations.",
            input=concat_prompt
        )

        return response.output_text


    def llm_python_code_builder(
        self,
        hf_model_json: dict,
        component_name: str,
        python_template_path: str = "example_docker_component/python_template.py"
    ) -> str:
        """
        Generates a Python DUUI component based on Hugging Face model information.

        Args:
            hf_model_json: Dictionary containing Hugging Face model information
                Required fields:
                - model_id: str (e.g., "debajyotimaz/codemix_hate")
                - inferred_task: str (e.g., "text-classification")
                - id2label: dict (e.g., {"0": "Non-hateful", "1": "Hateful"})
                - num_labels: int
                - architectures: list[str]
            component_name: Unified component name for consistent file naming
                (e.g., "hate" -> duui_hate.py, duui_hate.lua)
            python_template_path: Path to the Python template file

        Returns:
            Generated Python code as string
        """
        if self.llm_disabled:
            return "# LLM disabled, cannot generate Python script"

        # Load prompt template
        prompt_python = utils.load_prompt_template("src/prompts/python_code_builder.txt")

        # Load Python template
        python_template = utils.load_prompt_template(python_template_path)

        # Format HF model JSON for the prompt
        hf_model_json_str = json.dumps(hf_model_json, indent=2, ensure_ascii=False)

        # Build final prompt
        concat_prompt = (
            prompt_python
            .replace("{{hf_model_json}}", hf_model_json_str)
            .replace("{{python_template}}", python_template)
            .replace("{{component_name}}", component_name)
        )

        response = self.client.responses.create(
            model=self.model,
            instructions="You are a DUUI Python code generator. Generate a complete Python FastAPI script. Output ONLY the Python code, no explanations.",
            input=concat_prompt
        )

        return response.output_text


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

        return response.model_dump()



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
    
