import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Make tests work regardless of absolute machine path: add repo `src` to sys.path
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

import llm_wrapper

HGF_NSFW_MODEL = {

            "model_id": "eliasalbouzidi/distilbert-nsfw-text-classifier",
            "revision": None,
            "transformers_version": "5.2.0",
            "torch_available": True,
            "device_used": "cpu",
            "model_type": "distilbert",
            "architectures": [
                "DistilBertForSequenceClassification"
            ],
            "num_labels": 2,
            "id2label": {
                "0": "safe",
                "1": "nsfw"
            },
            "label2id": {
                "nsfw": 1,
                "safe": 0
            },
            "problem_type": "single_label_classification",
            "tokenizer_class": "TokenizersBackend",
            "tokenizer_name_or_path": "eliasalbouzidi/distilbert-nsfw-text-classifier",
            "vocab_size": 30522,
            "model_max_length": 512,
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "inferred_task": "text-classification",
            "output_semantics_hint": "Sequence-level classification: output logits shape [batch, num_labels].",
            "recommended_postprocess": "softmax over logits; pick argmax/top-k.",
            "probe": {
                "output_is_modeloutput": True,
                "output_tuple_len": 1,
                "fields": {
                "logits": {
                    "shape": [
                    2,
                    2
                    ],
                    "dtype": "torch.float32"
                },
                "loss": {
                    "shape": None,
                    "dtype": "None"
                }
                },
                "available_keys": [
                "logits"
                ]
            },
            "warnings": []
            }
TYPESYSTEM_NSFW =  """
    <?xml version="1.0" encoding="UTF-8"?>
    <!-- DUUI TypeSystem XML for NSFW text classification -->
    <typeSystemDescription xmlns="http://uima.apache.org/resourceSpecifier">
        <name>NSFWTextClassificationTypeSystem</name>
        <description>TypeSystem for NSFW text classification using eliasalbouzidi/distilbert-nsfw-text-classifier</description>
        <version>1.0</version>

        <types>
            <typeDescription>
                <name>org.texttechnologylab.annotation.NSFW</name>
                <description>NSFW classification result from eliasalbouzidi/distilbert-nsfw-text-classifier</description>
                <supertypeName>uima.tcas.Annotation</supertypeName>

                <features>
                    <featureDescription>
                        <name>label</name>
                        <description>Classification label</description>
                        <rangeTypeName>uima.cas.String</rangeTypeName>
                    </featureDescription>
                    <featureDescription>
                        <name>score</name>
                        <description>Confidence score (0.0 to 1.0)</description>
                        <rangeTypeName>uima.cas.Double</rangeTypeName>
                    </featureDescription>
                </features>
            </typeDescription>
        </types>
    </typeSystemDescription>            
"""


class TestLLMWrapper(unittest.TestCase):
    def test_init_sets_llm_disabled_from_env(self):
        with patch.dict(os.environ, {"LLM_DISABLE": "true"}), \
             patch("llm_wrapper.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            wrapper = llm_wrapper.LLMWrapper()

        self.assertTrue(wrapper.llm_disabled)
        self.assertEqual(wrapper.model, llm_wrapper.MODEL_NAME_2)

    def test_gen_response_calls_openai(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("llm_wrapper.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses.parse.return_value = MagicMock(output_text="ok")
            wrapper = llm_wrapper.LLMWrapper()

            out = wrapper.gen_response("hi", "inst")

        self.assertEqual(out, "ok")
        mock_client.responses.parse.assert_called_once_with(
            model=wrapper.model,
            instructions="inst",
            input="hi",
        )

    def test_llm_code_description(self):
        test_file = """
        class Settings(BaseSettings):
            # Name of this annotator
            annotator_name: str
            # Version of this annotator
            annotator_version: str
            # Log level
            log_level: str
            # model_name
            model_name: str
            # Name of this annotator
            model_version: str
            #cach_size
            model_cache_size: int
            # url of the model
            model_source: str
            # language of the model
            model_lang: str
        """
        llm = llm_wrapper.LLMWrapper()
        response = llm.llm_code_description(test_file)
        print(type(response))
        print(response)
        self.assertIs(type(response), dict)
        self.assertIsNotNone(response["description"])
        self.assertIs(list, type(response["keywords"]))

    def test_llm_rewrite_query(self):
        test_query_1 = """
                    package org.example;

            import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
            import org.apache.uima.UIMAException;
            import org.apache.uima.fit.factory.JCasFactory;
            import org.apache.uima.fit.util.JCasUtil;
            import org.apache.uima.jcas.JCas;
            import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
            import org.texttechnologylab.annotation.Hate;
            import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
            import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
            import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

            import java.util.Arrays;
            import java.util.Collection;
            import java.util.List;

            public class testHate {

                static DUUIComposer composer;
                static JCas cas;

                static String url = "http://127.0.0.1:9714";

                public static void main(String[] args) throws Exception {
                    // initialize DUUI composer and remote driver
                    composer = new DUUIComposer()
                            .withSkipVerification(true)
                            .withLuaContext(new DUUILuaContext().withJsonLibrary());

                    DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
                    composer.addDriver(remoteDriver);

                    cas = JCasFactory.createJCas();

                    List<String> sentences = Arrays.asList(
                            "I hate hate it. How can you do that bad thing to me! HOW!",
                            "I very happy to be here. I love this place."
                    );

                    createCas("en", sentences);

                    // run the DUUI pipeline on the CAS
                    composer.run(cas);

                    // print out detected Hate annotations with their scores
                    Collection<Hate> all_hate = JCasUtil.select(cas, Hate.class);
                    for (Hate hate : all_hate) {
                        int begin = hate.getBegin();
                        int end = hate.getEnd();
                        double hate_i = hate.getHate();
                        double non_hate = hate.getNonHate();
                        String label = (hate_i < non_hate) ? "NonHate" : "HATE";
                        System.out.println(begin + "_" + end + " -> " + label +
                                " (hate=" + hate_i + ", non_hate=" + non_hate + ")");
                    }
                }

                static void createCas(String language, List<String> sentences) throws UIMAException {
                    cas.setDocumentLanguage(language);

                    StringBuilder sb = new StringBuilder();
                    for (String sentence : sentences) {
                        Sentence sentenceAnnotation = new Sentence(cas, sb.length(), sb.length() + sentence.length());
                        sentenceAnnotation.addToIndexes();
                        sb.append(sentence).append(" ");
                    }

                    cas.setDocumentText(sb.toString());
                }
            } das ist mein code warum kommt keine antwort auch wenn der docker läuft
        """
        llm = llm_wrapper.LLMWrapper()
        response = llm.llm_rewrite_query(test_query_1)
        print(type(response))
        self.assertIs(type(response), dict)
        self.assertIsNotNone(response["description"])
        self.assertIs(list, type(response["subqueries"]))
        self.assertGreater(len(response["subqueries"], 2))


    def test_typesystem_builder(self):
        llm = llm_wrapper.LLMWrapper()

        response = llm.llm_typesystem_builder(hf_model_json=HGF_NSFW_MODEL)
        print(response)

    def test_code_description_labels(self):
        file = ""
        llm = llm_wrapper.LLMWrapper()
        response = llm.llm_code_description()

    def test_lua_code_generation(self):
        llm = llm_wrapper.LLMWrapper()
        response = llm.llm_lua_code_builder(HGF_NSFW_MODEL, TYPESYSTEM_NSFW)
        print(response)

    def test_gen_dockerfile(self):
        llm = llm_wrapper.LLMWrapper()
        response = llm.llm_dockerfile_builder(HGF_NSFW_MODEL, "NSFW")
        print(response)

    def test_gen_dockerbuilder(self):
        llm = llm_wrapper.LLMWrapper()
        response = llm.llm_docker_build_builder(HGF_NSFW_MODEL, "NSFW")
        print(response)

    def test_gen_pythonModel(self):
        llm = llm_wrapper.LLMWrapper()
        response = llm.llm_python_code_builder(HGF_NSFW_MODEL, "NSFW")
        print(response)

    def test_lua_script_gen_new(self):
        llm = llm_wrapper.LLMWrapper()
        
        output_lua = llm.llm_lua_code_builder(hf_model_json=HGF_NSFW_MODEL)
        print(output_lua)
        self.assertTrue(True)



if __name__ == "__main__":
    unittest.main()
