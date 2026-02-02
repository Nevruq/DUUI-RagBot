import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(1, "/home/nev/Documents/Bachelor/DUUI-RagBot/src")

import llm_wrapper


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

    def test_llm_code_description_disabled(self):
        with patch.dict(os.environ, {"LLM_DISABLE": "1"}), \
             patch("llm_wrapper.OpenAI") as mock_openai, \
             patch("llm_wrapper.utils.load_prompt_template", return_value="PROMPT"):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            wrapper = llm_wrapper.LLMWrapper()

            out = wrapper.llm_code_description("code")

        self.assertIn("N.A", out)
        mock_client.responses.parse.assert_not_called()

    def test_llm_code_description_enabled(self):
        with patch.dict(os.environ, {"LLM_DISABLE": "false"}), \
             patch("llm_wrapper.OpenAI") as mock_openai, \
             patch("llm_wrapper.utils.load_prompt_template", return_value="PROMPT"):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses.parse.return_value = MagicMock(output_text="RESULT")
            wrapper = llm_wrapper.LLMWrapper()

            out = wrapper.llm_code_description("code")

        self.assertEqual(out, "RESULT")
        mock_client.responses.parse.assert_called_once()

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



if __name__ == "__main__":
    unittest.main()
