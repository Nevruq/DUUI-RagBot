import unittest
import sys
import os
import chromadb
from unittest.mock import patch, MagicMock
from pathlib import Path

# Make tests work regardless of absolute machine path: add repo `src` to sys.path
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

import llm_wrapper
import RAG



class TestRAG(unittest.TestCase):
    def test_query_results(self):
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
            } Wie starte ich den Docker dafür
        """
        query_results_reg = RAG.query_results(query_input=test_query_1, collection_name="all_data_v1", n_results=3)
        filtered_query = llm_wrapper.LLMWrapper().llm_rewrite_query(test_query_1)
        print(filtered_query)
        query_results_filtert = RAG.query_results(query_input=filtered_query["subqueries"][0], collection_name="all_data_v1", n_results=3)
        print(query_results_reg["metadatas"])
        print(query_results_filtert["metadatas"])
        self.assertTrue(True)


    def test_distinct_file_query(self):
        

if __name__ == "__main__":
    unittest.main()
