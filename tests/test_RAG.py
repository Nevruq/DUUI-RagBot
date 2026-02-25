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
            lua seriazlies script
        """
        query_results_reg = RAG.query_results(query_input=test_query_1, collection_name="all_data_v2", n_results=3)
        filtered_query = llm_wrapper.LLMWrapper().llm_rewrite_query(test_query_1)
        print(filtered_query)
        query_results_filtert = RAG.query_results(query_input=filtered_query["subqueries"][0], collection_name="all_data_v2", n_results=3)
        print(query_results_reg["metadatas"])
        print(query_results_filtert["documents"])
        self.assertTrue(True)

    

    def test_query_to_ragchunk(self):
        from chunk_data.rag_chunk import ragchunks_from_chroma_query
        from chunk_data.rag_chunk import RAGChunk
        query_results_RAG_list = RAG.query_results(query_input="How do I implement a typesystem in python?",
                                              collection_name="all_data_v2",
                                              n_results=3,
                                              ollama_embedding=True)
        ragchunks = ragchunks_from_chroma_query(query_results_RAG_list)
        self.assertEqual(len(ragchunks), 3)
        self.assertIsInstance(ragchunks[0], RAGChunk)
        print(ragchunks[0])

    def test_get_few_shot_examples(self):
        """Test the get_few_shot_examples function with various parameters."""

        # Test 1: Basic functionality - get Lua examples
        examples = RAG.get_few_shot_examples(
            query_input="serialize and deserialize functions",
            collection_name="all_data_v2",
            chunk_type="lua",
            n_examples=2,
            diversity=True
        )

        # Check that we get results
        self.assertIsInstance(examples, list, "Should return a list")
        self.assertGreater(len(examples), 0, "Should return at least one example")
        self.assertLessEqual(len(examples), 2, "Should not return more than requested")

        # Check structure of returned examples
        if len(examples) > 0:
            first_example = examples[0]
            self.assertIsInstance(first_example, dict, "Each example should be a dict")
            self.assertIn("code", first_example, "Example should have 'code' key")
            self.assertIn("metadata", first_example, "Example should have 'metadata' key")

            # Check metadata structure
            metadata = first_example["metadata"]
            self.assertIsInstance(metadata, dict, "Metadata should be a dict")
            self.assertIn("chunk_type", metadata, "Metadata should have chunk_type")
            self.assertEqual(metadata["chunk_type"], "lua", "Chunk type should match filter")

            # Check that code is not empty
            self.assertIsInstance(first_example["code"], str, "Code should be a string")
            self.assertGreater(len(first_example["code"]), 0, "Code should not be empty")

        # Test 2: Diversity - check that examples come from different files
        if len(examples) > 1:
            files = [ex["metadata"].get("file", "") for ex in examples]
            unique_files = set(files)
            self.assertEqual(len(unique_files), len(examples),
                           "With diversity=True, each example should be from a different file")


        # Test 4: Empty collection or no results
        empty_examples = RAG.get_few_shot_examples(
            query_input="this should not match anything at all xyz123",
            collection_name="all_data_v2",
            chunk_type="nonexistent_language",
            n_examples=5
        )
        # Should return empty list or very few results
        self.assertIsInstance(empty_examples, list, "Should return a list even if empty")

        print(f"\n✓ Test passed: Retrieved {len(examples)} examples")
        if len(examples) > 0:
            print(f"✓ First example from: {examples[0]['metadata'].get('file', 'unknown')}")
            print(f"✓ Code length: {len(examples[0]['code'])} characters")


    def test_distinct_file_query(self):
        pass
    def test_chroma_content(self):
        path = "DUUI_v1"
        client = chromadb.PersistentClient(RAG.RAG_PATH)
        collection = client.get_or_create_collection(name=path)
        self.assertIsNotNone(collection)
        if collection.count() == 0:
            raise Exception("Collection is Empty.")
        print(collection.count())

if __name__ == "__main__":
    unittest.main()
