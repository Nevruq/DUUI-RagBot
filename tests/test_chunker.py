import unittest
import sys
from unittest.mock import patch

sys.path.insert(1, "/home/nev/Documents/Bachelor/DUUI-RagBot/src")

import chunker
import utils

TEST_FILE = "src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"

class TestChunker(unittest.TestCase):

    def test_safe_read(self):
        code = chunker._safe_read(path=TEST_FILE)
        self.assertIsInstance(code, str)
        self.assertTrue(code)


    def test_parse_description_response(self):
        test_string = str({"description": "N.A", "keywords": ["file:unknown", "code", "summary"]})
        parsed_string = chunker._parse_description_response(test_string)

        self.assertIsInstance(parsed_string, dict)


    def test_gen_code_description(self):
        text="def foo(x):\n    return x + 1\n"
        result = chunker._gen_code_description(text)

        # test if outputs of the LLM reponse suit the proper formatting
        self.assertIsInstance(result, dict)
        self.assertIn("description", result)
        self.assertIn("keywords", result)
        self.assertIsInstance(result["keywords"], list)
        # When the formatting was not correct, we test for the dafault return value
        self.assertEqual(result["description"], "N.A")
        
    def test_chunk_python_code(self):
        text="def foo(x):\n    return x + 1\n"
        chunks = chunker.chunk_python_code(code=text, file_path=TEST_FILE)
        
        self.assertTrue(chunks)
        chunk = chunks[0]
        # testing just one chunk
        self.assertIsInstance(chunk, chunker.CodeChunk)
        self.assertIn("code_description", chunk.meta)
        self.assertIn("keywords", chunk.meta)
        self.assertIsInstance(chunk.meta["code_description"], str)
        self.assertIsInstance(chunk.meta["keywords"], str)
        print(chunk)

    def test_chunk_to_chroma_item(self):
        test_chunk = chunker.CodeChunk(
            text="def foo(x):\n    return x + 1\n",
            file="src/example.py",
            language="python",
            symbol_type="function",
            symbol_name="foo",
            start_line=10,
            end_line=11,
            code_description="Adds 1 to the input and returns the result.",
            keywords=["file:example.py", "function", "math"],
        )
        chunk_formated = chunker.chunk_to_chroma_item(test_chunk)
        self.assertTrue(True)
        print(chunk_formated["metadata"])


if __name__ == "__main__":
    unittest.main()
