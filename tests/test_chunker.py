import unittest
import sys
import json
from unittest.mock import patch

sys.path.insert(1, "/home/nev/Documents/Bachelor/DUUI-RagBot/src")

import chunk_data.rag_chunk as rc
import chunk_data.chunk_java as cj
import utils

TEST_FILE_PY = "src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"
TEST_FILE_JAVA = "src/data/duui-uima/duui-Hate/src/test/java/org/hucompute/textimager/uima/hate/MultiTestHate.java"

class TestChunker(unittest.TestCase):

    def test_safe_read(self):
        code = rc._safe_read(path=TEST_FILE_JAVA)
        self.assertIsInstance(code, str)
        self.assertTrue(code)


    def test_parse_description_response(self):
        test_string = str({"description": "N.A", "keywords": ["file:unknown", "code", "summary"]})
        parsed_string = rc._parse_description_response(test_string)

        self.assertIsInstance(parsed_string, dict)

        
    def test_chunk_java_code(self):
        chunks = cj.chunk_java_file(path=TEST_FILE_JAVA)
        self.assertTrue(chunks)
        chunk = chunks[0]
        print(chunk.description)
        code_desc = cj._gen_code_description(chunk.text)
        print(code_desc)
        chunk.append_llm_data(code_desc)
        print(chunk.description)
        

    def test_chunk_to_chroma_item(self):
        test_chunk = rc.CodeChunk(
            text="def foo(x):\n    return x + 1\n",
            file="src/example.py",
            language="python",
            symbol_type="function",
            symbol_name="foo",
            start_line=10,
            end_line=11,
            description="Adds 1 to the input and returns the result.",
            keywords=["file:example.py", "function", "math"],
        )
        chunk_formated = rc.chunk_to_chroma_item(test_chunk)
        self.assertTrue(True)
        print(chunk_formated["metadata"])
    
    def test_to_json(self):
        test_chunk = rc.RAGChunk(
            text="class Foo { void bar() {} }",
            file="src/example.java",
            language="java",
            symbol_type="class",
            symbol_name="Foo",
            start_line=1,
            end_line=1,
            description="Simple Java class.",
            keywords=["file:example.java", "class", "java"],
            chunk_type="code",
            repo_id="repo::test",
        )
        with patch.object(rc.RAGChunk, "gen_embedding_meta", return_value=[0.0, 0.1]):
            json_item = test_chunk.to_json_item()

        self.assertIsInstance(json_item, dict)
        try:
            json.dumps(json_item)
        except TypeError as exc:
            self.fail(f"to_json_item returned non-JSON-serializable data: {exc}")
        print(json_item)
        with open("src/data/json_test_dump.txt", "a") as f:
            f.write(json.dumps(json_item))
    
    def test_chunk_to_jsonl(self):
        chunks = cj.chunk_java_file(TEST_FILE_JAVA)
        utils.write_ragchunks_jsonl(chunks=chunks, path="src/data/jsonl_test.jsonl")

    def test_infer_file(self):
        chunks = rc.chunk_file(TEST_FILE_JAVA)
        print(chunks[0].chunk_type)        


if __name__ == "__main__":
    unittest.main()
