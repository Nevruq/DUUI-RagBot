import chromadb
import ollama
import unittest
import sys
from pathlib import Path

# Make tests work regardless of absolute machine path: add repo `src` to sys.path
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

import import_data


TEST_FILE_PY = "src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"
TEST_FILE_JAVA = "src/data/duui-uima/duui-Hate/src/test/java/org/hucompute/textimager/uima/hate/MultiTestHate.java"
TEST_FILE_DOCKER = "src/data/duui-uima/duui-Hate/src/main/docker/Dockerfile"

class TestChunker(unittest.TestCase):

    def test_repo_id(self):
        chunks_java = import_data.chunk_file(TEST_FILE_JAVA)
        chunks_other = import_data.chunk_file(TEST_FILE_DOCKER)
        test = chunks_java[0]
        test_d = chunks_other[0]
        print(test_d.repo_id)
        #self.assertEqual("java", test)  
        self.assertEqual(test.repo_id, test_d.repo_id) 

    def test_query_results(self):
        pass

    def test_whole_pipeline(self):
        """
        Docstring for test_whole_pipeline. Given a example file. Chunking it -> Generate Description  (THREADING) -> Cast to JSON Item.
        """
        file_path = "src/data/duui-uima/duui-Hate"

        from utils import filter_files
        from import_data import load_data

        files = filter_files(file_path)[:2]
        print(len(files))
        load_data(LIST_FILES=files, output_jsonl="src/data/DUUI_v1.jsonl")
    

        

if __name__ == "__main__":
    unittest.main()



