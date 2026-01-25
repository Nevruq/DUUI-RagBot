import chromadb
import ollama
import unittest
import sys

sys.path.insert(1, "/home/nev/Documents/Bachelor/DUUI-RagBot/src")

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


if __name__ == "__main__":
    unittest.main()



