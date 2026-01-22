import unittest
import sys
from unittest.mock import patch

sys.path.insert(1, "/home/nev/Documents/Bachelor/DUUI-RagBot/src/")

from chunk_data.chunk_java import chunk_java_file
import utils

TEST_FILE_PY = "src/data/duui-uima/duui-Hate/src/main/python/duui_hate.py"
TEST_FILE_JAVA ="src/data/duui-uima/duui-Hate/src/test/java/org/hucompute/textimager/uima/hate/MultiTestHate.java"

class TestChunker(unittest.TestCase):

    def test_safe_read(self):

        self.assertTrue(True)

    def test_chunk_java_file(self):
        chunks = chunk_java_file(TEST_FILE_JAVA)
        print(len(chunks))
        



if __name__ == "__main__":
    unittest.main()
