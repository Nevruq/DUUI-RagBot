import unittest
import sys
import tempfile
from pathlib import Path

# Make tests work regardless of absolute machine path: add repo `src` to sys.path
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

import utils


class TestUtils(unittest.TestCase):
    def test_load_prompt_template_reads_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompt.txt"
            path.write_text("hello prompt", encoding="utf-8")
            content = utils.load_prompt_template(str(path))
        self.assertEqual(content, "hello prompt")


    def test_filter_files(self):
        test_path = "src/data/duui-uima/duui-Hate"
        filter_files = utils.filter_files(test_path)
        self.assertIsNotNone(filter_files)
        filter_files_py = utils.filter_files(test_path, set(".py"))
        print(filter_files)
        print(filter_files_py)

    def test_xml_validater(self):
        test_path = "src/data/UIMATypeSystem/src/main/resources/desc/type/TextTechnologySentimentBert.xml"
        errors = utils.validate_typesystem(test_path)
        self.assertEqual(len(errors), 0)

    def test_filter_files(self):
        test_path = "src/data/duui-uima/duui-Hate"
        filter_files = utils.filter_files(test_path)
        self.assertIsNotNone(filter_files)
        filter_files_py = utils.filter_files(test_path, set(".py"))
        print(filter_files)
        print(filter_files_py)

if __name__ == "__main__":
    unittest.main()
