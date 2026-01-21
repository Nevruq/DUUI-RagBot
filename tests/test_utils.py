import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(1, "/home/nev/Documents/Bachelor/DUUI-RagBot/src")

import utils


class TestUtils(unittest.TestCase):
    def test_load_prompt_template_reads_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompt.txt"
            path.write_text("hello prompt", encoding="utf-8")
            content = utils.load_prompt_template(str(path))
        self.assertEqual(content, "hello prompt")

    def test_short_no_truncation(self):
        text = "short text"
        self.assertEqual(utils.short(text, n=50), text)

    def test_short_truncation(self):
        text = "a" * 2500
        shortened = utils.short(text, n=2000)
        self.assertIn("... [truncated] ...", shortened)
        self.assertTrue(len(shortened) < len(text))


if __name__ == "__main__":
    unittest.main()
