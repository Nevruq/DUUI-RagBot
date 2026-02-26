import unittest
import sys
from pathlib import Path

# Make tests work regardless of absolute machine path: add repo `src` to sys.path
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

import DUUIComponent.load_hf_model as lhf

class TestHGF(unittest.TestCase):
    def test_inspectHFModel(self):
        hf_information = lhf.inspect_hf_model("eliasalbouzidi/distilbert-nsfw-text-classifier", return_json=True)
        print(hf_information)
        self.assertTrue(True)
        

if __name__ == "__main__":
    unittest.main()



