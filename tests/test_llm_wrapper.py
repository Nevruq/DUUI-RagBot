import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(1, "/home/nev/Documents/Bachelor/DUUI-RagBot/src")

import llm_wrapper


class TestLLMWrapper(unittest.TestCase):
    def test_init_sets_llm_disabled_from_env(self):
        with patch.dict(os.environ, {"LLM_DISABLE": "true"}), \
             patch("llm_wrapper.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            wrapper = llm_wrapper.LLMWrapper()

        self.assertTrue(wrapper.llm_disabled)
        self.assertEqual(wrapper.model, llm_wrapper.MODEL_NAME_2)

    def test_gen_response_calls_openai(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("llm_wrapper.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses.parse.return_value = MagicMock(output_text="ok")
            wrapper = llm_wrapper.LLMWrapper()

            out = wrapper.gen_response("hi", "inst")

        self.assertEqual(out, "ok")
        mock_client.responses.parse.assert_called_once_with(
            model=wrapper.model,
            instructions="inst",
            input="hi",
        )

    def test_llm_code_description_disabled(self):
        with patch.dict(os.environ, {"LLM_DISABLE": "1"}), \
             patch("llm_wrapper.OpenAI") as mock_openai, \
             patch("llm_wrapper.utils.load_prompt_template", return_value="PROMPT"):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            wrapper = llm_wrapper.LLMWrapper()

            out = wrapper.llm_code_description("code")

        self.assertIn("N.A", out)
        mock_client.responses.parse.assert_not_called()

    def test_llm_code_description_enabled(self):
        with patch.dict(os.environ, {"LLM_DISABLE": "false"}), \
             patch("llm_wrapper.OpenAI") as mock_openai, \
             patch("llm_wrapper.utils.load_prompt_template", return_value="PROMPT"):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.responses.parse.return_value = MagicMock(output_text="RESULT")
            wrapper = llm_wrapper.LLMWrapper()

            out = wrapper.llm_code_description("code")

        self.assertEqual(out, "RESULT")
        mock_client.responses.parse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
