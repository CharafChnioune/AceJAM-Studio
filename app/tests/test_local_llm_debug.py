import unittest
from unittest.mock import patch

import local_llm


class LocalLLMDebugPrintTest(unittest.TestCase):
    def test_ollama_chat_prints_full_request_and_response(self):
        with patch.object(local_llm, "ACEJAM_PRINT_LLM_IO", True), \
            patch.object(local_llm, "_http_json", return_value={"message": {"content": "{\"ok\": true}"}}), \
            patch("builtins.print") as mocked_print:
            result = local_llm.ollama_chat(
                "unit-model",
                [{"role": "user", "content": "unit prompt"}],
                options={"num_predict": 8},
                json_format=True,
            )

        printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertEqual(result, "{\"ok\": true}")
        self.assertIn("[acejam_llm_io][BEGIN ollama_chat_request_json", printed)
        self.assertIn("/api/chat", printed)
        self.assertIn("unit prompt", printed)
        self.assertIn("[acejam_llm_io][BEGIN ollama_chat_response_json", printed)
        self.assertIn("{\"ok\": true}", printed)

    def test_lmstudio_chat_prints_full_request_and_response(self):
        response = {"choices": [{"message": {"content": "{\"title\": \"Unit\"}"}}]}
        with patch.object(local_llm, "ACEJAM_PRINT_LLM_IO", True), \
            patch.object(local_llm, "_http_json", return_value=response), \
            patch("builtins.print") as mocked_print:
            result = local_llm.lmstudio_chat(
                "unit-lmstudio-model",
                [{"role": "user", "content": "unit lmstudio prompt"}],
                options={"max_tokens": 16},
                json_format=True,
            )

        printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertEqual(result, "{\"title\": \"Unit\"}")
        self.assertIn("[acejam_llm_io][BEGIN lmstudio_chat_request_json", printed)
        self.assertIn("/chat/completions", printed)
        self.assertIn("unit lmstudio prompt", printed)
        self.assertIn("[acejam_llm_io][BEGIN lmstudio_chat_response_json", printed)
        self.assertIn("{\"title\": \"Unit\"}", printed)


if __name__ == "__main__":
    unittest.main()
