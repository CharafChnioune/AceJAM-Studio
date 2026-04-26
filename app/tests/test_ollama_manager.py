import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


os.environ.setdefault("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "1")
acejam_app = importlib.import_module("app")
local_llm = importlib.import_module("local_llm")


class FakeOllamaClient:
    def list(self):
        return SimpleNamespace(
            models=[
                SimpleNamespace(
                    model="qwen3:4b",
                    size=4_000_000_000,
                    modified_at="2026-04-25T00:00:00Z",
                    digest="abc",
                    details=SimpleNamespace(family="qwen", parameter_size="4B", quantization_level="Q4"),
                ),
                SimpleNamespace(
                    model="nomic-embed-text:latest",
                    size=300_000_000,
                    modified_at="2026-04-25T00:00:00Z",
                    digest="def",
                    details=SimpleNamespace(family="nomic", parameter_size="", quantization_level=""),
                ),
            ]
        )

    def ps(self):
        return SimpleNamespace(models=[SimpleNamespace(model="qwen3:4b")])

    def pull(self, model, stream=True):
        yield SimpleNamespace(status="pulling manifest")
        yield SimpleNamespace(status="downloading", completed=50, total=100, digest="sha256:abcd")
        yield SimpleNamespace(status="success")

    def embed(self, model, input):
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])

    def chat(self, model, messages, think=False, options=None):
        return SimpleNamespace(message=SimpleNamespace(content="OK"))

    def show(self, model):
        return SimpleNamespace(model=model, details={"family": "qwen"})


class OllamaManagerTest(unittest.TestCase):
    def setUp(self):
        acejam_app._ollama_pull_jobs.clear()

    def test_catalog_splits_chat_and_embedding_models(self):
        with patch.object(acejam_app, "_ollama_client", return_value=FakeOllamaClient()):
            data = acejam_app._ollama_model_catalog()

        self.assertTrue(data["ready"])
        self.assertEqual(data["chat_models"], ["qwen3:4b"])
        self.assertEqual(data["embedding_models"], ["nomic-embed-text:latest"])
        self.assertEqual(data["running_models"], ["qwen3:4b"])

    def test_pull_worker_records_streaming_progress(self):
        with patch.object(acejam_app, "_ollama_client", return_value=FakeOllamaClient()):
            acejam_app._set_ollama_pull_job("job1", model="new-model:latest", kind="chat")
            acejam_app._ollama_pull_worker("job1", "new-model:latest")

        job = acejam_app._ollama_pull_job("job1")
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(job["progress"], 100)
        self.assertIn("new-model:latest installed", "\n".join(job["logs"]))

    def test_missing_selected_model_starts_pull(self):
        with patch.object(acejam_app, "_ollama_client", return_value=FakeOllamaClient()):
            with self.assertRaises(acejam_app.OllamaPullStarted) as raised:
                acejam_app._ensure_ollama_model_or_start_pull("missing-model:latest", "test", "chat")

        self.assertEqual(raised.exception.model_name, "missing-model:latest")
        self.assertIn(raised.exception.job["state"], {"queued", "running", "succeeded"})

    def test_api_ollama_test_embedding_uses_embed(self):
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_ollama_client", return_value=FakeOllamaClient()):
            response = client.post("/api/ollama/test", json={"model": "nomic-embed-text:latest", "kind": "embedding"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["dimensions"], 3)

    def test_lmstudio_catalog_splits_chat_and_embedding_models(self):
        def fake_http(method, url, payload=None, timeout=30.0):
            self.assertEqual(method, "GET")
            self.assertIn("/api/v1/models", url)
            return {
                "models": [
                    {"key": "qwen-local", "type": "llm", "format": "gguf", "quantization": "Q4_K_M", "loaded_instances": [{"identifier": "qwen-local", "config": {"context_length": 8192}}]},
                    {"key": "embed-local", "type": "embedding", "format": "gguf"},
                ]
            }

        with patch.object(local_llm, "_http_json", side_effect=fake_http):
            data = local_llm.lmstudio_model_catalog()

        self.assertTrue(data["ready"])
        self.assertEqual(data["chat_models"], ["qwen-local"])
        self.assertEqual(data["embedding_models"], ["embed-local"])
        self.assertEqual(data["loaded_models"], ["qwen-local"])
        self.assertEqual(data["details"][0]["loaded_context_length"], 8192)

    def test_lmstudio_load_model_sets_context_with_native_payload(self):
        calls = []

        def fake_http(method, url, payload=None, timeout=30.0):
            calls.append((method, url, payload, timeout))
            return {"status": "loaded", "load_config": {"context_length": 32768}}

        with patch.object(local_llm, "_http_json", side_effect=fake_http):
            result = local_llm.lmstudio_load_model("qwen-local", kind="chat", context_length=32768)

        self.assertTrue(result["success"])
        self.assertEqual(calls[0][0], "POST")
        self.assertIn("/api/v1/models/load", calls[0][1])
        self.assertEqual(calls[0][2]["model"], "qwen-local")
        self.assertNotIn("model_key", calls[0][2])
        self.assertEqual(calls[0][2]["context_length"], 32768)
        self.assertTrue(calls[0][2]["echo_load_config"])

    def test_lmstudio_chat_filters_non_openai_options(self):
        seen_payloads = []

        def fake_http(method, url, payload=None, timeout=30.0):
            seen_payloads.append(payload)
            return {"choices": [{"message": {"content": "OK"}}]}

        with patch.object(local_llm, "_http_json", side_effect=fake_http):
            result = local_llm.lmstudio_chat(
                "qwen-local",
                [{"role": "user", "content": "Reply OK."}],
                options={"temperature": 0.1, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.1, "num_ctx": 32768},
            )

        self.assertEqual(result, "OK")
        self.assertEqual(seen_payloads[0]["temperature"], 0.1)
        self.assertEqual(seen_payloads[0]["top_p"], 0.8)
        self.assertEqual(seen_payloads[0]["top_k"], 20)
        self.assertEqual(seen_payloads[0]["repeat_penalty"], 1.1)
        self.assertNotIn("num_ctx", seen_payloads[0])

    def test_api_local_llm_test_uses_lmstudio_chat(self):
        client = TestClient(acejam_app.app)

        def fake_catalog():
            return {"ready": True, "models": ["qwen-local"], "chat_models": ["qwen-local"], "embedding_models": [], "details": []}

        with patch.object(acejam_app, "lmstudio_model_catalog", side_effect=fake_catalog), patch.object(local_llm, "lmstudio_model_catalog", side_effect=fake_catalog), patch.object(local_llm, "_http_json", return_value={"choices": [{"message": {"content": "OK"}}]}):
            response = client.post("/api/local-llm/test", json={"provider": "lmstudio", "model": "qwen-local", "kind": "chat"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["provider"], "lmstudio")
        self.assertEqual(data["response"], "OK")

    def test_api_local_llm_test_uses_lmstudio_embeddings(self):
        client = TestClient(acejam_app.app)

        def fake_catalog():
            return {"ready": True, "models": ["embed-local"], "chat_models": [], "embedding_models": ["embed-local"], "details": []}

        with patch.object(acejam_app, "lmstudio_model_catalog", side_effect=fake_catalog), patch.object(local_llm, "lmstudio_model_catalog", side_effect=fake_catalog), patch.object(local_llm, "_http_json", return_value={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}):
            response = client.post("/api/local-llm/test", json={"provider": "lmstudio", "model": "embed-local", "kind": "embedding"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["dimensions"], 4)


if __name__ == "__main__":
    unittest.main()
