import importlib
import os
import tempfile
import unittest
from pathlib import Path
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
                SimpleNamespace(
                    model="x/flux2-klein:4b",
                    size=4_000_000_000,
                    modified_at="2026-04-25T00:00:00Z",
                    digest="ghi",
                    details=SimpleNamespace(family="flux", parameter_size="4B", quantization_level="Q4"),
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


class FakeOllamaClientNoImage(FakeOllamaClient):
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
            ]
        )


class OllamaManagerTest(unittest.TestCase):
    def setUp(self):
        acejam_app._ollama_pull_jobs.clear()

    def test_catalog_splits_chat_and_embedding_models(self):
        with patch.object(acejam_app, "_ollama_client", return_value=FakeOllamaClient()):
            data = acejam_app._ollama_model_catalog()

        self.assertTrue(data["ready"])
        self.assertEqual(data["chat_models"], ["qwen3:4b"])
        self.assertEqual(data["embedding_models"], ["nomic-embed-text:latest"])
        self.assertEqual(data["image_models"], ["x/flux2-klein:4b"])
        self.assertEqual(data["running_models"], ["qwen3:4b"])

    def test_ollama_kind_infers_image_generation_models(self):
        self.assertEqual(acejam_app._ollama_kind_from_model_name("x/flux2-klein:9b-bf16"), "chat")
        self.assertEqual(acejam_app._ollama_kind_from_model_name("x/z-image-turbo:bf16"), "chat")
        self.assertEqual(acejam_app._ollama_kind_from_model_name("nomic-embed-text:latest"), "embedding")
        self.assertEqual(acejam_app._ollama_kind_from_model_name("qwen3:4b"), "chat")

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

    def test_planner_llm_settings_clamp_and_map_to_ollama_options(self):
        # Context-length clamp ceiling raised to 262144 (Qwen 3.6 native max)
        # so power users can push to native context. Use 999999 here to
        # hit the new ceiling.
        payload = {
            "planner_creativity_preset": "wild",
            "planner_temperature": 3.0,
            "planner_top_p": -1,
            "planner_top_k": 999,
            "planner_repeat_penalty": 0.1,
            "planner_seed": "42",
            "planner_max_tokens": 99999,
            "planner_context_length": 999999,
            "planner_timeout": 99999999,
        }

        settings = local_llm.planner_llm_settings_from_payload(payload)
        options = local_llm.planner_llm_options_for_provider("ollama", payload)

        self.assertEqual(settings["planner_temperature"], 2.0)
        self.assertEqual(settings["planner_top_p"], 0.0)
        self.assertEqual(settings["planner_top_k"], 200)
        self.assertEqual(settings["planner_repeat_penalty"], 0.8)
        self.assertEqual(settings["planner_seed"], 42)
        self.assertEqual(settings["planner_max_tokens"], 8192)
        # New ceiling is 262144 (Qwen native context max)
        self.assertEqual(settings["planner_context_length"], 262144)
        self.assertEqual(settings["planner_timeout"], 2592000.0)
        self.assertEqual(options["temperature"], 2.0)
        self.assertEqual(options["top_p"], 0.0)
        self.assertEqual(options["top_k"], 200)
        self.assertEqual(options["repeat_penalty"], 0.8)
        self.assertEqual(options["seed"], 42)
        self.assertEqual(options["num_ctx"], 262144)
        self.assertEqual(options["num_predict"], 8192)

    def test_planner_llm_default_timeout_allows_large_local_album_calls(self):
        settings = local_llm.planner_llm_settings_from_payload({})
        self.assertEqual(settings["planner_timeout"], 604800.0)

    def test_planner_llm_settings_map_to_lmstudio_without_context_option(self):
        options = local_llm.planner_llm_options_for_provider(
            "lmstudio",
            {
                "planner_temperature": 0.7,
                "planner_top_p": 0.94,
                "planner_top_k": 64,
                "planner_repeat_penalty": 1.2,
                "planner_seed": "123",
                "planner_max_tokens": 3072,
                "planner_context_length": 16384,
                "planner_timeout": 120,
            },
        )

        self.assertEqual(options["temperature"], 0.7)
        self.assertEqual(options["top_p"], 0.94)
        self.assertEqual(options["top_k"], 64)
        self.assertEqual(options["repeat_penalty"], 1.2)
        self.assertEqual(options["seed"], 123)
        self.assertEqual(options["max_tokens"], 3072)
        self.assertEqual(options["timeout"], 120.0)
        self.assertNotIn("num_ctx", options)
        self.assertNotIn("num_predict", options)

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

    def test_structured_outputs_are_sent_to_ollama_and_lmstudio(self):
        schema = {"type": "object", "properties": {"payload": {"type": "object"}}, "required": ["payload"]}
        seen = []

        def fake_http(method, url, payload=None, timeout=30.0):
            seen.append((url, payload))
            if "/api/chat" in url:
                return {"message": {"content": "{\"payload\": {}}"}, "done_reason": "stop"}
            return {"choices": [{"message": {"content": "{\"payload\": {}}"}, "finish_reason": "stop"}]}

        with patch.object(local_llm, "_http_json", side_effect=fake_http), \
            patch.object(local_llm, "resolve_model", side_effect=lambda provider, model, kind="chat": model):
            ollama = local_llm.chat_completion_response(
                "ollama",
                "qwen",
                [{"role": "user", "content": "json"}],
                json_schema=schema,
            )
            lmstudio = local_llm.chat_completion_response(
                "lmstudio",
                "qwen-local",
                [{"role": "user", "content": "json"}],
                json_schema=schema,
            )

        self.assertFalse(ollama["truncated"])
        self.assertFalse(lmstudio["truncated"])
        self.assertEqual(seen[0][1]["format"], schema)
        self.assertEqual(seen[1][1]["response_format"]["type"], "json_schema")
        self.assertEqual(seen[1][1]["response_format"]["json_schema"]["schema"], schema)

    def test_local_llm_settings_and_catalog_endpoints_ignore_image_generation_settings(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "local_llm_settings.json"
            with patch.object(acejam_app, "LOCAL_LLM_SETTINGS_PATH", settings_path), \
                patch.object(acejam_app, "_ollama_client", return_value=FakeOllamaClient()), \
                patch.object(acejam_app, "lmstudio_model_catalog", return_value={"ready": True, "provider": "lmstudio", "host": "http://localhost:1234", "models": ["mlx-chat"], "chat_models": ["mlx-chat"], "embedding_models": [], "image_models": [], "details": [{"name": "mlx-chat", "provider": "lmstudio", "kind": "chat", "format": "mlx"}], "loaded_models": []}):
                saved = client.post(
                    "/api/local-llm/settings",
                    json={
                        "provider": "lmstudio",
                        "chat_model": "mlx-chat",
                        "embedding_provider": "ollama",
                        "embedding_model": "nomic-embed-text:latest",
                        "art_model": "x/flux2-klein:4b",
                        "planner_max_tokens": 99999,
                    },
                )
                catalog = client.get("/api/local-llm/catalog")

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["settings"]["planner_max_tokens"], 8192)
        self.assertEqual(saved.json()["settings"]["art_model"], "")
        self.assertFalse(saved.json()["settings"]["auto_single_art"])
        self.assertFalse(saved.json()["settings"]["auto_album_art"])
        data = catalog.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["image_models"], [])
        self.assertEqual(data["settings"]["provider"], "lmstudio")

    def test_art_generation_endpoint_is_disabled(self):
        client = TestClient(acejam_app.app)
        response = client.post(
            "/api/art/generate",
            json={"scope": "single", "model": "x/flux2-klein:4b", "prompt": "cover art"},
        )

        self.assertEqual(response.status_code, 410)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("disabled", data["error"].lower())

    def test_default_local_llm_settings_disable_art_generation(self):
        with patch.object(acejam_app, "_ollama_client", return_value=FakeOllamaClientNoImage()):
            settings = acejam_app._local_llm_default_settings()

        self.assertEqual(settings["art_model"], "")
        self.assertFalse(settings["auto_single_art"])
        self.assertFalse(settings["auto_album_art"])


if __name__ == "__main__":
    unittest.main()
