import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import official_runner


class OfficialRunnerTest(unittest.TestCase):
    def test_parse_seeds(self):
        self.assertEqual(official_runner._parse_seeds("1, 2,3"), [1, 2, 3])
        self.assertIsNone(official_runner._parse_seeds(""))
        self.assertIsNone(official_runner._parse_seeds("-1"))

    def test_filter_generation_params_drops_lm_only_fields(self):
        class FakeGenerationParams:
            __dataclass_fields__ = {"caption": object(), "lyrics": object(), "thinking": object()}

        filtered = official_runner._filter_generation_params(
            {
                "caption": "cinematic rap",
                "lyrics": "[Verse]\nline",
                "thinking": True,
                "repetition_penalty": 1.0,
                "sample_query": "write it",
            },
            FakeGenerationParams,
        )

        self.assertEqual(filtered, {"caption": "cinematic rap", "lyrics": "[Verse]\nline", "thinking": True})

    def test_call_compat_drops_unknown_initialize_kwargs(self):
        calls = {}

        def initialize_service(project_root, config_path, device="auto"):
            calls.update({"project_root": project_root, "config_path": config_path, "device": device})
            return "ready", True

        status, ready = official_runner._call_compat(
            initialize_service,
            project_root="/tmp/project",
            config_path="acestep-v15-turbo",
            device="auto",
            use_mlx_dit=False,
        )

        self.assertEqual((status, ready), ("ready", True))
        self.assertEqual(calls, {"project_root": "/tmp/project", "config_path": "acestep-v15-turbo", "device": "auto"})

    def test_create_sample_action_uses_lm_without_dit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor"
            package = vendor / "acestep"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "handler.py").write_text(
                "class AceStepHandler:\n"
                "    def __init__(self):\n"
                "        raise AssertionError('DiT should not initialize for create_sample')\n",
                encoding="utf-8",
            )
            (package / "llm_inference.py").write_text(
                "class LLMHandler:\n"
                "    def __init__(self, persistent_storage_path=None):\n"
                "        self.persistent_storage_path = persistent_storage_path\n"
                "    def initialize(self, **kwargs):\n"
                "        return ('ready', True)\n",
                encoding="utf-8",
            )
            (package / "inference.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass, asdict

                    class GenerationConfig:
                        def __init__(self, **kwargs):
                            self.kwargs = kwargs

                    class GenerationParams:
                        def __init__(self, **kwargs):
                            self.kwargs = kwargs

                    def generate_music(**kwargs):
                        raise AssertionError("generate_music should not run")

                    @dataclass
                    class Result:
                        caption: str = "bright pop"
                        lyrics: str = "[Verse]\\nhello"
                        bpm: int = 120
                        duration: float = 60.0
                        keyscale: str = "C major"
                        language: str = "en"
                        timesignature: str = "4"
                        instrumental: bool = False
                        status_message: str = "ok"
                        success: bool = True
                        error: str | None = None
                        def to_dict(self):
                            return asdict(self)

                    def create_sample(**kwargs):
                        return Result()

                    def format_sample(**kwargs):
                        return Result()

                    def understand_music(**kwargs):
                        return Result()
                    """
                ),
                encoding="utf-8",
            )
            request = {
                "action": "create_sample",
                "vendor_dir": str(vendor),
                "model_cache_dir": str(root / "cache"),
                "checkpoint_dir": str(root / "checkpoints"),
                "lm_model": "acestep-5Hz-lm-1.7B",
                "params": {"sample_query": "make a hook", "vocal_language": "en"},
            }
            request_path = root / "request.json"
            response_path = root / "response.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            saved_modules = {
                name: module
                for name, module in list(sys.modules.items())
                if name == "acestep" or name.startswith("acestep.")
            }
            saved_path = list(sys.path)
            for name in saved_modules:
                sys.modules.pop(name, None)
            try:
                official_runner._run(request_path, response_path)
            finally:
                sys.path[:] = saved_path
                for name in list(sys.modules):
                    if name == "acestep" or name.startswith("acestep."):
                        sys.modules.pop(name, None)
                sys.modules.update(saved_modules)

            data = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertTrue(data["success"])
            self.assertEqual(data["caption"], "bright pop")


if __name__ == "__main__":
    unittest.main()
