import json
import sys
import tempfile
import threading
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import official_runner


class OfficialRunnerTest(unittest.TestCase):
    def _fake_generate_music_execute_module(self):
        module_names = [
            "acestep.core",
            "acestep.core.generation",
            "acestep.core.generation.handler",
            "acestep.core.generation.handler.generate_music_execute",
        ]
        saved = {name: sys.modules.get(name) for name in module_names}
        for name in module_names[:-1]:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        leaf = types.ModuleType(module_names[-1])

        class GenerateMusicExecuteMixin:
            def _run_generate_music_service_with_progress(
                self,
                progress,
                actual_batch_size,
                audio_duration,
                inference_steps,
                timesteps,
                service_inputs,
                refer_audios,
                guidance_scale,
                actual_seed_list,
                audio_cover_strength,
                cover_noise_strength,
                use_adg,
                cfg_interval_start,
                cfg_interval_end,
                shift,
                infer_method,
                sampler_mode="euler",
                velocity_norm_threshold=0.0,
                velocity_ema_factor=0.0,
                repaint_crossfade_frames=10,
                repaint_injection_ratio=0.5,
            ):
                raise AssertionError("original threaded method should not run for active MLX")

        leaf.GenerateMusicExecuteMixin = GenerateMusicExecuteMixin
        sys.modules[module_names[-1]] = leaf
        return GenerateMusicExecuteMixin, saved

    def _restore_modules(self, saved):
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

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

    def test_resolve_backend_preserves_mlx_on_apple_silicon(self):
        with patch.object(official_runner.sys, "platform", "darwin"), \
            patch.object(official_runner.platform, "machine", return_value="arm64"):
            self.assertEqual(official_runner._resolve_backend("mlx"), "mlx")
            self.assertEqual(official_runner._resolve_backend("auto"), "mlx")

    def test_resolve_backend_downgrades_mlx_off_apple_silicon(self):
        with patch.object(official_runner.sys, "platform", "linux"), \
            patch.object(official_runner.platform, "machine", return_value="x86_64"):
            self.assertEqual(official_runner._resolve_backend("mlx"), "pt")
            self.assertEqual(official_runner._resolve_backend("auto"), "pt")

    def test_mlx_progress_patch_runs_service_generate_inline(self):
        GenerateMusicExecuteMixin, saved = self._fake_generate_music_execute_module()
        official_runner._patch_mlx_thread_stream(object)
        try:
            calls = []

            class DummyProgressThread:
                def join(self, timeout=None):
                    calls.append(("progress_join", timeout))

            class DummyModel:
                def generate_audio(self, **kwargs):
                    raise AssertionError("PyTorch fallback should not run")

            class Dummy(GenerateMusicExecuteMixin):
                use_mlx_dit = True
                mlx_decoder = object()

                def __init__(self):
                    self.model = DummyModel()

                def _start_diffusion_progress_estimator(self, **kwargs):
                    calls.append(("progress", kwargs["infer_steps"]))
                    return threading.Event(), DummyProgressThread()

                def service_generate(self, **kwargs):
                    calls.append(("service", threading.current_thread().name, kwargs["captions"]))
                    return {"target_latents": "ok"}

            def progress(*args, **kwargs):
                calls.append(("progress_callback", args, kwargs))

            result = Dummy()._run_generate_music_service_with_progress(
                progress=progress,
                actual_batch_size=1,
                audio_duration=30,
                inference_steps=8,
                timesteps=None,
                service_inputs={
                    "captions_batch": ["bright schlager"],
                    "global_captions_batch": None,
                    "lyrics_batch": ["[Verse]\nHallo"],
                    "metas_batch": [{}],
                    "vocal_languages_batch": ["de"],
                    "target_wavs_tensor": None,
                    "repainting_start_batch": [0],
                    "repainting_end_batch": [-1],
                    "instructions_batch": [""],
                    "audio_code_hints_batch": [""],
                    "should_return_intermediate": False,
                    "chunk_mask_modes_batch": ["auto"],
                },
                refer_audios=None,
                guidance_scale=8.0,
                actual_seed_list=[123],
                audio_cover_strength=1.0,
                cover_noise_strength=0.0,
                use_adg=False,
                cfg_interval_start=0.0,
                cfg_interval_end=1.0,
                shift=1.0,
                infer_method="ode",
            )

            self.assertEqual(result["outputs"], {"target_latents": "ok"})
            self.assertIn(("service", threading.current_thread().name, ["bright schlager"]), calls)
            self.assertIn(("progress", 8), calls)
        finally:
            self._restore_modules(saved)

    def test_mlx_progress_patch_blocks_pytorch_fallback(self):
        GenerateMusicExecuteMixin, saved = self._fake_generate_music_execute_module()
        official_runner._patch_mlx_thread_stream(object)
        try:
            class DummyModel:
                def generate_audio(self, **kwargs):
                    return {"unexpected": True}

            class Dummy(GenerateMusicExecuteMixin):
                use_mlx_dit = True
                mlx_decoder = object()

                def __init__(self):
                    self.model = DummyModel()

                def service_generate(self, **kwargs):
                    return self.model.generate_audio()

            with self.assertRaisesRegex(RuntimeError, "PyTorch/MPS fallback"):
                Dummy()._run_generate_music_service_with_progress(
                    progress=lambda *args, **kwargs: None,
                    actual_batch_size=1,
                    audio_duration=30,
                    inference_steps=8,
                    timesteps=None,
                    service_inputs={
                        "captions_batch": ["bright schlager"],
                        "lyrics_batch": ["[Verse]\nHallo"],
                        "metas_batch": [{}],
                        "vocal_languages_batch": ["de"],
                        "target_wavs_tensor": None,
                        "repainting_start_batch": [0],
                        "repainting_end_batch": [-1],
                        "instructions_batch": [""],
                        "audio_code_hints_batch": [""],
                        "should_return_intermediate": False,
                    },
                    refer_audios=None,
                    guidance_scale=8.0,
                    actual_seed_list=[123],
                    audio_cover_strength=1.0,
                    cover_noise_strength=0.0,
                    use_adg=False,
                    cfg_interval_start=0.0,
                    cfg_interval_end=1.0,
                    shift=1.0,
                    infer_method="ode",
                )
        finally:
            self._restore_modules(saved)

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
