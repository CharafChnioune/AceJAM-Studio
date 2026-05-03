import json
import tempfile
import unittest
from pathlib import Path

import download_models


class DownloadModelsTest(unittest.TestCase):
    def test_all_download_list_covers_supported_models_and_shared_components(self):
        models = download_models.default_download_models()

        self.assertIn("acestep-v15-turbo", models)
        self.assertIn("acestep-v15-xl-sft", models)
        self.assertIn("acestep-5Hz-lm-4B", models)
        self.assertIn("main", models)
        self.assertIn("acestep-captioner", models)
        self.assertIn("acestep-transcriber", models)
        self.assertIn("ace-step-v1.5-1d-vae-stable-audio-format", models)
        self.assertIn("vae", models)
        self.assertIn("Qwen3-Embedding-0.6B", models)
        self.assertNotIn("auto", models)
        self.assertNotIn("none", models)
        self.assertNotIn("acestep-v15-turbo-rl", models)

    def test_diffusers_export_readiness_accepts_model_index_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "acestep-v15-xl-turbo-diffusers"
            path.mkdir()
            (path / "model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "AceStepPipeline",
                        "condition_encoder": ["ace_step", "AceStepConditionEncoder"],
                        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                        "text_encoder": ["transformers", "Qwen3Model"],
                        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                        "transformer": ["diffusers", "AceStepTransformer1DModel"],
                        "vae": ["diffusers", "AutoencoderOobleck"],
                    }
                ),
                encoding="utf-8",
            )
            for component in ["condition_encoder", "text_encoder", "transformer", "vae"]:
                component_path = path / component
                component_path.mkdir()
                (component_path / "config.json").write_text("{}", encoding="utf-8")
                (component_path / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            (path / "scheduler").mkdir()
            (path / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer").mkdir()
            (path / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

            self.assertTrue(download_models.checkpoint_dir_ready(path))


if __name__ == "__main__":
    unittest.main()
