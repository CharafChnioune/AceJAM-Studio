import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import mflux_manager


class MfluxManagerTests(unittest.TestCase):
    def test_model_catalog_exposes_required_presets(self):
        catalog = mflux_manager.mflux_models()
        ids = {item["id"] for item in catalog["models"]}

        self.assertIn("qwen-image", ids)
        self.assertIn("flux2-klein-9b", ids)
        self.assertIn("z-image-turbo", ids)
        self.assertIn("z-image", ids)
        self.assertIn("seedvr2", ids)
        self.assertEqual(catalog["defaults"]["generate"], "qwen-image")
        self.assertEqual(catalog["defaults"]["train_lora"], "flux2-klein-9b")
        commands = {item["id"]: item["command"] for item in catalog["models"]}
        self.assertEqual(commands["flux2-klein-9b"], "mflux-generate-flux2")
        self.assertEqual(commands["z-image-turbo"], "mflux-generate-z-image-turbo")
        self.assertEqual(commands["seedvr2"], "mflux-upscale-seedvr2")
        self.assertEqual(commands["depth-pro"], "mflux-save-depth")

    def test_status_blocks_non_apple_mlx(self):
        with patch.object(mflux_manager.sys, "platform", "linux"), \
             patch.object(mflux_manager.platform, "machine", return_value="x86_64"), \
             patch.object(mflux_manager.importlib.util, "find_spec", return_value=True):
            status = mflux_manager.mflux_status()

        self.assertFalse(status["ready"])
        self.assertIn("Apple Silicon", status["blocking_reason"])

    def test_status_reports_action_command_readiness(self):
        def fake_which(command):
            return "/bin/echo" if command in {"mflux-generate-qwen", "mflux-train"} else None

        with patch.object(mflux_manager.sys, "platform", "darwin"), \
             patch.object(mflux_manager.platform, "machine", return_value="arm64"), \
             patch.object(mflux_manager, "MFLUX_ENV_DIR", Path("/tmp/acejam-no-mflux-env")), \
             patch.object(mflux_manager.importlib.util, "find_spec", return_value=True), \
             patch.object(mflux_manager.shutil, "which", side_effect=fake_which):
            status = mflux_manager.mflux_status()

        self.assertTrue(status["action_readiness"]["generate"]["ready"])
        self.assertIn("mflux-generate-flux2-edit", status["action_readiness"]["edit"]["missing_commands"])
        self.assertIn("mflux-upscale-seedvr2", status["action_readiness"]["upscale"]["missing_commands"])

    def test_action_command_builder_uses_uploads_and_multi_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            results = root / "results"
            uploads.mkdir()
            results.mkdir()
            source = uploads / "source.png"
            source.write_bytes(b"png")
            lora = root / "adapter.safetensors"
            lora.write_bytes(b"lora")

            with patch.object(mflux_manager, "DATA_DIR", root), \
                 patch.object(mflux_manager, "MFLUX_UPLOADS_DIR", uploads), \
                 patch.object(mflux_manager, "MFLUX_RESULTS_DIR", results), \
                 patch.object(mflux_manager, "MFLUX_ENV_DIR", root / "no-mflux-env"), \
                 patch.object(mflux_manager.shutil, "which", lambda command: f"/usr/bin/{command}"):
                command = mflux_manager._build_mflux_command(  # noqa: SLF001
                    {
                        "action": "edit",
                        "prompt": "turn it into premium cover art",
                        "model_id": "flux2-klein-9b",
                        "image_path": str(source),
                        "lora_adapters": [{"path": str(lora), "scale": 0.5, "model_id": "flux2-klein-9b"}],
                    },
                    results / "out.png",
                )

        self.assertEqual(command[0], "/usr/bin/mflux-generate-flux2-edit")
        self.assertIn("--image-path", command)
        self.assertIn("--lora-paths", command)
        self.assertIn("--lora-scales", command)

    def test_inpaint_requires_mask_and_lora_family_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            uploads.mkdir()
            source = uploads / "source.png"
            source.write_bytes(b"png")
            lora = root / "zstyle.safetensors"
            lora.write_bytes(b"lora")

            with patch.object(mflux_manager, "DATA_DIR", root), \
                 patch.object(mflux_manager, "MFLUX_UPLOADS_DIR", uploads), \
                 patch.object(mflux_manager, "MFLUX_ENV_DIR", root / "no-mflux-env"), \
                 patch.object(mflux_manager.shutil, "which", lambda command: f"/usr/bin/{command}"):
                with self.assertRaisesRegex(RuntimeError, "required"):
                    mflux_manager._build_mflux_command(  # noqa: SLF001
                        {"action": "inpaint", "prompt": "fill", "model_id": "flux2-klein-9b", "image_path": str(source)},
                        root / "out.png",
                    )
                with self.assertRaisesRegex(RuntimeError, "not flux2"):
                    mflux_manager._build_mflux_command(  # noqa: SLF001
                        {
                            "action": "generate",
                            "prompt": "cover",
                            "model_id": "flux2-klein-9b",
                            "lora_adapters": [{"path": str(lora), "model_id": "z-image"}],
                        },
                        root / "out.png",
                    )

    def test_create_job_runs_mocked_runner_and_persists_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "jobs"
            results_dir = Path(tmp) / "results"
            jobs_dir.mkdir()
            results_dir.mkdir()

            def fake_runner(job_id, payload):
                return {
                    "result_id": "mock-result",
                    "image_url": "/media/mflux/mock-result/image.png",
                    "model_id": payload["model_id"],
                    "action": payload["action"],
                    "logs": ["mock ok"],
                }

            ready = {
                "apple_silicon": True,
                "mlx_available": True,
                "mflux_available": True,
                "cli_available": False,
                "blocking_reason": "",
            }
            with patch.object(mflux_manager, "MFLUX_JOBS_DIR", jobs_dir), \
                 patch.object(mflux_manager, "MFLUX_RESULTS_DIR", results_dir), \
                 patch.object(mflux_manager, "mflux_status", return_value=ready):
                job = mflux_manager.mflux_create_job({"prompt": "cover", "model_id": "qwen-image"}, runner=fake_runner)
                for _ in range(50):
                    snapshot = mflux_manager.mflux_get_job(job["id"])
                    if snapshot and snapshot.get("state") == "succeeded":
                        break
                    time.sleep(0.02)
                snapshot = mflux_manager.mflux_get_job(job["id"])

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertEqual(snapshot["result_summary"]["image_url"], "/media/mflux/mock-result/image.png")
        self.assertEqual(snapshot["result"]["model_id"], "qwen-image")

    def test_image_lora_registry_reads_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "album-style"
            adapter.mkdir()
            (adapter / "adapter.safetensors").write_bytes(b"mock")
            (adapter / "mflux_adapter.json").write_text(
                json.dumps({"display_name": "Album Style", "trigger_tag": "albumstyle", "model_id": "flux2-klein-9b"}),
                encoding="utf-8",
            )

            with patch.object(mflux_manager, "MFLUX_LORAS_DIR", root):
                adapters = mflux_manager.mflux_list_lora_adapters()

        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0]["display_name"], "Album Style")
        self.assertEqual(adapters[0]["model_id"], "flux2-klein-9b")
        self.assertEqual(adapters[0]["family"], "flux2")
        self.assertTrue(adapters[0]["generation_loadable"])

    def test_training_dataset_summary_validates_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.png").write_bytes(b"png")
            (root / "one.txt").write_text("trigger, album art", encoding="utf-8")
            (root / "two.png").write_bytes(b"png")

            summary = mflux_manager.mflux_summarize_training_dataset(str(root), "txt2img")

        self.assertEqual(summary["image_count"], 2)
        self.assertEqual(summary["caption_count"], 1)
        self.assertEqual(summary["missing_caption_count"], 1)


if __name__ == "__main__":
    unittest.main()
