import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx_video_manager


class MlxVideoManagerTests(unittest.TestCase):
    def test_catalog_exposes_draft_first_presets(self):
        catalog = mlx_video_manager.mlx_video_models()
        ids = {item["id"] for item in catalog["models"]}

        self.assertIn("ltx2-fast-draft", ids)
        self.assertIn("ltx23-fast-draft", ids)
        self.assertIn("ltx2-final-hq", ids)
        self.assertIn("ltx23-final-hq", ids)
        self.assertIn("wan21-reality-480p", ids)
        self.assertIn("wan22-lightning-draft", ids)
        self.assertEqual(catalog["defaults"]["t2v"], "ltx23-fast-draft")
        self.assertEqual(catalog["defaults"]["final"], "ltx23-final-hq")
        self.assertIn("song_video", catalog["by_action"])

    def test_status_blocks_non_apple_mlx(self):
        with patch.object(mlx_video_manager.sys, "platform", "linux"), \
             patch.object(mlx_video_manager.platform, "machine", return_value="x86_64"):
            status = mlx_video_manager.mlx_video_status(check_help=False)

        self.assertFalse(status["ready"])
        self.assertIn("Apple Silicon", status["blocking_reason"])

    def test_command_registry_uses_console_and_python_m_fallback(self):
        with patch.object(mlx_video_manager, "_command_candidate", side_effect=lambda name: f"/env/bin/{name}" if "ltx" in name else None), \
             patch.object(mlx_video_manager, "_env_python", return_value=Path("/env/bin/python")), \
             patch.object(Path, "is_file", return_value=True):
            self.assertEqual(mlx_video_manager._engine_command("ltx")[0], "/env/bin/mlx_video.ltx_2.generate")  # noqa: SLF001
            self.assertEqual(
                mlx_video_manager._engine_command("wan"),  # noqa: SLF001
                ["/env/bin/python", "-m", "mlx_video.models.wan_2.generate"],
            )

    def test_help_status_parses_cli_capabilities(self):
        help_text = "--output-path --end-image --enhance-prompt --spatial-upscaler --tiling --audio-file --lora-path"
        with patch.object(mlx_video_manager, "_engine_command", return_value=["mlx_video.ltx_2.generate"]), \
             patch.object(mlx_video_manager, "_run_probe", return_value=(True, help_text)):
            status = mlx_video_manager._help_status("ltx")  # noqa: SLF001

        self.assertTrue(status["help_ok"])
        self.assertEqual(status["output_flag"], "--output-path")
        self.assertTrue(status["capabilities"]["end_image"])
        self.assertTrue(status["capabilities"]["spatial_upscaler"])

    def test_ltx_command_validates_dimensions_frames_sources_and_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            uploads.mkdir()
            source = uploads / "source.png"
            source.write_bytes(b"png")
            audio = uploads / "song.wav"
            audio.write_bytes(b"wav")
            lora = root / "ltx-style.safetensors"
            lora.write_bytes(b"lora")

            with patch.object(mlx_video_manager, "DATA_DIR", root), \
                 patch.object(mlx_video_manager, "MLX_VIDEO_UPLOADS_DIR", uploads), \
                 patch.object(mlx_video_manager, "_engine_command", return_value=["mlx_video.ltx_2.generate"]):
                command, model, warnings = mlx_video_manager._build_mlx_video_command(  # noqa: SLF001
                    {
                        "action": "song_video",
                        "prompt": "cinematic music video",
                        "model_id": "ltx2-fast-draft",
                        "width": 513,
                        "height": 321,
                        "num_frames": 34,
                        "image_path": str(source),
                        "audio_path": str(audio),
                        "lora_adapters": [{"path": str(lora), "scale": 0.65}],
                    },
                    root / "out.mp4",
                )

        self.assertEqual(model["id"], "ltx2-fast-draft")
        self.assertIn("--width", command)
        self.assertEqual(command[command.index("--width") + 1], "512")
        self.assertEqual(command[command.index("--height") + 1], "320")
        self.assertEqual(command[command.index("--num-frames") + 1], "33")
        self.assertIn("--audio-file", command)
        self.assertIn("--lora-path", command)
        self.assertEqual(warnings, [])

    def test_ltx23_command_passes_string_tiling_and_real_upscaler_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            uploads.mkdir()
            source = uploads / "source.png"
            source.write_bytes(b"png")

            caps = {
                "output_path": True,
                "image": True,
                "audio_file": True,
                "enhance_prompt": True,
                "spatial_upscaler": True,
                "tiling": True,
                "ltx_lora": True,
                "end_image": True,
            }
            with patch.object(mlx_video_manager, "DATA_DIR", root), \
                 patch.object(mlx_video_manager, "MLX_VIDEO_UPLOADS_DIR", uploads), \
                 patch.object(mlx_video_manager, "_engine_command", return_value=["mlx_video.ltx_2.generate"]), \
                 patch.object(mlx_video_manager, "_engine_capabilities", return_value=caps):
                command, model, warnings = mlx_video_manager._build_mlx_video_command(  # noqa: SLF001
                    {
                        "action": "i2v",
                        "prompt": "camera moves gently",
                        "model_id": "ltx23-fast-draft",
                        "image_path": str(source),
                        "tiling": "aggressive",
                        "spatial_upscaler": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
                    },
                    root / "out.mp4",
                )

        self.assertEqual(model["id"], "ltx23-fast-draft")
        self.assertIn("--tiling", command)
        self.assertEqual(command[command.index("--tiling") + 1], "aggressive")
        self.assertIn("--spatial-upscaler", command)
        self.assertEqual(
            command[command.index("--spatial-upscaler") + 1],
            "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
        )
        self.assertEqual(warnings, [])

    def test_ltx_end_frame_requires_pr23_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            uploads.mkdir()
            source = uploads / "source.png"
            source.write_bytes(b"png")
            end = uploads / "end.png"
            end.write_bytes(b"png")

            caps = {
                "output_path": True,
                "image": True,
                "audio_file": True,
                "enhance_prompt": True,
                "spatial_upscaler": True,
                "tiling": True,
                "ltx_lora": True,
                "end_image": True,
            }
            with patch.object(mlx_video_manager, "DATA_DIR", root), \
                 patch.object(mlx_video_manager, "MLX_VIDEO_UPLOADS_DIR", uploads), \
                 patch.object(mlx_video_manager, "_engine_command", return_value=["mlx_video.ltx_2.generate"]), \
                 patch.object(mlx_video_manager, "_engine_capabilities", return_value=caps):
                command, _model, _warnings = mlx_video_manager._build_mlx_video_command(  # noqa: SLF001
                    {
                        "action": "i2v",
                        "prompt": "camera moves from first to last frame",
                        "model_id": "ltx2-fast-draft",
                        "image_path": str(source),
                        "end_image_path": str(end),
                        "enhance_prompt": True,
                        "tiling": True,
                    },
                    root / "out.mp4",
                )

            self.assertIn("--end-image", command)
            self.assertIn("--enhance-prompt", command)
            self.assertIn("--tiling", command)

            caps["end_image"] = False
            with patch.object(mlx_video_manager, "DATA_DIR", root), \
                 patch.object(mlx_video_manager, "MLX_VIDEO_UPLOADS_DIR", uploads), \
                 patch.object(mlx_video_manager, "_engine_command", return_value=["mlx_video.ltx_2.generate"]), \
                 patch.object(mlx_video_manager, "_engine_capabilities", return_value=caps):
                with self.assertRaisesRegex(RuntimeError, "end-frame"):
                    mlx_video_manager._build_mlx_video_command(  # noqa: SLF001
                        {
                            "action": "i2v",
                            "prompt": "blocked",
                            "model_id": "ltx2-fast-draft",
                            "image_path": str(source),
                            "end_image_path": str(end),
                        },
                        root / "out.mp4",
                    )

    def test_wan_command_requires_model_dir_and_preserves_480p_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "Wan2.1-T2V-MLX"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            lora = root / "high.safetensors"
            lora.write_bytes(b"lora")

            with patch.object(mlx_video_manager, "_engine_command", return_value=["mlx_video.wan_2.generate"]):
                command, model, _warnings = mlx_video_manager._build_mlx_video_command(  # noqa: SLF001
                    {
                        "action": "t2v",
                        "prompt": "real life street scene",
                        "model_id": "wan21-reality-480p",
                        "model_dir": str(model_dir),
                        "width": 832,
                        "height": 480,
                        "num_frames": 82,
                        "tiling": "conservative",
                        "lora_adapters": [{"path": str(lora), "scale": 1.0, "role": "high"}],
                    },
                    root / "out.mp4",
                )

        self.assertEqual(model["id"], "wan21-reality-480p")
        self.assertEqual(command[command.index("--width") + 1], "832")
        self.assertEqual(command[command.index("--height") + 1], "480")
        self.assertEqual(command[command.index("--num-frames") + 1], "81")
        self.assertIn("--tiling", command)
        self.assertEqual(command[command.index("--tiling") + 1], "conservative")
        self.assertIn("--lora-high", command)

    def test_wan_command_prefers_no_negative_prompt_and_supports_lightning_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "Wan2.2-T2V-MLX"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model":"wan2.2"}', encoding="utf-8")
            high = root / "high.safetensors"
            low = root / "low.safetensors"
            high.write_bytes(b"high")
            low.write_bytes(b"low")

            with patch.object(mlx_video_manager, "_engine_command", return_value=["mlx_video.wan_2.generate"]):
                command, model, _warnings = mlx_video_manager._build_mlx_video_command(  # noqa: SLF001
                    {
                        "action": "t2v",
                        "prompt": "lightning clip",
                        "model_id": "wan22-lightning-draft",
                        "model_dir": str(model_dir),
                        "negative_prompt": "blurry, low quality",
                        "no_negative_prompt": True,
                        "trim_first_frames": 2,
                        "lora_adapters": [
                            {"path": str(high), "scale": 1.0, "role": "high"},
                            {"path": str(low), "scale": 0.9, "role": "low"},
                        ],
                    },
                    root / "out.mp4",
                )

        self.assertEqual(model["id"], "wan22-lightning-draft")
        self.assertIn("--no-negative-prompt", command)
        self.assertNotIn("--negative-prompt", command)
        self.assertIn("--trim-first-frames", command)
        self.assertEqual(command[command.index("--trim-first-frames") + 1], "2")
        self.assertIn("--lora-high", command)
        self.assertIn("--lora-low", command)

    def test_tokenizer_issue_26_guard_blocks_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "LTX-2.3"):
            mlx_video_manager._build_mlx_video_command(  # noqa: SLF001
                {
                    "prompt": "video",
                    "model_id": "ltx2-fast-draft",
                    "model_repo": "Lightricks/LTX-2.3",
                    "text_encoder_repo": "Lightricks/LTX-2",
                },
                Path(tempfile.gettempdir()) / "out.mp4",
            )

    def test_create_job_runs_mocked_runner_and_persists_mp4_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "jobs"
            results_dir = Path(tmp) / "results"
            jobs_dir.mkdir()
            results_dir.mkdir()

            def fake_runner(job_id, payload):
                return {
                    "result_id": "mock-video",
                    "video_url": "/media/mlx-video/mock-video/out.mp4",
                    "poster_url": "/media/mlx-video/mock-video/out.jpg",
                    "model_id": payload["model_id"],
                    "action": payload["action"],
                    "logs": ["mock ok"],
                }

            with patch.object(mlx_video_manager, "MLX_VIDEO_JOBS_DIR", jobs_dir), \
                 patch.object(mlx_video_manager, "MLX_VIDEO_RESULTS_DIR", results_dir):
                job = mlx_video_manager.mlx_video_create_job(
                    {"prompt": "video", "model_id": "ltx2-fast-draft"},
                    runner=fake_runner,
                )
                for _ in range(50):
                    snapshot = mlx_video_manager.mlx_video_get_job(job["id"])
                    if snapshot and snapshot.get("state") == "succeeded":
                        break
                    time.sleep(0.02)
                snapshot = mlx_video_manager.mlx_video_get_job(job["id"])

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertEqual(snapshot["result_summary"]["video_url"], "/media/mlx-video/mock-video/out.mp4")
        self.assertEqual(snapshot["result"]["model_id"], "ltx2-fast-draft")

    def test_song_video_muxes_source_audio_and_prefers_muxed_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            raw = results_dir / "clip.mp4"
            audio = results_dir / "song.wav"
            raw.write_bytes(b"video")
            audio.write_bytes(b"audio")
            seen = {}

            def fake_run(command, **_kwargs):
                seen["command"] = command
                Path(command[-1]).write_bytes(b"muxed")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(mlx_video_manager.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch.object(mlx_video_manager.subprocess, "run", side_effect=fake_run):
                muxed = mlx_video_manager._mux_source_audio(raw, audio, "clip-result")  # noqa: SLF001
                self.assertTrue(Path(muxed["path"]).is_file())

            command = seen["command"]
        self.assertEqual(command[0], "/usr/bin/ffmpeg")
        self.assertIn("-map", command)
        self.assertIn("0:v:0", command)
        self.assertIn("1:a:0", command)
        self.assertIn("-shortest", command)
        self.assertTrue(str(muxed["url"]).endswith("-source-audio.mp4"))

    def test_create_song_video_job_defaults_to_replace_source_audio_policy(self):
        def fake_runner(_job_id, payload):
            return {
                "result_id": "mock-video",
                "raw_video_url": "/media/mlx-video/mock-video/raw.mp4",
                "muxed_video_url": "/media/mlx-video/mock-video/muxed.mp4",
                "primary_video_url": "/media/mlx-video/mock-video/muxed.mp4",
                "video_url": "/media/mlx-video/mock-video/muxed.mp4",
                "poster_url": "/media/mlx-video/mock-video/out.jpg",
                "model_id": payload["model_id"],
                "action": payload["action"],
                "audio_policy": payload["audio_policy"],
                "mux_audio": payload["mux_audio"],
                "postprocess_status": "muxed",
                "logs": ["mock ok"],
            }

        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "jobs"
            results_dir = Path(tmp) / "results"
            jobs_dir.mkdir()
            results_dir.mkdir()
            with patch.object(mlx_video_manager, "MLX_VIDEO_JOBS_DIR", jobs_dir), \
                 patch.object(mlx_video_manager, "MLX_VIDEO_RESULTS_DIR", results_dir):
                job = mlx_video_manager.mlx_video_create_job(
                    {"action": "song_video", "prompt": "music video", "audio_path": "/tmp/source.wav"},
                    runner=fake_runner,
                )
                for _ in range(50):
                    snapshot = mlx_video_manager.mlx_video_get_job(job["id"])
                    if snapshot and snapshot.get("state") == "succeeded":
                        break
                    time.sleep(0.02)
                snapshot = mlx_video_manager.mlx_video_get_job(job["id"])

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["payload"]["audio_policy"], "replace_with_source")
        self.assertTrue(snapshot["payload"]["mux_audio"])
        self.assertEqual(snapshot["result_summary"]["video_url"], "/media/mlx-video/mock-video/muxed.mp4")
        self.assertEqual(snapshot["result_summary"]["raw_video_url"], "/media/mlx-video/mock-video/raw.mp4")
        self.assertEqual(snapshot["result_summary"]["postprocess_status"], "muxed")

    def test_video_lora_registry_reads_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "wan-lightning"
            adapter.mkdir()
            (adapter / "high_noise_model.safetensors").write_bytes(b"mock")
            (adapter / "mlx_video_lora.json").write_text(
                json.dumps({"display_name": "Wan Lightning", "family": "wan22", "role": "high"}),
                encoding="utf-8",
            )

            with patch.object(mlx_video_manager, "MLX_VIDEO_LORAS_DIR", root):
                adapters = mlx_video_manager.mlx_video_list_loras()

        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0]["display_name"], "Wan Lightning")
        self.assertEqual(adapters[0]["family"], "wan22")
        self.assertEqual(adapters[0]["role"], "high")
        self.assertTrue(adapters[0]["generation_loadable"])

    def test_cli_job_metadata_preserves_string_tiling_and_video_cli_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "result"
            result_dir.mkdir()
            output = result_dir / "out.mp4"

            def fake_run(command, **_kwargs):
                output.write_bytes(b"video")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            model = {
                "id": "wan22-final-hq",
                "label": "Wan2.2 Final HQ",
                "engine": "wan",
                "preset": "wan_final",
                "default_width": 1280,
                "default_height": 704,
                "default_frames": 81,
                "default_fps": 24,
                "guide_scale": "3.0,4.0",
                "shift": 12.0,
                "trim_first_frames": 1,
            }
            with patch.object(mlx_video_manager, "_build_mlx_video_command", return_value=(["mlx_video.wan_2.generate", "--output-path", str(output)], model, [])), \
                 patch.object(mlx_video_manager, "_engine_capabilities", return_value={"tiling": True}), \
                 patch.object(mlx_video_manager, "_patch_status", return_value={"matches_target_ref": True}), \
                 patch.object(mlx_video_manager, "_result_dir", return_value=result_dir), \
                 patch.object(mlx_video_manager, "_set_job", return_value=None), \
                 patch.object(mlx_video_manager.subprocess, "run", side_effect=fake_run), \
                 patch.object(mlx_video_manager.shutil, "which", return_value=None):
                metadata = mlx_video_manager._run_cli_job(  # noqa: SLF001
                    "job-1",
                    {
                        "action": "t2v",
                        "prompt": "final wan render",
                        "tiling": "conservative",
                        "negative_prompt": "blurry, low quality",
                        "no_negative_prompt": False,
                        "trim_first_frames": 2,
                        "guide_scale": "2.0,3.0",
                        "shift": 9,
                    },
                )

        self.assertEqual(metadata["tiling"], "conservative")
        self.assertEqual(metadata["negative_prompt"], "blurry, low quality")
        self.assertFalse(metadata["no_negative_prompt"])
        self.assertEqual(metadata["trim_first_frames"], 2)
        self.assertEqual(metadata["guide_scale"], "2.0,3.0")
        self.assertEqual(metadata["shift"], 9)

    def test_video_attachments_filter_by_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attachments.json"
            path.write_text(
                json.dumps(
                    [
                        {"target_type": "song", "target_id": "a", "result_id": "one"},
                        {"target_type": "album", "target_id": "b", "result_id": "two"},
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(mlx_video_manager, "MLX_VIDEO_ATTACHMENTS_PATH", path):
                self.assertEqual(len(mlx_video_manager.mlx_video_list_attachments()), 2)
                song = mlx_video_manager.mlx_video_list_attachments(target_type="song", target_id="a")

        self.assertEqual(song[0]["result_id"], "one")


if __name__ == "__main__":
    unittest.main()
