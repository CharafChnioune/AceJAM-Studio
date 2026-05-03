import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lora_trainer import (
    AceTrainingManager,
    TrainingJob,
    build_epoch_audition_caption,
    default_training_device,
    default_epoch_audition_lyrics,
    fit_epoch_audition_lyrics,
    model_from_variant,
    model_to_variant,
    normalize_training_song_model,
    training_device_policy,
    training_precision_for_device,
    utc_now,
)


class CaptureTrainingManager(AceTrainingManager):
    def require_ready(self):
        return None

    def _start_job(self, *, kind, command, params, paths):
        return {"kind": kind, "command": command, "params": params, "paths": paths}


class AuditionTrainingManager(AceTrainingManager):
    def __init__(self, *args, fail_epoch: int | None = None, **kwargs):
        self.commands = []
        self.audition_requests = []
        self.fail_epoch = fail_epoch
        super().__init__(*args, audition_runner=self._audition_runner, **kwargs)

    def _run_command_step(self, job_id, command, log_path, *, stage):
        self.commands.append((stage, list(command)))
        epoch = int(command[command.index("--epochs") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        checkpoint = output_dir / "checkpoints" / f"epoch_{epoch}_loss_0.1000"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self._append_log(log_path, f"{stage}\n")

    def _audition_runner(self, request):
        self.audition_requests.append(dict(request))
        if int(request["epoch"]) == self.fail_epoch:
            raise RuntimeError("audition boom")
        epoch = int(request["epoch"])
        return {
            "success": True,
            "result_id": f"audition-{epoch}",
            "audios": [{"result_id": f"audition-{epoch}", "audio_url": f"/media/results/audition-{epoch}/take-1.wav"}],
        }


class ResumeTrainingManager(AuditionTrainingManager):
    def require_ready(self):
        return None


class ImmediateThread:
    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class FakeNanProcess:
    pid = 12345

    def __init__(self):
        self.stdout = iter(["Epoch 2/2, Step 90, Loss: nan\n", "this should not be read\n"])
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        return -15 if self.terminated else 0


class LoraTrainerTest(unittest.TestCase):
    def make_manager(self, root: Path) -> CaptureTrainingManager:
        return CaptureTrainingManager(
            base_dir=root,
            data_dir=root / "data",
            model_cache_dir=root / "model_cache",
            release_models=lambda: None,
        )

    def make_peft_adapter(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"weights")
        return path

    def test_model_variant_mapping(self):
        self.assertEqual(model_to_variant("auto"), "xl_sft")
        self.assertEqual(model_to_variant("acestep-v15-turbo"), "turbo")
        self.assertEqual(model_to_variant("acestep-v15-xl-base"), "xl_base")
        self.assertEqual(model_to_variant("my-finetune"), "my-finetune")
        self.assertEqual(normalize_training_song_model("auto"), "acestep-v15-xl-sft")
        self.assertEqual(model_from_variant("turbo"), "acestep-v15-turbo")
        self.assertEqual(model_from_variant("xl_sft"), "acestep-v15-xl-sft")

    def test_default_training_device_prefers_mps_on_darwin_auto(self):
        with patch("lora_trainer.sys.platform", "darwin"), \
            patch("lora_trainer.platform.machine", return_value="arm64"), \
            patch("lora_trainer._torch_mps_available", return_value=True), \
            patch("lora_trainer._torch_cuda_available", return_value=True):
            self.assertEqual(default_training_device("auto"), "mps")

    def test_default_training_device_blocks_cpu_on_apple_silicon_mps(self):
        with patch("lora_trainer.sys.platform", "darwin"), \
            patch("lora_trainer.platform.machine", return_value="arm64"), \
            patch("lora_trainer._torch_mps_available", return_value=True), \
            patch.dict("lora_trainer.os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CPU LoRA training is blocked"):
                default_training_device("cpu")
            policy = training_device_policy()
            self.assertTrue(policy["cpu_blocked"])
            self.assertEqual(policy["default"], "mps")

    def test_default_training_device_allows_cpu_with_debug_override(self):
        with patch("lora_trainer.sys.platform", "darwin"), \
            patch("lora_trainer.platform.machine", return_value="arm64"), \
            patch("lora_trainer._torch_mps_available", return_value=True), \
            patch.dict("lora_trainer.os.environ", {"ACEJAM_ALLOW_CPU_TRAINING": "1"}, clear=True):
            self.assertEqual(default_training_device("cpu"), "cpu")

    def test_default_training_device_uses_cuda_then_cpu_for_auto(self):
        with patch("lora_trainer.sys.platform", "linux"), \
            patch("lora_trainer._torch_mps_available", return_value=False), \
            patch("lora_trainer._torch_cuda_available", return_value=True):
            self.assertEqual(default_training_device("auto"), "cuda")
        with patch("lora_trainer.sys.platform", "linux"), \
            patch("lora_trainer._torch_mps_available", return_value=False), \
            patch("lora_trainer._torch_cuda_available", return_value=False):
            self.assertEqual(default_training_device("auto"), "cpu")
            self.assertEqual(default_training_device("cpu"), "cpu")

    def test_training_precision_for_mps_forces_fp32_by_default(self):
        with patch.dict("lora_trainer.os.environ", {}, clear=True):
            self.assertEqual(training_precision_for_device("mps", "auto"), "fp32")
            self.assertEqual(training_precision_for_device("mps", "fp16"), "fp32")
            self.assertEqual(training_precision_for_device("mps", "16-mixed"), "fp32")
        with patch.dict("lora_trainer.os.environ", {"ACEJAM_ALLOW_MPS_FP16_TRAINING": "1"}, clear=True):
            self.assertEqual(training_precision_for_device("mps", "fp16"), "fp16")
            self.assertEqual(training_precision_for_device("mps", "auto"), "auto")
        self.assertEqual(training_precision_for_device("cuda", "auto"), "auto")
        self.assertEqual(training_precision_for_device("cpu", "auto"), "auto")

    def test_scan_and_save_dataset_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "songs"
            dataset.mkdir()
            audio = dataset / "track-one.wav"
            audio.write_bytes(b"not a real wav but still discoverable")
            (dataset / "track-one.lyrics.txt").write_text("[Verse]\nhello", encoding="utf-8")
            (dataset / "track-one.caption.txt").write_text("bright synth pop", encoding="utf-8")
            (dataset / "track-one.json").write_text(json.dumps({"bpm": 128, "keyscale": "C major"}), encoding="utf-8")

            manager = self.make_manager(root)
            scanned = manager.scan_dataset(dataset)
            self.assertEqual(len(scanned["files"]), 1)
            self.assertEqual(scanned["files"][0]["caption"], "bright synth pop")

            saved = manager.save_dataset(scanned["files"], dataset_id="unit", metadata={"custom_tag": "acejam"})
            payload = json.loads(Path(saved["dataset_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["custom_tag"], "acejam")
            self.assertEqual(payload["samples"][0]["audio_path"], str(audio.resolve()))
            self.assertEqual(payload["samples"][0]["lyrics"], "[Verse]\nhello")

    def test_one_click_labeling_keeps_user_language_and_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            labels = manager.label_entries(
                [
                    {
                        "path": str(root / "song.wav"),
                        "filename": "song.wav",
                        "caption": "bright schlager",
                        "lyrics": "",
                        "language": "unknown",
                    }
                ],
                trigger_tag="charafstyle",
                language="de",
            )

            self.assertEqual(labels[0]["language"], "de")
            self.assertEqual(labels[0]["lyrics"], "[Instrumental]")
            self.assertTrue(labels[0]["caption"].startswith("charafstyle, "))
            self.assertEqual(labels[0]["label_source"], "deterministic_filename")

    def test_one_click_defaults_are_lora_and_dynamic_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            with patch("lora_trainer.default_training_device", return_value="mps"):
                params = manager._one_click_params(
                    {
                        "dataset_id": "unit",
                        "import_root": str(root),
                        "trigger_tag": "voicehook",
                        "language": "nl",
                        "sample_count": 12,
                    },
                    dataset_id="unit",
                    import_root=root,
                )

            self.assertEqual(params["adapter_type"], "lora")
            self.assertEqual(params["tag_position"], "prepend")
            self.assertEqual(params["train_batch_size"], 1)
            self.assertEqual(params["gradient_accumulation"], 4)
            self.assertEqual(params["rank"], 64)
            self.assertEqual(params["alpha"], 128)
            self.assertEqual(params["dropout"], 0.1)
            self.assertEqual(params["training_shift"], 3.0)
            self.assertEqual(params["training_seed"], 42)
            self.assertIsNone(params["train_epochs"])
            self.assertEqual(params["save_every_n_epochs"], 1)
            self.assertEqual(params["song_model"], "acestep-v15-xl-sft")
            self.assertEqual(params["model_variant"], "xl_sft")
            self.assertFalse(params["epoch_audition"]["enabled"])
            self.assertEqual(params["device"], "mps")
            self.assertEqual(params["precision"], "fp32")
            self.assertEqual(manager.auto_epochs(20), 800)
            self.assertEqual(manager.auto_epochs(21), 500)
            self.assertEqual(manager.auto_epochs(101), 300)

    def test_one_click_explicit_cpu_device_is_preserved_when_debug_override_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            with patch("lora_trainer.sys.platform", "darwin"), \
                patch("lora_trainer.platform.machine", return_value="arm64"), \
                patch("lora_trainer._torch_mps_available", return_value=True), \
                patch.dict("lora_trainer.os.environ", {"ACEJAM_ALLOW_CPU_TRAINING": "1"}, clear=True):
                params = manager._one_click_params(
                    {
                        "dataset_id": "unit",
                        "import_root": str(root),
                        "trigger_tag": "voicehook",
                        "language": "nl",
                        "device": "cpu",
                    },
                    dataset_id="unit",
                    import_root=root,
                )

            self.assertEqual(params["device"], "cpu")
            self.assertEqual(params["precision"], "auto")

    def test_one_click_auto_song_model_uses_xl_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            with patch("lora_trainer.default_training_device", return_value="mps"):
                params = manager._one_click_params(
                    {
                        "dataset_id": "unit",
                        "import_root": str(root),
                        "trigger_tag": "voicehook",
                        "song_model": "auto",
                    },
                    dataset_id="unit",
                    import_root=root,
                )

        self.assertEqual(params["song_model"], "acestep-v15-xl-sft")
        self.assertEqual(params["model_variant"], "xl_sft")

    def test_one_click_explicit_variant_keeps_audition_model_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            with patch("lora_trainer.default_training_device", return_value="mps"):
                params = manager._one_click_params(
                    {
                        "dataset_id": "unit",
                        "import_root": str(root),
                        "trigger_tag": "voicehook",
                        "song_model": "auto",
                        "model_variant": "turbo",
                    },
                    dataset_id="unit",
                    import_root=root,
                )

        self.assertEqual(params["song_model"], "acestep-v15-turbo")
        self.assertEqual(params["model_variant"], "turbo")

    def test_preprocess_command_uses_vendor_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            dataset_json = root / "dataset.json"
            dataset_json.write_text('{"samples":[]}', encoding="utf-8")
            job = manager.start_preprocess({"dataset_json": str(dataset_json), "song_model": "acestep-v15-xl-turbo"})
            command = job["command"]
            self.assertIn("-m", command)
            self.assertIn("acestep.training_v2.cli.train_fixed", command)
            self.assertIn("xl_turbo", command)

    def test_train_command_supports_lokr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor_dir = root / "tensors"
            tensor_dir.mkdir()
            manager = self.make_manager(root)
            job = manager.start_train(
                {
                    "tensor_dir": str(tensor_dir),
                    "song_model": "acestep-v15-turbo",
                    "adapter_type": "lokr",
                    "train_epochs": 3,
                    "device": "mps",
                }
            )
            command = job["command"]
            self.assertIn("fixed", command)
            self.assertIn("--adapter-type", command)
            self.assertIn("lokr", command)
            self.assertIn("--lokr-weight-decompose", command)
            self.assertEqual(command[command.index("--save-every") + 1], "1")
            self.assertEqual(command[command.index("--device") + 1], "mps")
            self.assertEqual(command[command.index("--precision") + 1], "fp32")
            self.assertEqual(job["params"]["precision"], "fp32")

    def test_train_command_carries_trigger_tag_for_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor_dir = root / "tensors"
            tensor_dir.mkdir()
            manager = self.make_manager(root)
            job = manager.start_train(
                {
                    "tensor_dir": str(tensor_dir),
                    "song_model": "acestep-v15-xl-sft",
                    "trigger_tag": "charaf hook",
                    "adapter_type": "lora",
                    "train_epochs": 3,
                    "device": "mps",
                }
            )

            command = job["command"]
            self.assertEqual(job["params"]["trigger_tag"], "charaf hook")
            self.assertEqual(job["params"]["display_name"], "charaf hook")
            self.assertEqual(job["params"]["save_every_n_epochs"], 1)
            self.assertEqual(job["params"]["device"], "mps")
            self.assertEqual(command[command.index("--save-every") + 1], "1")
            self.assertEqual(command[command.index("--device") + 1], "mps")
            self.assertEqual(command[command.index("--precision") + 1], "fp32")
            self.assertEqual(job["params"]["precision"], "fp32")
            self.assertTrue(Path(job["paths"]["output_dir"]).name.startswith("charaf-hook-"))

    def test_manual_train_accepts_epoch_audition_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor_dir = root / "tensors"
            tensor_dir.mkdir()
            manager = self.make_manager(root)
            job = manager.start_train(
                {
                    "tensor_dir": str(tensor_dir),
                    "song_model": "acestep-v15-turbo",
                    "trigger_tag": "charaf hook",
                    "language": "en",
                    "train_epochs": 2,
                    "epoch_audition_enabled": True,
                    "epoch_audition_genre": "rap",
                    "epoch_audition_caption": "bright pop",
                    "epoch_audition_lyrics": "[Verse]\nUser draft line should stay out of the generated test",
                    "epoch_audition_seed": 123,
                    "epoch_audition_scale": 0.75,
                }
            )

            audition = job["params"]["epoch_audition"]
            self.assertTrue(audition["enabled"])
            self.assertEqual(audition["duration"], 20)
            self.assertIn("charaf hook", audition["caption"])
            self.assertIn("bright pop", audition["caption"])
            self.assertIn("clear intelligible vocal", audition["caption"])
            self.assertEqual(audition["lyrics_source"], "genre_default")
            self.assertEqual(audition["genre"], "rap")
            self.assertEqual(audition["genre_profile"], "rap")
            self.assertIn("Every bar lands clean", audition["lyrics"])
            self.assertNotIn("charaf hook", audition["lyrics"])
            self.assertNotIn("User draft line", audition["lyrics"])
            self.assertEqual(audition["user_lyrics"], "[Verse]\nUser draft line should stay out of the generated test")
            self.assertEqual(audition["seed"], 123)
            self.assertEqual(audition["scale"], 0.75)
            self.assertEqual(audition["vocal_language"], "en")

    def test_epoch_audition_defaults_do_not_require_user_lyrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor_dir = root / "tensors"
            tensor_dir.mkdir()
            manager = self.make_manager(root)
            job = manager.start_train(
                {
                    "tensor_dir": str(tensor_dir),
                    "song_model": "acestep-v15-turbo",
                    "trigger_tag": "Pac",
                    "language": "en",
                    "train_epochs": 2,
                    "epoch_audition_enabled": True,
                    "epoch_audition_genre": "rap",
                    "epoch_audition_caption": "west coast rap, hip hop",
                }
            )

            audition = job["params"]["epoch_audition"]
            self.assertTrue(audition["enabled"])
            self.assertEqual(audition["genre_profile"], "rap")
            self.assertEqual(audition["lyrics_source"], "genre_default")
            self.assertIn("Pac", audition["caption"])
            self.assertIn("west coast rap", audition["caption"])
            self.assertIn("hip hop drums", audition["caption"])
            self.assertIn("Every bar lands clean", audition["lyrics"])
            self.assertNotIn("Pac", audition["lyrics"])

    def test_default_epoch_audition_lyrics_keep_trigger_out_of_song_text(self):
        lyrics, meta = default_epoch_audition_lyrics("drill rap, dark piano", trigger_tag="Pac", lyrics_hint="Pac forever")
        caption = build_epoch_audition_caption("drill rap, dark piano", trigger_tag="Pac")

        self.assertEqual(meta["lyrics_source"], "genre_default")
        self.assertEqual(meta["genre_profile"], "rap")
        self.assertFalse(meta["trigger_in_lyrics"])
        self.assertIn("Pac", caption)
        self.assertIn("drill rap", caption)
        self.assertIn("clear intelligible vocal", caption)
        self.assertNotIn("Pac", lyrics)

    def test_epoch_audition_lyrics_are_fitted_for_twenty_seconds(self):
        lyrics = "\n".join(
            [
                "[Final Chorus - rap, apocalyptic, choir vocals, full climax]",
                "Count the room and keep it moving",
                "Name the chair and keep it true",
                "Every lie receives a number",
                "Every shadow turns to proof",
                "No more myth and no more static",
                "[Verse 4 - rap, acapella start, then drums return]",
                "Borrowed soil and fountain pens",
                "Concrete learned the mother tongue",
                "Every crack became a chorus",
                "Every curb knew what was done",
                "[drums return, building energy]",
                "Arrangement note should not become lyrics",
                "[Outro - fading, acapella, choir hum]",
                "You signed the wrong silence",
            ]
        )

        fitted, meta = fit_epoch_audition_lyrics(lyrics, duration=20)

        sung_lines = [line for line in fitted.splitlines() if line.strip() and not line.startswith("[")]
        self.assertLessEqual(len(fitted), 360)
        self.assertLessEqual(len(sung_lines), 4)
        self.assertEqual(meta["action"], "fit_for_20s")
        self.assertEqual(meta["max_sung_lines"], 4)
        self.assertTrue(meta["timed_structure"])
        self.assertEqual(meta["time_slices"][0]["start"], 0.0)
        self.assertEqual(meta["time_slices"][-1]["end"], 20.0)
        self.assertIn("[Chorus - seconds", fitted)
        self.assertIn("[Verse - seconds", fitted)
        self.assertNotIn("Final Chorus -", fitted)
        self.assertNotIn("Verse 4 -", fitted)
        self.assertNotIn("[drums return", fitted)
        self.assertNotIn("Arrangement note", fitted)

    def test_epoch_audition_empty_verse_before_outro_stays_a_verse(self):
        lyrics = "\n".join(
            [
                "[Final Chorus - rap, apocalyptic, choir vocals, full climax]",
                "Count the room and keep it moving",
                "Name the chair and keep it true",
                "[Verse 4 - rap, acapella start, then drums return]",
                "[Outro - acapella, close-mic vocal]",
                "Borrowed soil and fountain pens",
                "Concrete learned the mother tongue",
            ]
        )

        fitted, _ = fit_epoch_audition_lyrics(lyrics, duration=20)

        self.assertIn("[Verse - seconds", fitted)
        self.assertIn("Borrowed soil", fitted)
        self.assertNotIn("[Outro]", fitted)

    def test_training_command_step_fails_on_nonfinite_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = AceTrainingManager(base_dir=root, data_dir=root / "data", model_cache_dir=root / "model_cache")
            job = TrainingJob(
                id="nanjob",
                kind="train",
                state="running",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=["python", "train.py"],
                params={},
                paths={},
                log_path=str(root / "job.log"),
            )
            with manager._lock:
                manager._write_job_unlocked(job)
            fake = FakeNanProcess()

            with patch("lora_trainer.subprocess.Popen", return_value=fake), \
                self.assertRaisesRegex(RuntimeError, "non-finite loss"):
                manager._run_command_step("nanjob", ["python", "train.py"], Path(job.log_path), stage="train epoch 2/300")

            self.assertTrue(fake.terminated)
            self.assertIn("Loss: nan", Path(job.log_path).read_text(encoding="utf-8"))

    def test_epoch_auditions_run_once_per_epoch_with_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = AuditionTrainingManager(base_dir=root, data_dir=root / "data", model_cache_dir=root / "model_cache")
            output_dir = root / "data" / "lora_training" / "unit"
            job = TrainingJob(
                id="auditionjob",
                kind="train",
                state="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=["python", "train.py"],
                params={},
                paths={},
                log_path=str(root / "job.log"),
            )
            with manager._lock:
                manager._write_job_unlocked(job)
            command = ["python", "train.py", "--output-dir", str(output_dir), "--epochs", "2", "--save-every", "99"]
            params = {
                "adapter_type": "lora",
                "trigger_tag": "charaf hook",
                "song_model": "acestep-v15-turbo",
                "model_variant": "turbo",
                "language": "en",
                "training_seed": 42,
                "epoch_audition": {
                    "enabled": True,
                    "caption": "charaf hook, bright pop",
                    "lyrics": "[Verse]\nLine one",
                    "duration": 20,
                    "seed": 77,
                    "scale": 0.65,
                },
            }

            manager._run_train_command_with_epoch_auditions(
                "auditionjob",
                command,
                output_dir,
                Path(job.log_path),
                epochs=2,
                params=params,
            )

            self.assertEqual(len(manager.commands), 2)
            self.assertTrue(all(cmd[cmd.index("--save-every") + 1] == "1" for _, cmd in manager.commands))
            self.assertTrue(all("--scheduler-epochs" in cmd for _, cmd in manager.commands))
            self.assertEqual([item["epoch"] for item in manager.audition_requests], [1, 2])
            self.assertEqual(manager.audition_requests[0]["duration"], 20)
            self.assertIn("[Verse - seconds 0-20", manager.audition_requests[0]["lyrics"])
            self.assertIn("Line one", manager.audition_requests[0]["lyrics"])
            self.assertEqual(manager.audition_requests[0]["vocal_language"], "en")
            self.assertEqual(manager.audition_requests[0]["lyrics_fit"]["action"], "fit_for_20s")
            self.assertTrue(manager.audition_requests[0]["lyrics_fit"]["timed_structure"])
            self.assertTrue(manager.audition_requests[0]["checkpoint_path"].endswith("epoch_1_loss_0.1000"))
            stored = manager.get_job("auditionjob")
            self.assertEqual([item["status"] for item in stored["result"]["epoch_auditions"]], ["succeeded", "succeeded"])

    def test_epoch_audition_failure_is_recorded_and_training_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = AuditionTrainingManager(base_dir=root, data_dir=root / "data", model_cache_dir=root / "model_cache", fail_epoch=1)
            output_dir = root / "data" / "lora_training" / "unit"
            job = TrainingJob(
                id="failaudition",
                kind="train",
                state="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=["python", "train.py"],
                params={},
                paths={},
                log_path=str(root / "job.log"),
            )
            with manager._lock:
                manager._write_job_unlocked(job)
            command = ["python", "train.py", "--output-dir", str(output_dir), "--epochs", "2"]
            params = {
                "adapter_type": "lora",
                "song_model": "acestep-v15-turbo",
                "model_variant": "turbo",
                "epoch_audition": {"enabled": True, "caption": "test", "lyrics": "[Verse]\nLine", "duration": 20},
            }

            manager._run_train_command_with_epoch_auditions(
                "failaudition",
                command,
                output_dir,
                Path(job.log_path),
                epochs=2,
                params=params,
            )

            self.assertEqual(len(manager.commands), 2)
            stored = manager.get_job("failaudition")
            auditions = stored["result"]["epoch_auditions"]
            self.assertEqual([item["status"] for item in auditions], ["failed", "succeeded"])
            self.assertIn("audition boom", auditions[0]["error"])

    def test_startup_marks_running_epoch_auditions_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = TrainingJob(
                id="stalejob",
                kind="one_click_train",
                state="running",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=["python", "train.py"],
                params={},
                paths={},
                log_path=str(root / "data" / "lora_jobs" / "stalejob" / "job.log"),
                result={
                    "epoch_auditions": [
                        {
                            "epoch": 2,
                            "checkpoint_path": "/tmp/checkpoint",
                            "status": "running",
                            "error": "",
                            "audio_url": "",
                            "result_id": "",
                        }
                    ]
                },
            )
            job_dir = root / "data" / "lora_jobs" / "stalejob"
            job_dir.mkdir(parents=True)
            (job_dir / "job.json").write_text(json.dumps(job.to_dict()), encoding="utf-8")

            manager = AceTrainingManager(base_dir=root, data_dir=root / "data", model_cache_dir=root / "model_cache")

            stored = manager.get_job("stalejob")
            self.assertEqual(stored["state"], "failed")
            self.assertEqual(stored["error"], "Job was interrupted by an app restart")
            audition = stored["result"]["epoch_auditions"][0]
            self.assertEqual(audition["status"], "failed")
            self.assertEqual(audition["error"], "Interrupted by app restart")

    def test_resume_job_forces_mps_and_retries_failed_latest_audition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ResumeTrainingManager(base_dir=root, data_dir=root / "data", model_cache_dir=root / "model_cache")
            tensor_dir = root / "data" / "lora_tensors" / "resume"
            output_dir = root / "data" / "lora_training" / "resume"
            checkpoint = output_dir / "checkpoints" / "epoch_1_loss_0.9130"
            tensor_dir.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            job = TrainingJob(
                id="resumejob",
                kind="one_click_train",
                state="failed",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=[
                    "acejam-one-click-lora",
                    "--dataset-dir",
                    "/stale/tensors",
                    "--output-dir",
                    "/stale/output",
                    "--log-dir",
                    "/stale/runs",
                    "--epochs",
                    "3",
                    "--device",
                    "cpu",
                    "--precision",
                    "auto",
                ],
                params={
                    "adapter_type": "lora",
                    "trigger_tag": "charaf hook",
                    "song_model": "acestep-v15-turbo",
                    "model_variant": "turbo",
                    "train_epochs": 3,
                    "training_seed": 42,
                    "device": "cpu",
                    "epoch_audition": {
                        "enabled": True,
                        "caption": "charaf hook",
                        "lyrics": "[Verse]\nLine",
                        "duration": 20,
                    },
                },
                paths={
                    "tensor_output": str(tensor_dir),
                    "output_dir": str(output_dir),
                    "final_adapter": str(output_dir / "final"),
                    "log_dir": str(output_dir / "runs"),
                },
                log_path=str(root / "data" / "lora_jobs" / "resumejob" / "job.log"),
                result={
                    "epochs": 3,
                    "epoch_auditions": [
                        {
                            "epoch": 1,
                            "checkpoint_path": str(checkpoint),
                            "status": "failed",
                            "error": "module name can't contain dot",
                        }
                    ],
                },
            )
            with manager._lock:
                manager._write_job_unlocked(job)

            with patch("lora_trainer.default_training_device", return_value="mps"), \
                patch("lora_trainer.threading.Thread", ImmediateThread):
                resumed = manager.resume_job("resumejob")

            self.assertEqual(resumed["params"]["device"], "mps")
            self.assertEqual(resumed["params"]["precision"], "fp32")
            self.assertEqual([item["epoch"] for item in manager.audition_requests], [1, 2, 3])
            self.assertEqual(manager.audition_requests[0]["lora_adapter_name"], "epoch_1_epoch_1_loss_0_9130")
            self.assertEqual([stage for stage, _ in manager.commands], ["train epoch 2/3", "train epoch 3/3"])
            first_command = manager.commands[0][1]
            self.assertEqual(first_command[first_command.index("--device") + 1], "mps")
            self.assertEqual(first_command[first_command.index("--precision") + 1], "fp32")
            self.assertEqual(first_command[first_command.index("--resume-from") + 1], str(checkpoint))
            self.assertEqual(first_command[first_command.index("--scheduler-epochs") + 1], "3")

    def test_resume_job_retries_incomplete_latest_audition_from_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ResumeTrainingManager(base_dir=root, data_dir=root / "data", model_cache_dir=root / "model_cache")
            tensor_dir = root / "data" / "lora_tensors" / "resume"
            output_dir = root / "data" / "lora_training" / "resume"
            checkpoint_1 = output_dir / "checkpoints" / "epoch_1_loss_0.9130"
            checkpoint_2 = output_dir / "checkpoints" / "epoch_2_loss_0.8726"
            tensor_dir.mkdir(parents=True)
            checkpoint_1.mkdir(parents=True)
            checkpoint_2.mkdir(parents=True)
            job = TrainingJob(
                id="resumeepoch2",
                kind="one_click_train",
                state="failed",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=[
                    "acejam-one-click-lora",
                    "--dataset-dir",
                    "/stale/tensors",
                    "--output-dir",
                    "/stale/output",
                    "--log-dir",
                    "/stale/runs",
                    "--epochs",
                    "3",
                    "--device",
                    "cpu",
                    "--precision",
                    "auto",
                ],
                params={
                    "adapter_type": "lora",
                    "trigger_tag": "charaf hook",
                    "song_model": "acestep-v15-xl-sft",
                    "model_variant": "xl_sft",
                    "train_epochs": 3,
                    "training_seed": 42,
                    "device": "cpu",
                    "epoch_audition": {
                        "enabled": True,
                        "caption": "charaf hook",
                        "lyrics": "[Verse]\nLine",
                        "duration": 20,
                    },
                },
                paths={
                    "tensor_output": str(tensor_dir),
                    "output_dir": str(output_dir),
                    "final_adapter": str(output_dir / "final"),
                    "log_dir": str(output_dir / "runs"),
                },
                log_path=str(root / "data" / "lora_jobs" / "resumeepoch2" / "job.log"),
                result={
                    "epochs": 3,
                    "epoch_auditions": [
                        {
                            "epoch": 1,
                            "checkpoint_path": str(checkpoint_1),
                            "status": "succeeded",
                            "result_id": "ok-1",
                            "audio_url": "/media/results/ok-1/take.wav",
                        },
                        {
                            "epoch": 2,
                            "checkpoint_path": str(checkpoint_2),
                            "status": "running",
                            "result_id": "",
                            "audio_url": "",
                        },
                    ],
                },
            )
            with manager._lock:
                manager._write_job_unlocked(job)

            with patch("lora_trainer.default_training_device", return_value="mps"), \
                patch("lora_trainer.threading.Thread", ImmediateThread):
                resumed = manager.resume_job("resumeepoch2")

            self.assertEqual(resumed["params"]["device"], "mps")
            self.assertEqual([item["epoch"] for item in manager.audition_requests], [2, 3])
            self.assertEqual(manager.audition_requests[0]["checkpoint_path"], str(checkpoint_2))
            self.assertEqual(manager.audition_requests[0]["lora_adapter_name"], "epoch_2_epoch_2_loss_0_8726")
            self.assertEqual([stage for stage, _ in manager.commands], ["train epoch 3/3"])
            command = manager.commands[0][1]
            self.assertEqual(command[command.index("--resume-from") + 1], str(checkpoint_2))
            self.assertEqual(command[command.index("--device") + 1], "mps")
            self.assertEqual(command[command.index("--precision") + 1], "fp32")

    def test_lokr_epoch_auditions_are_skipped_but_save_every_epoch_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = AuditionTrainingManager(base_dir=root, data_dir=root / "data", model_cache_dir=root / "model_cache")
            output_dir = root / "data" / "lora_training" / "unit"
            job = TrainingJob(
                id="lokraudit",
                kind="train",
                state="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=["python", "train.py"],
                params={},
                paths={},
                log_path=str(root / "job.log"),
            )
            with manager._lock:
                manager._write_job_unlocked(job)
            command = ["python", "train.py", "--output-dir", str(output_dir), "--epochs", "2", "--save-every", "99"]
            params = {
                "adapter_type": "lokr",
                "epoch_audition": {"enabled": True, "caption": "test", "lyrics": "[Verse]\nLine", "duration": 20},
            }

            manager._run_train_command_with_epoch_auditions(
                "lokraudit",
                command,
                output_dir,
                Path(job.log_path),
                epochs=2,
                params=params,
            )

            self.assertEqual(len(manager.commands), 1)
            self.assertEqual(manager.commands[0][1][manager.commands[0][1].index("--save-every") + 1], "1")
            self.assertEqual(manager.audition_requests, [])
            stored = manager.get_job("lokraudit")
            self.assertIn("LoKr", stored["result"]["epoch_auditions_skipped_reason"])

    def test_register_adapter_uses_trigger_tag_and_stable_duplicate_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            source = self.make_peft_adapter(root / "training" / "final")

            first = manager.register_adapter(
                source,
                trigger_tag="charaf hook",
                adapter_type="lora",
                model_variant="xl_sft",
                song_model="acestep-v15-xl-sft",
                job_id="job-one",
            )
            second = manager.register_adapter(
                source,
                trigger_tag="charaf hook",
                adapter_type="lora",
                model_variant="xl_sft",
                song_model="acestep-v15-xl-sft",
                job_id="job-two",
            )

            self.assertEqual(Path(first["path"]).name, "charaf-hook")
            self.assertEqual(Path(second["path"]).name, "charaf-hook-2")
            metadata = json.loads((Path(first["path"]) / "acejam_adapter.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["display_name"], "charaf hook")
            self.assertEqual(metadata["trigger_tag"], "charaf hook")
            self.assertEqual(metadata["adapter_type"], "lora")
            self.assertEqual(metadata["model_variant"], "xl_sft")
            self.assertEqual(metadata["job_id"], "job-one")
            self.assertIn("source_paths", metadata)

            adapters = manager.list_adapters()
            exported = [item for item in adapters if item["source"] == "exports"]
            self.assertEqual([item["display_name"] for item in exported], ["charaf hook", "charaf hook"])
            self.assertTrue(all(item["is_loadable"] for item in exported))
            self.assertTrue(all(item["trigger_tag"] == "charaf hook" for item in exported))

    def test_lokr_adapter_is_listed_but_not_generation_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            adapter_dir = root / "data" / "loras" / "lokr-style"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "lokr_weights.safetensors").write_bytes(b"weights")
            (adapter_dir / "acejam_adapter.json").write_text(
                json.dumps({"display_name": "LoKr Style", "trigger_tag": "lokr-style", "adapter_type": "lokr"}),
                encoding="utf-8",
            )

            adapters = manager.list_adapters()

            self.assertEqual(len(adapters), 1)
            self.assertEqual(adapters[0]["display_name"], "LoKr Style")
            self.assertEqual(adapters[0]["adapter_type"], "lokr")
            self.assertFalse(adapters[0]["is_loadable"])

    def test_job_result_registers_manual_training_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            final_adapter = self.make_peft_adapter(root / "data" / "lora_training" / "manual" / "final")
            job = TrainingJob(
                id="manualjob",
                kind="train",
                state="succeeded",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=[],
                params={
                    "trigger_tag": "manual tag",
                    "display_name": "manual tag",
                    "adapter_type": "lora",
                    "model_variant": "turbo",
                    "song_model": "acestep-v15-turbo",
                    "epochs": 3,
                },
                paths={
                    "dataset_dir": str(root / "tensors"),
                    "output_dir": str(final_adapter.parent),
                    "final_adapter": str(final_adapter),
                    "log_dir": str(final_adapter.parent / "runs"),
                },
                log_path=str(root / "job.log"),
            )

            result = manager._job_result(job)

            self.assertTrue(result["adapter_exists"])
            self.assertEqual(Path(result["registered_adapter_path"]).name, "manual-tag")
            self.assertEqual(result["display_name"], "manual tag")
            metadata = json.loads((Path(result["registered_adapter_path"]) / "acejam_adapter.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["trigger_tag"], "manual tag")
            self.assertEqual(metadata["epochs"], 3)

    def test_tensorboard_runs_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self.make_manager(root)
            run_dir = root / "data" / "lora_training" / "demo" / "runs"
            run_dir.mkdir(parents=True)
            (run_dir / "events.out.tfevents.unit").write_text("event", encoding="utf-8")

            runs = manager.tensorboard_runs()

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["name"], "demo")


if __name__ == "__main__":
    unittest.main()
