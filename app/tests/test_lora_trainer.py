import json
import tempfile
import unittest
from pathlib import Path

from lora_trainer import AceTrainingManager, TrainingJob, model_to_variant, utc_now


class CaptureTrainingManager(AceTrainingManager):
    def require_ready(self):
        return None

    def _start_job(self, *, kind, command, params, paths):
        return {"kind": kind, "command": command, "params": params, "paths": paths}


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
        self.assertEqual(model_to_variant("acestep-v15-turbo"), "turbo")
        self.assertEqual(model_to_variant("acestep-v15-xl-base"), "xl_base")
        self.assertEqual(model_to_variant("my-finetune"), "my-finetune")

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
            self.assertEqual(manager.auto_epochs(20), 800)
            self.assertEqual(manager.auto_epochs(21), 500)
            self.assertEqual(manager.auto_epochs(101), 300)

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
                }
            )
            command = job["command"]
            self.assertIn("fixed", command)
            self.assertIn("--adapter-type", command)
            self.assertIn("lokr", command)
            self.assertIn("--lokr-weight-decompose", command)

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
                }
            )

            self.assertEqual(job["params"]["trigger_tag"], "charaf hook")
            self.assertEqual(job["params"]["display_name"], "charaf hook")
            self.assertTrue(Path(job["paths"]["output_dir"]).name.startswith("charaf-hook-"))

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
