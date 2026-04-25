import json
import tempfile
import unittest
from pathlib import Path

from lora_trainer import AceTrainingManager, model_to_variant


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
