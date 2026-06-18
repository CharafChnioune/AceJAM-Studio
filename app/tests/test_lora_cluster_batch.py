import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_lora_cluster_batch import derive_trigger_tag, run_batch


class FakeManager:
    def __init__(self, results_by_trigger):
        self.results_by_trigger = results_by_trigger
        self.started_payloads = []
        self.job_sequences = {}

    def active_job(self):
        return None

    def start_one_click_train(self, payload):
        trigger = str(payload["trigger_tag"])
        job_id = f"job-{len(self.started_payloads) + 1}"
        self.started_payloads.append(dict(payload))
        sequence = self.results_by_trigger[trigger]
        self.job_sequences[job_id] = [dict(item) for item in sequence]
        return {"id": job_id, "state": "queued"}

    def get_job(self, job_id):
        sequence = self.job_sequences[job_id]
        if len(sequence) > 1:
            return sequence.pop(0)
        return dict(sequence[0])


class LoraClusterBatchTest(unittest.TestCase):
    def make_adapter(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"weights")
        return path

    def test_trigger_tag_strips_lora_prefix(self):
        self.assertEqual(derive_trigger_tag("lora_westcoast_gfunk"), "westcoast_gfunk")
        self.assertEqual(derive_trigger_tag("westcoast_gfunk"), "westcoast_gfunk")

    def test_batch_skips_completed_entries_with_registered_and_mirror_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cluster_root = root / "clusters"
            mirror_root = root / "mirror"
            state_file = root / "state.json"
            log_file = root / "batch.log"
            folder = cluster_root / "lora_skip_me"
            folder.mkdir(parents=True)
            registered = self.make_adapter(root / "app" / "data" / "loras" / "skip_me")
            mirrored = self.make_adapter(mirror_root / "skip_me")
            state_file.write_text(
                json.dumps(
                    {
                        "entries": {
                            folder.name: {
                                "folder_name": folder.name,
                                "status": "completed",
                                "registered_path": str(registered),
                                "mirror_path": str(mirrored),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            manager = FakeManager({})
            summary = run_batch(
                manager=manager,
                cluster_root=cluster_root,
                mirror_root=mirror_root,
                state_file=state_file,
                log_file=log_file,
                poll_interval=0.01,
            )

            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(manager.started_payloads, [])

    def test_batch_retrains_completed_entry_if_mirror_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cluster_root = root / "clusters"
            mirror_root = root / "mirror"
            state_file = root / "state.json"
            log_file = root / "batch.log"
            folder = cluster_root / "lora_retry_me"
            folder.mkdir(parents=True)
            registered = self.make_adapter(root / "registered" / "retry_me")
            state_file.write_text(
                json.dumps(
                    {
                        "entries": {
                            folder.name: {
                                "folder_name": folder.name,
                                "status": "completed",
                                "registered_path": str(registered),
                                "mirror_path": str(mirror_root / "retry_me"),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            manager = FakeManager(
                {
                    "retry_me": [
                        {"id": "job-1", "state": "running", "stage": "train"},
                        {"id": "job-1", "state": "succeeded", "result": {"registered_adapter_path": str(registered)}},
                    ]
                }
            )

            with patch("train_lora_cluster_batch.time.sleep", return_value=None):
                summary = run_batch(
                    manager=manager,
                    cluster_root=cluster_root,
                    mirror_root=mirror_root,
                    state_file=state_file,
                    log_file=log_file,
                    poll_interval=0.01,
                )

            self.assertEqual(summary["completed"], 1)
            self.assertEqual(len(manager.started_payloads), 1)
            self.assertTrue((mirror_root / "retry_me" / "adapter_model.safetensors").is_file())

    def test_batch_mirrors_successful_registered_adapter_and_writes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cluster_root = root / "clusters"
            mirror_root = root / "mirror"
            state_file = root / "state.json"
            log_file = root / "batch.log"
            folder = cluster_root / "lora_westcoast_gfunk"
            folder.mkdir(parents=True)
            registered = self.make_adapter(root / "app" / "data" / "loras" / "westcoast_gfunk")
            output_dir = root / "training" / "westcoast_gfunk-job-1"
            (output_dir / "checkpoints").mkdir(parents=True)
            (output_dir / "checkpoints" / "epoch_10.bin").write_bytes(b"x")
            manager = FakeManager(
                {
                    "westcoast_gfunk": [
                        {"id": "job-1", "state": "queued", "stage": "queued"},
                        {
                            "id": "job-1",
                            "state": "succeeded",
                            "result": {
                                "registered_adapter_path": str(registered),
                                "generation_trigger_tag": "westcoast_gfunk",
                                "output_dir": str(output_dir),
                            },
                        },
                    ]
                }
            )

            with patch("train_lora_cluster_batch.time.sleep", return_value=None):
                summary = run_batch(
                    manager=manager,
                    cluster_root=cluster_root,
                    mirror_root=mirror_root,
                    state_file=state_file,
                    log_file=log_file,
                    poll_interval=0.01,
                )

            self.assertEqual(summary["completed"], 1)
            self.assertEqual(manager.started_payloads[0]["save_every_n_epochs"], 10)
            self.assertEqual(manager.started_payloads[0]["epoch_audition_every_n_epochs"], 10)
            self.assertEqual(manager.started_payloads[0]["auto_load"], False)
            self.assertEqual(manager.started_payloads[0]["custom_tag"], "westcoast_gfunk")
            self.assertTrue((mirror_root / "westcoast_gfunk" / "adapter_model.safetensors").is_file())
            self.assertFalse(output_dir.exists())

            saved = json.loads(state_file.read_text(encoding="utf-8"))
            entry = saved["entries"][folder.name]
            self.assertEqual(entry["status"], "completed")
            self.assertEqual(entry["trigger_tag"], "westcoast_gfunk")
            self.assertEqual(entry["generation_trigger_tag"], "westcoast_gfunk")
            self.assertEqual(entry["registered_path"], str(registered.resolve()))
            self.assertEqual(entry["mirror_path"], str((mirror_root / "westcoast_gfunk").resolve()))
            self.assertEqual(entry["training_output_dir"], str(output_dir))

    def test_batch_can_keep_training_artifacts_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cluster_root = root / "clusters"
            mirror_root = root / "mirror"
            state_file = root / "state.json"
            log_file = root / "batch.log"
            folder = cluster_root / "lora_keep_me"
            folder.mkdir(parents=True)
            registered = self.make_adapter(root / "app" / "data" / "loras" / "keep_me")
            output_dir = root / "training" / "keep_me-job-1"
            (output_dir / "final").mkdir(parents=True)
            (output_dir / "final" / "adapter.bin").write_bytes(b"x")
            manager = FakeManager(
                {
                    "keep_me": [
                        {"id": "job-1", "state": "running", "stage": "train"},
                        {
                            "id": "job-1",
                            "state": "succeeded",
                            "result": {
                                "registered_adapter_path": str(registered),
                                "generation_trigger_tag": "keep_me",
                                "output_dir": str(output_dir),
                            },
                        },
                    ]
                }
            )

            with patch("train_lora_cluster_batch.time.sleep", return_value=None):
                summary = run_batch(
                    manager=manager,
                    cluster_root=cluster_root,
                    mirror_root=mirror_root,
                    state_file=state_file,
                    log_file=log_file,
                    poll_interval=0.01,
                    cleanup_training_artifacts=False,
                )

            self.assertEqual(summary["completed"], 1)
            self.assertTrue(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
