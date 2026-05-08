import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "1")
acejam_app = importlib.import_module("app")


class GenerationJobSummaryTest(unittest.TestCase):
    def test_payload_summary_includes_planned_memory_trigger_and_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "pac-epoch-34"
            adapter.mkdir()
            (adapter / "acejam_adapter.json").write_text(
                json.dumps(
                    {
                        "display_name": "2pac epoch 34",
                        "song_model": "acestep-v15-xl-sft",
                        "model_variant": "xl_sft",
                        "trigger_tag_raw": "2pac",
                        "generation_trigger_tag": "pac",
                        "trigger_source": "training",
                        "trigger_aliases": ["2pac"],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(acejam_app, "_IS_APPLE_SILICON", True):
                summary = acejam_app._generation_payload_summary(
                    {
                        "task_type": "text2music",
                        "song_model": "acestep-v15-xl-sft",
                        "device": "mps",
                        "duration": 180,
                        "batch_size": 8,
                        "use_lora": True,
                        "lora_adapter_path": str(adapter),
                        "lora_scale": 1.0,
                        "style_profile": "rap",
                        "caption": "West Coast hip hop beat, clear rap vocal",
                        "lyrics": "[Verse]\nStreet lights move while the drumline snaps\n[Chorus]\nHands up high when the hook comes back",
                    }
                )

        self.assertEqual(summary["requested_take_count"], 8)
        self.assertEqual(summary["actual_runner_batch_size"], 1)
        self.assertEqual(summary["memory_policy"]["policy"], "mps_standard_model_sequential_takes")
        self.assertEqual(summary["lora_trigger_tag"], "pac")
        self.assertEqual(summary["lora_trigger_source"], "training")
        self.assertEqual(summary["lora_trigger_conditioning_audit"]["status"], "planned")
        self.assertEqual(summary["style_profile"], "rap")
        self.assertEqual(summary["style_conditioning_audit"]["status"], "pass")

    def test_result_summary_preserves_lora_trigger_fields(self):
        summary = acejam_app._generation_result_summary(
            {
                "success": True,
                "use_lora_trigger": True,
                "lora_trigger_tag": "pac",
                "lora_trigger_source": "training",
                "lora_trigger_applied": True,
                "lora_trigger_conditioning_audit": {"status": "applied", "trigger_source": "training"},
            }
        )

        self.assertTrue(summary["use_lora_trigger"])
        self.assertEqual(summary["lora_trigger_tag"], "pac")
        self.assertEqual(summary["lora_trigger_source"], "training")
        self.assertTrue(summary["lora_trigger_applied"])
        self.assertEqual(summary["lora_trigger_conditioning_audit"]["status"], "applied")


if __name__ == "__main__":
    unittest.main()
