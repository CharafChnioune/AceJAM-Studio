import importlib
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


os.environ.setdefault("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "1")
acejam_app = importlib.import_module("app")


class PromptAssistantTest(unittest.TestCase):
    def test_all_prompt_modes_load_system_prompts(self):
        for mode, info in acejam_app.PROMPT_ASSISTANT_MODES.items():
            with self.subTest(mode=mode):
                prompt = acejam_app._prompt_assistant_system_prompt(mode)
                self.assertIn("ace_lm_model", prompt)
                self.assertIn("planner_lm_provider", prompt)
                self.assertGreater(len(prompt), 200)
                self.assertTrue((acejam_app.BASE_DIR.parent / info["file"]).is_file())

    def test_prompt_assistant_run_parses_custom_payload_and_preserves_local_provider(self):
        raw = """
ACEJAM_PASTE_BLOCKS
Title: Neon Win

ACEJAM_PAYLOAD_JSON
{
  "task_type": "text2music",
  "song_model": "acestep-v15-xl-sft",
	  "ace_lm_model": "acestep-5Hz-lm-4B",
	  "planner_lm_provider": "ace",
	  "artist_name": "Neon Harbor",
	  "title": "Neon Win",
  "caption": "melodic rap, 808 bass, piano, male rap vocal, crisp modern mix",
  "negative_tags": "muddy mix, weak hook",
  "lyrics": "[Verse]\\nWe put the light on the dashboard\\n\\n[Chorus]\\nNeon win, we came too far",
  "duration": 60,
  "bpm": 124,
  "key_scale": "D minor",
  "time_signature": "4",
  "inference_steps": 50,
  "guidance_scale": 7.0,
  "audio_format": "wav",
  "auto_score": true,
  "auto_lrc": true
}
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw):
            response = client.post(
                "/api/prompt-assistant/run",
                json={
                    "mode": "custom",
                    "user_prompt": "make a hit",
                    "planner_lm_provider": "lmstudio",
                    "planner_model": "local-qwen",
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["payload"]["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
        self.assertEqual(data["payload"]["planner_lm_provider"], "lmstudio")
        self.assertEqual(data["payload"]["planner_model"], "local-qwen")
        self.assertNotIn("planner_ollama_model", data["payload"])
        self.assertTrue(data["payload"]["thinking"])
        self.assertTrue(data["payload"]["use_format"])
        self.assertTrue(data["payload"]["use_cot_lyrics"])
        self.assertFalse(data["payload"]["auto_score"])
        self.assertFalse(data["payload"]["auto_lrc"])
        self.assertEqual(data["payload"]["artist_name"], "Neon Harbor")
        self.assertEqual(data["payload"]["title"], "Neon Win")

    def test_prompt_assistant_run_parses_album_payload(self):
        raw = """
ACEJAM_ALBUM_SETTINGS_JSON
{
  "concept": "A focused test album",
  "song_model_strategy": "xl_sft_final",
  "ace_lm_model": "auto",
  "tracks": [
    {
	      "track_number": 1,
	      "artist_name": "Track Signal",
	      "title": "Track One",
      "caption": "pop, piano, female vocal, radio-ready",
      "lyrics": "[Verse]\\nLine one\\n\\n[Chorus]\\nHook"
    }
  ]
}
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw):
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "album", "user_prompt": "album", "ollama_model": "llama3.2"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()["payload"]
        self.assertEqual(payload["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
        self.assertEqual(payload["planner_lm_provider"], "ollama")
        self.assertTrue(payload["thinking"])
        self.assertTrue(payload["use_format"])
        self.assertTrue(payload["use_cot_lyrics"])
        self.assertEqual(payload["song_model_strategy"], "xl_sft_final")
        self.assertEqual(payload["tracks"][0]["artist_name"], "Track Signal")
        self.assertEqual(payload["tracks"][0]["title"], "Track One")

    def test_prompt_assistant_malformed_json_returns_raw_text(self):
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value="not json today"):
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "custom", "user_prompt": "bad json", "ollama_model": "llama3.2"},
            )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("not json today", data["raw_text"])


if __name__ == "__main__":
    unittest.main()
