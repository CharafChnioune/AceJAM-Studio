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
                self.assertIn(acejam_app.PROMPT_KIT_VERSION, prompt)
                self.assertIn("copy_paste_block", prompt)
                self.assertNotIn('"planner_lm_provider": "ollama"', prompt)
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
  "song_intent": {
    "genre_family": "rap",
    "subgenre": "melodic rap",
    "mood": "triumphant",
    "energy": "medium-high",
    "vocal_type": "male rap lead",
    "language": "en",
    "drum_groove": "crisp trap drums",
    "bass_low_end": "808 bass",
    "melodic_identity": "bright piano motif",
    "texture_space": "wide neon ambience",
    "mix_master": "crisp modern mix",
    "custom_tags": ["anthemic hook"],
    "caption": "melodic rap, crisp trap drums, 808 bass, bright piano motif, male rap lead, wide neon ambience, crisp modern mix"
  },
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
        self.assertEqual(data["payload"]["ace_lm_model"], "none")
        self.assertEqual(data["payload"]["planner_lm_provider"], "lmstudio")
        self.assertEqual(data["payload"]["planner_model"], "local-qwen")
        self.assertNotIn("planner_ollama_model", data["payload"])
        self.assertFalse(data["payload"]["thinking"])
        self.assertFalse(data["payload"]["use_format"])
        self.assertFalse(data["payload"]["use_cot_lyrics"])
        self.assertEqual(data["prompt_kit_version"], acejam_app.PROMPT_KIT_VERSION)
        self.assertEqual(data["payload"]["prompt_kit_version"], acejam_app.PROMPT_KIT_VERSION)
        self.assertIn("section_map", data["payload"])
        self.assertIn("quality_checks", data["payload"])
        self.assertFalse(data["payload"]["auto_score"])
        self.assertFalse(data["payload"]["auto_lrc"])
        self.assertEqual(data["payload"]["artist_name"], "Neon Harbor")
        self.assertEqual(data["payload"]["title"], "Neon Win")
        self.assertEqual(data["payload"]["song_intent"]["genre_family"], "rap")
        self.assertEqual(data["payload"]["song_intent"]["caption"], data["payload"]["caption"])
        self.assertIn("anthemic hook", data["payload"]["song_intent"]["custom_tags"])

    def test_prompt_assistant_passes_planner_settings_to_local_llm(self):
        raw = """
ACEJAM_PASTE_BLOCKS
Title: Neon Win

ACEJAM_PAYLOAD_JSON
{
  "task_type": "text2music",
  "title": "Neon Win",
  "artist_name": "Neon Harbor",
  "caption": "rap drums, 808 bass, piano, male rap vocal, polished mix",
  "lyrics": "[Verse]\\nConcrete numbers on the kitchen wall\\n[Chorus]\\nNeon win",
  "duration": 60
}
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw) as runner:
            response = client.post(
                "/api/prompt-assistant/run",
                json={
                    "mode": "custom",
                    "user_prompt": "make a rap song",
                    "planner_lm_provider": "lmstudio",
                    "planner_model": "local-qwen",
                    "planner_creativity_preset": "creative",
                    "planner_temperature": 0.77,
                    "planner_top_p": 0.96,
                    "planner_top_k": 88,
                    "planner_repeat_penalty": 1.05,
                    "planner_seed": "1234",
                    "planner_max_tokens": 3072,
                    "planner_context_length": 16384,
                    "planner_timeout": 90,
                },
            )

        self.assertEqual(response.status_code, 200)
        settings = runner.call_args.args[5]
        self.assertEqual(settings["planner_creativity_preset"], "creative")
        self.assertEqual(settings["planner_temperature"], 0.77)
        self.assertEqual(settings["planner_top_p"], 0.96)
        self.assertEqual(settings["planner_top_k"], 88)
        self.assertEqual(settings["planner_repeat_penalty"], 1.05)
        self.assertEqual(settings["planner_seed"], 1234)
        self.assertEqual(settings["planner_max_tokens"], 3072)
        self.assertEqual(settings["planner_context_length"], 16384)
        self.assertEqual(settings["planner_timeout"], 90.0)
        self.assertEqual(response.json()["payload"]["planner_temperature"], 0.77)

    def test_prompt_assistant_uses_staged_calls_and_carries_previous_payload(self):
        stage_intent = """
{"payload":{"caption":"glitch rap, off-beat drums, dark synths, 808 bass, male rap vocal, crisp modern mix","song_intent":{"genre_family":"rap","subgenre":"glitch rap","mood":"dark","energy":"mid-tempo","vocal_type":"male rap vocal","language":"en","drum_groove":"off-beat drums","bass_low_end":"808 bass","melodic_identity":"dark synths","mix_master":"crisp modern mix","genre_modules":["rap"],"style_tags":["glitch rap"],"rhythm_tags":["off-beat drums"],"instrument_tags":["808 bass","dark synths"],"vocal_tags":["male rap vocal"],"production_tags":["crisp modern mix"],"negative_tags":["muddy mix"]}}}
"""
        stage_writing = """
{"payload":{"title":"Consistent Signal","artist_name":"Neon Harbor","lyrics":"[Verse]\\nDark synths keep the signal steady\\n\\n[Chorus]\\nConsistent signal, never let it go","duration":60,"bpm":92,"key_scale":"D minor","time_signature":"4","song_intent":{"genre_family":"rap","style_tags":["glitch rap"],"instrument_tags":["808 bass","dark synths"]}}}
"""
        stage_render = """
{"payload":{"task_type":"text2music","song_model":"acestep-v15-xl-sft","quality_profile":"chart_master","song_intent":{"task_mode":"text2music","model_strategy":"chart_master","rhythm_tags":["off-beat drums"],"production_tags":["crisp modern mix"]}}}
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", side_effect=[stage_intent, stage_writing, stage_render]) as runner:
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "custom", "user_prompt": "make a coherent glitch rap song", "planner_lm_provider": "lmstudio", "planner_model": "local-qwen"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(runner.call_count, 3)
        second_stage_payload = runner.call_args_list[1].args[4]
        self.assertEqual(second_stage_payload["ai_fill_stage"], "song_writing")
        self.assertEqual(second_stage_payload["previous_ai_payload"]["song_intent"]["genre_family"], "rap")
        data = response.json()["payload"]
        self.assertEqual(data["title"], "Consistent Signal")
        self.assertEqual(data["song_intent"]["genre_family"], "rap")
        self.assertIn("glitch rap", data["song_intent"]["style_tags"])
        self.assertIn("808 bass", data["song_intent"]["instrument_tags"])
        self.assertIn("off-beat drums", data["song_intent"]["rhythm_tags"])

    def test_prompt_assistant_derives_song_intent_when_model_omits_it(self):
        raw = """
{"payload":{"title":"Derived Intent","artist_name":"Neon Harbor","caption":"melodic rap, crisp trap drums, 808 bass, piano, male rap vocal, polished mix","lyrics":"[Verse]\\nLine one\\n\\n[Chorus]\\nHook","duration":60}}
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw):
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "custom", "user_prompt": "make a melodic rap song", "planner_lm_provider": "lmstudio", "planner_model": "local-qwen"},
            )

        self.assertEqual(response.status_code, 200)
        intent = response.json()["payload"]["song_intent"]
        self.assertEqual(intent["caption"], "melodic rap, crisp trap drums, 808 bass, piano, male rap vocal, polished mix")
        self.assertNotIn(None, intent["genre_modules"])
        self.assertTrue(intent["genre_modules"])
        self.assertIn("808 bass", intent["instrument_tags"])
        self.assertTrue(intent["vocal_tags"])

    def test_prompt_assistant_local_retries_truncated_structured_json(self):
        calls = []

        def fake_chat(provider, model, messages, options=None, json_schema=None, **kwargs):
            calls.append({"provider": provider, "model": model, "options": options, "schema": json_schema})
            if len(calls) == 1:
                return {"content": "{\"payload\":{\"title\":\"Cut", "done_reason": "length", "truncated": True}
            return {
                "content": '{"payload":{"title":"Closed JSON","caption":"rap drums, 808 bass, clear vocal","lyrics":"[Verse]\\nLine one","duration":60}}',
                "done_reason": "stop",
                "truncated": False,
            }

        with patch.object(acejam_app, "_ensure_ollama_model_or_start_pull", return_value=None), \
            patch.object(acejam_app, "local_llm_chat_completion_response", side_effect=fake_chat):
            raw = acejam_app._run_prompt_assistant_local(
                "System prompt",
                "Make a song",
                "ollama",
                "qwen3:4b",
                {"lyrics": "long lyrics " * 400, "title": "Draft"},
                {"planner_max_tokens": 2048, "planner_context_length": 8192},
                mode="custom",
            )

        self.assertIn("Closed JSON", raw)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["schema"]["required"], ["payload"])
        self.assertEqual(calls[1]["options"]["num_predict"], 8192)
        self.assertEqual(calls[1]["options"]["num_ctx"], 32768)

    def test_prompt_assistant_truncated_json_error_is_user_friendly(self):
        client = TestClient(acejam_app.app)
        raw = 'ACEJAM_PAYLOAD_JSON\n{"title":"Cut off"'

        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw):
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "custom", "user_prompt": "make a song", "planner_lm_provider": "ollama", "planner_model": "qwen3:4b"},
            )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("AI Fill response was truncated", data["error"])
        self.assertNotIn("JSON object was not closed", data["error"])

    def test_prompt_assistant_ace_step_writer_routes_to_official_lm(self):
        client = TestClient(acejam_app.app)
        official_payload = {
            "success": True,
            "ace_lm_model": acejam_app.ACE_LM_PREFERRED_MODEL,
            "title": "Official Writer",
            "artist_name": "Five Hertz",
            "caption": "alt pop, live drums, sub bass, clear vocal, polished mix",
            "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook",
            "bpm": 108,
            "key_scale": "C minor",
            "time_signature": "4",
        }

        with patch.object(acejam_app, "_run_official_lm_aux", return_value=official_payload) as official, \
            patch.object(acejam_app, "_run_prompt_assistant_local") as local:
            response = client.post(
                "/api/prompt-assistant/run",
                json={
                    "mode": "custom",
                    "user_prompt": "write a sharp alt pop song",
                    "planner_lm_provider": "ace_step_lm",
                    "planner_model": acejam_app.ACE_LM_PREFERRED_MODEL,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(local.called)
        action, body = official.call_args.args
        self.assertEqual(action, "create_sample")
        self.assertEqual(body["planner_lm_provider"], "ace_step_lm")
        self.assertEqual(body["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
        payload = response.json()["payload"]
        self.assertEqual(payload["planner_lm_provider"], "ace_step_lm")
        self.assertEqual(payload["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
        self.assertEqual(payload["title"], "Official Writer")

    def test_prompt_assistant_blocks_trainer_mode_and_hides_prompt_listing(self):
        client = TestClient(acejam_app.app)
        prompts = client.get("/api/prompt-assistant/prompts")
        self.assertEqual(prompts.status_code, 200)
        self.assertNotIn("trainer", {item["mode"] for item in prompts.json()["prompts"]})
        self.assertIn("image", {item["mode"] for item in prompts.json()["prompts"]})
        self.assertIn("video", {item["mode"] for item in prompts.json()["prompts"]})

        response = client.post(
            "/api/prompt-assistant/run",
            json={"mode": "trainer", "user_prompt": "label my dataset", "planner_model": "local-qwen"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])
        self.assertIn("disabled for Trainer", response.json()["error"])

    def test_prompt_assistant_run_parses_image_payload_without_music_validation(self):
        raw = """
ACEJAM_PAYLOAD_JSON
{
  "action": "generate",
  "prompt": "premium square album artwork, neon street, cinematic lighting, no text",
  "model_id": "qwen-image",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "seed": -1,
  "lora_adapters": []
}
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw):
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "image", "user_prompt": "make album cover art", "planner_lm_provider": "lmstudio", "planner_model": "local-qwen"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsNone(data["validation"])
        self.assertEqual(data["payload"]["action"], "generate")
        self.assertEqual(data["payload"]["model_id"], "qwen-image")
        self.assertEqual(data["payload"]["width"], 1024)

    def test_prompt_assistant_run_parses_video_song_payload_with_mux_policy(self):
        raw = """
ACEJAM_PAYLOAD_JSON
{
  "action": "song_video",
  "prompt": "real-life night street music video, handheld camera, moody neon, no text",
  "model_id": "ltx2-fast-draft",
  "width": 512,
  "height": 320,
  "num_frames": 33,
  "steps": 8
}
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw):
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "video", "user_prompt": "make a short music video", "planner_lm_provider": "lmstudio", "planner_model": "local-qwen"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsNone(data["validation"])
        self.assertEqual(data["payload"]["action"], "song_video")
        self.assertEqual(data["payload"]["audio_policy"], "replace_with_source")
        self.assertTrue(data["payload"]["mux_audio"])

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
        self.assertEqual(payload["ace_lm_model"], "none")
        self.assertEqual(payload["planner_lm_provider"], "ollama")
        self.assertFalse(payload["thinking"])
        self.assertFalse(payload["use_format"])
        self.assertFalse(payload["use_cot_lyrics"])
        self.assertEqual(payload["prompt_kit_version"], acejam_app.PROMPT_KIT_VERSION)
        self.assertIn("section_map", payload["tracks"][0])
        self.assertIn("quality_checks", payload["tracks"][0])
        self.assertEqual(payload["tracks"][0]["ace_lm_model"], "none")
        self.assertEqual(payload["song_model_strategy"], "xl_sft_final")
        self.assertEqual(payload["tracks"][0]["artist_name"], "Track Signal")
        self.assertEqual(payload["tracks"][0]["title"], "Track One")

    def test_prompt_assistant_album_contract_repairs_returned_titles(self):
        raw = """
ACEJAM_ALBUM_SETTINGS_JSON
{
  "concept": "A safe repair album",
  "tracks": [
    {
      "track_number": 1,
      "artist_name": "Track Signal",
      "title": "Generated Season",
      "caption": "pop, piano, female vocal",
      "lyrics": "[Verse]\\nWe open doors"
    }
  ]
}
"""
        user_prompt = """
Album: Exact Assistant Brief
Concept: A safe repair album.
Track 1: "Lantern Keys" (Produced by Ada North)
(BPM: 88 | Style: warm boom-bap)
The Vibe: dusty piano and soft brass.
The Narrative: a locksmith helps everyone back into their apartments.
Lyrics:
"Turn the lantern keys"
"""
        client = TestClient(acejam_app.app)
        with patch.object(acejam_app, "_run_prompt_assistant_local", return_value=raw):
            response = client.post(
                "/api/prompt-assistant/run",
                json={"mode": "album", "user_prompt": user_prompt, "planner_lm_provider": "lmstudio", "planner_model": "local-qwen"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()["payload"]
        self.assertTrue(payload["input_contract_applied"])
        self.assertEqual(payload["album_title"], "Exact Assistant Brief")
        self.assertEqual(payload["tracks"][0]["title"], "Lantern Keys")
        self.assertEqual(payload["tracks"][0]["producer_credit"], "Ada North")
        self.assertEqual(payload["tracks"][0]["bpm"], 88)
        self.assertIn("Turn the lantern keys", payload["tracks"][0]["lyrics"])

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
