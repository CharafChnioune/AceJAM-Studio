import importlib
import json
import os
import unittest
from typing import Any
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
                self.assertTrue((acejam_app.BASE_DIR / "prompts" / info["file"]).is_file())

    def test_music_modes_get_full_acestep_reference_block(self):
        # Music modes must surface authoring rules + tag library + producer/rap cookbooks
        # + worked examples so Ollama/LM Studio can pattern-match Dre/No I.D./Metro requests.
        for mode in ("simple", "custom", "song", "news", "improve"):
            with self.subTest(mode=mode):
                prompt = acejam_app._prompt_assistant_system_prompt(mode)
                self.assertIn("## ACE-Step Authoring Rules", prompt)
                self.assertIn("## ACE-Step Tag Library", prompt)
                self.assertIn("## Producer-Format Cookbook", prompt)
                self.assertIn("## Rap-Mode Cookbook", prompt)
                self.assertIn("## Worked Examples", prompt)
                self.assertIn("Dr. Dre / G-funk era", prompt)
                self.assertIn("[Verse - rap]", prompt)
                self.assertIn("[Hook]", prompt)

    def test_album_wizard_fill_dispatches_through_crewai_micro(self):
        """Album mode in the prompt-assistant handler must route through the
        CrewAI Micro Tasks director (`plan_album`) instead of the legacy
        single-Ollama-call planner. Each wizard field is filled by a
        specialised agent (Topline Hook Writer, Tier-1 Lyric Writer, Sonic
        Tags Engineer, etc.). The legacy 'MLX Media's Album AI Fill planner'
        prompt was deleted; this test pins the new contract.
        """
        # The legacy single-call album system-prompt builder is removed.
        self.assertFalse(hasattr(acejam_app, "_album_prompt_assistant_system_prompt"))

        # Album mode is dispatched to the crew before reaching the staged
        # path, so plan_album receives crewai_micro engine + the user's
        # concept + the input_tracks list (when present).
        captured: dict[str, Any] = {}

        def fake_plan_album(**kwargs):
            captured.update(kwargs)
            return {
                "tracks": [
                    {
                        "track_number": 1,
                        "title": "Concrete Canyons",
                        "role": "opener",
                        "duration": 210.0,
                        "artist_name": "Ada North",
                        "producer_credit": "Dr. Dre",
                        "caption": "G-funk, 90s G-funk, talkbox lead, head-nod groove",
                        "tags": "G-funk, 90s G-funk, talkbox lead, head-nod groove",
                        "tag_list": ["G-funk", "talkbox lead"],
                        "lyrics": "[Intro]\nFrom the bottom (yeah)\n[Verse - rap]\nMama working doubles I was sleeping on the floor",
                        "lyrics_lines": ["[Intro]", "From the bottom (yeah)", "[Verse - rap]", "Mama working doubles I was sleeping on the floor"],
                        "bpm": 78,
                        "key_scale": "A minor",
                        "time_signature": "4",
                        "vocal_language": "en",
                        "production_team": {"executive_producer": "AceJAM Director"},
                        "quality_report": {"hit_angle": "low-end opener", "section_plan": ["[Intro]", "[Verse - rap]"]},
                    }
                ],
                "album_title": "Test Album",
                "concept": "test concept",
                "num_tracks": 1,
                "success": True,
                "warnings": [],
                "planning_engine": "crewai_micro",
                "crewai_used": True,
            }

        with patch("album_crew.plan_album", fake_plan_album):
            result = acejam_app._run_prompt_assistant_album_crew(
                body={"mode": "album", "user_prompt": "Album about systems and ledgers"},
                user_prompt="Album about systems and ledgers",
                current_payload={"num_tracks": 1, "track_duration": 210, "language": "en"},
                planner_provider="ollama",
                planner_model="qwen-test",
            )

        # CrewAI engine was forced regardless of input
        self.assertEqual(captured["options"]["agent_engine"], "crewai_micro")
        self.assertEqual(captured["concept"], "Album about systems and ledgers")
        self.assertEqual(captured["language"], "en")
        self.assertTrue(captured["use_crewai"])

        # Wizard payload shape: top-level metadata + tracks[] with edit-ready fields
        self.assertTrue(result["success"])
        payload = result["payload"]
        self.assertEqual(payload["agent_engine"], "crewai_micro")
        self.assertTrue(payload["crewai_used"])
        self.assertEqual(payload["concept"], "test concept")
        self.assertEqual(payload["album_title"], "Test Album")
        self.assertEqual(payload["num_tracks"], 1)
        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["album_writer_mode"], "per_track_writer_loop")
        self.assertEqual(payload["song_model_strategy"], "single_model_album")
        self.assertEqual(payload["final_song_model"], "acestep-v15-xl-sft")
        self.assertEqual(payload["quality_profile"], "chart_master")

        # First track is fully populated (artist, producer, caption, lyrics, BPM, etc.)
        track = payload["tracks"][0]
        self.assertEqual(track["title"], "Concrete Canyons")
        self.assertEqual(track["artist_name"], "Ada North")
        self.assertEqual(track["producer_credit"], "Dr. Dre")
        self.assertEqual(track["bpm"], 78)
        self.assertEqual(track["key_scale"], "A minor")
        self.assertIn("talkbox lead", track["caption"])
        self.assertIn("Mama working doubles", track["lyrics"])
        # Inference defaults are pre-filled so the wizard is not blank
        self.assertEqual(track["inference_steps"], 50)
        self.assertEqual(track["audio_format"], "wav32")
        # Production team is exposed in the wizard so user sees the crew makeup
        self.assertIn("executive_producer", track["production_team"])
        # Quality report mirrors the crew gate output for visibility
        self.assertIn("hit_angle", track["quality_report"])

    def test_album_wizard_fill_rejects_empty_concept(self):
        result = acejam_app._run_prompt_assistant_album_crew(
            body={"mode": "album"},
            user_prompt="",
            current_payload={},
            planner_provider="ollama",
            planner_model="qwen-test",
        )
        self.assertFalse(result["success"])
        self.assertIn("Album concept is empty", result["error"])

    def test_staged_system_prompt_inherits_acestep_reference_block(self):
        base = acejam_app._prompt_assistant_system_prompt("custom")
        staged = acejam_app._prompt_assistant_stage_system_prompt(
            base, "custom", "stage_one", "Test stage instruction.", 0, 3,
        )
        self.assertIn("## Producer-Format Cookbook", staged)
        self.assertIn("## Worked Examples", staged)
        self.assertIn("MULTI-PASS AI FILL STAGE 1/3", staged)
        self.assertIn("stage_one", staged)

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

    def test_prompt_assistant_ace_step_writer_falls_back_to_local(self):
        client = TestClient(acejam_app.app)
        local_raw = json.dumps(
            {
                "ACEJAM_PAYLOAD_JSON": {
                    "title": "Local Writer",
                    "artist_name": "Local Artist",
                    "caption": "alt pop, live drums, clean vocal, polished mix",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook",
                    "bpm": 108,
                    "key_scale": "C minor",
                    "time_signature": "4",
                }
            }
        )

        with patch.object(acejam_app, "_run_official_lm_aux") as official, \
            patch.object(acejam_app, "_run_prompt_assistant_local_staged", return_value=local_raw) as local:
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
        self.assertFalse(official.called, "ACE-Step 5Hz writer must never be reachable from prompt-assistant")
        self.assertTrue(local.called)
        provider_passed = local.call_args.args[2]
        self.assertIn(provider_passed, {"ollama", "lmstudio"})

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

    # NOTE: tests `test_prompt_assistant_run_parses_album_payload` and
    # `test_prompt_assistant_album_preserves_role_aware_track_durations` were
    # removed when the legacy single-Ollama-call album planner was deleted.
    # Album wizard fill now dispatches to the CrewAI Micro Tasks director
    # via `_run_prompt_assistant_album_crew` — covered by
    # `test_album_wizard_fill_dispatches_through_crewai_micro` above. The
    # `_normalize_prompt_assistant_payload` helpers below still cover the
    # role-aware duration + fixed-duration shape because they are unit-level
    # contracts independent of the wire protocol.

    def test_prompt_assistant_album_fills_missing_durations_from_role(self):
        payload, warnings = acejam_app._normalize_prompt_assistant_payload(
            "album",
            {
                "concept": "Role defaults",
                "duration_mode": "ai_per_track",
                "track_duration": 180,
                "tracks": [
                    {"track_number": 1, "title": "Door Opens", "role": "intro", "caption": "cinematic intro"},
                    {"track_number": 2, "title": "Big Hook", "role": "single", "caption": "radio single"},
                    {"track_number": 3, "title": "Credits", "role": "outro", "caption": "soft outro"},
                ],
            },
            {"planner_lm_provider": "ollama", "ollama_model": "llama3.2"},
        )

        self.assertEqual(warnings, [])
        self.assertEqual(payload["album_writer_mode"], "per_track_writer_loop")
        self.assertEqual([track["duration"] for track in payload["tracks"]], [90.0, 210.0, 90.0])
        self.assertEqual([track["duration_source"] for track in payload["tracks"]], ["role_default", "role_default", "role_default"])

    def test_prompt_assistant_album_fixed_duration_forces_all_tracks(self):
        payload, _warnings = acejam_app._normalize_prompt_assistant_payload(
            "album",
            {
                "concept": "Fixed duration",
                "duration_mode": "fixed",
                "track_duration": 180,
                "tracks": [
                    {"track_number": 1, "title": "Intro", "role": "intro", "duration": 75},
                    {"track_number": 2, "title": "Single", "role": "single", "duration": 230},
                ],
            },
            {"planner_lm_provider": "ollama", "ollama_model": "llama3.2"},
        )

        self.assertEqual(payload["duration_mode"], "fixed")
        self.assertEqual([track["duration"] for track in payload["tracks"]], [180.0, 180.0])

    # NOTE: `test_prompt_assistant_album_contract_repairs_returned_titles`
    # was removed with the legacy single-call album path. The CrewAI bridge
    # passes user paste content through `extract_user_album_contract` to
    # `plan_album` via `input_tracks`, where contract repair already runs in
    # the album crew (covered by tests in test_songwriting_toolkit.py).

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
