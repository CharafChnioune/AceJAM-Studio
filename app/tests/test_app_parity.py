import importlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import album_crew as album_crew_module
from fastapi.testclient import TestClient


os.environ.setdefault("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "1")
acejam_app = importlib.import_module("app")


# Shared long-form lyrics fixture for album/parity routing tests.
# Long enough to clear the boosted lyric_density_gate min_lines/min_words bar at
# 30s-60s durations after the 2026-04 lyric length plan revisions.
_LONG_TEST_LYRICS = (
    "[Intro]\nQuiet click before the lights catch on\n"
    "Bright route opens for the calm intake\n"
    "Steady pulse, every meter holds\n\n"
    "[Verse]\nWe test the bright route under city lights\n"
    "Every model enters clearly through the night\n"
    "Clean chords carry the signal so the morning sounds right\n"
    "Kick drum locks the cadence tight\n"
    "Piano flickers under city light\n"
    "Low end moves but the words stay clear\n"
    "Signal in front for every ear\n"
    "Hook coming forward like an honest heartbeat\n\n"
    "[Pre-Chorus]\nLights line up, cue the wave, cue the wave\n"
    "Every meter ready, give the mic a save\n"
    "Counters resolve and the verse takes shape\n\n"
    "[Chorus]\nEvery model plays it loud, every model rides the cloud\n"
    "Every take keeps timing proud, every voice declared aloud\n"
    "Unit Signal rides tonight, the hook lands clean and bright\n"
    "Hands come up when the chorus hits, clear lead cuts through the mix\n"
    "Every line lands right on time, no blurred words inside the rhyme\n"
    "Bright route open, clean and tight, every render lands just right\n\n"
    "[Verse 2]\nNight moves through the master, takes a steady breath\n"
    "Compressor holding gentle, never loses depth\n"
    "Mics all warm and honest, every tail decay\n"
    "Bright route polished smooth, never gets in the way\n"
    "Bus settles soft and even, headroom holds the seam\n"
    "Every register answering the dream\n\n"
    "[Chorus]\nEvery model plays it loud, every model rides the cloud\n"
    "Every take keeps timing proud, every voice declared aloud\n"
    "Unit Signal rides tonight, the hook lands clean and bright\n\n"
    "[Outro]\nThe final note stays clean and warm\n"
    "Seven paths land bright through the storm\n"
    "Bass rolls out and the voice stays near\n"
    "One last hook for the engineer\n"
    "Timing stays locked as the room lets go\n"
    "Last chord settle, last meter calm\n"
)


class AppParityTest(unittest.TestCase):
    def _mock_direct_album_plan(self, tracks):
        planned_tracks = []
        for index, item in enumerate(tracks, start=1):
            track = dict(item)
            track.setdefault("track_number", index)
            track.setdefault("caption", track.get("tags") or "clear pop vocal, crisp drums, radio mix")
            track.setdefault("description", track.get("caption") or track.get("tags") or "")
            track.setdefault("language", "en")
            track.setdefault("vocal_language", track.get("language") or "en")
            track.setdefault("duration", 30)
            track.setdefault("bpm", 95)
            track.setdefault("key_scale", "A minor")
            track.setdefault("time_signature", "4")
            track["agent_complete_payload"] = True
            track["agent_director_version"] = "unit-director"
            track["payload_gate_status"] = "pass"
            track["ace_lm_model"] = "none"
            track["allow_supplied_lyrics_lm"] = False
            track["thinking"] = False
            track["use_format"] = False
            track["use_cot_metas"] = False
            track["use_cot_caption"] = False
            track["use_cot_lyrics"] = False
            track["use_cot_language"] = False
            planned_tracks.append(track)

        return {
            "success": True,
            "tracks": planned_tracks,
            "logs": ["mock direct AceJAM Director plan"],
            "planning_engine": "acejam_agents",
            "custom_agents_used": True,
            "crewai_used": False,
            "toolbelt_fallback": False,
            "agent_debug_dir": "",
            "agent_rounds": [],
            "agent_repair_count": 0,
            "memory_enabled": False,
            "context_chunks": 0,
            "retrieval_rounds": 0,
            "input_contract_applied": False,
            "contract_repair_count": 0,
            "blocked_unsafe_count": 0,
            "toolkit_report": {},
        }

    def _write_ready_checkpoint(self, root: Path, name: str, weight_name: str = "model.safetensors") -> Path:
        path = root / "checkpoints" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / weight_name).write_bytes(b"weights")
        return path

    def _write_ready_diffusers_pipeline(self, root: Path, name: str = "acestep-v15-xl-turbo-diffusers") -> Path:
        path = root / "checkpoints" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "AceStepPipeline",
                    "condition_encoder": ["ace_step", "AceStepConditionEncoder"],
                    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                    "text_encoder": ["transformers", "Qwen3Model"],
                    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                    "transformer": ["diffusers", "AceStepTransformer1DModel"],
                    "vae": ["diffusers", "AutoencoderOobleck"],
                }
            ),
            encoding="utf-8",
        )
        for component in ["condition_encoder", "text_encoder", "transformer", "vae"]:
            component_path = path / component
            component_path.mkdir()
            (component_path / "config.json").write_text("{}", encoding="utf-8")
            (component_path / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        (path / "scheduler").mkdir()
        (path / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
        (path / "tokenizer").mkdir()
        (path / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
        return path

    def test_online_lyrics_title_candidates_cover_common_2pac_variants(self):
        candidates = acejam_app._lyrics_title_candidates("Bomb First (Intro)")
        self.assertIn("Bomb First My Second Reply", candidates)
        self.assertIn("Hellrazor", acejam_app._lyrics_title_candidates("Hell Razor"))
        self.assertIn("Nothing To Lose", acejam_app._lyrics_title_candidates("Nothin To Lose"))
        self.assertIn("California Love", acejam_app._lyrics_title_candidates("Califonia Love"))
        self.assertIn("Only Fear Of Death", acejam_app._lyrics_title_candidates("Only Fear Death"))
        self.assertIn("Whatz Ya Phone Number", acejam_app._lyrics_title_candidates("What'z Ya Phone Number"))

        self.assertEqual(
            acejam_app._genius_slug("2Pac", "To Live & Die In L.A."),
            "2pac-to-live-and-die-in-la",
        )
        self.assertEqual(
            acejam_app._strip_artist_prefix_from_title("2Pac", "2Pac - Ready 4 Whatever"),
            "Ready 4 Whatever",
        )

    def test_nested_generation_metadata_is_canonicalized(self):
        payload = acejam_app._merge_nested_generation_metadata(
            {
                "caption": "cinematic pop",
                "metas": {
                    "bpm": 122,
                    "duration": 180,
                    "keyscale": "D minor",
                    "timesignature": "4",
                    "language": "en",
                },
                "user_metadata": {"vocal_language": "fr"},
            }
        )

        self.assertEqual(payload["bpm"], 122)
        self.assertEqual(payload["duration"], 180)
        self.assertEqual(payload["key_scale"], "D minor")
        self.assertEqual(payload["time_signature"], "4")
        self.assertEqual(payload["vocal_language"], "en")

    def test_album_lyrics_cleaner_removes_section_timing_preamble(self):
        cleaned = acejam_app.strip_ace_step_lyrics_leakage(
            "[Intro] (0-8 sec)\n"
            "  - [Verse] (8-42 sec)\n"
            "  - [Chorus] (42-60 sec)\n\n"
            "Lyrics:\n"
            "Neon bakery lights keep calling us home.\n"
            "[Verse]\nThe ovens glow again\n"
            "[Chorus]\nNeon bakery lights keep calling us home.\n"
        )

        self.assertNotIn("(0-8 sec)", cleaned)
        self.assertNotIn("Lyrics:", cleaned)
        self.assertIn("[Verse]", cleaned)
        self.assertIn("Neon bakery lights keep calling us home.", cleaned)

    def test_album_lyrics_cleaner_removes_ace_step_timing_blocks(self):
        cleaned = acejam_app.strip_ace_step_lyrics_leakage(
            "[Verse]\nThe ovens glow again\n\n"
            "[ACE-Step]\n"
            "Tag: [Chorus]\n"
            "Start: 00:30\n"
            "End: 00:46\n"
            "Vocal Role: lyrics\n\n"
            "[Chorus]\nNeon bakery lights keep calling us home.\n"
        )

        self.assertNotIn("[ACE-Step]", cleaned)
        self.assertNotIn("Vocal Role:", cleaned)
        self.assertNotIn("Start:", cleaned)
        self.assertIn("[Chorus]", cleaned)
        self.assertIn("Neon bakery lights keep calling us home.", cleaned)

    def test_album_lyrics_cleaner_removes_bracketed_metadata_tail(self):
        cleaned = acejam_app.strip_ace_step_lyrics_leakage(
            "[Verse]\nThe ovens glow again\n"
            "[Chorus]\nNeon bakery lights keep calling us home.\n"
            "[ACE-Step metadata]\n"
            "tag_list: pop, funk, groovy\n"
            "[Intro]\nThis should not be kept\n"
        )

        self.assertNotIn("[ACE-Step metadata]", cleaned)
        self.assertNotIn("tag_list:", cleaned)
        self.assertNotIn("This should not be kept", cleaned)
        self.assertIn("Neon bakery lights keep calling us home.", cleaned)

    def test_album_lyrics_cleaner_removes_contract_metadata_sections(self):
        cleaned = acejam_app.strip_ace_step_lyrics_leakage(
            "[Final Chorus]\nWe built this thing from nothing at all\n"
            "[Producer Credit: Studio House]\n"
            "[Locked Title: Neon Bakery Lights]\n"
            "[Duration: 60.0 seconds]\n\n"
            "[Required phrases]\n"
            "Neon bakery lights keep calling us home.\n"
        )

        self.assertNotIn("Producer Credit", cleaned)
        self.assertNotIn("Locked Title", cleaned)
        self.assertNotIn("[Duration:", cleaned)
        self.assertNotIn("[Required phrases]", cleaned)
        self.assertIn("Neon bakery lights keep calling us home.", cleaned)

    def test_direct_album_agent_payload_rejects_metadata_in_caption(self):
        report = acejam_app._validate_direct_album_agent_payload(
            {
                "title": "Concrete Canyons",
                "producer_credit": "Dr. Dre",
                "caption": "West Coast rap, 95 BPM, A minor, 4/4 time, Dr. Dre production, polished mix",
                "lyrics": "[Verse]\nClean voices move\n[Chorus]\nClean hook returns\n[Outro]\nLights fade",
            }
        )

        self.assertFalse(report["gate_passed"])
        issue_ids = {item["id"] for item in report["issues"]}
        self.assertIn("metadata_or_credit_in_caption", issue_ids)

    def test_direct_album_agent_payload_rejects_underfilled_long_rap_lyrics(self):
        sections = ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]", "[Break]", "[Bridge]", "[Final Chorus]", "[Outro]"]
        lyrics = "\n".join(
            line
            for section in sections
            for line in (section, "Concrete shadows move")
        )

        report = acejam_app._validate_direct_album_agent_payload(
            {
                "title": "Concrete Canyons",
                "duration": 240,
                "caption": "cinematic West Coast rap, hip-hop drums, male rap lead, clear chorus, polished modern mix",
                "style": "West Coast rap",
                "lyrics": lyrics,
            }
        )

        self.assertFalse(report["gate_passed"])
        issue_ids = {item["id"] for item in report["issues"]}
        self.assertIn("lyrics_under_length", issue_ids)
        self.assertIn("lyrics_too_few_lines", issue_ids)
        self.assertGreaterEqual(report["lyric_duration_fit"]["min_words"], 340)

    def test_direct_album_agent_payload_rejects_rap_caption_without_rap_vocal_or_groove(self):
        lyrics = "\n".join(
            ["[Intro]"]
            + ["Concrete truth keeps knocking on the door" for _ in range(8)]
            + ["[Verse 1]"]
            + ["Every block remembers what the suits ignore" for _ in range(12)]
            + ["[Hook]"]
            + ["We keep the truth alive when pressure gets raw" for _ in range(6)]
            + ["[Outro]", "Concrete talks clear when the night gets low"]
        )

        report = acejam_app._validate_direct_album_agent_payload(
            {
                "title": "Concrete Canyons",
                "duration": 90,
                "caption": "cinematic orchestral strings, brass swells, taiko drums, epic score, polished mix",
                "style": "West Coast rap",
                "lyrics": lyrics,
            }
        )

        self.assertFalse(report["gate_passed"])
        issue_ids = {item["id"] for item in report["issues"]}
        self.assertIn("genre_intent_missing_rap_vocal", issue_ids)
        self.assertIn("genre_intent_missing_rap_groove", issue_ids)

    def test_unreleased_model_is_not_downloadable(self):
        self.assertNotIn("acestep-v15-turbo-rl", acejam_app._downloadable_model_names())

        status = acejam_app._model_runtime_status("acestep-v15-turbo-rl")

        self.assertEqual(status["status"], "unreleased")
        self.assertFalse(status["downloadable"])
        self.assertIn("unreleased", status["error"])

    def test_song_models_require_real_weights_and_shared_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = root / "checkpoints" / "acestep-v15-turbo"
            incomplete.mkdir(parents=True)
            (incomplete / "config.json").write_text("{}", encoding="utf-8")

            with patch.object(acejam_app, "MODEL_CACHE_DIR", root):
                self.assertFalse(acejam_app._checkpoint_dir_ready(incomplete))
                self.assertNotIn("acestep-v15-turbo", acejam_app._installed_acestep_models())

                self._write_ready_checkpoint(root, "acestep-v15-turbo")
                self.assertNotIn("acestep-v15-turbo", acejam_app._installed_acestep_models())

                self._write_ready_checkpoint(root, "vae", "diffusion_pytorch_model.safetensors")
                self._write_ready_checkpoint(root, "Qwen3-Embedding-0.6B")
                self.assertIn("acestep-v15-turbo", acejam_app._installed_acestep_models())

    def test_startup_skips_incomplete_checkpoint_without_transformers_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = root / "checkpoints" / "acestep-v15-turbo"
            incomplete.mkdir(parents=True)
            (incomplete / "config.json").write_text("{}", encoding="utf-8")

            with patch.object(acejam_app, "MODEL_CACHE_DIR", root), \
                patch.object(acejam_app.handler, "initialize_service", side_effect=AssertionError("should not load")):
                status, ready = acejam_app._initialize_acestep_handler("acestep-v15-turbo")

            self.assertFalse(ready)
            self.assertIn("not fully downloaded", status)
            self.assertIn("missing usable weight files", status)

    def test_helper_checkpoint_readiness_accepts_ckpt_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoints" / "ace-step-v1.5-1d-vae-stable-audio-format"
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "checkpoint.ckpt").write_bytes(b"weights")

            self.assertTrue(acejam_app._checkpoint_dir_ready(path))

    def test_diffusers_export_readiness_accepts_model_index_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_ready_diffusers_pipeline(root)

            with patch.object(acejam_app, "MODEL_CACHE_DIR", root):
                self.assertTrue(acejam_app._is_model_installed("acestep-v15-xl-turbo-diffusers"))
                self.assertEqual(acejam_app._checkpoint_status_reason(path), "ready")
                self.assertNotIn("acestep-v15-xl-turbo-diffusers", acejam_app._installed_acestep_models())

    def test_diffusers_export_reports_missing_pipeline_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_ready_diffusers_pipeline(root)
            (path / "transformer" / "diffusion_pytorch_model.safetensors").unlink()

            with patch.object(acejam_app, "MODEL_CACHE_DIR", root):
                self.assertFalse(acejam_app._is_model_installed("acestep-v15-xl-turbo-diffusers"))
                self.assertIn("transformer weights", acejam_app._checkpoint_status_reason(path))

    def test_official_parity_payload_exposes_wrappers_and_runtime(self):
        source_status = {
            "vendor_dir": str(acejam_app.OFFICIAL_ACE_STEP_DIR),
            "local_commit": "local-sha",
            "local_branch": "main",
            "dirty": False,
            "dirty_files": [],
            "remote_url": "https://github.com/ace-step/ACE-Step-1.5.git",
            "remote_main_head": "remote-sha",
            "behind_main": True,
            "status": "behind",
            "sync_confirm": acejam_app.ACE_STEP_VENDOR_SYNC_CONFIRM,
            "sync_mode": "patch-preserving manual endpoint",
            "update_note": "Status only.",
        }
        with patch.object(acejam_app, "_official_source_status", return_value=source_status):
            payload = acejam_app._official_parity_payload()

        self.assertTrue(payload["success"])
        self.assertIn("/v1/init", payload["manifest"]["api_endpoints"])
        self.assertIn("/v1/training/start_lokr", payload["manifest"]["api_endpoints"])
        self.assertIn("backend_hash", payload["manifest"]["runtime"])
        self.assertIn("api_key_enabled", payload["manifest"]["runtime"])
        self.assertEqual(payload["manifest"]["runtime"]["source_status"]["status"], "behind")
        self.assertTrue(payload["manifest"]["source_status"]["behind_main"])
        self.assertTrue(any("behind official main" in item for item in payload["manifest"]["recommended_actions"]))
        self.assertIn("quality_policy", payload["manifest"])
        self.assertIn("schema_parity", payload["manifest"])
        self.assertIn("settings_registry", payload["manifest"])
        self.assertIn("model_registry", payload["manifest"])
        self.assertIn("main", payload["manifest"]["model_registry"])
        self.assertIn("acestep-v15-turbo-shift1", payload["manifest"]["model_registry"])
        self.assertIn("acestep-v15-turbo-continuous", payload["manifest"]["model_registry"])
        self.assertIn("acestep-v15-xl-sft", payload["manifest"]["boot_quality_models"])
        self.assertIn("acestep-v15-sft", payload["manifest"]["boot_quality_models"])
        self.assertIn("acestep-captioner", payload["manifest"]["boot_quality_models"])
        self.assertNotIn("acestep-v15-xl-base", payload["manifest"]["boot_quality_models"])
        self.assertNotIn("acestep-5Hz-lm-4B", payload["manifest"]["boot_quality_models"])
        self.assertEqual(payload["manifest"]["model_registry"]["acestep-v15-turbo-rl"]["status"], "unreleased")
        self.assertIn("component_status", payload["manifest"]["core_bundle"])
        self.assertIn("boot_downloads", payload["manifest"]["runtime"])
        self.assertEqual(payload["manifest"]["settings_registry"]["version"], "ace-step-settings-parity-2026-04-26")
        self.assertEqual(payload["manifest"]["quality_policy"]["sft_base_models"]["inference_steps"], 50)
        self.assertEqual(payload["manifest"]["quality_policy"]["balanced_pro_models"]["inference_steps"], 50)
        self.assertEqual(payload["manifest"]["quality_policy"]["default_profile"], "chart_master")
        self.assertEqual(payload["manifest"]["quality_policy"]["turbo_models"]["inference_steps"], 8)

    def test_official_downloadable_names_include_helpers_and_new_render_models(self):
        downloads = acejam_app._downloadable_model_names()

        self.assertIn("main", downloads)
        self.assertIn("acestep-v15-turbo-shift1", downloads)
        self.assertIn("acestep-v15-turbo-continuous", downloads)
        self.assertIn("acestep-captioner", downloads)
        self.assertIn("acestep-transcriber", downloads)
        self.assertNotIn("acestep-v15-turbo-rl", downloads)

    def test_boot_download_bundle_queues_sft_models_and_helpers_without_ace_lm(self):
        names = acejam_app._boot_download_model_names()

        self.assertIn("acestep-v15-xl-sft", names)
        self.assertIn("acestep-v15-sft", names)
        self.assertIn("acestep-captioner", names)
        self.assertIn("acestep-transcriber", names)
        self.assertNotIn("main", names)
        self.assertNotIn("acestep-v15-xl-base", names)
        self.assertNotIn(acejam_app.ACE_LM_PREFERRED_MODEL, names)
        self.assertNotIn("acestep-5Hz-lm-1.7B", names)
        self.assertNotIn("acestep-5Hz-lm-0.6B", names)

        started: list[str] = []
        with patch.object(acejam_app, "ACEJAM_BOOT_DOWNLOAD_ENABLED", True), \
            patch.dict(os.environ, {"ACEJAM_SKIP_MODEL_INIT_FOR_TESTS": "0"}), \
            patch.object(acejam_app, "_is_model_installed", return_value=False), \
            patch.object(acejam_app, "_start_model_download", side_effect=lambda model: started.append(model) or {"model_name": model}):
            status = acejam_app._queue_boot_model_downloads()

        self.assertTrue(status["enabled"])
        self.assertEqual(status["queued"], names)
        self.assertEqual(started, names)

    def test_quality_profile_defaults_choose_highest_practical_models(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}):
            self.assertEqual(
                acejam_app._song_model_for_quality_profile(None, "chart_master", "text2music"),
                "acestep-v15-xl-sft",
            )
            self.assertEqual(
                acejam_app._song_model_for_quality_profile("auto", "chart_master", "extract"),
                "acestep-v15-xl-base",
            )
            self.assertEqual(
                acejam_app._song_model_for_quality_profile("auto", "chart_master", "lego"),
                "acestep-v15-xl-base",
            )

    def test_docs_daily_simple_defaults_use_turbo_and_auto_metadata(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-turbo"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", "acestep-5Hz-lm-1.7B"}):
            params = acejam_app._parse_generation_payload(
                {
                    "ui_mode": "simple",
                    "task_type": "text2music",
                    "song_model": "auto",
                    "caption": "modern pop, clean drums",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                }
            )

        self.assertEqual(params["quality_profile"], "docs_daily")
        self.assertEqual(params["song_model"], "acestep-v15-xl-turbo")
        self.assertEqual(params["inference_steps"], 8)
        self.assertEqual(params["guidance_scale"], 7.0)
        self.assertEqual(params["shift"], 3.0)
        self.assertEqual(params["audio_format"], "flac")
        self.assertEqual(params["duration"], -1.0)
        self.assertIsNone(params["bpm"])
        self.assertEqual(params["key_scale"], "")
        self.assertEqual(params["time_signature"], "")

    def test_metadata_locks_can_force_auto_even_with_concrete_values(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "caption": "bright pop, crisp drums",
            "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
            "duration": 180,
            "bpm": 120,
            "key_scale": "D minor",
            "time_signature": "4",
            "metadata_locks": {"duration": False, "bpm": False, "key_scale": False, "time_signature": False},
        }
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(payload)

        self.assertEqual(params["duration"], -1.0)
        self.assertIsNone(params["bpm"])
        self.assertEqual(params["key_scale"], "")
        self.assertEqual(params["time_signature"], "")

    def test_generation_prompt_normalizes_problematic_2pac_token_for_vocal_clarity(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "caption": "2pac, west coast rap, clear male rap vocal",
            "tags": "2Pac, hip hop",
            "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
        }
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(payload)

        self.assertIn("pac, west coast rap", params["caption"])
        self.assertNotIn("2pac", params["caption"].lower())
        self.assertIn("generation_prompt_token_2pac_normalized_to_pac_for_vocal_clarity", params["payload_warnings"])

    def test_parse_generation_payload_applies_rap_style_profile_to_caption_and_lyrics(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "style_profile": "rap",
            "caption": "west coast night drive",
            "tags": "hard drums",
            "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
        }
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(payload)

        self.assertEqual(params["style_profile"], "rap")
        self.assertIn("rap", params["caption"].lower())
        self.assertIn("hip hop", params["caption"].lower())
        self.assertIn("[Verse - rap, rhythmic spoken flow]", params["lyrics"])
        self.assertIn("[Chorus - rap hook]", params["lyrics"])
        self.assertEqual(params["style_conditioning_audit"]["status"], "pass")

    def test_audio_wizards_send_style_profile(self):
        for path in [
            "web/src/wizards/SimpleWizard.tsx",
            "web/src/wizards/CustomWizard.tsx",
            "web/src/wizards/SourceAudioWizard.tsx",
            "web/src/wizards/AlbumWizard.tsx",
            "web/src/wizards/NewsWizard.tsx",
        ]:
            text = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
            self.assertIn("AudioStyleSelector", text, path)
            self.assertIn("style_profile", text, path)
            self.assertIn("AudioBackendSelector", text, path)
            self.assertIn("audio_backend", text, path)
            self.assertIn("use_mlx_dit", text, path)

    def test_audio_backend_defaults_to_mps_torch_and_allows_explicit_mlx(self):
        with patch.object(acejam_app, "_IS_APPLE_SILICON", True), \
            patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            defaulted = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "rap, hard drums",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "duration": 30,
                }
            )
            explicit_mlx = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "rap, hard drums",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "duration": 30,
                    "audio_backend": "mlx",
                }
            )

        self.assertEqual(defaulted["audio_backend"], "mps_torch")
        self.assertFalse(defaulted["use_mlx_dit"])
        self.assertEqual(defaulted["device"], "mps")
        self.assertEqual(defaulted["dtype"], "float32")
        self.assertEqual(explicit_mlx["audio_backend"], "mlx")
        self.assertTrue(explicit_mlx["use_mlx_dit"])

    def test_official_request_forces_mlx_backend_even_with_stale_flag(self):
        with patch.object(acejam_app, "_IS_APPLE_SILICON", True), \
            patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "rap, hard drums",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "duration": 30,
                    "audio_backend": "mlx",
                    "use_mlx_dit": False,
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))

        self.assertEqual(request["audio_backend"], "mlx")
        self.assertTrue(request["use_mlx_dit"])
        self.assertEqual(request["requested_audio_backend"], "mlx")
        self.assertTrue(request["requested_use_mlx_dit"])
        self.assertEqual(request["audio_backend_contract"]["enforced_at"], "official_request")

    def test_mlx_runner_response_must_confirm_active_mlx(self):
        request = {"audio_backend": "mlx", "use_mlx_dit": True}

        with tempfile.TemporaryDirectory() as tmp, \
            patch.object(acejam_app, "OFFICIAL_ACE_STEP_DIR", Path(tmp)), \
            patch.object(acejam_app, "OFFICIAL_RUNNER_SCRIPT", Path(tmp) / "runner.py"), \
            patch.object(acejam_app, "_official_runner_timeout_seconds", return_value=5), \
            patch.object(acejam_app, "_official_service_generation_timeout_seconds", return_value=5), \
            patch.object(acejam_app, "subprocess") as subprocess_mock:
            runner = Path(tmp) / "runner.py"
            runner.write_text("# runner", encoding="utf-8")

            class DummyPipe:
                def readline(self):
                    return ""

                def close(self):
                    return None

            class DummyProcess:
                stdout = DummyPipe()
                stderr = DummyPipe()

                def wait(self, timeout=None):
                    response_path = Path(subprocess_mock.Popen.call_args.args[0][3])
                    response_path.write_text(
                        json.dumps({"success": True, "audio_backend_status": {"effective_mlx_dit_active": False}}),
                        encoding="utf-8",
                    )
                    return 0

            subprocess_mock.Popen.return_value = DummyProcess()

            with self.assertRaisesRegex(RuntimeError, "did not activate MLX DiT"):
                acejam_app._run_official_runner_request(request, Path(tmp) / "work")

    def test_query_result_returns_acejam_result_for_ui_rendering(self):
        task_id = "unit-query-result"
        result = {
            "success": True,
            "runner": "official",
            "result_id": "result-query",
            "params": {"caption": "bright pop", "lyrics": "[Verse]\nLine", "bpm": None, "duration": -1, "key_scale": "", "time_signature": ""},
            "audios": [
                {
                    "id": "take-1",
                    "result_id": "result-query",
                    "audio_url": "/media/results/result-query/take.wav",
                    "download_url": "/media/results/result-query/take.wav",
                    "seed": "123",
                }
            ],
        }
        with acejam_app._api_generation_tasks_lock:
            acejam_app._api_generation_tasks.pop(task_id, None)
        acejam_app._set_api_generation_task(task_id, status=1, state="succeeded", result=result)

        item = acejam_app._official_query_item(task_id)

        self.assertEqual(item["status"], 1)
        self.assertEqual(item["acejam_result"]["result_id"], "result-query")
        self.assertEqual(item["acejam_result"]["audios"][0]["audio_url"], "/media/results/result-query/take.wav")

    def test_new_expert_fields_reach_official_request_payload(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "West Coast rap, tight drums, 808 bass, talkbox lead, male rap vocal, polished loud mix",
                    "lyrics": "[Verse]\nConcrete lines land clean\n\n[Chorus]\nThe hook comes back clear",
                    "duration": 30,
                    "audio_format": "wav32",
                    "dcw_enabled": False,
                    "dcw_mode": "low",
                    "dcw_scaler": 0.07,
                    "dcw_high_scaler": 0.03,
                    "dcw_wavelet": "db4",
                    "retake_seed": "321",
                    "retake_variance": 0.25,
                    "flow_edit_morph": True,
                    "flow_edit_source_caption": "source vocal pocket",
                    "flow_edit_source_lyrics": "[Verse]\nsource line",
                    "flow_edit_n_min": 0.8,
                    "flow_edit_n_max": 0.2,
                    "flow_edit_n_avg": 4,
                    "analysis_only": True,
                    "full_analysis_only": False,
                    "extract_codes_only": True,
                    "use_tiled_decode": False,
                    "is_format_caption": True,
                    "track_name": "vocals",
                    "track_classes": ["vocals", "drums"],
                    "song_intent": {"genre_family": "rap", "caption": "structured intent caption"},
                    "source_task_intent": "clean source vocal before morph",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))

        official_params = request["params"]
        self.assertEqual(request["audio_backend"], "mps_torch")
        self.assertFalse(request["use_mlx_dit"])
        self.assertFalse(official_params["dcw_enabled"])
        self.assertEqual(official_params["dcw_mode"], "low")
        self.assertEqual(official_params["dcw_scaler"], 0.07)
        self.assertEqual(official_params["dcw_high_scaler"], 0.03)
        self.assertEqual(official_params["dcw_wavelet"], "db4")
        self.assertEqual(official_params["retake_seed"], "321")
        self.assertEqual(official_params["retake_variance"], 0.25)
        self.assertTrue(official_params["flow_edit_morph"])
        self.assertEqual(official_params["flow_edit_source_caption"], "source vocal pocket")
        self.assertEqual(official_params["flow_edit_source_lyrics"], "[Verse]\nsource line")
        self.assertEqual(official_params["flow_edit_n_min"], 0.2)
        self.assertEqual(official_params["flow_edit_n_max"], 0.8)
        self.assertEqual(official_params["flow_edit_n_avg"], 4)
        self.assertNotIn("analysis_only", official_params)
        self.assertEqual(request["official_api_fields"]["analysis_only"], True)
        self.assertEqual(request["official_api_fields"]["extract_codes_only"], True)
        self.assertEqual(request["official_api_fields"]["use_tiled_decode"], False)
        self.assertEqual(request["official_api_fields"]["is_format_caption"], True)
        self.assertEqual(request["official_api_fields"]["track_name"], "vocals")
        self.assertEqual(request["official_api_fields"]["track_classes"], ["vocals", "drums"])
        self.assertEqual(params["song_intent"]["genre_family"], "rap")
        self.assertEqual(params["source_task_intent"], "clean source vocal before morph")

    def test_official_response_wrapper_shape(self):
        wrapped = acejam_app._official_api_response({"ok": True})

        self.assertEqual(wrapped["code"], 200)
        self.assertIsNone(wrapped["error"])
        self.assertEqual(wrapped["data"], {"ok": True})
        self.assertIn("timestamp", wrapped)

    def test_config_and_toolkit_expose_song_intent_schema(self):
        client = TestClient(acejam_app.app)

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft", "acestep-v15-xl-base"}):
            config_response = client.get("/api/config")
            toolkit_response = client.get("/api/songwriting_toolkit")

        self.assertEqual(config_response.status_code, 200)
        self.assertEqual(toolkit_response.status_code, 200)
        for payload in [config_response.json()["songwriting_toolkit"], toolkit_response.json()]:
            schema = payload["song_intent_schema"]
            self.assertEqual(schema["counts"]["genre_modules"], 26)
            self.assertGreaterEqual(schema["counts"]["tag_taxonomy_terms"], 300)
            self.assertGreaterEqual(schema["counts"]["lyric_meta_tags"], 60)
            self.assertEqual(schema["counts"]["valid_languages"], 51)
            self.assertEqual(schema["counts"]["track_stems"], 12)
            self.assertIn("cover-nofsq", schema["capabilities"]["all_task_modes"])
            self.assertIn("acestep-v15-xl-base", schema["capabilities"]["model_support"]["complete"])

    # Removed: test_ai_fill_hydrates_song_intent_builder_chip_groups asserted
    # specific JS strings in app/index.html, which was removed in v0.2 when the
    # React + shadcn UI replaced the legacy Python SPA. Equivalent hydration
    # logic now lives in app/web/src/components/wizard/AIPromptStep.tsx and is
    # validated via the React build's TypeScript checks plus Playwright smoke
    # tests against /api/prompt-assistant/run.

    def test_official_runner_timeout_expands_for_long_mlx_generations(self):
        request_payload = {
            "audio_backend": "mlx",
            "use_mlx_dit": True,
            "lm_backend": "mlx",
            "params": {"duration": 365, "inference_steps": 64},
            "config": {"batch_size": 3},
        }

        timeout = acejam_app._official_runner_timeout_seconds(request_payload, requested_timeout=3600)

        self.assertGreater(timeout, 7200)
        self.assertLessEqual(timeout, acejam_app.ACEJAM_OFFICIAL_RUNNER_MAX_TIMEOUT_SECONDS)
        self.assertGreaterEqual(acejam_app.ACEJAM_GENERATE_ADVANCED_TIME_LIMIT_SECONDS, acejam_app.ACEJAM_OFFICIAL_RUNNER_TIMEOUT_SECONDS)

    def test_official_service_timeout_tracks_long_xl_sft_runner_timeout(self):
        request_payload = {
            "lm_backend": "mlx",
            "song_model": "acestep-v15-xl-sft",
            "params": {"duration": 180, "inference_steps": 50},
            "config": {"batch_size": 1},
        }

        runner_timeout = acejam_app._official_runner_timeout_seconds(request_payload, requested_timeout=3600)
        service_timeout = acejam_app._official_service_generation_timeout_seconds(request_payload, runner_timeout)

        self.assertGreater(service_timeout, 600)
        self.assertLess(service_timeout, runner_timeout)

    def test_official_memory_plan_serializes_xl_sft_takes_on_mps(self):
        payload = {
            "song_model": "acestep-v15-xl-sft",
            "task_type": "text2music",
            "device": "mps",
            "batch_size": 3,
            "duration": 180,
            "inference_steps": 50,
            "shift": 1.0,
            "use_lora": True,
            "lora_scale": 0.15,
            "seed": "42",
            "use_random_seed": False,
        }

        with patch.object(acejam_app, "_IS_APPLE_SILICON", True):
            plan = acejam_app._official_generation_memory_plan(payload)

        self.assertTrue(plan["force_runner_batch_size_one"])
        self.assertTrue(plan["sequential"])
        self.assertEqual(plan["requested_take_count"], 3)
        self.assertEqual(plan["actual_runner_batch_size"], 1)
        self.assertEqual(plan["render_pass_count"], 3)
        self.assertEqual(
            [acejam_app._official_take_params(payload, plan, index)["seed"] for index in range(3)],
            ["42", "43", "44"],
        )
        self.assertTrue(all(acejam_app._official_take_params(payload, plan, index)["batch_size"] == 1 for index in range(3)))

    def test_official_memory_plan_keeps_turbo_batching_policy(self):
        payload = {
            "song_model": "acestep-v15-turbo",
            "task_type": "text2music",
            "device": "mps",
            "batch_size": 3,
            "duration": 180,
            "inference_steps": 8,
            "shift": 3.0,
            "seed": "42",
        }

        with patch.object(acejam_app, "_IS_APPLE_SILICON", True):
            plan = acejam_app._official_generation_memory_plan(payload)

        self.assertFalse(plan["force_runner_batch_size_one"])
        self.assertFalse(plan["sequential"])
        self.assertEqual(plan["requested_take_count"], 3)
        self.assertEqual(plan["actual_runner_batch_size"], 3)
        self.assertEqual(plan["render_pass_count"], 1)

    def test_memory_error_gate_blocks_recommended_take(self):
        result = {
            "success": True,
            "error_type": "memory_error",
            "recommended_take": {"id": "take-1"},
            "audios": [{"id": "take-1", "is_recommended_take": True}],
        }

        gate = acejam_app._apply_vocal_intelligibility_gate_to_result(
            result,
            {"lyrics": "[Verse]\nClear words", "caption": "rap vocal", "instrumental": False},
            attempt=1,
            max_attempts=1,
        )

        self.assertEqual(gate["status"], "error")
        self.assertEqual(gate["reason"], "memory_error_before_audio")
        self.assertFalse(result["success"])
        self.assertTrue(result["needs_review"])
        self.assertNotIn("recommended_take", result)
        self.assertFalse(result["audios"][0]["is_recommended_take"])

    def test_timeout_error_gate_blocks_recommended_take(self):
        result = {
            "success": True,
            "error_type": "timeout_error",
            "recommended_take": {"id": "take-1"},
            "audios": [{"id": "take-1", "is_recommended_take": True}],
        }

        gate = acejam_app._apply_vocal_intelligibility_gate_to_result(
            result,
            {"lyrics": "[Verse]\nClear words", "caption": "rap vocal", "instrumental": False},
            attempt=1,
            max_attempts=1,
        )

        self.assertEqual(gate["status"], "error")
        self.assertEqual(gate["reason"], "timeout_before_audio")
        self.assertFalse(result["success"])
        self.assertTrue(result["needs_review"])
        self.assertNotIn("recommended_take", result)
        self.assertFalse(result["audios"][0]["is_recommended_take"])

    def test_audio_generation_memory_cleanup_unloads_local_llms(self):
        with patch.object(acejam_app, "_unload_llm_models_for_generation") as unload, \
            patch.object(acejam_app, "_cleanup_accelerator_memory") as cleanup, \
            patch.object(acejam_app.gc, "collect") as collect, \
            patch.object(acejam_app, "_mps_memory_snapshot", side_effect=lambda label: {"label": label}):
            event = acejam_app._prepare_audio_generation_memory("unit", release_handler=False)

        unload.assert_called_once()
        collect.assert_called_once()
        cleanup.assert_called_once()
        self.assertEqual(event["before"]["label"], "unit:before_cleanup")
        self.assertEqual(event["after"]["label"], "unit:after_cleanup")

    def test_react_ui_surfaces_takes_and_memory_policy(self):
        web_src = Path(acejam_app.BASE_DIR) / "web" / "src"
        custom = (web_src / "wizards" / "CustomWizard.tsx").read_text(encoding="utf-8")
        tracker = (web_src / "components" / "JobTracker.tsx").read_text(encoding="utf-8")

        self.assertIn("Aantal takes", custom)
        self.assertIn("XL-SFT/Base rendert takes een voor een op MPS", custom)
        self.assertIn('label: "Takes"', custom)
        self.assertIn("Requested takes", tracker)
        self.assertIn("Runner batch", tracker)
        self.assertIn("Memory policy", tracker)
        self.assertIn("Volledige tracks klaar", tracker)
        self.assertIn("Elke kaart hieronder is een volledig gerenderde track/take", tracker)

    def test_community_endpoint_refreshes_library_from_disk(self):
        client = TestClient(acejam_app.app)
        original_feed = list(acejam_app._feed_songs)
        with tempfile.TemporaryDirectory() as tmp:
            songs_dir = Path(tmp) / "songs"
            song_dir = songs_dir / "song123"
            song_dir.mkdir(parents=True)
            (song_dir / "take.wav").write_bytes(b"RIFF0000WAVE")
            (song_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "id": "song123",
                        "title": "Disk Refresh",
                        "artist_name": "AceJAM",
                        "audio_file": "take.wav",
                        "created_at": "2026-05-03T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            try:
                acejam_app._feed_songs.clear()
                with patch.object(acejam_app, "SONGS_DIR", songs_dir):
                    response = client.get("/api/community")
            finally:
                acejam_app._feed_songs[:] = original_feed

        self.assertEqual(response.status_code, 200)
        songs = response.json()
        self.assertEqual(songs[0]["id"], "song123")
        self.assertEqual(songs[0]["audio_url"], "/media/songs/song123/take.wav")

    def test_library_endpoint_shows_result_wavs_when_songs_empty(self):
        client = TestClient(acejam_app.app)
        original_feed = list(acejam_app._feed_songs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            songs = root / "songs"
            results = root / "results"
            result_dir = results / "result123"
            songs.mkdir()
            result_dir.mkdir(parents=True)
            (result_dir / "take.wav").write_bytes(b"RIFF0000WAVE")
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "id": "result123",
                        "title": "Result Only",
                        "artist_name": "MLX Media",
                        "created_at": "2026-05-07T00:00:00+00:00",
                        "audios": [{"id": "take-1", "filename": "take.wav", "audio_url": "/media/results/result123/take.wav"}],
                    }
                ),
                encoding="utf-8",
            )
            try:
                acejam_app._feed_songs.clear()
                with patch.object(acejam_app, "SONGS_DIR", songs), patch.object(acejam_app, "RESULTS_DIR", results):
                    response = client.get("/api/library")
            finally:
                acejam_app._feed_songs[:] = original_feed

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["counts"]["results"], 1)
        self.assertEqual(data["items"][0]["source"], "result")
        self.assertEqual(data["items"][0]["audio_url"], "/media/results/result123/take.wav")

    def test_library_endpoint_dedupes_result_audio_with_saved_song(self):
        client = TestClient(acejam_app.app)
        original_feed = list(acejam_app._feed_songs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            songs = root / "songs"
            results = root / "results"
            song_dir = songs / "song123"
            result_dir = results / "result123"
            song_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            (song_dir / "take.wav").write_bytes(b"RIFF0000WAVE")
            (song_dir / "meta.json").write_text(
                json.dumps({"id": "song123", "title": "Saved", "audio_file": "take.wav", "result_id": "result123"}),
                encoding="utf-8",
            )
            (result_dir / "take.wav").write_bytes(b"RIFF0000WAVE")
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "id": "result123",
                        "title": "Saved",
                        "audios": [{"id": "take-1", "filename": "take.wav", "song_id": "song123"}],
                    }
                ),
                encoding="utf-8",
            )
            try:
                acejam_app._feed_songs.clear()
                with patch.object(acejam_app, "SONGS_DIR", songs), patch.object(acejam_app, "RESULTS_DIR", results):
                    response = client.get("/api/library")
            finally:
                acejam_app._feed_songs[:] = original_feed

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["counts"]["songs"], 1)
        self.assertEqual(data["counts"]["results"], 0)
        self.assertEqual(data["items"][0]["source"], "song")

    def test_library_endpoint_shows_mlx_video_results(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            video_results = Path(tmp) / "video-results"
            result_dir = video_results / "vid1234"
            result_dir.mkdir(parents=True)
            (result_dir / "clip-source-audio.mp4").write_bytes(b"mp4")
            (result_dir / "mlx_video_result.json").write_text(
                json.dumps(
                    {
                        "result_id": "vid1234",
                        "filename": "clip-source-audio.mp4",
                        "primary_video_url": "/media/mlx-video/vid1234/clip-source-audio.mp4",
                        "raw_video_url": "/media/mlx-video/vid1234/clip.mp4",
                        "model_label": "LTX Fast Draft",
                        "prompt": "cinematic street clip",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(acejam_app, "MLX_VIDEO_RESULTS_DIR", video_results):
                response = client.get("/api/library")

        self.assertEqual(response.status_code, 200)
        video = next(item for item in response.json()["items"] if item["kind"] == "video")
        self.assertEqual(video["video_url"], "/media/mlx-video/vid1234/clip-source-audio.mp4")
        self.assertEqual(video["model_label"], "LTX Fast Draft")

    def test_library_endpoint_shows_mflux_image_results(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            image_results = Path(tmp) / "mflux-results"
            result_dir = image_results / "img1234"
            result_dir.mkdir(parents=True)
            (result_dir / "cover.png").write_bytes(b"png")
            (result_dir / "mflux_result.json").write_text(
                json.dumps(
                    {
                        "result_id": "img1234",
                        "filename": "cover.png",
                        "image_url": "/media/mflux/img1234/cover.png",
                        "model_label": "Qwen Image",
                        "prompt": "premium album art",
                        "width": 1024,
                        "height": 1024,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(acejam_app, "MFLUX_RESULTS_DIR", image_results):
                response = client.get("/api/library")

        self.assertEqual(response.status_code, 200)
        image = next(item for item in response.json()["items"] if item["kind"] == "image")
        self.assertEqual(image["image_url"], "/media/mflux/img1234/cover.png")
        self.assertEqual(image["model_label"], "Qwen Image")
        self.assertEqual(response.json()["counts"]["images"], 1)

    def test_library_delete_result_take_updates_metadata(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            result_dir = results / "result123"
            result_dir.mkdir(parents=True)
            (result_dir / "one.wav").write_bytes(b"one")
            (result_dir / "two.wav").write_bytes(b"two")
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "id": "result123",
                        "audios": [
                            {"id": "take-1", "filename": "one.wav"},
                            {"id": "take-2", "filename": "two.wav"},
                        ],
                        "recommended_take": {"audio_id": "take-1"},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(acejam_app, "RESULTS_DIR", results):
                response = client.post(
                    "/api/library/delete",
                    json={"kind": "result-audio", "result_id": "result123", "audio_id": "take-1", "confirm": "DELETE"},
                )

            saved = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
            one_exists = (result_dir / "one.wav").exists()
            two_exists = (result_dir / "two.wav").exists()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(one_exists)
        self.assertTrue(two_exists)
        self.assertEqual([item["id"] for item in saved["audios"]], ["take-2"])
        self.assertNotIn("recommended_take", saved)

    def test_library_delete_last_result_take_removes_result_folder(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            result_dir = results / "result123"
            result_dir.mkdir(parents=True)
            (result_dir / "one.wav").write_bytes(b"one")
            (result_dir / "result.json").write_text(
                json.dumps({"id": "result123", "audios": [{"id": "take-1", "filename": "one.wav"}]}),
                encoding="utf-8",
            )
            with patch.object(acejam_app, "RESULTS_DIR", results):
                response = client.post(
                    "/api/library/delete",
                    json={"kind": "result-audio", "result_id": "result123", "audio_id": "take-1", "confirm": "DELETE"},
                )
            result_dir_exists = result_dir.exists()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(result_dir_exists)

    def test_library_delete_video_removes_result_job_and_attachment(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_results = root / "video-results"
            jobs = root / "jobs"
            attachments = root / "attachments.json"
            result_dir = video_results / "vid1234"
            result_dir.mkdir(parents=True)
            jobs.mkdir()
            (result_dir / "clip.mp4").write_bytes(b"mp4")
            (result_dir / "mlx_video_result.json").write_text(json.dumps({"result_id": "vid1234", "filename": "clip.mp4"}), encoding="utf-8")
            (jobs / "job123.json").write_text(json.dumps({"id": "job123", "result_id": "vid1234"}), encoding="utf-8")
            attachments.write_text(json.dumps([{"target_type": "song", "target_id": "song123", "result_id": "vid1234"}]), encoding="utf-8")
            with patch.object(acejam_app, "MLX_VIDEO_RESULTS_DIR", video_results), \
                patch.object(acejam_app, "MLX_VIDEO_JOBS_DIR", jobs), \
                patch.object(acejam_app, "MLX_VIDEO_ATTACHMENTS_PATH", attachments):
                response = client.post(
                    "/api/library/delete",
                    json={"kind": "video", "result_id": "vid1234", "confirm": "DELETE"},
                )
            result_dir_exists = result_dir.exists()
            job_exists = (jobs / "job123.json").exists()
            saved_attachments = json.loads(attachments.read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(result_dir_exists)
        self.assertFalse(job_exists)
        self.assertEqual(saved_attachments, [])

    def test_library_delete_image_removes_mflux_result_and_job(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_results = root / "mflux-results"
            jobs = root / "jobs"
            result_dir = image_results / "img1234"
            result_dir.mkdir(parents=True)
            jobs.mkdir()
            (result_dir / "cover.png").write_bytes(b"png")
            (result_dir / "mflux_result.json").write_text(json.dumps({"result_id": "img1234", "filename": "cover.png"}), encoding="utf-8")
            (jobs / "job123.json").write_text(json.dumps({"id": "job123", "result_id": "img1234"}), encoding="utf-8")
            with patch.object(acejam_app, "MFLUX_RESULTS_DIR", image_results):
                response = client.post(
                    "/api/library/delete",
                    json={"kind": "image", "result_id": "img1234", "confirm": "DELETE"},
                )
            result_dir_exists = result_dir.exists()
            job_exists = (jobs / "job123.json").exists()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(result_dir_exists)
        self.assertFalse(job_exists)

    def test_library_bulk_delete_removes_audio_image_and_video(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            image_results = root / "mflux-results"
            video_results = root / "video-results"
            video_jobs = root / "video-jobs"
            video_attachments = root / "video-attachments.json"
            audio_dir = results / "res123"
            image_dir = image_results / "img123"
            video_dir = video_results / "vid123"
            audio_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)
            video_jobs.mkdir()
            (audio_dir / "take.wav").write_bytes(b"audio")
            (audio_dir / "result.json").write_text(
                json.dumps({"id": "res123", "audios": [{"id": "take-1", "filename": "take.wav"}]}),
                encoding="utf-8",
            )
            (image_dir / "cover.png").write_bytes(b"png")
            (image_dir / "mflux_result.json").write_text(json.dumps({"result_id": "img123", "filename": "cover.png"}), encoding="utf-8")
            (video_dir / "clip.mp4").write_bytes(b"mp4")
            (video_dir / "mlx_video_result.json").write_text(json.dumps({"result_id": "vid123", "filename": "clip.mp4"}), encoding="utf-8")
            (video_jobs / "job.json").write_text(json.dumps({"result_id": "vid123"}), encoding="utf-8")
            video_attachments.write_text(json.dumps([{"result_id": "vid123", "target_id": "res123"}]), encoding="utf-8")

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "MFLUX_RESULTS_DIR", image_results), \
                patch.object(acejam_app, "MLX_VIDEO_RESULTS_DIR", video_results), \
                patch.object(acejam_app, "MLX_VIDEO_JOBS_DIR", video_jobs), \
                patch.object(acejam_app, "MLX_VIDEO_ATTACHMENTS_PATH", video_attachments):
                response = client.post(
                    "/api/library/delete",
                    json={
                        "confirm": "DELETE",
                        "items": [
                            {"kind": "result-audio", "result_id": "res123", "audio_id": "take-1"},
                            {"kind": "image", "result_id": "img123"},
                            {"kind": "video", "result_id": "vid123"},
                        ],
                    },
                )

            data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["success"])
        self.assertEqual(data["deleted"]["results"], 1)
        self.assertEqual(data["deleted"]["images"], 1)
        self.assertEqual(data["deleted"]["videos"], 1)
        self.assertFalse(audio_dir.exists())
        self.assertFalse(image_dir.exists())
        self.assertFalse(video_dir.exists())

    def test_library_delete_rejects_path_traversal_filename(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            result_dir = results / "result123"
            result_dir.mkdir(parents=True)
            (result_dir / "result.json").write_text(json.dumps({"id": "result123", "audios": []}), encoding="utf-8")
            with patch.object(acejam_app, "RESULTS_DIR", results):
                response = client.post(
                    "/api/library/delete",
                    json={"kind": "result-audio", "result_id": "result123", "filename": "../evil.wav", "confirm": "DELETE"},
                )

        self.assertEqual(response.status_code, 400)

    # Removed: test_results_show_saved_library_link asserted strings inside the
    # legacy Python SPA (app/index.html) that no longer exists post v0.2. The
    # React Library page (app/web/src/pages/Library.tsx) covers this flow.

    def test_generate_advanced_http_uses_worker_thread_and_background_cleanup(self):
        client = TestClient(acejam_app.app)
        to_thread_calls = []

        async def fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append(func)
            return func(*args, **kwargs)

        with patch.object(acejam_app.asyncio, "to_thread", new=fake_to_thread), \
            patch.object(acejam_app, "_run_advanced_generation", return_value={"success": True, "audios": []}) as run, \
            patch.object(acejam_app, "_schedule_accelerator_cleanup") as cleanup:
            response = client.post("/api/generate_advanced", json={"title": "Threaded"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(to_thread_calls, [run])
        run.assert_called_once()
        cleanup.assert_called_once_with("api_generate_advanced")

    def test_generate_portfolio_http_uses_worker_thread_and_background_cleanup(self):
        client = TestClient(acejam_app.app)
        to_thread_calls = []

        async def fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append(func)
            return func(*args, **kwargs)

        with patch.object(acejam_app.asyncio, "to_thread", new=fake_to_thread), \
            patch.object(acejam_app, "_run_model_portfolio_generation", return_value={"success": True, "audios": []}) as run, \
            patch.object(acejam_app, "_schedule_accelerator_cleanup") as cleanup:
            response = client.post("/api/generate_portfolio", json={"title": "Threaded"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(to_thread_calls, [run])
        run.assert_called_once()
        cleanup.assert_called_once_with("api_generate_portfolio")

    def test_official_api_key_accepts_body_token(self):
        client = TestClient(acejam_app.app)
        previous = os.environ.get("ACESTEP_API_KEY")
        os.environ["ACESTEP_API_KEY"] = "unit-secret"
        try:
            missing = client.post("/create_random_sample", json={"sample_type": "simple_mode"})
            allowed = client.post("/create_random_sample", json={"sample_type": "simple_mode", "ai_token": "unit-secret"})
        finally:
            if previous is None:
                os.environ.pop("ACESTEP_API_KEY", None)
            else:
                os.environ["ACESTEP_API_KEY"] = previous

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json()["code"], 200)

    def test_studio_generation_keeps_ace_lm_off_even_when_requested(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-base",
            "caption": "cinematic pop, live drums, rich vocal chain",
            "lyrics": "[Verse]\nThe city hums tonight\n[Chorus]\nWe live, we burn, we shine",
            "ace_lm_model": "acestep-5Hz-lm-4B",
            "thinking": True,
            "use_format": True,
            "sample_query": "make this a polished full song",
            "planner_ollama_model": "llama3.1:8b",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-base"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", "acestep-5Hz-lm-4B"}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["ace_lm_model"], "none")
        self.assertEqual(normalized["planner_lm_provider"], "ollama")
        self.assertEqual(normalized["planner_ollama_model"], "llama3.1:8b")
        self.assertFalse(normalized["thinking"])
        self.assertFalse(normalized["use_format"])
        self.assertEqual(normalized["sample_query"], "")
        self.assertTrue(normalized["sample_mode"] is False)
        self.assertEqual(normalized["inference_steps"], 50)

    def test_planner_settings_stay_separate_from_ace_step_lm_controls(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-base",
            "caption": "cinematic pop, live drums, rich vocal chain",
            "lyrics": "[Verse]\nLine one\n[Chorus]\nHook",
            "planner_lm_provider": "lmstudio",
            "planner_model": "qwen-local",
            "planner_temperature": 0.9,
            "planner_top_p": 0.97,
            "planner_top_k": 90,
            "planner_repeat_penalty": 1.04,
            "planner_seed": "777",
            "planner_max_tokens": 4096,
            "planner_context_length": 12288,
            "planner_timeout": 240,
            "lm_temperature": 0.55,
            "lm_top_p": 0.7,
            "lm_top_k": 12,
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-base"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["planner_lm_provider"], "lmstudio")
        self.assertEqual(normalized["planner_model"], "qwen-local")
        self.assertEqual(normalized["planner_temperature"], 0.9)
        self.assertEqual(normalized["planner_top_p"], 0.97)
        self.assertEqual(normalized["planner_top_k"], 90)
        self.assertEqual(normalized["planner_repeat_penalty"], 1.04)
        self.assertEqual(normalized["planner_seed"], 777)
        self.assertEqual(normalized["planner_max_tokens"], 4096)
        self.assertEqual(normalized["planner_context_length"], 12288)
        self.assertEqual(normalized["planner_timeout"], 240.0)
        self.assertEqual(normalized["ace_lm_model"], "none")

    def test_album_options_include_planner_settings(self):
        options = acejam_app._album_options_from_payload(
            {
                "concept": "Album about pressure",
                "planner_lm_provider": "lmstudio",
                "planner_model": "qwen-local",
                "planner_creativity_preset": "wild",
                "planner_temperature": 1.1,
                "planner_top_p": 0.98,
                "planner_top_k": 100,
                "planner_repeat_penalty": 1.0,
                "planner_seed": "222",
                "planner_max_tokens": 5120,
                "planner_context_length": 16384,
                "planner_timeout": 300,
                "lm_temperature": 0.42,
            }
        )

        self.assertEqual(options["planner_lm_provider"], "lmstudio")
        self.assertEqual(options["planner_model"], "qwen-local")
        self.assertEqual(options["planner_creativity_preset"], "wild")
        self.assertEqual(options["planner_temperature"], 1.1)
        self.assertEqual(options["planner_top_p"], 0.98)
        self.assertEqual(options["planner_top_k"], 100)
        self.assertEqual(options["planner_repeat_penalty"], 1.0)
        self.assertEqual(options["planner_seed"], 222)
        self.assertEqual(options["planner_max_tokens"], 5120)
        self.assertEqual(options["planner_context_length"], 16384)
        self.assertEqual(options["planner_timeout"], 300.0)
        self.assertEqual(options["lm_temperature"], 0.42)
        self.assertEqual(options["embedding_lm_provider"], "ollama")

    def test_studio_generation_with_supplied_lyrics_leaves_ace_lm_format_off(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-base",
            "caption": "cinematic pop, live drums, rich vocal chain",
            "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-base"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["ace_lm_model"], "none")
        self.assertFalse(normalized["thinking"])
        self.assertFalse(normalized["use_format"])
        self.assertFalse(normalized["use_cot_metas"])
        self.assertFalse(normalized["use_cot_caption"])
        self.assertFalse(normalized["use_cot_lyrics"])
        self.assertFalse(normalized["use_cot_language"])
        self.assertEqual(normalized["inference_steps"], 50)
        self.assertEqual(normalized["guidance_scale"], 7.0)
        self.assertEqual(normalized["shift"], 1.0)
        self.assertTrue(normalized["use_adg"])
        self.assertEqual(normalized["batch_size"], 1)
        self.assertEqual(normalized["sampler_mode"], "heun")
        self.assertEqual(normalized["audio_format"], "wav32")
        self.assertEqual(normalized["runner_plan"], "official")

    def test_text2music_supplied_lyrics_forces_direct_render_globally(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "caption": "clear pop vocal, crisp drums, radio mix",
            "lyrics": "[Verse]\nLine one lands clearly\n\n[Chorus]\nHook line stays bright",
            "duration": 30,
            "audio_format": "wav32",
            "thinking": True,
            "sample_mode": True,
            "sample_query": "rewrite and generate audio codes",
            "use_format": True,
            "use_cot_metas": True,
            "use_cot_caption": True,
            "use_cot_language": True,
            "use_cot_lyrics": True,
            "audio_code_string": "<|audio_code_1|><|audio_code_2|>",
            "src_result_id": "stale-source",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertFalse(normalized["thinking"])
        self.assertFalse(normalized["sample_mode"])
        self.assertEqual(normalized["sample_query"], "")
        self.assertFalse(normalized["use_format"])
        self.assertFalse(normalized["use_cot_metas"])
        self.assertFalse(normalized["use_cot_caption"])
        self.assertFalse(normalized["use_cot_language"])
        self.assertFalse(normalized["use_cot_lyrics"])
        self.assertEqual(normalized["audio_code_string"], "")
        self.assertIsNone(normalized["src_audio"])
        self.assertIn("audio_code_hints_cleared_for_text2music_direct_render", normalized["payload_warnings"])
        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(normalized, Path(tmp))
        self.assertFalse(request["requires_lm"])
        self.assertIsNone(request["lm_model"])
        self.assertEqual(request["params"]["audio_codes"], "")
        self.assertIsNone(request["params"]["src_audio"])

    def test_text2music_defaults_and_key_aliases_reach_official_request(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "bright pop, crisp drums",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "duration": 30,
                    "key_scale": "Am",
                    "audio_format": "wav32",
                }
            )

        self.assertIsNone(params["bpm"])
        self.assertEqual(params["key_scale"], acejam_app.DEFAULT_KEY_SCALE)
        self.assertEqual(params["time_signature"], "")
        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))
        self.assertIsNone(request["params"]["bpm"])
        self.assertEqual(request["params"]["keyscale"], "A minor")
        self.assertEqual(request["params"]["timesignature"], "")

    def test_auto_key_stays_empty_for_ace_step_payload(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "bright pop, crisp drums",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "duration": 30,
                    "bpm": "auto",
                    "key_scale": "auto",
                    "audio_format": "wav32",
                }
            )

        self.assertIsNone(params["bpm"])
        self.assertEqual(params["key_scale"], "")
        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))
        self.assertIsNone(request["params"]["bpm"])
        self.assertEqual(request["params"]["keyscale"], "")

    def test_parse_generation_payload_preserves_album_quality_gate_metadata(self):
        gate = {
            "version": "album-payload-quality-gate-2026-04-29",
            "status": "auto_repair",
            "gate_passed": True,
        }
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "bright pop, steady groove, piano, clear lead vocal, polished studio mix",
                    "lyrics": "[Verse]\nLine one lands clearly\n\n[Chorus]\nHook line stays bright",
                    "duration": 30,
                    "payload_quality_gate": gate,
                    "payload_gate_status": "auto_repair",
                    "payload_gate_passed": True,
                    "tag_coverage": {"status": "pass"},
                    "caption_integrity": {"status": "pass"},
                    "lyric_duration_fit": {"status": "pass"},
                    "repair_actions": ["caption_rebuilt_from_tag_dimensions"],
                }
            )

        self.assertEqual(params["payload_quality_gate"], gate)
        self.assertEqual(params["payload_gate_status"], "auto_repair")
        self.assertTrue(params["payload_gate_passed"])
        self.assertEqual(params["tag_coverage"]["status"], "pass")
        self.assertEqual(params["caption_integrity"]["status"], "pass")
        self.assertEqual(params["lyric_duration_fit"]["status"], "pass")
        self.assertEqual(params["repair_actions"], ["caption_rebuilt_from_tag_dimensions"])

    def test_invalid_key_validation_returns_field_error(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}):
            validation = acejam_app._validate_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-turbo",
                    "caption": "rap, hard drums",
                    "lyrics": "[Verse]\nA line for the beat\n\n[Chorus]\nHook for the street",
                    "key_scale": "H minor",
                    "ace_lm_model": "none",
                }
            )

        self.assertFalse(validation["valid"])
        self.assertIn("key_scale", validation["field_errors"])
        self.assertIn("Unsupported ACE-Step key scale", validation["field_errors"]["key_scale"])

    def test_generation_metadata_audit_flags_missing_legacy_request_metadata(self):
        params = {
            "bpm": 95,
            "key_scale": "A minor",
            "time_signature": "4",
            "duration": 180,
            "song_model": "acestep-v15-xl-sft",
            "lm_backend": "mlx",
            "inference_steps": 64,
            "guidance_scale": 8.0,
            "shift": 3.0,
            "audio_format": "wav32",
            "batch_size": 3,
            "ace_step_text_budget": {
                "source_lyrics_char_count": 100,
                "runtime_lyrics_char_count": 80,
                "lyrics_overflow_action": "none",
            },
        }

        audit = acejam_app._generation_metadata_audit(
            params,
            {"params": {"bpm": None, "keyscale": "", "duration": 180, "timesignature": "4"}},
        )

        self.assertFalse(audit["metadata_present"])
        self.assertEqual(audit["missing"], ["bpm", "keyscale"])
        self.assertFalse(audit["bpm"]["present"])
        self.assertFalse(audit["keyscale"]["present"])

    def test_audio_quality_audit_and_take_recommendation(self):
        params = {
            "quality_profile": "chart_master",
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "caption": "bright pop, crisp drums",
            "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
            "duration": 2.0,
            "bpm": 120,
            "key_scale": "A minor",
            "time_signature": "4",
            "batch_size": 2,
            "inference_steps": 64,
            "lm_backend": "mlx",
            "settings_compliance": {"valid": True, "version": "unit"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 48000
            t = acejam_app.np.linspace(0, 2.0, sr * 2, endpoint=False)
            clean = (0.45 * acejam_app.np.sin(2 * acejam_app.np.pi * 440 * t)).astype("float32")
            quiet = (0.02 * acejam_app.np.sin(2 * acejam_app.np.pi * 440 * t)).astype("float32")
            clean_path = root / "clean.wav"
            quiet_path = root / "quiet.wav"
            acejam_app.sf.write(str(clean_path), clean, sr)
            acejam_app.sf.write(str(quiet_path), quiet, sr)

            clean_audit = acejam_app._audio_quality_audit(clean_path, params, seed="1")
            quiet_audit = acejam_app._audio_quality_audit(quiet_path, params, seed="2")

        metadata = acejam_app._generation_metadata_audit(params)
        readiness = acejam_app.hit_readiness_report(params, task_type="text2music", song_model="acestep-v15-xl-sft")
        audios = [
            {"id": "take-1", "filename": "clean.wav", "audio_quality_audit": clean_audit, "metadata_adherence": acejam_app._metadata_adherence(params, metadata, clean_audit)},
            {"id": "take-2", "filename": "quiet.wav", "audio_quality_audit": quiet_audit, "metadata_adherence": acejam_app._metadata_adherence(params, metadata, quiet_audit)},
        ]
        pro = acejam_app._build_pro_quality_audit(params, audios, metadata, readiness)

        self.assertEqual(clean_audit["status"], "pass")
        self.assertIn("low_peak", quiet_audit["issues"])
        self.assertEqual(pro["recommended_take"]["audio_id"], "take-1")
        self.assertTrue(audios[0]["is_recommended_take"])

    def test_payload_validation_exposes_pro_quality_preflight(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}):
            validation = acejam_app._validate_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-turbo",
                    "caption": "rap, hard drums",
                    "lyrics": "[Verse]\nA line for the beat\n\n[Chorus]\nHook for the street",
                    "ace_lm_model": "none",
                }
            )

        self.assertIn("hit_readiness", validation)
        self.assertIn("runtime_planner", validation)
        self.assertIn("effective_settings", validation)
        self.assertEqual(validation["hit_readiness"]["version"], acejam_app.PRO_QUALITY_AUDIT_VERSION)
        self.assertEqual(validation["settings_coverage"].get("status"), "complete")

    def test_xl_sft_defaults_to_chart_master_50_steps_shift_1(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "caption": "upbeat German schlager, bright brass, accordion",
            "lyrics": "[Verse]\nHeute Nacht leuchtet die Stadt\n\n[Chorus]\nWir singen weiter",
            "duration": 180,
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["inference_steps"], 50)
        self.assertEqual(normalized["shift"], 1.0)
        self.assertEqual(normalized["song_model"], "acestep-v15-xl-sft")
        self.assertEqual(normalized["duration"], 180)

    def test_xl_sft_stale_turbo_settings_are_corrected(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "quality_profile": "chart_master",
            "caption": "west coast rap, clear male vocal, crisp drums",
            "lyrics": "[Verse]\nClear words move through the city tonight\n\n[Chorus]\nEvery line lands bright",
            "duration": 20,
            "inference_steps": 8,
            "shift": 3.0,
            "ace_lm_model": "none",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}):
            normalized = acejam_app._parse_generation_payload(payload)
            official = acejam_app._official_request_payload(dict(normalized), Path("/tmp/acejam-test"))

        self.assertEqual(normalized["inference_steps"], 50)
        self.assertEqual(normalized["shift"], 1.0)
        self.assertEqual(official["params"]["inference_steps"], 50)
        self.assertEqual(official["params"]["shift"], 1.0)

    def test_simple_custom_helpers_always_use_local_writer(self):
        client = TestClient(acejam_app.app)
        local_payload = {
            "title": "Local Helper",
            "tags": "pop, polished vocal, premium mix",
            "lyrics": "[Verse]\nLine\n\n[Chorus]\nHook",
        }

        with patch.object(acejam_app, "_run_official_lm_aux") as official, \
            patch.object(acejam_app, "compose", return_value=json.dumps(local_payload)):
            response = client.post("/api/create_sample", json={"description": "make a hit", "duration": 180})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Local Helper")
        self.assertFalse(official.called)

        with patch.object(acejam_app, "_run_official_lm_aux") as official, \
            patch.object(acejam_app, "compose", return_value=json.dumps(local_payload)):
            response = client.post("/api/format_sample", json={"caption": "polish this", "duration": 180})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(official.called)

        with patch.object(acejam_app, "_run_official_lm_aux") as official, \
            patch.object(acejam_app, "compose", return_value=json.dumps(local_payload)):
            response = client.post(
                "/api/create_sample",
                json={
                    "description": "make a hit",
                    "duration": 180,
                    "planner_lm_provider": "ace_step_lm",
                    "planner_model": acejam_app.ACE_LM_PREFERRED_MODEL,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(official.called, "ACE-Step 5Hz LM must never be reachable from /api/create_sample")

    def test_simple_helper_respects_explicit_lm_opt_out(self):
        client = TestClient(acejam_app.app)
        fallback = {
            "title": "Ollama Helper",
            "tags": "pop, clean vocal",
            "lyrics": "[Verse]\nLine\n\n[Chorus]\nHook",
        }

        with patch.object(acejam_app, "_run_official_lm_aux") as official, \
            patch.object(acejam_app, "compose", return_value=json.dumps(fallback)):
            response = client.post(
                "/api/create_sample",
                json={"description": "make a hook", "duration": 60, "ace_lm_model": "none"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(official.called)
        self.assertEqual(response.json()["engine"], "ollama")

    def test_official_helper_lm_sampling_defaults_are_best_quality(self):
        params = acejam_app._official_aux_params({})

        self.assertEqual(params["lm_temperature"], 0.85)
        self.assertEqual(params["lm_top_k"], 0)
        self.assertEqual(params["lm_top_p"], 0.9)

    def test_official_generation_payload_omits_lm_only_repetition_penalty(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "German schlager, bright brass, singalong chorus",
                    "lyrics": "[Verse]\nWir tanzen durch die Nacht\n\n[Chorus]\nLa la, das Herz erwacht",
                    "duration": 30,
                    "lm_repetition_penalty": 1.4,
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))

        self.assertEqual(params["lm_repetition_penalty"], 1.0)
        self.assertNotIn("repetition_penalty", request["params"])
        self.assertIs(request["acejam_skip_lora_base_backup"], True)

    def test_official_generation_payload_fits_overlong_lyrics_before_runner(self):
        long_lyrics = "<think>draft</think>\nI will now write it.\n[Verse]\n" + "\n".join(
            f"Runtime line {index}" for index in range(1200)
        )
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "German schlager, bright brass, singalong chorus",
                    "lyrics": long_lyrics,
                    "duration": 180,
                    "ace_lm_model": "none",
                    "audio_format": "wav32",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))

        self.assertLessEqual(len(request["params"]["lyrics"]), acejam_app.ACE_STEP_LYRICS_CHAR_LIMIT)
        self.assertNotIn("<think>", request["params"]["lyrics"])
        self.assertNotIn("I will now write", request["params"]["lyrics"])
        self.assertEqual(request["ace_step_text_budget"]["source_lyrics_char_count"], len(long_lyrics))
        self.assertLessEqual(
            request["ace_step_text_budget"]["runtime_lyrics_char_count"],
            acejam_app.ACE_STEP_LYRICS_CHAR_LIMIT,
        )

    def test_strict_overlong_exact_lyrics_fails_before_runner(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}):
            validation = acejam_app._validate_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "pop, bright hook",
                    "lyrics": "[Verse]\n" + ("line\n" * 1200),
                    "duration": 180,
                    "lyrics_overflow_policy": "error",
                    "ace_lm_model": "none",
                }
            )

        self.assertFalse(validation["valid"])
        self.assertIn("4096", validation["field_errors"]["lyrics"])

    def test_explicit_lm_none_survives_disabled_cot_flags(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "German schlager, bright brass, singalong chorus",
                    "lyrics": "[Verse]\nWir tanzen durch die Nacht\n\n[Chorus]\nLa la, das Herz erwacht",
                    "duration": 30,
                    "audio_format": "wav32",
                    "ace_lm_model": "none",
                    "thinking": False,
                    "use_format": False,
                    "use_cot_caption": False,
                    "use_cot_language": False,
                    "use_cot_metas": False,
                    "use_cot_lyrics": False,
                    "lm_backend": "mlx",
                }
            )

        self.assertEqual(params["ace_lm_model"], "none")
        self.assertFalse(params["thinking"])
        self.assertFalse(params["use_format"])
        self.assertFalse(params["use_cot_lyrics"])
        self.assertEqual(params["runner_plan"], "official")

    def test_official_runner_log_redacts_prompt_and_audio_codes(self):
        prompt_line = "formatted_prompt_with_cot=<|im_start|>user secrets\n"
        codes_line = "Debug output text: <|audio_code_1|><|audio_code_2|><|audio_code_3|>\n"

        self.assertNotIn("user secrets", acejam_app._redact_official_runner_log_line(prompt_line))
        self.assertIn("redacted", acejam_app._redact_official_runner_log_line(codes_line))

    # Removed: test_ui_uses_keyscale_selects_and_auto_bpm_default + the Local
    # AI writer selector test inspected DOM ids inside the legacy Python SPA
    # (app/index.html) which is gone post v0.2. Equivalent React widgets live
    # in app/web/src/wizards/CustomWizard.tsx and Settings.tsx.

    # Removed: test_ui_global_model_advice_uses_state_mode (legacy SPA gone).

    def test_album_payload_options_normalize_planning_engine(self):
        direct = acejam_app._album_options_from_payload({"concept": "one song"}, song_model="auto")
        micro = acejam_app._album_options_from_payload({"concept": "one song", "agent_engine": "legacy_crewai"}, song_model="auto")

        self.assertEqual(direct["agent_engine"], "acejam_agents")
        self.assertEqual(micro["agent_engine"], "crewai_micro")

    # Removed: test_lora_one_click_ui_and_adapterbar_are_exposed and the
    # epoch-audition / trainer device / song-model UI assertions all targeted
    # the legacy Python SPA (app/index.html). Equivalent React widgets live in
    # app/web/src/wizards/TrainerWizard.tsx (file picker, hyperparameters,
    # auto-transcribe, genre, training-job progress) and Settings.tsx (LoRA
    # load/scale/unload). Their behavior is validated via TS type checks +
    # Playwright smoke tests; the live API contract is covered elsewhere in
    # this file.

    def test_react_lora_trainer_exposes_test_wav_genres_and_playback(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        trainer = (web_src / "wizards" / "TrainerWizard.tsx").read_text(encoding="utf-8")
        tracker = (web_src / "components" / "JobTracker.tsx").read_text(encoding="utf-8")

        self.assertIn("/api/lora/epoch-audition/genres", trainer)
        self.assertIn("epoch_audition_genre", trainer)
        self.assertIn("Test-WAV genre", trainer)
        self.assertIn("Rap / Hip-hop", trainer)
        self.assertIn("Soul / R&B", trainer)
        self.assertIn("folderInputRef", trainer)
        self.assertIn("configureFolderInput", trainer)
        self.assertIn('id="trainer-folder-input"', trainer)
        self.assertIn('htmlFor="trainer-folder-input"', trainer)
        self.assertIn('setAttribute("webkitdirectory"', trainer)
        self.assertIn('removeAttribute("accept")', trainer)
        self.assertIn("openFolderPicker", trainer)
        self.assertIn("Map kiezen", trainer)
        self.assertIn("directoryInput.webkitdirectory = true", trainer)
        self.assertIn("Losse bestanden kiezen", trainer)
        self.assertIn("stats.audio === 0", trainer)
        self.assertIn("trigger tag is pas nodig bij het starten van de training", trainer)
        self.assertIn('fd.append("files", item.file, item.relativePath', trainer)
        self.assertIn('fd.append("genre_label_mode"', trainer)
        self.assertIn("AI per track als fallback", trainer)
        self.assertIn("MusicBrainz artist-tags", trainer)
        self.assertIn("genre_label_source", trainer)
        self.assertIn("uploadItemsFromDataTransfer", trainer)
        self.assertIn("Auto stop bij loss-plateau", trainer)
        self.assertIn("autoEpochTarget", trainer)
        self.assertIn("max_epochs_or_loss_plateau", trainer)
        self.assertIn("WaveformPlayer", tracker)
        self.assertIn("Loss & early stop", tracker)
        self.assertIn("best_loss_epoch", tracker)
        self.assertIn("Epoch ${text(item.epoch)} test-WAV", tracker)

    def test_react_lora_selector_filters_peft_lora_and_uses_tag_labels(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        lora_lib = (web_src / "lib" / "lora.ts").read_text(encoding="utf-8")
        selector = (web_src / "components" / "wizard" / "LoraSelector.tsx").read_text(encoding="utf-8")
        api_ts = (web_src / "lib" / "api.ts").read_text(encoding="utf-8")

        self.assertIn("export const getLoraAdapters", api_ts)
        self.assertIn('adapterType === "lora"', lora_lib)
        self.assertIn("adapter.generation_loadable === true || adapter.is_loadable === true", lora_lib)
        for label_source in ["adapter.display_name", "adapter.trigger_tag", "adapter.label", "adapter.name"]:
            self.assertIn(label_source, lora_lib)
        self.assertIn('value={NONE}>Geen LoRA', selector)
        self.assertIn("getLoraAdapters", selector)
        self.assertIn("isGenerationLoraAdapter", selector)
        self.assertIn("loraTriggerOptions", lora_lib)
        self.assertIn("adapter.generation_trigger_tag", lora_lib)
        self.assertIn("adapter.trigger_aliases", lora_lib)
        self.assertIn("Trigger tag activeren", selector)
        self.assertIn("Wordt opgeslagen/gebruikt als", (web_src / "wizards" / "TrainerWizard.tsx").read_text(encoding="utf-8"))
        self.assertIn("LoRA trigger source", (web_src / "components" / "JobTracker.tsx").read_text(encoding="utf-8"))
        self.assertIn("songModelFromLoraVariant", lora_lib)
        self.assertIn('return "acestep-v15-xl-sft"', lora_lib)

    def test_react_generation_wizards_send_lora_payload_fields(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        wizard_paths = [
            web_src / "wizards" / "SimpleWizard.tsx",
            web_src / "wizards" / "CustomWizard.tsx",
            web_src / "wizards" / "SourceAudioWizard.tsx",
            web_src / "wizards" / "AlbumWizard.tsx",
            web_src / "wizards" / "NewsWizard.tsx",
        ]

        for path in wizard_paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("LoraSelector", text, path.name)
            self.assertIn("normalizeLoraSelection", text, path.name)
            self.assertIn("lora_adapter_name", text, path.name)
            self.assertIn("lora_adapter_path", text, path.name)
            self.assertIn("use_lora_trigger", text, path.name)
            self.assertIn("lora_trigger_tag", text, path.name)
            self.assertIn("lora_scale", text, path.name)
            self.assertIn("adapter_model_variant", text, path.name)
            self.assertIn("adapter_song_model", text, path.name)
            self.assertIn('form.setValue("song_model", selection.adapter_song_model', text, path.name)

        schemas = (web_src / "lib" / "schemas.ts").read_text(encoding="utf-8")
        for field in [
            "use_lora",
            "lora_adapter_path",
            "lora_adapter_name",
            "use_lora_trigger",
            "lora_trigger_tag",
            "lora_scale",
            "adapter_model_variant",
            "adapter_song_model",
        ]:
            self.assertIn(field, schemas)

    def test_react_generation_wizards_default_to_xl_sft_with_base_only_guard(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        schemas = (web_src / "lib" / "schemas.ts").read_text(encoding="utf-8")
        source = (web_src / "wizards" / "SourceAudioWizard.tsx").read_text(encoding="utf-8")

        for path in [
            web_src / "wizards" / "SimpleWizard.tsx",
            web_src / "wizards" / "CustomWizard.tsx",
            web_src / "wizards" / "AlbumWizard.tsx",
            web_src / "wizards" / "NewsWizard.tsx",
            web_src / "wizards" / "RepaintWizard.tsx",
            web_src / "wizards" / "ExtractWizard.tsx",
            web_src / "wizards" / "LegoWizard.tsx",
            web_src / "wizards" / "CompleteWizard.tsx",
        ]:
            text = path.read_text(encoding="utf-8")
            self.assertIn("acestep-v15-xl-sft", text, path.name)
            self.assertNotIn('defaultModel: "acestep-v15-xl-base"', text, path.name)

        self.assertIn('song_model: "acestep-v15-xl-sft"', schemas)
        self.assertIn('song_model: config.defaultModel ?? "acestep-v15-xl-sft"', source)
        self.assertIn('["acestep-v15-xl-sft", "ACE-Step v1.5 XL SFT (aanbevolen)"]', source)
        self.assertIn("BASE_ONLY_VARIANTS", source)
        self.assertIn("baseOnlyModelError", source)
        self.assertIn("ACE-Step v1.5 XL Base", source)
        self.assertIn("isValid: !!source?.uploadId && !baseOnlyModelError", source)

    def test_react_ai_fill_blocks_next_while_pending(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        ai_step = (web_src / "components" / "wizard" / "AIPromptStep.tsx").read_text(encoding="utf-8")
        wizard_paths = [
            web_src / "wizards" / "SimpleWizard.tsx",
            web_src / "wizards" / "CustomWizard.tsx",
            web_src / "wizards" / "SourceAudioWizard.tsx",
            web_src / "wizards" / "AlbumWizard.tsx",
            web_src / "wizards" / "NewsWizard.tsx",
        ]

        self.assertIn("onPendingChange?: (pending: boolean) => void", ai_step)
        self.assertIn("onPendingChange?.(aiFill.isPending)", ai_step)

        for path in wizard_paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("aiPromptPending", text, path.name)
            self.assertIn("onPendingChange={setAiPromptPending}", text, path.name)
            self.assertIn("!aiPromptPending", text, path.name)

    def test_react_album_ai_fill_sends_full_payload_and_embedding_settings(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        api_ts = (web_src / "lib" / "api.ts").read_text(encoding="utf-8")
        settings_store = (web_src / "store" / "settings.ts").read_text(encoding="utf-8")
        ai_step = (web_src / "components" / "wizard" / "AIPromptStep.tsx").read_text(encoding="utf-8")
        album = (web_src / "wizards" / "AlbumWizard.tsx").read_text(encoding="utf-8")
        settings = (web_src / "pages" / "Settings.tsx").read_text(encoding="utf-8")

        self.assertIn("current_payload?: Record<string, unknown>", api_ts)
        self.assertIn("embedding_provider?: string", api_ts)
        self.assertIn("embedding_model?: string", api_ts)
        self.assertIn("currentPayload?: Record<string, unknown>", ai_step)
        self.assertIn("current_payload: currentPayload", ai_step)
        self.assertIn("embeddingModelDetails", ai_step)
        self.assertIn("AI Memory / RAG embedding", ai_step)
        self.assertIn("groupedEmbeddingModels", ai_step)
        self.assertIn("setEmbedding", ai_step)
        self.assertIn("/api/local-llm/settings", ai_step)
        self.assertIn("embedding_lm_provider: embeddingProvider", ai_step)
        self.assertIn("embedding_model: name", ai_step)
        self.assertIn("embeddingProvider", settings_store)
        self.assertIn("setEmbedding", settings_store)
        self.assertIn("currentPayload={albumCurrentPayload}", album)
        self.assertIn("embedding_provider: embeddingProvider", album)
        self.assertIn("embedding_model: embeddingModel", album)
        self.assertIn("AI Memory / RAG embeddings", album)
        self.assertIn("Qwen3-Embedding-0.6B", album)
        self.assertIn("AI Memory / RAG embeddings", settings)
        self.assertIn("/api/local-llm/settings", settings)
        self.assertIn("/api/local-llm/test", settings)
        self.assertIn("ACE-Step audio text encoder blijft: Qwen3-Embedding-0.6B", settings)

    def test_lora_status_and_adapters_expose_display_name_and_trigger(self):
        client = TestClient(acejam_app.app)

        class StubTrainingManager:
            def status(self):
                return {"ready": True, "trainer_device_policy": {"default": "mps", "cpu_blocked": True}}

            def list_adapters(self):
                return [
                    {
                        "name": "charaf-hook",
                        "display_name": "charaf hook",
                        "trigger_tag": "charaf hook",
                        "adapter_type": "lora",
                        "path": "/tmp/charaf-hook",
                        "is_loadable": True,
                    }
                ]

        with patch.object(acejam_app, "training_manager", StubTrainingManager()), \
            patch.object(acejam_app.handler, "get_lora_status", return_value={"loaded": False}):
            status = client.get("/api/lora/status").json()
            adapters = client.get("/api/lora/adapters").json()

        self.assertEqual(status["adapters"][0]["display_name"], "charaf hook")
        self.assertEqual(status["adapters"][0]["trigger_tag"], "charaf hook")
        self.assertEqual(adapters["adapters"][0]["display_name"], "charaf hook")

    def test_official_runner_uses_safe_lora_adapter_name_for_decimal_checkpoints(self):
        official_runner = importlib.import_module("official_runner")

        class StubHandler:
            def __init__(self):
                self.calls = []
                self._base_decoder = None

            def add_lora(self, path, adapter_name=None):
                self.calls.append(("add_lora", path, adapter_name))
                return "✅ LoRA loaded"

            def set_lora_scale(self, scale):
                self.calls.append(("set_lora_scale", scale))
                return "✅ scale"

            def set_use_lora(self, use):
                self.calls.append(("set_use_lora", use))
                return "✅ use"

        handler = StubHandler()
        result = official_runner._apply_lora_request(
            handler,
            {
                "use_lora": True,
                "lora_adapter_path": "/tmp/checkpoints/epoch_1_loss_0.9130",
                "lora_adapter_name": "epoch_1_loss_0.9130",
                "lora_scale": 0.7,
            },
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["adapter_name"], "epoch_1_loss_0_9130")
        self.assertEqual(handler._base_decoder, {})
        self.assertEqual(handler.calls[0], ("add_lora", "/tmp/checkpoints/epoch_1_loss_0.9130", "epoch_1_loss_0_9130"))

    def test_xl_sft_lora_is_blocked_on_turbo_before_runner_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "xl-sft-adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/models/acestep-v15-xl-sft"}),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            payload = {
                "task_type": "text2music",
                "song_model": "acestep-v15-turbo",
                "caption": "hip hop",
                "lyrics": "[Verse]\nClear line\n\n[Chorus]\nClear hook",
                "use_lora": True,
                "lora_adapter_path": str(adapter),
            }

            with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}), \
                patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none"}):
                with self.assertRaises(ValueError) as raised:
                    acejam_app._parse_generation_payload(payload)

            self.assertIn("trained for acestep-v15-xl-sft", str(raised.exception))

    def test_lora_resume_endpoint_delegates_to_training_manager(self):
        client = TestClient(acejam_app.app)

        class StubTrainingManager:
            def resume_job(self, job_id):
                return {"id": job_id, "state": "queued", "params": {"device": "mps"}}

        with patch.object(acejam_app, "training_manager", StubTrainingManager()):
            response = client.post("/api/lora/jobs/resumejob/resume")

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["job"]["id"], "resumejob")
        self.assertEqual(payload["job"]["params"]["device"], "mps")

    def test_lora_payload_reaches_generation_and_official_runner(self):
        adapter_path = "/tmp/unit-adapter"
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "bright pop, crisp drums",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "duration": 30,
                    "audio_format": "wav32",
                    "seed": 314,
                    "use_random_seed": False,
                    "use_lora": True,
                    "lora_adapter_path": adapter_path,
                    "lora_adapter_name": "unit",
                    "use_lora_trigger": True,
                    "lora_trigger_tag": "2pac",
                    "lora_scale": 0.65,
                    "adapter_model_variant": "xl_sft",
                }
            )

        self.assertTrue(params["use_lora"])
        self.assertEqual(params["lora_adapter_path"], adapter_path)
        self.assertTrue(params["use_lora_trigger"])
        self.assertEqual(params["lora_trigger_tag"], "pac")
        self.assertRegex(params["caption"], r"(?i)(?<![a-z0-9])pac(?![a-z0-9])")
        self.assertNotRegex(params["lyrics"], r"(?i)(?<![a-z0-9])pac(?![a-z0-9])")
        self.assertEqual(params["lora_trigger_conditioning_audit"]["status"], "applied")
        self.assertEqual(params["lora_scale"], 0.65)
        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))
        self.assertTrue(request["use_lora"])
        self.assertEqual(request["lora_adapter_path"], adapter_path)
        self.assertEqual(request["lora_adapter_name"], "unit")
        self.assertEqual(request["params"]["caption"], params["caption"])
        self.assertEqual(request["lora_scale"], 0.65)
        self.assertEqual(request["params"]["seed"], 314)
        self.assertEqual(request["config"]["seeds"], "314")

    def test_lora_metadata_trigger_is_auto_applied_without_ui_trigger_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/models/acestep-v15-xl-sft"}),
                encoding="utf-8",
            )
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter_dir / "acejam_adapter.json").write_text(
                json.dumps(
                    {
                        "display_name": "2Pac Epoch",
                        "trigger_tag_raw": "2pac",
                        "generation_trigger_tag": "pac",
                        "trigger_aliases": ["pac", "2pac"],
                        "trigger_source": "training",
                        "adapter_type": "lora",
                        "model_variant": "xl_sft",
                        "song_model": "acestep-v15-xl-sft",
                        "quality_status": "ready",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
                patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
                params = acejam_app._parse_generation_payload(
                    {
                        "task_type": "text2music",
                        "song_model": "acestep-v15-xl-sft",
                        "caption": "rap, west coast drums",
                        "lyrics": "[Verse - rap]\nLine one",
                        "duration": 30,
                        "use_lora": True,
                        "lora_adapter_path": str(adapter_dir),
                        "lora_scale": 1.0,
                    }
                )

        self.assertTrue(params["use_lora_trigger"])
        self.assertEqual(params["lora_trigger_tag"], "pac")
        self.assertEqual(params["lora_trigger_source"], "training")
        self.assertRegex(params["caption"], r"(?i)(?<![a-z0-9])pac(?![a-z0-9])")
        self.assertNotRegex(params["lyrics"], r"(?i)(?<![a-z0-9])pac(?![a-z0-9])")
        self.assertEqual(params["lora_trigger_conditioning_audit"]["trigger_source"], "training")

    def test_lora_trigger_tag_is_not_duplicated_in_caption(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "pac, rap, west coast hip hop",
                    "lyrics": "[Verse]\nClear line\n\n[Chorus]\nHook line",
                    "duration": 30,
                    "use_lora": True,
                    "lora_adapter_path": "/tmp/unit-adapter",
                    "use_lora_trigger": True,
                    "lora_trigger_tag": "pac",
                    "adapter_model_variant": "xl_sft",
                }
            )

        self.assertEqual(
            len(re.findall(r"(?i)(?<![a-z0-9])pac(?![a-z0-9])", params["caption"])),
            1,
        )
        self.assertEqual(params["lora_trigger_conditioning_audit"]["status"], "present")

    def test_no_lora_payload_strips_stale_trigger_from_caption(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "pac, rap, west coast hip hop",
                    "lyrics": "[Verse - rap]\nClear line\n\n[Chorus - rap hook]\nHook line",
                    "duration": 30,
                    "use_lora": False,
                    "use_lora_trigger": True,
                    "lora_trigger_tag": "2pac",
                }
            )

        self.assertFalse(params["use_lora"])
        self.assertFalse(params["use_lora_trigger"])
        self.assertEqual(params["lora_trigger_tag"], "")
        self.assertNotRegex(params["caption"], r"(?i)(?<![a-z0-9])pac(?![a-z0-9])")
        self.assertIn("rap", params["caption"])
        self.assertTrue(params["lora_trigger_conditioning_audit"]["stripped_from_caption"])
        self.assertIn("lora_trigger_stripped_for_no_lora", params["payload_warnings"])

    def test_lora_preflight_baseline_strips_trigger_conditioning(self):
        params = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "quality_profile": "chart_master",
            "caption": "pac, rap, west coast hip hop",
            "lyrics": "[Verse - rap]\nClear line\n\n[Chorus - rap hook]\nHook line",
            "duration": 180,
            "use_lora": True,
            "use_lora_trigger": True,
            "lora_trigger_tag": "pac",
            "lora_adapter_path": "/tmp/unit-adapter",
            "lora_adapter_name": "unit",
            "lora_scale": 1.0,
            "payload_warnings": [],
            "repair_actions": [],
        }

        baseline = acejam_app._lora_preflight_attempt_params(
            params,
            use_lora=False,
            scale=0.0,
            label="baseline",
        )

        self.assertFalse(baseline["use_lora"])
        self.assertFalse(baseline["use_lora_trigger"])
        self.assertEqual(baseline["lora_trigger_tag"], "")
        self.assertEqual(baseline["lora_scale"], 0.0)
        self.assertNotRegex(baseline["caption"], r"(?i)(?<![a-z0-9])pac(?![a-z0-9])")
        self.assertIn("rap", baseline["caption"])
        self.assertIn("lora_trigger_stripped_for_no_lora_preflight", baseline["payload_warnings"])

    def test_raw_lora_trigger_caption_can_be_preserved_for_ab_test(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "2pac, rap, west coast hip hop",
                    "lyrics": "[Verse - rap]\nClear line\n\n[Chorus - rap hook]\nHook line",
                    "duration": 30,
                    "use_lora": True,
                    "lora_adapter_path": "/tmp/unit-adapter",
                    "lora_adapter_name": "unit",
                    "use_lora_trigger": False,
                    "lora_trigger_tag": "2pac",
                    "lora_scale": 1.0,
                    "adapter_model_variant": "xl_sft",
                    "preserve_raw_lora_trigger_caption": True,
                }
            )

        self.assertTrue(params["use_lora"])
        self.assertFalse(params["use_lora_trigger"])
        self.assertEqual(params["lora_trigger_tag"], "")
        self.assertRegex(params["caption"], r"(?i)(?<![a-z0-9])2pac(?![a-z0-9])")
        self.assertNotRegex(params["lyrics"], r"(?i)(?<![a-z0-9])2pac(?![a-z0-9])")
        self.assertIn("generation_prompt_token_2pac_preserved_for_lora_trigger_test", params["payload_warnings"])

    def test_lora_epoch_audition_uses_private_generation_without_library_save(self):
        captured = {}

        def fake_generation(params):
            captured.update(params)
            return {
                "success": True,
                "result_id": "audition-result",
                "audios": [{"id": "take-1", "result_id": "audition-result", "filename": "take-1.wav", "audio_url": "/media/results/audition-result/take-1.wav"}],
            }
        gate_transcripts = [{
            "path": str(acejam_app.RESULTS_DIR / "audition-result" / "take-1.wav"),
            "status": "pass",
            "passed": True,
            "blocking": False,
            "text": "Line one hook line clear vocal",
            "word_count": 6,
            "keyword_hits": ["line", "hook"],
            "missing_keywords": [],
            "issue": "",
        }]

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}), \
            patch.object(acejam_app, "_run_lora_preflight_verifier", return_value=None), \
            patch.object(acejam_app, "_transcribe_audio_paths", return_value=gate_transcripts), \
            patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_generation):
            result = acejam_app._run_lora_epoch_audition(
                {
                    "epoch": 2,
                    "trigger_tag": "charaf hook",
                    "checkpoint_path": "/tmp/checkpoints/epoch_2_loss_0.1",
                    "caption": "charaf hook, bright pop",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "vocal_language": "en",
                    "duration": 20,
                    "seed": 123,
                    "lora_scale": 0.7,
                    "song_model": "acestep-v15-xl-sft",
                    "model_variant": "xl_sft",
                }
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["result_id"], "audition-result")
        self.assertEqual(captured["task_type"], "text2music")
        self.assertEqual(captured["duration"], 20)
        self.assertIn("[Verse - clean pop vocal]", captured["lyrics"])
        self.assertIn("[Chorus - bright pop hook]", captured["lyrics"])
        self.assertIn("Line one", captured["lyrics"])
        self.assertIn("Hook line", captured["lyrics"])
        self.assertIn("clear intelligible vocal", captured["caption"])
        self.assertEqual(captured["vocal_language"], "en")
        self.assertEqual(captured["seed"], "123")
        self.assertEqual(acejam_app.EPOCH_AUDITION_INFERENCE_STEPS, 50)
        self.assertEqual(captured["inference_steps"], acejam_app.EPOCH_AUDITION_INFERENCE_STEPS)
        self.assertEqual(captured["ace_lm_model"], "none")
        self.assertFalse(captured["thinking"])
        self.assertFalse(captured["sample_mode"])
        self.assertFalse(captured["use_format"])
        self.assertFalse(captured["use_cot_metas"])
        self.assertFalse(captured["use_cot_caption"])
        self.assertFalse(captured["use_cot_lyrics"])
        self.assertFalse(captured["use_cot_language"])
        self.assertFalse(captured["save_to_library"])
        self.assertTrue(captured["lora_preflight_required"])
        self.assertTrue(captured["use_lora"])
        self.assertEqual(captured["lora_adapter_path"], "/tmp/checkpoints/epoch_2_loss_0.1")
        self.assertEqual(captured["lora_scale"], 0.7)
        if acejam_app._IS_APPLE_SILICON:
            self.assertEqual(captured["device"], "mps")
            self.assertEqual(captured["dtype"], "float32")
            self.assertEqual(captured["audio_backend"], "mps_torch")
            self.assertFalse(captured["use_mlx_dit"])
        self.assertIn(result["lyrics_fit"]["action"], {"none", "fit_for_20s"})
        self.assertTrue(result["lyrics_fit"]["timed_structure"])

    def test_lora_epoch_audition_auto_model_matches_checkpoint_variant(self):
        captured = {}

        def fake_generation(params):
            captured.update(params)
            return {
                "success": True,
                "result_id": "audition-result",
                "audios": [{"id": "take-1", "result_id": "audition-result", "filename": "take-1.wav", "audio_url": "/media/results/audition-result/take-1.wav"}],
            }
        gate_transcripts = [{
            "path": str(acejam_app.RESULTS_DIR / "audition-result" / "take-1.wav"),
            "status": "pass",
            "passed": True,
            "blocking": False,
            "text": "Line one clear vocal",
            "word_count": 4,
            "keyword_hits": ["line"],
            "missing_keywords": [],
            "issue": "",
        }]

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo", "acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}), \
            patch.object(acejam_app, "_run_lora_preflight_verifier", return_value=None), \
            patch.object(acejam_app, "_transcribe_audio_paths", return_value=gate_transcripts), \
            patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_generation):
            result = acejam_app._run_lora_epoch_audition(
                {
                    "epoch": 1,
                    "trigger_tag": "pac",
                    "checkpoint_path": "/tmp/checkpoints/epoch_1_loss_1.3433",
                    "caption": "pac",
                    "lyrics": "[Verse]\nLine one",
                    "song_model": "auto",
                    "model_variant": "turbo",
                }
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["song_model"], "acestep-v15-turbo")
        self.assertEqual(captured["adapter_model_variant"], "turbo")

    def test_lora_epoch_audition_fits_long_lyrics_and_language_for_wav_test(self):
        captured = {}
        long_lyrics = "\n".join(
            [
                "[Final Chorus - rap, apocalyptic, choir vocals, full climax]",
                "Count that room and keep it moving",
                "Name that chair and tell it true",
                "Every lie receives a number",
                "Every shadow turns to proof",
                "No more myth and no more static",
                "[Verse 4 - rap, acapella start, then drums return]",
                "Borrowed soil and fountain pens",
                "Concrete learned the mother tongue",
                "Every crack became a chorus",
                "Every curb knew what was done",
                "[drums return, building energy]",
                "Arrangement note should not be sung",
                "[Outro - fading, acapella, choir hum]",
                "You signed the wrong silence",
            ]
        )

        def fake_generation(params):
            captured.update(params)
            return {
                "success": True,
                "result_id": "audition-result",
                "audios": [{"id": "take-1", "result_id": "audition-result", "filename": "take-1.wav", "audio_url": "/media/results/audition-result/take-1.wav"}],
            }
        gate_transcripts = [{
            "path": str(acejam_app.RESULTS_DIR / "audition-result" / "take-1.wav"),
            "status": "pass",
            "passed": True,
            "blocking": False,
            "text": "Count that room every lie hook line clear",
            "word_count": 8,
            "keyword_hits": ["count", "lie"],
            "missing_keywords": [],
            "issue": "",
        }]

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}), \
            patch.object(acejam_app, "_run_lora_preflight_verifier", return_value=None), \
            patch.object(acejam_app, "_transcribe_audio_paths", return_value=gate_transcripts), \
            patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_generation):
            result = acejam_app._run_lora_epoch_audition(
                {
                    "epoch": 1,
                    "trigger_tag": "pac",
                    "checkpoint_path": "/tmp/checkpoints/epoch_1_loss_0.9130",
                    "caption": "pac, west coast rap, hip hop, gangster",
                    "lyrics": long_lyrics,
                    "vocal_language": "en",
                    "seed": 42,
                    "lora_scale": 1.0,
                    "song_model": "acestep-v15-xl-sft",
                    "model_variant": "xl_sft",
                }
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["duration"], 30)
        self.assertEqual(captured["vocal_language"], "en")
        self.assertEqual(captured["seed"], "42")
        self.assertEqual(captured["inference_steps"], acejam_app.EPOCH_AUDITION_INFERENCE_STEPS)
        sung_lines = [line for line in captured["lyrics"].splitlines() if line.strip() and not line.startswith("[")]
        self.assertLessEqual(len(captured["lyrics"]), 360)
        self.assertLessEqual(len(sung_lines), 6)
        self.assertIn("[Chorus - rap", captured["lyrics"])
        self.assertIn("[Verse - rap", captured["lyrics"])
        self.assertIn("clear intelligible vocal", captured["caption"])
        self.assertNotIn("Final Chorus -", captured["lyrics"])
        self.assertNotIn("Verse 4 -", captured["lyrics"])
        self.assertNotIn("[drums return", captured["lyrics"])
        self.assertNotIn("Arrangement note", captured["lyrics"])
        if acejam_app._IS_APPLE_SILICON:
            self.assertEqual(captured["device"], "mps")
            self.assertEqual(captured["dtype"], "float32")
            self.assertEqual(captured["audio_backend"], "mps_torch")
            self.assertFalse(captured["use_mlx_dit"])
        self.assertEqual(result["lyrics_fit"]["action"], "fit_for_30s")

    def test_lora_upload_path_sanitizer_preserves_relative_folders(self):
        self.assertEqual(str(acejam_app._safe_lora_upload_relative_path("dataset/sub/song.wav")), "dataset/sub/song.wav")
        self.assertEqual(str(acejam_app._safe_lora_upload_relative_path("../evil.wav")), "evil.wav")

    def test_lora_dataset_import_accepts_nested_folder_uploads(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "dataset"
            labels = [{"filename": "artist/session/song.aiff", "caption": "rap", "lyrics": "lyrics"}]
            with patch.object(acejam_app.training_manager, "import_root_for", return_value=target), \
                patch.object(acejam_app.training_manager, "scan_dataset", return_value={"files": labels}), \
                patch.object(acejam_app.training_manager, "label_entries", return_value=labels):
                response = client.post(
                    "/api/lora/dataset/import-folder",
                    data={"dataset_id": "nested-dataset", "trigger_tag": "pac", "language": "en"},
                    files=[
                        ("files", ("artist/session/song.aiff", b"FORM....AIFF", "audio/aiff")),
                        ("files", ("artist/session/song.txt", b"[Verse]\nlyrics", "text/plain")),
                    ],
                )

            data = response.json()
            self.assertTrue(data["success"])
            self.assertIn("artist/session/song.aiff", data["copied_files"])
            self.assertIn("artist/session/song.txt", data["copied_files"])
            self.assertTrue((target / "artist" / "session" / "song.aiff").exists())
            self.assertTrue((target / "artist" / "session" / "song.txt").exists())

    def test_training_genre_label_prefers_sidecar_metadata_before_musicbrainz_or_ai(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "Artist - Track.wav"
            audio.write_bytes(b"fake")
            (Path(tmp) / "Artist - Track.json").write_text(
                json.dumps({"genre": "neo soul", "style_profile": "soul", "caption_tags": "warm soul vocal"}),
                encoding="utf-8",
            )

            with patch.object(acejam_app, "_detect_bpm_key", return_value=(88, "C minor")), \
                patch.object(acejam_app, "_search_lyrics_online", return_value="[Verse]\nReal lyric line"), \
                patch.object(acejam_app, "_musicbrainz_artist_tags", side_effect=AssertionError("MusicBrainz should not run")), \
                patch.object(acejam_app, "local_llm_chat_completion_response", side_effect=AssertionError("AI should not run")):
                result = acejam_app._training_lookup_online_lyrics(
                    audio,
                    {"language": "en", "genre_label_mode": "ai_auto"},
                )

            self.assertEqual(result["genre"], "neo soul")
            self.assertEqual(result["style_profile"], "soul")
            self.assertEqual(result["genre_label_source"], "metadata")
            self.assertIn("warm soul vocal", result["caption"])

    def test_training_genre_label_uses_musicbrainz_before_ai_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "Artist - Track.wav"
            audio.write_bytes(b"fake")

            with patch.object(acejam_app, "_detect_bpm_key", return_value=(94, "A minor")), \
                patch.object(acejam_app, "_search_lyrics_online", return_value="[Verse]\nReal lyric line"), \
                patch.object(acejam_app, "_musicbrainz_artist_tags", return_value=["hip hop", "rap"]), \
                patch.object(acejam_app, "local_llm_chat_completion_response", side_effect=AssertionError("AI should not run")):
                result = acejam_app._training_lookup_online_lyrics(
                    audio,
                    {"language": "en", "genre_label_mode": "ai_auto"},
                )

            self.assertEqual(result["genre"], "hip hop, rap")
            self.assertEqual(result["style_profile"], "rap")
            self.assertEqual(result["genre_label_source"], "musicbrainz")
            self.assertIn("hip hop", result["caption"].lower())
            self.assertIn("[Verse - rap", result["lyrics"])

    def test_training_genre_label_uses_local_llm_only_when_metadata_and_musicbrainz_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "Unknown - Track.wav"
            audio.write_bytes(b"fake")
            llm_response = {
                "content": json.dumps(
                    {
                        "genre": "country, americana",
                        "style_profile": "country",
                        "caption_tags": "warm acoustic guitars, heartfelt country vocal",
                        "confidence": 0.82,
                        "reason": "Folder and lyrics indicate country storytelling.",
                    }
                )
            }

            with patch.object(acejam_app, "_detect_bpm_key", return_value=(96, "G major")), \
                patch.object(acejam_app, "_search_lyrics_online", return_value="[Verse]\nReal lyric line"), \
                patch.object(acejam_app, "_musicbrainz_artist_tags", return_value=[]), \
                patch.object(acejam_app, "_resolve_local_llm_model_selection", return_value="qwen-local"), \
                patch.object(acejam_app, "local_llm_chat_completion_response", return_value=llm_response):
                result = acejam_app._training_lookup_online_lyrics(
                    audio,
                    {
                        "language": "en",
                        "genre_label_mode": "ai_auto",
                        "genre_label_provider": "lmstudio",
                        "genre_label_model": "qwen-local",
                    },
                )

            self.assertEqual(result["genre"], "country, americana")
            self.assertEqual(result["style_profile"], "country")
            self.assertEqual(result["genre_label_source"], "ai_local_llm")
            self.assertEqual(result["genre_label_provider"], "lmstudio")
            self.assertEqual(result["genre_label_model"], "qwen-local")
            self.assertIn("warm acoustic guitars", result["caption"])

    def test_official_runner_stream_keeps_conditioning_prompt_blocks_by_default(self):
        state = {}
        lines = [
            "2026-04-26 21:43:35 | INFO | acestep.core.generation.handler.conditioning_text:_prepare_text_conditioning_inputs:122 - text_prompt:\n",
            "# Caption\n",
            "SECRET CAPTION\n",
            "2026-04-26 21:43:35 | INFO | acestep.core.generation.handler.conditioning_text:_prepare_text_conditioning_inputs:124 - lyrics_text:\n",
            "# Lyric\n",
            "SECRET LYRIC\n",
            "2026-04-26 21:43:35 | INFO | acestep.core.generation.handler.service_generate_execute:_execute_service_generate_diffusion:137 - [service_generate] Generating audio... (DiT backend: MLX (native))\n",
        ]

        with patch.object(acejam_app, "ACEJAM_REDACT_OFFICIAL_LOG_TEXT", False):
            rendered = "".join(acejam_app._redact_official_runner_stream_line(line, state) for line in lines)

        self.assertIn("text_prompt:", rendered)
        self.assertIn("lyrics_text:", rendered)
        self.assertIn("DiT backend: MLX (native)", rendered)
        self.assertIn("SECRET CAPTION", rendered)
        self.assertIn("SECRET LYRIC", rendered)

    def test_official_runner_stream_redacts_conditioning_prompt_blocks_when_requested(self):
        state = {}
        lines = [
            "2026-04-26 21:43:35 | INFO | acestep.core.generation.handler.conditioning_text:_prepare_text_conditioning_inputs:122 - text_prompt:\n",
            "# Caption\n",
            "SECRET CAPTION\n",
            "2026-04-26 21:43:35 | INFO | acestep.core.generation.handler.conditioning_text:_prepare_text_conditioning_inputs:124 - lyrics_text:\n",
            "# Lyric\n",
            "SECRET LYRIC\n",
            "2026-04-26 21:43:35 | INFO | acestep.core.generation.handler.service_generate_execute:_execute_service_generate_diffusion:137 - [service_generate] Generating audio... (DiT backend: MLX (native))\n",
        ]

        with patch.object(acejam_app, "ACEJAM_REDACT_OFFICIAL_LOG_TEXT", True):
            rendered = "".join(acejam_app._redact_official_runner_stream_line(line, state) for line in lines)

        self.assertIn("conditioning prompt", rendered)
        self.assertIn("conditioning lyrics", rendered)
        self.assertIn("DiT backend: MLX (native)", rendered)
        self.assertNotIn("SECRET CAPTION", rendered)
        self.assertNotIn("SECRET LYRIC", rendered)

    def test_official_request_payload_prints_full_conditioning_payload_when_enabled(self):
        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            params = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "unit full caption, crisp vocal, clean mix",
                    "lyrics": "[Verse]\nUnit full lyric line one\n\n[Chorus]\nUnit full hook line",
                    "duration": 30,
                    "audio_format": "wav32",
                    "ace_lm_model": "none",
                    "thinking": False,
                    "use_format": False,
                    "use_cot_caption": False,
                    "use_cot_lyrics": False,
                }
            )

        with tempfile.TemporaryDirectory() as tmp, \
            patch.object(acejam_app, "ACEJAM_PRINT_ACE_PAYLOAD", True), \
            patch("builtins.print") as mocked_print:
            acejam_app._official_request_payload(params, Path(tmp))
            printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
            self.assertTrue((Path(tmp) / "ace_step_terminal_payload.txt").exists())
            self.assertTrue((Path(tmp) / "ace_step_terminal_payload.json").exists())

        self.assertIn("[ace_step_payload][BEGIN caption", printed)
        self.assertIn("[ace_step_payload][BEGIN lyrics", printed)
        self.assertIn("unit full caption", printed)
        self.assertIn("Unit full lyric line one", printed)

    def test_lm_backend_preserves_runtime_default_for_studio_payloads(self):
        expected_default = acejam_app.ACE_LM_BACKEND_DEFAULT
        expected_mlx = "mlx" if acejam_app._IS_APPLE_SILICON else expected_default
        self.assertEqual(acejam_app._normalize_lm_backend("mlx"), expected_mlx)
        self.assertEqual(acejam_app._normalize_lm_backend("auto"), expected_default)
        self.assertEqual(
            acejam_app._normalize_lm_backend("pt"),
            expected_default if acejam_app._IS_APPLE_SILICON else "pt",
        )

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-base"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            normalized = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-base",
                    "caption": "cinematic pop, polished vocal",
                    "lyrics": "[Verse]\nLine one\n\n[Chorus]\nHook line",
                    "lm_backend": "mlx",
                }
            )

        self.assertEqual(normalized["lm_backend"], expected_mlx)

    def test_mac_mlx_xl_repetition_guard_disables_known_bad_dcw_codes_combo(self):
        with patch.dict(os.environ, {"ACEJAM_MAC_MLX_XL_REPETITION_GUARD": "1"}), \
            patch.object(acejam_app, "_IS_APPLE_SILICON", True), \
            patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            normalized = acejam_app._parse_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-xl-sft",
                    "caption": "rap, hip hop, rhythmic spoken-word vocal",
                    "lyrics": "[Verse - rap]\nLine one lands in the pocket\n\n[Chorus - rap hook]\nHook line repeats clean",
                    "duration": 30,
                    "audio_backend": "mlx",
                    "use_mlx_dit": True,
                    "lm_backend": "mlx",
                    "dcw_enabled": True,
                    "audio_cover_strength": 1.0,
                }
            )

        self.assertFalse(normalized["dcw_enabled"])
        self.assertEqual(normalized["audio_cover_strength"], 0.0)
        self.assertTrue(
            any(str(item).startswith("mac_mlx_xl_repetition_guard:parse") for item in normalized["payload_warnings"])
        )

    def test_vendor_mlx_single_seed_patch_present(self):
        source = (Path(acejam_app.BASE_DIR) / "vendor" / "ACE-Step-1.5" / "acestep" / "llm_inference.py").read_text(encoding="utf-8")

        self.assertIn('"seeds": seeds', source)
        self.assertIn("seed=seeds[0] if seeds else None", source)
        self.assertIn("seed_base + step * 1000003", source)

    def test_lm_backend_alone_does_not_require_ace_lm(self):
        validation = acejam_app._validate_generation_payload(
            {
                "task_type": "text2music",
                "song_model": "acestep-v15-turbo",
                "caption": "rap, hard drums",
                "lyrics": "[Verse]\nA line for the beat\n\n[Chorus]\nHook for the street",
                "ace_lm_model": "none",
                "lm_backend": "pt",
            }
        )

        self.assertNotIn("ace_lm_model", validation["field_errors"])
        self.assertEqual(validation["normalized_payload"]["ace_lm_model"], "none")
        self.assertEqual(validation["settings_policy_version"], "ace-step-settings-parity-2026-04-26")
        self.assertIn("settings_compliance", validation)

    def test_album_description_does_not_become_sample_query(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-turbo",
            "title": "Description Safe",
            "description": "metadata scene for the album card",
            "caption": "rap, hard drums, cinematic bass",
            "lyrics": "[Verse]\nA real lyric line\n\n[Chorus]\nWe keep the route clean",
            "ace_lm_model": "none",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["sample_query"], "")
        self.assertTrue(normalized["requires_official_runner"])
        self.assertEqual(normalized["runner_plan"], "official")

    def test_studio_lm_policy_strips_ace_lm_controls(self):
        with patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            validation = acejam_app._validate_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-turbo",
                    "caption": "rap, hard drums",
                    "lyrics": "[Verse]\nThe rhythm carries\n[Chorus]\nWe ride the night",
                    "sample_query": "write a song",
                    "ace_lm_model": "auto",
                    "use_official_lm": True,
                }
            )

        self.assertNotIn("ace_lm_model", validation["field_errors"])
        self.assertEqual(validation["normalized_payload"]["ace_lm_model"], "none")
        self.assertEqual(validation["normalized_payload"]["sample_query"], "")

    def test_album_all_models_calls_advanced_generation_for_each_model(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            model = payload["song_model"]
            result_id = f"result-{len(calls):02d}"
            return {
                "success": True,
                "result_id": result_id,
                "active_song_model": model,
                "runner": "mock",
                "params": payload,
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": result_id,
                        "filename": acejam_app._numbered_audio_filename(
                            payload["title"],
                            model,
                            payload["audio_format"],
                            artist_name=payload.get("artist_name"),
                            track_number=1,
                            variant=1,
                        ),
                        "audio_url": f"/media/results/{result_id}/take-1.wav",
                        "download_url": f"/media/results/{result_id}/take-1.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        request_payload = {
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "all_models_album",
            "ace_lm_model": "none",
            "album_use_ace_lm_for_supplied_lyrics": False,
            "track_variants": 1,
            "save_to_library": False,
            "use_lora": False,
            "vocal_clarity_recovery": False,
            "tracks": [
                {
                    "track_number": 1,
                    "artist_name": "Unit Signal",
                    "title": "Seven Model Test",
                    "tags": (
                        "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                        "dynamic hook arrangement, crisp modern mix"
                    ),
                    "lyrics": (
                        "[Intro]\nQuiet click, the seven models wake\n"
                        "Bright route opens for the calm intake\n\n"
                        "[Verse]\nWe test the bright route under city lights\n"
                        "Every model enters clearly through the night\n"
                        "Clean chords carry the signal so the morning sounds right\n"
                        "The chorus waits patient for the open release\n"
                        "Steady piano walking down a calm street\n"
                        "Hook coming forward like an honest heartbeat\n"
                        "Polish on the master, every meter in line\n"
                        "Bright route lifting smooth and fine\n\n"
                        "[Pre-Chorus]\nLights line up, cue the wave, cue the wave\n"
                        "Every meter ready, give the mic a save\n\n"
                        "[Chorus]\nEvery model plays it loud, every model rides the cloud\n"
                        "Every take keeps timing proud, every voice declared aloud\n"
                        "Seven engines share the light, Unit Signal rides tonight\n"
                        "Bright route open, clean and tight, every render lands just right\n"
                        "Hook on top of every wave, hook on top of every save\n\n"
                        "[Verse 2]\nNight moves through the master, takes a steady breath\n"
                        "Compressor holding gentle, never loses depth\n"
                        "Mics all warm and honest, every tail decay\n"
                        "Bright route polished smooth, never gets in the way\n\n"
                        "[Chorus]\nEvery model plays it loud, every model rides the cloud\n"
                        "Every take keeps timing proud, every voice declared aloud\n\n"
                        "[Outro]\nThe final note stays clean and warm\n"
                        "Seven paths land bright through the storm\n"
                        "Last chord settle, last meter calm\n"
                    ),
                    "duration": 60,
                    "bpm": 120,
                    "key_scale": "C minor",
                    "time_signature": "4",
                }
            ],
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value=set(acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS)), \
            patch.object(album_crew_module, "plan_album", return_value=self._mock_direct_album_plan(request_payload["tracks"])), \
            patch.object(acejam_app, "_validate_generation_payload", return_value={"valid": True, "payload_warnings": []}), \
            patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation), \
            patch.object(acejam_app, "_write_album_manifest", side_effect=lambda album_id, manifest: {**manifest, "album_id": album_id}):
            raw = acejam_app.generate_album(
                concept="unit test album",
                num_tracks=1,
                track_duration=30,
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertTrue(data["success"])
        self.assertEqual(len(calls), len(acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS))
        self.assertEqual([payload["song_model"] for payload in calls], acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS)
        self.assertTrue(all(payload["artist_name"] == "Unit Signal" for payload in calls))
        self.assertTrue(all(payload["ace_lm_model"] == "none" for payload in calls))
        self.assertTrue(all(not payload["thinking"] for payload in calls))
        self.assertTrue(all(not payload["use_format"] for payload in calls))
        self.assertTrue(all(not payload["use_cot_metas"] for payload in calls))
        self.assertTrue(all(not payload["use_cot_caption"] for payload in calls))
        self.assertTrue(all(not payload["use_cot_lyrics"] for payload in calls))
        self.assertTrue(all(not payload["use_cot_language"] for payload in calls))
        self.assertTrue(all(payload.get("audio_code_string", "") == "" for payload in calls))
        self.assertTrue(all(payload.get("src_audio_id", "") == "" for payload in calls))
        self.assertTrue(all(payload.get("reference_audio_id", "") == "" for payload in calls))
        self.assertTrue(all(not payload["use_lora"] for payload in calls))
        self.assertEqual(len(data["model_albums"]), len(acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS))
        self.assertIn("album_family_id", data)
        self.assertEqual(data["tracks"][0]["model_results"][0]["album_model"], acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS[0])

    def test_album_single_model_applies_lora_trigger_scale_to_each_track(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            return {
                "success": True,
                "result_id": "album-lora-01",
                "active_song_model": payload["song_model"],
                "runner": "mock",
                "params": {
                    **payload,
                    "lora_trigger_applied": True,
                    "lora_trigger_conditioning_audit": {"status": "applied", "caption_only": True},
                },
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": "album-lora-01",
                        "filename": "take.wav",
                        "audio_url": "/media/results/album-lora-01/take.wav",
                        "download_url": "/media/results/album-lora-01/take.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        request_payload = {
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "single_model_album",
            "song_model": "acestep-v15-xl-sft",
            "ace_lm_model": "none",
            "track_variants": 1,
            "save_to_library": False,
            "use_lora": True,
            "lora_adapter_path": "/tmp/album-charaf-hook",
            "lora_adapter_name": "charaf hook",
            "use_lora_trigger": True,
            "lora_trigger_tag": "pac",
            "lora_scale": 1.0,
            "adapter_song_model": "acestep-v15-xl-sft",
            "tracks": [
                {
                    "track_number": 1,
                    "artist_name": "Unit Signal",
                    "title": "LoRA Album Test",
                    "tags": "rap, hip hop, rhythmic spoken-word vocal, hard drums, polished mix",
                    "lyrics": (
                        "[Verse - rap]\n"
                        + "\n".join(f"Bar {idx} lands with pressure in the skyline tonight" for idx in range(16))
                        + "\n[Chorus - rap hook]\n"
                        + "\n".join(f"Hook keeps calling from the avenue line {idx}" for idx in range(6))
                        + "\n[Verse 2 - rap]\n"
                        + "\n".join(f"Second verse keeps the cadence locked in time {idx}" for idx in range(10))
                        + "\n[Outro]\nFade the drums but keep the message moving\n"
                    ),
                    "duration": 60,
                    "bpm": 92,
                    "key_scale": "D minor",
                    "time_signature": "4",
                }
            ],
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(album_crew_module, "plan_album", return_value=self._mock_direct_album_plan(request_payload["tracks"])), \
            patch.object(acejam_app, "_validate_generation_payload", return_value={"valid": True, "payload_warnings": []}), \
            patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation), \
            patch.object(acejam_app, "_write_album_manifest", side_effect=lambda album_id, manifest: {**manifest, "album_id": album_id}):
            raw = acejam_app.generate_album(
                concept="unit test album lora",
                num_tracks=1,
                track_duration=60,
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertTrue(data["success"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["song_model"], "acestep-v15-xl-sft")
        self.assertTrue(calls[0]["use_lora"])
        self.assertEqual(calls[0]["lora_adapter_name"], "charaf hook")
        self.assertEqual(calls[0]["lora_scale"], 1.0)
        self.assertEqual(calls[0]["lora_trigger_tag"], "pac")
        self.assertTrue(calls[0]["lora_trigger_applied"])
        self.assertEqual(calls[0]["adapter_song_model"], "acestep-v15-xl-sft")
        self.assertEqual(data["tracks"][0]["lora_adapter_name"], "charaf hook")
        self.assertEqual(data["tracks"][0]["lora_scale"], 1.0)
        self.assertEqual(data["tracks"][0]["lora_trigger_tag"], "pac")
        self.assertTrue(data["tracks"][0]["lora_trigger_applied"])

    def test_album_lora_model_mismatch_blocks_before_render(self):
        request_payload = {
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "single_model_album",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "track_variants": 1,
            "save_to_library": False,
            "use_lora": True,
            "lora_adapter_path": "/tmp/album-xl-sft",
            "lora_adapter_name": "xl sft lora",
            "lora_scale": 1.0,
            "adapter_song_model": "acestep-v15-xl-sft",
            "tracks": [
                {
                    "track_number": 1,
                    "artist_name": "Unit Signal",
                    "title": "Mismatch",
                    "tags": "rap, hard drums",
                    "lyrics": "[Verse - rap]\nWe test the mismatch before render",
                    "duration": 60,
                }
            ],
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo", "acestep-v15-xl-sft"}):
            raw = acejam_app.generate_album(
                concept="unit test mismatch",
                num_tracks=1,
                track_duration=60,
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertFalse(data["success"])
        self.assertIn("Selected LoRA was trained for acestep-v15-xl-sft", data["error"])

    def test_album_skips_planning_failed_track_and_renders_remaining_tracks(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            return {
                "success": True,
                "result_id": "partial-album-01",
                "active_song_model": payload["song_model"],
                "runner": "mock",
                "params": payload,
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": "partial-album-01",
                        "filename": "take.wav",
                        "audio_url": "/media/results/partial-album-01/take.wav",
                        "download_url": "/media/results/partial-album-01/take.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        request_payload = {
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "selected",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "track_variants": 1,
            "save_to_library": False,
            "tracks": [
                {
                    "track_number": 1,
                    "title": "Failed First",
                    "tags": "West Coast hip-hop, boom-bap drums, 808 bass, male rap vocal, polished mix",
                    "lyrics": "",
                    "duration": 30,
                },
                {
                    "track_number": 2,
                    "title": "Second Still Renders",
                    "tags": "West Coast hip-hop, boom-bap drums, 808 bass, male rap vocal, polished mix",
                    "lyrics": "[Verse]\nSecond track keeps the signal live\n[Chorus]\nSecond track keeps the signal live\n[Outro]\nSecond track lands clear",
                    "duration": 30,
                    "bpm": 92,
                    "key_scale": "A minor",
                    "time_signature": "4",
                },
            ],
        }
        planned = self._mock_direct_album_plan(request_payload["tracks"])
        planned["tracks"][0].update(
            {
                "planning_status": "failed",
                "skip_render": True,
                "planning_error": "Hook Agent failed semantic validation",
                "error": "Hook Agent failed semantic validation",
                "agent_complete_payload": False,
                "payload_gate_status": "planning_failed",
                "payload_gate_passed": False,
            }
        )
        planned["planning_status"] = "partial"
        planned["planning_failed_count"] = 1
        planned["planning_failures"] = [{"track_number": 1, "title": "Failed First", "error": "Hook Agent failed semantic validation"}]

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}), \
            patch.object(album_crew_module, "plan_album", return_value=planned), \
            patch.object(acejam_app, "_validate_direct_album_agent_payload", return_value={"gate_passed": True, "status": "pass", "issues": [], "blocking_issues": []}), \
            patch.object(acejam_app, "_validate_generation_payload", return_value={"valid": True, "payload_warnings": []}), \
            patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation), \
            patch.object(acejam_app, "_write_album_manifest", side_effect=lambda album_id, manifest: {**manifest, "album_id": album_id}):
            raw = acejam_app.generate_album(
                concept="unit test album",
                num_tracks=2,
                track_duration=30,
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertFalse(data["success"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["title"], "Second Still Renders")
        self.assertEqual(data["generated_count"], 1)
        self.assertEqual(data["expected_renders"], 2)
        self.assertEqual(data["failed_count"], 1)
        self.assertIn("Failed tracks", data["error"])
        self.assertEqual(data["model_albums"][0]["tracks"][0]["planning_status"], "failed")
        self.assertTrue(data["model_albums"][0]["tracks"][1]["generated"])

    def test_album_job_publishes_full_tracks_as_they_finish(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            result_id = f"album-progress-{len(calls)}"
            return {
                "success": True,
                "result_id": result_id,
                "active_song_model": payload["song_model"],
                "runner": "mock",
                "params": payload,
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": result_id,
                        "filename": "take.wav",
                        "audio_url": f"/media/results/{result_id}/take.wav",
                        "download_url": f"/media/results/{result_id}/take.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        request_payload = {
            "album_job_id": "album-progress-job",
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "selected",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "track_variants": 1,
            "save_to_library": False,
            "tracks": [
                {
                    "track_number": 1,
                    "title": "First Full Track",
                    "tags": "rap, hip hop, rhythmic spoken-word vocal, hard drums",
                    "lyrics": _LONG_TEST_LYRICS,
                    "duration": 30,
                    "bpm": 92,
                    "key_scale": "A minor",
                    "time_signature": "4",
                },
                {
                    "track_number": 2,
                    "title": "Second Full Track",
                    "tags": "rap, hip hop, rhythmic spoken-word vocal, hard drums",
                    "lyrics": _LONG_TEST_LYRICS,
                    "duration": 30,
                    "bpm": 92,
                    "key_scale": "A minor",
                    "time_signature": "4",
                },
            ],
        }
        updates = []
        original_set_album_job = acejam_app._set_album_job

        def capture_album_job(job_id, **kwargs):
            updates.append((job_id, kwargs))
            return original_set_album_job(job_id, **kwargs)

        try:
            with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}), \
                patch.object(album_crew_module, "plan_album", return_value=self._mock_direct_album_plan(request_payload["tracks"])), \
                patch.object(acejam_app, "_validate_direct_album_agent_payload", return_value={"gate_passed": True, "status": "pass", "issues": [], "blocking_issues": []}), \
                patch.object(acejam_app, "_validate_generation_payload", return_value={"valid": True, "payload_warnings": []}), \
                patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation), \
                patch.object(acejam_app, "_write_album_manifest", side_effect=lambda album_id, manifest: {**manifest, "album_id": album_id}), \
                patch.object(acejam_app, "_set_album_job", side_effect=capture_album_job):
                raw = acejam_app.generate_album(
                    concept="unit test album",
                    num_tracks=2,
                    track_duration=30,
                    request_json=json.dumps(request_payload),
                )
        finally:
            acejam_app._album_jobs.pop("album-progress-job", None)

        data = json.loads(raw)
        self.assertTrue(data["success"])
        progress_results = [
            kwargs["result"]
            for _, kwargs in updates
            if isinstance(kwargs.get("result"), dict) and kwargs["result"].get("full_tracks_ready")
        ]
        self.assertGreaterEqual(len(progress_results), 2)
        self.assertEqual(progress_results[0]["full_tracks_ready"], 1)
        self.assertEqual(progress_results[0]["completed_audio_count"], 1)
        self.assertEqual(progress_results[0]["audios"][0]["audio_url"], "/media/results/album-progress-1/take.wav")
        self.assertEqual(progress_results[-1]["full_tracks_ready"], 2)
        self.assertEqual(progress_results[-1]["completed_audio_count"], 2)

    def test_album_supplied_vocal_lyrics_can_explicitly_disable_ace_lm_formatting(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            return {
                "success": True,
                "result_id": "album-lm-01",
                "active_song_model": payload["song_model"],
                "runner": "mock",
                "params": payload,
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": "album-lm-01",
                        "filename": "take.wav",
                        "audio_url": "/media/results/album-lm-01/take.wav",
                        "download_url": "/media/results/album-lm-01/take.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        request_payload = {
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "selected",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": acejam_app.ACE_LM_PREFERRED_MODEL,
            "thinking": True,
            "use_format": True,
            "use_cot_caption": True,
            "vocal_clarity_recovery": False,
            "track_variants": 1,
            "save_to_library": False,
            "tracks": [
                {
                    "track_number": 1,
                    "artist_name": "Unit Signal",
                    "title": "Format The Hook",
                    "tags": (
                        "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                        "dynamic hook arrangement, crisp modern mix"
                    ),
                    "lyrics": _LONG_TEST_LYRICS,
                    "duration": 30,
                    "bpm": 120,
                    "key_scale": "C minor",
                    "time_signature": "4",
                }
            ],
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}), \
            patch.object(album_crew_module, "plan_album", return_value=self._mock_direct_album_plan(request_payload["tracks"])), \
            patch.object(acejam_app, "_validate_generation_payload", return_value={"valid": True, "payload_warnings": []}), \
            patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation), \
            patch.object(acejam_app, "_write_album_manifest", side_effect=lambda album_id, manifest: {**manifest, "album_id": album_id}):
            raw = acejam_app.generate_album(
                concept="unit test album",
                num_tracks=1,
                track_duration=30,
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertTrue(data["success"])
        self.assertEqual(len(calls), 1)
        payload = calls[0]
        self.assertEqual(payload["ace_lm_model"], "none")
        self.assertFalse(payload["allow_supplied_lyrics_lm"])
        self.assertFalse(payload["thinking"])
        self.assertFalse(payload["use_format"])
        self.assertFalse(payload["use_cot_caption"])
        self.assertFalse(payload["sample_mode"])
        self.assertFalse(payload["use_cot_metas"])
        self.assertFalse(payload["use_cot_lyrics"])
        self.assertFalse(payload["use_cot_language"])
        self.assertEqual(payload["lm_backend"], acejam_app._normalize_lm_backend(acejam_app.ACE_LM_BACKEND_DEFAULT))

    def test_direct_agent_album_rewrite_respects_explicit_clarity_optout(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            return {
                "success": True,
                "result_id": "album-lm-override-01",
                "active_song_model": payload["song_model"],
                "runner": "mock",
                "params": payload,
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": "album-lm-override-01",
                        "filename": "take.wav",
                        "audio_url": "/media/results/album-lm-override-01/take.wav",
                        "download_url": "/media/results/album-lm-override-01/take.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        request_payload = {
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "selected",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "album_allow_ace_lm_rewrite": True,
            "vocal_clarity_recovery": False,
            "track_variants": 1,
            "save_to_library": False,
            "tracks": [
                {
                    "track_number": 1,
                    "artist_name": "Unit Signal",
                    "title": "Format The Hook",
                    "tags": "pop, steady groove, piano, clear lead vocal, uplifting mood, dynamic hook arrangement, crisp modern mix",
                    "lyrics": _LONG_TEST_LYRICS,
                    "duration": 30,
                    "bpm": 120,
                    "key_scale": "C minor",
                    "time_signature": "4",
                }
            ],
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}), \
            patch.object(album_crew_module, "plan_album", return_value=self._mock_direct_album_plan(request_payload["tracks"])), \
            patch.object(acejam_app, "_validate_generation_payload", return_value={"valid": True, "payload_warnings": []}), \
            patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation), \
            patch.object(acejam_app, "_write_album_manifest", side_effect=lambda album_id, manifest: {**manifest, "album_id": album_id}):
            raw = acejam_app.generate_album(
                concept="unit test album",
                num_tracks=1,
                track_duration=30,
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertTrue(data["success"])
        payload = calls[0]
        self.assertEqual(payload["ace_lm_model"], "none")
        self.assertFalse(payload["allow_supplied_lyrics_lm"])
        self.assertFalse(payload["thinking"])
        self.assertFalse(payload["use_format"])
        self.assertFalse(payload["use_cot_caption"])

    def test_album_auto_vocal_clarity_recovery_keeps_direct_lyrics_render_for_agent_payload(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            return {
                "success": True,
                "result_id": "album-clarity-01",
                "active_song_model": payload["song_model"],
                "runner": "mock",
                "params": payload,
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": "album-clarity-01",
                        "filename": "take.wav",
                        "audio_url": "/media/results/album-clarity-01/take.wav",
                        "download_url": "/media/results/album-clarity-01/take.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        request_payload = {
            "agent_engine": "editable_plan",
            "toolbelt_only": True,
            "song_model_strategy": "selected",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "track_variants": 1,
            "save_to_library": False,
            "tracks": [
                {
                    "track_number": 1,
                    "artist_name": "Unit Signal",
                    "title": "Clear The Hook",
                    "tags": "West Coast rap, boom-bap drums, 808 bass, piano sample motif, male rap vocal, dynamic hook response, gritty street texture, punchy polished mix",
                    "lyrics": _LONG_TEST_LYRICS,
                    "duration": 30,
                    "bpm": 120,
                    "key_scale": "C minor",
                    "time_signature": "4",
                }
            ],
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}), \
            patch.object(album_crew_module, "plan_album", return_value=self._mock_direct_album_plan(request_payload["tracks"])), \
            patch.object(acejam_app, "_validate_generation_payload", return_value={"valid": True, "payload_warnings": []}), \
            patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation), \
            patch.object(acejam_app, "_write_album_manifest", side_effect=lambda album_id, manifest: {**manifest, "album_id": album_id}):
            raw = acejam_app.generate_album(
                concept="unit test album",
                num_tracks=1,
                track_duration=30,
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertTrue(data["success"])
        payload = calls[0]
        self.assertEqual(payload["ace_lm_model"], "none")
        self.assertFalse(payload["allow_supplied_lyrics_lm"])
        self.assertFalse(payload["thinking"])
        self.assertFalse(payload["use_format"])
        self.assertFalse(payload["use_cot_caption"])
        self.assertFalse(payload["use_cot_metas"])
        self.assertFalse(payload["use_cot_lyrics"])
        self.assertFalse(payload["use_cot_language"])
        self.assertTrue(payload["vocal_clarity_recovery"])

    def test_vocal_clarity_recovery_normalizes_caption_without_lm_defaults(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "vocal_clarity_recovery": True,
            "title": "Clear Hook",
            "artist_name": "Unit Signal",
            "caption": "West Coast rap, piano, drums, male rap vocal, polished mix",
            "lyrics": "[Verse]\nWe test the bright route\n[Chorus]\nThe hook lands clean",
            "duration": 30,
            "bpm": 120,
            "key_scale": "C minor",
            "time_signature": "4",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["ace_lm_model"], "none")
        self.assertFalse(normalized["thinking"])
        self.assertFalse(normalized["use_format"])
        self.assertFalse(normalized["use_cot_caption"])
        self.assertFalse(normalized["use_cot_language"])
        self.assertFalse(normalized["use_cot_metas"])
        self.assertFalse(normalized["use_cot_lyrics"])
        self.assertIn("clear intelligible English rap vocal", normalized["caption"])
        self.assertIn("vocal_clarity_recovery_caption_traits", normalized["payload_warnings"])

    def test_vocal_clarity_recovery_keeps_false_ui_lm_switches(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "vocal_clarity_recovery": True,
            "title": "Clear UI Hook",
            "artist_name": "Unit Signal",
            "caption": "cinematic rap, male vocal, polished mix",
            "lyrics": "[Verse]\nEvery word is clear\n[Chorus]\nAlbus in the light",
            "duration": 30,
            "bpm": 95,
            "key_scale": "A minor",
            "time_signature": "4",
            "thinking": False,
            "use_format": False,
            "use_cot_caption": False,
            "use_cot_language": False,
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["ace_lm_model"], "none")
        self.assertFalse(normalized["thinking"])
        self.assertFalse(normalized["use_format"])
        self.assertFalse(normalized["use_cot_caption"])
        self.assertFalse(normalized["use_cot_language"])
        self.assertFalse(normalized["use_cot_metas"])
        self.assertFalse(normalized["use_cot_lyrics"])

    def test_album_options_preserve_selected_model_from_payload(self):
        opts = acejam_app._album_options_from_payload(
            {
                "song_model_strategy": "selected",
                "song_model": "acestep-v15-xl-sft",
                "requested_song_model": "acestep-v15-xl-sft",
            },
            song_model="auto",
        )

        self.assertEqual(opts["song_model_strategy"], "selected")
        self.assertEqual(opts["requested_song_model"], "acestep-v15-xl-sft")

    def test_album_options_extract_contract_from_concept_payload(self):
        opts = acejam_app._album_options_from_payload(
            {
                "concept": (
                    'Album: Midnight Bakery\n'
                    'Track 1: "Neon Bakery Lights" (Produced by Studio House)\n'
                    "(BPM: 95 | Key: A minor | Style: upbeat city-pop / pop-funk)\n"
                    "The Vibe: rubber bass and bright piano.\n"
                    "The Narrative: the lights come back on."
                ),
                "num_tracks": 1,
                "language": "en",
                "song_model_strategy": "selected",
                "song_model": "acestep-v15-turbo",
            },
            song_model="auto",
        )

        contract = opts["user_album_contract"]
        self.assertTrue(contract["applied"])
        self.assertEqual(contract["album_title"], "Midnight Bakery")
        self.assertEqual(contract["tracks"][0]["locked_title"], "Neon Bakery Lights")
        self.assertEqual(contract["tracks"][0]["producer_credit"], "Studio House")
        self.assertEqual(contract["tracks"][0]["key_scale"], "A minor")

    def test_album_options_prefers_clean_user_prompt_over_polluted_generated_concept(self):
        clean_prompt = (
            "Album: You Buried the Wrong Man\n"
            "Track 1: Concrete Canyons (Prod. Dr. Dre)\n"
            "Vibe: Low-end rumble, sirens, West Coast weight\n"
            "Verse: They paved them blocks just to hide what's real,\n"
            "Naming Drop Style: \"Death Row\""
        )
        polluted_concept = (
            clean_prompt
            + "\nTrack 1: \"Concrete Canyons West Coast rap with dark orchestral elements [Intro] leaked old lyrics [Verse 1] more leaked text\"\n"
            + "[Chorus]\nInstrumental break\n[Outro]\nStrings fade away"
        )

        opts = acejam_app._album_options_from_payload(
            {
                "raw_user_prompt": clean_prompt,
                "user_prompt": clean_prompt,
                "concept": polluted_concept,
                "tracks": [{"title": "Generated Old Title", "lyrics": "[Intro]\nold generated lyrics"}],
                "num_tracks": 1,
                "language": "en",
                "song_model_strategy": "selected",
                "song_model": "acestep-v15-xl-sft",
            },
            song_model="auto",
        )

        contract = opts["user_album_contract"]
        self.assertEqual(len(contract["tracks"]), 1)
        self.assertEqual(contract["tracks"][0]["locked_title"], "Concrete Canyons")
        self.assertNotIn("Generated Old Title", contract["concept"])
        self.assertNotIn("Instrumental break", contract["concept"])

    def test_generate_album_selected_model_queues_download_instead_of_empty_strategy_error(self):
        request_payload = {
            "song_model_strategy": "selected",
            "song_model": "acestep-v15-xl-sft",
            "requested_song_model": "acestep-v15-xl-sft",
            "tracks": [],
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo"}), \
            patch.object(acejam_app, "_start_model_download", return_value={"id": "download-xl-sft", "status": "queued"}):
            raw = acejam_app.generate_album(
                concept="safe selected album",
                num_tracks=1,
                track_duration=30,
                song_model="auto",
                request_json=json.dumps(request_payload),
            )

        data = json.loads(raw)
        self.assertEqual(data["download_models"], ["acestep-v15-xl-sft"])
        self.assertNotIn("No album models resolved", json.dumps(data))

    def test_album_scaffold_respects_ai_per_track_durations(self):
        logs: list[str] = []
        scaffold = album_crew_module._build_album_track_scaffold(
            concept="smart duration album",
            num_tracks=3,
            track_duration=180,
            language="en",
            opts={"duration_mode": "ai_per_track"},
            contract={},
            bible_payload={
                "tracks": [
                    {"track_number": 1, "title": "Intro", "role": "intro", "duration": 90},
                    {"track_number": 2, "title": "Single", "role": "single", "duration": 210},
                    {"track_number": 3, "title": "Outro", "role": "outro", "duration": 75},
                ]
            },
            logs=logs,
        )

        self.assertEqual([track["duration"] for track in scaffold], [90.0, 210.0, 75.0])
        self.assertFalse([line for line in logs if "Ignored agent duration hint" in line])

    def test_album_scaffold_fixed_duration_forces_fallback(self):
        logs: list[str] = []
        scaffold = album_crew_module._build_album_track_scaffold(
            concept="fixed duration album",
            num_tracks=2,
            track_duration=180,
            language="en",
            opts={"duration_mode": "fixed"},
            contract={},
            bible_payload={
                "tracks": [
                    {"track_number": 1, "title": "Intro", "role": "intro", "duration": 90},
                    {"track_number": 2, "title": "Single", "role": "single", "duration": 210},
                ]
            },
            logs=logs,
        )

        self.assertEqual([track["duration"] for track in scaffold], [180.0, 180.0])
        self.assertTrue([line for line in logs if "Ignored agent duration hint" in line])

    def test_album_download_filename_contains_track_title_and_model(self):
        filename = acejam_app._numbered_audio_filename(
            "My Big Hook!",
            "acestep-v15-xl-sft",
            "wav",
            artist_name="The Artist",
            track_number=2,
            variant=3,
        )
        fallback = acejam_app._numbered_audio_filename("My Big Hook!", "acestep-v15-xl-sft", "wav", track_number=2, variant=3)

        self.assertEqual(filename, "02-The-Artist--My-Big-Hook--xl-sft--v3.wav")
        self.assertEqual(fallback, "02-MLX-Media--My-Big-Hook--xl-sft--v3.wav")

    def test_ollama_album_defaults_prefer_selected_27b_and_embedding(self):
        catalog = {
            "ready": True,
            "chat_models": [acejam_app.DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL, "other:latest"],
            "embedding_models": [acejam_app.DEFAULT_ALBUM_EMBEDDING_MODEL],
            "models": [acejam_app.DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL, acejam_app.DEFAULT_ALBUM_EMBEDDING_MODEL],
        }

        with patch.object(acejam_app, "_ollama_model_catalog", return_value=catalog):
            self.assertEqual(
                acejam_app._resolve_ollama_model_selection("", "chat", "album planning"),
                acejam_app.DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            )
            self.assertEqual(
                acejam_app._resolve_ollama_model_selection("", "embedding", "album embeddings"),
                acejam_app.DEFAULT_ALBUM_EMBEDDING_MODEL,
            )

    def test_album_job_worker_records_progress_and_result(self):
        job_id = "unitjob123"
        payload = {
            "concept": "unit album",
            "num_tracks": 1,
            "track_duration": 180,
            "ollama_model": acejam_app.DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            "embedding_model": acejam_app.DEFAULT_ALBUM_EMBEDDING_MODEL,
        }

        fake_result = {
            "success": True,
            "album_id": "album-one",
            "album_family_id": "family-one",
            "family_download_url": "/api/album-families/family-one/download",
            "audios": [{"audio_url": "/x.wav"}],
            "tracks": [{"title": "One"}],
            "generated_count": 7,
            "expected_renders": 7,
            "logs": ["done"],
        }

        with patch.object(acejam_app, "generate_album", return_value=json.dumps(fake_result)):
            acejam_app._album_job_worker(job_id, payload)

        job = acejam_app._album_job_snapshot(job_id)
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(job["planner_model"], acejam_app.DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL)
        self.assertEqual(job["embedding_model"], acejam_app.DEFAULT_ALBUM_EMBEDDING_MODEL)
        self.assertFalse(job["memory_enabled"])
        self.assertEqual(job["album_family_id"], "family-one")
        self.assertEqual(job["download_url"], "/api/album-families/family-one/download")

    def test_album_plan_job_worker_records_tracks_without_generation(self):
        job_id = "planjob123"
        payload = {
            "concept": "unit album",
            "num_tracks": 2,
            "track_duration": 180,
            "ollama_model": acejam_app.DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            "embedding_model": acejam_app.DEFAULT_ALBUM_EMBEDDING_MODEL,
        }
        fake_result = {
            "success": True,
            "tracks": [{"title": "One"}, {"title": "Two"}],
            "logs": ["planned"],
            "planner_model": acejam_app.DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            "embedding_model": acejam_app.DEFAULT_ALBUM_EMBEDDING_MODEL,
            "planning_engine": "acejam_agents",
            "custom_agents_used": True,
            "crewai_used": False,
            "toolbelt_fallback": False,
            "agent_debug_dir": "/tmp/album_plan_debug",
        }

        with patch.object(acejam_app, "_run_album_plan_from_payload", return_value=fake_result):
            acejam_app._album_plan_job_worker(job_id, payload)

        job = acejam_app._album_job_snapshot(job_id)
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(job["job_type"], "album_plan")
        self.assertEqual(job["planned_count"], 2)
        self.assertEqual(job["result"]["tracks"][0]["title"], "One")
        self.assertEqual(job["planning_engine"], "acejam_agents")
        self.assertTrue(job["custom_agents_used"])
        self.assertFalse(job["crewai_used"])
        self.assertFalse(job["toolbelt_fallback"])
        self.assertEqual(job["agent_debug_dir"], "/tmp/album_plan_debug")

    def test_song_portfolio_renders_every_model_with_model_defaults(self):
        calls = []

        def fake_generation(payload):
            calls.append(dict(payload))
            model = payload["song_model"]
            return {
                "success": True,
                "result_id": f"portfolio-{len(calls)}",
                "active_song_model": model,
                "runner": "mock",
                "params": payload,
                "payload_warnings": [],
                "audios": [
                    {
                        "id": "take-1",
                        "result_id": f"portfolio-{len(calls)}",
                        "filename": acejam_app._numbered_audio_filename(
                            payload["title"],
                            model,
                            payload["audio_format"],
                            artist_name=payload.get("artist_name"),
                            variant=1,
                        ),
                        "audio_url": f"/media/results/portfolio-{len(calls)}/take.wav",
                        "download_url": f"/media/results/portfolio-{len(calls)}/take.wav",
                        "title": payload["title"],
                        "seed": payload["seed"],
                    }
                ],
            }

        payload = {
            "ui_mode": "custom",
            "task_type": "text2music",
            "artist_name": "Portfolio Pulse",
            "title": "Portfolio Hook",
            "caption": "pop rap, bright hook, crisp modern mix",
            "lyrics": "[Verse]\nWe build it clean\n\n[Chorus]\nEvery model gets the scene",
            "duration": 30,
            "bpm": 120,
            "key_scale": "C minor",
            "time_signature": "4",
            "audio_format": "wav",
            "save_to_library": False,
            "thinking": True,
            "sample_mode": True,
            "sample_query": "rewrite this into a different sample",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value=set(acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS)), \
            patch.object(acejam_app, "_run_advanced_generation", side_effect=fake_generation):
            data = acejam_app._run_model_portfolio_generation(payload)

        self.assertTrue(data["success"])
        self.assertEqual([item["song_model"] for item in calls], acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS)
        self.assertTrue(all(item["artist_name"] == "Portfolio Pulse" for item in calls))
        self.assertEqual(len(data["audios"]), len(acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS))
        self.assertTrue(all(item["batch_size"] == 1 for item in calls))
        self.assertTrue(all(item["ace_lm_model"] == "none" for item in calls))
        self.assertTrue(all(not item["thinking"] for item in calls))
        self.assertTrue(all(not item["sample_mode"] for item in calls))
        self.assertTrue(all(item["sample_query"] == "" for item in calls))
        self.assertTrue(all(not item["use_cot_lyrics"] for item in calls))
        by_model = {item["song_model"]: item for item in calls}
        self.assertEqual(by_model["acestep-v15-turbo"]["inference_steps"], 8)
        self.assertEqual(by_model["acestep-v15-turbo-shift1"]["inference_steps"], 8)
        self.assertEqual(by_model["acestep-v15-turbo-continuous"]["inference_steps"], 8)
        self.assertEqual(by_model["acestep-v15-sft"]["inference_steps"], 50)
        self.assertEqual(by_model["acestep-v15-turbo"]["guidance_scale"], 7.0)
        self.assertEqual(by_model["acestep-v15-sft"]["guidance_scale"], 7.0)
        self.assertEqual(by_model["acestep-v15-turbo"]["shift"], 3.0)
        self.assertEqual(by_model["acestep-v15-sft"]["shift"], 1.0)
        self.assertEqual(data["render_strategy"], "all_models_song")

    def test_delete_generated_outputs_preserves_uploads_and_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            songs = root / "songs"
            results = root / "results"
            albums = root / "albums"
            uploads = root / "uploads"
            loras = root / "loras"
            for folder in [songs, results, albums, uploads, loras]:
                folder.mkdir(parents=True)
            (songs / "song1").mkdir()
            (songs / "song1" / "meta.json").write_text(json.dumps({"id": "song1", "result_id": "result1"}), encoding="utf-8")
            (songs / "song1" / "song.wav").write_text("audio", encoding="utf-8")
            (results / "result1").mkdir()
            (results / "result1" / "take.wav").write_text("audio", encoding="utf-8")
            (albums / "family1").mkdir()
            (albums / "family1" / "album.json").write_text("{}", encoding="utf-8")
            (uploads / "keep").mkdir()
            (uploads / "keep" / "source.wav").write_text("upload", encoding="utf-8")
            (loras / "adapter.safetensors").write_text("adapter", encoding="utf-8")

            with patch.object(acejam_app, "SONGS_DIR", songs), \
                patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "ALBUMS_DIR", albums), \
                patch.object(acejam_app, "UPLOADS_DIR", uploads), \
                patch.object(acejam_app, "LORA_EXPORTS_DIR", loras), \
                patch.object(acejam_app, "_feed_songs", [{"id": "song1"}]):
                before = acejam_app._count_generated_outputs()
                self.assertEqual(before["songs"], 1)
                self.assertEqual(before["results"], 1)
                self.assertEqual(before["albums"], 1)
                with self.assertRaises(Exception):
                    acejam_app._delete_generated_outputs("WRONG")
                deleted = acejam_app._delete_generated_outputs("DELETE_GENERATED_OUTPUTS")

            self.assertTrue(deleted["success"])
            self.assertTrue((uploads / "keep" / "source.wav").is_file())
            self.assertTrue((loras / "adapter.safetensors").is_file())
            self.assertEqual(list(songs.iterdir()), [])
            self.assertEqual(list(results.iterdir()), [])
            self.assertEqual(list(albums.iterdir()), [])

    def test_ace_lm_cleanup_preview_requires_smoked_abliterated_4b(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            original = checkpoints / "acestep-5Hz-lm-4B"
            original.mkdir()
            (original / "config.json").write_text("{}", encoding="utf-8")
            (original / "model.safetensors").write_text("weights", encoding="utf-8")

            abliterated_root = root / "ace_lm_abliterated"
            abliterated_root.mkdir()
            abliterated = checkpoints / "acestep-5Hz-lm-4B-abliterated"
            abliterated.mkdir()
            (abliterated / "config.json").write_text("{}", encoding="utf-8")
            (abliterated / "model.safetensors").write_text("weights", encoding="utf-8")

            with patch.object(acejam_app, "MODEL_CACHE_DIR", root), \
                patch.object(acejam_app, "ACE_LM_ABLITERATED_DIR", abliterated_root):
                locked = acejam_app._ace_lm_cleanup_preview()
                self.assertFalse(locked["safe_to_cleanup"])
                (abliterated / "acejam_smoke_passed.json").write_text("{}", encoding="utf-8")
                unlocked = acejam_app._ace_lm_cleanup_preview()

            self.assertTrue(unlocked["safe_to_cleanup"])
            self.assertEqual(unlocked["delete_candidates"][0]["model"], "acestep-5Hz-lm-4B")

    def test_private_upload_is_gated_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "acestep-5Hz-lm-4B-abliterated"
            model.mkdir()
            (model / "acejam_smoke_passed.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(Exception):
                acejam_app._ace_lm_private_upload({"repo_id": "user/model", "model_path": str(model), "confirm": "NOPE"})

    def test_vocal_transcript_score_rejects_filler_repetition(self):
        score = acejam_app._score_vocal_transcript(
            "Albus yeah yeah yeah yeah yeah ah oh oh",
            ["albus"],
        )

        self.assertFalse(score["passed"])
        self.assertIn("asr_filler_ratio", score["issue"])
        self.assertIn("asr_repeat_yeah", score["issue"])

    def test_vocal_transcript_score_rejects_phrase_loops(self):
        score = acejam_app._score_vocal_transcript(
            "I love you I love you I love you I love you I love you I love you concrete remembers everything lost",
            ["concrete", "remembers", "everything", "lost"],
        )

        self.assertFalse(score["passed"])
        self.assertIn("asr_phrase_loop", score["issue"])

    def test_vocal_transcript_score_accepts_clear_words(self):
        score = acejam_app._score_vocal_transcript(
            "Albus in de stad lichten schijnen helder elke regel klinkt nu klaar",
            ["albus", "stad", "helder", "regel", "klaar"],
        )

        self.assertTrue(score["passed"])
        self.assertGreaterEqual(score["word_count"], acejam_app.ACEJAM_VOCAL_INTELLIGIBILITY_MIN_WORDS)
        self.assertGreaterEqual(len(score["keyword_hits"]), acejam_app.ACEJAM_VOCAL_INTELLIGIBILITY_MIN_KEYWORDS)

    def test_vocal_intelligibility_gate_updates_recommended_take(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            result_dir = results / "gatepass"
            result_dir.mkdir()
            for filename in ["take1.wav", "take2.wav"]:
                (result_dir / filename).write_text("audio", encoding="utf-8")
            result = {
                "success": True,
                "result_id": "gatepass",
                "audios": [
                    {"id": "take-1", "filename": "take1.wav", "pro_quality_score": 95},
                    {"id": "take-2", "filename": "take2.wav", "pro_quality_score": 80},
                ],
                "recommended_take": {"audio_id": "take-1"},
            }
            (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "vocal_intelligibility_gate": True,
            }
            transcripts = [
                {"path": str(result_dir / "take1.wav"), "status": "fail", "passed": False, "blocking": True, "text": ",!", "word_count": 0, "keyword_hits": [], "missing_keywords": ["albus"], "issue": "asr_words_0_keywords_0"},
                {"path": str(result_dir / "take2.wav"), "status": "pass", "passed": True, "blocking": False, "text": "Albus rises bright with every word clear tonight", "word_count": 8, "keyword_hits": ["albus", "rises", "bright"], "missing_keywords": [], "issue": ""},
            ]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_transcribe_audio_paths", return_value=transcripts):
                gate = acejam_app._apply_vocal_intelligibility_gate_to_result(result, params, attempt=1, max_attempts=3)

            self.assertTrue(gate["passed"])
            self.assertEqual(result["recommended_take"]["audio_id"], "take-2")
            saved = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["vocal_intelligibility_gate"]["status"], "pass")

    def test_vocal_intelligibility_gate_failure_clears_recommended_take(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            result_dir = results / "gatefail"
            result_dir.mkdir()
            (result_dir / "take.wav").write_text("audio", encoding="utf-8")
            result = {
                "success": True,
                "result_id": "gatefail",
                "audios": [{"id": "take-1", "filename": "take.wav", "is_recommended_take": True}],
                "recommended_take": {"audio_id": "take-1"},
            }
            (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "vocal_intelligibility_gate": True,
            }
            transcripts = [{
                "path": str(result_dir / "take.wav"),
                "status": "fail",
                "passed": False,
                "blocking": True,
                "text": "so oh",
                "word_count": 2,
                "keyword_hits": [],
                "missing_keywords": ["albus"],
                "issue": "asr_words_2_min_8",
            }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_transcribe_audio_paths", return_value=transcripts):
                gate = acejam_app._apply_vocal_intelligibility_gate_to_result(result, params, attempt=1, max_attempts=1)

            self.assertEqual(gate["status"], "fail")
            self.assertFalse(result["success"])
            self.assertNotIn("recommended_take", result)
            self.assertFalse(result["audios"][0]["is_recommended_take"])
            saved = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("recommended_take", saved)
            self.assertFalse(saved["audios"][0]["is_recommended_take"])

    def test_vocal_intelligibility_gate_treats_unavailable_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            result_dir = results / "asrunavailable"
            result_dir.mkdir()
            (result_dir / "take.wav").write_text("audio", encoding="utf-8")
            result = {
                "success": True,
                "result_id": "asrunavailable",
                "audios": [{"id": "take-1", "filename": "take.wav"}],
            }
            (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "vocal_intelligibility_gate": True,
            }
            transcripts = [{
                "path": str(result_dir / "take.wav"),
                "status": "unavailable",
                "passed": True,
                "blocking": False,
                "text": "",
                "word_count": 0,
                "keyword_hits": [],
                "missing_keywords": ["albus"],
                "issue": "asr_model_unavailable",
            }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_transcribe_audio_paths", return_value=transcripts):
                gate = acejam_app._apply_vocal_intelligibility_gate_to_result(result, params, attempt=1, max_attempts=3)

            self.assertEqual(gate["status"], "needs_review")
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["blocking"])
            self.assertFalse(result["success"])
            self.assertTrue(result["needs_review"])
            self.assertIn("vocal_intelligibility_verifier_error", result["payload_warnings"])

    def test_vocal_intelligibility_generation_retries_until_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 3,
            }
            calls = []

            def fake_once(attempt_params):
                result_id = f"retry{len(calls) + 1}"
                calls.append(attempt_params)
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {"success": True, "result_id": result_id, "audios": [{"id": "take-1", "filename": "take.wav"}]}
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                passed = len(calls) == 2
                return [{
                    "path": str(paths[0]),
                    "status": "pass" if passed else "fail",
                    "passed": passed,
                    "blocking": not passed,
                    "text": "Albus rises bright with every word clear tonight" if passed else ",!",
                    "word_count": 8 if passed else 0,
                    "keyword_hits": ["albus", "rises", "bright"] if passed else [],
                    "missing_keywords": [] if passed else ["albus"],
                    "issue": "" if passed else "asr_words_0_keywords_0",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual(result["result_id"], "retry2")
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["vocal_intelligibility_gate"]["status"], "pass")

    def test_vocal_intelligibility_records_no_lora_and_turbo_as_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "song_model": "acestep-v15-xl-sft",
                "quality_profile": "chart_master",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "shift": 1.0,
                "use_lora": True,
                "lora_adapter_path": "/tmp/adapter",
                "lora_adapter_name": "adapter",
                "lora_scale": 0.45,
                "adapter_model_variant": "xl_sft",
                "payload_warnings": [],
                "album_metadata": {},
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 3,
                "vocal_intelligibility_model_rescue": True,
                "vocal_intelligibility_model_rescue_after": 1,
            }
            calls = []

            def fake_once(attempt_params):
                calls.append(dict(attempt_params))
                result_id = f"loraretry{len(calls)}"
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {"success": True, "result_id": result_id, "audios": [{"id": "take-1", "filename": "take.wav"}]}
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                passed = calls[-1].get("use_lora") is False
                return [{
                    "path": str(paths[0]),
                    "status": "pass" if passed else "fail",
                    "passed": passed,
                    "blocking": not passed,
                    "text": "Albus rises bright with every word clear tonight" if passed else "so oh",
                    "word_count": 8 if passed else 2,
                    "keyword_hits": ["albus", "rises", "bright"] if passed else [],
                    "missing_keywords": [] if passed else ["albus"],
                    "issue": "" if passed else "asr_words_2_min_8",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_lora_quality_for_params", return_value={"quality_status": "verified", "quality_reasons": [], "audition_passed": True, "metadata": {}}), \
                patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft", "acestep-v15-turbo"}), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual(
                [call["song_model"] for call in calls],
                [
                    "acestep-v15-xl-sft",
                    "acestep-v15-xl-sft",
                    "acestep-v15-xl-sft",
                    "acestep-v15-xl-sft",
                    "acestep-v15-turbo",
                ],
            )
            self.assertTrue(all(call["use_lora"] for call in calls[:3]))
            self.assertFalse(calls[3]["use_lora"])
            self.assertFalse(calls[4]["use_lora"])
            self.assertFalse(result["success"])
            self.assertEqual(result["attempt_role"], "primary")
            self.assertEqual(result["vocal_intelligibility_gate"]["status"], "fail")
            self.assertTrue(result["diagnostic_attempts"][0]["passed"])
            self.assertEqual(result["diagnostic_attempts"][0]["actual_song_model"], "acestep-v15-xl-sft")

    def test_lora_preflight_quarantines_review_adapter_before_long_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "acejam_adapter.json").write_text(
                json.dumps(
                    {
                        "display_name": "review adapter",
                        "adapter_type": "lora",
                        "model_variant": "xl_sft",
                        "song_model": "acestep-v15-xl-sft",
                        "quality_status": "needs_review",
                    }
                ),
                encoding="utf-8",
            )
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "song_model": "acestep-v15-xl-sft",
                "quality_profile": "chart_master",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "shift": 1.0,
                "use_lora": True,
                "lora_adapter_path": str(adapter),
                "lora_adapter_name": "review adapter",
                "lora_scale": 0.45,
                "adapter_model_variant": "xl_sft",
                "adapter_metadata": {
                    "adapter_type": "lora",
                    "model_variant": "xl_sft",
                    "song_model": "acestep-v15-xl-sft",
                    "quality_status": "needs_review",
                },
                "lora_preflight_required": True,
                "payload_warnings": [],
                "album_metadata": {},
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 3,
            }
            calls = []

            def fake_once(attempt_params):
                calls.append(dict(attempt_params))
                result_id = f"preflight{len(calls)}"
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {"success": True, "result_id": result_id, "audios": [{"id": "take-1", "filename": "take.wav"}]}
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                passed = calls[-1].get("use_lora") is False
                return [{
                    "path": str(paths[0]),
                    "status": "pass" if passed else "fail",
                    "passed": passed,
                    "blocking": not passed,
                    "text": "Albus rises bright with every word clear tonight" if passed else "I don't know I don't know",
                    "word_count": 8 if passed else 5,
                    "keyword_hits": ["albus", "rises", "bright"] if passed else [],
                    "missing_keywords": [] if passed else ["albus"],
                    "issue": "" if passed else "asr_phrase_loop_i_don't_know",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual([call["use_lora"] for call in calls], [False, True])
            self.assertEqual(calls[1]["lora_scale"], 0.45)
            self.assertFalse(result["success"])
            self.assertEqual(result["lora_preflight"]["status"], "failed_audition")
            self.assertEqual(result["lora_preflight"]["requested_scale"], 0.45)
            metadata = json.loads((adapter / "acejam_adapter.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["quality_status"], "failed_audition")

    def test_vocal_intelligibility_generation_records_turbo_as_diagnostic_not_primary_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "song_model": "acestep-v15-xl-sft",
                "quality_profile": "chart_master",
                "inference_steps": 64,
                "guidance_scale": 8.0,
                "shift": 3.0,
                "infer_method": "ode",
                "sampler_mode": "heun",
                "use_adg": False,
                "audio_format": "wav32",
                "payload_warnings": [],
                "album_metadata": {},
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 3,
            }
            calls = []

            def fake_once(attempt_params):
                calls.append(dict(attempt_params))
                result_id = f"modelrescue{len(calls)}"
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {
                    "success": True,
                    "result_id": result_id,
                    "audios": [{"id": "take-1", "filename": "take.wav"}],
                    "active_song_model": attempt_params.get("song_model"),
                    "song_model": attempt_params.get("song_model"),
                    "payload_warnings": list(attempt_params.get("payload_warnings") or []),
                }
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                passed = calls[-1].get("song_model") == "acestep-v15-turbo"
                return [{
                    "path": str(paths[0]),
                    "status": "pass" if passed else "fail",
                    "passed": passed,
                    "blocking": not passed,
                    "text": "Albus rises bright with every word clear tonight" if passed else "I don't know I don't know",
                    "word_count": 8 if passed else 5,
                    "keyword_hits": ["albus", "rises", "bright"] if passed else [],
                    "missing_keywords": [] if passed else ["albus"],
                    "issue": "" if passed else "asr_keywords_0_min_2",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft", "acestep-v15-turbo"}), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual(
                [call["song_model"] for call in calls],
                ["acestep-v15-xl-sft", "acestep-v15-xl-sft", "acestep-v15-xl-sft", "acestep-v15-turbo"],
            )
            self.assertTrue(all(call["inference_steps"] == 50 for call in calls[:3]))
            self.assertTrue(all(call["shift"] == 1.0 for call in calls[:3]))
            self.assertEqual(calls[3]["inference_steps"], 8)
            self.assertEqual(calls[3]["shift"], 3.0)
            self.assertEqual(result["result_id"], "modelrescue3")
            self.assertFalse(result["success"])
            self.assertEqual(result["attempt_role"], "primary")
            self.assertEqual(result["vocal_intelligibility_gate"]["status"], "fail")
            self.assertEqual(result["diagnostic_attempts"][0]["actual_song_model"], "acestep-v15-turbo")
            self.assertTrue(result["diagnostic_attempts"][0]["passed"])
            self.assertIn("vocal_diagnostic_passed_primary_failed", result["payload_warnings"])

    def test_long_xl_sft_render_blocks_when_vocal_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "song_model": "acestep-v15-xl-sft",
                "quality_profile": "chart_master",
                "inference_steps": 8,
                "guidance_scale": 8.0,
                "shift": 3.0,
                "duration": 180,
                "vocal_preflight_required": True,
                "payload_warnings": [],
                "album_metadata": {},
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 3,
            }
            calls = []

            def fake_once(attempt_params):
                calls.append(dict(attempt_params))
                result_id = f"preblock{len(calls)}"
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {
                    "success": True,
                    "result_id": result_id,
                    "active_song_model": attempt_params.get("song_model"),
                    "song_model": attempt_params.get("song_model"),
                    "audios": [{"id": "take-1", "filename": "take.wav"}],
                    "payload_warnings": list(attempt_params.get("payload_warnings") or []),
                }
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                passed = calls[-1].get("song_model") == "acestep-v15-turbo"
                return [{
                    "path": str(paths[0]),
                    "status": "pass" if passed else "fail",
                    "passed": passed,
                    "blocking": not passed,
                    "text": "Albus rises bright with every word clear tonight" if passed else "thank you thank you",
                    "word_count": 8 if passed else 4,
                    "keyword_hits": ["albus", "rises", "bright"] if passed else [],
                    "missing_keywords": [] if passed else ["albus"],
                    "issue": "" if passed else "asr_phrase_loop_thank_you",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft", "acestep-v15-turbo"}), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual([call["song_model"] for call in calls], ["acestep-v15-xl-sft", "acestep-v15-turbo"])
            self.assertEqual(calls[0]["duration"], acejam_app.ACEJAM_LORA_PREFLIGHT_DURATION_SECONDS)
            self.assertEqual(calls[0]["inference_steps"], 50)
            self.assertEqual(calls[0]["shift"], 1.0)
            self.assertEqual(calls[1]["inference_steps"], 8)
            self.assertEqual(calls[1]["shift"], 3.0)
            self.assertFalse(result["success"])
            self.assertEqual(result["vocal_preflight"]["status"], "failed")
            self.assertEqual(result["diagnostic_attempts"][0]["actual_song_model"], "acestep-v15-turbo")
            self.assertTrue(result["diagnostic_attempts"][0]["passed"])

    def test_long_xl_sft_render_does_not_create_30s_preflight_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "song_model": "acestep-v15-xl-sft",
                "quality_profile": "chart_master",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "shift": 1.0,
                "duration": 180,
                "payload_warnings": [],
                "album_metadata": {},
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 1,
            }
            calls = []

            def fake_once(attempt_params):
                calls.append(dict(attempt_params))
                result_id = f"fullsong{len(calls)}"
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {
                    "success": True,
                    "result_id": result_id,
                    "active_song_model": attempt_params.get("song_model"),
                    "song_model": attempt_params.get("song_model"),
                    "audios": [{"id": "take-1", "filename": "take.wav"}],
                    "payload_warnings": list(attempt_params.get("payload_warnings") or []),
                }
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                return [{
                    "path": str(paths[0]),
                    "status": "pass",
                    "passed": True,
                    "blocking": False,
                    "text": "Albus rises bright with every word clear tonight",
                    "word_count": 8,
                    "keyword_hits": ["albus", "rises", "bright"],
                    "missing_keywords": [],
                    "issue": "",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["duration"], 180)
            self.assertNotIn("vocal_preflight", result)
            self.assertTrue(result["success"])

    def test_vocal_intelligibility_generation_fails_loudly_after_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 2,
            }
            calls = []

            def fake_once(_attempt_params):
                result_id = f"fail0{len(calls) + 1}"
                calls.append(result_id)
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {"success": True, "result_id": result_id, "audios": [{"id": "take-1", "filename": "take.wav"}]}
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                return [{
                    "path": str(paths[0]),
                    "status": "fail",
                    "passed": False,
                    "blocking": True,
                    "text": ",!",
                    "word_count": 0,
                    "keyword_hits": [],
                    "missing_keywords": ["albus"],
                    "issue": "asr_words_0_keywords_0",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual(calls, ["fail01", "fail02"])
            self.assertFalse(result["success"])
            self.assertEqual(result["attempt_role"], "primary")
            self.assertIn("Vocal intelligibility gate failed after 2 primary attempt", result["error"])
            for result_id in calls:
                saved = json.loads((results / result_id / "result.json").read_text(encoding="utf-8"))
                self.assertFalse(saved["success"])
                self.assertIn("Vocal intelligibility", saved["error"])

    def test_vocal_intelligibility_verifier_error_does_not_retry_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            params = {
                "task_type": "text2music",
                "instrumental": False,
                "lyrics": "[Chorus]\nAlbus rises bright",
                "title": "Albus Test",
                "artist_name": "Acejam",
                "vocal_language": "en",
                "vocal_intelligibility_gate": True,
                "vocal_intelligibility_attempts": 3,
            }
            calls = []

            def fake_once(_attempt_params):
                result_id = "asrerr"
                calls.append(result_id)
                result_dir = results / result_id
                result_dir.mkdir()
                (result_dir / "take.wav").write_text("audio", encoding="utf-8")
                result = {"success": True, "result_id": result_id, "audios": [{"id": "take-1", "filename": "take.wav"}]}
                (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return result

            def fake_asr(paths, **_kwargs):
                return [{
                    "path": str(paths[0]),
                    "status": "error",
                    "passed": False,
                    "blocking": True,
                    "text": "",
                    "word_count": 0,
                    "keyword_hits": [],
                    "missing_keywords": ["albus"],
                    "issue": "asr verifier exploded",
                }]

            with patch.object(acejam_app, "RESULTS_DIR", results), \
                patch.object(acejam_app, "_parse_generation_payload", return_value=params), \
                patch.object(acejam_app, "_run_advanced_generation_once", side_effect=fake_once), \
                patch.object(acejam_app, "_transcribe_audio_paths", side_effect=fake_asr):
                result = acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual(calls, ["asrerr"])
            self.assertFalse(result["success"])
            self.assertEqual(result["vocal_intelligibility_gate"]["status"], "needs_review")
            self.assertFalse(result["vocal_intelligibility_gate"]["blocking"])
            self.assertNotIn("recommended_take", result)
            self.assertIn("vocal_intelligibility_verifier_error", result["payload_warnings"])
            saved = json.loads((results / "asrerr" / "result.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["success"])
            self.assertEqual(saved["vocal_intelligibility_gate"]["status"], "needs_review")

    def test_vocal_intelligibility_asr_timeout_returns_advisory_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "take.wav"
            audio_path.write_text("audio", encoding="utf-8")
            with patch.object(acejam_app, "ACEJAM_VOCAL_ASR_MODEL", "local-whisper"), \
                patch.object(acejam_app, "ACEJAM_VOCAL_ASR_TIMEOUT", 5), \
                patch.object(acejam_app.subprocess, "run", side_effect=acejam_app.subprocess.TimeoutExpired(["asr"], 5)):
                transcripts = acejam_app._transcribe_audio_paths([audio_path], language="en", expected_keywords=["albus"])

            self.assertEqual(transcripts[0]["status"], "error")
            self.assertFalse(transcripts[0]["blocking"])
            self.assertIn("timed out after 5s", transcripts[0]["issue"])

    def test_vocal_asr_finds_pinokio_ffmpeg_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pinokio"
            app_dir = root / "api" / "MLX-Media.git" / "app"
            ffmpeg_dir = root / "bin" / "ffmpeg-env" / "bin"
            ffmpeg_dir.mkdir(parents=True)
            (ffmpeg_dir / "ffmpeg").write_text("#!/bin/sh\n", encoding="utf-8")

            with patch.object(acejam_app, "BASE_DIR", app_dir), \
                patch.object(acejam_app.shutil, "which", return_value=None):
                self.assertIn(str(ffmpeg_dir.resolve()), acejam_app._ffmpeg_subprocess_bin_dirs())

    def test_react_ai_fill_persists_wizard_drafts(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        store = (web_src / "store" / "wizard.ts").read_text(encoding="utf-8")
        draft_hook = (web_src / "hooks" / "useWizardDraft.ts").read_text(encoding="utf-8")
        custom = (web_src / "wizards" / "CustomWizard.tsx").read_text(encoding="utf-8")
        wizard_paths = [
            web_src / "wizards" / "SimpleWizard.tsx",
            web_src / "wizards" / "CustomWizard.tsx",
            web_src / "wizards" / "SourceAudioWizard.tsx",
            web_src / "wizards" / "AlbumWizard.tsx",
            web_src / "wizards" / "NewsWizard.tsx",
        ]

        self.assertIn("drafts: Record<string, Record<string, unknown> | undefined>", store)
        self.assertIn("setDraft", store)
        self.assertIn("clearDraft", store)
        self.assertIn("drafts: state.drafts", store)
        self.assertIn("form.watch", draft_hook)
        self.assertIn("mergeWizardDraft", draft_hook)
        self.assertIn("usePromptMirror", draft_hook)
        self.assertIn("normalizeAceStepRenderDraft", draft_hook)
        self.assertIn("Number(normalized.inference_steps) < 50", draft_hook)
        self.assertIn("normalized.shift = 1", draft_hook)
        self.assertIn("docsCorrectRenderDefaults(nextModel)", custom)
        self.assertNotIn("docsCorrectRenderDefaults(nextProfile)", custom)

        for path in wizard_paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("useWizardDraft", text, path.name)
            self.assertIn("mergeWizardDraft", text, path.name)
            self.assertIn("draftState.saveNow", text, path.name)
            self.assertIn("draftState.clear", text, path.name)

    def test_react_ollama_image_generation_ui_is_removed(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        api_ts = (web_src / "lib" / "api.ts").read_text(encoding="utf-8")
        settings = (web_src / "pages" / "Settings.tsx").read_text(encoding="utf-8")
        wizard_text = "\n".join(
            (web_src / "wizards" / name).read_text(encoding="utf-8")
            for name in [
                "SimpleWizard.tsx",
                "CustomWizard.tsx",
                "SourceAudioWizard.tsx",
                "AlbumWizard.tsx",
                "NewsWizard.tsx",
            ]
        )

        self.assertNotIn("DEFAULT_OLLAMA_IMAGE_MODEL", api_ts)
        self.assertNotIn("generateArt", api_ts)
        self.assertNotIn("Album- &amp; track-art", settings)
        self.assertNotIn("ArtGenerator", wizard_text)
        self.assertFalse((web_src / "components" / "art" / "ArtGenerator.tsx").exists())
        self.assertNotIn("aravhawk/flux:11.9bf16", settings)
        self.assertNotIn("aravhawk/flux:11.9bf16", wizard_text)

    def test_react_mflux_image_studio_routes_and_payloads_exist(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        app_tsx = (web_src / "App.tsx").read_text(encoding="utf-8")
        api_ts = (web_src / "lib" / "api.ts").read_text(encoding="utf-8")
        settings = (web_src / "pages" / "Settings.tsx").read_text(encoding="utf-8")
        image_wizard = (web_src / "wizards" / "ImageWizard.tsx").read_text(encoding="utf-8")
        image_trainer = (web_src / "wizards" / "ImageTrainerWizard.tsx").read_text(encoding="utf-8")
        art_maker = (web_src / "components" / "mflux" / "MfluxArtMaker.tsx").read_text(encoding="utf-8")
        home = (web_src / "pages" / "Home.tsx").read_text(encoding="utf-8")

        self.assertIn('path="/wizard/image"', app_tsx)
        self.assertIn('path="/wizard/image-trainer"', app_tsx)
        self.assertIn("MLX Media", app_tsx)
        self.assertIn("getMfluxStatus", api_ts)
        self.assertIn("startMfluxJob", api_ts)
        self.assertIn("uploadMfluxImage", api_ts)
        self.assertIn("startMfluxLoraTraining", api_ts)
        self.assertIn("attachMfluxArt", api_ts)
        self.assertIn('TabsTrigger value="mflux"', settings)
        self.assertIn("MFLUX runtime", settings)
        self.assertIn("MFLUX commands", settings)
        self.assertIn("lora_adapters", image_wizard)
        self.assertIn('mode="image"', image_wizard)
        self.assertIn("useWizardStore", image_wizard)
        self.assertIn("ImageUploadBox", image_wizard)
        self.assertIn("mask_path", image_wizard)
        self.assertIn("upscale_factor", image_wizard)
        self.assertIn("model_id", image_wizard)
        self.assertIn("dataset_path", image_trainer)
        self.assertIn("dataset_type", image_trainer)
        self.assertIn("preview_prompt", image_trainer)
        self.assertIn("startMfluxLoraTraining", image_trainer)
        self.assertIn("target_type", art_maker)
        self.assertIn('action: "upscale"', art_maker)
        self.assertIn("MFLUX Art Maker", art_maker)
        self.assertIn("Image Studio", home)

    def test_mflux_art_attach_endpoint_returns_metadata(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "mock-image"
            result_dir.mkdir(parents=True)
            (result_dir / "cover.png").write_bytes(b"png")
            (result_dir / "mflux_result.json").write_text(
                json.dumps(
                    {
                        "result_id": "mock-image",
                        "filename": "cover.png",
                        "image_url": "/media/mflux/mock-image/cover.png",
                        "prompt": "cover art",
                        "model_label": "Qwen Image",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(acejam_app, "MFLUX_RESULTS_DIR", root):
                response = client.post(
                    "/api/mflux/art/attach",
                    json={"source_result_id": "mock-image", "target_type": "mflux"},
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["art"]["url"], "/media/mflux/mock-image/cover.png")
        self.assertEqual(data["art"]["source"], "mflux")
        self.assertIn("path", data["art"])

    def test_mflux_upload_endpoint_accepts_images_and_rejects_audio(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(acejam_app, "MFLUX_UPLOADS_DIR", root):
                ok = client.post("/api/mflux/uploads", files={"file": ("cover.png", b"png", "image/png")})
                bad = client.post("/api/mflux/uploads", files={"file": ("voice.wav", b"wav", "audio/wav")})

        self.assertEqual(ok.status_code, 200)
        self.assertTrue(ok.json()["success"])
        self.assertIn("/media/mflux/uploads/", ok.json()["url"])
        self.assertEqual(bad.status_code, 400)

    def test_react_mlx_video_studio_routes_and_payloads_exist(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        app_tsx = (web_src / "App.tsx").read_text(encoding="utf-8")
        api_ts = (web_src / "lib" / "api.ts").read_text(encoding="utf-8")
        settings = (web_src / "pages" / "Settings.tsx").read_text(encoding="utf-8")
        video_wizard = (web_src / "wizards" / "VideoWizard.tsx").read_text(encoding="utf-8")
        library = (web_src / "pages" / "Library.tsx").read_text(encoding="utf-8")
        source_audio = (web_src / "wizards" / "SourceAudioWizard.tsx").read_text(encoding="utf-8")
        news = (web_src / "wizards" / "NewsWizard.tsx").read_text(encoding="utf-8")
        image_wizard = (web_src / "wizards" / "ImageWizard.tsx").read_text(encoding="utf-8")
        tracker = (web_src / "components" / "JobTracker.tsx").read_text(encoding="utf-8")
        home = (web_src / "pages" / "Home.tsx").read_text(encoding="utf-8")

        self.assertIn('path="/wizard/video"', app_tsx)
        self.assertIn("getMlxVideoStatus", api_ts)
        self.assertIn("startMlxVideoJob", api_ts)
        self.assertIn("uploadMlxVideoMedia", api_ts)
        self.assertIn("registerMlxVideoModelDir", api_ts)
        self.assertIn("getMlxVideoAttachments", api_ts)
        self.assertIn("startMlxVideoLoraTraining", api_ts)
        self.assertIn("listLibrary", api_ts)
        self.assertIn("deleteLibraryItem", api_ts)
        self.assertIn("deleteLibraryItems", api_ts)
        self.assertIn('TabsTrigger value="video"', settings)
        self.assertIn("MLX video runtime", settings)
        self.assertIn("Wan model directories", settings)
        self.assertIn("Video-LoRA training", settings)
        self.assertIn("PR #23 end frame", settings)
        self.assertIn("capabilities", settings)
        self.assertIn("Video Studio", home)
        self.assertIn("source_job_id", video_wizard)
        self.assertIn('mode="video"', video_wizard)
        self.assertIn("audio_policy", video_wizard)
        self.assertIn("replace_with_source", video_wizard)
        self.assertIn("primary_video_url", video_wizard)
        self.assertIn("end_image_path", video_wizard)
        self.assertIn("enhance_prompt", video_wizard)
        self.assertIn("spatial_upscaler", video_wizard)
        self.assertIn("tiling", video_wizard)
        self.assertIn("attachMlxVideo", video_wizard)
        self.assertIn("lora_adapters", video_wizard)
        self.assertIn("Make Final", video_wizard)
        self.assertIn("listLibrary", library)
        self.assertIn("deleteLibraryItem", library)
        self.assertIn("deleteLibraryItems", library)
        self.assertIn("Delete from disk", library)
        self.assertIn("Selecteer zichtbaar", library)
        self.assertIn("Selecteer alles", library)
        self.assertIn("Delete selected", library)
        self.assertIn('TabsTrigger value="results"', library)
        self.assertIn('TabsTrigger value="images"', library)
        self.assertIn('TabsTrigger value="videos"', library)
        self.assertIn("<WaveformPlayer", library)
        self.assertIn("<img", library)
        self.assertIn("<video", library)
        self.assertIn("getMlxVideoAttachments", library)
        self.assertIn('to="/wizard/video"', library)
        self.assertIn('navigate("/wizard/video"', source_audio)
        self.assertIn('navigate("/wizard/video"', news)
        self.assertIn('navigate("/wizard/video"', image_wizard)
        self.assertIn("mlx-video", tracker)

    def test_react_music_wizards_send_after_render_automation_fields(self):
        web_src = Path(__file__).resolve().parents[1] / "web" / "src"
        schemas = (web_src / "lib" / "schemas.ts").read_text(encoding="utf-8")
        automation = (web_src / "components" / "wizard" / "AutomationFields.tsx").read_text(encoding="utf-8")
        wizard_text = "\n".join(
            (web_src / "wizards" / name).read_text(encoding="utf-8")
            for name in ["SimpleWizard.tsx", "CustomWizard.tsx", "AlbumWizard.tsx", "NewsWizard.tsx"]
        )

        for field in ["auto_song_art", "auto_album_art", "auto_video_clip", "art_prompt", "video_prompt"]:
            self.assertIn(field, schemas)
            self.assertIn(field, wizard_text)
        self.assertIn("After audio render", automation)
        self.assertIn("AutomationFields", wizard_text)
        self.assertIn("auto_video_clip: v.auto_video_clip", wizard_text)

    def test_mlx_video_upload_endpoint_accepts_image_audio_and_rejects_text(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(acejam_app, "MLX_VIDEO_UPLOADS_DIR", root):
                image = client.post("/api/mlx-video/uploads", files={"file": ("frame.png", b"png", "image/png")})
                audio = client.post("/api/mlx-video/uploads", files={"file": ("song.wav", b"wav", "audio/wav")})
                bad = client.post("/api/mlx-video/uploads", files={"file": ("notes.txt", b"txt", "text/plain")})

        self.assertEqual(image.status_code, 200)
        self.assertEqual(audio.status_code, 200)
        self.assertIn("/media/mlx-video/uploads/", image.json()["url"])
        self.assertEqual(audio.json()["media_kind"], "audio")
        self.assertEqual(bad.status_code, 400)

    def test_mlx_video_attachments_endpoint_filters_targets(self):
        client = TestClient(acejam_app.app)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attachments.json"
            path.write_text(
                json.dumps(
                    [
                        {"target_type": "song", "target_id": "track-a", "result_id": "one"},
                        {"target_type": "album", "target_id": "album-b", "result_id": "two"},
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(acejam_app, "mlx_video_list_attachments", lambda target_type=None, target_id=None: [
                     item for item in json.loads(path.read_text(encoding="utf-8"))
                     if (not target_type or item["target_type"] == target_type) and (not target_id or item["target_id"] == target_id)
                 ]):
                response = client.get("/api/mlx-video/attachments?target_type=song&target_id=track-a")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["attachments"][0]["result_id"], "one")

    def test_mlx_video_lora_train_endpoint_reports_upstream_unavailable(self):
        client = TestClient(acejam_app.app)
        response = client.post("/api/mlx-video/lora/train", json={"dataset_path": "/tmp/nope"})

        self.assertEqual(response.status_code, 501)
        self.assertFalse(response.json()["success"])
        self.assertFalse(response.json()["available"])
        self.assertIn("upstream mlx-video", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
