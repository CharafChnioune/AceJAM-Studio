import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import album_crew as album_crew_module
from fastapi.testclient import TestClient


os.environ.setdefault("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "1")
acejam_app = importlib.import_module("app")


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
        self.assertIn("acestep-captioner", payload["manifest"]["boot_quality_models"])
        self.assertIn("acestep-v15-xl-sft", payload["manifest"]["boot_quality_models"])
        self.assertEqual(payload["manifest"]["model_registry"]["acestep-v15-turbo-rl"]["status"], "unreleased")
        self.assertIn("component_status", payload["manifest"]["core_bundle"])
        self.assertIn("boot_downloads", payload["manifest"]["runtime"])
        self.assertEqual(payload["manifest"]["settings_registry"]["version"], "ace-step-settings-parity-2026-04-26")
        self.assertEqual(payload["manifest"]["quality_policy"]["sft_base_models"]["inference_steps"], 64)
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

    def test_boot_download_bundle_queues_best_quality_and_helper_models(self):
        names = acejam_app._boot_download_model_names()

        self.assertIn("main", names)
        self.assertIn("acestep-v15-xl-sft", names)
        self.assertIn("acestep-v15-xl-base", names)
        self.assertIn(acejam_app.ACE_LM_PREFERRED_MODEL, names)
        self.assertIn("acestep-captioner", names)
        self.assertIn("acestep-transcriber", names)

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
                    "lm_repetition_penalty": 1.15,
                    "song_intent": {"genre_family": "rap", "caption": "structured intent caption"},
                    "source_task_intent": "clean source vocal before morph",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))

        official_params = request["params"]
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
        self.assertEqual(request["official_api_fields"]["lm_repetition_penalty"], 1.15)
        self.assertEqual(request["lm_sampling"]["repetition_penalty"], 1.15)
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
            self.assertEqual(schema["counts"]["tag_taxonomy_terms"], 184)
            self.assertEqual(schema["counts"]["lyric_meta_tags"], 36)
            self.assertEqual(schema["counts"]["valid_languages"], 51)
            self.assertEqual(schema["counts"]["track_stems"], 12)
            self.assertIn("cover-nofsq", schema["capabilities"]["all_task_modes"])
            self.assertIn("acestep-v15-xl-base", schema["capabilities"]["model_support"]["complete"])

    def test_ai_fill_hydrates_song_intent_builder_chip_groups(self):
        html = (acejam_app.BASE_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn("resolveIntentKnownOptionValue", html)
        self.assertIn('setIntentGroupValues("genre_style", [intent.subgenre, intent.style_tags, payload.tags]);', html)
        self.assertIn('setIntentGroupValues("speed_rhythm", [intent.energy, intent.rhythm_tags]);', html)
        self.assertIn('setIntentGroupValues("instruments", [intent.instrument_tags]);', html)
        self.assertIn('setIntentGroupValues("timbre_texture", [intent.texture_space, intent.production_tags]);', html)
        self.assertIn('setIntentGroupValues("negative_control", [intent.negative_tags, payload.negative_tags]);', html)

    def test_official_runner_timeout_expands_for_long_mlx_generations(self):
        request_payload = {
            "lm_backend": "mlx",
            "params": {"duration": 365, "inference_steps": 64},
            "config": {"batch_size": 3},
        }

        timeout = acejam_app._official_runner_timeout_seconds(request_payload, requested_timeout=3600)

        self.assertGreater(timeout, 7200)
        self.assertLessEqual(timeout, acejam_app.ACEJAM_OFFICIAL_RUNNER_MAX_TIMEOUT_SECONDS)
        self.assertGreaterEqual(acejam_app.ACEJAM_GENERATE_ADVANCED_TIME_LIMIT_SECONDS, acejam_app.ACEJAM_OFFICIAL_RUNNER_TIMEOUT_SECONDS)

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

    def test_results_show_saved_library_link(self):
        html = (acejam_app.BASE_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn("audio.library_url", html)
        self.assertIn(">Library</a>", html)
        self.assertIn("if (payload.save_to_library) await loadLibrary();", html)

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

    def test_studio_generation_uses_hybrid_ace_lm_only_when_requested(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-base",
            "caption": "cinematic pop, live drums, rich vocal chain",
            "lyrics": "",
            "ace_lm_model": "acestep-5Hz-lm-4B",
            "thinking": True,
            "use_format": True,
            "sample_query": "make this a polished full song",
            "planner_ollama_model": "llama3.1:8b",
        }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-base"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", "acestep-5Hz-lm-4B"}):
            normalized = acejam_app._parse_generation_payload(payload)

        self.assertEqual(normalized["ace_lm_model"], "acestep-5Hz-lm-4B")
        self.assertEqual(normalized["planner_lm_provider"], "ollama")
        self.assertEqual(normalized["planner_ollama_model"], "llama3.1:8b")
        self.assertTrue(normalized["thinking"])
        self.assertTrue(normalized["use_format"])
        self.assertTrue(normalized["sample_mode"] is False)
        self.assertEqual(normalized["inference_steps"], 64)
        self.assertEqual(normalized["quality_profile"], "chart_master")
        self.assertEqual(normalized["runner_plan"], "official")

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
        self.assertEqual(normalized["lm_temperature"], 0.55)
        self.assertEqual(normalized["lm_top_p"], 0.7)
        self.assertEqual(normalized["lm_top_k"], 12)

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
        self.assertEqual(normalized["inference_steps"], 64)
        self.assertEqual(normalized["guidance_scale"], 8.0)
        self.assertEqual(normalized["shift"], 3.0)
        self.assertTrue(normalized["use_adg"])
        self.assertEqual(normalized["batch_size"], 3)
        self.assertEqual(normalized["sampler_mode"], "heun")
        self.assertEqual(normalized["audio_format"], "wav")
        self.assertEqual(normalized["runner_plan"], "fast")

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

        self.assertEqual(params["bpm"], acejam_app.DEFAULT_BPM)
        self.assertEqual(params["key_scale"], acejam_app.DEFAULT_KEY_SCALE)
        self.assertEqual(params["time_signature"], "4")
        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))
        self.assertEqual(request["params"]["bpm"], 95)
        self.assertEqual(request["params"]["keyscale"], "A minor")
        self.assertEqual(request["params"]["timesignature"], "4")

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

    def test_xl_sft_defaults_to_chart_master_64_steps(self):
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

        self.assertEqual(normalized["inference_steps"], 64)
        self.assertEqual(normalized["shift"], 3.0)
        self.assertEqual(normalized["song_model"], "acestep-v15-xl-sft")
        self.assertEqual(normalized["duration"], 180)

    def test_simple_custom_helpers_default_to_local_writer_unless_ace_engine_selected(self):
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

        official_payload = {
            "success": True,
            "title": "Official Helper",
            "caption": "pop, polished vocal, premium mix",
            "lyrics": "[Verse]\nLine\n\n[Chorus]\nHook",
        }
        with patch.object(acejam_app, "_run_official_lm_aux", return_value=official_payload) as official:
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
        self.assertEqual(response.json()["title"], "Official Helper")
        action, body = official.call_args.args
        self.assertEqual(action, "create_sample")
        self.assertEqual(body["planner_lm_provider"], "ace_step_lm")
        self.assertEqual(body["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
        self.assertTrue(body["use_official_lm"])

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

        self.assertEqual(params["lm_repetition_penalty"], 1.4)
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

    def test_ui_uses_keyscale_selects_and_bpm_default(self):
        html = (Path(acejam_app.BASE_DIR) / "index.html").read_text(encoding="utf-8")

        self.assertIn('<input id="bpm" type="number" min="30" max="300" value="95"', html)
        self.assertIn('<select id="key-scale"', html)
        self.assertIn('data-field="key_scale" data-key-scale-select', html)
        self.assertIn('data-field="keyscale" data-key-scale-select', html)
        self.assertIn("Pro Quality", html)
        self.assertIn("lora-health-grid", html)
        self.assertNotIn('id="key-scale" type="text"', html)

    def test_ui_has_single_local_ai_writer_planner_selector(self):
        html = (Path(acejam_app.BASE_DIR) / "index.html").read_text(encoding="utf-8")

        self.assertIn("Local AI Writer/Planner", html)
        self.assertIn('id="settings-local-ai-slot"', html)
        self.assertIn('id="planner-provider"', html)
        self.assertIn('id="ollama-model"', html)
        self.assertIn('id="embedding-provider"', html)
        self.assertIn('id="embedding-model"', html)
        self.assertIn('id="art-model"', html)
        self.assertIn('id="ollama-test-art-btn"', html)
        self.assertIn("body:not(.mode-settings) #local-ai-panel", html)
        self.assertIn("ACE-Step Audio Model", html)
        self.assertIn('id="album-agent-engine"', html)
        self.assertIn("AceJAM Direct (recommended)", html)
        self.assertIn("CrewAI Micro Tasks (experimental)", html)
        self.assertIn("agent_engine:", html)
        self.assertIn("planning_engine", html)
        self.assertIn("local_ai_writer", html)
        self.assertIn("ace_step_audio", html)
        self.assertIn("ace_step_optional_lm", html)
        self.assertNotIn('id="ai-fill-provider"', html)
        self.assertNotIn('id="ai-fill-ollama"', html)
        self.assertNotIn('id="album-planner-provider"', html)
        self.assertNotIn('id="album-ollama"', html)
        self.assertNotIn('id="album-embed-provider"', html)
        self.assertNotIn('id="album-embed"', html)
        self.assertNotIn('id="local-llm-provider"', html)
        self.assertNotIn("Legacy LM", html)
        self.assertNotIn("Planner provider", html)
        self.assertNotIn("Album planner", html)

    def test_ui_global_model_advice_uses_state_mode(self):
        html = (Path(acejam_app.BASE_DIR) / "index.html").read_text(encoding="utf-8")

        self.assertIn("selectedActualSongModel(state.mode)", html)
        self.assertNotIn("selectedActualSongModel(mode) : model", html)

    def test_album_payload_options_normalize_planning_engine(self):
        direct = acejam_app._album_options_from_payload({"concept": "one song"}, song_model="auto")
        micro = acejam_app._album_options_from_payload({"concept": "one song", "agent_engine": "legacy_crewai"}, song_model="auto")

        self.assertEqual(direct["agent_engine"], "acejam_agents")
        self.assertEqual(micro["agent_engine"], "crewai_micro")

    def test_lora_one_click_ui_and_adapterbar_are_exposed(self):
        html = (Path(acejam_app.BASE_DIR) / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="dataset-folder" type="file" webkitdirectory directory multiple', html)
        self.assertIn('id="dataset-language"', html)
        self.assertIn('id="lora-one-click-train"', html)
        self.assertIn('id="generation-lora-select"', html)
        self.assertIn('id="generation-lora-use"', html)
        self.assertIn('id="generation-lora-scale"', html)
        self.assertIn('id="album-lora-select"', html)
        self.assertIn('id="album-lora-use"', html)
        self.assertIn('id="album-lora-scale"', html)
        self.assertIn("loadableGenerationAdapters", html)
        self.assertIn('renderAdapterSelect("album-lora-select"', html)
        self.assertIn('id="epoch-audition-enabled"', html)
        self.assertIn('id="epoch-audition-genre"', html)
        self.assertIn('id="epoch-audition-caption"', html)
        self.assertIn('id="epoch-audition-lyrics"', html)
        self.assertIn('id="epoch-audition-lyrics-preview"', html)
        self.assertIn('id="epoch-audition-seed"', html)
        self.assertIn('id="epoch-audition-scale"', html)
        self.assertIn('id="train-device"', html)
        self.assertIn('id="train-device-note"', html)
        self.assertIn("renderTrainerDevicePolicy", html)
        self.assertIn('device: $("train-device")?.value || "auto"', html)
        self.assertIn("function selectedTrainerSongModel()", html)
        self.assertIn('return value === "auto" ? "acestep-v15-xl-sft" : value;', html)
        self.assertIn("song_model: selectedTrainerSongModel()", html)
        self.assertIn("epoch_audition_duration: 20", html)
        self.assertIn("epoch_audition_genre: genre", html)
        self.assertIn("full-quality 64-step WAV", html)
        self.assertIn("Standard test lyrics", html)
        self.assertIn("Test genre", html)
        self.assertNotIn("Extra test tags", html)
        self.assertNotIn("Lyric notes", html)
        self.assertIn("...epochAuditionPayload()", html)
        self.assertIn("renderLoraAuditions(job)", html)

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
                    "lora_scale": 0.65,
                    "adapter_model_variant": "xl_sft",
                }
            )

        self.assertTrue(params["use_lora"])
        self.assertEqual(params["lora_adapter_path"], adapter_path)
        self.assertEqual(params["lora_scale"], 0.65)
        with tempfile.TemporaryDirectory() as tmp:
            request = acejam_app._official_request_payload(params, Path(tmp))
        self.assertTrue(request["use_lora"])
        self.assertEqual(request["lora_adapter_path"], adapter_path)
        self.assertEqual(request["lora_adapter_name"], "unit")
        self.assertEqual(request["lora_scale"], 0.65)
        self.assertEqual(request["params"]["seed"], 314)
        self.assertEqual(request["config"]["seeds"], "314")

    def test_lora_epoch_audition_uses_private_generation_without_library_save(self):
        captured = {}

        def fake_generation(params):
            captured.update(params)
            return {
                "success": True,
                "result_id": "audition-result",
                "audios": [{"result_id": "audition-result", "audio_url": "/media/results/audition-result/take-1.wav"}],
            }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}), \
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
        self.assertIn("[Verse - seconds", captured["lyrics"])
        self.assertIn("[Chorus - seconds", captured["lyrics"])
        self.assertIn("Line one", captured["lyrics"])
        self.assertIn("Hook line", captured["lyrics"])
        self.assertIn("clear intelligible vocal", captured["caption"])
        self.assertEqual(captured["vocal_language"], "en")
        self.assertEqual(captured["seed"], "123")
        self.assertEqual(acejam_app.EPOCH_AUDITION_INFERENCE_STEPS, 64)
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
        self.assertTrue(captured["use_lora"])
        self.assertEqual(captured["lora_adapter_path"], "/tmp/checkpoints/epoch_2_loss_0.1")
        self.assertEqual(captured["lora_scale"], 0.7)
        self.assertEqual(result["lyrics_fit"]["action"], "fit_for_20s")
        self.assertTrue(result["lyrics_fit"]["timed_structure"])

    def test_lora_epoch_audition_auto_model_matches_checkpoint_variant(self):
        captured = {}

        def fake_generation(params):
            captured.update(params)
            return {
                "success": True,
                "result_id": "audition-result",
                "audios": [{"result_id": "audition-result", "audio_url": "/media/results/audition-result/take-1.wav"}],
            }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-turbo", "acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}), \
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
                "audios": [{"result_id": "audition-result", "audio_url": "/media/results/audition-result/take-1.wav"}],
            }

        with patch.object(acejam_app, "_installed_acestep_models", return_value={"acestep-v15-xl-sft"}), \
            patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}), \
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
        self.assertEqual(captured["duration"], 20)
        self.assertEqual(captured["vocal_language"], "en")
        self.assertEqual(captured["seed"], "42")
        self.assertEqual(captured["inference_steps"], acejam_app.EPOCH_AUDITION_INFERENCE_STEPS)
        sung_lines = [line for line in captured["lyrics"].splitlines() if line.strip() and not line.startswith("[")]
        self.assertLessEqual(len(captured["lyrics"]), 360)
        self.assertLessEqual(len(sung_lines), 4)
        self.assertIn("[Chorus - seconds", captured["lyrics"])
        self.assertIn("[Verse - seconds", captured["lyrics"])
        self.assertIn("clear intelligible vocal", captured["caption"])
        self.assertNotIn("Final Chorus -", captured["lyrics"])
        self.assertNotIn("Verse 4 -", captured["lyrics"])
        self.assertNotIn("[drums return", captured["lyrics"])
        self.assertNotIn("Arrangement note", captured["lyrics"])
        self.assertEqual(result["lyrics_fit"]["action"], "fit_for_20s")

    def test_lora_upload_path_sanitizer_preserves_relative_folders(self):
        self.assertEqual(str(acejam_app._safe_lora_upload_relative_path("dataset/sub/song.wav")), "dataset/sub/song.wav")
        self.assertEqual(str(acejam_app._safe_lora_upload_relative_path("../evil.wav")), "evil.wav")

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
        self.assertFalse(normalized["requires_official_runner"])
        self.assertEqual(normalized["runner_plan"], "fast")

    def test_explicit_lm_controls_upgrade_none_to_4b_default(self):
        with patch.object(acejam_app, "_installed_lm_models", return_value={"auto", "none", acejam_app.ACE_LM_PREFERRED_MODEL}):
            validation = acejam_app._validate_generation_payload(
                {
                    "task_type": "text2music",
                    "song_model": "acestep-v15-turbo",
                    "caption": "rap, hard drums",
                    "lyrics": "",
                    "sample_query": "write a song",
                    "ace_lm_model": "none",
                }
            )

        self.assertNotIn("ace_lm_model", validation["field_errors"])
        self.assertEqual(validation["normalized_payload"]["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)

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
            "use_lora": True,
            "lora_adapter_path": "/tmp/album-charaf-hook",
            "lora_adapter_name": "charaf hook",
            "lora_scale": 0.7,
            "adapter_model_variant": "xl_sft",
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
                        "[Verse]\nWe test the bright route\nEvery model enters clearly\n"
                        "Clean chords carry the signal\nThe chorus waits for release\n\n"
                        "[Chorus]\nEvery model plays it loud\nEvery take keeps timing proud\n"
                        "Seven engines share the light\nUnit Signal rides tonight\n\n"
                        "[Outro]\nThe final note stays clean\nSeven paths land bright"
                    ),
                    "duration": 30,
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
        self.assertTrue(all(payload["use_lora"] for payload in calls))
        self.assertTrue(all(payload["lora_adapter_path"] == "/tmp/album-charaf-hook" for payload in calls))
        self.assertTrue(all(payload["lora_adapter_name"] == "charaf hook" for payload in calls))
        self.assertTrue(all(payload["lora_scale"] == 0.7 for payload in calls))
        self.assertTrue(all(payload["adapter_model_variant"] == "xl_sft" for payload in calls))
        self.assertEqual(len(data["model_albums"]), len(acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS))
        self.assertIn("album_family_id", data)
        self.assertEqual(data["tracks"][0]["model_results"][0]["album_model"], acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS[0])

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
                    "lyrics": (
                        "[Verse]\nWe test the bright route\nEvery model enters clearly\n"
                        "Clean chords carry the signal\nThe chorus waits for release\n\n"
                        "[Chorus]\nEvery model plays it loud\nEvery take keeps timing proud\n"
                        "Unit Signal rides tonight\nThe hook lands clean and bright\n\n"
                        "[Outro]\nThe final note stays clean\nSeven paths land bright"
                    ),
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
                    "lyrics": (
                        "[Verse]\nWe test the bright route\nEvery model enters clearly\n"
                        "Clean chords carry the signal\nThe chorus waits for release\n\n"
                        "[Chorus]\nEvery model plays it loud\nEvery take keeps timing proud\n"
                        "Unit Signal rides tonight\nThe hook lands clean and bright\n\n"
                        "[Outro]\nThe final note stays clean\nSeven paths land bright"
                    ),
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
                    "lyrics": (
                        "[Verse]\nWe test the bright route\nEvery model enters clearly\n"
                        "Clean chords carry the signal\nThe chorus waits for release\n"
                        "Kick drum locks the cadence tight\nPiano flickers under city light\n"
                        "Low end moves but the words stay clear\nSignal in front for every ear\n\n"
                        "[Chorus]\nEvery model plays it loud\nEvery take keeps timing proud\n"
                        "Unit Signal rides tonight\nThe hook lands clean and bright\n"
                        "Hands come up when the chorus hits\nClear lead cuts through the mix\n"
                        "Every line lands right on time\nNo blurred words inside the rhyme\n\n"
                        "[Outro]\nThe final note stays clean\nSeven paths land bright\n"
                        "Bass rolls out and the voice stays near\nOne last hook for the engineer\n"
                        "Timing stays locked as the room lets go\nBright route echoes when the credits roll\n"
                        "Clear words ring through the final snare\nUnit Signal leaves the hook in air"
                    ),
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
        self.assertEqual(fallback, "02-AceJAM--My-Big-Hook--xl-sft--v3.wav")

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
        self.assertEqual(by_model["acestep-v15-sft"]["inference_steps"], 64)
        self.assertEqual(by_model["acestep-v15-turbo"]["guidance_scale"], 7.0)
        self.assertEqual(by_model["acestep-v15-sft"]["guidance_scale"], 8.0)
        self.assertEqual(by_model["acestep-v15-turbo"]["shift"], 3.0)
        self.assertEqual(by_model["acestep-v15-sft"]["shift"], 3.0)
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

            self.assertEqual(gate["status"], "error")
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["blocking"])
            self.assertTrue(result["success"])
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

    def test_vocal_intelligibility_generation_rescues_to_turbo_model(self):
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

            self.assertEqual([call["song_model"] for call in calls], ["acestep-v15-xl-sft", "acestep-v15-turbo"])
            self.assertEqual(calls[1]["inference_steps"], 8)
            self.assertEqual(result["result_id"], "modelrescue2")
            self.assertEqual(result["vocal_intelligibility_gate"]["status"], "pass")
            self.assertIn(
                "vocal_intelligibility_model_rescue:acestep-v15-xl-sft->acestep-v15-turbo",
                result["payload_warnings"],
            )

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
                with self.assertRaisesRegex(RuntimeError, "Vocal intelligibility gate failed after 2 attempt"):
                    acejam_app._run_advanced_generation({"title": "ignored"})

            self.assertEqual(calls, ["fail01", "fail02"])
            for result_id in calls:
                saved = json.loads((results / result_id / "result.json").read_text(encoding="utf-8"))
                self.assertFalse(saved["success"])
                self.assertEqual(saved["error"], "Vocal intelligibility gate rejected every take.")

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
            self.assertTrue(result["success"])
            self.assertEqual(result["vocal_intelligibility_gate"]["status"], "error")
            self.assertFalse(result["vocal_intelligibility_gate"]["blocking"])
            self.assertIn("vocal_intelligibility_verifier_error", result["payload_warnings"])
            saved = json.loads((results / "asrerr" / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["success"])
            self.assertEqual(saved["vocal_intelligibility_gate"]["status"], "error")

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


if __name__ == "__main__":
    unittest.main()
