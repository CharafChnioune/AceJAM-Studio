import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


os.environ.setdefault("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "1")
acejam_app = importlib.import_module("app")


class AppParityTest(unittest.TestCase):
    def _write_ready_checkpoint(self, root: Path, name: str, weight_name: str = "model.safetensors") -> Path:
        path = root / "checkpoints" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / weight_name).write_bytes(b"weights")
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

    def test_official_parity_payload_exposes_wrappers_and_runtime(self):
        payload = acejam_app._official_parity_payload()

        self.assertTrue(payload["success"])
        self.assertIn("/v1/init", payload["manifest"]["api_endpoints"])
        self.assertIn("/v1/training/start_lokr", payload["manifest"]["api_endpoints"])
        self.assertIn("backend_hash", payload["manifest"]["runtime"])
        self.assertIn("api_key_enabled", payload["manifest"]["runtime"])
        self.assertIn("quality_policy", payload["manifest"])
        self.assertIn("schema_parity", payload["manifest"])
        self.assertIn("settings_registry", payload["manifest"])
        self.assertEqual(payload["manifest"]["settings_registry"]["version"], "ace-step-settings-parity-2026-04-26")
        self.assertEqual(payload["manifest"]["quality_policy"]["sft_base_models"]["inference_steps"], 64)
        self.assertEqual(payload["manifest"]["quality_policy"]["balanced_pro_models"]["inference_steps"], 50)
        self.assertEqual(payload["manifest"]["quality_policy"]["default_profile"], "chart_master")
        self.assertEqual(payload["manifest"]["quality_policy"]["turbo_models"]["inference_steps"], 8)

    def test_official_response_wrapper_shape(self):
        wrapped = acejam_app._official_api_response({"ok": True})

        self.assertEqual(wrapped["code"], 200)
        self.assertIsNone(wrapped["error"])
        self.assertEqual(wrapped["data"], {"ok": True})
        self.assertIn("timestamp", wrapped)

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

        self.assertEqual(normalized["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
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

    def test_simple_custom_helpers_default_to_official_4b_lm(self):
        client = TestClient(acejam_app.app)
        official_payload = {
            "success": True,
            "title": "Official Helper",
            "caption": "pop, polished vocal, premium mix",
            "lyrics": "[Verse]\nLine\n\n[Chorus]\nHook",
        }

        with patch.object(acejam_app, "_run_official_lm_aux", return_value=official_payload) as official:
            response = client.post("/api/create_sample", json={"description": "make a hit", "duration": 180})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Official Helper")
        action, body = official.call_args.args
        self.assertEqual(action, "create_sample")
        self.assertEqual(body["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
        self.assertTrue(body["use_official_lm"])

        with patch.object(acejam_app, "_run_official_lm_aux", return_value=official_payload) as official:
            response = client.post("/api/format_sample", json={"caption": "polish this", "duration": 180})

        self.assertEqual(response.status_code, 200)
        action, body = official.call_args.args
        self.assertEqual(action, "format_sample")
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

    def test_lora_status_and_adapters_expose_display_name_and_trigger(self):
        client = TestClient(acejam_app.app)

        class StubTrainingManager:
            def status(self):
                return {"ready": True}

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

    def test_lora_upload_path_sanitizer_preserves_relative_folders(self):
        self.assertEqual(str(acejam_app._safe_lora_upload_relative_path("dataset/sub/song.wav")), "dataset/sub/song.wav")
        self.assertEqual(str(acejam_app._safe_lora_upload_relative_path("../evil.wav")), "evil.wav")

    def test_official_runner_stream_redacts_conditioning_prompt_blocks(self):
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

        rendered = "".join(acejam_app._redact_official_runner_stream_line(line, state) for line in lines)

        self.assertIn("conditioning prompt", rendered)
        self.assertIn("conditioning lyrics", rendered)
        self.assertIn("DiT backend: MLX (native)", rendered)
        self.assertNotIn("SECRET CAPTION", rendered)
        self.assertNotIn("SECRET LYRIC", rendered)

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

    def test_album_supplied_vocal_lyrics_use_ace_lm_formatting_by_default(self):
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
            "song_model_strategy": "selected",
            "song_model": "acestep-v15-turbo",
            "ace_lm_model": "none",
            "thinking": False,
            "use_format": False,
            "use_cot_caption": False,
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
        self.assertEqual(payload["ace_lm_model"], acejam_app.ACE_LM_PREFERRED_MODEL)
        self.assertTrue(payload["allow_supplied_lyrics_lm"])
        self.assertTrue(payload["thinking"])
        self.assertTrue(payload["use_format"])
        self.assertTrue(payload["use_cot_caption"])
        self.assertFalse(payload["sample_mode"])
        self.assertFalse(payload["use_cot_metas"])
        self.assertFalse(payload["use_cot_lyrics"])
        self.assertFalse(payload["use_cot_language"])
        self.assertEqual(payload["lm_backend"], acejam_app._normalize_lm_backend(acejam_app.ACE_LM_BACKEND_DEFAULT))

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
        self.assertTrue(all(item["ace_lm_model"] == acejam_app.ACE_LM_PREFERRED_MODEL for item in calls))
        self.assertTrue(all(not item["thinking"] for item in calls))
        self.assertTrue(all(not item["sample_mode"] for item in calls))
        self.assertTrue(all(item["sample_query"] == "" for item in calls))
        self.assertTrue(all(not item["use_cot_lyrics"] for item in calls))
        self.assertEqual(calls[0]["inference_steps"], 8)
        self.assertEqual(calls[2]["inference_steps"], 64)
        self.assertEqual(calls[0]["guidance_scale"], 7.0)
        self.assertEqual(calls[2]["guidance_scale"], 8.0)
        self.assertEqual(calls[0]["shift"], 3.0)
        self.assertEqual(calls[2]["shift"], 3.0)
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


if __name__ == "__main__":
    unittest.main()
