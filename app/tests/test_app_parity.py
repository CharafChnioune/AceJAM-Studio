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

    def test_unreleased_model_is_not_downloadable(self):
        self.assertNotIn("acestep-v15-turbo-rl", acejam_app._downloadable_model_names())

        status = acejam_app._model_runtime_status("acestep-v15-turbo-rl")

        self.assertEqual(status["status"], "unreleased")
        self.assertFalse(status["downloadable"])
        self.assertIn("unreleased", status["error"])

    def test_official_parity_payload_exposes_wrappers_and_runtime(self):
        payload = acejam_app._official_parity_payload()

        self.assertTrue(payload["success"])
        self.assertIn("/v1/init", payload["manifest"]["api_endpoints"])
        self.assertIn("/v1/training/start_lokr", payload["manifest"]["api_endpoints"])
        self.assertIn("backend_hash", payload["manifest"]["runtime"])
        self.assertIn("api_key_enabled", payload["manifest"]["runtime"])
        self.assertIn("quality_policy", payload["manifest"])
        self.assertIn("schema_parity", payload["manifest"])
        self.assertEqual(payload["manifest"]["quality_policy"]["sft_base_models"]["inference_steps"], 64)
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
        self.assertFalse(normalized["use_cot_lyrics"])
        self.assertEqual(normalized["inference_steps"], 64)
        self.assertEqual(normalized["guidance_scale"], 8.0)
        self.assertEqual(normalized["shift"], 1.0)
        self.assertEqual(normalized["sampler_mode"], "heun")
        self.assertEqual(normalized["audio_format"], "wav")
        self.assertEqual(normalized["runner_plan"], "fast")

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

        self.assertEqual(params["lm_temperature"], 1.0)
        self.assertEqual(params["lm_top_k"], 40)
        self.assertEqual(params["lm_top_p"], 1.0)

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
            "song_model_strategy": "all_models_album",
            "ace_lm_model": "none",
            "track_variants": 1,
            "save_to_library": False,
            "tracks": [
                {
                    "track_number": 1,
                    "artist_name": "Unit Signal",
                    "title": "Seven Model Test",
                    "tags": "pop, radio ready, crisp modern mix",
                    "lyrics": "[Verse]\nWe test the route\n\n[Chorus]\nEvery model plays it loud",
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
        self.assertEqual(len(data["model_albums"]), len(acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS))
        self.assertIn("album_family_id", data)
        self.assertEqual(data["tracks"][0]["model_results"][0]["album_model"], acejam_app.ALBUM_MODEL_PORTFOLIO_MODELS[0])

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
        self.assertTrue(job["memory_enabled"])
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
            "planning_engine": "crewai",
            "crewai_used": True,
            "toolbelt_fallback": False,
            "crewai_output_log_file": "/tmp/album_plan_planjob123.json",
        }

        with patch.object(acejam_app, "_run_album_plan_from_payload", return_value=fake_result):
            acejam_app._album_plan_job_worker(job_id, payload)

        job = acejam_app._album_job_snapshot(job_id)
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(job["job_type"], "album_plan")
        self.assertEqual(job["planned_count"], 2)
        self.assertEqual(job["result"]["tracks"][0]["title"], "One")
        self.assertEqual(job["planning_engine"], "crewai")
        self.assertTrue(job["crewai_used"])
        self.assertFalse(job["toolbelt_fallback"])
        self.assertEqual(job["crewai_output_log_file"], "/tmp/album_plan_planjob123.json")

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
        self.assertTrue(all(not item["use_cot_lyrics"] for item in calls))
        self.assertEqual(calls[0]["inference_steps"], 8)
        self.assertEqual(calls[2]["inference_steps"], 64)
        self.assertEqual(calls[0]["guidance_scale"], 7.0)
        self.assertEqual(calls[2]["guidance_scale"], 8.0)
        self.assertEqual(calls[0]["shift"], 3.0)
        self.assertEqual(calls[2]["shift"], 1.0)
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
