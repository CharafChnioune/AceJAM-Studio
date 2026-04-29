import unittest

from studio_core import (
    ACE_STEP_CAPTION_CHAR_LIMIT,
    ACE_STEP_LYRICS_CHAR_LIMIT,
    ACE_STEP_LM_MODELS,
    DEFAULT_BPM,
    DEFAULT_KEY_SCALE,
    KNOWN_ACE_STEP_MODELS,
    LM_MODEL_PROFILES,
    MODEL_PROFILES,
    OFFICIAL_ACE_STEP_MANIFEST,
    OFFICIAL_UNRELEASED_MODELS,
    PRO_QUALITY_AUDIT_VERSION,
    ace_step_settings_compliance,
    ace_step_settings_registry,
    build_task_instruction,
    docs_best_model_settings,
    ensure_task_supported,
    apply_ace_step_text_budget,
    hit_readiness_report,
    lm_model_profile,
    lm_model_profiles_for_models,
    model_profile,
    model_profiles_for_models,
    needs_vocal_lyrics,
    normalize_generation_text_fields,
    normalize_key_scale,
    normalize_task_type,
    official_fields_used,
    official_manifest,
    normalize_track_names,
    parse_timesteps,
    recommended_lm_model,
    recommended_song_model,
    runtime_planner_report,
    safe_id,
    studio_ui_schema,
    supported_tasks_for_model,
    VALID_KEY_SCALES,
)


class StudioCoreTest(unittest.TestCase):
    def test_task_aliases(self):
        self.assertEqual(normalize_task_type("simple"), "text2music")
        self.assertEqual(normalize_task_type("custom"), "text2music")
        self.assertEqual(normalize_task_type("remix"), "cover")

    def test_model_capabilities(self):
        self.assertIn("complete", supported_tasks_for_model("acestep-v15-base"))
        self.assertNotIn("complete", supported_tasks_for_model("acestep-v15-turbo"))
        with self.assertRaises(ValueError):
            ensure_task_supported("acestep-v15-xl-turbo", "extract")
        ensure_task_supported("acestep-v15-xl-base", "extract")

    def test_known_model_profiles_are_complete(self):
        required = {"label", "summary", "best_for", "steps", "cfg", "quality", "speed", "tasks"}
        for name in KNOWN_ACE_STEP_MODELS:
            with self.subTest(name=name):
                self.assertIn(name, MODEL_PROFILES)
                profile = model_profile(name, installed=name == "acestep-v15-turbo")
                self.assertTrue(required.issubset(profile))
                self.assertEqual(profile["tasks"], supported_tasks_for_model(name))
                self.assertIn("installed", profile)
                self.assertIn("dropdown_label", profile)

    def test_official_manifest_covers_models_endpoints_and_params(self):
        manifest = official_manifest()
        self.assertEqual(manifest["manifest_version"], OFFICIAL_ACE_STEP_MANIFEST["manifest_version"])
        self.assertIn("unreleased", manifest["status_values"])
        self.assertIn("not_applicable", manifest["status_values"])
        self.assertIn("acestep-v15-turbo-rl", manifest["dit_models"])
        self.assertEqual(manifest["dit_models"]["acestep-v15-turbo-rl"]["status"], "unreleased")
        self.assertIn("acestep-v15-turbo-rl", OFFICIAL_UNRELEASED_MODELS)
        for endpoint in ["/release_task", "/query_result", "/v1/init", "/v1/stats", "/v1/audio", "/v1/training/start_lokr"]:
            self.assertIn(endpoint, manifest["api_endpoints"])
            self.assertIn(manifest["api_endpoints"][endpoint]["status"], {"supported", "guarded"})
        for field in ["instruction", "reference_audio", "src_audio", "thinking", "lm_cfg_scale", "timesteps"]:
            self.assertIn(field, manifest["generation_params"])
        for field in ["batch_size", "allow_lm_batch", "use_random_seed", "audio_format"]:
            self.assertIn(field, manifest["generation_config"])
        for alias in ["prompt", "lm_model_path", "reference_audio_path", "src_audio_path"]:
            self.assertTrue(any(alias in values for values in manifest["payload_aliases"].values()))
        self.assertIn("user_metadata", manifest["acejam_extension_params"]["fields"])
        self.assertIn("lm_repetition_penalty", manifest["acejam_extension_params"]["fields"])
        self.assertIn("ACESTEP_OFFLOAD_TO_CPU", manifest["runtime_controls"])
        self.assertIn("lokr_train", manifest["training_features"])

    def test_model_profile_fallback_and_installed_flags(self):
        profile = model_profile("acestep-v15-custom-base", installed=True)
        self.assertTrue(profile["installed"])
        self.assertIn("complete", profile["tasks"])
        profiles = model_profiles_for_models(["acestep-v15-turbo", "acestep-v15-custom-base"], {"acestep-v15-turbo"})
        self.assertTrue(profiles["acestep-v15-turbo"]["installed"])
        self.assertFalse(profiles["acestep-v15-custom-base"]["installed"])
        self.assertEqual(recommended_song_model({"acestep-v15-turbo"}), "acestep-v15-turbo")

    def test_lm_profiles_are_complete(self):
        required = {"label", "summary", "best_for", "quality", "speed", "tasks", "installed"}
        for name in ACE_STEP_LM_MODELS:
            with self.subTest(name=name):
                self.assertIn(name, LM_MODEL_PROFILES)
                profile = lm_model_profile(name, installed=name == "acestep-5Hz-lm-1.7B")
                self.assertTrue(required.issubset(profile))
        profiles = lm_model_profiles_for_models(ACE_STEP_LM_MODELS, {"acestep-5Hz-lm-1.7B"})
        self.assertTrue(profiles["none"]["installed"])
        self.assertTrue(profiles["auto"]["installed"])
        self.assertTrue(profiles["acestep-5Hz-lm-1.7B"]["installed"])
        self.assertEqual(recommended_lm_model({"acestep-5Hz-lm-1.7B"}), "acestep-5Hz-lm-1.7B")
        self.assertEqual(
            recommended_lm_model({"acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"}),
            "acestep-5Hz-lm-4B",
        )

    def test_instruction_templates(self):
        self.assertIn("vocals", build_task_instruction("extract", "vocals"))
        self.assertIn("drums", build_task_instruction("lego", ["drums"]))
        self.assertIn("drums, bass", build_task_instruction("complete", ["drums", "bass"]))

    def test_track_normalization(self):
        self.assertEqual(normalize_track_names("backing vocals, drums, nope"), ["backing_vocals", "drums"])

    def test_timesteps(self):
        self.assertEqual(parse_timesteps("1, 0.5, 0"), [1.0, 0.5, 0.0])
        self.assertIsNone(parse_timesteps(""))

    def test_official_keyscale_registry_and_aliases(self):
        self.assertEqual(DEFAULT_BPM, 95)
        self.assertEqual(DEFAULT_KEY_SCALE, "A minor")
        self.assertEqual(len(VALID_KEY_SCALES), 70)
        for key in ["A minor", "C major", "F# minor", "F♯ minor", "Bb major", "B♭ major"]:
            self.assertIn(key, VALID_KEY_SCALES)

        self.assertEqual(normalize_key_scale("Am"), "A minor")
        self.assertEqual(normalize_key_scale("C Major"), "C major")
        self.assertEqual(normalize_key_scale("F# Minor"), "F# minor")
        self.assertEqual(normalize_key_scale("auto"), "")
        with self.assertRaises(ValueError):
            normalize_key_scale("H major")

    def test_text2music_vocal_lyrics_guard(self):
        self.assertTrue(needs_vocal_lyrics(task_type="text2music", instrumental=False, lyrics=""))
        self.assertTrue(needs_vocal_lyrics(task_type="text2music", instrumental=False, lyrics="[Instrumental]"))
        self.assertFalse(needs_vocal_lyrics(task_type="text2music", instrumental=True, lyrics=""))
        self.assertFalse(needs_vocal_lyrics(task_type="text2music", instrumental=False, lyrics="[Verse]\nReal lyric"))
        self.assertFalse(needs_vocal_lyrics(task_type="cover", instrumental=False, lyrics=""))
        self.assertFalse(needs_vocal_lyrics(task_type="text2music", instrumental=False, lyrics="", sample_mode=True))
        self.assertFalse(needs_vocal_lyrics(task_type="text2music", instrumental=False, lyrics="", sample_query="write it"))

    def test_payload_normalization_moves_lyrics_out_of_caption(self):
        payload = normalize_generation_text_fields(
            {
                "title": "Night Shift",
                "caption": "[Verse]\nI clock in when the city sleeps\n[Chorus]\nWe rise",
                "description": "[Verse]\nI clock in when the city sleeps\n[Chorus]\nWe rise",
                "lyrics": "",
                "instrumental": False,
            },
            task_type="text2music",
        )
        self.assertIn("[Verse]", payload["lyrics"])
        self.assertNotIn("[Verse]", payload["caption"])
        self.assertEqual(payload["description"], "")
        self.assertIn("Moved likely lyrics", " ".join(payload["payload_warnings"]))
        self.assertEqual(payload["lyrics_source"], "caption_repaired")
        second_pass = normalize_generation_text_fields(payload, task_type="text2music")
        self.assertEqual(second_pass["lyrics_source"], "caption_repaired")
        self.assertEqual(second_pass["caption_source"], "repaired_from_tags")

    def test_payload_normalization_keeps_tags_as_caption(self):
        payload = normalize_generation_text_fields(
            {
                "caption": "Dutch rap, warm keys, 808 bass, emotional hook",
                "lyrics": "[Verse]\nreal words",
            },
            task_type="text2music",
        )
        self.assertEqual(payload["caption"], "Dutch rap, warm keys, 808 bass, emotional hook")
        self.assertIn("Dutch rap", payload["tag_list"])
        self.assertEqual(payload["payload_warnings"], [])

    def test_payload_normalization_sets_instrumental_lyrics(self):
        payload = normalize_generation_text_fields({"caption": "ambient piano", "instrumental": True}, task_type="text2music")
        self.assertEqual(payload["lyrics"], "[Instrumental]")
        self.assertEqual(payload["lyrics_source"], "instrumental_default")

    def test_ace_step_text_budget_cleans_and_fits_runtime_text(self):
        polluted = (
            "<think>private chain</think>\n"
            "I will now generate the final lyrics.\n"
            "[Verse]\n"
            + "\n".join(f"Line {index} for the song" for index in range(900))
        )

        payload = apply_ace_step_text_budget(
            {"caption": "x" * (ACE_STEP_CAPTION_CHAR_LIMIT + 20), "lyrics": polluted},
            task_type="text2music",
        )

        self.assertLessEqual(len(payload["caption"]), ACE_STEP_CAPTION_CHAR_LIMIT)
        self.assertLessEqual(len(payload["lyrics"]), ACE_STEP_LYRICS_CHAR_LIMIT)
        self.assertNotIn("<think>", payload["lyrics"])
        self.assertNotIn("I will now generate", payload["lyrics"])
        self.assertEqual(payload["ace_step_text_budget"]["source_lyrics_char_count"], len(polluted))
        self.assertLessEqual(payload["ace_step_text_budget"]["runtime_lyrics_char_count"], ACE_STEP_LYRICS_CHAR_LIMIT)
        self.assertIn("Lyrics fit", " ".join(payload["payload_warnings"]))

    def test_official_field_detection(self):
        self.assertEqual(official_fields_used({"audio_format": "wav", "thinking": False}), [])
        self.assertNotIn("sample_query", official_fields_used({"description": "metadata only"}))
        self.assertIn("thinking", official_fields_used({"thinking": True}))
        self.assertIn("audio_format", official_fields_used({"audio_format": "mp3"}))
        self.assertIn("lm_temperature", official_fields_used({"lm_temperature": 1.1}))
        self.assertIn("lm_repetition_penalty", official_fields_used({"lm_repetition_penalty": 1.2}))
        self.assertIn("offload_to_cpu", official_fields_used({"offload_to_cpu": True}))

    def test_ui_schema_includes_custom_controls(self):
        schema = studio_ui_schema()
        self.assertEqual(schema["payload_validation_endpoint"], "/api/payload/validate")
        self.assertIn("payload_contract_version", schema)
        self.assertIn("bpm", schema["custom_sections"]["music_metadata"])
        self.assertEqual(schema["custom_sections"]["ace_step_lm"], [])
        self.assertIn("ollama_model", schema["custom_sections"]["ollama_planner"])
        self.assertIn("mp3", schema["audio_formats"])
        self.assertIn("thinking", schema["official_only_fields"])
        self.assertEqual(schema["official_parity_endpoint"], "/api/ace-step/parity")
        self.assertIn("runtime", schema["custom_sections"])
        self.assertEqual(schema["quality_policy"]["turbo_models"]["inference_steps"], 8)
        self.assertEqual(schema["quality_policy"]["sft_base_models"]["inference_steps"], 64)
        self.assertEqual(schema["quality_policy"]["balanced_pro_models"]["inference_steps"], 50)
        self.assertEqual(schema["default_quality_profile"], "chart_master")
        self.assertEqual(schema["default_bpm"], 95)
        self.assertEqual(schema["default_key_scale"], "A minor")
        self.assertEqual(len(schema["valid_keyscales"]), 70)
        self.assertEqual(schema["ranges"]["bpm"], [30, 300])
        self.assertEqual(schema["options"]["key_scale"][0], "auto")
        self.assertIn("A minor", schema["options"]["key_scale"])
        self.assertEqual(schema["ace_step_coverage"]["status"], "complete")
        self.assertEqual(schema["payload_contract_version"], "2026-04-26")
        self.assertEqual(schema["settings_policy_version"], "ace-step-settings-parity-2026-04-26")
        self.assertIn("ace_step_settings_registry", schema)

    def test_ace_step_settings_registry_covers_official_fields(self):
        registry = ace_step_settings_registry()
        settings = registry["settings"]
        manifest = official_manifest()
        for field in manifest["generation_params"]:
            with self.subTest(field=field):
                self.assertIn(field, settings)
        for field in manifest["generation_config"]:
            with self.subTest(field=field):
                self.assertIn(field, settings)
        self.assertEqual(registry["profiles"]["official_defaults"]["sampler_mode"], "euler")
        self.assertEqual(registry["profiles"]["official_defaults"]["audio_format"], "flac")
        self.assertEqual(registry["default_quality_profile"], "chart_master")
        self.assertEqual(registry["coverage"]["status"], "complete")
        self.assertEqual(registry["profiles"]["chart_master"]["models"]["acestep-v15-xl-sft"]["inference_steps"], 64)
        self.assertEqual(registry["profiles"]["balanced_pro"]["models"]["acestep-v15-xl-sft"]["inference_steps"], 50)
        self.assertEqual(registry["profiles"]["docs_recommended"]["lm"]["lm_temperature"], 0.85)
        self.assertEqual(registry["profiles"]["docs_recommended"]["lm"]["lm_top_k"], 0)
        self.assertEqual(registry["profiles"]["docs_recommended"]["lm"]["lm_top_p"], 0.9)
        self.assertEqual(registry["settings"]["use_cot_lyrics"]["status"], "reserved")
        self.assertEqual(registry["settings"]["cot_caption"]["status"], "read_only_lm_output")
        self.assertEqual(registry["pro_quality"]["version"], PRO_QUALITY_AUDIT_VERSION)
        self.assertIn("complete", registry["task_policy"]["source_locked_duration"])

    def test_settings_compliance_marks_ignored_and_overrides(self):
        compliance = ace_step_settings_compliance(
            {
                "guidance_scale": 8.0,
                "use_adg": True,
                "thinking": True,
                "timesteps": [1.0, 0.5, 0.0],
                "audio_format": "mp3",
            },
            task_type="cover",
            song_model="acestep-v15-turbo",
            runner_plan="fast",
        )
        self.assertFalse(compliance["valid"])
        self.assertIn("guidance_scale", compliance["ignored"])
        self.assertIn("use_adg", compliance["ignored"])
        self.assertIn("thinking", compliance["ignored"])
        self.assertIn("audio_format", compliance["unsupported"])
        self.assertIn("timesteps_override_steps_shift", compliance["notes"])
        self.assertIn("duration_source_locked", compliance["notes"])

    def test_hit_readiness_and_runtime_planner_reports(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "quality_profile": "chart_master",
            "caption": "German schlager, bright brass, accordion sparkle",
            "lyrics": "[Verse]\nHeute Nacht klingt die Stadt\n\n[Chorus]\nWir singen weiter",
            "bpm": 95,
            "key_scale": "A minor",
            "time_signature": "4",
            "duration": 180,
            "batch_size": 3,
            "inference_steps": 64,
            "lm_backend": "mlx",
        }

        readiness = hit_readiness_report(payload, task_type="text2music", song_model="acestep-v15-xl-sft")
        planner = runtime_planner_report(payload, task_type="text2music", song_model="acestep-v15-xl-sft")

        self.assertEqual(readiness["version"], PRO_QUALITY_AUDIT_VERSION)
        self.assertGreaterEqual(readiness["score"], 80)
        self.assertIn(readiness["status"], {"pass", "warn"})
        self.assertEqual(planner["takes"], 3)
        self.assertEqual(planner["steps"], 64)
        self.assertIn(planner["risk"], {"medium", "high"})

    def test_hit_readiness_fails_caption_and_fallback_lyric_leakage(self):
        payload = {
            "task_type": "text2music",
            "song_model": "acestep-v15-xl-sft",
            "quality_profile": "chart_master",
            "caption": "Track 1: Broken Caption\n[Verse]\nThis is lyric leakage",
            "lyrics": "[Verse]\nMorning finds the you on the floor\n\n[Chorus]\nThe you is here",
            "bpm": 95,
            "key_scale": "A minor",
            "time_signature": "4",
            "duration": 60,
            "batch_size": 3,
            "inference_steps": 64,
            "lm_backend": "mlx",
        }

        readiness = hit_readiness_report(payload, task_type="text2music", song_model="acestep-v15-xl-sft")
        statuses = {check["id"]: check["status"] for check in readiness["checks"]}

        self.assertEqual(readiness["status"], "fail")
        self.assertEqual(statuses["caption_integrity"], "fail")
        self.assertEqual(statuses["no_fallback_artifacts"], "fail")

    def test_docs_best_model_settings(self):
        self.assertEqual(docs_best_model_settings("acestep-v15-turbo")["inference_steps"], 8)
        self.assertEqual(docs_best_model_settings("acestep-v15-turbo")["guidance_scale"], 7.0)
        self.assertEqual(docs_best_model_settings("acestep-v15-turbo-shift3")["inference_steps"], 8)
        self.assertEqual(docs_best_model_settings("acestep-v15-sft")["inference_steps"], 64)
        self.assertEqual(docs_best_model_settings("acestep-v15-base")["inference_steps"], 64)
        self.assertEqual(docs_best_model_settings("acestep-v15-xl-turbo")["inference_steps"], 8)
        self.assertEqual(docs_best_model_settings("acestep-v15-xl-sft")["inference_steps"], 64)
        self.assertEqual(docs_best_model_settings("acestep-v15-xl-base")["inference_steps"], 64)
        self.assertEqual(docs_best_model_settings("acestep-v15-xl-sft")["shift"], 3.0)
        self.assertFalse(docs_best_model_settings("acestep-v15-xl-sft")["use_adg"])
        self.assertTrue(docs_best_model_settings("acestep-v15-xl-base")["use_adg"])

    def test_safe_id(self):
        self.assertEqual(safe_id("abc123_DEF-9"), "abc123_DEF-9")
        with self.assertRaises(ValueError):
            safe_id("../bad")


if __name__ == "__main__":
    unittest.main()
