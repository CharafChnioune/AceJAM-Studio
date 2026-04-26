import unittest

from studio_core import (
    ACE_STEP_LM_MODELS,
    KNOWN_ACE_STEP_MODELS,
    LM_MODEL_PROFILES,
    MODEL_PROFILES,
    OFFICIAL_ACE_STEP_MANIFEST,
    OFFICIAL_UNRELEASED_MODELS,
    build_task_instruction,
    docs_best_model_settings,
    ensure_task_supported,
    lm_model_profile,
    lm_model_profiles_for_models,
    model_profile,
    model_profiles_for_models,
    needs_vocal_lyrics,
    normalize_generation_text_fields,
    normalize_task_type,
    official_fields_used,
    official_manifest,
    normalize_track_names,
    parse_timesteps,
    recommended_lm_model,
    recommended_song_model,
    safe_id,
    studio_ui_schema,
    supported_tasks_for_model,
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

    def test_official_field_detection(self):
        self.assertEqual(official_fields_used({"audio_format": "wav", "thinking": True}), [])
        self.assertNotIn("sample_query", official_fields_used({"description": "metadata only"}))
        self.assertIn("thinking", official_fields_used({"thinking": False}))
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

    def test_docs_best_model_settings(self):
        self.assertEqual(docs_best_model_settings("acestep-v15-turbo")["inference_steps"], 8)
        self.assertEqual(docs_best_model_settings("acestep-v15-turbo")["guidance_scale"], 7.0)
        self.assertEqual(docs_best_model_settings("acestep-v15-xl-sft")["inference_steps"], 64)
        self.assertEqual(docs_best_model_settings("acestep-v15-xl-sft")["shift"], 1.0)

    def test_safe_id(self):
        self.assertEqual(safe_id("abc123_DEF-9"), "abc123_DEF-9")
        with self.assertRaises(ValueError):
            safe_id("../bad")


if __name__ == "__main__":
    unittest.main()
