import unittest

from studio_core import (
    ACE_STEP_LM_MODELS,
    KNOWN_ACE_STEP_MODELS,
    LM_MODEL_PROFILES,
    MODEL_PROFILES,
    build_task_instruction,
    ensure_task_supported,
    lm_model_profile,
    lm_model_profiles_for_models,
    model_profile,
    model_profiles_for_models,
    normalize_task_type,
    normalize_track_names,
    parse_timesteps,
    recommended_lm_model,
    recommended_song_model,
    safe_id,
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

    def test_instruction_templates(self):
        self.assertIn("vocals", build_task_instruction("extract", "vocals"))
        self.assertIn("drums", build_task_instruction("lego", ["drums"]))
        self.assertIn("drums, bass", build_task_instruction("complete", ["drums", "bass"]))

    def test_track_normalization(self):
        self.assertEqual(normalize_track_names("backing vocals, drums, nope"), ["backing_vocals", "drums"])

    def test_timesteps(self):
        self.assertEqual(parse_timesteps("1, 0.5, 0"), [1.0, 0.5, 0.0])
        self.assertIsNone(parse_timesteps(""))

    def test_safe_id(self):
        self.assertEqual(safe_id("abc123_DEF-9"), "abc123_DEF-9")
        with self.assertRaises(ValueError):
            safe_id("../bad")


if __name__ == "__main__":
    unittest.main()
