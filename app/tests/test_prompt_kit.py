import unittest

from prompt_kit import (
    GENRE_MODULES,
    LANGUAGE_PRESETS,
    PROMPT_KIT_METADATA_FIELDS,
    PROMPT_KIT_OUTPUT_CONTRACT_FIELDS,
    PROMPT_KIT_VERSION,
    VALIDATION_CHECKLIST,
    infer_genre_modules,
    is_sparse_lyric_genre,
    kit_metadata_defaults,
    language_preset,
    prompt_kit_payload,
    prompt_kit_system_block,
    section_map_for,
)


class PromptKitTest(unittest.TestCase):
    def test_language_presets_are_complete(self):
        expected = {
            "en", "nl", "fr", "es", "pt", "de", "ar", "tr", "hi", "ur", "pa", "ja", "ko",
            "zh", "yue", "id", "ms", "sw", "pcm", "it", "pl", "he", "ru", "unknown",
        }
        self.assertEqual(set(LANGUAGE_PRESETS), expected)
        self.assertIn("no Pinyin", language_preset("zh")["notes"])
        self.assertIn("no romaji", language_preset("ja")["notes"])
        self.assertIn("Arabic script", language_preset("ar")["script"])
        self.assertIn("Hebrew", language_preset("he")["script"])
        self.assertIn("Cyrillic", language_preset("ru")["script"])
        self.assertIn("Dutch-English blend", language_preset("nl")["notes"])

    def test_genre_modules_and_payload_contract(self):
        self.assertEqual(len(GENRE_MODULES), 26)
        for key in ["hiphop", "boom_bap", "trap", "drill", "rnb", "amapiano", "techno", "cinematic", "ambient", "kpop_jpop"]:
            self.assertIn(key, GENRE_MODULES)
        payload = prompt_kit_payload()
        self.assertEqual(payload["version"], PROMPT_KIT_VERSION)
        self.assertEqual(len(PROMPT_KIT_OUTPUT_CONTRACT_FIELDS), 24)
        self.assertEqual(PROMPT_KIT_OUTPUT_CONTRACT_FIELDS[8], "ace_caption")
        self.assertIn("generation_settings", PROMPT_KIT_OUTPUT_CONTRACT_FIELDS)
        self.assertIn("output_contract_fields", payload)
        self.assertIn("copy_paste_block", PROMPT_KIT_METADATA_FIELDS)
        self.assertIn("validation_checklist", payload)
        self.assertIn("troubleshooting_matrix", payload)
        self.assertGreaterEqual(len(VALIDATION_CHECKLIST), 8)

    def test_prompt_kit_does_not_expose_adlib_terms(self):
        payload_text = str(prompt_kit_payload()).lower()
        self.assertNotIn("ad-lib", payload_text)
        self.assertNotIn("ad libs", payload_text)
        self.assertNotIn("ad-libs", payload_text)

    def test_genre_routing_and_section_maps(self):
        rap_modules = infer_genre_modules("raw boom bap rap with internal rhyme")
        self.assertEqual(rap_modules[0]["slug"], "boom_bap")
        techno_sections = section_map_for(240, "instrumental techno club track", instrumental=True)
        techno_tags = [item["tag"] for item in techno_sections]
        self.assertIn("[Drop]", techno_tags)
        self.assertIn("[Final Drop]", techno_tags)
        self.assertTrue(is_sparse_lyric_genre("ambient techno score"))
        rap_sections = section_map_for(240, "rap")
        rap_tags = [item["tag"] for item in rap_sections]
        self.assertIn("[Verse 1]", rap_tags)
        self.assertIn("[Chorus]", rap_tags)
        self.assertIn("[Verse 3 - Beat Switch]", rap_tags)

    def test_metadata_defaults_and_system_block(self):
        meta = kit_metadata_defaults("text2music", "ko", "k-pop bright synths", 220)
        self.assertEqual(meta["prompt_kit_version"], PROMPT_KIT_VERSION)
        self.assertEqual(meta["target_language"], "ko")
        self.assertIn("kpop_jpop", meta["genre_modules"])
        self.assertIn("quality_checks", meta)
        block = prompt_kit_system_block("custom")
        self.assertIn(PROMPT_KIT_VERSION, block)
        self.assertIn("Never hardcode planner_lm_provider to ollama", block)
        self.assertIn("song_intent", block)
        self.assertIn("concrete sonic portrait", block)


if __name__ == "__main__":
    unittest.main()
