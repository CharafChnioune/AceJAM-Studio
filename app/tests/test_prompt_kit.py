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
        self.assertIn("[Verse 2]", rap_tags)
        self.assertNotIn("[Verse 3 - rap]", rap_tags)

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

    def test_system_block_appends_acestep_full_reference_for_music_modes(self):
        block = prompt_kit_system_block("custom")
        self.assertIn("## ACE-Step Authoring Rules", block)
        self.assertIn("## ACE-Step Tag Library", block)
        self.assertIn("## Producer-Format Cookbook", block)
        self.assertIn("## Rap-Mode Cookbook", block)
        self.assertIn("## Songwriter Craft Cookbook", block)
        self.assertIn("## Lyric Anti-Patterns", block)
        self.assertIn("## Worked Examples", block)
        self.assertIn("Dr. Dre / G-funk era", block)
        self.assertIn("No I.D. / Common-era boom bap", block)
        self.assertIn("Metro Boomin / dark trap", block)
        self.assertIn("J Dilla / Soulquarian feel", block)
        self.assertIn("DJ Premier / 90s boom bap", block)
        # New cookbook entries
        self.assertIn("Pete Rock / golden-age soul boom bap", block)
        self.assertIn("Havoc / Mobb Deep production", block)
        # Era-specific Dre split (G-funk vs Chronic 2001 / Get Rich era)
        self.assertIn("Dr. Dre / Chronic 2001 + Get Rich era", block)
        self.assertIn("Mike Elizondo live bass guitar", block)
        self.assertIn("Million Dollar Mix polish", block)
        self.assertIn("In Da Club bounce", block)
        # Production-grade upgrades surface signature elements
        self.assertIn("whistle synth lead", block)  # Dre signature
        self.assertIn("replayed funk interpolation", block)  # Dre sample treatment
        self.assertIn("SP-1200 played bassline", block)  # No I.D. signature
        self.assertIn("filtered choir", block)  # Metro signature
        self.assertIn("mono low end", block)  # mono/wide-split signal
        self.assertIn("[Hook]", block)
        self.assertIn("[Verse - rap]", block)
        self.assertIn("[Verse - whispered]", block)
        self.assertIn("Modifier syntax", block)
        self.assertIn("Background vocals use parentheses", block)
        self.assertIn("ALL CAPS", block)
        # Vocal-technique words live in caption tags (taxonomy), section
        # modifiers like [Verse - whispered] are still valid in lyrics.
        self.assertIn("ad-libs", block)
        self.assertIn("falsetto", block)
        self.assertIn("call-and-response", block)
        # Authoring rule explicitly states standalone brackets are wrong.
        self.assertIn("standalone bracket lines", block)
        # New 16-bar floor + craft + anti-pattern rules surface in authoring rules
        self.assertIn("16 bars", block)
        self.assertIn("Six-dimension beat-spec", block)
        self.assertIn("Sample-source rule", block)
        self.assertIn("Drum kit triad rule", block)
        self.assertIn("Mono-bass / wide-pad split", block)
        # Songwriter craft cookbook surfaces per-artist signatures
        self.assertIn("eminem signature", block)
        self.assertIn("nas signature", block)
        self.assertIn("kendrick signature", block)
        self.assertIn("tupac signature", block)
        # Anti-patterns block lists cliches
        self.assertIn("neon dreams", block)
        self.assertIn("shattered dreams", block)
        # Worked examples carry concrete ad-lib parens patterns
        self.assertIn("(yeah)", block)
        self.assertIn("(skrrt)", block)

    def test_system_block_for_album_includes_producer_cookbook(self):
        block = prompt_kit_system_block("album")
        self.assertIn("## Producer-Format Cookbook", block)
        self.assertIn("## Rap-Mode Cookbook", block)
        self.assertIn("## Worked Examples", block)
        self.assertIn("[Verse - rap]", block)

    def test_system_block_skips_cookbook_for_non_music_modes(self):
        for mode in ("image", "video", "trainer"):
            block = prompt_kit_system_block(mode)
            self.assertNotIn("## Producer-Format Cookbook", block)
            self.assertNotIn("## Rap-Mode Cookbook", block)
            self.assertNotIn("## Worked Examples", block)
            self.assertNotIn("## ACE-Step Tag Library", block)

    def test_system_block_for_source_audio_modes_has_taxonomy_but_no_producer(self):
        for mode in ("cover", "repaint", "extract", "lego", "complete"):
            block = prompt_kit_system_block(mode)
            self.assertIn("## ACE-Step Tag Library", block)
            self.assertIn("## ACE-Step Authoring Rules", block)
            self.assertNotIn("## Producer-Format Cookbook", block)
            self.assertNotIn("## Worked Examples", block)


if __name__ == "__main__":
    unittest.main()
