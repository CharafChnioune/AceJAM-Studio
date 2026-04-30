import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import album_crew as album_crew_module
from album_crew import (
    CREWAI_AGENT_MAX_ITER,
    CREWAI_AGENT_MAX_RETRY_LIMIT,
    CREWAI_LLM_CONTEXT_WINDOW,
    CREWAI_LLM_MAX_TOKENS,
    CREWAI_LLM_NUM_PREDICT,
    CREWAI_LLM_TIMEOUT_SECONDS,
    CREWAI_LMSTUDIO_MAX_TOKENS,
    CREWAI_LMSTUDIO_NO_THINK_PREFILL,
    CREWAI_LMSTUDIO_PIN_CONTEXT,
    CREWAI_MEMORY_CONTENT_LIMIT,
    CREWAI_MEMORY_DIR,
    CREWAI_RESPECT_CONTEXT_WINDOW,
    CREWAI_TASK_MAX_RETRIES,
    CREWAI_VERBOSE,
    DEFAULT_ALBUM_EMBEDDING_MODEL,
    DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
    AlbumBiblePayloadModel,
    TrackProductionPayloadModel,
    _album_genre_hint,
    _compact_track_memory_record,
    _coerce_agent_lyrics_payload,
    _crewai_llm_kwargs,
    _crewai_task_callback,
    _kickoff_crewai_compact,
    _empty_response_fallback_text,
    _is_lmstudio_model_crash,
    _is_empty_response_payload,
    _json_object_from_text,
    _lyric_like_text,
    _lmstudio_no_think_args,
    _lmstudio_no_think_messages,
    _make_llm,
    _select_crewai_tools,
    _make_album_memory,
    _make_album_memory_writer,
    _remember_compact,
    _ollama_embedder_config,
    _strip_thinking_blocks,
    _track_json_guardrail_factory,
    create_album_bible_crew,
    create_album_crew,
    create_track_production_crew,
    crewai_output_log_path,
    plan_album,
    preflight_album_local_llm,
)
from ace_step_track_prompt_template import (
    ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION,
    CAPTION_DIMENSIONS,
    compact_full_tag_library,
    render_track_prompt_template,
)
from songwriting_toolkit import (
    ALBUM_FINAL_MODEL,
    ALBUM_MODEL_PORTFOLIO_MODELS,
    CRAFT_TOOLS,
    LYRIC_META_TAGS,
    TAG_TAXONOMY,
    build_album_plan,
    album_model_portfolio,
    album_models_for_strategy,
    choose_song_model,
    derive_artist_name,
    lyric_stats,
    lyric_length_plan,
    make_crewai_tools,
    expand_lyrics_for_duration,
    normalize_album_tracks,
    normalize_artist_name,
    parse_duration_seconds,
    sanitize_artist_references,
    toolkit_payload,
)
from prompt_kit import PROMPT_KIT_VERSION
from user_album_contract import extract_user_album_contract


class SongwritingToolkitTest(unittest.TestCase):
    SAFE_CONTRACT_PROMPT = """
Album: Market Lights
Concept: A safe city album about rebuilding trust after a long blackout.
Track 1: "Morning Market" (Produced by Ada North)
(BPM: 92 | Key: A minor | Style: warm boom-bap)
The Vibe: brass stabs, vinyl dust, calm crowd energy.
The Narrative: neighbors reopen stalls and trade stories instead of rumors.
Lyrics:
"Open the shutters"
"Coffee on the corner"
Naming Drop: "market bell"

Track 2: "Rooftop Letters" (Produced by Mira South)
(BPM: 104 | Style: melodic house)
The Vibe: soft synth pulse and bright handclaps.
The Narrative: friends read old letters on a roof as the lights come back.
Lyrics:
"Letters in the skyline"
"""

    def _valid_album_test_lyrics(self, required_phrases=None, sections=None, lines_per_section=8):
        required = [str(item) for item in (required_phrases or []) if str(item)]
        section_names = sections or ["Verse", "Chorus", "Outro"]
        lines = []
        phrase_index = 0
        for section in section_names:
            lines.append(f"[{section}]")
            for line_index in range(lines_per_section):
                if phrase_index < len(required):
                    lines.append(required[phrase_index])
                    phrase_index += 1
                else:
                    lines.append(f"Market lanterns carry hopeful voices number {section.lower()} {line_index}")
        return "\n".join(lines)

    def test_user_album_contract_extracts_safe_album_prompt(self):
        contract = extract_user_album_contract(self.SAFE_CONTRACT_PROMPT, 2, "en")

        self.assertTrue(contract["applied"])
        self.assertEqual(contract["album_title"], "Market Lights")
        self.assertEqual(contract["track_count"], 2)
        self.assertEqual(contract["tracks"][0]["locked_title"], "Morning Market")
        self.assertEqual(contract["tracks"][0]["producer_credit"], "Ada North")
        self.assertEqual(contract["tracks"][0]["bpm"], 92)
        self.assertEqual(contract["tracks"][0]["key_scale"], "A minor")
        self.assertEqual(contract["tracks"][0]["style"], "warm boom-bap")
        self.assertIn("Open the shutters", contract["tracks"][0]["required_phrases"])
        self.assertEqual(contract["tracks"][0]["content_policy_status"], "safe")

    def test_user_album_contract_extracts_required_hook_phrase(self):
        contract = extract_user_album_contract(
            """
Album: Midnight Bakery
Track 1: "Neon Bakery Lights" (Produced by Studio House)
BPM: 95 | Duration: 2:40 | Style: upbeat city-pop
The Vibe: rubber bass and bright piano.
The Narrative: the block gathers after midnight.
Required hook phrase: Neon bakery lights keep calling us home.
""",
            1,
            "en",
        )

        track = contract["tracks"][0]
        self.assertEqual(track["duration"], 160)
        self.assertEqual(track["narrative"], "the block gathers after midnight.")
        self.assertIn("Neon bakery lights keep calling us home.", track["required_phrases"])
        self.assertNotIn("[]", track["required_phrases"])

    def test_user_album_contract_required_phrase_matching_is_punctuation_tolerant(self):
        from user_album_contract import apply_user_album_contract_to_track

        contract = extract_user_album_contract(
            """
Album: Concrete Test
Track 1: "Concrete Canyons" (Produced by Studio House)
Lyrics:
"They paved them blocks just to hide what’s real,"
"Death Row… East Coast… Closed doors…"
""",
            1,
            "en",
        )
        logs = []
        track = {
            "track_number": 1,
            "title": "Concrete Canyons",
            "lyrics": "They paved them blocks just to hide what's real\nDeath Row... East Coast... Closed doors",
        }

        repaired = apply_user_album_contract_to_track(track, contract, 0, logs)

        self.assertNotIn("required_phrase", repaired["contract_repaired_fields"])
        self.assertNotIn("required_phrase", " ".join(logs))

    def test_user_album_contract_handles_prod_shorthand_and_verse_label(self):
        contract = extract_user_album_contract(
            """
Album: You Buried the Wrong Man
Track 1: Concrete Canyons (Prod. Dr. Dre)
(BPM: 78 | Style: Heavy West Coast G-Funk)
Vibe: Low-end rumble, sirens, West Coast weight
Verse: They paved them blocks just to hide what's real,
Boardroom smiles while they cut them deals.
Naming Drop Style: "Death Row", "East Coast", "Closed doors"
""",
            1,
            "en",
        )

        track = contract["tracks"][0]
        self.assertEqual(track["locked_title"], "Concrete Canyons")
        self.assertEqual(track["producer_credit"], "Dr. Dre")
        self.assertEqual(track["bpm"], 78)
        self.assertEqual(track["style"], "Heavy West Coast G-Funk")
        self.assertEqual(track["vibe"], "Low-end rumble, sirens, West Coast weight")
        self.assertIn("They paved them blocks", track["required_lyrics"])
        self.assertIn("Death Row", track["required_phrases"])
        self.assertNotIn("Death Row", track["style"])

    def test_build_album_plan_uses_locked_user_titles_and_producers(self):
        result = build_album_plan(
            self.SAFE_CONTRACT_PROMPT,
            2,
            60,
            {
                "installed_models": ["acestep-v15-turbo"],
                "song_model_strategy": "best_installed",
                "language": "en",
            },
        )

        self.assertEqual([track["title"] for track in result["tracks"]], ["Morning Market", "Rooftop Letters"])
        self.assertEqual(result["tracks"][0]["producer_credit"], "Ada North")
        self.assertEqual(result["tracks"][0]["bpm"], 92)
        self.assertTrue(result["tracks"][0]["input_contract_applied"])
        self.assertEqual(result["tracks"][0]["settings_policy_version"], "ace-step-settings-parity-2026-04-26")
        self.assertIn("settings_compliance", result["tracks"][0])
        self.assertNotIn("Protocol", result["tracks"][0]["title"])
        self.assertNotIn("Season", result["tracks"][1]["title"])

    def test_all_lyrics_pass_through_without_blocking(self):
        contract = extract_user_album_contract(
            """
Track 1: "Safe Title" (Produced by Ada North)
Lyrics:
kill all the rivals
""",
            1,
            "en",
        )

        self.assertEqual(contract["blocked_unsafe_count"], 0)
        self.assertEqual(contract["tracks"][0]["content_policy_status"], "safe")
        self.assertIn("kill all the rivals", contract["tracks"][0]["required_lyrics"])
        self.assertTrue(len(contract["tracks"][0]["required_phrases"]) > 0)

    def test_toolkit_payload_has_official_dimensions_and_all_tools(self):
        payload = toolkit_payload({"acestep-v15-turbo"})
        for category in [
            "genre_style",
            "mood_atmosphere",
            "instruments",
            "timbre_texture",
            "era_reference",
            "production_style",
            "vocal_character",
            "speed_rhythm",
            "structure_hints",
            "track_stems",
        ]:
            self.assertIn(category, TAG_TAXONOMY)
            self.assertGreaterEqual(len(payload["tag_taxonomy"][category]), 8)
        self.assertIn("performance_modifiers", LYRIC_META_TAGS)
        tool_names = {tool["name"] for tool in CRAFT_TOOLS}
        self.assertTrue(
            {
                "ModelAdvisorTool",
                "ModelPortfolioTool",
                "PerModelSettingsTool",
                "AlbumRenderMatrixTool",
                "FilenamePlannerTool",
                "XLModelPolicyTool",
                "TagLibraryTool",
                "LyricLengthTool",
                "GenerationSettingsTool",
                "ArrangementTool",
                "VocalPerformanceTool",
                "RhymeFlowTool",
                "MetaphorWorldTool",
                "HookDoctorTool",
                "ClicheGuardTool",
                "AlbumArcTool",
                "AlbumContinuityTool",
                "InspirationRadarTool",
                "CaptionPolisherTool",
                "ConflictCheckerTool",
                "MixMasterTool",
                "HitScoreTool",
                "TrackRepairTool",
                "LanguagePresetTool",
                "GenreModuleTool",
                "SectionMapTool",
                "IterationPlanTool",
                "TroubleshootingTool",
                "ValidationChecklistTool",
                "NegativeControlTool",
                "AceStepSettingsPolicyTool",
                "HitReadinessTool",
                "RuntimePlannerTool",
                "TaskApplicabilityTool",
                "ModelCompatibilityTool",
            }.issubset(tool_names)
        )
        self.assertEqual(payload["prompt_kit_version"], PROMPT_KIT_VERSION)
        self.assertIn("prompt_kit", payload)
        self.assertIn("language_presets", payload)
        self.assertIn("genre_modules", payload)
        self.assertIn("validation_checklist", payload)
        self.assertIn("ace_step_settings_registry", payload)
        self.assertEqual(payload["ace_step_settings_registry"]["version"], "ace-step-settings-parity-2026-04-26")

    def test_model_advisor_uses_only_installed_models(self):
        info = choose_song_model({"acestep-v15-turbo"}, "best_installed", "auto")
        self.assertTrue(info["ok"])
        self.assertEqual(info["model"], "acestep-v15-turbo")

        missing = choose_song_model({"acestep-v15-turbo"}, "selected", "acestep-v15-xl-turbo")
        self.assertFalse(missing["ok"])
        self.assertIn("not installed", missing["error"])

        detail = choose_song_model({"acestep-v15-sft", "acestep-v15-turbo"}, "maximum_detail", "auto")
        self.assertEqual(detail["model"], "acestep-v15-sft")

        final_missing = choose_song_model({"acestep-v15-turbo"}, "xl_sft_final", "auto")
        self.assertFalse(final_missing["ok"])
        self.assertEqual(final_missing["model"], ALBUM_FINAL_MODEL)
        self.assertTrue(final_missing["locked_final_model"])

        final_ready = choose_song_model({ALBUM_FINAL_MODEL, "acestep-v15-turbo"}, "xl_sft_final", "auto")
        self.assertTrue(final_ready["ok"])
        self.assertEqual(final_ready["model"], ALBUM_FINAL_MODEL)

    def test_all_models_album_portfolio_has_seven_models(self):
        portfolio = album_model_portfolio({"acestep-v15-turbo"})
        self.assertEqual([item["model"] for item in portfolio], ALBUM_MODEL_PORTFOLIO_MODELS)
        self.assertEqual(len(portfolio), 7)
        self.assertTrue(portfolio[0]["installed"])
        self.assertTrue(portfolio[1]["download_required"])

        resolved = album_models_for_strategy("all_models_album", {"acestep-v15-turbo"})
        self.assertEqual([item["model"] for item in resolved], ALBUM_MODEL_PORTFOLIO_MODELS)

    def test_selected_album_strategy_uses_requested_model_even_when_missing(self):
        resolved = album_models_for_strategy("selected", {"acestep-v15-turbo"}, "acestep-v15-xl-sft")

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["model"], "acestep-v15-xl-sft")
        self.assertFalse(resolved[0]["installed"])
        self.assertTrue(resolved[0]["download_required"])

    def test_artist_references_pass_through_unchanged(self):
        cleaned, notes = sanitize_artist_references("Dutch rap like Nas and Eminem")
        self.assertIn("Nas", cleaned)
        self.assertIn("Eminem", cleaned)
        self.assertEqual(len(notes), 0)

    def test_lyric_length_scales_with_duration(self):
        short = lyric_length_plan(30, "dense", genre_hint="rap")
        long = lyric_length_plan(240, "dense", genre_hint="rap")
        ten_minutes = lyric_length_plan(600, "rap_dense", genre_hint="rap")
        self.assertLess(short["target_words"], long["target_words"])
        self.assertLess(long["target_words"], ten_minutes["target_words"])
        self.assertGreaterEqual(short["min_words"], 45)
        self.assertGreaterEqual(long["min_words"], 430)
        self.assertLessEqual(ten_minutes["max_lyrics_chars"], 4096)
        self.assertGreaterEqual(short["min_lines"], len(short["sections"]))

    def test_sparse_genres_use_sparse_section_timeline(self):
        plan = lyric_length_plan(240, "dense", genre_hint="instrumental techno club track")
        self.assertEqual(plan["density"], "sparse")
        self.assertLess(plan["target_words"], 180)
        self.assertIn("Drop", " ".join(plan["sections"]))

    def test_duration_parser_accepts_minute_second_strings(self):
        self.assertEqual(parse_duration_seconds("2:55"), 175)
        self.assertEqual(parse_duration_seconds("3 min 15 sec"), 195)
        self.assertEqual(parse_duration_seconds(240), 240)

    def test_artist_name_is_original_and_fallback_safe(self):
        self.assertEqual(normalize_artist_name("DJ Test Pilot"), "Dj Test Pilot")
        self.assertEqual(normalize_artist_name("like Nas", "AceJAM"), "AceJAM")
        self.assertNotEqual(derive_artist_name("Neon Win", "melodic rap", "808 bass"), "AceJAM")

    def test_normalize_track_repairs_creative_crewai_fields_and_short_lyrics(self):
        tracks = normalize_album_tracks(
            [
                {
                    "title": "Rivers of Red Ink",
                    "tags": ["glitchy syncopation", "weird rhythm", "neon lights strobe"],
                    "lyrics": "[Verse]\nShort setup\n\n[Chorus]\nTiny hook",
                    "bpm": 120,
                    "key_scale": "E Minor Jelly",
                    "time_signature": "7/8 Intro -> 4/4 Verse",
                    "duration": "2:55",
                    "seed": "neon strobe eviction notices digital sheen",
                    "infer_method": "Euler-Shift3",
                    "sampler_mode": "Weird-Rhythm",
                    "audio_format": "24-bit WAV",
                }
            ],
            {
                "concept": "dense rap album with cinematic hooks",
                "sanitized_concept": "dense rap album with cinematic hooks",
                "num_tracks": 1,
                "track_duration": 180,
                "lyric_density": "dense",
                "structure_preset": "auto",
                "key_strategy": "related",
                "installed_models": [ALBUM_FINAL_MODEL],
                "song_model_strategy": "xl_sft_final",
            },
        )
        track = tracks[0]
        stats = lyric_stats(track["lyrics"])
        self.assertTrue(track["artist_name"])
        self.assertEqual(track["duration"], 175)
        self.assertEqual(track["key_scale"], "E minor")
        self.assertEqual(track["time_signature"], "4")
        self.assertRegex(track["seed"], r"^\d+$")
        self.assertEqual(track["infer_method"], "ode")
        self.assertEqual(track["sampler_mode"], "heun")
        self.assertEqual(track["audio_format"], "wav32")
        self.assertGreaterEqual(stats["word_count"], track["tool_report"]["length_plan"]["min_words"])
        self.assertGreaterEqual(stats["line_count"], track["tool_report"]["length_plan"]["min_lines"])
        self.assertTrue(track["tool_report"]["length_ok"])
        self.assertEqual(track["prompt_kit_version"], PROMPT_KIT_VERSION)
        self.assertIn("section_map", track)
        self.assertIn("negative_control", track)
        self.assertIn("quality_checks", track)
        self.assertFalse(track["use_format"])

    def test_acejam_agent_normalization_does_not_append_generic_filler(self):
        tracks = normalize_album_tracks(
            [
                {
                    "title": "Concrete Canyons",
                    "description": "A heavy atmospheric West Coast hip-hop track with deep bass and sirens.",
                    "tags": "hip-hop, 808 bass, trap hi-hats, male rap vocal, crisp modern mix, melancholic",
                    "lyrics": "[Verse]\nThey paved them blocks just to hide what is real\n\n[Chorus]\nConcrete canyons keep shaking",
                    "duration": 240,
                    "bpm": 78,
                }
            ],
            {
                "concept": (
                    "Album: You Buried the Wrong Man\n"
                    "Track 1: Concrete Canyons\n"
                    "Track 2: Rooftop Letters (Style: melodic house)"
                ),
                "sanitized_concept": (
                    "Album: You Buried the Wrong Man\n"
                    "Track 1: Concrete Canyons\n"
                    "Track 2: Rooftop Letters (Style: melodic house)"
                ),
                "num_tracks": 1,
                "track_duration": 240,
                "lyric_density": "dense",
                "structure_preset": "auto",
                "installed_models": [ALBUM_FINAL_MODEL],
                "song_model_strategy": "xl_sft_final",
                "agent_engine": "acejam_agents",
                "strict_album_agents": True,
                "disable_auto_lyric_expansion": True,
            },
        )

        track = tracks[0]
        self.assertIn("They paved them blocks", track["lyrics"])
        self.assertNotIn("Morning lifts", track["lyrics"])
        self.assertNotIn("pressure into perfume", track["lyrics"])
        self.assertNotIn("A heavy atmospheric", track["tags"])
        self.assertNotEqual(track["tool_report"]["length_plan"]["density"], "sparse")
        self.assertIn(track["payload_gate_status"], {"fail", "review_needed"})

    def test_build_album_plan_returns_duration_ready_tracks(self):
        result = build_album_plan(
            "Dutch luxury rap album with cinematic hooks",
            3,
            120,
            {
                "installed_models": [ALBUM_FINAL_MODEL, "acestep-v15-turbo"],
                "song_model_strategy": "xl_sft_final",
                "lyric_density": "dense",
                "tag_packs": ["genre_style", "production_style"],
                "custom_tags": "male rap vocal, radio ready",
            },
        )
        self.assertEqual(len(result["tracks"]), 3)
        for track in result["tracks"]:
            self.assertEqual(track["song_model"], ALBUM_FINAL_MODEL)
            self.assertEqual(track["inference_steps"], 64)
            self.assertEqual(track["guidance_scale"], 8.0)
            self.assertEqual(track["shift"], 3.0)
            self.assertEqual(track["quality_profile"], "chart_master")
            self.assertIn("production_team", track)
            self.assertIn("studio_engineer", track["production_team"])
            self.assertTrue(track["lyrics"].strip())
            self.assertTrue(track["tool_report"]["length_score"] > 0)
            self.assertIn("bpm", track)
            self.assertEqual(track["prompt_kit_version"], PROMPT_KIT_VERSION)

    def test_editable_plan_preserves_track_model_choice(self):
        result = plan_album(
            "two track pop EP",
            num_tracks=1,
            track_duration=60,
            options={
                "installed_models": ["acestep-v15-turbo", "acestep-v15-sft"],
                "song_model_strategy": "best_installed",
            },
            use_crewai=False,
            input_tracks=[{"title": "Manual Model", "song_model": "acestep-v15-sft"}],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["tracks"][0]["song_model"], "acestep-v15-sft")
        self.assertEqual(result["tracks"][0]["model_advice"]["strategy"], "selected")
        self.assertEqual(result["planning_engine"], "editable_plan")
        self.assertFalse(result["crewai_used"])

    def test_xl_sft_final_overrides_editable_track_model_choice(self):
        result = plan_album(
            "two track pop EP",
            num_tracks=1,
            track_duration=60,
            options={
                "installed_models": [ALBUM_FINAL_MODEL, "acestep-v15-turbo"],
                "song_model_strategy": "xl_sft_final",
            },
            use_crewai=False,
            input_tracks=[{"title": "Manual Model", "song_model": "acestep-v15-turbo"}],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["tracks"][0]["song_model"], ALBUM_FINAL_MODEL)
        self.assertTrue(result["tracks"][0]["production_team"]["final_model_policy"]["locked"])

    def test_crewai_tool_factory_exposes_real_tools_when_available(self):
        tools = make_crewai_tools(
            {
                "installed_models": ["acestep-v15-turbo"],
                "track_duration": 90,
                "song_model_strategy": "best_installed",
            }
        )
        if tools:
            names = {getattr(tool, "name", "") for tool in tools}
            self.assertIn("ModelAdvisorTool", names)
            self.assertIn("ModelPortfolioTool", names)
            self.assertIn("AlbumRenderMatrixTool", names)
            self.assertIn("XLModelPolicyTool", names)
            self.assertIn("GenerationSettingsTool", names)
            self.assertIn("ConflictCheckerTool", names)
            self.assertIn("HitReadinessTool", names)
            self.assertIn("RuntimePlannerTool", names)

    def test_album_plan_fallback_injects_tool_reports(self):
        result = plan_album(
            "club rap with big hooks",
            num_tracks=2,
            track_duration=75,
            language="en",
            options={
                "installed_models": ["acestep-v15-turbo"],
                "song_model_strategy": "best_installed",
                "lyric_density": "balanced",
            },
            use_crewai=False,
        )
        self.assertTrue(result["success"])
        self.assertEqual(len(result["tracks"]), 2)
        self.assertEqual(result["planning_engine"], "toolbelt")
        self.assertFalse(result["crewai_used"])
        self.assertFalse(result["toolbelt_fallback"])
        self.assertIn("toolkit_report", result)
        self.assertIn("tool_report", result["tracks"][0])

    def test_album_plan_streams_monitor_logs(self):
        streamed = []
        long_concept = "short folk album " + ("with gentle safe details " * 40)
        result = plan_album(
            long_concept,
            num_tracks=1,
            track_duration=45,
            language="en",
            options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
            use_crewai=False,
            log_callback=streamed.append,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_engine"], "toolbelt")
        self.assertTrue(streamed[0].startswith("Concept preview: "))
        self.assertLessEqual(len(streamed[0]), 240)
        self.assertIn("Toolbelt fallback planned 1 tracks.", streamed)

    def test_acejam_agents_default_plans_with_direct_json_calls(self):
        required = ["Open the shutters", "Coffee on the corner", "market bell"]
        lyrics = self._valid_album_test_lyrics(required)
        bible_payload = {
            "album_bible": {"album_title": "Market Lights", "concept": "safe", "arc": "repair", "motifs": ["market bell"]},
            "tracks": [{
                "track_number": 1,
                "title": "Generated Rename",
                "producer_credit": "Wrong Producer",
                "description": "wrong",
                "tags": "modern pop, steady groove, bass, brass, drums, clear lead vocal, hopeful warm atmosphere, dynamic hook arrangement, polished studio mix",
                "tag_list": ["modern pop", "steady groove", "bass", "brass", "drums", "clear lead vocal", "hopeful warm atmosphere", "dynamic hook arrangement", "polished studio mix"],
                "duration": 45,
                "bpm": 120,
            }],
        }
        writer_payload = {
            "description": "neighbors reopen stalls and trade stories instead of rumors.",
            "tags": bible_payload["tracks"][0]["tags"],
            "tag_list": bible_payload["tracks"][0]["tag_list"],
            "lyrics": lyrics,
            "hook_promise": "Open the shutters into a brighter block.",
        }

        def fake_agent_json_call(*, agent_name, **_kwargs):
            if agent_name == "Album Bible Agent":
                return bible_payload
            if agent_name == "Track Blueprint Agent":
                return bible_payload["tracks"][0]
            if agent_name == "Track Writer Agent":
                return writer_payload
            if agent_name == "Track Finalizer Agent":
                return {**writer_payload, "title": "Another Rename", "bpm": 140}
            raise AssertionError(agent_name)

        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "embedding_model": "embed-local",
            "warnings": [],
            "errors": [],
        }), patch.object(album_crew_module, "local_llm_embed", return_value=[1.0, 0.0, 0.0]), patch.object(album_crew_module, "_agent_json_call", side_effect=fake_agent_json_call):
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
                planner_provider="ollama",
                embedding_provider="ollama",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_engine"], "acejam_agents")
        self.assertTrue(result["custom_agents_used"])
        self.assertFalse(result["crewai_used"])
        self.assertFalse(result["toolbelt_fallback"])
        self.assertEqual(result["tracks"][0]["title"], "Morning Market")
        self.assertEqual(result["tracks"][0]["producer_credit"], "Ada North")
        self.assertEqual(result["tracks"][0]["bpm"], 92)
        self.assertIn("Open the shutters", result["tracks"][0]["lyrics"])
        self.assertIn(result["tracks"][0]["payload_gate_status"], {"pass", "auto_repair"})
        self.assertTrue(result["memory_enabled"])
        self.assertGreaterEqual(result["context_chunks"], 1)
        self.assertTrue(any("AceJAM Agents produced" in line for line in result["logs"]))

    def test_acejam_agents_continue_when_album_bible_agent_fails(self):
        lyrics = self._valid_album_test_lyrics(["Open the shutters"], lines_per_section=8)

        def fake_agent_json_call(*, agent_name, **_kwargs):
            if agent_name == "Album Bible Agent":
                raise album_crew_module.AceJamAgentError("empty bible response")
            if agent_name == "Track Blueprint Agent":
                return {
                    "track_number": 1,
                    "title": "Wrong Rename",
                    "producer_credit": "Wrong Producer",
                    "bpm": 120,
                    "duration": 180,
                    "key_scale": "C major",
                    "style": "modern pop",
                    "vibe": "bright rebuild energy",
                    "narrative": "neighbors rebuild trust",
                    "description": "neighbors rebuild trust",
                    "tag_list": ["modern pop", "steady groove", "bass", "bright drums", "clear lead vocal", "emotional atmosphere", "dynamic hook arrangement", "polished studio mix"],
                    "tags": "modern pop, steady groove, bass, bright drums, clear lead vocal, emotional atmosphere, dynamic hook arrangement, polished studio mix",
                    "hook_promise": "Open the shutters.",
                }
            if agent_name in {"Track Writer Agent", "Track Finalizer Agent"}:
                return {
                    "description": "neighbors reopen stalls and trade stories instead of rumors.",
                    "tags": "modern pop, steady groove, bass, bright drums, clear lead vocal, emotional atmosphere, dynamic hook arrangement, polished studio mix",
                    "tag_list": ["modern pop", "steady groove", "bass", "bright drums", "clear lead vocal", "emotional atmosphere", "dynamic hook arrangement", "polished studio mix"],
                    "lyrics": lyrics,
                    "hook_promise": "Open the shutters into a brighter block.",
                }
            raise AssertionError(agent_name)

        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "embedding_model": "embed-local",
            "warnings": [],
            "errors": [],
        }), patch.object(album_crew_module, "local_llm_embed", return_value=[1.0, 0.0, 0.0]), patch.object(album_crew_module, "_agent_json_call", side_effect=fake_agent_json_call):
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
                planner_provider="ollama",
                embedding_provider="ollama",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["tracks"][0]["title"], "Morning Market")
        self.assertEqual(result["tracks"][0]["duration"], 45)
        self.assertEqual(result["tracks"][0]["producer_credit"], "Ada North")
        self.assertIn("empty bible response", result["album_bible_agent_error"])
        self.assertTrue(any("Album Bible Agent failed explicitly" in line for line in result["logs"]))
        self.assertTrue(any("Ignored blueprint duration hint" in line for line in result["logs"]))

    def test_acejam_agents_scaffold_fills_missing_tracks_from_short_bible(self):
        lyrics = self._valid_album_test_lyrics(["Open the shutters"], lines_per_section=8)
        prompt = """
Album: Market Lights
Concept: A safe city album about rebuilding trust after a long blackout.
Track 1: "Morning Market" (Produced by Ada North)
(BPM: 92 | Key: A minor | Style: warm boom-bap)
The Vibe: brass stabs, vinyl dust, calm crowd energy.
The Narrative: neighbors reopen stalls and trade stories instead of rumors.

Track 2: "Rooftop Letters" (Produced by Mira South)
(BPM: 104 | Style: melodic house)
The Vibe: soft synth pulse and bright handclaps.
The Narrative: friends read old letters on a roof as the lights come back.
"""
        bible_payload = {
            "album_bible": {"album_title": "Market Lights", "concept": "safe city recovery", "arc": "five chapters"},
            "tracks": [
                {"track_number": 1, "title": "Morning Market", "tags": "modern pop, steady groove, bass, brass, drums, clear lead vocal, hopeful warm atmosphere, dynamic hook arrangement, polished studio mix"},
                {"track_number": 2, "title": "Rooftop Letters", "tags": "modern pop, steady groove, synth, handclap drums, clear lead vocal, hopeful bright atmosphere, dynamic hook arrangement, polished studio mix"},
            ],
        }

        blueprint_calls = {"count": 0}

        def fake_agent_json_call(*, agent_name, user_prompt="", **_kwargs):
            if agent_name == "Album Bible Agent":
                return bible_payload
            if agent_name == "Track Blueprint Agent":
                blueprint_calls["count"] += 1
                number = blueprint_calls["count"]
                return {
                    "track_number": number,
                    "title": f"Generated Track {number}",
                    "producer_credit": f"Producer {number}",
                    "bpm": 95 + number,
                    "key_scale": "A minor",
                    "style": "modern pop",
                    "vibe": "bright rebuild energy",
                    "narrative": f"chapter {number} of the blackout recovery",
                    "description": f"chapter {number} of the blackout recovery",
                    "tag_list": ["modern pop", "steady groove", "bass", "bright drums", "clear lead vocal", "emotional atmosphere", "dynamic hook arrangement", "polished studio mix"],
                    "tags": "modern pop, steady groove, bass, bright drums, clear lead vocal, emotional atmosphere, dynamic hook arrangement, polished studio mix",
                    "hook_promise": f"Track {number} brings the lights back.",
                }
            if agent_name == "Track Writer Agent":
                is_house = "Rooftop Letters" in user_prompt or "melodic house" in user_prompt
                track_lyrics = self._valid_album_test_lyrics([], sections=["Intro", "Build", "Drop", "Chorus"], lines_per_section=2) if is_house else lyrics
                track_tags = "modern house, steady groove, synth, handclap drums, clear vocal chops, bright atmosphere, dynamic drop arrangement, polished studio mix" if is_house else "modern pop, steady groove, bass, bright drums, clear lead vocal, emotional atmosphere, dynamic hook arrangement, polished studio mix"
                return {
                    "description": "neighbors rebuild trust with a clear chorus.",
                    "tags": track_tags,
                    "tag_list": track_tags.split(", "),
                    "lyrics": track_lyrics,
                    "hook_promise": "Bring the lights back.",
                }
            if agent_name == "Track Finalizer Agent":
                is_house = "Rooftop Letters" in user_prompt or "melodic house" in user_prompt
                track_lyrics = self._valid_album_test_lyrics([], sections=["Intro", "Build", "Drop", "Chorus"], lines_per_section=2) if is_house else lyrics
                track_tags = "modern house, steady groove, synth, handclap drums, clear vocal chops, bright atmosphere, dynamic drop arrangement, polished studio mix" if is_house else "modern pop, steady groove, bass, bright drums, clear lead vocal, emotional atmosphere, dynamic hook arrangement, polished studio mix"
                return {
                    "description": "neighbors rebuild trust with a clear chorus.",
                    "tags": track_tags,
                    "tag_list": track_tags.split(", "),
                    "lyrics": track_lyrics,
                    "hook_promise": "Bring the lights back.",
                }
            raise AssertionError(agent_name)

        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "embedding_model": "embed-local",
            "warnings": [],
            "errors": [],
        }), patch.object(album_crew_module, "local_llm_embed", return_value=[1.0, 0.0, 0.0]), patch.object(album_crew_module, "_agent_json_call", side_effect=fake_agent_json_call):
            result = plan_album(
                prompt,
                num_tracks=5,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
                planner_provider="ollama",
                embedding_provider="ollama",
            )

        self.assertTrue(result["success"])
        self.assertEqual(len(result["tracks"]), 5)
        self.assertEqual(result["tracks"][0]["title"], "Morning Market")
        self.assertEqual(result["tracks"][1]["title"], "Rooftop Letters")
        self.assertTrue(all(int(track.get("duration") or 0) == 45 for track in result["tracks"]))
        self.assertIn("Bible returned 2 optional blueprint hint(s); scaffold requires 5.", result["logs"])
        self.assertTrue(all(track.get("payload_gate_status") in {"pass", "auto_repair"} for track in result["tracks"]))
        self.assertTrue(result["memory_enabled"])
        self.assertGreaterEqual(result["retrieval_rounds"], 1)

    def test_acejam_agent_prompt_contains_full_contract_template_and_counter(self):
        template = render_track_prompt_template(
            user_album_contract={"album_title": "Market Lights"},
            ace_step_payload_contract={"version": "unit"},
            lyric_length_plan=lyric_length_plan(45, "dense", "auto", "warm boom-bap"),
            language_preset={"code": "en"},
            blueprint={"track_number": 1, "title": "Morning Market"},
            album_bible={"concept": "safe"},
        )
        prompt = album_crew_module._track_writer_prompt(
            concept=self.SAFE_CONTRACT_PROMPT,
            album_bible={"concept": "safe"},
            blueprint={"track_number": 1, "title": "Morning Market"},
            previous_summaries=[],
            track_prompt_template=template,
            index=0,
            total=2,
        )

        self.assertIn("FULL_ORIGINAL_ALBUM_PROMPT", prompt)
        self.assertIn("Track 1: \"Morning Market\"", prompt)
        self.assertIn("TRACK COUNTER: you are writing track 1 of 2", prompt)
        self.assertIn("FULL_TAG_LIBRARY", prompt)
        self.assertIn("LYRIC_LENGTH_PLAN", prompt)

    def test_acejam_agents_empty_response_fails_loudly_without_toolbelt(self):
        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": False,
            "embedding_model": "embed-local",
            "warnings": [],
            "errors": [],
        }), patch.object(album_crew_module, "_agent_json_call", side_effect=album_crew_module.AceJamAgentError("empty response")):
            result = plan_album(
                "safe city recovery album",
                num_tracks=1,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
                planner_provider="ollama",
                embedding_provider="ollama",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["planning_engine"], "acejam_agents")
        self.assertTrue(result["custom_agents_used"])
        self.assertFalse(result["crewai_used"])
        self.assertFalse(result["toolbelt_fallback"])
        self.assertEqual(result["tracks"], [])
        self.assertTrue(any("deterministic toolbelt fallback was not used" in line for line in result["logs"]))

    def test_agent_json_parser_repairs_raw_newlines_inside_lyrics_string(self):
        raw = (
            '{\n'
            '  "title": "Lanterns on the Pier",\n'
            '  "lyrics": "[Intro]\nWarm piano enters\n[Chorus]\nLanterns on the pier keep calling us home.",\n'
            '  "tags": "cinematic pop-rap, boom-bap drums, clear male rap vocal"\n'
            '}'
        )

        parsed = _json_object_from_text(raw)

        self.assertEqual(parsed["title"], "Lanterns on the Pier")
        self.assertIn("[Chorus]", parsed["lyrics"])
        self.assertIn("calling us home", parsed["lyrics"])

    def test_agent_json_call_uses_plain_ollama_transport_by_default(self):
        calls = []

        def fake_call_agent_llm(**kwargs):
            calls.append(kwargs.get("json_format"))
            return '{"ok": true}'

        with patch.object(album_crew_module, "_call_agent_llm", side_effect=fake_call_agent_llm):
            payload = album_crew_module._agent_json_call(
                agent_name="Album Bible Agent",
                provider="ollama",
                model_name="qwen-local",
                user_prompt="Return {\"ok\": true}",
                logs=[],
                debug_options={},
                schema_name="unit",
                max_retries=0,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(calls, [False])

    def test_agent_payload_lyrics_lines_are_joined(self):
        payload = _coerce_agent_lyrics_payload({
            "title": "Lanterns on the Pier",
            "lyrics_lines": ["[Intro]", "Warm piano enters", "[Chorus]", "Lanterns call us home"],
        })

        self.assertEqual(payload["lyrics"], "[Intro]\nWarm piano enters\n[Chorus]\nLanterns call us home")
        self.assertEqual(payload["lyrics_lines"][2], "[Chorus]")

    def test_vocal_album_track_clears_spurious_instrumental_flag(self):
        tracks = normalize_album_tracks([
            {
                "title": "Lanterns on the Pier",
                "description": "coastal rebuild anthem",
                "tags": "cinematic pop-rap, boom-bap drums, warm piano, male rap vocal, hopeful, dynamic hook, polished studio mix",
                "lyrics": "[Verse]\nWe relight the harbor\n[Chorus]\nLanterns on the pier keep calling us home",
                "instrumental": True,
                "section_map": [{"tag": "[Build]"}, {"tag": "[Drop]"}],
                "duration": 60,
            }
        ], {
            "agent_engine": "acejam_agents",
            "strict_album_agents": True,
            "installed_models": ["acestep-v15-xl-sft"],
            "song_model_strategy": "xl_sft_final",
            "track_duration": 60,
            "lyric_density": "dense",
            "structure_preset": "auto",
        })

        self.assertFalse(tracks[0]["instrumental"])
        self.assertEqual(tracks[0]["production_team"]["prompt_kit"]["section_map"][1]["tag"], "[Verse]")

    def test_crewai_bible_parse_repair_still_uses_track_crew(self):
        class FakeCrew:
            def __init__(self, output):
                self.output = output

            def kickoff(self):
                return self.output

        class FakeMemory:
            def remember(self, *args, **kwargs):
                return None

        track_payload = {
            "track_number": 1,
            "artist_name": "Harbor Lights",
            "title": "Morning Market",
            "description": "safe hopeful opener",
            "tags": "hopeful pop, acoustic guitar",
            "lyrics": "[Verse]\nWe clean the windows\n[Chorus]\nMorning market shines\n[Outro]\nHome again",
            "bpm": 90,
            "key_scale": "C major",
            "time_signature": "4",
            "language": "en",
            "duration": 45,
        }

        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "warnings": [],
            "embedding_model": "embed-local",
            "memory_dir": str(CREWAI_MEMORY_DIR),
        }), patch.object(album_crew_module, "_make_album_memory_writer", return_value=FakeMemory()), \
            patch.object(album_crew_module, "create_album_bible_crew", return_value=FakeCrew(SimpleNamespace(raw='{"album_bible": {"concept": "safe"}, "tracks": ['))), \
            patch.object(album_crew_module, "create_track_production_crew", return_value=FakeCrew(SimpleNamespace(raw=json.dumps(track_payload)))):
            result = plan_album(
                "safe city recovery album",
                num_tracks=1,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed", "agent_engine": "legacy_crewai"},
                use_crewai=True,
                planner_provider="lmstudio",
                embedding_provider="lmstudio",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_engine"], "crewai")
        self.assertTrue(result["crewai_used"])
        self.assertEqual(len(result["tracks"]), 1)
        self.assertTrue(any("CrewAI bible JSON parse repair" in line for line in result["logs"]))

    def test_crewai_contract_repairs_agent_renames(self):
        class FakeCrew:
            def __init__(self, output):
                self.output = output

            def kickoff(self):
                return self.output

        class FakeMemory:
            def remember(self, *args, **kwargs):
                return None

        contract_prompt = """
Album: Exact Brief
Concept: A safe two-song record about neighborhood repair.
Track 1: "Lantern Keys" (Produced by Ada North)
(BPM: 88 | Style: warm boom-bap)
The Vibe: dusty piano and soft brass.
The Narrative: a locksmith helps everyone back into their apartments.
Lyrics:
"Turn the lantern keys"
"""
        bible_payload = {
            "album_bible": {"concept": "safe", "arc": "one", "motifs": []},
            "tracks": [{
                "track_number": 1,
                "artist_name": "Wrong Artist",
                "title": "Generated Season",
                "description": "wrong rename",
                "tags": "hopeful pop",
                "duration": 45,
                "bpm": 120,
            }],
        }
        produced_payload = {
            "track_number": 1,
            "artist_name": "Wrong Artist",
            "title": "Another Rename",
            "description": "wrong final",
            "tags": "hopeful pop",
            "lyrics": "[Verse]\nWe find the room\n[Chorus]\nHome again",
            "bpm": 120,
            "key_scale": "C major",
            "time_signature": "4",
            "language": "en",
            "duration": 45,
        }

        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "warnings": [],
            "embedding_model": "embed-local",
            "memory_dir": str(CREWAI_MEMORY_DIR),
        }), patch.object(album_crew_module, "_make_album_memory_writer", return_value=FakeMemory()), \
            patch.object(album_crew_module, "create_album_bible_crew", return_value=FakeCrew(SimpleNamespace(raw=json.dumps(bible_payload)))), \
            patch.object(album_crew_module, "create_track_production_crew", return_value=FakeCrew(SimpleNamespace(raw=json.dumps(produced_payload)))):
            result = plan_album(
                contract_prompt,
                num_tracks=1,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed", "agent_engine": "legacy_crewai"},
                use_crewai=True,
                planner_provider="lmstudio",
                embedding_provider="lmstudio",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_engine"], "crewai")
        self.assertEqual(result["tracks"][0]["title"], "Lantern Keys")
        self.assertEqual(result["tracks"][0]["producer_credit"], "Ada North")
        self.assertEqual(result["tracks"][0]["bpm"], 88)
        self.assertIn("Turn the lantern keys", result["tracks"][0]["lyrics"])
        self.assertTrue(result["input_contract_applied"])
        self.assertGreaterEqual(result["contract_repair_count"], 1)
        self.assertTrue(any("Contract repaired: track 1" in line for line in result["logs"]))

    def test_crewai_blueprint_duration_is_clamped_to_requested_track_duration(self):
        class FakeCrew:
            def __init__(self, output):
                self.output = output

            def kickoff(self):
                return self.output

        class FakeMemory:
            def remember(self, *args, **kwargs):
                return None

        captured_blueprints = []
        bible_payload = {
            "album_bible": {"concept": "safe", "arc": "one", "motifs": []},
            "tracks": [{
                "track_number": 1,
                "artist_name": "Harbor Lights",
                "title": "Morning Market",
                "description": "safe hopeful opener",
                "tags": "hopeful pop",
                "duration": 195,
            }],
        }

        def fake_track_crew(album_bible, blueprint, *args, **kwargs):
            captured_blueprints.append(dict(blueprint))
            produced = {
                **blueprint,
                "lyrics": "[Verse]\nClean windows shine\n[Chorus]\nHome again\n[Outro]\nMorning stays",
                "language": "en",
            }
            return FakeCrew(SimpleNamespace(raw=json.dumps(produced)))

        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "warnings": [],
            "embedding_model": "embed-local",
            "memory_dir": str(CREWAI_MEMORY_DIR),
        }), patch.object(album_crew_module, "_make_album_memory_writer", return_value=FakeMemory()), \
            patch.object(album_crew_module, "create_album_bible_crew", return_value=FakeCrew(SimpleNamespace(raw=json.dumps(bible_payload)))), \
            patch.object(album_crew_module, "create_track_production_crew", side_effect=fake_track_crew):
            result = plan_album(
                "safe city recovery album",
                num_tracks=1,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed", "agent_engine": "legacy_crewai"},
                use_crewai=True,
                planner_provider="lmstudio",
                embedding_provider="lmstudio",
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured_blueprints[0]["duration"], 45)
        self.assertEqual(result["tracks"][0]["duration"], 45.0)

    def test_track_json_parse_repair_does_not_force_album_toolbelt_fallback(self):
        class FakeCrew:
            def __init__(self, output):
                self.output = output

            def kickoff(self):
                return self.output

        class FakeMemory:
            def remember(self, *args, **kwargs):
                return None

        bible_payload = {
            "album_bible": {"concept": "safe", "arc": "one", "motifs": []},
            "tracks": [{
                "track_number": 1,
                "artist_name": "Harbor Lights",
                "title": "Repair Crews",
                "description": "safe repair scene",
                "tags": "hopeful acoustic pop",
                "duration": 20,
            }],
        }
        malformed_track = SimpleNamespace(
            raw="No JSON here, only notes.",
            tasks_output=[
                SimpleNamespace(raw="[Verse]\nWe fix the roof\n[Chorus]\nHome again\n[Outro]\nLights on"),
                SimpleNamespace(raw="No JSON here, only notes."),
            ],
        )

        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "warnings": [],
            "embedding_model": "embed-local",
            "memory_dir": str(CREWAI_MEMORY_DIR),
        }), patch.object(album_crew_module, "_make_album_memory_writer", return_value=FakeMemory()), \
            patch.object(album_crew_module, "create_album_bible_crew", return_value=FakeCrew(SimpleNamespace(raw=json.dumps(bible_payload)))), \
            patch.object(album_crew_module, "create_track_production_crew", return_value=FakeCrew(malformed_track)):
            result = plan_album(
                "safe city recovery album",
                num_tracks=1,
                track_duration=20,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed", "agent_engine": "legacy_crewai"},
                use_crewai=True,
                planner_provider="lmstudio",
                embedding_provider="lmstudio",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_engine"], "crewai")
        self.assertFalse(result["toolbelt_fallback"])
        self.assertTrue(any("CrewAI JSON repair used production text" in line for line in result["logs"]))
        self.assertIn("[Verse]", result["tracks"][0]["lyrics"])

    def test_crewai_verbose_default_enabled(self):
        self.assertTrue(CREWAI_VERBOSE)

    def test_crewai_log_path_is_job_scoped_and_sanitized(self):
        path = crewai_output_log_path("job/with spaces")

        self.assertEqual(path.parent.name, "crewai_logs")
        self.assertEqual(path.name, "album_plan_job_with_spaces.json")

    def test_crewai_kickoff_captures_verbose_stdout_without_prompt_log(self):
        class FakeCrew:
            def kickoff(self):
                print("SECRET FULL TASK PROMPT")
                return SimpleNamespace(raw="ok")

        logs = []
        result = _kickoff_crewai_compact(FakeCrew(), logs, "unit crew", "/tmp/crewai.json")

        self.assertEqual(result.raw, "ok")
        self.assertTrue(any("CrewAI verbose captured for unit crew" in line for line in logs))
        self.assertFalse(any("SECRET FULL TASK PROMPT" in line for line in logs))
        self.assertTrue(any("/tmp/crewai.json" in line for line in logs))

    def test_crewai_task_callback_redacts_lyrics_preview(self):
        logs = []
        callback = _crewai_task_callback(logs)

        callback(SimpleNamespace(agent="Track JSON Finalizer", raw=json.dumps({
            "track_number": 1,
            "title": "Safe Title",
            "duration": 45,
            "lyrics": "[Verse]\nSECRET LINE",
        })))

        self.assertEqual(len(logs), 1)
        self.assertIn("title=Safe Title", logs[0])
        self.assertIn("lyrics=redacted", logs[0])
        self.assertNotIn("SECRET LINE", logs[0])

    def test_json_object_parser_uses_last_valid_object_after_reasoning(self):
        raw = (
            "Thinking through the schema.\n"
            "```json\n{\"draft\": true}\n```\n"
            "Final answer follows.\n"
            "{\"track_number\": 5, \"title\": \"Repair Crews\", \"duration\": 20}"
        )

        parsed = _json_object_from_text(raw)

        self.assertEqual(parsed["track_number"], 5)
        self.assertEqual(parsed["title"], "Repair Crews")

    def test_normalize_album_tracks_treats_none_strings_as_defaults(self):
        tracks = normalize_album_tracks([
            {
                "title": "Repair Crews",
                "description": "hopeful repair song",
                "lyrics": "[Verse]\nWe fix the roof\n[Chorus]\nHome again\n[Outro]\nLights on",
                "duration": "20s",
                "bpm": "None",
                "inference_steps": "C",
                "guidance_scale": "not numeric",
                "shift": "None",
                "seed": "None",
            }
        ], {
            "installed_models": ["acestep-v15-turbo"],
            "song_model_strategy": "best_installed",
            "track_duration": 20,
            "lyric_density": "balanced",
        })

        self.assertEqual(len(tracks), 1)
        self.assertIsInstance(tracks[0]["bpm"], int)
        self.assertIsInstance(tracks[0]["inference_steps"], int)
        self.assertIsInstance(tracks[0]["guidance_scale"], float)
        self.assertIsInstance(tracks[0]["shift"], float)
        self.assertEqual(tracks[0]["seed"], "-1")

    def test_normalize_album_tracks_repairs_string_schema_fields(self):
        tracks = normalize_album_tracks([
            {
                "title": "Schema Repair",
                "description": "upbeat city-pop repair song",
                "tags": "pop, steady groove, piano, clear lead vocal, uplifting mood, dynamic hook arrangement, polished studio mix",
                "lyrics": (
                    "[Intro]\nLights are coming back\n"
                    "[Verse]\nNeighbors gather by the bakery door\nFresh warm bread gives everybody more\n"
                    "[Chorus]\nWe sing the lights back on tonight\nWe sing the lights back on tonight\n"
                    "[Verse 2]\nClean drums move under every line\n"
                    "[Final Chorus]\nWe sing the lights back on tonight"
                ),
                "duration": 45,
                "quality_checks": "quality_checks",
                "contract_compliance": "contract_compliance",
                "settings_compliance": "settings_compliance",
                "genre_profile": "genre_profile",
                "section_map": "section_map",
                "iteration_plan": "iteration_plan",
                "community_risk_notes": "community_risk_notes",
                "troubleshooting_hints": "troubleshooting_hints",
                "variations": "variations",
                "negative_control": "negative_control",
                "tag_coverage": "tag_coverage",
                "lyric_duration_fit": "lyric_duration_fit",
                "caption_integrity": "caption_integrity",
                "repair_actions": "repair_actions",
                "prompt_kit_version": "bad-version",
                "auto_score": True,
                "auto_lrc": True,
                "return_audio_codes": True,
            }
        ], {
            "installed_models": ["acestep-v15-turbo"],
            "song_model_strategy": "best_installed",
            "track_duration": 45,
            "lyric_density": "balanced",
        })

        self.assertEqual(len(tracks), 1)
        self.assertIsInstance(tracks[0]["quality_checks"], dict)
        self.assertIsInstance(tracks[0]["contract_compliance"], dict)
        self.assertIsInstance(tracks[0]["settings_compliance"], dict)
        self.assertIsInstance(tracks[0]["genre_profile"], dict)
        self.assertIsInstance(tracks[0]["section_map"], list)
        self.assertIsInstance(tracks[0]["iteration_plan"], list)
        self.assertIsInstance(tracks[0]["community_risk_notes"], list)
        self.assertIsInstance(tracks[0]["troubleshooting_hints"], list)
        self.assertIsInstance(tracks[0]["variations"], list)
        self.assertIsInstance(tracks[0]["negative_control"], list)
        self.assertIsInstance(tracks[0]["tag_coverage"], dict)
        self.assertIsInstance(tracks[0]["repair_actions"], list)
        self.assertEqual(tracks[0]["prompt_kit_version"], PROMPT_KIT_VERSION)
        self.assertFalse(tracks[0]["auto_score"])
        self.assertFalse(tracks[0]["auto_lrc"])
        self.assertFalse(tracks[0]["return_audio_codes"])

    def test_crewai_lyric_extractor_strips_metadata_tail(self):
        raw = (
            "Thought: done\nFinal Answer:\n```\n"
            "**[Intro]**\nThe lights come back\n[Verse]\nThe bakery opens wide\n**[Chorus]**\nWarm bread, bright street\n[Outro]\nGood night\n"
            "```\n**ACE-Step Metadata:**\n- **Song Model:** acestep-v15-turbo\nmetadata:\nbpm: 95\nkey_scale: A minor\n"
        )

        lyrics = _lyric_like_text(raw)

        self.assertIn("[Intro]", lyrics)
        self.assertIn("[Chorus]", lyrics)
        self.assertNotIn("**[Chorus]**", lyrics)
        self.assertNotIn("ACE-Step Metadata", lyrics)
        self.assertNotIn("metadata:", lyrics)
        self.assertNotIn("Song Model:", lyrics)
        self.assertNotIn("bpm:", lyrics)

    def test_crewai_tool_filter_keeps_only_allowed_tools(self):
        class DummyTool:
            def __init__(self, name, description=""):
                self.name = name
                self.description = description

        tools = [
            DummyTool("LyricLengthTool", "Tool Name: lyric_length_tool\nTool Description: ok"),
            DummyTool("AceStepCoverageAuditTool", "Tool Name: ace_step_coverage_audit_tool\nTool Description: huge"),
            DummyTool("HookDoctorTool", "Tool Name: hook_doctor_tool\nTool Description: ok"),
        ]

        selected = _select_crewai_tools(tools, {"lyric_length_tool", "hook_doctor_tool"})

        self.assertEqual([tool.name for tool in selected], ["LyricLengthTool", "HookDoctorTool"])

    def test_album_genre_hint_ignores_producer_credit_house(self):
        hint = _album_genre_hint({
            "sanitized_concept": 'Track 1: "Neon Bakery Lights" (Produced by Studio House)',
            "user_album_contract": {
                "tracks": [{
                    "style": "upbeat city-pop / pop-funk",
                    "vibe": "rubber bass, bright piano stabs",
                    "narrative": "neighbors gather after a blackout",
                }]
            },
        })

        self.assertIn("city-pop", hint)
        self.assertIn("pop-funk", hint)
        self.assertNotIn("Studio House", hint)

    def test_fallback_lyric_expansion_avoids_stopword_subject_artifacts(self):
        lyrics = expand_lyrics_for_duration(
            title="The You",
            concept="A hopeful pop song about repair",
            lyrics="",
            duration=60,
            language="en",
            density="balanced",
            structure_preset="auto",
        )

        lowered = lyrics.lower()
        self.assertNotIn("morning finds the", lowered)
        self.assertNotIn("the you", lowered)
        self.assertNotIn("the was", lowered)
        self.assertIn("morning lifts", lowered)

    def test_album_crew_uses_ollama_llm_without_native_tool_parser(self):
        kwargs = _crewai_llm_kwargs("qwen3.6:27b-instruct-general")
        llm = _make_llm("qwen3.6:27b-instruct-general")

        self.assertNotIn("context_window_size", kwargs)
        self.assertEqual(llm.provider, "ollama")
        self.assertEqual(llm.model, "qwen3.6:27b-instruct-general")
        self.assertEqual(llm.timeout, CREWAI_LLM_TIMEOUT_SECONDS)
        self.assertEqual(llm.max_tokens, CREWAI_LLM_MAX_TOKENS)
        options = llm.additional_params["extra_body"]["options"]
        self.assertEqual(options["num_ctx"], CREWAI_LLM_CONTEXT_WINDOW)
        self.assertEqual(options["num_predict"], CREWAI_LLM_NUM_PREDICT)
        params = llm._prepare_completion_params([{"role": "user", "content": "Reply OK."}], tools=None)
        self.assertNotIn("num_ctx", params)
        self.assertNotIn("num_predict", params)
        self.assertNotIn("context_window_size", params)
        self.assertEqual(params["extra_body"]["options"]["num_ctx"], CREWAI_LLM_CONTEXT_WINDOW)
        self.assertFalse(llm.supports_function_calling())

    def test_album_crew_uses_openai_compatible_lmstudio_payload(self):
        kwargs = _crewai_llm_kwargs("qwen-local", "lmstudio")
        llm = _make_llm("qwen-local", "lmstudio")

        self.assertNotIn("num_ctx", kwargs)
        self.assertNotIn("num_predict", kwargs)
        self.assertNotIn("context_window_size", kwargs)
        self.assertNotIn("additional_params", kwargs)
        self.assertEqual(llm.provider, "openai")
        self.assertEqual(llm.model, "qwen-local")
        self.assertTrue(str(llm.base_url).rstrip("/").endswith("/v1"))
        self.assertEqual(llm.timeout, CREWAI_LLM_TIMEOUT_SECONDS)
        self.assertEqual(llm.max_tokens, CREWAI_LMSTUDIO_MAX_TOKENS)
        self.assertEqual(llm.additional_params, {})
        params = llm._prepare_completion_params([{"role": "user", "content": "Reply OK."}], tools=None)
        self.assertNotIn("num_ctx", params)
        self.assertNotIn("num_predict", params)
        self.assertNotIn("context_window_size", params)
        self.assertNotIn("extra_body", params)
        self.assertEqual(params["model"], "qwen-local")
        self.assertEqual(params["max_tokens"], CREWAI_LMSTUDIO_MAX_TOKENS)

    def test_lmstudio_crewai_kwargs_do_not_prefix_model(self):
        kwargs = _crewai_llm_kwargs("huihui-qwen-local", "lmstudio")

        self.assertEqual(kwargs["model"], "huihui-qwen-local")
        self.assertEqual(kwargs["provider"], "openai")
        self.assertNotIn("api_base", kwargs)
        self.assertNotIn("additional_params", kwargs)
        self.assertNotIn("context_window_size", kwargs)

    def test_lmstudio_crews_disable_native_json_schema(self):
        opts = {"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed", "lyric_density": "balanced"}
        bible_crew = create_album_bible_crew(
            "safe city rebuild album",
            1,
            60,
            "qwen-local",
            "en",
            "embed-local",
            opts,
            "lmstudio",
            "lmstudio",
        )
        track_crew = create_track_production_crew(
            {"concept": "safe city rebuild album", "arc": "recovery", "motifs": ["lights"]},
            {"track_number": 1, "title": "Lights Return", "description": "hopeful opener"},
            1,
            60,
            "qwen-local",
            "en",
            "embed-local",
            opts,
            "lmstudio",
            "lmstudio",
        )

        self.assertIsNone(bible_crew.tasks[-1].output_json)
        self.assertIsNone(track_crew.tasks[-1].output_json)

    def test_compact_crews_wire_verbose_callbacks_and_output_log_file(self):
        opts = {"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed", "lyric_density": "balanced"}
        step_callback = lambda *_args, **_kwargs: None
        task_callback = lambda *_args, **_kwargs: None
        log_file = "/tmp/acejam_compact_crewai_test.json"
        bible_crew = create_album_bible_crew(
            "safe city rebuild album",
            1,
            60,
            "qwen-local",
            "en",
            "embed-local",
            opts,
            "lmstudio",
            "lmstudio",
            step_callback=step_callback,
            task_callback=task_callback,
            output_log_file=log_file,
        )
        track_crew = create_track_production_crew(
            {"concept": "safe city rebuild album", "arc": "recovery", "motifs": ["lights"]},
            {"track_number": 1, "title": "Lights Return", "description": "hopeful opener"},
            1,
            60,
            "qwen-local",
            "en",
            "embed-local",
            opts,
            "lmstudio",
            "lmstudio",
            step_callback=step_callback,
            task_callback=task_callback,
            output_log_file=log_file,
        )

        for crew in (bible_crew, track_crew):
            self.assertTrue(crew.verbose)
            self.assertIs(crew.step_callback, step_callback)
            self.assertIs(crew.task_callback, task_callback)
            self.assertEqual(crew.output_log_file, log_file)
            for agent in crew.agents:
                self.assertTrue(agent.verbose)
                self.assertIs(agent.step_callback, step_callback)

    def test_lmstudio_no_think_is_injected_once(self):
        messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Return JSON."}]

        updated = _lmstudio_no_think_messages(messages)
        again = _lmstudio_no_think_messages(updated)
        args, kwargs = _lmstudio_no_think_args((), {"messages": messages}, "lmstudio")
        other_args, other_kwargs = _lmstudio_no_think_args((messages,), {}, "ollama")

        self.assertEqual(args, ())
        self.assertIn("/no_think", kwargs["messages"][1]["content"])
        self.assertEqual(CREWAI_LMSTUDIO_NO_THINK_PREFILL, "")
        self.assertEqual(kwargs["messages"], again)
        self.assertEqual(sum(1 for item in again if item.get("role") == "assistant" and item.get("content") == "<think>\n\n</think>\n\n"), 0)
        self.assertEqual(other_args[0], messages)
        self.assertEqual(other_kwargs, {})

    def test_lmstudio_preflight_loads_chat_context_and_embedding(self):
        catalog = {
            "ready": True,
            "models": ["qwen-local", "embed-local"],
            "chat_models": ["qwen-local"],
            "embedding_models": ["embed-local"],
            "loaded_models": ["qwen-local"],
            "details": [
                {"name": "qwen-local", "kind": "chat", "loaded": True, "loaded_context_length": 4096},
                {"name": "embed-local", "kind": "embedding", "loaded": False, "loaded_context_length": 0},
            ],
        }
        load_calls = []

        def fake_load(model_name, kind="chat", context_length=None):
            load_calls.append((model_name, kind, context_length))
            return {"success": True}

        with patch.object(album_crew_module, "lmstudio_model_catalog", side_effect=[catalog, catalog]), \
            patch.object(album_crew_module, "lmstudio_load_model", side_effect=fake_load), \
            patch.object(album_crew_module, "local_llm_test_model", return_value={"success": True}), \
            patch.object(album_crew_module, "local_llm_embed", return_value=[0.1, 0.2, 0.3]):
            result = preflight_album_local_llm("lmstudio", "qwen-local", "lmstudio", "embed-local")

        self.assertTrue(result["ok"])
        expected_context = CREWAI_LLM_CONTEXT_WINDOW if CREWAI_LMSTUDIO_PIN_CONTEXT else None
        self.assertIn(("qwen-local", "chat", expected_context), load_calls)
        self.assertIn(("embed-local", "embedding", None), load_calls)

    def test_lmstudio_preflight_reloads_overwide_pinned_context(self):
        catalog = {
            "ready": True,
            "models": ["qwen-local", "embed-local"],
            "chat_models": ["qwen-local"],
            "embedding_models": ["embed-local"],
            "loaded_models": ["qwen-local", "embed-local"],
            "details": [
                {"name": "qwen-local", "kind": "chat", "loaded": True, "loaded_context_length": CREWAI_LLM_CONTEXT_WINDOW * 2},
                {"name": "embed-local", "kind": "embedding", "loaded": True, "loaded_context_length": 2048},
            ],
        }
        load_calls = []

        def fake_load(model_name, kind="chat", context_length=None):
            load_calls.append((model_name, kind, context_length))
            return {"success": True}

        with patch.object(album_crew_module, "lmstudio_model_catalog", side_effect=[catalog, catalog]), \
            patch.object(album_crew_module, "lmstudio_load_model", side_effect=fake_load), \
            patch.object(album_crew_module, "local_llm_test_model", return_value={"success": True}), \
            patch.object(album_crew_module, "local_llm_embed", return_value=[0.1, 0.2, 0.3]):
            result = preflight_album_local_llm("lmstudio", "qwen-local", "lmstudio", "embed-local")

        self.assertTrue(result["ok"])
        if CREWAI_LMSTUDIO_PIN_CONTEXT:
            self.assertIn(("qwen-local", "chat", CREWAI_LLM_CONTEXT_WINDOW), load_calls)
        else:
            self.assertNotIn(("qwen-local", "chat", CREWAI_LLM_CONTEXT_WINDOW), load_calls)

    def test_lmstudio_model_crash_retry_reloads_and_retries(self):
        calls = []

        def fake_call(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise RuntimeError("Error code: 400 - model has crashed without additional information. (Exit code: null)")
            return "OK"

        with patch.object(album_crew_module, "lmstudio_load_model", return_value={"success": True}) as load_model, \
            patch.object(album_crew_module.time, "sleep"):
            llm = _make_llm("qwen-local", "lmstudio")
            object.__setattr__(llm, "_acejam_original_call", fake_call)
            result = llm.call([{"role": "user", "content": "Reply OK."}])

        self.assertTrue(_is_lmstudio_model_crash("model has crashed without additional information"))
        self.assertIn("OK", result)
        self.assertEqual(len(calls), 2)
        load_model.assert_called_once()

    def test_album_crew_uses_long_running_agent_limits(self):
        step_callback = lambda *_args, **_kwargs: None
        task_callback = lambda *_args, **_kwargs: None
        crew = create_album_crew(
            "one cinematic rap track",
            num_tracks=1,
            track_duration=60,
            ollama_model="qwen3.6:27b-instruct-general",
            language="en",
            embedding_model="nomic-embed-text",
            options={"installed_models": ["acestep-v15-turbo"]},
            step_callback=step_callback,
            task_callback=task_callback,
            output_log_file="/tmp/acejam_album_crew_test.json",
        )

        self.assertEqual(len(crew.agents), 8)
        self.assertEqual(len(crew.tasks), 8)
        self.assertTrue(crew.verbose)
        self.assertIs(crew.step_callback, step_callback)
        self.assertIs(crew.task_callback, task_callback)
        self.assertEqual(crew.output_log_file, "/tmp/acejam_album_crew_test.json")
        self.assertTrue(crew.memory.read_only)
        for agent in crew.agents:
            self.assertTrue(agent.verbose)
            self.assertIs(agent.step_callback, step_callback)
            self.assertEqual(agent.max_iter, CREWAI_AGENT_MAX_ITER)
            self.assertIsNone(agent.max_execution_time)
            self.assertEqual(agent.max_retry_limit, CREWAI_AGENT_MAX_RETRY_LIMIT)
            self.assertEqual(agent.respect_context_window, CREWAI_RESPECT_CONTEXT_WINDOW)
        for task in crew.tasks:
            self.assertEqual(task.guardrail_max_retries, CREWAI_TASK_MAX_RETRIES)

    def test_album_crew_memory_uses_local_ollama_defaults(self):
        from crewai.tools.memory_tools import create_memory_tools

        memory = _make_album_memory(DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL, DEFAULT_ALBUM_EMBEDDING_MODEL)

        self.assertEqual(memory.llm.provider, "ollama")
        self.assertEqual(memory.llm.model, DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL)
        self.assertEqual(memory.embedder["provider"], "ollama")
        self.assertEqual(memory.embedder["config"]["model_name"], DEFAULT_ALBUM_EMBEDDING_MODEL)
        self.assertTrue(memory.read_only)
        self.assertEqual(str(memory.storage), str(CREWAI_MEMORY_DIR))
        self.assertNotIn("Save to memory", {getattr(tool, "name", "") for tool in create_memory_tools(memory)})

        writer = _make_album_memory_writer(DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL, DEFAULT_ALBUM_EMBEDDING_MODEL)
        self.assertFalse(writer.read_only)

    def test_compact_memory_writer_uses_explicit_safe_metadata(self):
        class FakeMemory:
            def __init__(self):
                self.calls = []

            def remember(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        fake = FakeMemory()
        long_lyrics = "[Verse]\n" + "full lyric line should not repeat forever\n" * 300
        content, metadata = _compact_track_memory_record(
            {
                "track_number": 1,
                "title": "Signal Room",
                "description": "opener",
                "tags": "cinematic rap, 808 bass",
                "duration": 180,
                "bpm": 92,
                "key_scale": "C minor",
                "time_signature": "4",
                "language": "en",
                "lyrics": long_lyrics,
            }
        )

        self.assertLessEqual(len(content), CREWAI_MEMORY_CONTENT_LIMIT)
        self.assertNotIn("full lyric line should not repeat forever", content)
        self.assertEqual(metadata["title"], "Signal Room")
        self.assertIn("lyric_words", metadata)

        ok = _remember_compact(
            fake,
            content,
            scope="/acejam_album_production/track",
            categories=["track_production"],
            metadata=metadata,
            importance=0.55,
        )

        self.assertTrue(ok)
        self.assertEqual(len(fake.calls), 1)
        args, kwargs = fake.calls[0]
        self.assertLessEqual(len(args[0]), CREWAI_MEMORY_CONTENT_LIMIT)
        self.assertEqual(kwargs["scope"], "/acejam_album_production/track")
        self.assertEqual(kwargs["categories"], ["track_production"])
        self.assertEqual(kwargs["importance"], 0.55)
        self.assertEqual(kwargs["root_scope"], "acejam_album_production")
        self.assertNotIn("lyrics", kwargs["metadata"])

    def test_crewai_structured_models_ignore_extra_fields(self):
        bible = AlbumBiblePayloadModel.model_validate(
            {
                "album_bible": {"concept": "cinematic rap", "title": "ignored"},
                "tracks": [{"track_number": 1, "title": "Signal Room", "unexpected": {"huge": "ignored"}}],
                "random": "ignored",
            }
        )
        track = TrackProductionPayloadModel.model_validate(
            {
                "track_number": 1,
                "title": "Signal Room",
                "lyrics": "[Verse]\nline",
                "extracted_metadata": {"title": "should not fail"},
            }
        )

        self.assertEqual(bible.album_bible.concept, "cinematic rap")
        self.assertEqual(bible.tracks[0].title, "Signal Room")
        self.assertFalse(hasattr(bible, "random"))
        self.assertEqual(track.title, "Signal Room")

    def test_empty_llm_response_returns_crewai_final_answer_fallback(self):
        fallback = _empty_response_fallback_text(DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL)

        self.assertIn("Final Answer:", fallback)
        self.assertIn("acejam_empty_response_fallback", fallback)
        self.assertIn("tracks", fallback)

    def test_empty_llm_response_tries_no_tool_recovery_before_fallback(self):
        calls = []

        def fake_call(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) <= 2:
                return ""
            return '{"ok": true}'

        with patch.object(album_crew_module.time, "sleep"):
            llm = _make_llm("qwen3.6:27b-instruct-general")
            object.__setattr__(llm, "_acejam_original_call", fake_call)
            result = llm.call([{"role": "user", "content": "Return JSON."}])

        self.assertEqual(result, '{"ok": true}')
        self.assertEqual(len(calls), 3)
        recovery_messages = calls[-1][0][0]
        self.assertIn("Do not call tools", recovery_messages[-1]["content"])

    def test_track_production_crew_has_compact_single_track_context(self):
        album_bible = {"concept": "cinematic rap", "arc": "rise and resolve", "motifs": ["signal"]}
        blueprint = {
            "track_number": 1,
            "title": "Signal Room",
            "description": "opener with heavy drums",
            "tags": "cinematic rap, 808 bass, male rap vocal",
            "duration": 180,
            "bpm": 92,
            "key_scale": "C minor",
            "time_signature": "4",
        }
        crew = create_track_production_crew(
            album_bible,
            blueprint,
            num_tracks=7,
            track_duration=180,
            ollama_model=DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            language="en",
            embedding_model=DEFAULT_ALBUM_EMBEDDING_MODEL,
            options={"installed_models": ["acestep-v15-turbo"], "concept": "cinematic rap"},
        )

        self.assertEqual(len(crew.tasks), 2)
        self.assertIn("one track", crew.agents[0].goal.lower())
        descriptions = "\n".join(task.description for task in crew.tasks)
        self.assertIn("Signal Room", descriptions)
        self.assertIn("ace-step-track-payload-contract", descriptions)
        self.assertIn("ACE-Step track prompt template", descriptions)
        self.assertIn("FULL_TAG_LIBRARY", descriptions)
        self.assertIn("lyrics_word_count", descriptions)
        self.assertIn("TagLibraryTool", descriptions)
        self.assertIn("PayloadGateTool", descriptions)
        self.assertNotIn("all tracks", descriptions.lower())
        self.assertLess(len(descriptions), 48000)
        self.assertNotIn("raw_lyrics", descriptions)
        self.assertTrue(crew.memory.read_only)
        self.assertIsNotNone(crew.tasks[-1].guardrail)
        for task in crew.tasks:
            self.assertEqual(task.guardrail_max_retries, CREWAI_TASK_MAX_RETRIES)
            if task.output_json is not None:
                self.assertIs(task.output_json, TrackProductionPayloadModel)

    def test_ace_step_track_prompt_template_contains_full_contract_blocks(self):
        library = compact_full_tag_library()
        lyric_plan = lyric_length_plan(120, "dense", genre_hint="rap")
        rendered = render_track_prompt_template(
            user_album_contract={"applied": True, "album_title": "Safe Brief"},
            ace_step_payload_contract={"version": "unit-contract"},
            lyric_length_plan=lyric_plan,
            language_preset={"code": "en", "language": "English"},
            blueprint={"track_number": 1, "title": "Signal Room"},
            album_bible={"concept": "safe concept"},
        )

        self.assertEqual(set(CAPTION_DIMENSIONS), {
            "genre_style",
            "rhythm_groove",
            "instrumentation",
            "vocal_style",
            "mood_atmosphere",
            "arrangement_energy",
            "mix_production",
        })
        self.assertIn("FULL_TAG_LIBRARY", rendered)
        self.assertIn("SELF_CHECK", rendered)
        self.assertIn("LyricCounterTool", rendered)
        for category in TAG_TAXONOMY:
            self.assertIn(category, rendered)
            self.assertIn(category, library["tag_taxonomy"])
        for category in LYRIC_META_TAGS:
            self.assertIn(category, rendered)
            self.assertIn(category, library["lyric_meta_tags"])
        self.assertIn(ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION, rendered)

    def test_crewai_toolkit_exposes_payload_validation_tools(self):
        tools = make_crewai_tools({
            "concept": "safe rap song",
            "track_duration": 120,
            "language": "en",
            "ace_step_track_payload_contract": {"version": "unit-contract"},
            "ace_step_track_prompt_template_version": ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION,
        })
        names = {getattr(tool, "name", "") for tool in tools}

        self.assertIn("AceStepPromptContractTool", names)
        self.assertIn("LyricCounterTool", names)
        self.assertIn("TagCoverageTool", names)
        self.assertIn("CaptionIntegrityTool", names)
        self.assertIn("PayloadGateTool", names)

    def test_track_json_guardrail_rejects_under_length_payload(self):
        lyric_plan = lyric_length_plan(240, "dense", genre_hint="rap")
        guardrail = _track_json_guardrail_factory(
            blueprint={
                "track_number": 1,
                "title": "Signal Room",
                "tags": "hip-hop, steady groove, 808 bass, male rap vocal, gritty mood, dynamic hook arrangement, polished studio mix",
                "duration": 240,
            },
            options={"track_duration": 240, "lyric_density": "dense", "structure_preset": "auto", "language": "en"},
            lyric_plan=lyric_plan,
        )
        output = SimpleNamespace(raw=json.dumps({
            "track_number": 1,
            "title": "Signal Room",
            "tags": "hip-hop, steady groove, 808 bass, male rap vocal, gritty mood, dynamic hook arrangement, polished studio mix",
            "lyrics": "[Verse]\nWe start\n[Chorus]\nSignal room",
            "duration": 240,
            "language": "en",
        }))

        ok, message = guardrail(output)

        self.assertFalse(ok)
        self.assertIn("lyrics_under_length", message)
        self.assertIn("TrackRepairTool", message)

    def test_track_json_guardrail_repairs_near_miss_and_returns_counts(self):
        lyric_plan = lyric_length_plan(240, "dense", genre_hint="heavy rap")
        lines = []
        lyric_lines = 0
        for section in lyric_plan["sections"]:
            lines.append(f"[{section}]")
            while lyric_lines < 84 and len(lines) < 110:
                lines.append(f"Blocks bend as drums hit cold stone hard {lyric_lines:02d}")
                lyric_lines += 1
                if lyric_lines % 8 == 0:
                    break
            if lyric_lines >= 84:
                break
        guardrail = _track_json_guardrail_factory(
            blueprint={
                "track_number": 1,
                "title": "Concrete Signal",
                "tags": "hip-hop, steady groove, 808 bass, male rap vocal, gritty mood, dynamic hook arrangement, polished studio mix",
                "duration": 240,
            },
            options={"track_duration": 240, "lyric_density": "dense", "structure_preset": "auto", "language": "en"},
            lyric_plan=lyric_plan,
        )
        output = SimpleNamespace(raw=json.dumps({
            "track_number": 1,
            "title": "Concrete Signal",
            "tags": "hip-hop, steady groove, 808 bass, male rap vocal, gritty mood, dynamic hook arrangement, polished studio mix",
            "lyrics": "\n".join(lines),
            "duration": 240,
            "language": "en",
        }))

        ok, repaired_json = guardrail(output)
        repaired = json.loads(repaired_json)

        self.assertTrue(ok)
        effective_min = repaired["lyric_duration_fit"]["plan"]["effective_min_lines"]
        self.assertLess(effective_min, lyric_plan["min_lines"])
        self.assertGreaterEqual(repaired["lyrics_line_count"], effective_min)
        self.assertGreaterEqual(len(repaired["caption_dimensions_covered"]), 7)

    def test_empty_crewai_payload_is_explicit_failure_marker(self):
        payload = json.loads(_empty_response_fallback_text("qwen-local").split("Final Answer: ", 1)[1])

        self.assertTrue(_is_empty_response_payload(payload))

    def test_empty_crewai_response_fails_loudly_without_toolbelt_fallback(self):
        class FakeCrew:
            def __init__(self, output):
                self.output = output

            def kickoff(self):
                return self.output

        class FakeMemory:
            def remember(self, *args, **kwargs):
                return None

        empty_payload = _empty_response_fallback_text("qwen-local")
        with patch.object(album_crew_module, "preflight_album_local_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "embed_ok": True,
            "warnings": [],
            "embedding_model": "embed-local",
            "memory_dir": str(CREWAI_MEMORY_DIR),
        }), patch.object(album_crew_module, "_make_album_memory_writer", return_value=FakeMemory()), \
            patch.object(album_crew_module, "create_album_bible_crew", return_value=FakeCrew(SimpleNamespace(raw=empty_payload))):
            result = plan_album(
                "safe city recovery album",
                num_tracks=1,
                track_duration=45,
                ollama_model="qwen-local",
                embedding_model="embed-local",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed", "agent_engine": "legacy_crewai"},
                use_crewai=True,
                planner_provider="lmstudio",
                embedding_provider="lmstudio",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["planning_engine"], "crewai")
        self.assertTrue(result["crewai_used"])
        self.assertFalse(result["toolbelt_fallback"])
        self.assertEqual(result["tracks"], [])
        self.assertTrue(any("deterministic toolbelt fallback was not used" in line for line in result["logs"]))

    def test_album_crew_ollama_embedding_config(self):
        config = _ollama_embedder_config("nomic-embed-text")

        self.assertEqual(config["provider"], "ollama")
        self.assertEqual(config["config"]["model_name"], "nomic-embed-text")
        self.assertIn("/api/embeddings", config["config"]["url"])

    def test_strip_thinking_blocks_handles_truncated_qwen_output(self):
        self.assertEqual(_strip_thinking_blocks("<think>hidden</think>\n\nOK"), "OK")
        self.assertEqual(_strip_thinking_blocks("<think>hidden but truncated"), "")


if __name__ == "__main__":
    unittest.main()
