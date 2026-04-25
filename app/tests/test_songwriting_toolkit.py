import unittest

from album_crew import (
    CREWAI_AGENT_MAX_ITER,
    CREWAI_AGENT_MAX_RETRY_LIMIT,
    CREWAI_LLM_CONTEXT_WINDOW,
    CREWAI_LLM_MAX_TOKENS,
    CREWAI_LLM_NUM_PREDICT,
    CREWAI_LLM_TIMEOUT_SECONDS,
    CREWAI_MEMORY_CONTENT_LIMIT,
    CREWAI_MEMORY_DIR,
    CREWAI_RESPECT_CONTEXT_WINDOW,
    CREWAI_TASK_MAX_RETRIES,
    DEFAULT_ALBUM_EMBEDDING_MODEL,
    DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
    AlbumBiblePayloadModel,
    TrackProductionPayloadModel,
    _compact_track_memory_record,
    _empty_response_fallback_text,
    _make_llm,
    _make_album_memory,
    _make_album_memory_writer,
    _remember_compact,
    _ollama_embedder_config,
    _strip_thinking_blocks,
    create_album_crew,
    create_track_production_crew,
    plan_album,
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
    normalize_album_tracks,
    normalize_artist_name,
    parse_duration_seconds,
    sanitize_artist_references,
    toolkit_payload,
)


class SongwritingToolkitTest(unittest.TestCase):
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
            }.issubset(tool_names)
        )

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

    def test_artist_references_become_technique_briefs(self):
        cleaned, notes = sanitize_artist_references("Dutch rap like Nas and Eminem")
        self.assertNotIn("Nas", cleaned)
        self.assertNotIn("Eminem", cleaned)
        self.assertIn("internal rhyme", cleaned)
        self.assertIn("multisyllabic rhyme", cleaned)
        self.assertEqual(len(notes), 2)

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
            self.assertEqual(track["shift"], 1.0)
            self.assertIn("production_team", track)
            self.assertIn("studio_engineer", track["production_team"])
            self.assertTrue(track["lyrics"].strip())
            self.assertTrue(track["tool_report"]["length_score"] > 0)
            self.assertIn("bpm", track)

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
        self.assertIn("toolkit_report", result)
        self.assertIn("tool_report", result["tracks"][0])

    def test_album_crew_uses_ollama_llm_without_native_tool_parser(self):
        llm = _make_llm("qwen3.6:27b-instruct-general")

        self.assertEqual(llm.provider, "ollama")
        self.assertEqual(llm.model, "qwen3.6:27b-instruct-general")
        self.assertEqual(llm.timeout, CREWAI_LLM_TIMEOUT_SECONDS)
        self.assertEqual(llm.max_tokens, CREWAI_LLM_MAX_TOKENS)
        self.assertEqual(llm.additional_params["num_ctx"], CREWAI_LLM_CONTEXT_WINDOW)
        self.assertEqual(llm.additional_params["num_predict"], CREWAI_LLM_NUM_PREDICT)
        self.assertEqual(llm.additional_params["context_window_size"], CREWAI_LLM_CONTEXT_WINDOW)
        self.assertFalse(llm.supports_function_calling())

    def test_album_crew_uses_long_running_agent_limits(self):
        crew = create_album_crew(
            "one cinematic rap track",
            num_tracks=1,
            track_duration=60,
            ollama_model="qwen3.6:27b-instruct-general",
            language="en",
            embedding_model="nomic-embed-text",
            options={"installed_models": ["acestep-v15-turbo"]},
        )

        self.assertEqual(len(crew.agents), 8)
        self.assertEqual(len(crew.tasks), 8)
        self.assertTrue(crew.memory.read_only)
        for agent in crew.agents:
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
        self.assertNotIn("all tracks", descriptions.lower())
        self.assertLess(len(descriptions), 10000)
        self.assertNotIn("raw_lyrics", descriptions)
        self.assertTrue(crew.memory.read_only)
        for task in crew.tasks:
            if task.output_json is not None:
                self.assertIs(task.output_json, TrackProductionPayloadModel)

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
