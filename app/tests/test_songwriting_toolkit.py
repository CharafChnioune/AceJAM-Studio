import json
import re
import contextlib
import tempfile
import unittest
from pathlib import Path
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
    _enforce_lyric_part_budget,
    _lyric_like_text,
    _lyric_part_targets,
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
    trim_lyrics_to_limit,
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

    def _valid_lyrics_part_payload(self, user_prompt: str, lines_per_section: int = 8):
        section_match = re.search(r"WRITE_THESE_SECTIONS_ONLY:\n(\[[\s\S]*?\])\n\nPART_TARGETS", user_prompt)
        if not section_match:
            section_match = re.search(r"WRITE_ONLY_THESE_SECTION_TAGS_EXACTLY:\n(\[[\s\S]*?\])\nHOOK_LINES", user_prompt)
        if not section_match:
            section_match = re.search(r"ONLY_ALLOWED_SECTION_TAGS:\n(\[[\s\S]*?\])\nFORBIDDEN_SECTION_TAGS", user_prompt)
        sections = json.loads(section_match.group(1)) if section_match else ["[Verse]"]
        min_lines_match = re.search(r"PART_MIN_VOCAL_LINES_APPROX:\s*(\d+)", user_prompt)
        if min_lines_match:
            required_part_lines = int(min_lines_match.group(1))
            lines_per_section = max(lines_per_section, int((required_part_lines + max(1, len(sections)) - 1) / max(1, len(sections))))
        part_match = re.search(r"PART_INDEX_REQUIRED:\s*(\d+)", user_prompt)
        part_index = int(part_match.group(1)) if part_match else 1
        phrase_match = re.search(r"REQUIRED_PHRASES_FOR_THIS_PART:\n(\[[\s\S]*?\])\n\nPREVIOUS_LYRIC_PARTS_CONTEXT", user_prompt)
        phrases = json.loads(phrase_match.group(1)) if phrase_match else []
        hook_match = re.search(r"HOOK_LINES_TO_USE_IN_CHORUS_OR_HOOK:\n(\[[\s\S]*?\])\n", user_prompt)
        hook_lines = json.loads(hook_match.group(1)) if hook_match else []
        lines = []
        phrase_index = 0
        for section in sections:
            section_text = str(section or "[Verse]").strip()
            lines.append(section_text if section_text.startswith("[") else f"[{section_text}]")
            label = re.sub(r"[^a-z0-9]+", " ", section_text.lower()).strip() or "verse"
            for line_index in range(lines_per_section):
                if "chorus" in label and line_index < len(hook_lines):
                    lines.append(str(hook_lines[line_index]))
                elif phrase_index < len(phrases):
                    lines.append(str(phrases[phrase_index]))
                    phrase_index += 1
                else:
                    lines.append(f"Market lanterns carry hopeful voices {label} {line_index}")
        return {
            "part_index": part_index,
            "sections": sections,
            "lyrics_lines": lines,
            "required_phrases_used": phrases,
            "hook_lines": [line for line in lines if "chorus" in line.lower()][:2],
            "word_count": len(" ".join(lines).split()),
            "line_count": len([line for line in lines if line and not line.startswith("[")]),
        }

    def test_album_agent_llm_options_use_planner_settings_for_ollama(self):
        options = album_crew_module._agent_llm_options(
            "ollama",
            "Track Lyrics Agent Part 1",
            {
                "planner_temperature": 0.82,
                "planner_top_p": 0.97,
                "planner_top_k": 77,
                "planner_repeat_penalty": 1.03,
                "planner_seed": "314",
                "planner_max_tokens": 4096,
                "planner_context_length": 16384,
                "planner_timeout": 180,
            },
        )

        self.assertEqual(options["temperature"], 0.82)
        self.assertEqual(options["top_p"], 0.97)
        self.assertEqual(options["top_k"], 77)
        self.assertEqual(options["repeat_penalty"], 1.03)
        self.assertEqual(options["seed"], 314)
        self.assertEqual(options["num_predict"], 4096)
        self.assertEqual(options["num_ctx"], 16384)
        self.assertEqual(options["timeout"], 180.0)

    def test_album_agent_llm_options_use_lmstudio_chat_supported_settings(self):
        options = album_crew_module._agent_llm_options(
            "lmstudio",
            "Track Caption Agent",
            {
                "planner_temperature": 0.68,
                "planner_top_p": 0.91,
                "planner_top_k": 44,
                "planner_repeat_penalty": 1.12,
                "planner_max_tokens": 2048,
                "planner_context_length": 32768,
                "planner_timeout": 120,
            },
        )

        self.assertEqual(options["temperature"], 0.68)
        self.assertEqual(options["top_p"], 0.91)
        self.assertEqual(options["top_k"], 44)
        self.assertEqual(options["repeat_penalty"], 1.12)
        self.assertEqual(options["max_tokens"], 2048)
        self.assertEqual(options["timeout"], 120.0)
        self.assertNotIn("num_ctx", options)

    def _micro_settings_payload(self, agent_name: str, user_prompt: str = ""):
        def json_after(label: str):
            marker = f"{label}:\n"
            start = user_prompt.find(marker)
            if start < 0:
                return {}
            text = user_prompt[start + len(marker):].lstrip()
            try:
                parsed, _end = json.JSONDecoder().raw_decode(text)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}

        current = json_after("CURRENT_TRACK_STATE")
        locked = json_after("LOCKED_TRACK_FIELDS")
        title = str(current.get("title") or locked.get("title") or locked.get("locked_title") or "")
        style = str(current.get("style") or locked.get("style") or "")
        is_house = title == "Rooftop Letters" or "melodic house" in style.lower()
        is_strict_rap = "STRICT_RAP_LOCK" in user_prompt
        tags = (
            "modern house, handclap drums, warm bassline, synth motif, clear vocal chops, dynamic drop arrangement, airy club texture, polished studio mix"
            if is_house
            else "West Coast hip-hop, boom-bap drums, 808 bass, piano sample motif, male rap vocal, dynamic hook response, gritty street texture, punchy polished rap mix"
            if is_strict_rap
            else "modern pop, bright drums, deep bass, piano motif, clear lead vocal, emotional hook lift, warm analog texture, polished studio mix"
        )
        if agent_name in {"Track BPM Agent", "BPM Agent"}:
            return {"bpm": 88 if title == "Lantern Keys" else (104 if is_house else 92)}
        if agent_name in {"Track Key Agent", "Key Agent"}:
            return {"key_scale": "A minor"}
        if agent_name in {"Track Time Signature Agent", "Time Signature Agent"}:
            return {"time_signature": "4"}
        if agent_name in {"Track Duration Agent", "Duration Agent"}:
            return {"duration": 45}
        if agent_name == "Track Language Agent":
            return {"language": "en", "vocal_language": "en"}
        if agent_name in {"Track Tag List Agent", "Tag Agent"}:
            return {
                "tag_list": tags.split(", "),
                "tags": tags,
                "caption_dimensions_covered": [
                    "primary_genre",
                    "drum_groove",
                    "low_end_bass",
                    "melodic_identity",
                    "vocal_delivery",
                    "arrangement_movement",
                    "texture_space",
                    "mix_master",
                ],
            }
        if agent_name in {"Track Caption Agent", "Caption Agent"}:
            return {"caption": tags}
        if agent_name == "Track Description Agent":
            return {"description": "neighbors rebuild trust with a clear chorus."}
        if agent_name in {"Track Hook Agent", "Hook Agent"}:
            if "Turn the lantern keys" in user_prompt:
                return {
                    "hook_title": "Lantern Keys",
                    "hook_lines": ["Turn the lantern keys", "Every hallway opens when the lights come on"],
                    "hook_promise": "Turn the lantern keys.",
                }
            return {
                "hook_title": "Open the Shutters",
                "hook_lines": ["Open the shutters into a brighter block", "Market bell ringing when the lights come on"],
                "hook_promise": "Open the shutters into a brighter block.",
            }
        if agent_name in {"Track Performance Agent", "Performance Agent"}:
            if is_strict_rap:
                return {
                    "performance_brief": "male rap lead, tight cadence, clear punchline delivery, locked pocket",
                    "negative_control": "no muddy vocals, no random syllables, no prompt text",
                    "genre_profile": "West Coast hip-hop with boom-bap drums, 808 bass, and rap vocal focus",
                }
            return {
                "performance_brief": "clear lead vocal, tight breath control, polished hook lift",
                "negative_control": "no muddy vocals, no random syllables, no prompt text",
                "genre_profile": "safe polished city pop with rap-friendly phrasing",
            }
        return None

    def _director_agent_payload(self, agent_name: str, user_prompt: str = ""):
        def json_after(label: str):
            marker = f"{label}:\n"
            start = user_prompt.find(marker)
            if start < 0:
                return {}
            text = user_prompt[start + len(marker):].lstrip()
            try:
                parsed, _end = json.JSONDecoder().raw_decode(text)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}

        if agent_name == "Album Intake Agent":
            return {
                "album_title": "Market Lights",
                "one_sentence_concept": "A safe city album about rebuilding trust after a long blackout.",
                "style_guardrails": ["compact ACE-Step captions", "complete section-tagged lyrics"],
                "track_roles": ["opener", "second chapter"],
            }
        if agent_name == "Track Concept Agent":
            counter_match = re.search(r"track\s+(\d+)\s+of\s+(\d+)", user_prompt, flags=re.I)
            track_number = int(counter_match.group(1)) if counter_match else 1
            locked = json_after("LOCKED_TRACK_FIELDS")
            title = str(locked.get("title") or locked.get("locked_title") or "").strip()
            if not title:
                title = f"Generated Track {track_number}"
            return {
                "title": title,
                "description": locked.get("narrative") or "neighbors reopen stalls and trade stories instead of rumors.",
                "style": locked.get("style") or ("melodic house" if title == "Rooftop Letters" else "warm boom-bap"),
                "vibe": locked.get("vibe") or ("soft synth pulse" if title == "Rooftop Letters" else "brass stabs, vinyl dust, calm crowd energy"),
                "narrative": locked.get("narrative") or ("friends read old letters as the lights return" if title == "Rooftop Letters" else "neighbors reopen stalls and trade stories instead of rumors"),
                "required_phrases": locked.get("required_phrases") or ["Open the shutters", "Coffee on the corner", "market bell"],
            }
        if agent_name == "Section Map Agent":
            return {
                "section_map": ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]", "[Bridge]", "[Final Chorus]", "[Outro]"],
                "rationale": "vocal track with clear hook and complete ending",
            }
        micro_payload = self._micro_settings_payload(agent_name, user_prompt)
        if micro_payload is not None:
            return micro_payload
        if agent_name.startswith("Track Lyrics Agent Part"):
            return self._valid_lyrics_part_payload(user_prompt, lines_per_section=3)
        if agent_name == "Final Payload Assembler":
            tags_payload = self._micro_settings_payload("Tag Agent", user_prompt) or {}
            caption_payload = self._micro_settings_payload("Caption Agent", user_prompt) or {}
            current = json_after("CURRENT_TRACK_STATE")
            title = str(current.get("title") or "Morning Market")
            lyrics_lines = current.get("lyrics_lines") if isinstance(current.get("lyrics_lines"), list) else []
            return {
                "track_number": 1,
                "title": title,
                "description": "neighbors reopen stalls and trade stories instead of rumors.",
                "caption": caption_payload.get("caption") or "modern pop, bright drums, deep bass, piano motif, clear lead vocal, emotional hook lift, warm analog texture, polished studio mix",
                "tags": tags_payload.get("tags") or "",
                "tag_list": tags_payload.get("tag_list") or [],
                "lyrics_lines": lyrics_lines,
                "bpm": current.get("bpm") or (88 if title == "Lantern Keys" else (104 if title == "Rooftop Letters" else 92)),
                "key_scale": "A minor",
                "time_signature": "4",
                "duration": 45,
                "language": "en",
                "performance_brief": "clear lead vocal, tight breath control, polished hook lift",
                "quality_checks": {"direct_payload": True},
            }
        raise AssertionError(agent_name)

    def _patch_director_agents(self, side_effect=None):
        def default_side_effect(*, agent_name, user_prompt="", logs=None, **_kwargs):
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            return self._director_agent_payload(agent_name, user_prompt)

        class _AgentPatch:
            def __enter__(self_inner):
                self_inner.stack = contextlib.ExitStack()
                self_inner.stack.enter_context(
                    patch.object(album_crew_module, "_agent_json_call", side_effect=side_effect or default_side_effect)
                )
                self_inner.stack.enter_context(
                    patch.object(album_crew_module, "_crewai_micro_block_call", side_effect=side_effect or default_side_effect)
                )
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return self_inner.stack.__exit__(exc_type, exc, tb)

        return (
            patch.object(album_crew_module, "preflight_album_agent_llm", return_value={
                "ok": True,
                "chat_ok": True,
                "warnings": [],
                "errors": [],
            }),
            _AgentPatch(),
        )

    def test_album_prompt_library_is_compact_and_does_not_inject_full_md(self):
        library = album_crew_module.AlbumAgentPromptLibrary(
            {
                "album_agent_genre_prompt": "West Coast rap with boom-bap drums",
                "album_agent_mood_vibe": "dark triumphant",
                "album_agent_vocal_type": "male rap lead",
                "album_agent_audience": "streaming release",
            },
            "en",
        )

        rules = library.system_rules()

        self.assertLess(len(rules), 2600)
        self.assertIn("caption<512 sound-only", rules)
        self.assertIn("MD reference not injected", rules)
        self.assertNotIn("Prompt kit excerpt source", rules)
        self.assertNotIn("FULL_TAG_LIBRARY", rules)

    def test_album_agent_engine_normalizes_legacy_to_crewai_micro(self):
        self.assertEqual(album_crew_module.normalize_album_agent_engine(None), "acejam_agents")
        self.assertEqual(album_crew_module.normalize_album_agent_engine("acejam_direct"), "acejam_agents")
        self.assertEqual(album_crew_module.normalize_album_agent_engine("legacy_crewai"), "crewai_micro")
        self.assertEqual(album_crew_module.album_agent_engine_label("crewai_micro"), "CrewAI Micro Tasks")

    def test_crewai_micro_llm_kwargs_use_planner_settings(self):
        ollama = album_crew_module._crewai_micro_llm_kwargs(
            "qwen-local",
            "ollama",
            "Track Lyrics Agent Part 1",
            {
                "planner_temperature": 0.9,
                "planner_top_p": 0.88,
                "planner_top_k": 61,
                "planner_repeat_penalty": 1.17,
                "planner_seed": "42",
                "planner_max_tokens": 2048,
                "planner_context_length": 12288,
                "planner_timeout": 240,
            },
        )
        self.assertEqual(ollama["provider"], "ollama")
        self.assertEqual(ollama["temperature"], 0.9)
        self.assertEqual(ollama["top_p"], 0.88)
        self.assertEqual(ollama["seed"], 42)
        self.assertEqual(ollama["max_tokens"], 2048)
        options = ollama["additional_params"]["extra_body"]["options"]
        self.assertEqual(options["top_k"], 61)
        self.assertEqual(options["repeat_penalty"], 1.17)
        self.assertEqual(options["num_ctx"], 12288)
        self.assertEqual(options["num_predict"], 2048)

        lmstudio = album_crew_module._crewai_micro_llm_kwargs(
            "qwen-local",
            "lmstudio",
            "Caption Agent",
            {
                "planner_temperature": 0.7,
                "planner_top_p": 0.91,
                "planner_top_k": 44,
                "planner_repeat_penalty": 1.08,
                "planner_max_tokens": 1024,
                "planner_context_length": 32768,
                "planner_timeout": 120,
            },
        )
        self.assertEqual(lmstudio["provider"], "openai")
        self.assertEqual(lmstudio["max_tokens"], 1024)
        self.assertNotIn("num_ctx", lmstudio)
        self.assertEqual(lmstudio["additional_params"]["top_k"], 44)
        self.assertEqual(lmstudio["additional_params"]["repeat_penalty"], 1.08)

    def test_crewai_micro_guardrail_parses_delimiter_blocks_and_rejects_json(self):
        guardrail = album_crew_module._crewai_micro_block_guardrail("caption_agent_payload")
        ok, _ = guardrail(SimpleNamespace(raw="******caption******\nclear rap vocal, hard drums\n******/caption******"))
        self.assertTrue(ok)

        ok, message = guardrail(SimpleNamespace(raw='{"caption":"clear rap vocal"}'))
        self.assertFalse(ok)
        self.assertIn("json_response_not_allowed", str(message))

    def test_crewai_micro_engine_routes_through_micro_call(self):
        direct_calls = []
        micro_calls = []

        def fake_direct(*, agent_name, **_kwargs):
            direct_calls.append(agent_name)
            raise AssertionError("direct runtime should not be used")

        def fake_micro(*, agent_name, user_prompt="", logs=None, **_kwargs):
            micro_calls.append(agent_name)
            if isinstance(logs, list):
                logs.append(f"CrewAI Micro Agent call: {agent_name} attempt 1 via patched test.")
            return self._director_agent_payload(agent_name, user_prompt)

        with patch.object(album_crew_module, "preflight_album_agent_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "warnings": [],
            "errors": [],
        }), patch.object(album_crew_module, "_agent_json_call", side_effect=fake_direct), patch.object(album_crew_module, "_crewai_micro_block_call", side_effect=fake_micro):
            result = plan_album(
                "one track pop EP",
                num_tracks=1,
                track_duration=45,
                options={
                    "installed_models": ["acestep-v15-turbo"],
                    "song_model_strategy": "best_installed",
                    "agent_engine": "crewai_micro",
                },
                planner_provider="ollama",
                embedding_provider="ollama",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_engine"], "crewai_micro")
        self.assertTrue(result["crewai_used"])
        self.assertFalse(direct_calls)
        self.assertIn("Album Intake Agent", micro_calls)

    def test_agent_block_instruction_demands_exact_shape(self):
        instruction = album_crew_module._agent_json_instruction("caption_agent_payload")

        self.assertIn("DELIMITER_RESPONSE_CONTRACT", instruction)
        self.assertIn("required blocks", instruction)
        self.assertIn("******caption******", instruction)
        self.assertIn("******/caption******", instruction)
        self.assertIn("Do not output JSON", instruction)

    def test_agent_block_parser_preserves_multiline_lyrics(self):
        raw = (
            "******part_index******\n1\n******/part_index******\n"
            "******sections******\n[Intro]\n[Verse 1]\n[Chorus]\n******/sections******\n"
            "******lyrics_lines******\n[Intro]\nWarm piano enters\n[Verse 1]\n******not_a_real_delimiter****** inside a lyric\n[Chorus]\nLanterns call us home\n******/lyrics_lines******"
        )

        payload = album_crew_module._parse_agent_block_payload(raw, "lyrics_part_1_payload")

        self.assertEqual(payload["part_index"], 1)
        self.assertEqual(payload["sections"], ["[Intro]", "[Verse 1]", "[Chorus]"])
        self.assertIn("******not_a_real_delimiter****** inside a lyric", payload["lyrics_lines"])
        self.assertEqual(payload["lyrics_lines"][0], "[Intro]")

    def test_agent_block_parser_fails_json_only_response(self):
        with self.assertRaisesRegex(ValueError, "json_response_not_allowed"):
            album_crew_module._parse_agent_block_payload('{"caption":"bright drums"}', "caption_agent_payload")

    def test_agent_block_parser_reports_missing_and_extra_blocks(self):
        with self.assertRaisesRegex(ValueError, "missing_block:one_sentence_concept,style_guardrails,track_roles"):
            album_crew_module._parse_agent_block_payload(
                "******album_title******\nMarket Lights\n******/album_title******",
                "album_intake_payload",
            )

        with self.assertRaisesRegex(ValueError, "extra_block:bpm"):
            album_crew_module._parse_agent_block_payload(
                "******caption******\nbright drums\n******/caption******\n"
                "******bpm******\n95\n******/bpm******",
                "caption_agent_payload",
            )

    def test_agent_block_call_retries_on_extra_blocks_for_known_schema(self):
        responses = iter([
            "******caption******\nbright drums\n******/caption******\n******bpm******\n95\n******/bpm******",
            "******caption******\nbright drums\n******/caption******",
        ])
        prompts = []

        def fake_call_agent_llm(**kwargs):
            prompts.append(kwargs.get("user_prompt") or "")
            return next(responses)

        with patch.object(album_crew_module, "_call_agent_llm", side_effect=fake_call_agent_llm):
            payload = album_crew_module._agent_json_call(
                agent_name="Caption Agent",
                provider="ollama",
                model_name="qwen-local",
                user_prompt="Return caption JSON",
                logs=[],
                debug_options={},
                schema_name="caption_agent_payload",
                max_retries=1,
            )

        self.assertEqual(payload, {"caption": "bright drums"})
        self.assertEqual(len(prompts), 2)
        self.assertIn("EXPECTED_BLOCK_SHAPE", prompts[1])

    def test_director_micro_call_order_sets_sonic_and_metadata_before_lyrics(self):
        calls = []

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            calls.append(agent_name)
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                "safe West Coast rap opener",
                num_tracks=1,
                track_duration=45,
                language="en",
                options={
                    "installed_models": ["acestep-v15-turbo"],
                    "song_model_strategy": "best_installed",
                    "album_agent_genre_prompt": "West Coast rap with clean hooks",
                },
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        expected = [
            "Album Intake Agent",
            "Track Concept Agent",
            "Tag Agent",
            "BPM Agent",
            "Key Agent",
            "Time Signature Agent",
            "Duration Agent",
            "Section Map Agent",
            "Hook Agent",
            "Track Lyrics Agent Part 1",
        ]
        positions = [calls.index(agent) for agent in expected]
        self.assertEqual(positions, sorted(positions))

    def test_director_lyrics_prompts_do_not_leak_producer_or_source_excerpt(self):
        lyric_prompts = []

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            if agent_name.startswith("Track Lyrics Agent Part"):
                lyric_prompts.append(user_prompt)
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={
                    "installed_models": ["acestep-v15-turbo"],
                    "song_model_strategy": "best_installed",
                    "album_agent_genre_prompt": "warm boom-bap with clean hooks",
                },
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertTrue(lyric_prompts)
        combined = "\n".join(lyric_prompts)
        self.assertNotIn("producer_credit", combined)
        self.assertNotIn("Ada North", combined)
        self.assertNotIn("source_excerpt", combined)

    def test_director_lyrics_prompts_are_section_bound_and_do_not_include_previous_body(self):
        lyric_prompts = []

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            if agent_name.startswith("Track Lyrics Agent Part"):
                lyric_prompts.append(user_prompt)
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(lyric_prompts), 1)
        first_prompt = lyric_prompts[0]
        self.assertIn("ONLY_ALLOWED_SECTION_TAGS", first_prompt)
        self.assertIn("FORBIDDEN_SECTION_TAGS_ALREADY_WRITTEN", first_prompt)
        self.assertIn("sections must equal ONLY_ALLOWED_SECTION_TAGS exactly", first_prompt)
        self.assertNotIn("PREVIOUS_LYRIC_PARTS_CONTEXT", first_prompt)
        self.assertNotIn('"section_map"', first_prompt)

    def test_director_retries_lyrics_part_when_sections_do_not_match(self):
        part_one_calls = 0

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal part_one_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Track Lyrics Agent Part 1":
                part_one_calls += 1
                if part_one_calls <= 3:
                    return {
                        "part_index": 1,
                        "sections": ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]"],
                        "lyrics_lines": [
                            "[Intro]", "Market opens softly",
                            "[Verse 1]", "Lanterns lift the morning",
                            "[Pre-Chorus]", "The bell begins to ring",
                            "[Chorus]", "This chorus arrived too early",
                        ],
                    }
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(part_one_calls, 4)
        self.assertTrue(any("Agent semantic validation retry: Track Lyrics Agent Part 1" in line for line in result["logs"]))
        self.assertTrue(any("sections_mismatch" in line for line in result["logs"]))

    def test_director_retries_lyrics_part_after_json_parse_failure(self):
        part_one_calls = 0

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal part_one_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Track Lyrics Agent Part 1":
                part_one_calls += 1
                if part_one_calls <= 2:
                    raise album_crew_module.AceJamAgentError(
                        "Track Lyrics Agent Part 1 failed to produce valid delimiter blocks after 2 attempt(s)"
                    )
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(part_one_calls, 3)
        self.assertTrue(any("Agent block validation retry: Track Lyrics Agent Part 1" in line for line in result["logs"]))

    def test_director_lyrics_part_falls_back_when_planner_refuses_json(self):
        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Track Lyrics Agent Part 1":
                raise album_crew_module.AceJamAgentError(
                    "Track Lyrics Agent Part 1 failed to produce valid delimiter blocks after 2 attempt(s): refusal text"
                )
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch, patch.object(album_crew_module, "ACEJAM_AGENT_GATE_REPAIR_RETRIES", 1):
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        lyrics = result["tracks"][0]["lyrics"]
        self.assertIn("[Intro]", lyrics)
        self.assertIn("[Verse 1]", lyrics)
        self.assertIn("[Chorus]", lyrics)
        self.assertTrue(any("Agent deterministic fallback: Track Lyrics Agent Part 1" in line for line in result["logs"]))

    def test_director_retries_hook_when_hook_lines_include_section_tags(self):
        hook_calls = 0

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal hook_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Hook Agent":
                hook_calls += 1
                if hook_calls <= 2:
                    return {
                        "hook_title": "Open the Shutters",
                        "hook_lines": ["[Chorus]", "Open the shutters into a brighter block"],
                        "hook_promise": "Open the shutters into a brighter block.",
                    }
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(hook_calls, 3)
        self.assertTrue(any("Agent semantic validation retry: Hook Agent" in line for line in result["logs"]))
        self.assertTrue(any("hook_lines_must_not_contain_section_tags" in line for line in result["logs"]))

    def test_director_retries_caption_when_it_leaks_metadata(self):
        caption_calls = 0

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal caption_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Caption Agent":
                caption_calls += 1
                if caption_calls == 1:
                    return {"caption": "92 BPM, A minor, Produced by Ada North, Morning Market comeback story"}
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(caption_calls, 2)
        caption = result["tracks"][0]["caption"]
        self.assertNotIn("BPM", caption)
        self.assertNotIn("Ada North", caption)
        self.assertNotIn("Morning Market", caption)
        self.assertTrue(any("Agent semantic validation retry: Caption Agent" in line for line in result["logs"]))

    def test_director_caption_falls_back_when_planner_connection_breaks(self):
        caption_calls = 0

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal caption_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Caption Agent":
                caption_calls += 1
                raise album_crew_module.AceJamAgentError(
                    "Caption Agent failed to produce valid delimiter blocks after 2 attempt(s): BrokenPipeError"
                )
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch, patch.object(album_crew_module, "ACEJAM_AGENT_GATE_REPAIR_RETRIES", 1):
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(caption_calls, 2)
        caption = result["tracks"][0]["caption"]
        self.assertTrue(caption)
        self.assertNotIn("Morning Market", caption)
        self.assertTrue(any("Agent deterministic fallback: Caption Agent" in line for line in result["logs"]))

    def test_director_performance_falls_back_when_planner_connection_breaks(self):
        performance_calls = 0

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal performance_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Performance Agent":
                performance_calls += 1
                raise album_crew_module.AceJamAgentError(
                    "Performance Agent failed to produce valid delimiter blocks after 2 attempt(s): BrokenPipeError"
                )
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch, patch.object(album_crew_module, "ACEJAM_AGENT_GATE_REPAIR_RETRIES", 1):
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(performance_calls, 2)
        track = result["tracks"][0]
        self.assertIn("lead vocal", track["performance_brief"])
        self.assertTrue(any("Agent deterministic fallback: Performance Agent" in line for line in result["logs"]))

    def test_final_gate_duplicate_sections_triggers_lyrics_regeneration(self):
        original_gate = album_crew_module._director_minimal_validate
        gate_calls = 0
        lyrics_calls = 0

        def fake_gate(track, section_tags, *args, **kwargs):
            nonlocal gate_calls
            gate_calls += 1
            if gate_calls == 1:
                return {
                    "version": album_crew_module.ACEJAM_ALBUM_DIRECTOR_VERSION,
                    "gate_passed": False,
                    "status": "fail",
                    "issues": ["duplicate_section_tags:[Verse 1]"],
                    "caption_chars": len(str(track.get("caption") or "")),
                    "lyrics_chars": len(str(track.get("lyrics") or "")),
                    "section_tags": section_tags,
                }
            return original_gate(track, section_tags, *args, **kwargs)

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal lyrics_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name.startswith("Track Lyrics Agent Part"):
                lyrics_calls += 1
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch, patch.object(album_crew_module, "_director_minimal_validate", side_effect=fake_gate):
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertGreaterEqual(gate_calls, 2)
        self.assertGreater(lyrics_calls, len(album_crew_module._director_section_groups(["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]", "[Bridge]", "[Final Chorus]", "[Outro]"])))
        self.assertTrue(any("Final gate repair retry: track 1 attempt 1/8: duplicate_section_tags:[Verse 1]" in line for line in result["logs"]))

    def test_final_gate_arrangement_leakage_uses_deterministic_sanitizer(self):
        original_gate = album_crew_module._director_minimal_validate
        gate_calls = 0
        lyrics_calls = 0
        blocked_line = "The orchestra swells like a stage direction"

        def fake_gate(track, section_tags, *args, **kwargs):
            nonlocal gate_calls
            gate_calls += 1
            if gate_calls == 1:
                return {
                    "version": album_crew_module.ACEJAM_ALBUM_DIRECTOR_VERSION,
                    "gate_passed": False,
                    "status": "fail",
                    "issues": ["non_rap_arrangement_lyric_leakage"],
                    "caption_chars": len(str(track.get("caption") or "")),
                    "lyrics_chars": len(str(track.get("lyrics") or "")),
                    "genre_adherence": {
                        "stats": {
                            "arrangement_lyric_scan": [
                                {"line": blocked_line, "status": "blocked", "match": "orchestra swells"}
                            ]
                        }
                    },
                    "section_tags": section_tags,
                }
            return original_gate(track, section_tags, *args, **kwargs)

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            nonlocal lyrics_calls
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name.startswith("Track Lyrics Agent Part"):
                lyrics_calls += 1
                payload = self._valid_lyrics_part_payload(user_prompt, lines_per_section=3)
                for idx, line in enumerate(payload["lyrics_lines"]):
                    if not str(line).startswith("["):
                        payload["lyrics_lines"][idx] = blocked_line
                        break
                return payload
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch, patch.object(album_crew_module, "_director_minimal_validate", side_effect=fake_gate):
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        expected_initial_parts = len(album_crew_module._director_section_groups(["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]", "[Bridge]", "[Final Chorus]", "[Outro]"]))
        self.assertTrue(result["success"])
        self.assertEqual(lyrics_calls, expected_initial_parts)
        self.assertNotIn(blocked_line, result["tracks"][0]["lyrics"])
        self.assertTrue(any("Final gate deterministic lyric sanitizer" in line for line in result["logs"]))

    def test_track_planning_failure_does_not_block_following_tracks(self):
        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Hook Agent" and "track 1 of 2" in user_prompt:
                return {
                    "hook_title": "Bad Hook",
                    "hook_lines": ["[Chorus]", "Still tagged"],
                    "hook_promise": "Still tagged.",
                }
            return self._director_agent_payload(agent_name, user_prompt)

        with tempfile.TemporaryDirectory() as tmp_dir:
            preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
            with preflight_patch, agent_patch, patch.object(album_crew_module, "ACEJAM_AGENT_GATE_REPAIR_RETRIES", 1):
                result = plan_album(
                    self.SAFE_CONTRACT_PROMPT,
                    num_tracks=2,
                    track_duration=45,
                    language="en",
                    options={
                        "installed_models": ["acestep-v15-turbo"],
                        "song_model_strategy": "best_installed",
                        "album_debug_dir": tmp_dir,
                    },
                    use_crewai=True,
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_status"], "partial")
        self.assertEqual(result["planning_failed_count"], 1)
        self.assertEqual(result["tracks"][0]["planning_status"], "failed")
        self.assertEqual(result["tracks"][1]["planning_status"], "completed")
        self.assertTrue(any("Track planning failed but album continues" in line for line in result["logs"]))

    def test_semantic_retry_hard_failure_logs_debug_jsonl(self):
        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            if agent_name == "Hook Agent":
                return {
                    "hook_title": "Bad Hook",
                    "hook_lines": ["[Chorus]", "Still tagged"],
                    "hook_promise": "Still tagged.",
                }
            return self._director_agent_payload(agent_name, user_prompt)

        with tempfile.TemporaryDirectory() as tmp_dir:
            preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
            with preflight_patch, agent_patch, patch.object(album_crew_module, "ACEJAM_AGENT_GATE_REPAIR_RETRIES", 2):
                result = plan_album(
                    self.SAFE_CONTRACT_PROMPT,
                    num_tracks=1,
                    track_duration=45,
                    language="en",
                    options={
                        "installed_models": ["acestep-v15-turbo"],
                        "song_model_strategy": "best_installed",
                        "album_debug_dir": tmp_dir,
                    },
                    use_crewai=True,
                )

            debug_jsonl = Path(tmp_dir) / "04_agent_responses.jsonl"
            self.assertFalse(result["success"])
            self.assertIn("Hook Agent failed semantic validation after 2 repair attempt", result["error"])
            self.assertTrue(debug_jsonl.exists())
            debug_text = debug_jsonl.read_text(encoding="utf-8")
            self.assertIn('"validation_issues"', debug_text)
            self.assertIn('"repair_attempt": 2', debug_text)

    def test_final_payload_assembler_is_deterministic_not_llm(self):
        calls = []

        def fake_agent_json_call(*, agent_name, user_prompt="", logs=None, **_kwargs):
            calls.append(agent_name)
            if isinstance(logs, list):
                logs.append(f"AceJAM Agent call: {agent_name} attempt 1 via patched test.")
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
            result = plan_album(
                self.SAFE_CONTRACT_PROMPT,
                num_tracks=1,
                track_duration=45,
                language="en",
                options={"installed_models": ["acestep-v15-turbo"], "song_model_strategy": "best_installed"},
                use_crewai=True,
            )

        self.assertTrue(result["success"])
        self.assertNotIn("Final Payload Assembler", calls)
        self.assertTrue(any("Final Payload Assembler: deterministic assembly" in line for line in result["logs"]))

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

    def test_user_album_contract_ignores_generated_lyrics_as_fake_track_title(self):
        clean_prompt = """
Album: You Buried the Wrong Man
Track 1: Concrete Canyons (Prod. Dr. Dre)
Vibe: Low-end rumble, sirens, West Coast weight
Verse: They paved them blocks just to hide what's real,
Naming Drop Style: "Death Row"
"""
        polluted_prompt = clean_prompt + """
Track 1: "Concrete Canyons West Coast rap with dark orchestral/cinematic elements [Intro] leaked old lyrics [Verse 1] more old lines [Chorus] Instrumental break"
Track 1: "Concrete Canyons"
Style: West Coast rap
"""
        contract = extract_user_album_contract(
            polluted_prompt,
            1,
            "en",
            {"raw_user_prompt": clean_prompt, "user_prompt": clean_prompt},
        )

        self.assertEqual(len(contract["tracks"]), 1)
        self.assertEqual(contract["tracks"][0]["locked_title"], "Concrete Canyons")
        self.assertNotIn("Instrumental break", contract["concept"])

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

    def test_all_models_album_portfolio_has_official_render_models(self):
        portfolio = album_model_portfolio({"acestep-v15-turbo"})
        self.assertEqual([item["model"] for item in portfolio], ALBUM_MODEL_PORTFOLIO_MODELS)
        self.assertEqual(len(portfolio), 9)
        self.assertIn("acestep-v15-turbo-shift1", ALBUM_MODEL_PORTFOLIO_MODELS)
        self.assertIn("acestep-v15-turbo-continuous", ALBUM_MODEL_PORTFOLIO_MODELS)
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
        self.assertGreaterEqual(long["min_words"], 340)
        self.assertIn("Verse 3 - Beat Switch", long["sections"])
        self.assertLessEqual(long["safe_lyrics_char_target"], 3600)
        self.assertLessEqual(ten_minutes["max_lyrics_chars"], 4096)
        self.assertGreaterEqual(short["min_lines"], len(short["sections"]))

    def test_rap_prompt_auto_selects_rap_dense_hit_targets(self):
        plan = lyric_length_plan(240, "balanced", genre_hint="West Coast rap, clear hook")

        self.assertEqual(plan["density"], "rap_dense")
        self.assertGreaterEqual(plan["min_words"], 430)
        self.assertGreaterEqual(plan["target_words"], 480)
        self.assertGreaterEqual(plan["min_lines"], 75)
        self.assertIn("Verse 3 - Beat Switch", plan["sections"])
        self.assertLessEqual(plan["safe_lyrics_char_target"], 3600)

    def test_trim_lyrics_to_limit_never_slices_mid_line_or_adds_outro(self):
        lyrics = "\n".join([
            "[Intro]",
            "Open the shutters",
            "[Verse]",
            *[f"Market lanterns carry hopeful voices number {idx:03d}" for idx in range(260)],
            "[Outro]",
            "Coffee on the corner",
        ])

        trimmed = trim_lyrics_to_limit(lyrics, 900)

        self.assertLessEqual(len(trimmed), 900)
        self.assertNotIn("Let it breathe", trimmed)
        self.assertNotRegex(trimmed, r"\b[a-zA-Z]\s*$")
        self.assertNotIn("number 0", trimmed[-12:])

    def test_lyric_part_targets_include_enforced_char_budget(self):
        plan = lyric_length_plan(240, "dense", genre_hint="rap")
        groups = [["Intro", "Verse 1"], ["Chorus", "Verse 2"], ["Bridge", "Final Chorus", "Outro"]]
        targets = _lyric_part_targets(plan, groups, 0)
        payload = {
            "lyrics_lines": [
                "[Intro]",
                *[f"Market lanterns carry hopeful voices number {idx:03d}" for idx in range(90)],
            ]
        }

        repaired = _enforce_lyric_part_budget(payload, plan, groups, 0)
        lyrics = "\n".join(repaired["lyrics_lines"])

        self.assertIn("max_chars", targets)
        self.assertLessEqual(len(lyrics), targets["max_chars"])
        self.assertTrue(repaired["quality_checks"]["budget_repaired"])

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
        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["tracks"][0]["song_model"], "acestep-v15-turbo")
        self.assertEqual(result["planning_engine"], "acejam_agents")
        self.assertFalse(result["crewai_used"])
        self.assertTrue(any("Editable album plan received" in line for line in result["logs"]))

    def test_xl_sft_final_overrides_editable_track_model_choice(self):
        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["tracks"][0]["quality_profile"], "chart_master")

    def test_editable_plan_runs_through_acejam_agents_when_enabled(self):
        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
            result = plan_album(
                "two track pop EP",
                num_tracks=1,
                track_duration=45,
                options={
                    "installed_models": ["acestep-v15-turbo", "acestep-v15-sft"],
                    "song_model_strategy": "best_installed",
                },
                use_crewai=True,
                input_tracks=[{"track_number": 1, "title": "Manual Model", "song_model": "acestep-v15-sft", "style": "warm boom-bap"}],
                planner_provider="ollama",
                embedding_provider="ollama",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["planning_engine"], "acejam_agents")
        self.assertTrue(result["custom_agents_used"])
        self.assertEqual(result["tracks"][0]["title"], "Manual Model")
        self.assertEqual(result["tracks"][0]["song_model"], "acestep-v15-turbo")
        self.assertTrue(any("Editable album plan received" in line for line in result["logs"]))
        self.assertTrue(any("AceJAM Agent call: BPM Agent" in line for line in result["logs"]))

    def test_micro_agent_tags_are_preserved_without_default_genre_inflation(self):
        tracks = normalize_album_tracks(
            [
                {
                    "title": "Concrete Canyons",
                    "description": "low-end opener",
                    "caption": "West Coast Hip-Hop, boom-bap drums, 808 bass, sirens, melodic rap vocal, hard-hitting drums, atmospheric mix",
                    "tags": "West Coast Hip-Hop, boom-bap drums, 808 bass, sirens, melodic rap vocal, hard-hitting drums, atmospheric mix",
                    "tag_list": [
                        "West Coast Hip-Hop",
                        "boom-bap drums",
                        "808 bass",
                        "sirens",
                        "melodic rap vocal",
                        "hard-hitting drums",
                        "atmospheric mix",
                    ],
                    "lyrics": "[Instrumental]",
                    "duration": 60,
                    "agent_micro_settings_flow": True,
                }
            ],
            {
                "strict_album_agents": True,
                "agent_engine": "acejam_agents",
                "installed_models": ["acestep-v15-turbo"],
                "song_model_strategy": "best_installed",
                "track_duration": 60,
                "language": "en",
            },
        )

        track = tracks[0]
        joined = " ".join(track["tag_list"]).lower()
        self.assertEqual(track["tag_list"][0], "West Coast Hip-Hop")
        self.assertNotIn("trap", joined)
        self.assertNotIn("drill", joined)
        self.assertNotIn("pop", joined)
        self.assertEqual(track["tags"], "West Coast Hip-Hop, boom-bap drums, 808 bass, sirens, melodic rap vocal, hard-hitting drums, atmospheric mix")

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
        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["planning_engine"], "acejam_agents")
        self.assertFalse(result["crewai_used"])
        self.assertFalse(result["toolbelt_fallback"])
        self.assertIn("toolkit_report", result)
        self.assertTrue(result["tracks"][0]["agent_complete_payload"])

    def test_album_plan_streams_monitor_logs(self):
        streamed = []
        long_concept = "short folk album " + ("with gentle safe details " * 40)
        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["planning_engine"], "acejam_agents")
        preview_line = next(line for line in streamed if line.startswith("Concept preview: "))
        self.assertLessEqual(len(preview_line), 240)
        self.assertTrue(any("AceJAM Director produced 1 direct ACE-Step track payload" in line for line in streamed))

    def test_acejam_agents_default_plans_with_direct_json_calls(self):
        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["tracks"][0]["payload_gate_status"], "pass")
        self.assertFalse(result["memory_enabled"])
        self.assertEqual(result["context_chunks"], 0)
        self.assertTrue(any("AceJAM Director produced" in line for line in result["logs"]))
        self.assertTrue(any("AceJAM Agent call: BPM Agent" in line for line in result["logs"]))
        self.assertTrue(any("AceJAM Agent call: Caption Agent" in line for line in result["logs"]))

    def test_acejam_agents_empty_intake_fails_loudly_without_toolbelt(self):
        def fake_agent_json_call(*, agent_name, user_prompt="", **_kwargs):
            if agent_name == "Album Intake Agent":
                raise album_crew_module.AceJamAgentError("empty intake response")
            return self._director_agent_payload(agent_name, user_prompt)

        preflight_patch, agent_patch = self._patch_director_agents(fake_agent_json_call)
        with preflight_patch, agent_patch:
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

        self.assertFalse(result["success"])
        self.assertEqual(result["planning_engine"], "acejam_agents")
        self.assertFalse(result["toolbelt_fallback"])
        self.assertIn("empty intake response", result["error"])

    def test_acejam_agents_scaffold_fills_missing_tracks_from_short_bible(self):
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
        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertTrue(all(track.get("payload_gate_status") == "pass" for track in result["tracks"]))
        self.assertFalse(result["memory_enabled"])
        self.assertEqual(result["retrieval_rounds"], 0)
        self.assertTrue(any("AceJAM Agent call: Track Concept Agent" in line for line in result["logs"]))

    def test_acejam_agent_prompt_contains_compact_contract_template_and_counter(self):
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

        self.assertIn("ORIGINAL_PROMPT_SIGNAL", prompt)
        self.assertNotIn("FULL_ORIGINAL_ALBUM_PROMPT", prompt)
        self.assertIn("Track 1: \"Morning Market\"", prompt)
        self.assertIn("TRACK COUNTER: you are writing track 1 of 2", prompt)
        self.assertIn("FULL_TAG_LIBRARY_COMPACT", prompt)
        self.assertIn("LYRIC_LENGTH_PLAN", prompt)

    def test_acejam_agents_empty_response_fails_loudly_without_toolbelt(self):
        with patch.object(album_crew_module, "preflight_album_agent_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "warnings": [],
            "errors": [],
        }), patch.object(album_crew_module, "_agent_json_call", side_effect=album_crew_module.AceJamAgentError("empty response")), patch.object(album_crew_module, "_crewai_micro_block_call", side_effect=album_crew_module.AceJamAgentError("empty response")):
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
        self.assertTrue(any("AceJAM Director planning failed loudly" in line for line in result["logs"]))

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

    def test_agent_block_call_uses_plain_ollama_transport_by_default(self):
        calls = []

        def fake_call_agent_llm(**kwargs):
            calls.append(kwargs.get("json_format"))
            return "******caption******\nbright drums\n******/caption******"

        with patch.object(album_crew_module, "_call_agent_llm", side_effect=fake_call_agent_llm):
            payload = album_crew_module._agent_json_call(
                agent_name="Caption Agent",
                provider="ollama",
                model_name="qwen-local",
                user_prompt="Return caption blocks",
                logs=[],
                debug_options={},
                schema_name="caption_agent_payload",
                max_retries=0,
            )

        self.assertEqual(payload, {"caption": "bright drums"})
        self.assertEqual(calls, [False])

    def test_agent_payload_lyrics_lines_are_joined(self):
        payload = _coerce_agent_lyrics_payload({
            "title": "Lanterns on the Pier",
            "lyrics_lines": ["[Intro]", "Warm piano enters", "[Chorus]", "Lanterns call us home"],
        })

        self.assertEqual(payload["lyrics"], "[Intro]\nWarm piano enters\n[Chorus]\nLanterns call us home")
        self.assertEqual(payload["lyrics_lines"][2], "[Chorus]")

    def test_agent_payload_lyrics_dict_lines_keep_section_tags(self):
        payload = _coerce_agent_lyrics_payload({
            "part_index": 1,
            "sections": ["[Intro]", "[Verse 1]", "[Chorus]"],
            "lyrics_lines": [
                {"section_tag": "[Intro]", "line": "Death Row... East Coast... Closed doors..."},
                {"section_tag": "[Verse 1]", "line": "They paved them blocks just to hide what's real,"},
                {"section_tag": "[Chorus]", "line": "But ghosts don't sleep when they settle scores."},
            ],
        })

        self.assertEqual(
            payload["lyrics_lines"],
            [
                "[Intro]",
                "Death Row... East Coast... Closed doors...",
                "[Verse 1]",
                "They paved them blocks just to hide what's real,",
                "[Chorus]",
                "But ghosts don't sleep when they settle scores.",
            ],
        )
        self.assertIn("[Verse 1]", payload["lyrics"])

    def test_agent_payload_lyrics_dict_lines_do_not_duplicate_repeated_sections(self):
        payload = _coerce_agent_lyrics_payload({
            "part_index": 1,
            "sections": ["[Intro]", "[Verse 1]", "[Pre-Chorus]"],
            "lyrics_lines": [
                {"section": "[Intro]", "line": "Concrete canyons closing in tight"},
                {"section": "[Verse 1]", "line": "They paved them blocks just to hide what's real,"},
                {"section": "[Verse 1]", "line": "Boardroom smiles while they cut them deals."},
                {"section": "[Pre-Chorus]", "line": "But ghosts don't sleep when they settle scores."},
            ],
        })

        self.assertEqual(
            payload["lyrics_lines"],
            [
                "[Intro]",
                "Concrete canyons closing in tight",
                "[Verse 1]",
                "They paved them blocks just to hide what's real,",
                "Boardroom smiles while they cut them deals.",
                "[Pre-Chorus]",
                "But ghosts don't sleep when they settle scores.",
            ],
        )

    def test_json_object_parser_repairs_raw_newline_and_missing_closer(self):
        raw = (
            '<think>bad wrapper</think>\n'
            '{"part_index":1,"sections":["[Intro]"],'
            '"lyrics_lines":[{"section_tag":"[Intro]","line":"First line\nSecond line"}]'
        )

        parsed = _json_object_from_text(raw)

        self.assertEqual(parsed["part_index"], 1)
        self.assertEqual(parsed["lyrics_lines"][0]["line"], "First line\nSecond line")

    def test_director_lyric_expansion_preserves_unique_section_tags(self):
        lines = [
            "[Intro]",
            "Death Row... East Coast... Closed doors...",
            "[Verse 1]",
            "They paved them blocks just to hide what's real,",
            "[Chorus]",
            "But ghosts don't sleep when they settle scores.",
        ]

        expanded, changed = album_crew_module._expand_director_lyrics_lines_to_fit(
            lines,
            {"vibe": "low-end rumble, sirens, West Coast weight"},
            min_words=90,
            min_lines=18,
            max_chars=1600,
        )

        self.assertTrue(changed)
        lyrics = "\n".join(expanded)
        self.assertEqual(lyrics.count("[Intro]"), 1)
        self.assertEqual(lyrics.count("[Verse 1]"), 1)
        self.assertEqual(lyrics.count("[Chorus]"), 1)
        stats = lyric_stats(lyrics)
        self.assertGreaterEqual(stats["word_count"], 90)
        self.assertGreaterEqual(stats["line_count"], 18)

    def test_director_section_groups_omits_empty_tail_group(self):
        groups = album_crew_module._director_section_groups([
            "[Intro]",
            "[Verse 1]",
            "[Chorus]",
            "[Break]",
        ])

        self.assertEqual(groups, [["[Intro]", "[Verse 1]", "[Chorus]"], ["[Break]"]])

    def test_agent_payload_distributes_tagless_lyrics_under_declared_sections(self):
        payload = _coerce_agent_lyrics_payload({
            "part_index": 1,
            "sections": ["[Intro]", "[Verse 1]", "[Pre-Chorus]"],
            "lyrics_lines": [
                "Death Row... East Coast... Closed doors...",
                "They paved them blocks just to hide what's real,",
                "Boardroom smiles while they cut them deals.",
                "Built ten towers off a fallen name",
                "Turned a man's life to a numbers game.",
                "Said it was peace, but I peeped that lie,",
            ],
        })

        self.assertEqual(payload["lyrics_lines"][0], "[Intro]")
        self.assertIn("[Verse 1]", payload["lyrics_lines"])
        self.assertIn("[Pre-Chorus]", payload["lyrics_lines"])
        self.assertNotIn("Death Row... East Coast... Closed doors...\n[Intro]", payload["lyrics"])

    def test_agent_payload_dict_lines_array_uses_declared_section(self):
        payload = _coerce_agent_lyrics_payload({
            "part_index": 1,
            "sections": ["[Intro]", "[Verse 1]"],
            "lyrics_lines": [
                {"tag": "[Intro]", "lines": ["Low-end rumble in the distance"]},
                {"tag": "[Verse 1]", "lines": ["They paved them blocks just to hide what's real,"]},
            ],
        })

        self.assertEqual(
            payload["lyrics_lines"],
            [
                "[Intro]",
                "Low-end rumble in the distance",
                "[Verse 1]",
                "They paved them blocks just to hide what's real,",
            ],
        )

    def test_lyrics_part_validator_rejects_text_before_first_section(self):
        issues = album_crew_module._validate_lyrics_part_payload(
            {
                "part_index": 1,
                "sections": ["[Intro]", "[Verse 1]"],
                "lyrics_lines": ["loose opening line", "[Intro]", "clean setup", "[Verse 1]", "clean verse"],
            },
            expected_sections=["[Intro]", "[Verse 1]"],
            forbidden_sections=[],
            expected_part_index=1,
        )

        self.assertIn("lyrics_before_first_section_tag", issues)

    def test_agent_payload_splits_colon_tag_lines_and_inserts_missing_sections(self):
        payload = _coerce_agent_lyrics_payload({
            "part_index": 1,
            "sections": ["[Intro]", "[Verse 1]", "[Pre-Chorus]"],
            "lyrics_lines": [
                "[Intro]: They paved them blocks just to hide what's real,",
                "Boardroom smiles while they cut them deals.",
                "[Verse 1]",
                "Same hands shaking be the ones that try-",
            ],
        })

        self.assertEqual(payload["lyrics_lines"][0], "[Intro]")
        self.assertEqual(payload["lyrics_lines"][1], "They paved them blocks just to hide what's real,")
        self.assertEqual(payload["lyrics_lines"].count("[Intro]"), 1)
        self.assertEqual(payload["lyrics_lines"].count("[Verse 1]"), 1)
        self.assertEqual(payload["lyrics_lines"].count("[Pre-Chorus]"), 1)

    def test_director_minimal_gate_rejects_duplicate_sections(self):
        report = album_crew_module._director_minimal_validate(
            {
                "title": "Lanterns on the Pier",
                "caption": "warm boom-bap, piano, clear lead vocal, hopeful hook, polished production",
                "lyrics": (
                    "[Intro]\nWarm piano enters\n"
                    "[Verse 1]\nWe open every window\n"
                    "[Chorus]\nLanterns call us home\n"
                    "[Verse 1]\nThe same verse returns by mistake\n"
                    "[Outro]\nHome again"
                ),
            },
            ["[Intro]", "[Verse 1]", "[Chorus]", "[Outro]"],
        )

        self.assertFalse(report["gate_passed"])
        self.assertIn("duplicate_section_tags:[Verse 1]", report["issues"])
        self.assertNotIn("metadata_or_credit_in_caption", report["issues"])

    def test_director_minimal_gate_rejects_underfilled_long_rap_lyrics(self):
        sections = ["[Intro]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Verse 2]", "[Break]", "[Bridge]", "[Final Chorus]", "[Outro]"]
        lyrics = "\n".join(
            line
            for section in sections
            for line in (section, "Concrete shadows move")
        )

        report = album_crew_module._director_minimal_validate(
            {
                "title": "Concrete Canyons",
                "duration": 240,
                "caption": "cinematic West Coast rap, hip-hop drums, male rap lead, clear chorus, polished modern mix",
                "style": "West Coast rap",
                "lyrics": lyrics,
            },
            sections,
        )

        self.assertFalse(report["gate_passed"])
        self.assertTrue(any(str(issue).startswith("lyrics_under_length:") for issue in report["issues"]))
        self.assertTrue(any(str(issue).startswith("lyrics_too_few_lines:") for issue in report["issues"]))
        self.assertGreaterEqual(report["lyric_duration_fit"]["min_words"], 340)
        self.assertGreaterEqual(report["lyric_duration_fit"]["min_lines"], 36)

    def test_director_minimal_gate_blocks_non_rap_payload_for_rap_request(self):
        sections = ["[Intro]", "[Verse 1]", "[Hook]", "[Verse 2]", "[Final Hook]", "[Outro]"]
        lyrics = "\n".join(
            line
            for section in sections
            for line in [section] + ["Concrete truth keeps knocking on the city door" for _ in range(8)]
        )

        report = album_crew_module._director_minimal_validate(
            {
                "title": "Concrete Canyons",
                "duration": 90,
                "caption": "cinematic orchestral strings, brass swells, taiko drums, epic score, polished mix",
                "style": "West Coast rap",
                "lyrics": lyrics,
            },
            sections,
            {"album_agent_genre_prompt": "West Coast rap"},
        )

        self.assertFalse(report["gate_passed"])
        self.assertIn("genre_intent_missing_rap_vocal", report["issues"])
        self.assertIn("genre_intent_missing_rap_groove", report["issues"])

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

        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["planning_engine"], "crewai_micro")
        self.assertTrue(result["crewai_used"])
        self.assertEqual(len(result["tracks"]), 1)
        self.assertTrue(any("AceJAM Agent call: Album Intake Agent" in line for line in result["logs"]))

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

        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["planning_engine"], "crewai_micro")
        self.assertEqual(result["tracks"][0]["title"], "Lantern Keys")
        self.assertEqual(result["tracks"][0]["producer_credit"], "Ada North")
        self.assertEqual(result["tracks"][0]["bpm"], 88)
        self.assertIn("Turn the lantern keys", result["tracks"][0]["lyrics"])
        self.assertTrue(result["input_contract_applied"])
        self.assertTrue(result["crewai_used"])

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

        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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

        preflight_patch, agent_patch = self._patch_director_agents()
        with preflight_patch, agent_patch:
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
        self.assertEqual(result["planning_engine"], "crewai_micro")
        self.assertFalse(result["toolbelt_fallback"])
        self.assertTrue(result["crewai_used"])
        self.assertIn("[Verse 1]", result["tracks"][0]["lyrics"])

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
                "tags": "modern pop, bright drums, deep bass, piano motif, clear lead vocal, emotional hook lift, warm analog texture, polished studio mix",
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
            "primary_genre",
            "drum_groove",
            "low_end_bass",
            "melodic_identity",
            "vocal_delivery",
            "arrangement_movement",
            "texture_space",
            "mix_master",
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
                "tags": "hip-hop, boom-bap drums, 808 bass, piano sample motif, male rap vocal, dynamic hook arrangement, gritty street texture, punchy polished rap mix",
                "duration": 240,
            },
            options={"track_duration": 240, "lyric_density": "dense", "structure_preset": "auto", "language": "en"},
            lyric_plan=lyric_plan,
        )
        output = SimpleNamespace(raw=json.dumps({
            "track_number": 1,
            "title": "Signal Room",
            "tags": "hip-hop, boom-bap drums, 808 bass, piano sample motif, male rap vocal, dynamic hook arrangement, gritty street texture, punchy polished rap mix",
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
                "tags": "hip-hop, boom-bap drums, 808 bass, piano sample motif, male rap vocal, dynamic hook arrangement, gritty street texture, punchy polished rap mix",
                "duration": 240,
            },
            options={"track_duration": 240, "lyric_density": "dense", "structure_preset": "auto", "language": "en"},
            lyric_plan=lyric_plan,
        )
        output = SimpleNamespace(raw=json.dumps({
            "track_number": 1,
            "title": "Concrete Signal",
            "tags": "hip-hop, boom-bap drums, 808 bass, piano sample motif, male rap vocal, dynamic hook arrangement, gritty street texture, punchy polished rap mix",
            "lyrics": "\n".join(lines),
            "duration": 240,
            "language": "en",
        }))

        ok, repaired_json = guardrail(output)
        repaired = json.loads(repaired_json)

        self.assertTrue(ok)
        effective_min = repaired["lyric_duration_fit"]["plan"].get("effective_min_lines") or repaired["lyric_duration_fit"]["plan"]["min_lines"]
        self.assertGreaterEqual(effective_min, 75)
        self.assertGreaterEqual(repaired["lyrics_line_count"], effective_min)
        self.assertGreaterEqual(len(repaired["caption_dimensions_covered"]), 8)

    def test_empty_crewai_payload_is_explicit_failure_marker(self):
        payload = json.loads(_empty_response_fallback_text("qwen-local").split("Final Answer: ", 1)[1])

        self.assertTrue(_is_empty_response_payload(payload))

    def test_empty_crewai_response_fails_loudly_without_toolbelt_fallback(self):
        with patch.object(album_crew_module, "preflight_album_agent_llm", return_value={
            "ok": True,
            "chat_ok": True,
            "warnings": [],
            "errors": [],
        }), patch.object(album_crew_module, "_agent_json_call", side_effect=album_crew_module.AceJamAgentError("empty response")), patch.object(album_crew_module, "_crewai_micro_block_call", side_effect=album_crew_module.AceJamAgentError("empty response")):
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
        self.assertEqual(result["planning_engine"], "crewai_micro")
        self.assertTrue(result["crewai_used"])
        self.assertFalse(result["toolbelt_fallback"])
        self.assertEqual(result["tracks"], [])
        self.assertTrue(any("CrewAI Micro Tasks planning failed loudly" in line for line in result["logs"]))

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
