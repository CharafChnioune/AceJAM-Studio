from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from prompt_kit import PROMPT_KIT_VERSION
from songwriting_toolkit import LYRIC_META_TAGS, TAG_TAXONOMY
from studio_core import ACE_STEP_CAPTION_CHAR_LIMIT, ACE_STEP_LYRICS_CHAR_LIMIT


ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION = "ace-step-track-prompt-template-2026-04-29"

CAPTION_DIMENSIONS = [
    "primary_genre",
    "drum_groove",
    "low_end_bass",
    "melodic_identity",
    "vocal_delivery",
    "arrangement_movement",
    "texture_space",
    "mix_master",
]

OUTPUT_SCHEMA_FIELDS = [
    "track_number",
    "artist_name",
    "title",
    "description",
    "tags",
    "tag_list",
    "lyrics",
    "bpm",
    "key_scale",
    "time_signature",
    "language",
    "duration",
    "song_model",
    "seed",
    "inference_steps",
    "guidance_scale",
    "shift",
    "infer_method",
    "sampler_mode",
    "audio_format",
    "quality_profile",
    "prompt_kit_version",
    "settings_policy_version",
    "settings_compliance",
    "quality_checks",
    "contract_compliance",
    "tag_coverage",
    "lyric_duration_fit",
    "caption_integrity",
    "payload_gate_status",
    "repair_actions",
    "lyrics_word_count",
    "lyrics_line_count",
    "lyrics_char_count",
    "section_count",
    "hook_count",
    "caption_dimensions_covered",
]


def compact_full_tag_library() -> dict[str, Any]:
    return {
        "caption_dimensions": CAPTION_DIMENSIONS,
        "tag_taxonomy": TAG_TAXONOMY,
        "lyric_meta_tags": LYRIC_META_TAGS,
        "caption_selection_rule": (
            "Use the full library for choosing tags, then select only compact terms that cover every "
            "required caption dimension. Do not dump every tag into the final caption."
        ),
    }


@dataclass(frozen=True)
class AceStepTrackPromptTemplate:
    """Shared prompt contract for album track agents that emit ACE-Step payloads."""

    version: str = ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION

    def render(
        self,
        *,
        user_album_contract: dict[str, Any],
        ace_step_payload_contract: dict[str, Any],
        lyric_length_plan: dict[str, Any],
        language_preset: dict[str, Any],
        blueprint: dict[str, Any],
        album_bible: dict[str, Any] | None = None,
    ) -> str:
        blocks = {
            "TEMPLATE_VERSION": self.version,
            "PROMPT_KIT_VERSION": PROMPT_KIT_VERSION,
            "USER_ALBUM_CONTRACT": user_album_contract or {},
            "ACE_STEP_NON_NEGOTIABLES": {
                "caption_max_chars": ACE_STEP_CAPTION_CHAR_LIMIT,
                "lyrics_max_chars": ACE_STEP_LYRICS_CHAR_LIMIT,
                "caption_role": "short musical portrait: genre, groove, instruments, vocal style, mood, arrangement energy, production mix",
                "lyrics_role": "temporal script: concise section/performance tags plus actual lyrics only; human craft with concrete scenes, coherent metaphor, memorable hooks, and genre-natural flow",
                "caption_forbidden": [
                    "lyrics",
                    "section tags",
                    "track headers",
                    "JSON/prose scaffolding",
                    "BPM/key/duration/model/seed",
                    "full user prompt",
                    "CrewAI/tool instructions",
                ],
                "lyrics_forbidden": [
                    "planning prose",
                    "thought/reasoning text",
                    "metadata blocks",
                    "placeholder lines",
                    "escaped literal \\n sequences",
                    "generic AI slogans like neon dreams, fire inside, we rise",
                ],
                "consistency_rule": "Caption and lyric section tags must agree on instruments, vocal role, mood, and energy.",
                "section_tag_rule": "Keep section tags concise; put complex sound detail in caption.",
                "craft_rule": "Rap needs cadence/internal rhyme/bar momentum; sung genres need vowel-friendly emotional clarity; EDM/instrumental should use sparse motifs instead of forced literary verses.",
            },
            "FULL_TAG_LIBRARY": compact_full_tag_library(),
            "LYRIC_LENGTH_PLAN": lyric_length_plan or {},
            "LANGUAGE_PRESET": language_preset or {},
            "TRACK_BLUEPRINT": blueprint or {},
            "ALBUM_BIBLE": album_bible or {},
            "ACE_STEP_PAYLOAD_CONTRACT": ace_step_payload_contract or {},
            "OUTPUT_SCHEMA": {
                "type": "single_json_object",
                "fields": OUTPUT_SCHEMA_FIELDS,
                "strict_rules": [
                    "Return exactly one track JSON object.",
                    "Preserve locked title, producer credit, BPM, style, vibe, narrative, and required phrases exactly.",
                    "Use tag_list for selected compact terms; use tags/caption as a comma-separated caption under 512 chars.",
                    "Lyrics must fit the length plan and stay under 4096 chars.",
                    "Populate all counter fields with deterministic values after using LyricCounterTool.",
                ],
            },
            "SELF_CHECK": [
                "Call TagLibraryTool or AceStepPromptContractTool before final JSON.",
                "Call LyricCounterTool with the final lyrics and copy its counts into the JSON.",
                "Call TagCoverageTool with caption/tags/tag_list; repair missing dimensions.",
                "Call CaptionIntegrityTool; repair leakage before final JSON.",
                "Call PayloadGateTool; if it reports fail, repair lyrics/caption/tags before final JSON.",
                "Reject your own output if lyrics_line_count or lyrics_word_count is below the length plan.",
                "Reject your own output if caption leaks prompt text or metadata.",
                "Reject your own output if escaped \\n remains in lyrics.",
            ],
        }
        return "\n\n".join(
            f"## {name}\n{json.dumps(value, ensure_ascii=True, indent=2)}"
            for name, value in blocks.items()
        )


def render_track_prompt_template(**kwargs: Any) -> str:
    return AceStepTrackPromptTemplate().render(**kwargs)
