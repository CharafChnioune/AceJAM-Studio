"""
Album generation using AceJAM's local-agent planner plus deterministic tools.

AceJAM Direct is the default album runtime. CrewAI Micro Tasks is available as
an experimental wrapper around the same tiny agent calls, delimiter parser, and
deterministic gates. Legacy large CrewAI constructors remain for import
compatibility only.
"""

from __future__ import annotations

import builtins
import contextlib
import errno
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Tuple


def _safe_print(*args: Any, **kwargs: Any) -> None:
    try:
        builtins.print(*args, **kwargs)
    except (BrokenPipeError, OSError) as exc:
        if isinstance(exc, BrokenPipeError) or getattr(exc, "errno", None) == errno.EPIPE:
            return
        raise


print = _safe_print

# CrewAI telemetry attempts to attach custom Memory objects as OpenTelemetry
# attributes. AceJAM uses local compact monitor logs instead.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from pydantic import BaseModel, ConfigDict, Field

from ace_step_track_prompt_template import (
    ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION,
    compact_full_tag_library,
    render_track_prompt_template,
)
from album_quality_gate import (
    build_genre_intent_contract,
    build_lyrical_craft_contract,
    build_producer_grade_sonic_contract,
    evaluate_album_payload_quality,
    evaluate_genre_adherence,
    lyric_craft_gate,
    lyric_density_gate,
    producer_grade_readiness,
)
from local_llm import (
    chat_completion as local_llm_chat_completion,
    embed as local_llm_embed,
    lmstudio_api_base_url,
    lmstudio_load_model,
    lmstudio_model_catalog,
    normalize_provider,
    ollama_host,
    planner_llm_options_for_provider,
    planner_llm_settings_from_payload,
    provider_label,
    test_model as local_llm_test_model,
)
from songwriting_toolkit import (
    ALBUM_FINAL_MODEL,
    ALBUM_MODEL_PORTFOLIO_MODELS,
    TAG_TAXONOMY,
    album_model_portfolio,
    build_album_plan,
    choose_song_model,
    lyric_length_plan,
    lyric_stats,
    make_crewai_tools,
    normalize_album_tracks,
    parse_duration_seconds,
    sanitize_artist_references,
    toolkit_payload,
)
from prompt_kit import (
    DEFAULT_NEGATIVE_CONTROL,
    PROMPT_KIT_VERSION,
    infer_genre_modules,
    is_sparse_lyric_genre,
    kit_metadata_defaults,
    language_preset,
    prompt_kit_payload,
    prompt_kit_system_block,
    section_map_for,
)
from studio_core import (
    ACE_STEP_LYRICS_CHAR_LIMIT,
    ACE_STEP_LYRICS_SAFE_HEADROOM,
    ACE_STEP_LYRICS_SOFT_TARGET_MAX,
    DEFAULT_BPM,
    DEFAULT_KEY_SCALE,
    DEFAULT_QUALITY_PROFILE,
    clamp_int,
    docs_best_model_settings,
    normalize_quality_profile,
)
from user_album_contract import (
    USER_ALBUM_CONTRACT_VERSION,
    apply_user_album_contract_to_track,
    apply_user_album_contract_to_tracks,
    contract_prompt_context,
    contract_track,
    extract_user_album_contract,
)


OLLAMA_BASE_URL = ollama_host()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CREWAI_LEGACY_MEMORY_DIR = DATA_DIR / "crewai_memory"
CREWAI_MEMORY_DIR = DATA_DIR / "crewai_album_memory_v2"
DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL = os.environ.get(
    "ACEJAM_ALBUM_PLANNER_OLLAMA_MODEL",
    "charaf/qwen3.6-27b-abliterated-mlx:mxfp4-instruct-general",
)
DEFAULT_ALBUM_EMBEDDING_MODEL = os.environ.get(
    "ACEJAM_ALBUM_EMBEDDING_MODEL",
    "charaf/qwen3-vl-embedding-8b:latest",
)
ALBUM_EMBEDDING_FALLBACK_MODELS = [
    DEFAULT_ALBUM_EMBEDDING_MODEL,
    "mxbai-embed-large:latest",
    "nomic-embed-text:latest",
]
CREWAI_LLM_TIMEOUT_SECONDS = int(os.environ.get("ACEJAM_CREWAI_LLM_TIMEOUT_SECONDS", "86400"))
CREWAI_EMPTY_RESPONSE_RETRIES = int(os.environ.get("ACEJAM_CREWAI_EMPTY_RESPONSE_RETRIES", "1"))
CREWAI_EMPTY_RESPONSE_RETRY_DELAY = float(os.environ.get("ACEJAM_CREWAI_EMPTY_RESPONSE_RETRY_DELAY", "8"))
CREWAI_AGENT_MAX_ITER = int(os.environ.get("ACEJAM_CREWAI_AGENT_MAX_ITER", "80"))
CREWAI_AGENT_MAX_RETRY_LIMIT = int(os.environ.get("ACEJAM_CREWAI_AGENT_MAX_RETRY_LIMIT", "8"))
CREWAI_TASK_MAX_RETRIES = int(os.environ.get("ACEJAM_CREWAI_TASK_MAX_RETRIES", "8"))
CREWAI_LLM_MAX_TOKENS = int(os.environ.get("ACEJAM_CREWAI_LLM_MAX_TOKENS", "12000"))
CREWAI_LLM_CONTEXT_WINDOW = int(os.environ.get("ACEJAM_CREWAI_LLM_CONTEXT_WINDOW", "65536"))
CREWAI_LLM_NUM_PREDICT = int(os.environ.get("ACEJAM_CREWAI_LLM_NUM_PREDICT", str(CREWAI_LLM_MAX_TOKENS)))
CREWAI_LMSTUDIO_MAX_TOKENS = int(os.environ.get("ACEJAM_CREWAI_LMSTUDIO_MAX_TOKENS", str(CREWAI_LLM_MAX_TOKENS)))
CREWAI_LMSTUDIO_DISABLE_THINKING = os.environ.get("ACEJAM_CREWAI_LMSTUDIO_DISABLE_THINKING", "1").lower() in {"1", "true", "yes"}
CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE = os.environ.get("ACEJAM_CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE", "/no_think").strip()
CREWAI_LMSTUDIO_NO_THINK_PREFILL = os.environ.get("ACEJAM_CREWAI_LMSTUDIO_NO_THINK_PREFILL", "")
CREWAI_LMSTUDIO_PIN_CONTEXT = os.environ.get("ACEJAM_CREWAI_LMSTUDIO_PIN_CONTEXT", "1").lower() in {"1", "true", "yes"}
CREWAI_LMSTUDIO_CRASH_RETRIES = int(os.environ.get("ACEJAM_CREWAI_LMSTUDIO_CRASH_RETRIES", "1"))
CREWAI_MEMORY_CONTENT_LIMIT = int(os.environ.get("ACEJAM_CREWAI_MEMORY_CONTENT_LIMIT", "1500"))
CREWAI_PROMPT_BUDGET_CHARS = int(os.environ.get("ACEJAM_CREWAI_PROMPT_BUDGET_CHARS", "24000"))
CREWAI_RESPECT_CONTEXT_WINDOW = os.environ.get("ACEJAM_CREWAI_RESPECT_CONTEXT_WINDOW", "0").lower() in {"1", "true", "yes"}
CREWAI_DEBUG_LLM_RESPONSES = os.environ.get("ACEJAM_CREWAI_DEBUG_LLM_RESPONSES", "0").lower() in {"1", "true", "yes"}
CREWAI_VERBOSE = os.environ.get("ACEJAM_CREWAI_VERBOSE", "1").lower() in {"1", "true", "yes"}
CREWAI_CAPTURE_STDIO = os.environ.get("ACEJAM_CREWAI_CAPTURE_STDIO", "1").lower() in {"1", "true", "yes"}
CREWAI_LIVE_TOOLS = os.environ.get("ACEJAM_CREWAI_LIVE_TOOLS", "0").lower() in {"1", "true", "yes"}
CREWAI_LOG_DIR = DATA_DIR / "crewai_logs"
CREWAI_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
CREWAI_LOG_DIR.mkdir(parents=True, exist_ok=True)
ALBUM_FINAL_DOCS_BEST = docs_best_model_settings(ALBUM_FINAL_MODEL)

ACEJAM_AGENT_ENGINE = "acejam_agents"
CREWAI_MICRO_AGENT_ENGINE = "crewai_micro"
ACTIVE_ALBUM_AGENT_ENGINES = {ACEJAM_AGENT_ENGINE, CREWAI_MICRO_AGENT_ENGINE}
ALBUM_AGENT_ENGINE_LABELS = {
    ACEJAM_AGENT_ENGINE: "AceJAM Direct",
    CREWAI_MICRO_AGENT_ENGINE: "CrewAI Micro Tasks",
}
ACEJAM_AGENT_JSON_RETRIES = int(os.environ.get("ACEJAM_AGENT_JSON_RETRIES", "2"))
ACEJAM_AGENT_BLOCK_RETRIES = int(os.environ.get("ACEJAM_AGENT_BLOCK_RETRIES", str(ACEJAM_AGENT_JSON_RETRIES)))
ACEJAM_AGENT_EMPTY_RETRIES = int(os.environ.get("ACEJAM_AGENT_EMPTY_RETRIES", "1"))
ACEJAM_AGENT_GATE_REPAIR_RETRIES = int(os.environ.get("ACEJAM_AGENT_GATE_REPAIR_RETRIES", "8"))
ALBUM_WRITER_MODE_PER_TRACK = "per_track_writer_loop"
ALBUM_WRITER_MODE_DEFAULT = os.environ.get("ACEJAM_ALBUM_WRITER_MODE", ALBUM_WRITER_MODE_PER_TRACK).strip() or ALBUM_WRITER_MODE_PER_TRACK
ALBUM_TRACK_GATE_REPAIR_RETRIES = max(0, min(3, int(os.environ.get("ACEJAM_ALBUM_TRACK_GATE_REPAIR_RETRIES", "3"))))
ACEJAM_AGENT_TEMPERATURE = float(os.environ.get("ACEJAM_AGENT_TEMPERATURE", "0.25"))
ACEJAM_AGENT_TOP_P = float(os.environ.get("ACEJAM_AGENT_TOP_P", "0.9"))
ACEJAM_AGENT_MEMORY_DEFAULT = os.environ.get("ACEJAM_AGENT_MEMORY_DEFAULT", "1").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_RETRIEVAL_TOP_K = int(os.environ.get("ACEJAM_AGENT_RETRIEVAL_TOP_K", "5"))
ACEJAM_AGENT_CONTEXT_CHUNK_CHARS = int(os.environ.get("ACEJAM_AGENT_CONTEXT_CHUNK_CHARS", "1400"))
ACEJAM_AGENT_OLLAMA_JSON_FORMAT = os.environ.get("ACEJAM_AGENT_OLLAMA_JSON_FORMAT", "0").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_SPLIT_TRACK_FLOW = os.environ.get("ACEJAM_AGENT_SPLIT_TRACK_FLOW", "1").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_MICRO_SETTINGS_FLOW = os.environ.get("ACEJAM_AGENT_MICRO_SETTINGS_FLOW", "1").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_LYRIC_PARTS = max(1, int(os.environ.get("ACEJAM_AGENT_LYRIC_PARTS", "4")))
ACEJAM_AGENT_ALBUM_BIBLE_LLM = os.environ.get("ACEJAM_AGENT_ALBUM_BIBLE_LLM", "0").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_BLUEPRINT_LLM = os.environ.get("ACEJAM_AGENT_BLUEPRINT_LLM", "0").lower() in {"1", "true", "yes"}
ACEJAM_ALBUM_DIRECTOR_VERSION = "acejam-album-director-prompt-first-v2-2026-05-01"
ACEJAM_PRINT_AGENT_IO_DEFAULT = os.environ.get("ACEJAM_PRINT_AGENT_IO", "1").lower() in {"1", "true", "yes", "on"}
ACEJAM_PROMPT_KIT_MD_PATH = Path(
    os.environ.get(
        "ACEJAM_PROMPT_KIT_MD_PATH",
        "/Users/charafchnioune/Desktop/code/ACE-Step_Multilingual_Hit_Prompt_Kit_Full.md",
    )
)

ACE_STEP_PAYLOAD_CONTRACT_VERSION = "ace-step-track-payload-contract-2026-04-29"


def normalize_album_agent_engine(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s-]+", "_", text)
    # CrewAI Micro Tasks is the album-wizard default. Each track field is filled
    # by a real CrewAI Agent/Task running through the crewai library so users
    # get a visible multi-agent run with full prompt-kit knowledge injection.
    # The legacy `acejam_agents` direct-Ollama path stays available for
    # diagnostic / fallback use when the user explicitly opts in.
    aliases = {
        "": CREWAI_MICRO_AGENT_ENGINE,
        "crewai": CREWAI_MICRO_AGENT_ENGINE,
        "crew_ai": CREWAI_MICRO_AGENT_ENGINE,
        "crewai_micro": CREWAI_MICRO_AGENT_ENGINE,
        "micro_crewai": CREWAI_MICRO_AGENT_ENGINE,
        "crewai_micro_tasks": CREWAI_MICRO_AGENT_ENGINE,
        "legacy_crewai": CREWAI_MICRO_AGENT_ENGINE,
        "acejam": ACEJAM_AGENT_ENGINE,
        "acejam_agent": ACEJAM_AGENT_ENGINE,
        "acejam_agents": ACEJAM_AGENT_ENGINE,
        "acejam_direct": ACEJAM_AGENT_ENGINE,
        "direct": ACEJAM_AGENT_ENGINE,
        "editable_plan": ACEJAM_AGENT_ENGINE,
    }
    return aliases.get(text, CREWAI_MICRO_AGENT_ENGINE)


def album_agent_engine_label(engine: Any) -> str:
    return ALBUM_AGENT_ENGINE_LABELS.get(normalize_album_agent_engine(engine), ALBUM_AGENT_ENGINE_LABELS[ACEJAM_AGENT_ENGINE])

ACE_STEP_CAPTION_DIMENSIONS = [
    "primary_genre",
    "drum_groove",
    "low_end_bass",
    "melodic_identity",
    "vocal_delivery",
    "arrangement_movement",
    "texture_space",
    "mix_master",
]

AGENT_EXACT_RESPONSE_SCHEMAS: dict[str, dict[str, Any]] = {
    "album_intake_payload": {
        "keys": ["album_title", "one_sentence_concept", "style_guardrails", "track_roles"],
        "example": {
            "album_title": "",
            "one_sentence_concept": "",
            "style_guardrails": [],
            "track_roles": [],
        },
    },
    "track_concept_payload": {
        "keys": ["title", "description", "style", "vibe", "narrative", "required_phrases"],
        "example": {
            "title": "",
            "description": "",
            "style": "",
            "vibe": "",
            "narrative": "",
            "required_phrases": [],
        },
    },
    "tag_agent_payload": {
        "keys": ["tag_list", "tags", "caption_dimensions_covered"],
        "example": {
            "tag_list": [],
            "tags": "",
            "caption_dimensions_covered": ACE_STEP_CAPTION_DIMENSIONS,
        },
    },
    "bpm_agent_payload": {"keys": ["bpm"], "example": {"bpm": DEFAULT_BPM}},
    "key_agent_payload": {"keys": ["key_scale"], "example": {"key_scale": DEFAULT_KEY_SCALE}},
    "time_signature_agent_payload": {"keys": ["time_signature"], "example": {"time_signature": "4"}},
    "duration_agent_payload": {"keys": ["duration"], "example": {"duration": 240}},
    "section_map_payload": {
        "keys": ["section_map", "rationale"],
        "example": {
            "section_map": ["[Intro]", "[Verse 1]", "[Chorus]", "[Verse 2]", "[Break]", "[Bridge]", "[Final Chorus]", "[Outro]"],
            "rationale": "",
        },
    },
    "hook_payload": {
        "keys": ["hook_title", "hook_lines", "hook_promise"],
        "example": {"hook_title": "", "hook_lines": [], "hook_promise": ""},
    },
    "lyric_craft_repair_payload": {
        "keys": ["sections", "lyrics_lines", "craft_fixes"],
        "example": {"sections": [], "lyrics_lines": [], "craft_fixes": []},
    },
    "caption_agent_payload": {"keys": ["caption"], "example": {"caption": ""}},
    "performance_agent_payload": {
        "keys": ["performance_brief", "negative_control", "genre_profile"],
        "example": {"performance_brief": "", "negative_control": "", "genre_profile": ""},
    },
    "final_payload": {
        "keys": [
            "track_number",
            "title",
            "description",
            "caption",
            "tags",
            "tag_list",
            "lyrics_lines",
            "bpm",
            "key_scale",
            "time_signature",
            "duration",
            "language",
            "performance_brief",
            "quality_checks",
        ],
        "example": {
            "track_number": 1,
            "title": "",
            "description": "",
            "caption": "",
            "tags": "",
            "tag_list": [],
            "lyrics_lines": [],
            "bpm": DEFAULT_BPM,
            "key_scale": DEFAULT_KEY_SCALE,
            "time_signature": "4",
            "duration": 240,
            "language": "en",
            "performance_brief": "",
            "quality_checks": {},
        },
    },
}

AGENT_BLOCK_RESPONSE_SCHEMAS: dict[str, dict[str, Any]] = {
    "album_intake_payload": {
        "fields": ["album_title", "one_sentence_concept", "style_guardrails", "track_roles"],
        "list_fields": {"style_guardrails", "track_roles"},
        "required_nonempty": {"album_title", "one_sentence_concept"},
    },
    "track_concept_payload": {
        "fields": ["title", "description", "style", "vibe", "narrative", "required_phrases"],
        "list_fields": {"required_phrases"},
        "required_nonempty": {"title", "description", "style"},
    },
    "tag_agent_payload": {
        "fields": ["tag_list", "caption_dimensions_covered"],
        "list_fields": {"tag_list", "caption_dimensions_covered"},
        "required_nonempty": {"tag_list"},
        "derived_fields": {"tags": "tag_list_csv"},
    },
    "bpm_agent_payload": {
        "fields": ["bpm"],
        "number_fields": {"bpm"},
        "required_nonempty": {"bpm"},
    },
    "key_agent_payload": {
        "fields": ["key_scale"],
        "required_nonempty": {"key_scale"},
    },
    "time_signature_agent_payload": {
        "fields": ["time_signature"],
        "required_nonempty": {"time_signature"},
    },
    "duration_agent_payload": {
        "fields": ["duration"],
        "number_fields": {"duration"},
        "required_nonempty": {"duration"},
    },
    "section_map_payload": {
        "fields": ["section_map", "rationale"],
        "list_fields": {"section_map"},
        "required_nonempty": {"section_map"},
    },
    "hook_payload": {
        "fields": ["hook_title", "hook_lines", "hook_promise"],
        "list_fields": {"hook_lines"},
        "required_nonempty": {"hook_lines"},
    },
    "lyric_craft_repair_payload": {
        "fields": ["sections", "lyrics_lines", "craft_fixes"],
        "list_fields": {"sections", "lyrics_lines", "craft_fixes"},
        "required_nonempty": {"sections", "lyrics_lines"},
    },
    "caption_agent_payload": {
        "fields": ["caption"],
        "required_nonempty": {"caption"},
    },
    "performance_agent_payload": {
        "fields": ["performance_brief", "negative_control", "genre_profile"],
        "required_nonempty": {"performance_brief"},
    },
    "final_payload": {
        "fields": [
            "track_number",
            "title",
            "description",
            "caption",
            "tag_list",
            "lyrics_lines",
            "bpm",
            "key_scale",
            "time_signature",
            "duration",
            "language",
            "performance_brief",
        ],
        "list_fields": {"tag_list", "lyrics_lines"},
        "number_fields": {"track_number", "bpm", "duration"},
        "derived_fields": {"tags": "tag_list_csv", "quality_checks": "deterministic_block_payload"},
    },
}

CAPTION_METADATA_RE = re.compile(
    r"\b(?:\d{2,3}\s*bpm|bpm\s*[:=]|\d+\/\d+\s*time|time\s*signature|"
    r"[A-G](?:#|b|♯|♭)?\s+(?:major|minor)|duration|seconds?|model|seed|producer|produced by|"
    r"prod\.|production\s+(?:by|credit))\b",
    re.I,
)

LANG_NAMES = {
    "en": "English", "ar": "Arabic", "az": "Azerbaijani", "bg": "Bulgarian",
    "bn": "Bengali", "ca": "Catalan", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "es": "Spanish", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "la": "Latin",
    "lt": "Lithuanian", "ms": "Malay", "ne": "Nepali", "nl": "Dutch",
    "no": "Norwegian", "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sr": "Serbian",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese",
    "instrumental": "Instrumental",
}


class _AceJamStructuredModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AlbumBibleModel(_AceJamStructuredModel):
    concept: str = ""
    arc: Any = ""
    motifs: list[Any] = Field(default_factory=list)
    sonic_palette: Any = ""
    continuity_rules: list[Any] = Field(default_factory=list)


class TrackBlueprintModel(_AceJamStructuredModel):
    track_number: int = 0
    artist_name: str = ""
    title: str = ""
    description: str = ""
    tags: Any = ""
    bpm: Any = 90
    key_scale: str = "C minor"
    time_signature: Any = "4"
    duration: Any = 180
    hook_promise: str = ""
    performance_brief: str = ""


class AlbumBiblePayloadModel(_AceJamStructuredModel):
    album_bible: AlbumBibleModel = Field(default_factory=AlbumBibleModel)
    tracks: list[TrackBlueprintModel] = Field(default_factory=list)


class TrackProductionPayloadModel(_AceJamStructuredModel):
    track_number: int = 0
    artist_name: str = ""
    title: str = ""
    description: str = ""
    tags: Any = ""
    lyrics: str = ""
    bpm: Any = 90
    key_scale: str = "C minor"
    time_signature: Any = "4"
    language: str = "en"
    duration: Any = 180
    song_model: str = ALBUM_FINAL_MODEL
    seed: Any = -1
    inference_steps: Any = ALBUM_FINAL_DOCS_BEST["inference_steps"]
    guidance_scale: Any = ALBUM_FINAL_DOCS_BEST["guidance_scale"]
    shift: Any = ALBUM_FINAL_DOCS_BEST["shift"]
    infer_method: str = "ode"
    sampler_mode: str = ALBUM_FINAL_DOCS_BEST["sampler_mode"]
    audio_format: str = ALBUM_FINAL_DOCS_BEST["audio_format"]
    auto_score: bool = False
    auto_lrc: bool = False
    return_audio_codes: bool = False
    save_to_library: bool = True
    tool_notes: Any = ""
    production_team: dict[str, Any] = Field(default_factory=dict)
    model_render_notes: Any = Field(default_factory=dict)
    settings_policy_version: str = ""
    settings_compliance: dict[str, Any] = Field(default_factory=dict)
    quality_checks: dict[str, Any] = Field(default_factory=dict)
    contract_compliance: dict[str, Any] = Field(default_factory=dict)
    tag_coverage: dict[str, Any] = Field(default_factory=dict)
    lyric_duration_fit: dict[str, Any] = Field(default_factory=dict)
    caption_integrity: dict[str, Any] = Field(default_factory=dict)
    payload_gate_status: str = ""
    repair_actions: list[Any] = Field(default_factory=list)
    lyrics_word_count: int = 0
    lyrics_line_count: int = 0
    lyrics_char_count: int = 0
    section_count: int = 0
    hook_count: int = 0
    caption_dimensions_covered: list[Any] = Field(default_factory=list)


def _clip_text(value: Any, limit: int = CREWAI_MEMORY_CONTENT_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _monitor_preview(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return _clip_text(text, limit)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _print_agent_io(options: dict[str, Any], label: str, payload: Any) -> None:
    if not _truthy((options or {}).get("print_agent_io"), ACEJAM_PRINT_AGENT_IO_DEFAULT):
        return
    text = payload if isinstance(payload, str) else json.dumps(_debug_jsonable(payload), ensure_ascii=False, indent=2)
    print(f"[acejam_agent_io][BEGIN {label} chars={len(text)}]", flush=True)
    print(text, flush=True)
    print(f"[acejam_agent_io][END {label}]", flush=True)


def _safe_job_id(value: Any) -> str:
    job_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "manual").strip())
    return (job_id or "manual")[:80]


def crewai_output_log_path(job_id: Any) -> Path:
    return CREWAI_LOG_DIR / f"album_plan_{_safe_job_id(job_id)}.json"


def _crewai_step_callback(logs: list[str] | None = None):
    def _callback(step: Any) -> None:
        kind = type(step).__name__
        tool = _monitor_preview(getattr(step, "tool", "") or getattr(step, "tool_name", ""), 80)
        tool_suffix = f" tool={tool}" if tool else ""
        line = f"CrewAI step: {kind}{tool_suffix}"
        if logs is not None:
            logs.append(line)
        else:
            print(f"[album_crew][crewai] {line}", flush=True)

    return _callback


def _safe_crewai_output_preview(raw: Any, limit: int = 220) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    cleaned = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        lowered = cleaned.lower()
        if '"lyrics"' in lowered or "[verse" in lowered or "[chorus" in lowered or "[outro" in lowered:
            return "content_preview_redacted=lyrics_or_full_track_text"
        return _monitor_preview(cleaned, limit)
    if isinstance(parsed, dict):
        has_lyrics = "lyrics" in parsed or "lyrics_content" in parsed
        pieces = []
        for key in ("track_number", "title", "duration", "bpm"):
            if parsed.get(key) not in (None, ""):
                pieces.append(f"{key}={_monitor_preview(parsed.get(key), 70)}")
        if has_lyrics:
            pieces.append("lyrics=redacted")
        if pieces:
            return ", ".join(pieces)
    return _monitor_preview(cleaned, limit)


def _crewai_task_callback(logs: list[str] | None = None):
    def _callback(output: Any) -> None:
        agent = _monitor_preview(getattr(output, "agent", ""), 90) or "agent"
        raw = str(getattr(output, "raw", "") or "")
        preview = _safe_crewai_output_preview(raw)
        suffix = f" preview={preview}" if preview else ""
        line = f"CrewAI task completed: agent={agent} output_chars={len(raw)}{suffix}"
        if logs is not None:
            logs.append(line)
        else:
            print(f"[album_crew][crewai] {line}", flush=True)

    return _callback


def _kickoff_crewai_compact(crew: Any, logs: list[str], label: str, output_log_file: str | None = None) -> Any:
    if not CREWAI_CAPTURE_STDIO:
        return crew.kickoff()
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return crew.kickoff()
    finally:
        captured = "\n".join(part for part in (stdout.getvalue(), stderr.getvalue()) if part)
        if captured.strip():
            lines = len(captured.splitlines())
            chars = len(captured)
            suffix = f"; full verbose log: {output_log_file}" if output_log_file else ""
            logs.append(f"CrewAI verbose captured for {label}: {lines} lines, {chars} chars{suffix}.")


def _compact_json(value: Any, limit: int = CREWAI_MEMORY_CONTENT_LIMIT) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, default=str, separators=(",", ":"))
    except TypeError:
        text = str(value)
    return _clip_text(text, limit)


def _debug_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _debug_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_debug_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_llm_debug_jsonl(path_value: Any, payload: dict[str, Any]) -> None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return
    try:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **_debug_jsonable(payload),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    except Exception as exc:
        print(f"[album_crew][llm-debug-log][warning] {type(exc).__name__}: {_monitor_preview(exc, 220)}", flush=True)


def _small_scalar(value: Any, limit: int = 160) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _clip_text(value, limit)


def _prompt_within_budget(*parts: Any) -> bool:
    text = "\n".join(str(part or "") for part in parts)
    return len(text) <= CREWAI_PROMPT_BUDGET_CHARS


def _task_output_json_dict(output: Any) -> dict[str, Any]:
    tasks_output = getattr(output, "tasks_output", None)
    if isinstance(tasks_output, list) and tasks_output:
        try:
            return _task_output_json_dict(tasks_output[-1])
        except Exception:
            pass
    json_dict = getattr(output, "json_dict", None)
    if isinstance(json_dict, dict):
        return json_dict
    pydantic_obj = getattr(output, "pydantic", None)
    if pydantic_obj is not None:
        if hasattr(pydantic_obj, "model_dump"):
            dumped = pydantic_obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(pydantic_obj, "dict"):
            dumped = pydantic_obj.dict()
            if isinstance(dumped, dict):
                return dumped
    raw = getattr(output, "raw", None)
    return _json_object_from_text(str(raw if raw is not None else output))


def _task_output_raw_text(output: Any) -> str:
    parts: list[str] = []
    tasks_output = getattr(output, "tasks_output", None)
    if isinstance(tasks_output, list):
        for item in tasks_output:
            raw = getattr(item, "raw", None)
            if raw is not None:
                parts.append(str(raw))
    raw = getattr(output, "raw", None)
    if raw is not None:
        parts.append(str(raw))
    if not parts and output is not None:
        parts.append(str(output))
    return "\n\n".join(part for part in parts if part)


def _lyric_like_text(raw: str) -> str:
    text = _strip_thinking_blocks(raw)
    match = re.search(r"\[(?:Intro|Verse|Pre-Chorus|Chorus|Bridge|Final Chorus|Outro|Vocal responses?)[^\]]*\]", text, flags=re.I)
    if match:
        snippet = text[match.start():]
        lines: list[str] = []
        for line in snippet.splitlines():
            stripped = line.strip()
            section_match = re.fullmatch(r"[*_`~\s]*(\[[^\]]+\])[*_`~\s]*", stripped)
            if section_match:
                line = section_match.group(1)
                stripped = line
            marker = re.sub(r"^[\s>*_`#-]+", "", stripped)
            marker = re.sub(r"[\s*_`#-]+$", "", marker).replace("**", "").strip()
            if lines and (
                stripped.startswith("```")
                or re.match(r"(?i)^(metadata|ace[-\s]?step metadata|bpm|key(?:[_\s-]?scale)?|time(?:[_\s-]?signature)?|language|duration|song(?:[_\s-]?model)?|quality(?:[_\s-]?profile)?|seed|inference(?:[_\s-]?steps)?|guidance(?:[_\s-]?scale)?|shift|infer(?:[_\s-]?method)?|sampler(?:[_\s-]?mode)?|audio(?:[_\s-]?format)?|model(?:[_\s-]?advice)?|title|artist|description|tags)\s*:", marker)
                or re.match(r"(?i)^final answer\s*:", marker)
                or stripped.startswith("{")
            ):
                break
            if stripped.startswith("```"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()
    return text


def _lyrics_richness_score(lyrics: Any, duration: float, density: str, structure_preset: str, genre_hint: str) -> float:
    text = str(lyrics or "")
    stats = lyric_stats(text)
    plan = lyric_length_plan(duration, density, structure_preset, genre_hint)
    lowered = text.lower()
    score = 0.0
    score += min(1.0, stats.get("word_count", 0) / max(1, int(plan.get("min_words") or 1))) * 4
    score += min(1.0, stats.get("line_count", 0) / max(1, int(plan.get("min_lines") or 1))) * 3
    score += min(1.0, stats.get("section_count", 0) / max(1, len(plan.get("sections") or []))) * 2
    if "[verse" in lowered:
        score += 1
    if "[chorus" in lowered or "[hook" in lowered or "[refrain" in lowered:
        score += 1
    return score


def _ace_step_track_payload_contract(
    lyric_plan: dict[str, Any],
    language: str,
    blueprint: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Machine-readable contract given to CrewAI before track production."""
    return {
        "version": ACE_STEP_PAYLOAD_CONTRACT_VERSION,
        "source_docs": {
            "caption": "ACE-Step GenerationParams.caption: max 512 chars; style, emotion, instruments, timbre only.",
            "lyrics": "ACE-Step GenerationParams.lyrics: max 4096 chars; temporal script with concise section/performance tags.",
            "metadata": "BPM, keyscale, timesignature, duration are dedicated metadata fields, never caption prose.",
        },
        "caption_contract": {
            "max_chars": 512,
            "required_dimensions": ACE_STEP_CAPTION_DIMENSIONS,
            "full_tag_library_tool": "Call TagLibraryTool before final JSON; it returns the complete caption tag taxonomy and lyric meta tag library.",
            "forbidden": [
                "lyrics",
                "section tags",
                "track headers",
                "JSON/prose scaffolding",
                "BPM/key/duration/model/seed",
                "full user prompt",
            ],
        },
        "lyrics_contract": {
            "language": language,
            "max_chars": int(lyric_plan.get("max_lyrics_chars") or 4096),
            "min_words": int(lyric_plan.get("min_words") or 0),
            "target_words": int(lyric_plan.get("target_words") or 0),
            "max_words": int(lyric_plan.get("max_words") or 0),
            "min_lines": int(lyric_plan.get("min_lines") or 0),
            "target_lines": int(lyric_plan.get("target_lines") or 0),
            "required_sections": lyric_plan.get("sections") or [],
            "required_stats_fields": ["lyrics_word_count", "lyrics_line_count", "lyrics_char_count", "section_count"],
            "line_rule": "Use short performable lines. Rap bars may be split across breath units; do not write prose paragraphs.",
            "full_lyric_tag_library_tool": "Call TagLibraryTool before final JSON; keep lyric tags concise per ACE-Step tutorial guidance.",
        },
        "locked_user_fields": {
            key: blueprint.get(key)
            for key in ("track_number", "title", "locked_title", "producer_credit", "bpm", "key_scale", "style", "vibe", "narrative", "required_phrases")
            if blueprint.get(key) not in (None, "", [])
        },
        "repair_instruction": (
            "Before final JSON, count words, vocal lines, chars, sections, hook count, and tag dimensions. "
            "If any required count is short, call TrackRepairTool or extend/reflow lyrics before output. "
            "Never claim payload_gate_status=pass unless these counts pass."
        ),
        "quality_profile": options.get("quality_profile"),
    }


class CrewAIEmptyResponseError(RuntimeError):
    """Raised when the local planner returns the explicit empty-response marker."""


def _is_empty_response_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("acejam_empty_response_fallback") is True:
        return True
    if isinstance(payload.get("album_bible"), dict) and payload.get("tracks") == [] and payload.get("error"):
        return bool(payload.get("acejam_empty_response_fallback"))
    return False


def _album_debug_file(options: dict[str, Any], name: str) -> Path | None:
    debug_dir = str((options or {}).get("album_debug_dir") or "").strip()
    if not debug_dir:
        return None
    safe_name = name.strip().lstrip("/").replace("..", "_")
    return Path(debug_dir) / safe_name


def _write_album_debug_json(options: dict[str, Any], name: str, payload: Any) -> str:
    path = _album_debug_file(options, name)
    if path is None:
        return ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_debug_jsonable(payload), ensure_ascii=True, indent=2), encoding="utf-8")
        _update_album_debug_index(options, {name: str(path)})
        return str(path)
    except Exception as exc:
        print(f"[album_crew][debug][warning] {type(exc).__name__}: {_monitor_preview(exc, 220)}", flush=True)
        return ""


def _append_album_debug_jsonl(options: dict[str, Any], name: str, payload: Any) -> str:
    path = _album_debug_file(options, name)
    if path is None:
        return ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(_debug_jsonable(payload) if isinstance(payload, dict) else {"payload": _debug_jsonable(payload)}),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        _update_album_debug_index(options, {name: str(path)})
        return str(path)
    except Exception as exc:
        print(f"[album_crew][debug][warning] {type(exc).__name__}: {_monitor_preview(exc, 220)}", flush=True)
        return ""


def _update_album_debug_index(options: dict[str, Any], files: dict[str, str]) -> None:
    path = _album_debug_file(options, "debug_index.json")
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if path.exists():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = {}
        payload.setdefault("version", "album-debug-index-2026-04-29")
        payload.setdefault("files", {})
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload["files"].update(files)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        return


def _track_gate_retry_message(report: dict[str, Any]) -> str:
    issues = report.get("blocking_issues") or report.get("issues") or []
    pieces: list[str] = []
    for issue in issues[:8]:
        issue_id = str(issue.get("id") or "quality_issue")
        detail = str(issue.get("detail") or "").strip()
        if issue_id == "lyrics_too_few_lines":
            pieces.append(f"{issue_id}: {detail}; call LyricCounterTool then TrackRepairTool or split long lines into breath units.")
        elif issue_id == "lyrics_under_length":
            pieces.append(f"{issue_id}: {detail}; extend verses, bridge, or final chorus while staying under 4096 chars.")
        elif issue_id == "tag_dimension_coverage":
            pieces.append(f"{issue_id}: {detail}; call TagCoverageTool and add compact caption terms for missing dimensions.")
        elif issue_id == "caption_leakage":
            pieces.append(f"{issue_id}: {detail}; call CaptionIntegrityTool and rewrite caption as sound terms only.")
        elif issue_id == "section_coverage_low":
            pieces.append(f"{issue_id}: {detail}; call SectionMapTool and include required section tags.")
        else:
            pieces.append(f"{issue_id}: {detail}".rstrip(": "))
    if not pieces:
        pieces.append(f"payload_gate_status={report.get('status') or 'unknown'}; call PayloadGateTool and repair before final JSON.")
    return "Album track guardrail failed; retry with repaired final JSON. " + " | ".join(pieces)


def _track_json_guardrail_factory(
    *,
    blueprint: dict[str, Any],
    options: dict[str, Any],
    lyric_plan: dict[str, Any],
) -> Callable[[Any], Tuple[bool, Any]]:
    def _guardrail(output: Any) -> Tuple[bool, Any]:
        try:
            payload = _task_output_json_dict(output)
        except Exception as exc:
            return False, f"Track JSON parse failed: {_monitor_preview(exc, 240)}. Return one strict JSON object only."
        if _is_empty_response_payload(payload):
            return False, (
                "CrewAI planner returned acejam_empty_response_fallback. Retry without tools if needed, "
                "but produce a real track JSON with lyrics, tags, caption counters, and metadata."
            )
        merged = {**dict(blueprint or {}), **dict(payload or {})}
        if not merged.get("caption") and merged.get("tags"):
            merged["caption"] = merged.get("tags")
        merged.setdefault("duration", blueprint.get("duration") or options.get("track_duration") or 180)
        merged.setdefault("language", options.get("language") or "en")
        report = evaluate_album_payload_quality(
            merged,
            options={
                **dict(options or {}),
                "track_duration": merged.get("duration") or options.get("track_duration") or 180,
                "lyric_density": options.get("lyric_density") or "dense",
                "structure_preset": options.get("structure_preset") or "auto",
            },
            repair=True,
        )
        public_report = {key: value for key, value in report.items() if key != "repaired_payload"}
        _append_album_debug_jsonl(
            options,
            "04_track_guardrails.jsonl",
            {
                "track_number": merged.get("track_number"),
                "title": merged.get("title"),
                "status": report.get("status"),
                "gate_passed": bool(report.get("gate_passed")),
                "report": public_report,
            },
        )
        if not report.get("gate_passed"):
            return False, _track_gate_retry_message(report)
        repaired = dict(report.get("repaired_payload") or merged)
        stats = lyric_stats(str(repaired.get("lyrics") or ""))
        repaired["lyrics_word_count"] = int(stats.get("word_count") or 0)
        repaired["lyrics_line_count"] = int(stats.get("line_count") or 0)
        repaired["lyrics_char_count"] = int(stats.get("char_count") or 0)
        repaired["section_count"] = int(stats.get("section_count") or 0)
        repaired["hook_count"] = sum(
            1 for section in stats.get("sections") or [] if re.search(r"chorus|hook|refrain", str(section), re.I)
        )
        repaired["lyric_duration_fit"] = report.get("lyric_duration_fit") or {}
        repaired["lyric_density_gate"] = report.get("lyric_density_gate") or {}
        repaired["producer_grade_sonic_contract"] = report.get("producer_grade_sonic_contract") or {}
        repaired["sonic_dna_coverage"] = report.get("sonic_dna_coverage") or {}
        repaired["producer_grade_readiness"] = report.get("producer_grade_readiness") or {}
        covered = [
            item.get("dimension")
            for item in ((report.get("tag_coverage") or {}).get("dimensions") or [])
            if item.get("status") == "pass"
        ]
        repaired["caption_dimensions_covered"] = covered
        repaired["quality_checks"] = {
            **(repaired.get("quality_checks") if isinstance(repaired.get("quality_checks"), dict) else {}),
            "guardrail_validated": "pass",
            "lyric_length_plan": {
                "min_words": lyric_plan.get("min_words"),
                "min_lines": lyric_plan.get("min_lines"),
                "max_lyrics_chars": lyric_plan.get("max_lyrics_chars"),
            },
        }
        return True, json.dumps(repaired, ensure_ascii=True)

    # CrewAI validates the raw function annotations, not only get_type_hints().
    # With postponed annotations enabled, make the return type concrete.
    _guardrail.__annotations__["return"] = Tuple[bool, Any]
    _guardrail.__annotations__["output"] = Any
    return _guardrail


def _prefer_production_lyrics(
    current_lyrics: Any,
    track_result: Any,
    *,
    duration: float,
    density: str,
    structure_preset: str,
    genre_hint: str,
) -> tuple[str, bool]:
    current = str(current_lyrics or "")
    candidate = _lyric_like_text(_task_output_raw_text(track_result))
    if not candidate or candidate == current:
        return current, False
    current_score = _lyrics_richness_score(current, duration, density, structure_preset, genre_hint)
    candidate_score = _lyrics_richness_score(candidate, duration, density, structure_preset, genre_hint)
    if candidate_score >= max(current_score + 1.0, 5.5):
        return candidate, True
    return current, False


def _compact_track_memory_record(track: dict[str, Any], *, include_lyrics_excerpt: bool = False) -> tuple[str, dict[str, Any]]:
    stats = lyric_stats(str(track.get("lyrics") or ""))
    caption = track.get("caption") or track.get("tags") or ""
    title = str(track.get("title") or "Untitled")
    lyrics_excerpt = ""
    if include_lyrics_excerpt:
        lyrics_excerpt = "\nlyrics_excerpt=" + _clip_text(track.get("lyrics"), 420)
    content = (
        f"Track {track.get('track_number') or '?'}: {title}\n"
        f"description={_clip_text(track.get('description'), 260)}\n"
        f"caption={_clip_text(caption, 360)}\n"
        f"duration={track.get('duration')} bpm={track.get('bpm')} key={track.get('key_scale')} "
        f"time={track.get('time_signature')} language={track.get('language')}\n"
        f"lyrics_stats=words:{stats.get('word_count')}, lines:{stats.get('line_count')}, "
        f"sections:{stats.get('section_count')}\n"
        f"quality={_clip_text(track.get('tool_report'), 420)}"
        f"{lyrics_excerpt}"
    )
    metadata = {
        "track_number": _small_scalar(track.get("track_number")),
        "title": _small_scalar(title),
        "duration": _small_scalar(track.get("duration")),
        "bpm": _small_scalar(track.get("bpm")),
        "key_scale": _small_scalar(track.get("key_scale")),
        "time_signature": _small_scalar(track.get("time_signature")),
        "language": _small_scalar(track.get("language")),
        "prompt_kit_version": _small_scalar(track.get("prompt_kit_version") or PROMPT_KIT_VERSION),
        "genre_modules": _clip_text(",".join(str(item) for item in (track.get("genre_modules") or [])), 120),
        "lyric_words": stats.get("word_count"),
        "lyric_lines": stats.get("line_count"),
        "lyric_sections": stats.get("section_count"),
    }
    return _clip_text(content), metadata


def _get_ollama_client():
    import ollama

    return ollama.Client(host=OLLAMA_BASE_URL)


def list_ollama_models() -> list[dict[str, Any]]:
    try:
        client = _get_ollama_client()
        response = client.list()
        return [{"name": m.model, "size": m.size} for m in response.models]
    except Exception as exc:
        print(f"[album_crew] Failed to list Ollama models: {exc}")
        return []


def ollama_model_names() -> list[str]:
    return [m["name"] for m in list_ollama_models()]


def test_ollama_model(model_name: str) -> dict[str, Any]:
    try:
        client = _get_ollama_client()
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            think=False,
        )
        return {"ok": True, "response": response.message.content[:100]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _ollama_model_installed(model_name: str) -> bool:
    model = str(model_name or "").strip()
    if not model:
        return False
    return model in set(ollama_model_names())


def _ollama_embed_works(client: Any, model_name: str) -> tuple[bool, str]:
    try:
        response = client.embed(model=model_name, input="AceJAM album memory preflight")
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings") or response.get("embedding")
        if embeddings:
            return True, ""
        return False, "Ollama returned no embeddings"
    except Exception as exc:
        return False, str(exc)


def preflight_album_ollama(ollama_model: str, embedding_model: str) -> dict[str, Any]:
    planner = str(ollama_model or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL).strip()
    embedder = str(embedding_model or DEFAULT_ALBUM_EMBEDDING_MODEL).strip()
    client = _get_ollama_client()
    installed = set(ollama_model_names())
    errors: list[str] = []
    warnings: list[str] = []
    if planner not in installed:
        errors.append(f"Planner model is not installed in Ollama: {planner}")
    if embedder not in installed:
        errors.append(f"Embedding model is not installed in Ollama: {embedder}")
    chat_ok = False
    embed_ok = False
    if not errors:
        try:
            response = client.chat(
                model=planner,
                messages=[{"role": "user", "content": "Reply OK. No explanation."}],
                options={"num_predict": 8},
                think=False,
            )
            chat_ok = bool(str(getattr(getattr(response, "message", None), "content", "") or "").strip())
        except Exception as exc:
            errors.append(f"Planner model test failed: {exc}")
        embed_ok, embed_error = _ollama_embed_works(client, embedder)
        if not embed_ok:
            for fallback in ALBUM_EMBEDDING_FALLBACK_MODELS:
                if fallback == embedder or fallback not in installed:
                    continue
                fallback_ok, fallback_error = _ollama_embed_works(client, fallback)
                if fallback_ok:
                    warnings.append(
                        f"Embedding model {embedder} failed embed() ({embed_error}); using {fallback} instead."
                    )
                    embedder = fallback
                    embed_ok = True
                    break
                warnings.append(f"Embedding fallback {fallback} failed embed(): {fallback_error}")
        if not embed_ok:
            errors.append(f"Embedding model test failed: {embed_error}")
    return {
        "ok": not errors,
        "planner_model": planner,
        "embedding_model": embedder,
        "chat_ok": chat_ok,
        "embed_ok": embed_ok,
        "memory_dir": str(CREWAI_MEMORY_DIR),
        "errors": errors,
        "warnings": warnings,
    }


def _strip_thinking_blocks(raw: Any) -> str:
    text = str(raw or "")
    # Extract content INSIDE think blocks in case the model puts all output there
    think_content = ""
    for match in re.finditer(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE):
        think_content += match.group(1)
    # Strip think blocks from the text
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"<think>[\s\S]*", "", stripped, flags=re.IGNORECASE)
    stripped = stripped.replace("</think>", "").strip()
    # If stripping left nothing useful but think blocks had content, use that content
    if not stripped and think_content.strip():
        stripped = think_content.strip()
    return stripped


def _empty_response_fallback_text(model_name: str) -> str:
    payload = {
        "acejam_empty_response_fallback": True,
        "error": f"{model_name} returned an empty response after CrewAI retries.",
        "album_bible": {"concept": "", "arc": "", "motifs": []},
        "tracks": [],
    }
    return "Thought: I now know the final answer\nFinal Answer: " + json.dumps(payload, ensure_ascii=True)


def _compact_tool_context(opts: dict[str, Any], track_duration: float, num_tracks: int, concept: str) -> dict[str, Any]:
    lang = str(opts.get("language") or "en")
    genre_modules = infer_genre_modules(concept, max_modules=2)
    sparse = is_sparse_lyric_genre(concept)
    length_plan = lyric_length_plan(
        track_duration,
        str(opts.get("lyric_density") or "dense"),
        str(opts.get("structure_preset") or "auto"),
        concept,
    )
    portfolio = album_model_portfolio(opts.get("installed_models"))
    arc = [
        "opener - immediate identity and strongest first impression",
        *["escalation - new scene, sharper rhythm, more pressure"] * max(0, num_tracks - 5),
        "climax - highest stakes and biggest hook",
        "cooldown - emotional consequence and contrast",
        "closer - resolution, callback, or final twist",
    ][:num_tracks]
    return {
        "lyric_length_plan": length_plan,
        "album_arc": arc,
        "album_model_portfolio": portfolio,
        "quality_target": opts.get("quality_target") or "hit",
        "tag_packs": opts.get("tag_packs") or ["genre_style", "mood_atmosphere", "instruments", "timbre_texture"],
        "artist_reference_notes": opts.get("artist_reference_notes") or [],
        "live_crewai_tools": CREWAI_LIVE_TOOLS,
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "language_preset": language_preset(lang),
        "genre_modules": genre_modules,
        "section_map": section_map_for(track_duration, concept, instrumental=sparse),
        "kit_metadata_defaults": kit_metadata_defaults(
            mode="album",
            language=lang,
            genre_hint=concept,
            duration=track_duration,
            instrumental=sparse,
        ),
        "user_album_contract": contract_prompt_context(opts.get("user_album_contract")),
    }


def _compact_agent_tool_context(opts: dict[str, Any], track_duration: float, num_tracks: int, concept: str) -> dict[str, Any]:
    """Tiny prompt context for local agents; full registries stay in debug/tooling."""
    full = _compact_tool_context(opts, track_duration, num_tracks, concept)
    length_plan = dict(full.get("lyric_length_plan") or {})
    return {
        "lyric_length_plan": {
            "duration": length_plan.get("duration"),
            "density": length_plan.get("density"),
            "sections": length_plan.get("sections"),
            "target_words": length_plan.get("target_words"),
            "min_words": length_plan.get("min_words"),
            "target_lines": length_plan.get("target_lines"),
            "min_lines": length_plan.get("min_lines"),
            "max_lyrics_chars": length_plan.get("max_lyrics_chars"),
        },
        "album_arc": full.get("album_arc") or [],
        "quality_target": full.get("quality_target") or "hit",
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "language_preset": {
            "code": (full.get("language_preset") or {}).get("code"),
            "name": (full.get("language_preset") or {}).get("name"),
            "script": (full.get("language_preset") or {}).get("script"),
            "notes": _clip_text((full.get("language_preset") or {}).get("notes"), 220),
        },
        "genre_modules": [
            {
                "slug": module.get("slug"),
                "label": module.get("label"),
                "density": module.get("density"),
                "caption_dna": (module.get("caption_dna") or [])[:5],
                "structure": _clip_text(module.get("structure"), 220),
            }
            for module in (full.get("genre_modules") or [])[:3]
            if isinstance(module, dict)
        ],
        "section_map": (full.get("section_map") or [])[:8],
        "user_album_contract": full.get("user_album_contract") or {},
    }


def preflight_album_local_llm(
    planner_provider: str,
    planner_model: str,
    embedding_provider: str,
    embedding_model: str,
) -> dict[str, Any]:
    provider_name = normalize_provider(planner_provider)
    embed_provider = normalize_provider(embedding_provider)
    if provider_name == "ollama" and embed_provider == "ollama":
        return preflight_album_ollama(planner_model, embedding_model)

    errors: list[str] = []
    warnings: list[str] = []
    chat_ok = False
    embed_ok = False
    selected_embedding = str(embedding_model or "").strip()

    def _lmstudio_detail(catalog: dict[str, Any], model: str) -> dict[str, Any]:
        for item in catalog.get("details") or []:
            if isinstance(item, dict) and str(item.get("name") or item.get("model") or "") == model:
                return item
        return {}

    def _load_lmstudio_if_needed(model: str, kind: str, catalog: dict[str, Any], requested_context: int | None = None) -> None:
        detail = _lmstudio_detail(catalog, model)
        loaded = bool(detail.get("loaded") or model in set(catalog.get("loaded_models") or []))
        loaded_context = int(detail.get("loaded_context_length") or 0)
        pinned_context = requested_context if CREWAI_LMSTUDIO_PIN_CONTEXT else None
        needs_context_reload = bool(pinned_context and loaded and loaded_context != pinned_context)
        if not loaded or needs_context_reload:
            lmstudio_load_model(model, kind=kind, context_length=pinned_context if kind == "chat" else None)
            if needs_context_reload:
                warnings.append(
                    f"Reloaded LM Studio {kind} model {model} with context_length={requested_context} "
                    f"(was {loaded_context or 'unknown'})."
                )
            else:
                warnings.append(f"Loaded LM Studio {kind} model {model} before CrewAI preflight.")

    try:
        catalog = lmstudio_model_catalog() if provider_name == "lmstudio" else {"ready": True, "chat_models": ollama_model_names()}
        if not catalog.get("ready"):
            errors.append(catalog.get("error") or f"{provider_label(provider_name)} is not reachable.")
        elif planner_model not in set(catalog.get("chat_models") or catalog.get("models") or []):
            errors.append(f"Planner model {planner_model} is not available in {provider_label(provider_name)}.")
        else:
            if provider_name == "lmstudio":
                _load_lmstudio_if_needed(planner_model, "chat", catalog, CREWAI_LLM_CONTEXT_WINDOW)
            test = local_llm_test_model(provider_name, planner_model, "chat")
            chat_ok = bool(test.get("success"))
    except Exception as exc:
        errors.append(f"Planner model preflight failed: {exc}")

    try:
        embed_catalog = lmstudio_model_catalog() if embed_provider == "lmstudio" else {"ready": True, "embedding_models": ollama_model_names()}
        if not embed_catalog.get("ready"):
            errors.append(embed_catalog.get("error") or f"{provider_label(embed_provider)} is not reachable.")
        else:
            embed_models = [str(item) for item in (embed_catalog.get("embedding_models") or []) if str(item).strip()]
            if selected_embedding not in set(embed_models):
                if embed_models:
                    fallback = embed_models[0]
                    warnings.append(f"Embedding model {selected_embedding or '(empty)'} was not available; using {fallback}.")
                    selected_embedding = fallback
                else:
                    errors.append(f"No embedding model is available in {provider_label(embed_provider)}.")
            if selected_embedding:
                if embed_provider == "lmstudio":
                    _load_lmstudio_if_needed(selected_embedding, "embedding", embed_catalog)
                vector = local_llm_embed(embed_provider, selected_embedding, "AceJAM album memory test")
                embed_ok = bool(vector)
                if not embed_ok:
                    errors.append(f"Embedding model {selected_embedding} returned no vector.")
    except Exception as exc:
        errors.append(f"Embedding model preflight failed: {exc}")

    return {
        "ok": not errors and chat_ok and embed_ok,
        "planner_provider": provider_name,
        "planner_model": planner_model,
        "embedding_provider": embed_provider,
        "embedding_model": selected_embedding,
        "chat_ok": chat_ok,
        "embed_ok": embed_ok,
        "errors": errors,
        "warnings": warnings,
        "memory_dir": str(CREWAI_MEMORY_DIR),
        "legacy_memory_dir": str(CREWAI_LEGACY_MEMORY_DIR),
    }


class AceJamAgentError(RuntimeError):
    """Raised when AceJAM's direct album agent loop cannot produce a valid payload."""


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    dot = sum(float(left[index]) * float(right[index]) for index in range(limit))
    left_norm = sum(float(left[index]) ** 2 for index in range(limit)) ** 0.5
    right_norm = sum(float(right[index]) ** 2 for index in range(limit)) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return float(dot / (left_norm * right_norm))


def _text_chunks(text: Any, *, chunk_chars: int = ACEJAM_AGENT_CONTEXT_CHUNK_CHARS) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    size = max(400, int(chunk_chars or ACEJAM_AGENT_CONTEXT_CHUNK_CHARS))
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", raw) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [raw]:
        if len(paragraph) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            for index in range(0, len(paragraph), size):
                piece = paragraph[index:index + size].strip()
                if piece:
                    chunks.append(piece)
            continue
        candidate = (current + "\n\n" + paragraph).strip() if current else paragraph
        if len(candidate) > size and current:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


class AlbumContextStore:
    """Small job-scoped RAG store for AceJAM album agents."""

    def __init__(
        self,
        *,
        options: dict[str, Any],
        provider: str,
        model: str,
        enabled: bool,
        logs: list[str],
    ) -> None:
        self.options = options
        self.provider = normalize_provider(provider)
        self.model = str(model or "").strip()
        self.logs = logs
        self.enabled = bool(enabled and self.model)
        debug_dir = str((options or {}).get("album_debug_dir") or "").strip()
        self.root = Path(debug_dir) / "context_store" if debug_dir else None
        self.chunks: list[dict[str, Any]] = []
        self.vectors: list[list[float]] = []
        self.retrieval_rounds = 0
        self.disabled_reason = "" if self.enabled else "embedding memory disabled"
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
            _update_album_debug_index(options, {"context_store/index.json": str(self.root / "index.json")})

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def disable(self, reason: str) -> None:
        self.enabled = False
        self.disabled_reason = str(reason or "embedding memory disabled")
        self._write_index()

    def add(self, kind: str, text: Any, metadata: dict[str, Any] | None = None) -> list[str]:
        ids: list[str] = []
        chunks = _text_chunks(text)
        if not chunks:
            return ids
        for chunk in chunks:
            chunk_id = f"ctx_{len(self.chunks) + 1:04d}"
            record = {
                "id": chunk_id,
                "kind": str(kind or "context"),
                "text": _clip_text(chunk, ACEJAM_AGENT_CONTEXT_CHUNK_CHARS),
                "metadata": _debug_jsonable(metadata or {}),
            }
            vector: list[float] = []
            if self.enabled:
                try:
                    vector = local_llm_embed(self.provider, self.model, chunk)
                    if not vector:
                        raise ValueError("embedding provider returned no vector")
                except Exception as exc:
                    self.logs.append(f"Agent memory disabled: {provider_label(self.provider)} embedding failed ({_monitor_preview(exc, 180)}).")
                    self.disable(str(exc))
                    vector = []
            self.chunks.append(record)
            self.vectors.append(vector)
            ids.append(chunk_id)
            if self.root:
                with (self.root / "chunks.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                with (self.root / "vectors.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"id": chunk_id, "dims": len(vector), "vector": vector}, ensure_ascii=True) + "\n")
        self._write_index()
        return ids

    def search(
        self,
        query: Any,
        *,
        top_k: int = ACEJAM_AGENT_RETRIEVAL_TOP_K,
        kinds: list[str] | None = None,
        track_number: int | None = None,
        label: str = "retrieval",
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        kind_set = {str(item) for item in (kinds or []) if str(item).strip()}
        query_text = str(query or "").strip()
        query_vector: list[float] = []
        if self.enabled and query_text:
            try:
                query_vector = local_llm_embed(self.provider, self.model, query_text)
            except Exception as exc:
                self.logs.append(f"Agent memory retrieval disabled: {_monitor_preview(exc, 180)}")
                self.disable(str(exc))
        scored: list[dict[str, Any]] = []
        for index, chunk in enumerate(self.chunks):
            if kind_set and str(chunk.get("kind")) not in kind_set:
                continue
            if track_number is not None:
                metadata = chunk.get("metadata") or {}
                meta_track = metadata.get("track_number")
                if meta_track not in (None, ""):
                    try:
                        if int(meta_track) != int(track_number):
                            continue
                    except Exception:
                        continue
            score = _cosine_similarity(query_vector, self.vectors[index]) if query_vector and index < len(self.vectors) else 0.0
            if not query_vector and query_text:
                haystack = str(chunk.get("text") or "").lower()
                tokens = {token for token in re.findall(r"[a-zA-Z0-9']{3,}", query_text.lower())}
                score = sum(1 for token in tokens if token in haystack) / max(1, len(tokens))
            scored.append({
                "id": chunk.get("id"),
                "kind": chunk.get("kind"),
                "score": round(float(score), 6),
                "text": chunk.get("text"),
                "metadata": chunk.get("metadata") or {},
            })
        scored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        results = scored[: max(1, int(top_k or ACEJAM_AGENT_RETRIEVAL_TOP_K))]
        self.retrieval_rounds += 1
        if self.root:
            with (self.root / "retrieval_trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label": label,
                    "query_preview": _monitor_preview(query_text, 240),
                    "result_ids": [item.get("id") for item in results],
                    "scores": [item.get("score") for item in results],
                }, ensure_ascii=True) + "\n")
        self._write_index()
        return results

    def block(
        self,
        query: Any,
        *,
        top_k: int = ACEJAM_AGENT_RETRIEVAL_TOP_K,
        kinds: list[str] | None = None,
        track_number: int | None = None,
        label: str = "retrieval",
    ) -> str:
        results = self.search(query, top_k=top_k, kinds=kinds, track_number=track_number, label=label)
        if not results:
            return "[]"
        compact = [
            {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "score": item.get("score"),
                "metadata": item.get("metadata"),
                "text": _clip_text(item.get("text"), 520),
            }
            for item in results
        ]
        return json.dumps(compact, ensure_ascii=True, indent=2)

    def _write_index(self) -> None:
        if not self.root:
            return
        payload = {
            "version": "acejam-album-context-store-2026-04-30",
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "disabled_reason": self.disabled_reason,
            "chunk_count": self.chunk_count,
            "retrieval_rounds": self.retrieval_rounds,
            "files": {
                "chunks": str(self.root / "chunks.jsonl"),
                "vectors": str(self.root / "vectors.jsonl"),
                "retrieval_trace": str(self.root / "retrieval_trace.jsonl"),
            },
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "index.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return


def preflight_album_agent_llm(
    planner_provider: str,
    planner_model: str,
    planner_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider_name = normalize_provider(planner_provider)
    model = str(planner_model or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL).strip()
    settings = planner_llm_settings_from_payload(planner_settings or {})
    requested_context = int(settings.get("planner_context_length") or CREWAI_LLM_CONTEXT_WINDOW)
    errors: list[str] = []
    warnings: list[str] = []
    chat_ok = False

    def _lmstudio_detail(catalog: dict[str, Any], model_name: str) -> dict[str, Any]:
        for item in catalog.get("details") or []:
            if isinstance(item, dict) and str(item.get("name") or item.get("model") or "") == model_name:
                return item
        return {}

    try:
        if provider_name == "lmstudio":
            catalog = lmstudio_model_catalog()
            if not catalog.get("ready"):
                errors.append(catalog.get("error") or "LM Studio is not reachable.")
            elif model not in set(catalog.get("chat_models") or catalog.get("models") or []):
                errors.append(f"Planner model {model} is not available in LM Studio.")
            else:
                detail = _lmstudio_detail(catalog, model)
                loaded = bool(detail.get("loaded") or model in set(catalog.get("loaded_models") or []))
                loaded_context = int(detail.get("loaded_context_length") or 0)
                if (not loaded) or (CREWAI_LMSTUDIO_PIN_CONTEXT and loaded_context != requested_context):
                    lmstudio_load_model(
                        model,
                        kind="chat",
                        context_length=requested_context if CREWAI_LMSTUDIO_PIN_CONTEXT else None,
                    )
                    warnings.append(
                        f"Loaded LM Studio chat model {model} for AceJAM Agents"
                        + (f" with context_length={requested_context}." if CREWAI_LMSTUDIO_PIN_CONTEXT else ".")
                    )
        test = local_llm_test_model(provider_name, model, "chat", settings)
        chat_ok = bool(test.get("success"))
        if not chat_ok:
            errors.append(str(test.get("error") or "chat preflight returned no success flag"))
    except Exception as exc:
        errors.append(f"Planner model preflight failed: {exc}")

    return {
        "ok": not errors and chat_ok,
        "planner_provider": provider_name,
        "planner_model": model,
        "planner_llm_settings": settings,
        "chat_ok": chat_ok,
        "errors": errors,
        "warnings": warnings,
    }


def _ollama_v1_base_url() -> str:
    return f"{OLLAMA_BASE_URL.rstrip('/')}/v1"


def _ollama_embedding_url() -> str:
    return f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"


def _ollama_embedder_config(model_name: str) -> dict[str, Any]:
    model = str(model_name or DEFAULT_ALBUM_EMBEDDING_MODEL).strip() or DEFAULT_ALBUM_EMBEDDING_MODEL
    return {
        "provider": "ollama",
        "config": {
            "model_name": model,
            "url": _ollama_embedding_url(),
        },
    }


def _local_embedder_config(provider: str, model_name: str) -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    model = str(model_name or DEFAULT_ALBUM_EMBEDDING_MODEL).strip() or DEFAULT_ALBUM_EMBEDDING_MODEL
    if provider_name == "lmstudio":
        return {
            "provider": "openai",
            "config": {
                "model": model,
                "api_base": lmstudio_api_base_url(),
                "api_key": os.environ.get("LMSTUDIO_API_TOKEN", "lm-studio"),
            },
        }
    return _ollama_embedder_config(model)


def _make_album_memory(
    planner_model: str,
    embedding_model: str,
    read_only: bool = True,
    planner_provider: str = "ollama",
    embedding_provider: str = "ollama",
):
    from crewai import Memory

    return Memory(
        llm=_make_llm(planner_model, planner_provider),
        storage=str(CREWAI_MEMORY_DIR),
        embedder=_local_embedder_config(embedding_provider, embedding_model),
        root_scope="acejam_album_production",
        read_only=read_only,
        consolidation_threshold=1.0,
        consolidation_limit=1,
        exploration_budget=0,
        query_analysis_threshold=100000,
    )


def _make_album_memory_writer(
    planner_model: str,
    embedding_model: str,
    planner_provider: str = "ollama",
    embedding_provider: str = "ollama",
):
    return _make_album_memory(planner_model, embedding_model, read_only=False, planner_provider=planner_provider, embedding_provider=embedding_provider)


def _remember_compact(
    memory: Any,
    content: Any,
    *,
    scope: str,
    categories: list[str],
    metadata: dict[str, Any] | None = None,
    importance: float = 0.5,
    logs: list[str] | None = None,
) -> bool:
    compact_content = _clip_text(content, CREWAI_MEMORY_CONTENT_LIMIT)
    safe_metadata = {
        str(key): _small_scalar(value)
        for key, value in (metadata or {}).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    try:
        memory.remember(
            compact_content,
            scope=scope,
            categories=[_clip_text(category, 80) for category in categories],
            metadata=safe_metadata,
            importance=float(importance),
            source="acejam",
            private=False,
            root_scope="acejam_album_production",
        )
        return True
    except Exception as exc:
        message = f"AceJAM compact memory save skipped: {exc}"
        if logs is not None:
            logs.append(message)
        print(f"[album_crew][memory] {message}", flush=True)
        return False


def _remember_album_bible(
    memory: Any,
    album_bible: dict[str, Any],
    blueprints: list[dict[str, Any]],
    logs: list[str] | None = None,
) -> None:
    content = (
        "Album bible summary\n"
        f"concept={_clip_text(album_bible.get('concept'), 360)}\n"
        f"arc={_clip_text(album_bible.get('arc'), 360)}\n"
        f"motifs={_compact_json(album_bible.get('motifs') or [], 360)}\n"
        f"tracks={_compact_json([{'n': b.get('track_number'), 'title': b.get('title'), 'role': b.get('description')} for b in blueprints], 620)}"
    )
    metadata = {
        "record_type": "album_bible",
        "track_count": len(blueprints),
        "concept": _clip_text(album_bible.get("concept"), 160),
    }
    _remember_compact(
        memory,
        content,
        scope="/acejam_album_production/album_bible",
        categories=["album_bible", "album_arc", "track_blueprints"],
        metadata=metadata,
        importance=0.6,
        logs=logs,
    )


def _remember_track(memory: Any, track: dict[str, Any], logs: list[str] | None = None) -> None:
    content, metadata = _compact_track_memory_record(track)
    metadata["record_type"] = "track_production"
    _remember_compact(
        memory,
        content,
        scope="/acejam_album_production/track",
        categories=["track_production", "lyrics_summary", "ace_step_payload"],
        metadata=metadata,
        importance=0.55,
        logs=logs,
    )


def _search_web(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        results = []
        for match in re.finditer(r'class="result__snippet">(.*?)</a>', html, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            text = re.sub(r"\s+", " ", text)
            if text and len(text) > 20:
                results.append(text)
            if len(results) >= 5:
                break
        if results:
            return "\n\n".join(results)
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()[:1000]
    except Exception as exc:
        return f"Search failed: {exc}"


def _crewai_llm_kwargs(model_name: str, provider: str = "ollama") -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    if provider_name == "lmstudio":
        return {
            "model": str(model_name or "").strip(),
            "provider": "openai",
            "base_url": lmstudio_api_base_url(),
            "api_key": os.environ.get("LMSTUDIO_API_TOKEN", "lm-studio"),
            "temperature": 0.72,
            "top_p": 0.92,
            "max_tokens": CREWAI_LMSTUDIO_MAX_TOKENS,
            "timeout": CREWAI_LLM_TIMEOUT_SECONDS,
        }
    return {
        "model": str(model_name or "").strip(),
        "provider": "ollama",
        "base_url": OLLAMA_BASE_URL,
        "api_base": _ollama_v1_base_url(),
        "temperature": 0.72,
        "top_p": 0.92,
        "max_tokens": CREWAI_LLM_MAX_TOKENS,
        "additional_params": {
            "extra_body": {
                "think": False,
                "options": {
                    "num_ctx": CREWAI_LLM_CONTEXT_WINDOW,
                    "num_predict": CREWAI_LLM_NUM_PREDICT,
                }
            },
        },
        "timeout": CREWAI_LLM_TIMEOUT_SECONDS,
    }


def _lmstudio_no_think_text(value: str) -> str:
    directive = CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE
    if not CREWAI_LMSTUDIO_DISABLE_THINKING or not directive or directive in value:
        return value
    return f"{directive}\n{value}"


def _lmstudio_no_think_messages(value: Any) -> Any:
    if not CREWAI_LMSTUDIO_DISABLE_THINKING or not CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE:
        return value
    if isinstance(value, str):
        return _lmstudio_no_think_text(value)
    if not isinstance(value, list):
        return value
    copied: list[Any] = []
    user_index: int | None = None
    for item in value:
        if isinstance(item, dict):
            next_item = dict(item)
            if user_index is None and str(next_item.get("role") or "").lower() == "user":
                user_index = len(copied)
            copied.append(next_item)
        else:
            copied.append(item)
    if any(
        isinstance(item, dict)
        and isinstance(item.get("content"), str)
        and CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE in item["content"]
        for item in copied
    ):
        return copied
    if user_index is not None and isinstance(copied[user_index], dict) and isinstance(copied[user_index].get("content"), str):
        copied[user_index]["content"] = _lmstudio_no_think_text(str(copied[user_index].get("content") or ""))
    else:
        copied.insert(0, {"role": "user", "content": CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE})
    prefill = CREWAI_LMSTUDIO_NO_THINK_PREFILL
    if prefill and not any(isinstance(item, dict) and item.get("content") == prefill for item in copied):
        copied.append({"role": "assistant", "content": prefill})
    return copied


def _lmstudio_no_think_args(
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    provider_name: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if normalize_provider(provider_name) != "lmstudio":
        return call_args, call_kwargs
    next_args = list(call_args)
    next_kwargs = dict(call_kwargs)
    if next_args:
        next_args[0] = _lmstudio_no_think_messages(next_args[0])
    elif "messages" in next_kwargs:
        next_kwargs["messages"] = _lmstudio_no_think_messages(next_kwargs["messages"])
    elif "prompt" in next_kwargs and isinstance(next_kwargs["prompt"], str):
        next_kwargs["prompt"] = _lmstudio_no_think_text(str(next_kwargs["prompt"]))
    return tuple(next_args), next_kwargs


def _empty_response_recovery_args(
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    recovery_instruction = (
        "Your previous local model response was empty. Do not call tools. "
        "Return the requested final answer now. If the task asks for JSON, return strict JSON only. "
        "If the task asks for production notes, return compact production notes with the requested counters."
    )
    next_args = list(call_args)
    next_kwargs = dict(call_kwargs)
    if next_args and isinstance(next_args[0], list):
        messages = [dict(item) if isinstance(item, dict) else item for item in next_args[0]]
        messages.append({"role": "user", "content": recovery_instruction})
        next_args[0] = messages
    elif isinstance(next_kwargs.get("messages"), list):
        messages = [dict(item) if isinstance(item, dict) else item for item in next_kwargs["messages"]]
        messages.append({"role": "user", "content": recovery_instruction})
        next_kwargs["messages"] = messages
    elif "prompt" in next_kwargs:
        next_kwargs["prompt"] = f"{next_kwargs.get('prompt')}\n\n{recovery_instruction}"
    elif next_args and isinstance(next_args[0], str):
        next_args[0] = f"{next_args[0]}\n\n{recovery_instruction}"
    return tuple(next_args), next_kwargs


def _is_lmstudio_model_crash(value: Any) -> bool:
    text = str(value or "").lower()
    return "model has crashed" in text or "exit code: null" in text


def _make_llm(model_name: str, provider: str = "ollama", debug_log_file: str | None = None):
    from crewai import LLM

    provider_name = normalize_provider(provider)
    llm = LLM(**_crewai_llm_kwargs(model_name, provider_name))
    # Qwen/Ollama models often emit raw <think> content. If CrewAI sends native
    # OpenAI-style tool schemas to Ollama, that content can crash Ollama's tool
    # parser before AceJAM gets a fallback. Keep tools available through CrewAI's
    # text tool loop, but do not advertise native function-calling support.
    llm.supports_function_calling = lambda: False
    original_call = llm.call
    object.__setattr__(llm, "_acejam_original_call", original_call)

    def _debug_print_response(label: str, value: Any, attempt: int) -> None:
        if not CREWAI_DEBUG_LLM_RESPONSES:
            return
        text = str(value if value is not None else "")
        print(
            f"\n[album_crew][llm][{label}] provider={provider_name} model={model_name} "
            f"attempt={attempt + 1}/{CREWAI_EMPTY_RESPONSE_RETRIES + 1} "
            f"type={type(value).__name__} chars={len(text)}",
            flush=True,
        )
        print("[album_crew][llm][content-begin]", flush=True)
        print(text, flush=True)
        print("[album_crew][llm][content-end]\n", flush=True)

    def _patched_call(*args, **kwargs):
        last_result: Any = None
        crash_retries = 0
        total_attempts = max(CREWAI_EMPTY_RESPONSE_RETRIES + 1, CREWAI_LMSTUDIO_CRASH_RETRIES + 1)
        for attempt in range(total_attempts):
            try:
                call_args, call_kwargs = _lmstudio_no_think_args(args, kwargs, provider_name)
                _append_llm_debug_jsonl(
                    debug_log_file,
                    {
                        "event": "request",
                        "provider": provider_name,
                        "model": model_name,
                        "attempt": attempt + 1,
                        "args": call_args,
                        "kwargs": call_kwargs,
                    },
                )
                result = getattr(llm, "_acejam_original_call")(*call_args, **call_kwargs)
            except Exception as exc:
                text = str(exc)
                _append_llm_debug_jsonl(
                    debug_log_file,
                    {
                        "event": "exception",
                        "provider": provider_name,
                        "model": model_name,
                        "attempt": attempt + 1,
                        "exception_type": type(exc).__name__,
                        "exception": text,
                    },
                )
                if "OpenAI API call failed" in text:
                    text = text.replace("OpenAI API call failed", f"{provider_label(provider_name)} CrewAI call failed for {model_name}")
                if "context length" in text.lower() or "input length" in text.lower():
                    if provider_name == "lmstudio":
                        text += (
                            f" [AceJAM limit: context_window={CREWAI_LLM_CONTEXT_WINDOW}, "
                            f"max_tokens={CREWAI_LMSTUDIO_MAX_TOKENS}; LM Studio context is set through /api/v1/models/load.]"
                        )
                    else:
                        text += (
                            f" [AceJAM limit: num_ctx={CREWAI_LLM_CONTEXT_WINDOW}, "
                            f"num_predict={CREWAI_LLM_NUM_PREDICT}. CrewAI auto-summarization is disabled by default.]"
                        )
                if provider_name == "lmstudio" and _is_lmstudio_model_crash(text) and crash_retries < CREWAI_LMSTUDIO_CRASH_RETRIES:
                    crash_retries += 1
                    print(
                        f"[album_crew][llm][lmstudio-crash-retry] model={model_name} "
                        f"retry={crash_retries}/{CREWAI_LMSTUDIO_CRASH_RETRIES}; reloading via /api/v1/models/load",
                        flush=True,
                    )
                    try:
                        lmstudio_load_model(
                            str(model_name or ""),
                            kind="chat",
                            context_length=CREWAI_LLM_CONTEXT_WINDOW if CREWAI_LMSTUDIO_PIN_CONTEXT else None,
                        )
                    except Exception as reload_exc:
                        print(
                            f"[album_crew][llm][lmstudio-reload-failed] model={model_name} "
                            f"{type(reload_exc).__name__}: {_monitor_preview(reload_exc, 260)}",
                            flush=True,
                        )
                    time.sleep(min(CREWAI_EMPTY_RESPONSE_RETRY_DELAY, 2.0))
                    continue
                print(
                    f"\n[album_crew][llm][exception] provider={provider_name} model={model_name} "
                    f"attempt={attempt + 1}/{total_attempts} "
                    f"{type(exc).__name__}: {_monitor_preview(text, 420)}\n",
                    flush=True,
                )
                raise
            last_result = result
            _append_llm_debug_jsonl(
                debug_log_file,
                {
                    "event": "response",
                    "provider": provider_name,
                    "model": model_name,
                    "attempt": attempt + 1,
                    "response_type": type(result).__name__,
                    "response": result,
                    "response_chars": len(str(result or "")),
                },
            )
            _debug_print_response("raw-response", result, attempt)
            if isinstance(result, str) and "<think" in result.lower():
                stripped = _strip_thinking_blocks(result)
                _debug_print_response("stripped-response", stripped, attempt)
                result = stripped
            if str(result or "").strip():
                return result
            print(
                f"[album_crew][llm][empty-response] provider={provider_name} model={model_name} "
                f"attempt={attempt + 1}/{CREWAI_EMPTY_RESPONSE_RETRIES + 1}; retrying={attempt < CREWAI_EMPTY_RESPONSE_RETRIES}",
                flush=True,
            )
            if attempt >= CREWAI_EMPTY_RESPONSE_RETRIES:
                break
            time.sleep(CREWAI_EMPTY_RESPONSE_RETRY_DELAY)
        try:
            recovery_args, recovery_kwargs = _empty_response_recovery_args(args, kwargs)
            recovery_args, recovery_kwargs = _lmstudio_no_think_args(recovery_args, recovery_kwargs, provider_name)
            _append_llm_debug_jsonl(
                debug_log_file,
                {
                    "event": "empty_response_recovery_request",
                    "provider": provider_name,
                    "model": model_name,
                    "args": recovery_args,
                    "kwargs": recovery_kwargs,
                },
            )
            recovery_result = getattr(llm, "_acejam_original_call")(*recovery_args, **recovery_kwargs)
            if isinstance(recovery_result, str) and "<think" in recovery_result.lower():
                recovery_result = _strip_thinking_blocks(recovery_result)
            _append_llm_debug_jsonl(
                debug_log_file,
                {
                    "event": "empty_response_recovery_response",
                    "provider": provider_name,
                    "model": model_name,
                    "response": recovery_result,
                    "response_chars": len(str(recovery_result or "")),
                },
            )
            if str(recovery_result or "").strip():
                return recovery_result
        except Exception as recovery_exc:
            _append_llm_debug_jsonl(
                debug_log_file,
                {
                    "event": "empty_response_recovery_exception",
                    "provider": provider_name,
                    "model": model_name,
                    "exception_type": type(recovery_exc).__name__,
                    "exception": str(recovery_exc),
                },
            )
        fallback = _empty_response_fallback_text(model_name)
        _append_llm_debug_jsonl(
            debug_log_file,
            {
                "event": "empty_response_fallback",
                "provider": provider_name,
                "model": model_name,
                "response": fallback,
            },
        )
        _debug_print_response("empty-response-fallback", fallback, CREWAI_EMPTY_RESPONSE_RETRIES)
        return fallback

    llm.call = _patched_call
    return llm


def _crew_task(
    *,
    description: str,
    expected_output: str,
    agent: Any,
    context: list[Any] | None = None,
    output_json: type[BaseModel] | None = None,
    guardrail: Callable[[Any], Any] | None = None,
):
    from crewai import Task

    kwargs: dict[str, Any] = {
        "description": description,
        "expected_output": expected_output,
        "agent": agent,
        "guardrail_max_retries": CREWAI_TASK_MAX_RETRIES,
    }
    if context is not None:
        kwargs["context"] = context
    if output_json is not None:
        kwargs["output_json"] = output_json
    if guardrail is not None:
        kwargs["guardrail"] = guardrail
    return Task(**kwargs)


def _json_from_text(raw: str) -> list[dict[str, Any]]:
    text = _strip_thinking_blocks(raw)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("tracks"), list):
            return [item for item in parsed["tracks"] if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    if text.lstrip().startswith(("[", "{")):
        raise ValueError("Crew result did not contain a valid JSON track array")
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"[\[{]", text))):
        try:
            parsed, _end = decoder.raw_decode(text[match.start():])
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict) and isinstance(parsed.get("tracks"), list):
                return [item for item in parsed["tracks"] if isinstance(item, dict)]
        except json.JSONDecodeError:
            continue
    raise ValueError("Crew result did not contain a valid JSON track array")


def _json_object_from_text(raw: str) -> dict[str, Any]:
    text = _strip_thinking_blocks(raw)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    candidates: list[str] = [text]
    sanitized = _escape_json_string_control_chars(text)
    if sanitized != text:
        candidates.append(sanitized)
    balanced = _balance_json_trailing_closers(sanitized)
    if balanced not in candidates:
        candidates.append(balanced)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass
    for candidate in candidates[1:]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except json.JSONDecodeError:
            pass
    if text.lstrip().startswith(("{", "[")):
        decoder = json.JSONDecoder()
        for candidate in candidates:
            try:
                parsed, _end = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Crew result did not contain a valid JSON object")
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in reversed(list(re.finditer(r"\{", candidate))):
            try:
                parsed, _end = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("Crew result did not contain a valid JSON object")


def _escape_json_string_control_chars(text: str) -> str:
    """Repair common local-LLM JSON where lyrics strings contain raw newlines."""
    out: list[str] = []
    in_string = False
    escaped = False
    for char in str(text or ""):
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
                continue
            if char == "\\":
                out.append(char)
                escaped = True
                continue
            if char == '"':
                out.append(char)
                in_string = False
                continue
            if char == "\n":
                out.append("\\n")
                continue
            if char == "\r":
                out.append("\\r")
                continue
            if char == "\t":
                out.append("\\t")
                continue
            out.append(char)
            continue
        out.append(char)
        if char == '"':
            in_string = True
            escaped = False
    return "".join(out)


def _balance_json_trailing_closers(text: str) -> str:
    """Append missing trailing JSON closers while ignoring braces inside strings."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in str(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
            else:
                return text
    return str(text or "") + "".join(reversed(stack)) if stack else str(text or "")


def _coerce_agent_lyrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize agent-friendly lyric arrays into the ACE-Step lyrics string."""
    if not isinstance(payload, dict):
        return payload
    result = dict(payload)

    def _section_key(section: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(section or "").lower())

    def _split_lyric_lines(value: Any) -> list[str]:
        return [line.strip() for line in str(value or "").splitlines() if line.strip()]

    def _is_section_line(line: str) -> bool:
        return bool(re.fullmatch(r"\[[^\]]+\]", str(line or "").strip()))

    def _expected_sections() -> list[str]:
        expected: list[str] = []
        sections_value = result.get("sections")
        if isinstance(sections_value, list):
            for section in sections_value:
                if isinstance(section, dict):
                    tag = _section_tag_line(section.get("section") or section.get("name") or section.get("tag") or "")
                else:
                    tag = _section_tag_line(section)
                if tag:
                    expected.append(tag)
        return expected

    def _distribute_lines_under_sections(raw_lines: list[str], expected_sections: list[str]) -> list[str]:
        lyric_only = [line for line in raw_lines if line and not _is_section_line(line)]
        if not expected_sections:
            return raw_lines
        if not lyric_only:
            return list(expected_sections)
        count = max(1, len(expected_sections))
        base = len(lyric_only) // count
        remainder = len(lyric_only) % count
        offset = 0
        distributed: list[str] = []
        for index, tag in enumerate(expected_sections):
            take = base + (1 if index < remainder else 0)
            distributed.append(tag)
            if take > 0:
                distributed.extend(lyric_only[offset : offset + take])
                offset += take
        if offset < len(lyric_only):
            distributed.extend(lyric_only[offset:])
        return distributed

    def _dict_section(value: dict[str, Any]) -> str:
        return _section_tag_line(value.get("section_tag") or value.get("section") or value.get("tag") or "")

    def _append_line_item(lines: list[str], value: Any, last_section_key: str) -> str:
        if isinstance(value, dict):
            section = _dict_section(value)
            section_key = _section_key(section)
            if section and section_key != last_section_key:
                lines.append(section)
                last_section_key = section_key
            primary = value.get("line") or value.get("text") or value.get("lyric") or ""
            for line in _split_lyric_lines(primary):
                if line != section:
                    lines.append(line)
            nested = value.get("lines") or value.get("lyrics_lines")
            if isinstance(nested, list):
                for item in nested:
                    last_section_key = _append_line_item(lines, item, last_section_key)
            elif isinstance(nested, str):
                for line in _split_lyric_lines(nested):
                    last_section_key = _append_line_item(lines, line, last_section_key)
            return last_section_key
        for line in _split_lyric_lines(value):
            tagged_line = re.match(r"^(\[[^\]]+\])\s*:\s*(.+)$", line)
            if tagged_line:
                tag = _section_tag_line(tagged_line.group(1))
                tag_key = _section_key(tag)
                if tag and tag_key != last_section_key:
                    lines.append(tag)
                    last_section_key = tag_key
                lyric = tagged_line.group(2).strip()
                if lyric:
                    lines.append(lyric)
                continue
            if _is_section_line(line):
                tag = _section_tag_line(line)
                tag_key = _section_key(tag)
                if tag_key and tag_key != last_section_key:
                    lines.append(tag)
                    last_section_key = tag_key
                continue
            lines.append(line)
        return last_section_key

    lines_value = result.get("lyrics_lines") or result.get("lyric_lines") or result.get("script_lines")
    if isinstance(lines_value, list):
        lines: list[str] = []
        last_section_key = ""
        for item in lines_value:
            last_section_key = _append_line_item(lines, item, last_section_key)
        if lines:
            expected_sections = _expected_sections()
            existing_keys = {_section_key(line) for line in lines if _is_section_line(line)}
            if expected_sections and not existing_keys:
                lines = _distribute_lines_under_sections(lines, expected_sections)
            else:
                for section_index, tag in enumerate(expected_sections):
                    tag_key = _section_key(tag)
                    if not tag_key or tag_key in existing_keys:
                        continue
                    next_keys = {_section_key(item) for item in expected_sections[section_index + 1 :]}
                    insert_at = len(lines)
                    if section_index == 0:
                        insert_at = 0
                    else:
                        for line_index, line in enumerate(lines):
                            if _is_section_line(line) and _section_key(line) in next_keys:
                                insert_at = line_index
                                break
                    lines.insert(insert_at, tag)
                    existing_keys.add(tag_key)
            joined = "\n".join(lines)
            current = str(result.get("lyrics") or "")
            if not current.strip() or len(joined) >= len(current):
                result["lyrics"] = joined
            result["lyrics_lines"] = lines

    sections_value = result.get("sections")
    if not str(result.get("lyrics") or "").strip() and isinstance(sections_value, list):
        lines: list[str] = []
        for section in sections_value:
            if isinstance(section, dict):
                title = _section_tag_line(section.get("section") or section.get("name") or section.get("tag") or "")
                if title:
                    lines.append(title)
                section_lines = section.get("lines") or section.get("lyrics") or []
                if isinstance(section_lines, str):
                    lines.extend(line.strip() for line in section_lines.splitlines() if line.strip())
                elif isinstance(section_lines, list):
                    last_section_key = _section_key(title)
                    for line in section_lines:
                        last_section_key = _append_line_item(lines, line, last_section_key)
            elif isinstance(section, str):
                text = section.strip()
                if text:
                    lines.append(text)
        if lines:
            result["lyrics"] = "\n".join(lines)
            result.setdefault("lyrics_lines", lines)
    return result


def _coerce_options(
    concept: str,
    num_tracks: int,
    track_duration: float,
    language: str,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    opts = dict(options or {})
    sanitized, artist_notes = sanitize_artist_references(concept)
    contract = opts.get("user_album_contract")
    if not isinstance(contract, dict):
        contract = extract_user_album_contract(concept, num_tracks, language, opts)
    opts.setdefault("song_model_strategy", "all_models_album")
    opts.setdefault("quality_target", "hit")
    opts["quality_profile"] = normalize_quality_profile(opts.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    opts.setdefault("lyric_density", "dense")
    opts.setdefault("rhyme_density", 0.8)
    opts.setdefault("metaphor_density", 0.7)
    opts.setdefault("hook_intensity", 0.85)
    opts.setdefault("structure_preset", "auto")
    opts.setdefault("bpm_strategy", "varied")
    opts.setdefault("key_strategy", "related")
    opts.setdefault("use_web_inspiration", False)
    opts.setdefault("track_variants", 1)
    opts.setdefault("inference_steps", ALBUM_FINAL_DOCS_BEST["inference_steps"])
    opts.setdefault("guidance_scale", ALBUM_FINAL_DOCS_BEST["guidance_scale"])
    opts.setdefault("shift", ALBUM_FINAL_DOCS_BEST["shift"])
    opts.setdefault("infer_method", ALBUM_FINAL_DOCS_BEST["infer_method"])
    opts.setdefault("sampler_mode", ALBUM_FINAL_DOCS_BEST["sampler_mode"])
    opts.setdefault("audio_format", ALBUM_FINAL_DOCS_BEST["audio_format"])
    opts.setdefault("auto_score", False)
    opts.setdefault("auto_lrc", False)
    opts.setdefault("return_audio_codes", False)
    opts.setdefault("save_to_library", True)
    opts.update(
        {
            "concept": concept,
            "sanitized_concept": sanitized,
            "artist_reference_notes": artist_notes,
            "num_tracks": int(num_tracks),
            "track_duration": float(track_duration),
            "language": language,
            "user_album_contract": contract,
            "album_title": opts.get("album_title") or contract.get("album_title") if isinstance(contract, dict) else opts.get("album_title"),
        }
    )
    return opts


def songwriting_toolkit(installed_models: set[str] | list[str] | None = None) -> dict[str, Any]:
    return toolkit_payload(installed_models)


def _output_json_for_provider(model: type[BaseModel], planner_provider: str) -> type[BaseModel] | None:
    # Always return None — thinking models (Qwen3, etc.) emit <think> blocks that
    # break CrewAI's Pydantic output_json validator before our _strip_thinking_blocks
    # can clean them. We parse the raw text ourselves after kickoff.
    return None


def _tool_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _select_crewai_tools(tools: list[Any], allowed_names: set[str]) -> list[Any]:
    allowed = {_tool_key(name) for name in allowed_names}
    selected: list[Any] = []
    for tool in tools:
        keys = {_tool_key(getattr(tool, "name", ""))}
        description = str(getattr(tool, "description", "") or "")
        match = re.search(r"Tool Name:\s*([A-Za-z0-9_ -]+)", description)
        if match:
            keys.add(_tool_key(match.group(1)))
        if keys & allowed:
            selected.append(tool)
    return selected or tools


def _album_genre_hint(opts: dict[str, Any]) -> str:
    contract = opts.get("user_album_contract") if isinstance(opts, dict) else {}
    parts: list[str] = []
    if isinstance(contract, dict):
        for track in contract.get("tracks") or []:
            if not isinstance(track, dict):
                continue
            parts.extend(
                str(track.get(key) or "")
                for key in ("style", "vibe", "narrative", "required_phrases")
                if track.get(key)
            )
    parts.extend(
        str(opts.get(key) or "")
        for key in ("custom_tags", "negative_tags", "sanitized_concept")
        if opts.get(key)
    )
    text = "\n".join(parts)
    text = re.sub(r"\(\s*produced\s+by\s+[^)]+\)", " ", text, flags=re.I)
    text = re.sub(r"\bproduced\s+by\s+[^\n.;]+", " ", text, flags=re.I)
    text = re.sub(r"\bproducer(?:_credit)?\s*:\s*[^\n.;]+", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def create_album_bible_crew(
    concept: str,
    num_tracks: int,
    track_duration: float,
    ollama_model: str,
    language: str = "en",
    embedding_model: str = DEFAULT_ALBUM_EMBEDDING_MODEL,
    options: dict[str, Any] | None = None,
    planner_provider: str = "ollama",
    embedding_provider: str = "ollama",
    step_callback: Callable[[Any], Any] | None = None,
    task_callback: Callable[[Any], Any] | None = None,
    output_log_file: str | None = None,
):
    """Compact CrewAI stage that returns album bible plus per-track blueprints."""
    from crewai import Agent, Crew, Process

    opts = _coerce_options(concept, num_tracks, track_duration, language, options)
    llm = _make_llm(ollama_model, planner_provider, str(opts.get("llm_debug_log_file") or ""))
    lang_preset = language_preset(language)
    genre_hint = _album_genre_hint(opts)
    matched_genres = infer_genre_modules(genre_hint, max_modules=2)
    length_plan = lyric_length_plan(
        track_duration,
        str(opts.get("lyric_density") or "dense"),
        str(opts.get("structure_preset") or "auto"),
        genre_hint,
    )
    tool_context = json.dumps(
        {
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "language_preset": lang_preset,
            "genre_modules": matched_genres,
            "section_map": section_map_for(track_duration, genre_hint, instrumental=is_sparse_lyric_genre(genre_hint)),
            "lyric_length_plan": length_plan,
            "album_model_portfolio": album_model_portfolio(opts.get("installed_models")),
            "user_album_contract": contract_prompt_context(opts.get("user_album_contract")),
        },
        ensure_ascii=True,
    )
    tools = _select_crewai_tools(
        make_crewai_tools(opts),
        {
            "album_arc_tool",
            "album_continuity_tool",
            "tag_library_tool",
            "genre_module_tool",
            "language_preset_tool",
            "lyric_length_tool",
            "section_map_tool",
            "model_portfolio_tool",
            "per_model_settings_tool",
            "generation_settings_tool",
            "conflict_checker_tool",
        },
    )
    cfg = dict(
        llm=llm,
        verbose=CREWAI_VERBOSE,
        allow_delegation=False,
        max_iter=CREWAI_AGENT_MAX_ITER,
        max_execution_time=None,
        max_retry_limit=CREWAI_AGENT_MAX_RETRY_LIMIT,
        respect_context_window=CREWAI_RESPECT_CONTEXT_WINDOW,
        step_callback=step_callback,
    )
    executive = Agent(
        role="Executive Producer and Album Bible Architect",
        goal=f"Plan exactly {num_tracks} locked, duration-realistic album track blueprints",
        backstory=(
            f"Use Prompt Kit {PROMPT_KIT_VERSION}. Treat the UserAlbumContract as authoritative. "
            "Do not rename, reorder, translate, replace, or reinterpret locked titles, album title, "
            "producer credits, BPM, style, vibe, narrative, or required phrases. "
            "Create a compact album bible: arc, sonic palette, motifs, and track roles."
        ),
        tools=tools,
        **cfg,
    )
    finalizer = Agent(
        role="ACE-Step Blueprint Engineer",
        goal="Return strict JSON blueprints that AceJAM can validate and produce one track at a time",
        backstory=(
            "Return JSON only. Keep lyrics out of the bible stage except required safe phrases. "
            "Use compact fields and preserve contract metadata for every track."
        ),
        tools=tools,
        **cfg,
    )
    task_plan = _crew_task(
        description=(
            f"Create a compact album_bible for exactly {num_tracks} tracks.\n"
            f"Concept: {opts['sanitized_concept']}\n"
            f"Language: {language}; preset: {json.dumps(lang_preset, ensure_ascii=True)}\n"
            f"Tool context: {_compact_json(tool_context, CREWAI_PROMPT_BUDGET_CHARS)}\n"
            "For each track include track_number, artist_name, title, description, tags, bpm, key_scale, "
            "time_signature, duration, hook_promise, performance_brief, producer_credit, style, vibe, narrative. "
            "Do not write full lyrics in this stage."
        ),
        expected_output="Compact album bible and numbered track blueprint notes.",
        agent=executive,
    )
    task_json = _crew_task(
        description=(
            'Return strict JSON only with this shape: {"album_bible":{"concept":"...","arc":"...",'
            '"motifs":[],"sonic_palette":"...","continuity_rules":[]},"tracks":[...]}. '
            f"Exactly {num_tracks} tracks. Preserve UserAlbumContract locked fields exactly. "
            "Do not include markdown fences, prose, thoughts, or full lyrics."
        ),
        expected_output="Strict JSON object with album_bible and tracks.",
        agent=finalizer,
        context=[task_plan],
        output_json=_output_json_for_provider(AlbumBiblePayloadModel, planner_provider),
    )
    return Crew(
        agents=[executive, finalizer],
        tasks=[task_plan, task_json],
        process=Process.sequential,
        memory=_make_album_memory(ollama_model, embedding_model, read_only=True, planner_provider=planner_provider, embedding_provider=embedding_provider),
        embedder=_local_embedder_config(embedding_provider, embedding_model),
        verbose=CREWAI_VERBOSE,
        step_callback=step_callback,
        task_callback=task_callback,
        output_log_file=output_log_file,
    )


def create_track_production_crew(
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    num_tracks: int,
    track_duration: float,
    ollama_model: str,
    language: str = "en",
    embedding_model: str = DEFAULT_ALBUM_EMBEDDING_MODEL,
    options: dict[str, Any] | None = None,
    planner_provider: str = "ollama",
    embedding_provider: str = "ollama",
    step_callback: Callable[[Any], Any] | None = None,
    task_callback: Callable[[Any], Any] | None = None,
    output_log_file: str | None = None,
):
    """Compact CrewAI stage that produces one track from a locked blueprint."""
    from crewai import Agent, Crew, Process

    opts = _coerce_options(str(options.get("concept") if isinstance(options, dict) else "") or str(album_bible.get("concept") or ""), num_tracks, track_duration, language, options)
    llm = _make_llm(ollama_model, planner_provider, str(opts.get("llm_debug_log_file") or ""))
    lang_preset = language_preset(language)
    blueprint = dict(blueprint or {})
    opts["track_duration"] = parse_duration_seconds(blueprint.get("duration") or track_duration, track_duration)
    lyric_plan = lyric_length_plan(
        opts["track_duration"],
        str(opts.get("lyric_density") or "dense"),
        str(opts.get("structure_preset") or "auto"),
        " ".join(str(blueprint.get(key) or "") for key in ("description", "tags", "style", "vibe", "narrative")),
    )
    payload_contract = _ace_step_track_payload_contract(lyric_plan, language, blueprint, opts)
    track_prompt_template = render_track_prompt_template(
        user_album_contract=contract_prompt_context(opts.get("user_album_contract")),
        ace_step_payload_contract=payload_contract,
        lyric_length_plan=lyric_plan,
        language_preset=lang_preset,
        blueprint=blueprint,
        album_bible=album_bible,
    )
    opts["ace_step_track_payload_contract"] = payload_contract
    opts["ace_step_track_prompt_template_version"] = ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION
    _append_album_debug_jsonl(
        opts,
        "03_track_prompt_templates.jsonl",
        {
            "track_number": blueprint.get("track_number"),
            "title": blueprint.get("title"),
            "template_version": ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION,
            "prompt_template": track_prompt_template,
        },
    )
    tools = _select_crewai_tools(
        make_crewai_tools(opts),
        {
            "ace_step_prompt_contract_tool",
            "tag_library_tool",
            "lyric_counter_tool",
            "tag_coverage_tool",
            "caption_integrity_tool",
            "payload_gate_tool",
            "lyric_length_tool",
            "section_map_tool",
            "arrangement_tool",
            "vocal_performance_tool",
            "rhyme_flow_tool",
            "metaphor_world_tool",
            "hook_doctor_tool",
            "caption_polisher_tool",
            "cliche_guard_tool",
            "conflict_checker_tool",
            "track_repair_tool",
            "validation_checklist_tool",
            "negative_control_tool",
            "effective_settings_tool",
            "hit_readiness_tool",
            "model_compatibility_tool",
        },
    )
    cfg = dict(
        llm=llm,
        verbose=CREWAI_VERBOSE,
        allow_delegation=False,
        max_iter=CREWAI_AGENT_MAX_ITER,
        max_execution_time=None,
        max_retry_limit=CREWAI_AGENT_MAX_RETRY_LIMIT,
        respect_context_window=CREWAI_RESPECT_CONTEXT_WINDOW,
        step_callback=step_callback,
    )
    producer = Agent(
        role="Track Production Team",
        goal="Produce exactly one track from a compact blueprint with complete lyrics and ACE-Step metadata",
        backstory=(
            f"Use Prompt Kit {PROMPT_KIT_VERSION}. You are producing one track, not all tracks. "
            "Preserve locked title, producer credit, BPM, style, vibe, narrative, and required phrases exactly. "
            "Write complete, duration-aware lyrics with ACE-Step section tags and no placeholders."
        ),
        tools=tools,
        **cfg,
    )
    finalizer = Agent(
        role="Track JSON Finalizer",
        goal="Return strict JSON for this one track only",
        backstory=(
            "Return a single JSON object only. Preserve all contract fields. "
            "Caption/tags describe sound; lyrics contain lyrics and section/performance tags only. "
            "Every track must satisfy the ACE-Step payload contract: tag dimensions covered, "
            "no prompt leakage in captions, and lyrics long enough for the chosen duration."
        ),
        tools=tools,
        **cfg,
    )
    compact_context = {
        "album_bible": album_bible,
        "blueprint": blueprint,
        "lyric_length_plan": lyric_plan,
        "language_preset": lang_preset,
        "section_map": section_map_for(blueprint.get("duration") or track_duration, str(blueprint.get("tags") or blueprint.get("description") or "")),
        "docs_best_model_settings": ALBUM_FINAL_DOCS_BEST,
        "quality_profile": opts.get("quality_profile"),
        "ace_step_payload_contract_version": ACE_STEP_PAYLOAD_CONTRACT_VERSION,
        "settings_policy": "Use EffectiveSettingsTool and ModelCompatibilityTool. Keep unsupported/reserved/read-only settings out of active payloads.",
    }
    task_produce = _crew_task(
        description=(
            f"Produce exactly one track from this compact blueprint: {_compact_json(blueprint, 3600)}\n"
            f"Album bible: {_compact_json(album_bible, 1800)}\n"
            f"Production context: {_compact_json(compact_context, 5200)}\n"
            f"ACE-Step track prompt template:\n{_clip_text(track_prompt_template, CREWAI_PROMPT_BUDGET_CHARS)}\n"
            "Write complete lyrics unless the blueprint is explicitly instrumental. "
            "Use AceStepPromptContractTool, TagLibraryTool, LyricLengthTool, LyricCounterTool, "
            "TagCoverageTool, CaptionIntegrityTool, PayloadGateTool, SectionMapTool, CaptionPolisherTool, "
            "and TrackRepairTool as needed. "
            "Your production notes must include a compact ACE_STEP_VALIDATION block with lyrics_word_count, "
            "lyrics_line_count, lyrics_char_count, section_count, hook_count, caption_dimensions_covered, "
            "caption_char_count, and any repairs made. Stay on this single track; do not rewrite album-wide plans."
        ),
        expected_output="Complete production notes and full lyrics for one track, including ACE_STEP_VALIDATION counts.",
        agent=producer,
    )
    task_json = _crew_task(
        description=(
            "Return strict JSON object only. Required fields: track_number, artist_name, title, description, tags, "
            "lyrics, bpm, key_scale, time_signature, language, duration, song_model, seed, inference_steps, "
            "guidance_scale, shift, infer_method, sampler_mode, audio_format, auto_score, auto_lrc, "
            "return_audio_codes, save_to_library, tool_notes, production_team, model_render_notes, "
            "quality_profile, prompt_kit_version, settings_policy_version, settings_compliance, quality_checks, contract_compliance, "
            "tag_coverage, lyric_duration_fit, caption_integrity, payload_gate_status, repair_actions, "
            "lyrics_word_count, lyrics_line_count, lyrics_char_count, section_count, hook_count, "
            "caption_dimensions_covered. "
            f"Apply this prompt template before output:\n{_clip_text(track_prompt_template, CREWAI_PROMPT_BUDGET_CHARS)} "
            "If lyrics_line_count is below min_lines, split long rap/sung lines into shorter breath units or extend sections. "
            "If lyrics_word_count is below min_words, extend verses/bridge/final chorus. "
            "If tags miss a required caption dimension, repair the caption. "
            "Preserve the blueprint title and locked fields exactly. No markdown fences."
        ),
        expected_output="Strict JSON object for exactly one produced track with passing validation counts.",
        agent=finalizer,
        context=[task_produce],
        output_json=_output_json_for_provider(TrackProductionPayloadModel, planner_provider),
        guardrail=_track_json_guardrail_factory(blueprint=blueprint, options=opts, lyric_plan=lyric_plan),
    )
    return Crew(
        agents=[producer, finalizer],
        tasks=[task_produce, task_json],
        process=Process.sequential,
        memory=_make_album_memory(ollama_model, embedding_model, read_only=True, planner_provider=planner_provider, embedding_provider=embedding_provider),
        embedder=_local_embedder_config(embedding_provider, embedding_model),
        verbose=CREWAI_VERBOSE,
        step_callback=step_callback,
        task_callback=task_callback,
        output_log_file=output_log_file,
    )


def create_album_crew(
    concept: str,
    num_tracks: int,
    track_duration: float,
    ollama_model: str,
    language: str = "en",
    embedding_model: str = "nomic-embed-text",
    options: dict[str, Any] | None = None,
    planner_provider: str = "ollama",
    embedding_provider: str = "ollama",
    step_callback: Callable[[Any], Any] | None = None,
    task_callback: Callable[[Any], Any] | None = None,
    output_log_file: str | None = None,
):
    """Single professional production crew that plans and produces an entire album."""
    from crewai import Agent, Crew, Process

    opts = _coerce_options(concept, num_tracks, track_duration, language, options)
    llm = _make_llm(ollama_model, planner_provider, str(opts.get("llm_debug_log_file") or ""))
    lang_name = LANG_NAMES.get(language, language)
    length_plan = lyric_length_plan(
        track_duration,
        str(opts.get("lyric_density") or "dense"),
        str(opts.get("structure_preset") or "auto"),
        opts["sanitized_concept"],
    )
    model_info = choose_song_model(
        set(opts.get("installed_models") or []),
        str(opts.get("song_model_strategy") or "best_installed"),
        str(opts.get("requested_song_model") or "auto"),
    )
    inspiration = ""
    if opts.get("use_web_inspiration"):
        queries = opts.get("inspiration_queries") or opts["sanitized_concept"]
        inspiration = _search_web(str(queries))[:1200]
    tool_context = dict(opts)
    tool_context["web_inspiration"] = inspiration
    tools = make_crewai_tools(tool_context)

    # Build genre-specific production context from matched modules
    matched_genres = infer_genre_modules(opts["sanitized_concept"], max_modules=2)
    genre_detail = ""
    for module in matched_genres[:2]:
        slug = module.get("slug", "")
        genre_detail += (
            f"\nGenre '{slug}' production guide:\n"
            f"  Caption DNA: {module.get('caption_dna', '')}\n"
            f"  Structure: {module.get('structure', '')}\n"
            f"  BPM: {module.get('bpm', '')}\n"
            f"  Keys: {module.get('keys', '')}\n"
            f"  Hook strategy: {module.get('hook_strategy', '')}\n"
            f"  Avoid: {module.get('avoid', '')}\n"
            f"  Density: {module.get('density', 'balanced')}\n"
        )

    # Language-specific production guidance
    lang_preset = language_preset(language)
    lang_guidance = (
        f"Language: {lang_name} (code: {lang_preset.get('vocal_language', language)}). "
        f"Script: {lang_preset.get('script', 'Latin')}. "
        f"Notes: {lang_preset.get('notes', '')}. "
        f"Romanization: {lang_preset.get('romanization', 'not applicable')}."
    )

    # Section tags reference for lyrics
    section_tags_ref = (
        "Available ACE-Step section tags: "
        "[Intro], [Verse], [Verse 1], [Verse 2], [Verse 3], [Pre-Chorus], [Chorus], [Hook], [Post-Chorus], "
        "[Bridge], [Final Chorus], [Outro], [Build], [Build-Up], [Drop], [Final Drop], [Breakdown], [Climax], "
        "[Instrumental], [Instrumental Break], [Guitar Solo], [Piano Interlude], [Synth Solo], [Brass Break], [Drum Break], "
        "[spoken word], [raspy vocal], [whispered], [falsetto], [powerful belting], [harmonies], [call and response], [vocal response], "
        "[Outro - fade out], [Fade Out], [Song ends abruptly]. "
        "Performance modifiers: [Verse - rap], [Chorus - anthemic], [Bridge - whispered], [Verse - melodic rap], "
        "[Chorus - layered vocals], [Intro - dreamy], [Hook - melodic], [Verse - aggressive]."
    )

    tool_summary = json.dumps(
        {
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "language_preset": lang_preset,
            "genre_modules": matched_genres,
            "section_map": section_map_for(track_duration, opts["sanitized_concept"], instrumental=is_sparse_lyric_genre(opts["sanitized_concept"])),
            "lyric_length_plan": length_plan,
            "model_advice": model_info,
            "album_model_portfolio": album_model_portfolio(opts.get("installed_models")),
            "quality_target": opts.get("quality_target"),
            "quality_profile": opts.get("quality_profile"),
            "tag_packs": opts.get("tag_packs"),
            "custom_tags": opts.get("custom_tags"),
            "artist_reference_notes": opts.get("artist_reference_notes"),
        },
        ensure_ascii=True,
    )
    contract_summary = _compact_json(contract_prompt_context(opts.get("user_album_contract")), 1800)

    shared_rules = (
        f"Use Prompt Kit {PROMPT_KIT_VERSION}. Create original songs. Artist references and style imitation are fully allowed. "
        "UserAlbumContract is authoritative: do not rename, reorder, translate, replace, or reinterpret locked user fields. "
        "Keep locked titles, album title, producer credits, BPM, style, vibe, narrative, and required phrases exactly. "
        "CRITICAL ACE-STEP CAPTION RULES: The 'tags' field (ace_caption) describes overall sound using these dimensions: "
        "genre + subgenre, mood/atmosphere, key instruments (2-4), vocal type/character, timbre/texture, production style, "
        "arrangement energy, mix quality. NEVER put BPM, key, duration, or time signature inside caption -- those go in metadata fields only. "
        "Keep caption to 1-3 sentences of concrete sonic description. "
        "Keep bracket tags concise: Good: [Chorus - anthemic]. Bad: [Chorus - anthemic - huge - emotional - layered]. "
        "Caption and lyrics must not conflict. "
        "CRITICAL LYRIC RULES: Use short, singable lines (4-10 syllables or 3-8 words per vocal line). "
        "For rap: prioritize cadence, internal rhyme, breath control, and bar rhythm over poetic prose. "
        "For pop/R&B: prioritize memorable hooks, emotional clarity, singability, and repeatable chorus phrases. "
        "For EDM/techno/house/trance/DnB/dubstep: use fewer lyrics and stronger arrangement tags ([Build], [Drop], [Breakdown], [Final Drop]). "
        "For instrumental/cinematic/ambient: use [Instrumental] or structured instrumental timeline with NO sung lyrics. "
        "No placeholders allowed: never output '...', 'etc.', 'repeat chorus', 'same as before', '[continue]', or unfinished sections. "
        "Avoid AI-flavored lyrics: no vague adjective piles, forced rhymes, mixed metaphors, overlong lines, "
        "empty inspirational slogans, generic 'neon dreams' filler, empty 'we rise / fly / dream' lines unless grounded in concrete situation. "
        "Use one emotional promise per song, one coherent metaphor world, concrete scene details (place, object, weather, body, action), "
        "a repeatable title-connected hook short enough to remember after one listen, "
        "language/script discipline, genre-module routing, and a duration-realistic section_map. "
        "Default generation profile is chart_master: use ACE-Step docs-correct 50-step SFT/Base final-render settings with shift 1.0, wav32 output, "
        "ADG only for Base/XL Base, and one album take per track/model unless the user explicitly asks for more. "
        "Use AceStepSettingsPolicyTool, ChartMasterProfileTool, AceStepCoverageAuditTool, EffectiveSettingsTool, "
        "AandRVariantPlanTool, TaskApplicabilityTool, and ModelCompatibilityTool for generation controls; "
        "do not invent settings, and do not treat read-only, reserved, ignored, or unsupported fields as active. "
        "Use the provided tools when useful. Output concrete, editable production data."
    )
    schema_rules = (
        "Use machine-safe values only: duration must be numeric seconds (60-600), bpm must be 30-300, "
        "key_scale must be like C minor or F# major, time_signature must be 2, 3, 4, or 6, "
        "infer_method must be ode or sde, sampler_mode must be euler or heun, audio_format must be wav/wav32/flac/ogg/mp3/opus/aac, "
        "timesteps overrides inference_steps and shift when present, use_cot_lyrics is reserved/future, "
        "and seed must be numeric or -1. Put colorful phrases in description/tool_notes, never in these fields. "
        "Final JSON must include settings_compliance when generation settings are present."
    )
    cfg = dict(
        llm=llm,
        verbose=CREWAI_VERBOSE,
        allow_delegation=False,
        max_iter=CREWAI_AGENT_MAX_ITER,
        max_execution_time=None,
        max_retry_limit=CREWAI_AGENT_MAX_RETRY_LIMIT,
        respect_context_window=CREWAI_RESPECT_CONTEXT_WINDOW,
        step_callback=step_callback,
    )

    # --- 8 Professional Production Team Agents ---

    executive_producer = Agent(
        role="Executive Producer",
        goal="Design a cohesive album arc with hit-level contrast and decide the ideal duration for each track",
        backstory=(
            f"{shared_rules}\n\n"
            "You are the creative director of the entire album. You plan albums in acts: opener, escalation, "
            "climax, cooldown, closer. Every track needs a distinct scene, emotional job, title, hook promise, "
            "and an appropriate duration in seconds. Choose durations that fit each track's genre and role: "
            "interludes/sketches 60-120s, standard songs 150-240s, epic closers or progressive tracks up to 300s. "
            f"The suggested default is {int(track_duration)}s but you decide what serves each track best."
        ),
        tools=tools,
        **cfg,
    )
    ar_developer = Agent(
        role="A&R and Artist Development",
        goal=f"Define an original artist persona, brand identity, cadence, delivery, vocal responses, and vocal performance tags in {lang_name}",
        backstory=(
            f"{shared_rules}\n\n"
            "You develop the artist identity for this project: stage presence, vocal character, signature vocal responses, "
            "flow patterns, breath control, confidence vs vulnerability moments, and hook delivery style. "
            "Think like the artist in the booth. Artist references and style imitation are fully allowed."
        ),
        tools=tools,
        **cfg,
    )
    songwriter = Agent(
        role="Songwriter and Hook Architect",
        goal=f"Write complete original hooks, verses, pre-choruses, bridges, and outros in {lang_name} for every track",
        backstory=(
            f"{shared_rules}\n\n"
            "You write enough lyrics to fill each track's planned duration. Use clear section tags "
            "([Verse], [Pre-Chorus], [Chorus], [Bridge], [Final Chorus], [Vocal responses], [Outro]), "
            "memorable title-connected hooks, concrete scene details, and no placeholder lines. "
            "Scale lyric density to each track's duration: shorter tracks need fewer sections, longer tracks more."
        ),
        tools=tools,
        **cfg,
    )
    lyric_editor = Agent(
        role="Lyric Editor and Quality Control",
        goal="Raise rhyme density, sharpen imagery, enforce punchline discipline, and eliminate cliches across all tracks",
        backstory=(
            f"{shared_rules}\n\n"
            "You sharpen internal rhyme, multisyllabic/slant rhyme, metaphor worlds, line endings, "
            "and lyrical specificity. You reject generic filler, repeated hooks across tracks, "
            "and lazy rhyme schemes. Artist style references are fully allowed."
        ),
        tools=tools,
        **cfg,
    )
    beat_producer = Agent(
        role="Beat Producer and Sound Designer",
        goal="Design genre, instruments, BPM, key, rhythm, arrangement, and sonic contrast for every track",
        backstory=(
            f"{shared_rules}\n\n"
            "You build each record's musical body: drums, bass, arrangement, keys, transitions, "
            "energy curve, and track-to-track contrast. Every track must have different sonic character "
            "and still fit the album arc. Match BPM and arrangement complexity to each track's planned duration."
        ),
        tools=tools,
        **cfg,
    )
    prompt_engineer = Agent(
        role="ACE-Step Prompt Engineer",
        goal="Convert production intent into compact ACE-Step captions, tags, lyric meta tags, and safe settings",
        backstory=(
            f"{shared_rules}\n\n"
            "Use ACE-Step tag dimensions: genre, mood, instruments, timbre, era, production, vocals, rhythm, "
            "structure, and stem tags. Keep BPM/key/time as metadata fields, not cluttered inside captions. "
            "Captions should be concise sonic descriptions that guide the diffusion model."
        ),
        tools=tools,
        **cfg,
    )
    studio_engineer = Agent(
        role="Studio Engineer and Mix/Master QA",
        goal="Set generation steps, guidance, shift, sampler, output format, score, LRC, and audio-code flags per track",
        backstory=(
            f"{shared_rules}\n\n"
            "Album generation renders a full model portfolio: Turbo, Turbo Shift3, SFT, Base, "
            "XL Turbo, XL SFT, and XL Base. You tune controllable per-track settings for quality, "
            "repeatability, and clean library metadata while AceJAM locks each model-specific album render."
        ),
        tools=tools,
        **cfg,
    )
    quality_editor = Agent(
        role="A&R Quality Editor and JSON Finalizer",
        goal="Reject weak songs, repair issues, and output strict JSON for AceJAM generation",
        backstory=(
            f"{shared_rules}\n\n"
            "You are the final gate. Reject generic lyrics, cliches, repeated hooks, tag conflicts, "
            "and under-length songs. Verify every duration is a realistic numeric value in seconds. "
            "Final output must be a JSON array only -- no prose, no markdown, no analysis."
        ),
        tools=tools,
        **cfg,
    )

    # --- 8 Sequential Tasks ---

    task_concept = _crew_task(
        description=(
            f"Plan exactly {num_tracks} tracks for this album.\n"
            f"Concept: {opts['sanitized_concept']}\n"
            f"{lang_guidance}\n"
            f"Tool context: {tool_summary}\n"
            f"UserAlbumContract: {contract_summary}\n"
            + (f"Current inspiration snippets:\n{inspiration}\n" if inspiration else "")
            + f"Genre production reference:{genre_detail}\n"
            "TRACK LENGTH CATEGORIES (do NOT specify exact seconds — duration will be calculated from lyrics):\n"
            "- 'intro' or 'interlude' or 'skit': short atmospheric piece (30-90s worth of content)\n"
            "- 'full_song': standard album hit (180-300s, the majority of tracks)\n"
            "- 'epic': progressive/cinematic closer (240-360s)\n"
            "A professional album should have mostly full_song tracks, with 1-2 intros/interludes max.\n"
            "Use AlbumArcTool and AlbumContinuityTool. Return for each track: title, role in album arc, "
            "unique scene, hook promise, genre/subgenre, and length category (intro/full_song/epic)."
        ),
        expected_output=f"Numbered plan for {num_tracks} distinct tracks with titles, scenes, hook promises, genres, and length categories.",
        agent=executive_producer,
    )
    task_performance = _crew_task(
        description=(
            f"For each planned track, define an original artist/performance brief in {lang_name}: "
            "persona, cadence, vocal character, vocal responses, delivery tags, and hook performance style. "
            f"{lang_guidance}\n"
            "Use VocalPerformanceTool and RhymeFlowTool when artist references or flow goals appear. "
            "Define: vocal type (male/female/group), vocal texture (raspy, smooth, breathy, powerful), "
            "delivery mode (rap, sing, spoken word, falsetto), vocal-response style, and hook delivery approach."
        ),
        expected_output="Performance and persona brief for every track.",
        agent=ar_developer,
        context=[task_concept],
    )
    task_lyrics = _crew_task(
        description=(
            f"Write FULL professional lyrics for all {num_tracks} tracks in {lang_name}.\n"
            f"{lang_guidance}\n"
            "DO NOT write short demos. Write COMPLETE album-quality songs. The track duration will be "
            "CALCULATED FROM YOUR LYRICS — more words = longer track. Write as much as the song needs.\n"
            "LENGTH GUIDE per track category:\n"
            "- full_song: complete [Intro] + [Verse 1] + [Pre-Chorus] + [Chorus] + [Verse 2] + [Pre-Chorus] + "
            "[Chorus] + [Bridge] + [Final Chorus] + [Outro]. Minimum 200 words, target 250-400 words.\n"
            "- epic: extended structure with 3+ verses, multiple bridges/solos, 300-500 words.\n"
            "- intro/interlude: short atmospheric content 20-60 words, or [Instrumental] with section tags.\n"
            "DURATION WILL BE DERIVED: singing ≈ 2.5 words/sec, rap ≈ 3.5 words/sec, sparse EDM ≈ 1.5 words/sec.\n"
            f"{section_tags_ref}\n"
            "HIT-WRITING QUALITY GATES:\n"
            "- Every vocal song needs a central emotional promise: what changes between verse and chorus?\n"
            "- Use one central metaphor or concrete situation per song. Do not jump between unrelated images.\n"
            "- Include sensory/physical details: place, object, weather, body feeling, sound, motion, memory.\n"
            "- The hook must be short enough to remember after one listen and strong enough to repeat.\n"
            "- Verses add new information; choruses simplify and intensify the emotion.\n"
            "- Rap verses need cadence, breath control, internal rhyme, and bar-to-bar momentum.\n"
            "- Pop/R&B hooks need emotional lift, vowel-friendly phrasing, and clear melodic stress.\n"
            "- EDM hooks should be sparse and placed around builds/drops rather than overfilling.\n"
            "- Lyrics must sound like a human wrote them for this artist and language, not a translated motivational poster.\n"
            "Keep sonic/instrument/production tags in caption only, not as random lyric lines. "
            "Keep each hook unique across the album. No placeholder lines. No AI filler."
        ),
        expected_output="Complete FULL-LENGTH professional lyrics for every track (minimum 200 words per full_song).",
        agent=songwriter,
        context=[task_concept, task_performance],
    )
    task_lyric_edit = _crew_task(
        description=(
            "Edit every lyric for internal/slant rhyme, metaphor focus, cliche cleanup, repeated-line checks, "
            "and hook memorability. Use RhymeFlowTool, MetaphorWorldTool, HookDoctorTool, ClicheGuardTool, "
            "HitScoreTool, and TrackRepairTool.\n"
            "ANTI-AI REWRITE CHECKLIST:\n"
            "- Remove vague adjective piles and empty inspirational slogans.\n"
            "- Replace generic images ('neon dreams', 'fire inside') with concrete specific details.\n"
            "- Ensure every line adds information or emotion, not filler.\n"
            "- Check that hooks are short (3-8 words) and repeatable.\n"
            "- Verify lines are 4-10 syllables for singing, or breathable bars for rap.\n"
            "- Confirm one metaphor world per song, not random image jumping.\n"
            "- Check language sounds natural, not like a translation."
        ),
        expected_output="Polished lyrics with quality notes and anti-AI confirmation for every track.",
        agent=lyric_editor,
        context=[task_concept, task_performance, task_lyrics],
    )
    task_sonic = _crew_task(
        description=(
            "For each planned track, design arrangement, genre, instruments, BPM, key_scale, time_signature, "
            "rhythm pocket, and energy movement. Use ArrangementTool and TagLibraryTool.\n"
            f"Genre production reference:{genre_detail}\n"
            "Every track must have different sonic character and still fit the album arc. "
            "Match BPM range and key to the genre conventions above. "
            "Design: primary instruments (2-4), rhythm style, bass type, production texture, "
            "energy curve within each track, and track-to-track sonic contrast across the album."
        ),
        expected_output="Sonic specification with BPM, key, instruments, and arrangement for every track.",
        agent=beat_producer,
        context=[task_concept, task_performance, task_lyric_edit],
    )
    task_prompt = _crew_task(
        description=(
            "Convert each track into ACE-Step-ready generation fields. Use ModelPortfolioTool, "
            "PerModelSettingsTool, AlbumRenderMatrixTool, ModelAdvisorTool, CaptionPolisherTool, "
            "ConflictCheckerTool, and TagLibraryTool.\n"
            "CAPTION (tags field) MUST describe these exact dimensions in 1-3 sentences:\n"
            "- Primary genre + subgenre (e.g., 'dark melodic trap')\n"
            "- Key instruments (2-4) (e.g., 'heavy 808 bass, rolling hi-hats, icy bell melody')\n"
            "- Mood/atmosphere (e.g., 'moody synth pads')\n"
            "- Vocal type and character (e.g., 'confident male rap vocal, autotuned melodic chorus')\n"
            "- Production/mix quality (e.g., 'crisp modern mix')\n"
            "Example caption: 'dark melodic trap with heavy 808 bass, rolling hi-hats, icy bell melody, "
            "moody synth pads, confident male rap vocal, autotuned melodic chorus, crisp modern mix'\n"
            "NEVER put BPM, key, duration, or time signature in caption -- metadata fields only.\n"
            "LYRICS must contain ONLY lyric text and section/performance tags. No instrument descriptions in lyrics.\n"
            f"{section_tags_ref}\n"
            f"Also preserve Prompt Kit {PROMPT_KIT_VERSION} metadata: target_language, language_notes, genre_profile, "
            "genre_modules, section_map, lyric_density_notes, workflow_mode, negative_control, quality_checks, "
            "troubleshooting_hints, and anti_ai_rewrite_notes. "
            f"{schema_rules}"
        ),
        expected_output="ACE-Step-ready prompt spec with proper caption DNA and clean lyrics for every track.",
        agent=prompt_engineer,
        context=[task_concept, task_performance, task_lyric_edit, task_sonic],
    )
    task_engineering = _crew_task(
        description=(
            "Set editable generation controls for every track: seed, inference_steps, guidance_scale, shift, "
            "infer_method, sampler_mode, audio_format, auto_score, auto_lrc, return_audio_codes, save_to_library. "
            "Use GenerationSettingsTool, PerModelSettingsTool, ChartMasterProfileTool, EffectiveSettingsTool, "
            "AceStepCoverageAuditTool, AandRVariantPlanTool, AlbumRenderMatrixTool, FilenamePlannerTool, "
            "and MixMasterTool. Plan strong defaults; AceJAM will render every track once with each portfolio model. "
            f"{schema_rules}"
        ),
        expected_output="Generation-control spec and mix/master QA for every track.",
        agent=studio_engineer,
        context=[task_concept, task_prompt],
    )
    task_json = _crew_task(
        description=(
            f"Combine the full album production into a strict JSON array of exactly {num_tracks} track objects. "
            "Each object must include: track_number, artist_name, title, description, tags, lyrics, bpm, key_scale, "
            "time_signature, language, duration, song_model, seed, inference_steps, guidance_scale, shift, "
            "infer_method, sampler_mode, audio_format, auto_score, auto_lrc, return_audio_codes, save_to_library, "
            "quality_profile, "
            "tool_notes, production_team, model_render_notes, prompt_kit_version, target_language, language_notes, "
            "genre_profile, genre_modules, section_map, lyric_density_notes, workflow_mode, negative_control, "
            "quality_checks, troubleshooting_hints, anti_ai_rewrite_notes, tag_coverage, lyric_duration_fit, "
            "caption_integrity, payload_gate_status, and repair_actions. "
            "Include contract_compliance with each locked field marked kept, repaired, or blocked.\n"
            "DURATION CALCULATION FROM LYRICS (MANDATORY):\n"
            "Count the words in the lyrics field for each track, then calculate duration:\n"
            "- Singing (pop/R&B/rock/soul/indie): duration = (word_count / 2.5) + 30 seconds for intro/outro/breaks\n"
            "- Rap (hip-hop/trap/drill/boom-bap): duration = (word_count / 3.5) + 20 seconds for beats/hooks\n"
            "- Sparse (EDM/house/techno/trance/DnB/dubstep): duration = (word_count / 1.5) + 60 seconds for builds/drops\n"
            "- Instrumental ([Instrumental] only): duration = 180-240 seconds\n"
            "- Intro/interlude/skit: duration = 30-90 seconds\n"
            "Round to nearest 10 seconds. Minimum 180s for full songs. Maximum 360s.\n"
            "VALIDATION BEFORE OUTPUT:\n"
            "- 'tags' (caption) contains genre, instruments, mood, vocal type, timbre/production, mix -- NO BPM/key/duration\n"
            "- 'lyrics' uses clear section tags, no instrument descriptions, no stacked bracket descriptors\n"
            "- Chorus/hook is simple enough to remember after one listen (3-8 words)\n"
            "- Lines are short enough to sing or rap (4-10 syllables)\n"
            "- No placeholders, no AI filler, no unfinished sections\n"
            "- Language script is correct for target language\n"
            "- Duration matches lyrics word count (calculated above)\n"
            "- Caption and lyrics do not conflict\n"
            "- tag_coverage and sonic_dna_coverage confirm genre, drums, low-end, melodic identity, vocal delivery, arrangement movement, texture, and mix/master\n"
            "- payload_gate_status is pass or auto_repair, never pass when lyrics are too short or caption leaks prompt text\n"
            f"{schema_rules} "
            f"For planning, song_model may be {ALBUM_FINAL_MODEL}; final audio generation will render all models: "
            f"{', '.join(ALBUM_MODEL_PORTFOLIO_MODELS)}. "
            "artist_name must be an original stage/project name, never a real artist reference. "
            "Also include tool_notes when you changed an artist reference into technique language. JSON array only. "
            "Do not include analysis, thoughts, markdown fences, or prose. The first character must be [ and the last character must be ]."
        ),
        expected_output="Valid JSON array of album tracks with durations calculated from lyrics word count.",
        agent=quality_editor,
        context=[task_concept, task_performance, task_lyrics, task_lyric_edit, task_sonic, task_prompt, task_engineering],
    )

    return Crew(
        agents=[
            executive_producer,
            ar_developer,
            songwriter,
            lyric_editor,
            beat_producer,
            prompt_engineer,
            studio_engineer,
            quality_editor,
        ],
        tasks=[
            task_concept,
            task_performance,
            task_lyrics,
            task_lyric_edit,
            task_sonic,
            task_prompt,
            task_engineering,
            task_json,
        ],
        process=Process.sequential,
        memory=_make_album_memory(ollama_model, embedding_model, read_only=True, planner_provider=planner_provider, embedding_provider=embedding_provider),
        embedder=_local_embedder_config(embedding_provider, embedding_model),
        verbose=CREWAI_VERBOSE,
        step_callback=step_callback,
        task_callback=task_callback,
        output_log_file=output_log_file,
    )


class _AlbumPlanLogs(list[str]):
    def __init__(self, callback: Callable[[str], None] | None = None):
        super().__init__()
        self._callback = callback

    def append(self, item: object) -> None:
        line = str(item)
        super().append(line)
        if self._callback:
            self._callback(line)

    def extend(self, items) -> None:
        for item in items:
            self.append(item)


def _agent_completion_cap(agent_name: str, provider_name: str) -> int:
    defaults = {
        "Album Bible Agent": 1800,
        "Track Blueprint Agent": 1600,
        "Track Settings Agent": 2600,
        "Track BPM Agent": 700,
        "Track Key Agent": 700,
        "Track Time Signature Agent": 700,
        "Track Duration Agent": 700,
        "Track Language Agent": 900,
        "Track Tag List Agent": 700,
        "Track Caption Agent": 800,
        "Track Description Agent": 1500,
        "Track Hook Agent": 1200,
        "Track Performance Agent": 1400,
        "Track Writer Agent": 12000,
        "Track Finalizer Agent": 9000,
        "Track Lyric Continuation Agent": 3200,
    }
    if str(agent_name or "").startswith("Track Lyrics Agent"):
        return max(900, int(os.environ.get("ACEJAM_AGENT_MAX_TOKENS_TRACK_LYRICS_AGENT", "1800")))
    env_key = "ACEJAM_AGENT_MAX_TOKENS_" + re.sub(r"[^A-Z0-9]+", "_", str(agent_name or "agent").upper()).strip("_")
    fallback = CREWAI_LMSTUDIO_MAX_TOKENS if provider_name == "lmstudio" else CREWAI_LLM_NUM_PREDICT
    return max(512, int(os.environ.get(env_key, defaults.get(str(agent_name), min(4500, int(fallback))))))


def _agent_llm_options(provider: str, agent_name: str = "", planner_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    completion_cap = _agent_completion_cap(agent_name, provider_name)
    timeouts = {
        "Album Bible Agent": 60,
        "Track Blueprint Agent": 75,
        "Track BPM Agent": 45,
        "Track Key Agent": 45,
        "Track Time Signature Agent": 45,
        "Track Duration Agent": 45,
        "Track Language Agent": 45,
        "Track Tag List Agent": 45,
        "Track Caption Agent": 45,
        "Track Description Agent": 60,
        "Track Hook Agent": 60,
        "Track Performance Agent": 60,
        "Track Settings Agent": 90,
        "Track Writer Agent": 150,
        "Track Finalizer Agent": 150,
        "Track Lyric Continuation Agent": 75,
    }
    default_timeout = timeouts.get(str(agent_name), 120)
    if str(agent_name or "").startswith("Track Lyrics Agent"):
        default_timeout = 75
    timeout = float(os.environ.get(
        "ACEJAM_AGENT_TIMEOUT_" + re.sub(r"[^A-Z0-9]+", "_", str(agent_name or "agent").upper()).strip("_"),
        default_timeout,
    ))
    source = dict(planner_settings or {})
    if "planner_temperature" not in source and "local_llm_temperature" not in source:
        source["planner_temperature"] = os.environ.get("ACEJAM_PLANNER_TEMPERATURE", ACEJAM_AGENT_TEMPERATURE)
    if "planner_top_p" not in source and "local_llm_top_p" not in source:
        source["planner_top_p"] = os.environ.get("ACEJAM_PLANNER_TOP_P", ACEJAM_AGENT_TOP_P)
    if "planner_context_length" not in source and "local_llm_context_length" not in source and "planner_num_ctx" not in source:
        source["planner_context_length"] = os.environ.get("ACEJAM_PLANNER_CONTEXT_LENGTH", os.environ.get("ACEJAM_AGENT_OLLAMA_NUM_CTX", "8192"))
    options = planner_llm_options_for_provider(
        provider_name,
        source,
        default_max_tokens=completion_cap,
        default_timeout=timeout,
    )
    return options


def _agent_schema_contract(schema_name: str) -> dict[str, Any] | None:
    key = str(schema_name or "").strip()
    if key.startswith("lyrics_part_") and key.endswith("_payload"):
        return {
            "keys": ["part_index", "sections", "lyrics_lines"],
            "example": {"part_index": 1, "sections": [], "lyrics_lines": []},
        }
    return AGENT_EXACT_RESPONSE_SCHEMAS.get(key)


def _agent_block_contract(schema_name: str) -> dict[str, Any] | None:
    key = str(schema_name or "").strip()
    if key.startswith("lyrics_part_") and key.endswith("_payload"):
        return {
            "fields": ["part_index", "sections", "lyrics_lines"],
            "list_fields": {"sections", "lyrics_lines"},
            "number_fields": {"part_index"},
            "required_nonempty": {"part_index", "sections", "lyrics_lines"},
        }
    if key.startswith("track_micro_") and key.endswith("_payload"):
        field = key[len("track_micro_"):-len("_payload")]
        if field == "tag_list":
            return {
                "fields": ["tag_list", "caption_dimensions_covered"],
                "list_fields": {"tag_list", "caption_dimensions_covered"},
                "required_nonempty": {"tag_list"},
                "derived_fields": {"tags": "tag_list_csv"},
            }
        if field == "performance_brief":
            return {
                "fields": ["performance_brief", "negative_control", "genre_profile"],
                "required_nonempty": {"performance_brief"},
            }
        if field == "language":
            return {
                "fields": ["language", "vocal_language"],
                "required_nonempty": {"language", "vocal_language"},
            }
        if field in {"bpm", "duration"}:
            return {
                "fields": [field],
                "number_fields": {field},
                "required_nonempty": {field},
            }
        return {
            "fields": [field],
            "required_nonempty": {field},
        }
    return AGENT_BLOCK_RESPONSE_SCHEMAS.get(key)


def _agent_block_template(schema_name: str) -> str:
    contract = _agent_block_contract(schema_name) or {}
    fields = list(contract.get("fields") or [])
    examples = {
        "album_title": "Album title",
        "one_sentence_concept": "One compact sentence",
        "style_guardrails": "Rule one\nRule two",
        "track_roles": "Track 1 role\nTrack 2 role",
        "title": "Track title",
        "description": "Short description",
        "style": "Genre and production style",
        "vibe": "Mood and energy",
        "narrative": "Narrative purpose",
        "required_phrases": "Phrase one\nPhrase two",
        "tag_list": "West Coast hip-hop\nboom-bap drums\n808 bass\npiano sample motif\nmale rap vocal\ndynamic hook response\ngritty street texture\npunchy polished rap mix",
        "caption_dimensions_covered": "\n".join(ACE_STEP_CAPTION_DIMENSIONS),
        "bpm": str(DEFAULT_BPM),
        "key_scale": DEFAULT_KEY_SCALE,
        "time_signature": "4",
        "duration": "240",
        "section_map": "[Intro]\n[Verse 1]\n[Chorus]\n[Verse 2]\n[Bridge]\n[Final Chorus]\n[Outro]",
        "rationale": "Compact structure note",
        "hook_title": "Hook title",
        "hook_lines": "Hook line one\nHook line two",
        "hook_promise": "What the hook delivers",
        "part_index": "1",
        "sections": "[Intro]\n[Verse 1]\n[Chorus]",
        "lyrics_lines": "[Intro]\nFirst lyric line\n[Verse 1]\nSecond lyric line\n[Chorus]\nHook lyric line",
        "craft_fixes": "Replaced generic phrase with concrete image\nTightened breath length",
        "caption": "West Coast hip-hop, boom-bap drums, 808 bass, piano sample motif, male rap vocal, dynamic hook response, gritty street texture, punchy polished rap mix",
        "performance_brief": "Clear lead vocal, tight cadence, confident pocket, drums and bass forward",
        "negative_control": "no muddy vocals, no gibberish, no prompt text",
        "genre_profile": "Rap-first performance with hip-hop drums, low-end focus, melodic motif, and punchy mix",
        "track_number": "1",
        "language": "en",
    }
    lines: list[str] = []
    for field in fields:
        lines.append(f"******{field}******")
        lines.append(examples.get(field, "content"))
        lines.append(f"******/{field}******")
    return "\n".join(lines)


def _agent_block_instruction(schema_name: str) -> str:
    contract = _agent_block_contract(schema_name)
    if not contract:
        return (
            f"Return only delimiter blocks for {schema_name}. No JSON, no markdown, no commentary, no thoughts. "
            "Every block must open with ******field_name****** and close with ******/field_name****** on its own line."
        )
    fields = list(contract.get("fields") or [])
    list_fields = sorted(str(item) for item in (contract.get("list_fields") or set()))
    scalar_fields = [field for field in fields if field not in set(list_fields)]
    return (
        f"Return only delimiter blocks for {schema_name}. No JSON, no markdown, no commentary, no thoughts.\n"
        "DELIMITER_RESPONSE_CONTRACT:\n"
        "- Your entire response must contain only the required blocks, in this exact order: "
        + ", ".join(fields)
        + "\n"
        "- Open each block on its own line as ******field_name****** and close it as ******/field_name******.\n"
        "- Field names are lowercase snake_case. Delimiters count only when they are the whole line.\n"
        "- Array fields use one item per line: "
        + (", ".join(list_fields) if list_fields else "none")
        + "\n"
        "- Scalar fields keep their text inside one block: "
        + (", ".join(scalar_fields) if scalar_fields else "none")
        + "\n"
        "- Do not output JSON, code fences, bullets outside blocks, analysis, explanations, or extra blocks.\n"
        "ANSWER EXACTLY LIKE THIS BLOCK SHAPE:\n"
        + _agent_block_template(schema_name)
    )


def _agent_json_instruction(schema_name: str) -> str:
    return _agent_block_instruction(schema_name)


_AGENT_BLOCK_DELIMITER_RE = re.compile(r"^\*{6}(/?)([a-z][a-z0-9_]*)\*{6}$")


def _parse_agent_block_payload(raw: str, schema_name: str) -> dict[str, Any]:
    contract = _agent_block_contract(schema_name)
    if not contract:
        raise ValueError(f"block_parse_failed:no_block_contract:{schema_name}")
    text = str(raw or "").strip()
    if not text:
        raise ValueError("block_parse_failed:empty_response")
    if text.startswith("{") or text.startswith("["):
        raise ValueError("block_parse_failed:json_response_not_allowed")
    expected = list(contract.get("fields") or [])
    expected_set = set(expected)
    blocks: dict[str, list[str]] = {}
    current_field = ""
    current_lines: list[str] = []
    for line_number, raw_line in enumerate(str(raw or "").splitlines(), start=1):
        stripped = raw_line.strip()
        delimiter = _AGENT_BLOCK_DELIMITER_RE.fullmatch(stripped)
        if delimiter:
            is_close = bool(delimiter.group(1))
            field = delimiter.group(2)
            if field not in expected_set:
                raise ValueError(f"extra_block:{field}")
            if is_close:
                if not current_field:
                    raise ValueError(f"block_parse_failed:orphan_close:{field}:line_{line_number}")
                if field != current_field:
                    raise ValueError(f"block_parse_failed:mismatched_close:{current_field}!={field}:line_{line_number}")
                # Tolerate duplicates: keep the LAST occurrence. Models often
                # echo an example block before the actual answer; the rule
                # used to be a hard fail, but rejecting the whole response
                # over a benign repetition wastes a full LLM call. Last-wins
                # picks the model's actual output instead of the example.
                blocks[field] = current_lines
                current_field = ""
                current_lines = []
            else:
                if current_field:
                    raise ValueError(f"block_parse_failed:nested_block:{field}:inside:{current_field}:line_{line_number}")
                # Re-opening a previously seen field is allowed (last-wins);
                # we'll overwrite the prior block on the matching close.
                current_field = field
                current_lines = []
            continue
        if current_field:
            current_lines.append(raw_line)
        elif stripped:
            raise ValueError(f"block_parse_failed:text_outside_blocks:line_{line_number}")
    if current_field:
        raise ValueError(f"block_parse_failed:unclosed_block:{current_field}")
    missing = [field for field in expected if field not in blocks]
    if missing:
        raise ValueError("missing_block:" + ",".join(missing))
    list_fields = set(contract.get("list_fields") or set())
    number_fields = set(contract.get("number_fields") or set())
    required_nonempty = set(contract.get("required_nonempty") or set())
    payload: dict[str, Any] = {}
    for field in expected:
        raw_lines = blocks.get(field) or []
        while raw_lines and not str(raw_lines[0]).strip():
            raw_lines = raw_lines[1:]
        while raw_lines and not str(raw_lines[-1]).strip():
            raw_lines = raw_lines[:-1]
        if field in list_fields:
            value = [str(line).strip() for line in raw_lines if str(line).strip()]
            if field in required_nonempty and not value:
                raise ValueError(f"empty_required_block:{field}")
            payload[field] = value
            continue
        value_text = "\n".join(raw_lines).strip()
        if field in required_nonempty and not value_text:
            raise ValueError(f"empty_required_block:{field}")
        if field in number_fields and value_text:
            number_match = re.search(r"-?\d+(?:\.\d+)?", value_text)
            if not number_match:
                raise ValueError(f"invalid_scalar:{field}")
            number_value = float(number_match.group(0))
            payload[field] = int(number_value) if number_value.is_integer() else number_value
        else:
            payload[field] = value_text
    for field, strategy in (contract.get("derived_fields") or {}).items():
        if strategy == "tag_list_csv":
            payload[field] = ", ".join(str(item).strip() for item in (payload.get("tag_list") or []) if str(item).strip())
        elif strategy == "deterministic_block_payload":
            payload[field] = {"deterministic_block_payload": True}
    return payload


def _validate_agent_response_shape(schema_name: str, payload: dict[str, Any]) -> None:
    contract = _agent_schema_contract(schema_name)
    if not contract or not isinstance(payload, dict):
        return
    expected = list(contract["keys"])
    actual = list(payload.keys())
    missing = [key for key in expected if key not in payload]
    extra = [key for key in actual if key not in expected]
    if missing or extra:
        details = []
        if missing:
            details.append("missing_keys=" + ",".join(missing))
        if extra:
            details.append("extra_keys=" + ",".join(extra))
        raise ValueError(f"{schema_name} did not match exact response contract: {'; '.join(details)}")
    if schema_name == "hook_payload":
        hook_lines = payload.get("hook_lines")
        if not isinstance(hook_lines, list):
            raise ValueError("hook_payload did not match exact response contract: hook_lines_must_be_array")
        tagged = [str(line) for line in hook_lines if re.search(r"\[[^\]]+\]", str(line or ""))]
        if tagged:
            raise ValueError("hook_payload did not match exact response contract: hook_lines_must_not_contain_section_tags")


def _agent_system_prompt(agent_name: str) -> str:
    name = str(agent_name or "AceJAM Agent")
    common = (
        f"You are {name}, a single-purpose ACE-Step album planning agent.\n"
        "Return only the requested delimiter blocks. No JSON, prose, markdown, thoughts, or extra blocks.\n"
        "Keep locked fields unchanged. Producer credit is metadata, never a lyric.\n"
        "ACE-Step fields are separate: caption=sound, lyrics=temporal script, BPM/key/time/duration=metadata.\n"
    )
    if name == "Album Bible Agent":
        return common + (
            "Task: album DNA only. Do not write lyrics or final captions. Keep it compact.\n"
        )
    if name == "Track Blueprint Agent":
        return common + (
            "Task: one track blueprint only. Preserve contract fields. No lyrics.\n"
        )
    if name in {"Tag Agent", "Track Tag List Agent", "Sonic DNA Agent"}:
        return common + (
            "Task: sonic DNA tags only. Cover genre/style, rhythm/groove, instruments, vocal style, mood, arrangement energy, mix. "
            "No BPM, key, duration, names, title, story, or lyrics.\n"
        )
    if name in {"Caption Agent", "Track Caption Agent"}:
        return common + (
            "Task: final ACE-Step caption only. Comma-separated sound prompt under 512 chars. "
            "Only sonic traits; no metadata, names, title, story, or lyrics.\n"
        )
    if name.startswith("Track ") and name.endswith("Agent") and "Lyrics" not in name and name not in {"Track Writer Agent", "Track Finalizer Agent"}:
        return common + (
            "Task: exactly one micro-setting. Do not plan the song and do not write lyrics.\n"
        )
    if name.startswith("Track Lyrics Agent") or name == "Track Lyric Continuation Agent":
        return common + (
            "Task: lyrics only. Use ONLY_ALLOWED_SECTION_TAGS exactly and once each. "
            "Do not write forbidden/previous section tags. Lines should be performable, usually 6-10 syllables. "
            "No caption, metadata, producer names, placeholders, or escaped literal \\n.\n"
        )
    if name in {"Track Writer Agent", "Track Finalizer Agent"}:
        return common + (
            "Task: assemble one ACE-Step track JSON from existing fields. Do not rewrite creative content.\n"
        )
    return common + "Task scope: follow the provided schema for this one agent call only.\n"


def _agent_full_system_prompt(
    *,
    agent_name: str,
    schema_name: str,
    extra_system: str = "",
    debug_options: dict[str, Any] | None = None,
) -> str:
    """Build the full per-agent system prompt: per-agent task scope from
    `_agent_system_prompt(agent_name)`, plus the album-mode `extra_system`
    block (which carries `prompt_kit_system_block("album")` with the tag
    library, producer/songwriter/anti-pattern cookbooks and worked examples),
    plus the thinking directive, plus the schema-specific JSON instruction.

    Earlier this function recursed into itself (`_agent_full_system_prompt`
    calling `_agent_full_system_prompt`) which crashed CrewAI Micro Tasks
    with `RecursionError: maximum recursion depth exceeded` the moment the
    Album Intake Agent tried to fire — keeping the album wizard empty even
    after a successful preflight. The fix mirrors the working composition
    in `_agent_json_call`."""
    system_prompt = _agent_system_prompt(agent_name)
    if extra_system:
        system_prompt += "\n" + str(extra_system).strip()
    if _truthy((debug_options or {}).get("planner_thinking"), False):
        system_prompt += "\nPlanner thinking is enabled: you may reason internally, but final visible content must be delimiter blocks only."
    else:
        system_prompt += "\nPlanner thinking is disabled: do not emit <think>, hidden reasoning, chain-of-thought, markdown, or prose."
    system_prompt += "\n" + _agent_json_instruction(schema_name)
    return system_prompt


def _call_agent_llm(
    *,
    agent_name: str,
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    options: dict[str, Any],
    logs: list[str],
    debug_options: dict[str, Any],
    attempt: int,
    json_format: bool = True,
) -> str:
    provider_name = normalize_provider(provider)
    user_content = user_prompt
    planner_thinking = _truthy((debug_options or {}).get("planner_thinking"), False)
    no_think_directive = str((debug_options or {}).get("planner_no_think_directive") or CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE or "/no_think").strip()
    if not planner_thinking and no_think_directive and no_think_directive not in user_content:
        user_content = f"{no_think_directive}\n\n{user_content}"
    elif provider_name == "lmstudio" and CREWAI_LMSTUDIO_DISABLE_THINKING and CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE:
        user_content = f"{CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE}\n\n{user_content}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt_chars = len(system_prompt) + len(user_content)
    _print_agent_io(
        debug_options,
        f"{agent_name.replace(' ', '_')}_prompt_attempt_{attempt}",
        {
            "agent": agent_name,
            "provider": provider_name,
            "model": model_name,
            "prompt_chars": prompt_chars,
            "system_chars": len(system_prompt),
            "user_chars": len(user_content),
            "messages": messages,
            "options": options,
        },
    )
    _append_album_debug_jsonl(
        debug_options,
        "03_agent_prompts.jsonl",
        {
            "agent": agent_name,
            "provider": provider_name,
            "model": model_name,
            "attempt": attempt,
            "prompt_chars": prompt_chars,
            "system_chars": len(system_prompt),
            "user_chars": len(user_content),
            "messages": messages,
            "options": options,
        },
    )
    logs.append(
        f"AceJAM Agent call: {agent_name} attempt {attempt} via {provider_label(provider_name)} "
        f"(prompt_chars={prompt_chars}, system={len(system_prompt)}, user={len(user_content)})."
    )
    started = time.perf_counter()
    raw = local_llm_chat_completion(provider_name, model_name, messages, options=options, json_format=json_format)
    elapsed = round(time.perf_counter() - started, 3)
    _print_agent_io(debug_options, f"{agent_name.replace(' ', '_')}_raw_response_attempt_{attempt}", raw)
    text = _strip_thinking_blocks(raw)
    _append_album_debug_jsonl(
        debug_options,
        "04_agent_responses.jsonl",
        {
            "agent": agent_name,
            "provider": provider_name,
            "model": model_name,
            "attempt": attempt,
            "elapsed": elapsed,
            "response_chars": len(text),
            "raw_response": str(raw or ""),
            "response": text,
        },
    )
    logs.append(f"AceJAM Agent response: {agent_name} {len(text)} chars in {elapsed}s (parse pending).")
    return text


def _agent_json_call(
    *,
    agent_name: str,
    provider: str,
    model_name: str,
    user_prompt: str,
    logs: list[str],
    debug_options: dict[str, Any],
    schema_name: str,
    extra_system: str = "",
    max_retries: int | None = None,
) -> dict[str, Any]:
    retries = ACEJAM_AGENT_BLOCK_RETRIES if max_retries is None else int(max_retries)
    attempts = max(1, retries + ACEJAM_AGENT_EMPTY_RETRIES + 1) if max_retries is None else max(1, retries + 1)
    options = _agent_llm_options(provider, agent_name, debug_options)
    system_prompt = _agent_system_prompt(agent_name)
    if extra_system:
        system_prompt += "\n" + str(extra_system).strip()
    if _truthy((debug_options or {}).get("planner_thinking"), False):
        system_prompt += "\nPlanner thinking is enabled: you may reason internally, but final visible content must be delimiter blocks only."
    else:
        system_prompt += "\nPlanner thinking is disabled: do not emit <think>, hidden reasoning, chain-of-thought, markdown, or prose."
    system_prompt += "\n" + _agent_json_instruction(schema_name)
    prompt = user_prompt
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            raw = _call_agent_llm(
                agent_name=agent_name,
                provider=provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                options=options,
                logs=logs,
                debug_options=debug_options,
                attempt=attempt,
                json_format=False,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _append_album_debug_jsonl(
                debug_options,
                "04_agent_responses.jsonl",
                {"agent": agent_name, "attempt": attempt, "error": last_error},
            )
            if attempt >= attempts:
                break
            prompt = (
                f"{user_prompt}\n\nRECOVERY: The previous {agent_name} call raised {last_error}. "
                "Return the requested delimiter blocks now."
            )
            continue
        if not raw.strip():
            last_error = "empty response"
            if attempt >= attempts:
                break
            logs.append(f"AceJAM Agent empty response: {agent_name}; retrying delimiter block output.")
            prompt = (
                f"{user_prompt}\n\nRECOVERY: Your previous response was empty. "
                "Return the requested delimiter blocks only. Do not call tools."
            )
            continue
        try:
            payload = _parse_agent_block_payload(raw, schema_name)
            if not isinstance(payload, dict):
                raise ValueError("block_parse_failed:root_was_not_object")
            _validate_agent_response_shape(schema_name, payload)
            parsed = _coerce_agent_lyrics_payload(payload)
            _print_agent_io(debug_options, f"{agent_name.replace(' ', '_')}_parsed_blocks_attempt_{attempt}", parsed)
            _append_album_debug_jsonl(
                debug_options,
                "04_agent_responses.jsonl",
                {"agent": agent_name, "attempt": attempt, "parsed_blocks": parsed},
            )
            logs.append(f"AceJAM Agent parsed delimiter blocks: {agent_name} attempt {attempt} ok.")
            return parsed
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _append_album_debug_jsonl(
                debug_options,
                "04_agent_responses.jsonl",
                {
                    "agent": agent_name,
                    "attempt": attempt,
                    "block_parse_error": last_error,
                    "response_preview": _monitor_preview(raw, 500),
                },
            )
            if attempt >= attempts:
                break
            logs.append(f"AceJAM Agent block repair: {agent_name}; {last_error}.")
            prompt = (
                f"{user_prompt}\n\nBLOCK REPAIR: The previous response failed delimiter-block parsing: {last_error}. "
                "Return exactly the required delimiter blocks, in order, with no JSON, markdown, commentary, or extra text.\n"
                "EXPECTED_BLOCK_SHAPE:\n"
                f"{_agent_block_template(schema_name)}"
            )
    raise AceJamAgentError(f"{agent_name} failed to produce valid delimiter blocks after {attempts} attempt(s): {last_error}")


def _crewai_micro_llm_kwargs(
    model_name: str,
    provider: str,
    agent_name: str,
    planner_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    options = _agent_llm_options(provider_name, agent_name, planner_settings)
    seed = options.get("seed")
    try:
        seed_value = int(seed) if seed not in (None, "") else None
    except Exception:
        seed_value = None
    max_tokens = int(options.get("num_predict") or options.get("max_tokens") or _agent_completion_cap(agent_name, provider_name))
    common: dict[str, Any] = {
        "model": str(model_name or "").strip(),
        "temperature": float(options.get("temperature") if options.get("temperature") is not None else ACEJAM_AGENT_TEMPERATURE),
        "top_p": float(options.get("top_p") if options.get("top_p") is not None else ACEJAM_AGENT_TOP_P),
        "max_tokens": max_tokens,
        "timeout": float(options.get("timeout") or CREWAI_LLM_TIMEOUT_SECONDS),
    }
    if seed_value is not None:
        common["seed"] = seed_value
    top_k = options.get("top_k")
    repeat_penalty = options.get("repeat_penalty")
    if provider_name == "lmstudio":
        params: dict[str, Any] = {}
        if top_k is not None:
            params["top_k"] = int(top_k)
        if repeat_penalty is not None:
            params["repeat_penalty"] = float(repeat_penalty)
        payload = {
            **common,
            "provider": "openai",
            "base_url": lmstudio_api_base_url(),
            "api_key": os.environ.get("LMSTUDIO_API_TOKEN", "lm-studio"),
        }
        if params:
            payload["additional_params"] = params
        return payload
    ollama_options: dict[str, Any] = {
        "num_ctx": int(options.get("num_ctx") or CREWAI_LLM_CONTEXT_WINDOW),
        "num_predict": max_tokens,
    }
    if top_k is not None:
        ollama_options["top_k"] = int(top_k)
    if repeat_penalty is not None:
        ollama_options["repeat_penalty"] = float(repeat_penalty)
    if seed_value is not None:
        ollama_options["seed"] = seed_value
    # `think: False` is the correct setting for ALL Ollama models including
    # Qwen3 :thinking variants. Direct probes confirmed: `think: True`
    # routes the answer to `message.thinking` and leaves `message.content`
    # empty — litellm only reads content, so CrewAI raises "Invalid response
    # from LLM call - None or empty." We pair `think: False` with the
    # `/no_think` directive injected into the user message inside
    # `_crewai_micro_block_call` so the model writes its answer directly to
    # `content` even when its tag advertises a reasoning mode.
    return {
        **common,
        "provider": "ollama",
        "base_url": OLLAMA_BASE_URL,
        "api_base": _ollama_v1_base_url(),
        "additional_params": {
            "extra_body": {
                "think": False,
                "options": ollama_options,
            },
        },
    }


def _build_ollama_native_llm_class():
    """Build the CrewAI-compatible Ollama-native LLM class. We construct it
    inside a factory because it must subclass `crewai.llms.base_llm.BaseLLM`
    for pydantic validation in `crewai.Agent` to pass, and crewai is
    imported lazily.

    Why subclass BaseLLM (not the public LLM facade): crewai.LLM uses a
    custom __new__ factory that returns a different concrete class
    (OpenAICompatibleCompletion) regardless of the cls argument. Subclassing
    LLM therefore fails with TypeError: LLM.__new__() missing argument.
    BaseLLM is the actual abstract base accepted by Agent.llm validation
    (`Annotated[str | BaseLLM | None, ...]`).
    """
    from crewai.llms.base_llm import BaseLLM as _BaseLLM

    class _OllamaNativeLLM(_BaseLLM):
        """BaseLLM subclass that hits Ollama's native /api/chat endpoint
        directly via httpx. Bypasses CrewAI's OpenAI-compat layer which
        drops the `reasoning` field for Qwen :thinking / DeepSeek-R1 /
        QwQ models — that bug left message.content empty and crashed every
        album agent with 'Invalid response from LLM call - None or empty.'
        Hitting /api/chat with `think: False` + `/no_think` directive
        produces a populated content field directly (verified by probe)."""

        llm_type: str = "ollama_native"

        # Pydantic-friendly attribute fields. Internal Ollama config kept
        # in private attributes so they bypass model validation.
        _ollama_base_url: str = ""
        _ollama_options_raw: dict[str, Any] = {}
        _ollama_timeout: float = 600.0

        def __init__(self, model_name: str, base_url: str, options: dict[str, Any], timeout: float = 600.0, **extra: Any):
            super().__init__(model=str(model_name or "").strip(), provider="ollama_native", **extra)
            object.__setattr__(self, "_ollama_base_url", str(base_url or "").rstrip("/"))
            object.__setattr__(self, "_ollama_options_raw", dict(options or {}))
            object.__setattr__(self, "_ollama_timeout", float(timeout))

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            try:
                return int(self._ollama_options_raw.get("num_ctx") or CREWAI_LLM_CONTEXT_WINDOW)
            except Exception:
                return CREWAI_LLM_CONTEXT_WINDOW

        @staticmethod
        def _ensure_no_think_directive(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not messages:
                return messages
            adjusted = [dict(m) for m in messages]
            for i in range(len(adjusted) - 1, -1, -1):
                if adjusted[i].get("role") == "user":
                    content = str(adjusted[i].get("content") or "")
                    if "/no_think" not in content:
                        adjusted[i]["content"] = "/no_think\n\n" + content
                    break
            return adjusted

        def call(
            self,
            messages,
            tools=None,
            callbacks=None,
            available_functions=None,
            from_task=None,
            from_agent=None,
            response_model=None,
        ):
            import httpx

            if isinstance(messages, str):
                payload_messages = [{"role": "user", "content": messages}]
            else:
                payload_messages = []
                for m in messages or []:
                    if isinstance(m, dict):
                        payload_messages.append({"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")})
                    else:
                        payload_messages.append({"role": str(getattr(m, "role", "user")), "content": str(getattr(m, "content", ""))})

            payload_messages = self._ensure_no_think_directive(payload_messages)

            body = {
                "model": self.model,
                "messages": payload_messages,
                "think": False,
                "stream": False,
                "options": dict(self._ollama_options_raw),
            }

            url = f"{self._ollama_base_url}/api/chat"
            try:
                with httpx.Client(timeout=self._ollama_timeout) as client:
                    response = client.post(url, json=body)
                    response.raise_for_status()
                    data = response.json()
            except Exception as exc:
                raise RuntimeError(f"OllamaNativeLLM HTTP call failed: {type(exc).__name__}: {exc}") from exc

            message = data.get("message") if isinstance(data, dict) else None
            if not isinstance(message, dict):
                return ""
            content = (message.get("content") or "").strip()
            if not content:
                content = (message.get("thinking") or "").strip()
            if not content:
                return ""
            stripped = _strip_thinking_blocks(content)
            return stripped if stripped.strip() else content

    return _OllamaNativeLLM


def _make_crewai_micro_llm(
    model_name: str,
    provider: str,
    agent_name: str,
    planner_settings: dict[str, Any] | None = None,
):
    provider_name = normalize_provider(provider)
    # Ollama: use our native-endpoint LLM that handles :thinking variants
    # correctly. CrewAI's OpenAI-compat layer drops the `reasoning` field
    # for those models, leaving content empty and crashing the agent.
    if provider_name == "ollama":
        kwargs = _crewai_micro_llm_kwargs(model_name, provider, agent_name, planner_settings)
        ollama_options = (kwargs.get("additional_params") or {}).get("extra_body", {}).get("options") or {}
        ollama_options.setdefault("temperature", kwargs.get("temperature", 0.45))
        ollama_options.setdefault("top_p", kwargs.get("top_p", 0.92))
        ollama_options.setdefault("num_predict", kwargs.get("max_tokens", 8192))
        ollama_native_cls = _build_ollama_native_llm_class()
        return ollama_native_cls(
            model_name=model_name,
            base_url=OLLAMA_BASE_URL,
            options=ollama_options,
            timeout=float(kwargs.get("timeout") or CREWAI_LLM_TIMEOUT_SECONDS),
        )

    from crewai import LLM

    llm = LLM(**_crewai_micro_llm_kwargs(model_name, provider, agent_name, planner_settings))
    llm.supports_function_calling = lambda: False

    # Patch `LLM.call` so Qwen :thinking / DeepSeek-R1 / QwQ variants stop
    # crashing CrewAI with "Invalid response from LLM call - None or empty."
    # Ollama's OpenAI-compat (/v1/chat/completions) endpoint routes the
    # answer for those tagged reasoning models into a non-standard
    # `reasoning` field and leaves `message.content` empty. Litellm preserves
    # the field verbatim. Our wrapper invokes the original call once, then
    # if the returned string is empty issues a single follow-up via
    # litellm.completion to read the full message dict and pull the answer
    # out of `reasoning` (or legacy `thinking`).
    original_call = llm.call

    def _diag(msg: str) -> None:
        """Write reasoning-fallback diagnostics to both stdout (Pinokio
        terminal) and a file the dev can tail from outside the app process."""
        try:
            print(msg, flush=True)
        except Exception:
            pass
        try:
            with open("/tmp/album_crew_diag.log", "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass

    def _call_with_reasoning_fallback(messages=None, *args, **kwargs):
        response = original_call(messages, *args, **kwargs) if messages is not None else original_call(*args, **kwargs)
        non_empty = bool(response) and isinstance(response, str) and bool(response.strip())
        _diag(
            f"[REASONING_FALLBACK] agent={agent_name} response_type={type(response).__name__} "
            f"len={len(response) if isinstance(response, str) else -1} non_empty={non_empty}"
        )
        if non_empty:
            return response
        # Empty/None — try to extract from reasoning or thinking field via
        # a follow-up litellm.completion that captures the full message dict.
        try:
            import litellm

            pending = messages if messages is not None else (args[0] if args else None) or kwargs.get("messages")
            if pending is None:
                _diag(f"[REASONING_FALLBACK] agent={agent_name} skipped: no messages to retry")
                return response
            completion_kwargs = {
                "model": llm.model,
                "messages": pending,
                "temperature": getattr(llm, "temperature", None),
                "max_tokens": getattr(llm, "max_tokens", None),
                "api_base": getattr(llm, "api_base", None),
                "timeout": getattr(llm, "timeout", None),
            }
            completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}
            _diag(
                f"[REASONING_FALLBACK] agent={agent_name} retrying via litellm.completion model={completion_kwargs.get('model')}"
            )
            raw = litellm.completion(**completion_kwargs)
            choice = raw.choices[0] if hasattr(raw, "choices") and raw.choices else None
            if choice is None:
                _diag(f"[REASONING_FALLBACK] agent={agent_name} retry got no choices")
                return response
            msg = getattr(choice, "message", None)
            if msg is None and isinstance(choice, dict):
                msg = choice.get("message")
            if msg is None:
                _diag(f"[REASONING_FALLBACK] agent={agent_name} retry message is None")
                return response
            keys_seen: list[str] = []
            content = ""
            source_used = "content"
            if isinstance(msg, dict):
                keys_seen = list(msg.keys())
                content = (msg.get("content") or "").strip()
                if not content:
                    content = (msg.get("reasoning") or "").strip()
                    source_used = "reasoning"
                if not content:
                    content = (msg.get("thinking") or "").strip()
                    source_used = "thinking"
            else:
                keys_seen = [k for k in dir(msg) if not k.startswith("_")][:15]
                content = (getattr(msg, "content", "") or "").strip()
                if not content:
                    content = (getattr(msg, "reasoning", "") or "").strip()
                    source_used = "reasoning"
                if not content:
                    content = (getattr(msg, "thinking", "") or "").strip()
                    source_used = "thinking"
                if not content:
                    # Last resort: dump message repr to see all available fields
                    _diag(f"[REASONING_FALLBACK] agent={agent_name} message_repr={repr(msg)[:500]}")
            _diag(
                f"[REASONING_FALLBACK] agent={agent_name} message_keys={keys_seen} "
                f"source={source_used} extracted_len={len(content)} preview={content[:120]!r}"
            )
            if content:
                # Strip <think>...</think> wrapper if model emitted both
                # reasoning channel and inline tags.
                stripped = _strip_thinking_blocks(content)
                if stripped.strip():
                    return stripped
                return content
        except Exception as exc:
            _diag(f"[REASONING_FALLBACK] agent={agent_name} fallback exception: {type(exc).__name__}: {exc}")
        return response

    llm.call = _call_with_reasoning_fallback  # type: ignore[method-assign]
    return llm


def _crewai_micro_block_guardrail(schema_name: str):
    def _guardrail(output: Any) -> Tuple[bool, Any]:
        try:
            raw = _task_output_raw_text(output)
            payload = _parse_agent_block_payload(_strip_thinking_blocks(raw), schema_name)
            _validate_agent_response_shape(schema_name, payload)
            return True, output
        except Exception as exc:
            return False, f"block_parse_failed:{type(exc).__name__}: {exc}"

    _guardrail.__annotations__["return"] = Tuple[bool, Any]
    _guardrail.__annotations__["output"] = Any
    return _guardrail


# Per-agent CrewAI personas. Each entry is a (role, goal, backstory) tuple
# tailored to the agent's job. Default fallback is the generic worker persona;
# specialised agents get domain-specific framing so the LLM acts like the
# right kind of expert (topline lyric writer for the lyrics agent, sonic
# engineer for the tag agent, etc.). The backstory carries the non-negotiable
# craft rules; the full ACE-Step reference still ships via system_rules().
_AGENT_PERSONAS: dict[str, tuple[str, str, str]] = {
    "Album Intake Agent": (
        "Album A&R Intake Producer",
        "Distil the user concept into an album bible: title, theme, emotional arc, target audience, sonic identity, motif words, and locked tracks. Never invent producers the user did not ask for. Never copy AI-cliche phrasing.",
        "You run intake for award-level studio releases. You read what the artist actually wants — concrete details, real references, the locked phrases — and you keep generic AI slop out of the bible.",
    ),
    "Track Concept Agent": (
        "Track Concept Producer (concrete style + vibe + narrative required)",
        (
            "Pitch one track that earns its slot on this album. ALL FOUR fields are REQUIRED non-empty: "
            "title (specific phrase), description (1 sentence hit-angle), style (one specific stack — examples: "
            "'modern trap with sub-808 + auto-tune lead', 'G-funk with Minimoog bass + talkbox lead + 90s polish', "
            "'boom-bap with chopped soul sample + dusty SP1200 drums', 'cinematic drill with sliding 808 + "
            "horror-string sample' — NEVER one generic word like 'rap' or 'pop'), vibe (emotional + textural — "
            "'menacing late-night confidence', 'triumphant horn-stab anthem', 'claustrophobic NYC street-noir'), "
            "narrative (1-2 sentences with a concrete actor + action + stake — 'A delivery driver counts the same "
            "twelve corners every night and starts narrating each light he runs'). Every track changes something "
            "the previous tracks did not."
        ),
        (
            "You write concept docs A&R execs read at lunch. Concrete title, single-sentence thesis, no genre mush, "
            "no 'inspirational journey' filler. The four fields you control (title/description/style/vibe/narrative) "
            "feed every other agent in the album crew — empty values cascade into empty captions, generic hooks, "
            "and unmotivated lyrics. Style is your most-stolen field: a Tag Engineer downstream copies its instrument/era "
            "tokens, a Hook Writer reads its mood, a Lyric Writer reads its narrative. Land all four with the kind of "
            "specificity Eminem's Slim Shady origin spec, Kendrick's m.A.A.d city day-in-the-life, or Travis Scott's "
            "Astroworld park-ride concept landed."
        ),
    ),
    "BPM Agent": (
        "Tempo Producer",
        "Pick the BPM that matches the producer reference, era and energy from the brief. Default range hip-hop 70-110 BPM, pop 95-128, ballad 60-85, trap 130-160 (half-time = 65-80 perceived).",
        "You own the click. BPM derives from the producer cookbook era + groove tag, never from a coin flip. 90 BPM Dre G-funk, 78 BPM Chronic 2001, 140 BPM Metro half-time, 92 BPM Pete Rock.",
    ),
    "Key Agent": (
        "Key & Scale Producer",
        "Pick a key/scale that fits the mood and producer reference. Minor for menacing/dark/cinematic; major for triumphant/anthemic/summer.",
        "You match key to vibe, not to randomness. C minor for Mobb Deep noir, A minor for Dre 2001 menace, G major for Just Blaze triumph, F# minor for Stoupe cinematic.",
    ),
    "Time Signature Agent": (
        "Meter Producer",
        "Default 4/4 unless the genre clearly calls for 3/4, 6/8 or 12/8.",
        "You keep the meter normal. 99% of hits are 4/4. Only switch when the user explicitly asks for waltz, ballad triplets or jazz odd meter.",
    ),
    "Duration Agent": (
        "Duration Producer",
        "Pick the duration from the track role and album arc: intro/skit 60-90s, single 180-240s, epic/closer 240-360s. Never pad to a round number.",
        "You match length to song role. Lead singles run 180-240s; cinematic closers can push 300s; skits stay short. Hit albums never have 9 tracks all at exactly 3:00.",
    ),
    "Tag Agent": (
        "ACE-Step Sonic Tags Engineer (2024-2026 chart-aware)",
        "Build a 12-24 token caption stack covering all six dimensions: drums (kick + snare + hat triad), bass character, sample/source + treatment, mix treatment, era marker, groove word. No bare 'sample'. No producer names in tags. Default to MODERN era markers when the request is contemporary ('2020s rap', '2010s trap', '2024 pop polish', 'retro disco-pop revival') — only use 'classic 90s G-funk' when the user explicitly asks for vintage.",
        "You think in ACE-Step's vocabulary. Every caption gets the drum triad, bass character, sample-source-with-treatment, mix descriptor, era token, groove word. You stack like Pete Rock built beats AND like Mustard / Pi'erre Bourne / Finneas / Carter Lang stack 2024-2026 productions — six layers, none missing. You know modern templates: Mustard ratchet (hyphy bass + finger-snaps + violin sample), Central Cee UK drill (sliding 808 + acoustic guitar chop + 25ms retune), Finneas bedroom pop (4-chord palette + sub-bass + close-mic), retro disco-pop revival (vintage bass walk + syncopated guitar chops + displaced-downbeat melody).",
    ),
    "Section Map Agent": (
        "Arrangement Producer",
        "Build the bracketed section sequence for this duration + genre. Rap tracks include [Verse - rap] sections; hooks repeat verbatim; bridges introduce NEW content; long tracks use [Beat Switch] or [Bridge - melodic rap] to keep momentum.",
        "You arrange songs the way mixtape engineers arrange them. Verse/Hook/Verse/Hook/Bridge/Final Hook for pop. Intro/Verse/Hook/Verse/Hook/Bridge/Verse/Hook/Outro for rap. Long tracks earn their length with section variety, not repetition.",
    ),
    "Hook Agent": (
        "Topline Hook Writer (2024-2026 chart-craft)",
        "Write a 2-4 line hook that passes the TikTok 30-second test and the hum-test: by 0:15 a stranger should grasp the thesis. Title-drop on chorus line 1 with vowel-lock rhyme through 3-4 successive end-words. Anthem-shout cadence (4-syllable chant a stadium can sing). Concrete proper noun (brand/place/name). One displaced-downbeat melody phrase (start on beat 2 not the 1). No cliche image bank, no polar 'I am X / I am Y' binary.",
        "You write 2024-2026 chart hooks the way Sabrina Carpenter writes 'that's that me, espresso', Kendrick writes 'they not like us', Billie Eilish writes 'birds of a feather, we should stick together I know', Beyonce writes 'this ain't Texas, ain't no hold 'em'. Hook FIRST, not buried. Vowel-lock stacked, concrete proper nouns, displaced downbeat, anthem-shout cadence. No 'neon dreams', no 'I am the saint I am the sinner' Nick-Cave-flagged AI tells.",
    ),
    "Track Lyrics Agent Part 1": (
        "Tier-1 Lyric Writer Part 1 (2024-2026 chart-craft)",
        "Write the first lyric block exactly matching ONLY_ALLOWED_SECTION_TAGS. Modern rap verses are 12 bars minimum (16+ for storytelling tracks Kendrick-concept / Nas-narrative); pop verses 8-12 lines. Stack multisyllabic mosaic rhymes with slant-dominant flow + perfect-rhyme landings. Every verse changes something. Force ONE concrete proper noun per verse (brand / place / name / time / object). Allow ONE deliberate metric overflow per song (Antonoff/Swift rant technique). No cliche phrases. No polar 'I am X / I am Y' binary.",
        "You ghost-write for 2024-2026 chart-toppers. Sabrina Carpenter humor + brand drops (Mountain Dew, Dior, jet-lag CVS), Kendrick 12-bar diss-track punch, Billie Eilish/Finneas conversational close-mic intimacy with idiom-flip titles, Central Cee UK-drill melodic-rap hybrid, Morgan Wallen acoustic-percussion country storytelling. Behind those, Eminem rhyme-stacking + Nas Hemingway-line specificity + 2Pac empathetic clarity stay as the craft floor. You never ship 'I feel sad', 'my heart is broken', 'we all', 'shattered dreams', 'I am the saint I am the sinner'. You write proper nouns, contradictions in the same verse (confidence + jet-lag), conversational micro-overflow lines.",
    ),
    "Track Lyrics Agent Part 2": (
        "Tier-1 Lyric Writer Part 2 (2024-2026 chart-craft)",
        "Continue the lyric for the next section group, exactly matching ONLY_ALLOWED_SECTION_TAGS. Never repeat content from previous parts. Verse 2 ESCALATES: new scene, new witness, time jump, OR reversal (never paraphrase V1). Rap verses minimum 12 bars (16+ for storytelling). Hook lines repeat verbatim from HOOK_LINES_TO_USE. Force one concrete proper noun per verse.",
        "You ghost-write for 2024-2026 chart hits. V2 must add what V1 didn't: 'Espresso' V2 jumps from after-party to chapel-to-ICU-to-CVS at dawn. 'Cruel Summer' V2 zooms to drunk-in-back-of-car detail. 'Birds of a Feather' V2 lands the songwriter's thesis. Same craft rules as Part 1: rhyme stacking, concrete proper nouns, contradictions, no cliches, every verse changes something, hook verbatim every chorus pass.",
    ),
    "Track Lyrics Agent Part 3": (
        "Tier-1 Lyric Writer Part 3 (2024-2026 chart-craft)",
        "Write the closing lyric block exactly matching ONLY_ALLOWED_SECTION_TAGS. Bridge OPTIONAL but if present write the 'rant bridge' (Antonoff/Swift): stream-of-consciousness, conversational diction, intrusive thoughts blended with metaphor, end on a shouted single-line thesis. Final chorus repeats hook verbatim. Outro lands the thesis in 1-3 short lines.",
        "You ghost-write for 2024-2026 chart hits. Closing sections need either a rant-bridge (Cruel Summer style: shout-line ending) OR straight to final hook + outro (Birds of a Feather has no bridge, Not Like Us has no bridge — modern norm). Outro is conversational, short, leaves the listener wanting one more spin.",
    ),
    "Caption Agent": (
        "Final Caption Polisher",
        "Polish the final 12-24 token ACE-Step caption. Cover all six dimensions (drums-triad, bass, sample-source + treatment, mix, era, groove). No bare 'sample'. No producer names. No BPM/key/title/lyric leakage. Stay under 512 chars.",
        "You're the caption-engineer who writes the prompt that ships to ACE-Step. You know exactly which words the model rewards (compound terms, era markers, groove words). No filler tags like 'modern production' or 'tight groove' without concrete instruments behind them.",
    ),
    "Performance Agent": (
        "Performance & Mix Brief Writer",
        "Write a vocal performance brief: persona, cadence, ad-lib placement, harmony stacking, energy curve, mix notes. Concrete instructions a session vocalist + mix engineer can act on.",
        "You write briefs that vocalists print and tape to the booth. Specific persona, specific cadence per section, specific ad-lib placement (not 'add ad-libs' but '(yeah!) on bar 4 of verse 2, (skrrt) on the second hook'). Mix notes call out which instruments duck for the verse and rise for the hook.",
    ),
}


def _agent_persona(agent_name: str, schema_name: str) -> tuple[str, str, str]:
    """Resolve per-agent (role, goal, backstory) for CrewAI. Falls back to a
    sensible specialist persona when the exact agent_name is not in the
    cookbook so new agents still get reasonable framing."""
    key = str(agent_name or "").strip()
    if key in _AGENT_PERSONAS:
        return _AGENT_PERSONAS[key]
    # Wildcard match for "Track Lyrics Agent Part N" beyond Part 3
    if key.startswith("Track Lyrics Agent Part"):
        base = _AGENT_PERSONAS.get("Track Lyrics Agent Part 2")
        if base:
            return base
    if key.startswith("Caption"):
        return _AGENT_PERSONAS.get("Caption Agent", ("Caption Specialist", "", ""))
    if key.startswith("Tag"):
        return _AGENT_PERSONAS.get("Tag Agent", ("Tag Specialist", "", ""))
    # Generic fallback — still better than the old "tiny worker" line.
    return (
        key or "AceJAM Album Worker",
        f"Return only delimiter blocks for {schema_name} with award-level production specificity and lyric craft.",
        (
            "You are a hit-album specialist for AceJAM. You apply the appended "
            "ACE-Step reference (tag library, producer cookbook, songwriter craft, "
            "anti-patterns, worked examples) to every output. You never ship AI "
            "slop, never paraphrase locked user fields, never put producer names "
            "in caption."
        ),
    )


def _agent_max_iter(agent_name: str) -> int:
    """Creative agents need iteration room; metadata agents can stay at 1."""
    creative_keys = ("Lyrics", "Hook", "Caption", "Section Map", "Track Concept", "Tag", "Performance")
    if any(key in (agent_name or "") for key in creative_keys):
        return 4
    return 1


def _crewai_micro_block_call(
    *,
    agent_name: str,
    provider: str,
    model_name: str,
    user_prompt: str,
    logs: list[str],
    debug_options: dict[str, Any],
    schema_name: str,
    extra_system: str = "",
    max_retries: int | None = None,
) -> dict[str, Any]:
    from crewai import Agent, Crew, Process, Task

    provider_name = normalize_provider(provider)
    retries = ACEJAM_AGENT_BLOCK_RETRIES if max_retries is None else int(max_retries)
    attempts = max(1, retries + ACEJAM_AGENT_EMPTY_RETRIES + 1) if max_retries is None else max(1, retries + 1)
    planner_settings = planner_llm_settings_from_payload(debug_options or {})
    options = _agent_llm_options(provider_name, agent_name, planner_settings)
    system_prompt = _agent_full_system_prompt(
        agent_name=agent_name,
        schema_name=schema_name,
        extra_system=extra_system,
        debug_options=debug_options,
    )
    prompt = user_prompt
    last_error = ""
    task_name = re.sub(r"[^a-z0-9]+", "_", str(agent_name or "agent").lower()).strip("_") or "agent"
    for attempt in range(1, attempts + 1):
        user_content = prompt
        planner_thinking = _truthy((debug_options or {}).get("planner_thinking"), False)
        no_think_directive = str((debug_options or {}).get("planner_no_think_directive") or CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE or "/no_think").strip()
        if not planner_thinking and no_think_directive and no_think_directive not in user_content:
            user_content = f"{no_think_directive}\n\n{user_content}"
        prompt_chars = len(system_prompt) + len(user_content)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        _print_agent_io(
            debug_options,
            f"{agent_name.replace(' ', '_')}_crewai_micro_prompt_attempt_{attempt}",
            {
                "agent": agent_name,
                "agent_runtime": CREWAI_MICRO_AGENT_ENGINE,
                "crewai_task_name": task_name,
                "provider": provider_name,
                "model": model_name,
                "prompt_chars": prompt_chars,
                "system_chars": len(system_prompt),
                "user_chars": len(user_content),
                "messages": messages,
                "options": options,
            },
        )
        _append_album_debug_jsonl(
            debug_options,
            "03_agent_prompts.jsonl",
            {
                "agent": agent_name,
                "agent_runtime": CREWAI_MICRO_AGENT_ENGINE,
                "crewai_task_name": task_name,
                "provider": provider_name,
                "model": model_name,
                "attempt": attempt,
                "prompt_chars": prompt_chars,
                "system_chars": len(system_prompt),
                "user_chars": len(user_content),
                "messages": messages,
                "options": options,
            },
        )
        logs.append(
            f"CrewAI Micro Agent call: {agent_name} attempt {attempt} via {provider_label(provider_name)} "
            f"(prompt_chars={prompt_chars}, system={len(system_prompt)}, user={len(user_content)})."
        )
        started = time.perf_counter()
        try:
            llm = _make_crewai_micro_llm(model_name, provider_name, agent_name, planner_settings)
            persona_role, persona_goal, persona_backstory = _agent_persona(agent_name, schema_name)
            agent_iter = _agent_max_iter(agent_name)
            micro_agent = Agent(
                role=persona_role,
                goal=persona_goal,
                backstory=persona_backstory,
                llm=llm,
                tools=[],
                verbose=False,
                allow_delegation=False,
                max_iter=agent_iter,
                max_retry_limit=1,
                respect_context_window=True,
                reasoning=False,
                system_template="{{ .System }}",
                prompt_template="{{ .Prompt }}",
                response_template="{{ .Response }}",
                use_system_prompt=True,
            )
            micro_task = Task(
                description=user_content,
                expected_output=(
                    f"Delimiter blocks for {schema_name} only. "
                    "Award-level craft (concrete imagery, multisyllabic mosaic rhymes for rap, six-dimension caption coverage). "
                    "No JSON, no commentary, no AI cliches (neon dreams / fire inside / shattered dreams / we rise), "
                    "no producer names in caption, no metadata leakage."
                ),
                agent=micro_agent,
                guardrail=_crewai_micro_block_guardrail(schema_name),
                guardrail_max_retries=2 if agent_iter > 1 else 0,
            )
            micro_crew = Crew(
                agents=[micro_agent],
                tasks=[micro_task],
                process=Process.sequential,
                memory=False,
                planning=False,
                verbose=False,
                cache=False,
            )
            result = _kickoff_crewai_compact(micro_crew, logs, f"CrewAI Micro {agent_name}")
            raw_crewai_output = _task_output_raw_text(result)
            raw = _strip_thinking_blocks(raw_crewai_output)
            elapsed = round(time.perf_counter() - started, 3)
        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 3)
            last_error = f"{type(exc).__name__}: {exc}"
            _append_album_debug_jsonl(
                debug_options,
                "04_agent_responses.jsonl",
                {
                    "agent": agent_name,
                    "agent_runtime": CREWAI_MICRO_AGENT_ENGINE,
                    "crewai_task_name": task_name,
                    "attempt": attempt,
                    "elapsed": elapsed,
                    "error": last_error,
                },
            )
            if attempt >= attempts:
                break
            logs.append(f"CrewAI Micro Agent exception: {agent_name}; {last_error}.")
            prompt = (
                f"{user_prompt}\n\nRECOVERY: The previous CrewAI Micro {agent_name} call raised {last_error}. "
                "Return the requested delimiter blocks now."
            )
            continue
        _print_agent_io(debug_options, f"{agent_name.replace(' ', '_')}_crewai_micro_raw_response_attempt_{attempt}", raw)
        _append_album_debug_jsonl(
            debug_options,
            "04_agent_responses.jsonl",
            {
                "agent": agent_name,
                "agent_runtime": CREWAI_MICRO_AGENT_ENGINE,
                "crewai_task_name": task_name,
                "provider": provider_name,
                "model": model_name,
                "attempt": attempt,
                "elapsed": elapsed,
                "response_chars": len(raw),
                "raw_crewai_output": str(raw_crewai_output or ""),
                "raw_response": str(raw_crewai_output or ""),
                "response": raw,
            },
        )
        logs.append(f"CrewAI Micro Agent response: {agent_name} {len(raw)} chars in {elapsed}s (parse pending).")
        if not raw.strip():
            last_error = "empty response"
            if attempt >= attempts:
                break
            logs.append(f"CrewAI Micro Agent empty response: {agent_name}; retrying delimiter block output.")
            prompt = (
                f"{user_prompt}\n\nRECOVERY: Your previous response was empty. "
                "Return the requested delimiter blocks only. Do not call tools."
            )
            continue
        try:
            payload = _parse_agent_block_payload(raw, schema_name)
            if not isinstance(payload, dict):
                raise ValueError("block_parse_failed:root_was_not_object")
            _validate_agent_response_shape(schema_name, payload)
            parsed = _coerce_agent_lyrics_payload(payload)
            _print_agent_io(debug_options, f"{agent_name.replace(' ', '_')}_crewai_micro_parsed_blocks_attempt_{attempt}", parsed)
            _append_album_debug_jsonl(
                debug_options,
                "04_agent_responses.jsonl",
                {
                    "agent": agent_name,
                    "agent_runtime": CREWAI_MICRO_AGENT_ENGINE,
                    "crewai_task_name": task_name,
                    "attempt": attempt,
                    "parsed_blocks": parsed,
                },
            )
            logs.append(f"CrewAI Micro Agent parsed delimiter blocks: {agent_name} attempt {attempt} ok.")
            return parsed
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _append_album_debug_jsonl(
                debug_options,
                "04_agent_responses.jsonl",
                {
                    "agent": agent_name,
                    "agent_runtime": CREWAI_MICRO_AGENT_ENGINE,
                    "crewai_task_name": task_name,
                    "attempt": attempt,
                    "block_parse_error": last_error,
                    "response_preview": _monitor_preview(raw, 500),
                },
            )
            if attempt >= attempts:
                break
            logs.append(f"CrewAI Micro Agent block repair: {agent_name}; {last_error}.")
            prompt = (
                f"{user_prompt}\n\nBLOCK REPAIR: The previous CrewAI Micro response failed delimiter-block parsing: {last_error}. "
                "Return exactly the required delimiter blocks, in order, with no JSON, markdown, commentary, or extra text.\n"
                "EXPECTED_BLOCK_SHAPE:\n"
                f"{_agent_block_template(schema_name)}"
            )
    raise AceJamAgentError(f"{agent_name} failed to produce valid CrewAI Micro delimiter blocks after {attempts} attempt(s): {last_error}")


def _public_gate_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in (report or {}).items() if key != "repaired_payload"}


def _track_summary_for_agent(track: dict[str, Any]) -> dict[str, Any]:
    stats = lyric_stats(str(track.get("lyrics") or ""))
    return {
        "track_number": track.get("track_number"),
        "title": track.get("title"),
        "producer_credit": track.get("producer_credit"),
        "bpm": track.get("bpm"),
        "key_scale": track.get("key_scale"),
        "caption": _clip_text(track.get("caption") or track.get("tags") or "", 360),
        "hook_promise": _clip_text(track.get("hook_promise") or "", 220),
        "lyrics_stats": {
            "words": stats.get("word_count"),
            "lines": stats.get("line_count"),
            "sections": stats.get("section_count"),
        },
    }


def _director_short_track_role(track: dict[str, Any]) -> bool:
    role = str((track or {}).get("role") or (track or {}).get("track_role") or "").strip().lower()
    title = str((track or {}).get("title") or "").strip().lower()
    return bool(re.search(r"\b(?:intro|outro|skit|interlude)\b", f"{role} {title}"))


def _director_is_rap_context(track: dict[str, Any], options: dict[str, Any] | None = None) -> bool:
    return bool(
        re.search(
            r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b",
            _director_track_genre_hint(track, options),
            re.I,
        )
    )


def _director_rap_bar_counts(track: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, int]:
    lyrics = str((track or {}).get("lyrics") or "")
    rap_context = _director_is_rap_context(track, options)
    counts: dict[str, int] = {}
    active_section = ""
    active_counts = False
    for raw_line in lyrics.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        if re.fullmatch(r"\[[^\]]+\]", line):
            active_section = line
            lower = line.lower()
            active_counts = bool("verse" in lower and (rap_context or re.search(r"\brap|hip[-\s]?hop|flow\b", lower, re.I)))
            if active_counts:
                counts.setdefault(active_section, 0)
            continue
        if active_counts and active_section:
            counts[active_section] = counts.get(active_section, 0) + 1
    return counts


def _director_lyrics_quality(track: dict[str, Any], options: dict[str, Any] | None = None, gate: dict[str, Any] | None = None) -> dict[str, Any]:
    duration = parse_duration_seconds(
        (track or {}).get("duration") or (options or {}).get("track_duration") or 180,
        (options or {}).get("track_duration") or 180,
    )
    genre_hint = _director_track_genre_hint(track, options)
    density = str((track or {}).get("lyric_density") or (options or {}).get("lyric_density") or "dense")
    structure_preset = str((track or {}).get("structure_preset") or (options or {}).get("structure_preset") or "auto")
    plan = lyric_length_plan(duration, density, structure_preset, genre_hint)
    stats = lyric_stats(str((track or {}).get("lyrics") or ""))
    min_words = int(plan.get("min_words") or 0)
    target_words = int(plan.get("target_words") or min_words)
    raw_min_lines = int(plan.get("min_lines") or 0)
    min_lines = _director_effective_min_lines(raw_min_lines, min_words)
    target_lines = int(plan.get("target_lines") or min_lines)
    rap_bar_counts = _director_rap_bar_counts(track, options)
    issues = list((gate or {}).get("issues") or [])
    gate_status = str((gate or {}).get("status") or ("pass" if not issues else "fail"))
    return {
        "version": "album-lyrics-quality-v1",
        "gate_status": gate_status,
        "word_count": int(stats.get("word_count") or 0),
        "line_count": int(stats.get("line_count") or 0),
        "char_count": int(stats.get("char_count") or 0),
        "section_count": int(stats.get("section_count") or 0),
        "hook_count": sum(
            1 for section in stats.get("sections") or [] if re.search(r"chorus|hook|refrain", str(section), re.I)
        ),
        "rap_bar_counts": rap_bar_counts,
        "target_words": target_words,
        "min_words": min_words,
        "target_lines": target_lines,
        "min_lines": min_lines,
        "raw_min_lines": raw_min_lines,
        "duration": duration,
        "is_rap": _director_is_rap_context(track, options),
        "is_short_role": _director_short_track_role(track),
        "issues": issues,
    }


def _set_track_stats(track: dict[str, Any]) -> dict[str, Any]:
    stats = lyric_stats(str(track.get("lyrics") or ""))
    track["lyrics_word_count"] = int(stats.get("word_count") or 0)
    track["lyrics_line_count"] = int(stats.get("line_count") or 0)
    track["lyrics_char_count"] = int(stats.get("char_count") or 0)
    track["section_count"] = int(stats.get("section_count") or 0)
    track["hook_count"] = sum(
        1 for section in stats.get("sections") or [] if re.search(r"chorus|hook|refrain", str(section), re.I)
    )
    track["lyrics_quality"] = _director_lyrics_quality(track)
    return track


def _director_effective_min_lines(raw_min_lines: int, min_words: int) -> int:
    raw = int(raw_min_lines or 0)
    if raw <= 0:
        return 0
    cap = max(36, int((int(min_words or 0) / 5.7) + 0.999))
    return min(raw, cap)


def _director_lyric_extension_lines(track: dict[str, Any]) -> list[str]:
    """Short deterministic bars used only when local agents undershoot duration."""
    return [
        "Low-end shadows roll beneath the street",
        "Sirens paint the glass with broken light",
        "Cold suits whisper while the drums hit hard",
        "Paper towers lean above the block",
        "Every signature leaves another scar",
        "Truth keeps breathing under poured cement",
        "Neon shakes across the courthouse steps",
        "Bassline crawling where the deals were kept",
        "Hands stay clean while the corners bleed",
        "Names get buried under polished greed",
        "Footsteps echo through the vacant floor",
        "Locks keep turning on a quiet war",
        "Streetlights flicker on the hidden cost",
        "Every profit counts what someone lost",
        "Marble halls can never mute the cries",
        "Concrete remembers every alibi",
        "Glass towers hum with a hollow shine",
        "Pressure rises on the dotted line",
        "Drums keep knocking through the city smoke",
        "Voices cut the silence when it broke",
        "Gold trim covers up the rust below",
        "Dark wheels move where the cameras go",
        "Every block can feel the ground shake",
        "Every smile can hide another take",
        "Sirens bend around the midnight bend",
        "Hard bass rolls like judgment in the wind",
        "Cold rain taps on the executive glass",
        "Street truth moves where the rumors pass",
        "No clean hands in the elevator light",
        "No sleep left when the truth takes flight",
        "Paper crowns fall when the bassline drops",
        "Concrete talks when the heartbeat stops",
        "Shadow deals melt in the morning heat",
        "Every drum hit lands beneath my feet",
        "City pressure riding through the snare",
        "Voices rise from underneath the stair",
        "Tall walls shake when the chorus lands",
        "Truth breaks loose from the quiet plans",
        "Deep subs rumble through the floor",
        "No closed room can hold it anymore",
    ]


def _expand_director_lyrics_lines_to_fit(
    lines: list[str],
    track: dict[str, Any],
    *,
    min_words: int,
    min_lines: int,
    max_chars: int,
) -> tuple[list[str], bool]:
    result = [str(line or "").strip() for line in lines if str(line or "").strip()]
    if not result:
        return result, False
    stats = lyric_stats("\n".join(result))
    if int(stats.get("word_count") or 0) >= int(min_words or 0) and int(stats.get("line_count") or 0) >= int(min_lines or 0):
        return result, False
    tag_indexes = [
        idx
        for idx, line in enumerate(result)
        if re.fullmatch(r"\[[^\]]+\]", line)
        and not re.search(r"\b(?:break|instrumental|interlude)\b", line, re.I)
    ]
    if not tag_indexes:
        tag_indexes = [idx for idx, line in enumerate(result) if re.fullmatch(r"\[[^\]]+\]", line)]
    if not tag_indexes:
        return result, False
    fillers = _director_lyric_extension_lines(track)
    changed = False
    fill_index = 0
    max_len = max(800, int(max_chars or ACE_STEP_LYRICS_SAFE_HEADROOM))
    while (
        (int(stats.get("word_count") or 0) < int(min_words or 0) or int(stats.get("line_count") or 0) < int(min_lines or 0))
        and fill_index < 160
    ):
        section_pos = tag_indexes[fill_index % len(tag_indexes)]
        insert_at = len(result)
        for idx in range(section_pos + 1, len(result)):
            if re.fullmatch(r"\[[^\]]+\]", result[idx]):
                insert_at = idx
                break
        line = fillers[fill_index % len(fillers)]
        candidate = result[:insert_at] + [line] + result[insert_at:]
        if len("\n".join(candidate)) > max_len:
            break
        result = candidate
        tag_indexes = [
            idx
            for idx, item in enumerate(result)
            if re.fullmatch(r"\[[^\]]+\]", item)
            and not re.search(r"\b(?:break|instrumental|interlude)\b", item, re.I)
        ] or [idx for idx, item in enumerate(result) if re.fullmatch(r"\[[^\]]+\]", item)]
        stats = lyric_stats("\n".join(result))
        changed = True
        fill_index += 1
    return result, changed


def _director_track_genre_hint(track: dict[str, Any], options: dict[str, Any] | None = None) -> str:
    parts: list[str] = []
    for source in (track or {}, options or {}):
        for key in (
            "caption",
            "tags",
            "description",
            "style",
            "vibe",
            "narrative",
            "genre_profile",
            "album_agent_genre_prompt",
            "genre_prompt",
            "album_agent_mood_vibe",
            "album_agent_vocal_type",
            "custom_tags",
            "sanitized_concept",
            "concept",
            "user_prompt",
        ):
            value = source.get(key) if isinstance(source, dict) else None
            if value:
                parts.append(" ".join(str(item) for item in value) if isinstance(value, list) else str(value))
        tag_list = source.get("tag_list") if isinstance(source, dict) else None
        if isinstance(tag_list, list):
            parts.extend(str(item) for item in tag_list if str(item).strip())
    text = "\n".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def _director_lyric_duration_fit(track: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
    lyrics = str((track or {}).get("lyrics") or "")
    duration = parse_duration_seconds(
        (track or {}).get("duration") or (options or {}).get("track_duration") or 180,
        (options or {}).get("track_duration") or 180,
    )
    genre_hint = _director_track_genre_hint(track, options)
    density = str((track or {}).get("lyric_density") or (options or {}).get("lyric_density") or "dense")
    structure_preset = str((track or {}).get("structure_preset") or (options or {}).get("structure_preset") or "auto")
    plan = lyric_length_plan(duration, density, structure_preset, genre_hint)
    stats = lyric_stats(lyrics)
    min_words = int(plan.get("min_words") or 0)
    raw_min_lines = int(plan.get("min_lines") or 0)
    min_lines = _director_effective_min_lines(raw_min_lines, min_words)
    issues: list[str] = []
    raw_instrumental = (track or {}).get("instrumental")
    instrumental = lyrics.strip().lower() == "[instrumental]" or str(raw_instrumental).strip().lower() in {"1", "true", "yes", "on"}
    if lyrics.strip() and not instrumental:
        if stats.get("word_count", 0) < min_words:
            issues.append(f"lyrics_under_length:{stats.get('word_count', 0)}/{min_words}_words")
        if stats.get("line_count", 0) < min_lines:
            issues.append(f"lyrics_too_few_lines:{stats.get('line_count', 0)}/{min_lines}_lines")
    density_gate = lyric_density_gate(
        lyrics,
        plan,
        duration=duration,
        genre_hint=genre_hint,
        instrumental=instrumental,
    )
    for issue in density_gate.get("issues") or []:
        issue_id = str(issue.get("id") if isinstance(issue, dict) else issue)
        detail = str(issue.get("detail") if isinstance(issue, dict) else "").strip()
        issues.append(f"{issue_id}:{detail}" if detail else issue_id)
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "duration": duration,
        "density": plan.get("density") or density,
        "genre_hint": _clip_text(genre_hint, 220),
        "word_count": int(stats.get("word_count") or 0),
        "line_count": int(stats.get("line_count") or 0),
        "min_words": min_words,
        "target_words": int(plan.get("target_words") or 0),
        "min_lines": min_lines,
        "target_lines": int(plan.get("target_lines") or 0),
        "raw_min_lines": raw_min_lines,
        "lyric_density_gate": density_gate,
    }


def _director_genre_validation_issues(
    payload: dict[str, Any],
    base: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    *,
    include_lyrics: bool = True,
) -> list[str]:
    merged = {**(base or {}), **(payload or {})}
    if not include_lyrics:
        merged.pop("lyrics", None)
        merged.pop("lyrics_lines", None)
    if "lyrics_lines" in merged and not merged.get("lyrics"):
        merged["lyrics"] = "\n".join(str(line) for line in (merged.get("lyrics_lines") or []))
    report = evaluate_genre_adherence(merged, options)
    issues: list[str] = []
    for issue in report.get("issues") or []:
        issue_id = str(issue.get("id") if isinstance(issue, dict) else issue)
        if issue_id:
            issues.append(issue_id)
    return issues


def _director_producer_grade_validation_issues(
    payload: dict[str, Any],
    base: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> list[str]:
    merged = {**(base or {}), **(payload or {})}
    report = producer_grade_readiness(merged, options=options)
    issues: list[str] = []
    for issue in report.get("issues") or []:
        issue_id = str(issue.get("id") if isinstance(issue, dict) else issue)
        detail = str(issue.get("detail") if isinstance(issue, dict) else "").strip()
        if issue_id:
            issues.append(f"{issue_id}:{detail}" if detail else issue_id)
    return issues


def _agent_payload_lines(payload: dict[str, Any]) -> list[str]:
    coerced = _coerce_agent_lyrics_payload(payload if isinstance(payload, dict) else {})
    lines_value = coerced.get("lyrics_lines") or coerced.get("lyric_lines") or coerced.get("script_lines")
    lines: list[str] = []
    if isinstance(lines_value, list):
        for item in lines_value:
            if isinstance(item, dict):
                text = str(item.get("line") or item.get("text") or item.get("section") or item.get("tag") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                lines.append(text)
    if not lines and str(coerced.get("lyrics") or "").strip():
        lines = [line.strip() for line in str(coerced.get("lyrics") or "").splitlines() if line.strip()]
    return lines


def _section_tag_line(section: str) -> str:
    text = str(section or "").strip()
    if not text:
        return ""
    inner = text.strip().strip("[]").strip()
    return f"[{inner}]" if inner else ""


def _ensure_part_section_tags(lines: list[str], section_group: list[str]) -> list[str]:
    result = [str(line or "").strip() for line in lines if str(line or "").strip()]
    if not section_group:
        return result
    existing = {re.sub(r"[^a-z0-9]+", "", line.lower()) for line in result if line.startswith("[")}
    prefixed: list[str] = []
    for section in section_group:
        tag = _section_tag_line(section)
        key = re.sub(r"[^a-z0-9]+", "", tag.lower())
        if key and key not in existing:
            prefixed.append(tag)
    return prefixed + result if prefixed else result


def _assemble_split_agent_track(
    *,
    blueprint: dict[str, Any],
    settings_payload: dict[str, Any],
    lyric_part_payloads: list[dict[str, Any]],
    section_groups: list[list[str]],
    language: str,
    duration: float,
) -> dict[str, Any]:
    lyric_lines: list[str] = []
    for idx, payload in enumerate(lyric_part_payloads):
        lines = _agent_payload_lines(payload)
        group = section_groups[idx] if idx < len(section_groups) else []
        lyric_lines.extend(_ensure_part_section_tags(lines, group))
    lyrics = "\n".join(line for line in lyric_lines if line.strip()).strip()
    tag_list = settings_payload.get("tag_list")
    if isinstance(tag_list, str):
        tag_list = [item.strip() for item in tag_list.split(",") if item.strip()]
    track = {
        **dict(blueprint or {}),
        **dict(settings_payload or {}),
        "lyrics": lyrics,
        "lyrics_lines": lyric_lines,
        "duration": duration,
        "language": settings_payload.get("language") or language,
        "vocal_language": settings_payload.get("vocal_language") or language,
        "caption": settings_payload.get("caption") or settings_payload.get("tags") or blueprint.get("caption") or blueprint.get("tags") or "",
        "tags": settings_payload.get("tags") or settings_payload.get("caption") or blueprint.get("tags") or "",
        "tag_list": tag_list or blueprint.get("tag_list") or [],
        "agent_split_flow": True,
        "agent_lyric_part_count": len(lyric_part_payloads),
        "agent_lyric_sections": [section for group in section_groups for section in group],
    }
    return _set_track_stats(track)


def _agent_gate_options(opts: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    return {
        **dict(opts or {}),
        "track_duration": track.get("duration") or opts.get("track_duration") or 180,
        "lyric_density": opts.get("lyric_density") or "dense",
        "structure_preset": opts.get("structure_preset") or "auto",
    }


def _album_bible_agent_prompt(
    *,
    concept: str,
    num_tracks: int,
    track_duration: float,
    language: str,
    opts: dict[str, Any],
    model_info: dict[str, Any],
    contract: dict[str, Any],
    retrieved_context: str = "",
) -> str:
    context = {
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "track_count": int(num_tracks or 0),
        "track_duration_seconds": int(parse_duration_seconds(track_duration, 180)),
        "language": language,
        "workflow": "album_bible_only_no_lyrics",
    }
    return (
        "You are the AceJAM Album Bible Agent. Build compact album-level creative DNA for an ACE-Step album.\n"
        "Do not write full lyrics in this stage. Do not decide the final track count; AceJAM builds an exact N-track scaffold deterministically. "
        "You may return optional track blueprint hints, but missing hints are fine and must not stop the album.\n"
        "Preserve user locked titles/order/producers/BPM/style/vibe/narrative when you mention them.\n\n"
        f"ORIGINAL_PROMPT_SIGNAL:\n{_clip_text(concept, 1400)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{_clip_text(retrieved_context or '[]', 2600)}\n\n"
        f"USER_ALBUM_CONTRACT:\n{json.dumps(contract_prompt_context(contract), ensure_ascii=True, indent=2)}\n\n"
        f"ALBUM_TOOL_CONTEXT:\n{json.dumps(context, ensure_ascii=True, indent=2)}\n\n"
        f"LANGUAGE: {language}\nTRACK_COUNT: {num_tracks}\nTRACK_DURATION_SECONDS: {track_duration}\n\n"
        "DURATION_RULE: every optional track hint must use TRACK_DURATION_SECONDS unless USER_ALBUM_CONTRACT explicitly locks a per-track duration.\n\n"
        "OUTPUT_SCHEMA:\n"
        "{\n"
        '  "album_bible": {"album_title":"", "concept":"", "arc":"", "sonic_palette":"", "recurring_motifs":[], "continuity_rules":[]},\n'
        f'  "tracks": [{{"track_number":1, "title":"", "producer_credit":"", "bpm":95, "key_scale":"A minor", "time_signature":"4", "duration":{int(parse_duration_seconds(track_duration, 180))}, "style":"", "vibe":"", "narrative":"", "description":"", "tag_list":[], "tags":"", "hook_promise":"", "performance_brief":"", "required_phrases":[]}}]\n'
        "}\n"
        "Return exactly one JSON object with album_bible and optional tracks hints. It is acceptable for tracks to be empty or shorter than TRACK_COUNT."
    )


def _agent_tag_library_summary() -> dict[str, Any]:
    return {
        "caption_dimensions": ACE_STEP_CAPTION_DIMENSIONS,
        "tag_examples": {
            "primary_genre": TAG_TAXONOMY.get("genre_style", [])[:14],
            "drum_groove": TAG_TAXONOMY.get("speed_rhythm", [])[:10] + ["boom-bap drums", "trap hi-hats", "punchy snare"],
            "low_end_bass": ["808 bass", "sub-bass", "deep low end", "synth bass", "bass guitar", "sliding 808s"],
            "melodic_identity": TAG_TAXONOMY.get("instruments", [])[:18] + ["soul chop", "piano motif", "synth lead"],
            "vocal_delivery": TAG_TAXONOMY.get("vocal_character", [])[:12],
            "arrangement_movement": TAG_TAXONOMY.get("structure_hints", [])[:10],
            "texture_space": TAG_TAXONOMY.get("timbre_texture", [])[:12] + TAG_TAXONOMY.get("mood_atmosphere", [])[:6],
            "mix_master": TAG_TAXONOMY.get("production_style", [])[:12],
        },
        "selection_rule": "Final caption/tags must be compact comma-separated sonic terms covering every caption dimension.",
    }


def _compact_album_bible_for_agent(album_bible: dict[str, Any]) -> dict[str, Any]:
    bible = dict(album_bible or {})
    return {
        "album_title": bible.get("album_title") or "",
        "concept": _clip_text(bible.get("concept") or "", 700),
        "arc": _clip_text(bible.get("arc") or "", 500),
        "sonic_palette": _clip_text(bible.get("sonic_palette") or "", 500),
        "recurring_motifs": (bible.get("recurring_motifs") or bible.get("motifs") or [])[:8],
        "continuity_rules": (bible.get("continuity_rules") or [])[:8],
    }


def _compact_blueprint_for_agent(blueprint: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "track_number",
        "title",
        "locked_title",
        "producer_credit",
        "engineer_credit",
        "artist_role",
        "bpm",
        "key_scale",
        "time_signature",
        "duration",
        "style",
        "vibe",
        "narrative",
        "description",
        "tag_list",
        "tags",
        "hook_promise",
        "performance_brief",
        "required_phrases",
        "language",
    ]
    compact: dict[str, Any] = {}
    for field in fields:
        value = (blueprint or {}).get(field)
        if value in (None, "", []):
            continue
        if isinstance(value, str):
            compact[field] = _clip_text(value, 800)
        elif isinstance(value, list):
            compact[field] = value[:16]
        else:
            compact[field] = value
    return compact


def _compact_track_agent_contract(blueprint: dict[str, Any], lyric_plan: dict[str, Any], *, include_schema: bool = False) -> dict[str, Any]:
    contract = {
        "ace_step_limits": {
            "caption_max_chars": 512,
            "lyrics_max_chars": 4096,
            "caption_role": "sonic tags only: genre, groove, instruments, vocal style, mood, arrangement energy, mix",
            "lyrics_role": "temporal script with concise section tags plus actual lyric lines only",
            "forbidden_caption_content": ["lyrics", "prompt text", "BPM/key/duration/model/seed", "JSON", "track headers", "prose sentences"],
            "forbidden_lyrics_content": ["metadata prose", "reasoning", "placeholders", "escaped literal \\n", "generic filler"],
        },
        "locked_track_fields": {
            key: blueprint.get(key)
            for key in [
                "track_number",
                "title",
                "producer_credit",
                "bpm",
                "key_scale",
                "time_signature",
                "duration",
                "style",
                "vibe",
                "narrative",
                "required_phrases",
            ]
            if blueprint.get(key) not in (None, "", [])
        },
        "LYRIC_LENGTH_PLAN": {
            "sections": lyric_plan.get("sections") or [],
            "target_words": lyric_plan.get("target_words"),
            "min_words": lyric_plan.get("min_words"),
            "target_lines": lyric_plan.get("target_lines"),
            "min_lines": lyric_plan.get("min_lines"),
            "max_lyrics_chars": lyric_plan.get("max_lyrics_chars") or 4096,
        },
        "FULL_TAG_LIBRARY_COMPACT": _agent_tag_library_summary(),
        "self_check": [
            "Keep locked fields exactly.",
            "Write enough unique short lines to meet min_words and min_lines.",
            "Include at least one chorus/hook/final chorus section.",
            "Ensure caption covers every tag dimension.",
            "Do not add generic filler or planning prose.",
        ],
    }
    if include_schema:
        contract["output_schema"] = {
            "track_number": "int",
            "artist_name": "string",
            "title": "locked string",
            "description": "short narrative summary",
            "tags": "comma-separated caption under 512 chars",
            "tag_list": "array of compact sonic tags",
            "lyrics_lines": "preferred array: one section tag or lyric line per string; backend joins with newlines",
            "lyrics": "optional full ACE-Step temporal script under 4096 chars; escape newlines as \\n if used",
            "bpm": "locked number",
            "key_scale": "locked string or A minor",
            "time_signature": "4",
            "language": "en",
            "duration": "seconds",
            "hook_promise": "short promise",
            "performance_brief": "short delivery note",
            "quality_checks": "object",
        }
    return contract


def _album_arc_role(index: int, total: int) -> str:
    position = int(index) + 1
    count = max(1, int(total or 1))
    if position == 1:
        return "opener - immediate identity and strongest first impression"
    if position == count:
        return "closer - resolution, callback, or final twist"
    if count >= 5 and position == count - 1:
        return "cooldown - emotional consequence and contrast"
    if position >= max(2, int(round(count * 0.65))):
        return "climax - highest stakes and biggest hook"
    return "escalation - new scene, sharper rhythm, more pressure"


def _default_missing_track_title(contract: dict[str, Any], index: int, total: int) -> str:
    album_title = str((contract or {}).get("album_title") or "Album").strip()
    role = _album_arc_role(index, total).split(" - ", 1)[0].title()
    safe_album = re.sub(r"[^A-Za-z0-9 ]+", "", album_title).strip() or "Album"
    return f"{safe_album} {role} {index + 1}"


def _baseline_caption_tags(blueprint: dict[str, Any], concept: str) -> list[str]:
    style = str(blueprint.get("style") or blueprint.get("description") or concept or "").lower()
    if re.search(r"schlager|accordion|akkordeon|brass|polka|volks", style):
        return [
            "German schlager pop",
            "steady dance groove",
            "sparkling accordion",
            "bright brass stabs",
            "warm lead vocal",
            "uplifting singalong mood",
            "dynamic chorus arrangement",
            "clean radio-ready mix",
        ]
    if re.search(r"rap|hip.?hop|boom.?bap|trap|drill|g.?funk", style):
        genre = "cinematic hip-hop"
        groove = "steady rap groove"
        vocal = "clear lead rap vocal"
    elif re.search(r"techno|house|trance|edm|club|dance", style):
        genre = "melodic electronic"
        groove = "driving four-on-the-floor groove"
        vocal = "clean vocal chops"
    else:
        genre = "modern pop"
        groove = "steady groove"
        vocal = "clear lead vocal"
    return [
        genre,
        groove,
        "deep bass",
        "bright drums",
        vocal,
        "emotional atmosphere",
        "dynamic hook arrangement",
        "polished studio mix",
    ]


def _hint_by_track_number(hints: list[dict[str, Any]], track_number: int) -> dict[str, Any]:
    for hint in hints:
        if int(hint.get("track_number") or 0) == int(track_number):
            return dict(hint)
    index = track_number - 1
    if 0 <= index < len(hints):
        return dict(hints[index])
    return {}


def _album_duration_mode(opts: dict[str, Any] | None) -> str:
    return "fixed" if str((opts or {}).get("duration_mode") or "").strip().lower() == "fixed" else "ai_per_track"


def _clamp_album_duration(value: Any, fallback: float) -> float:
    duration = parse_duration_seconds(value, fallback)
    return max(30.0, min(600.0, float(duration)))


def _album_track_role_hint(track: dict[str, Any], fallback_role: str = "") -> str:
    parts: list[str] = [fallback_role]
    for key in ("role", "album_arc_role", "title", "description", "narrative", "caption", "tags", "style", "vibe"):
        value = track.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif value not in (None, "", []):
            parts.append(str(value))
    return " ".join(parts).lower()


def _album_default_duration_for_role(track: dict[str, Any], fallback: float, role: str = "") -> float:
    text = _album_track_role_hint(track, role)
    if any(token in text for token in ("intro", "outro", "skit", "interlude", "breather")):
        return 90.0
    if any(token in text for token in ("extended", "epic", "cinematic")):
        return 270.0
    if any(token in text for token in ("single", "full_song", "full song", "opener", "climax", "closer")):
        return 210.0
    return _clamp_album_duration(fallback, 180.0)


def _album_duration_from_hint(
    hint: dict[str, Any],
    fallback: float,
    duration_mode: str,
    role: str = "",
) -> tuple[float, str]:
    if duration_mode == "fixed":
        return _clamp_album_duration(fallback, 180.0), "fixed_duration_mode"
    if hint.get("duration") not in (None, "", []):
        return _clamp_album_duration(hint.get("duration"), fallback), "ai_per_track"
    return _clamp_album_duration(_album_default_duration_for_role(hint, fallback, role), fallback), "role_default"


def _build_album_track_scaffold(
    *,
    concept: str,
    num_tracks: int,
    track_duration: float,
    language: str,
    opts: dict[str, Any],
    contract: dict[str, Any],
    bible_payload: dict[str, Any],
    logs: list[str],
) -> list[dict[str, Any]]:
    hints = [item for item in (bible_payload.get("tracks") or []) if isinstance(item, dict)]
    editable_hints = [item for item in (opts.get("editable_plan_tracks") or []) if isinstance(item, dict)]
    scaffold: list[dict[str, Any]] = []
    requested_duration = parse_duration_seconds(track_duration, 180)
    duration_mode = _album_duration_mode(opts)
    for index in range(max(0, int(num_tracks or 0))):
        track_number = index + 1
        role = _album_arc_role(index, num_tracks)
        hint = _hint_by_track_number(hints, track_number)
        initial_duration, duration_source = _album_duration_from_hint(hint, requested_duration, duration_mode, role)
        hint_duration = hint.get("duration")
        if duration_mode == "fixed" and hint_duration not in (None, "", []) and parse_duration_seconds(hint_duration, requested_duration) != requested_duration:
            logs.append(
                f"Ignored agent duration hint for track {track_number}: "
                f"{parse_duration_seconds(hint_duration, requested_duration)}s; job duration is {requested_duration}s."
            )
        slot: dict[str, Any] = {
            "track_number": track_number,
            "title": "",
            "duration": initial_duration,
            "duration_mode": duration_mode,
            "duration_source": duration_source,
            "bpm": hint.get("bpm") or 95,
            "key_scale": hint.get("key_scale") or "A minor",
            "time_signature": hint.get("time_signature") or "4",
            "language": language,
            "album_arc_role": role,
            "style": hint.get("style") or opts.get("genre_hint") or "cinematic pop",
            "vibe": hint.get("vibe") or role,
            "narrative": hint.get("narrative") or hint.get("description") or role,
            "description": hint.get("description") or hint.get("narrative") or role,
            "tags": hint.get("tags") or "",
            "tag_list": hint.get("tag_list") or [],
            "hook_promise": hint.get("hook_promise") or "",
            "performance_brief": hint.get("performance_brief") or "",
            "required_phrases": hint.get("required_phrases") or [],
            "scaffold_source": "deterministic_scaffold",
            "needs_agent_blueprint": True,
        }
        if hint:
            slot.update({key: value for key, value in hint.items() if value not in (None, "", [])})
            slot["duration"] = initial_duration
            slot["duration_mode"] = duration_mode
            slot["duration_source"] = duration_source
            slot["scaffold_source"] = "bible_hint"
        editable_hint = _hint_by_track_number(editable_hints, track_number)
        if editable_hint:
            editable_hint_duration, editable_duration_source = _album_duration_from_hint(editable_hint, requested_duration, duration_mode, role)
            raw_editable_duration = editable_hint.get("duration")
            if duration_mode == "fixed" and raw_editable_duration not in (None, "", []) and parse_duration_seconds(raw_editable_duration, requested_duration) != requested_duration:
                logs.append(
                    f"Ignored editable duration hint for track {track_number}: "
                    f"{parse_duration_seconds(raw_editable_duration, requested_duration)}s; job duration is {requested_duration}s."
                )
            for key, value in editable_hint.items():
                if key in {"lyrics", "lyrics_lines"}:
                    continue
                if value not in (None, "", []):
                    slot[key] = value
            slot["duration"] = editable_hint_duration
            slot["duration_mode"] = duration_mode
            slot["duration_source"] = editable_duration_source
            slot["editable_plan_scaffold"] = True
            slot["scaffold_source"] = "editable_plan"
        locked = contract_track(contract, track_number, index)
        if locked:
            locked_title = str(locked.get("locked_title") or "").strip()
            if locked_title:
                slot["title"] = locked_title
                slot["locked_title"] = locked_title
                slot["scaffold_source"] = "user_contract"
            for field in ["producer_credit", "engineer_credit", "artist_role", "bpm", "key_scale", "style", "vibe", "narrative", "required_phrases"]:
                if locked.get(field) not in (None, "", []):
                    slot[field] = locked.get(field)
            if locked.get("narrative"):
                slot["description"] = locked.get("narrative")
            elif locked.get("vibe"):
                slot["description"] = locked.get("vibe")
            if locked.get("required_lyrics"):
                slot["lyrics"] = locked.get("required_lyrics")
            if locked.get("duration"):
                slot["duration"] = parse_duration_seconds(locked.get("duration"), requested_duration)
                slot["duration_source"] = "user_contract"
            slot["needs_agent_blueprint"] = bool(not slot.get("tags") or not slot.get("hook_promise"))
        if not slot.get("title"):
            slot["title"] = _default_missing_track_title(contract, index, num_tracks)
            slot["generated_missing_track"] = True
        if not slot.get("tag_list"):
            slot["tag_list"] = _baseline_caption_tags(slot, concept)
        if not slot.get("tags"):
            slot["tags"] = ", ".join(str(item) for item in slot.get("tag_list") or [])
        scaffold.append(slot)
    scaffold = apply_user_album_contract_to_tracks(scaffold, contract, logs)
    return scaffold


def _merge_blueprint_payload(scaffold: dict[str, Any], payload: dict[str, Any], contract: dict[str, Any], index: int, logs: list[str]) -> dict[str, Any]:
    merged = {**dict(scaffold or {}), **dict(payload or {})}
    merged["track_number"] = int(scaffold.get("track_number") or index + 1)
    scaffold_duration = parse_duration_seconds(scaffold.get("duration") or 180, 180)
    duration_mode = str(scaffold.get("duration_mode") or "ai_per_track")
    payload_duration = payload.get("duration")
    if duration_mode == "fixed" and payload_duration not in (None, "", []) and parse_duration_seconds(payload_duration, scaffold_duration) != scaffold_duration:
        logs.append(
            f"Ignored blueprint duration hint for track {merged['track_number']}: "
            f"{parse_duration_seconds(payload_duration, scaffold_duration)}s; scaffold duration is {scaffold_duration}s."
        )
    if duration_mode == "fixed" or payload_duration in (None, "", []):
        merged["duration"] = scaffold_duration
    else:
        merged["duration"] = _clamp_album_duration(payload_duration, scaffold_duration)
        merged["duration_source"] = "blueprint"
    merged["duration_mode"] = duration_mode
    if not merged.get("title"):
        merged["title"] = scaffold.get("title") or f"Track {index + 1}"
    if not merged.get("tag_list"):
        merged["tag_list"] = _baseline_caption_tags(merged, "")
    if not merged.get("tags"):
        merged["tags"] = ", ".join(str(item) for item in merged.get("tag_list") or [])
    if not merged.get("description"):
        merged["description"] = merged.get("narrative") or merged.get("vibe") or merged.get("album_arc_role") or ""
    merged["needs_agent_blueprint"] = False
    return apply_user_album_contract_to_track(merged, contract, index, logs)


def _album_sequence_report(tracks: list[dict[str, Any]], contract: dict[str, Any], num_tracks: int) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    if len(tracks) != int(num_tracks or 0):
        issues.append({"id": "track_count_mismatch", "detail": f"{len(tracks)}/{num_tracks} tracks planned"})
    seen_titles: dict[str, int] = {}
    for index, track in enumerate(tracks):
        title = str(track.get("title") or "").strip()
        folded = title.casefold()
        if not title:
            issues.append({"id": "missing_title", "track_number": index + 1, "detail": "track title is empty"})
        elif folded in seen_titles:
            locked = bool(contract_track(contract, track.get("track_number"), index))
            if locked:
                issues.append({"id": "duplicate_locked_title", "track_number": index + 1, "detail": title})
            else:
                repaired = f"{title} Part {index + 1}"
                track["title"] = repaired
                repairs.append({"id": "duplicate_title_repaired", "track_number": index + 1, "from": title, "to": repaired})
        else:
            seen_titles[folded] = index + 1
        locked_item = contract_track(contract, track.get("track_number"), index)
        if locked_item and locked_item.get("locked_title") and str(track.get("title") or "") != str(locked_item.get("locked_title")):
            issues.append({
                "id": "locked_title_mismatch",
                "track_number": index + 1,
                "detail": f"{track.get('title')} != {locked_item.get('locked_title')}",
            })
        if str(track.get("payload_gate_status") or "") not in {"pass", "auto_repair"}:
            issues.append({
                "id": "payload_gate_not_passed",
                "track_number": index + 1,
                "detail": str(track.get("payload_gate_status") or "missing"),
            })
    bpms = {str(track.get("bpm") or "") for track in tracks if track.get("bpm")}
    keys = {str(track.get("key_scale") or "") for track in tracks if track.get("key_scale")}
    warnings: list[dict[str, Any]] = []
    if len(tracks) >= 3 and len(bpms) <= 1:
        warnings.append({"id": "low_bpm_contrast", "detail": "all planned tracks use the same BPM"})
    if len(tracks) >= 4 and len(keys) <= 1:
        warnings.append({"id": "low_key_contrast", "detail": "all planned tracks use the same key"})
    return {
        "version": "acejam-sequence-critic-2026-04-30",
        "gate_passed": not issues,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "warnings": warnings,
        "repairs": repairs,
        "repair_count": len(repairs),
        "track_count": len(tracks),
        "expected_track_count": int(num_tracks or 0),
    }


def _deterministic_album_bible(concept: str, contract: dict[str, Any], language: str, num_tracks: int) -> dict[str, Any]:
    title = str(contract.get("album_title") or "").strip()
    motifs: list[str] = []
    for item in contract.get("tracks") or []:
        for field in ("vibe", "narrative", "style"):
            text = str(item.get(field) or "").strip()
            if text and text not in motifs:
                motifs.append(_clip_text(text, 120))
            if len(motifs) >= 6:
                break
        if len(motifs) >= 6:
            break
    return {
        "album_title": title,
        "concept": _clip_text(contract.get("concept") or concept, 900),
        "arc": f"{num_tracks}-track {language or 'unknown'} album arc built from the locked user brief, with missing slots filled by AceJAM scaffold.",
        "sonic_palette": ", ".join(motifs[:4]) or "cohesive modern production, clear vocals, polished mix",
        "recurring_motifs": motifs[:6],
        "continuity_rules": [
            "Locked user-provided titles, producers, BPM, key, style, vibe, narrative, and required phrases remain authoritative.",
            "Generated missing tracks must extend the album concept without renaming or replacing locked tracks.",
            "Every final track must pass the ACE-Step payload quality gate before audio render.",
        ],
        "source": "deterministic_album_bible_after_agent_failure",
    }


def _track_blueprint_prompt(
    *,
    concept: str,
    album_bible: dict[str, Any],
    scaffold: dict[str, Any],
    contract: dict[str, Any],
    language: str,
    index: int,
    total: int,
    retrieved_context: str = "",
) -> str:
    return (
        "You are the AceJAM Track Blueprint Agent. Fill or enrich exactly one scaffold slot for an ACE-Step album.\n"
        f"TRACK COUNTER: you are planning track {index + 1} of {total}. Return this track only.\n"
        "Locked user fields are immutable. If the scaffold title came from the user, keep it exactly. "
        "If the slot is generated_missing_track=true, create a distinct title that fits the album arc.\n\n"
        f"ORIGINAL_PROMPT_SIGNAL:\n{_clip_text(concept, 1200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{_clip_text(retrieved_context or '[]', 2200)}\n\n"
        f"USER_ALBUM_CONTRACT:\n{json.dumps(contract_prompt_context(contract), ensure_ascii=True, indent=2)}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"SCAFFOLD_SLOT:\n{json.dumps(_compact_blueprint_for_agent(scaffold), ensure_ascii=True, indent=2)}\n\n"
        f"LANGUAGE: {language}\n\n"
        f"DURATION_SECONDS_LOCKED: {int(parse_duration_seconds(scaffold.get('duration') or 180, 180))}. Keep this duration exactly.\n\n"
        "OUTPUT_SCHEMA:\n"
        '{"track_number":1,"title":"","producer_credit":"","bpm":95,"key_scale":"A minor","time_signature":"4",'
        f'"duration":{int(parse_duration_seconds(scaffold.get("duration") or 180, 180))},"style":"","vibe":"","narrative":"","description":"","tag_list":[],"tags":"",'
        '"hook_promise":"","performance_brief":"","required_phrases":[]}\n'
        "Return strict JSON only. No lyrics yet."
    )


def _track_writer_prompt(
    *,
    concept: str,
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    previous_summaries: list[dict[str, Any]],
    track_prompt_template: str,
    lyric_plan: dict[str, Any] | None = None,
    index: int,
    total: int,
    retrieved_context: str = "",
) -> str:
    compact_contract = _compact_track_agent_contract(blueprint, lyric_plan or {}, include_schema=True)
    return (
        "You are the AceJAM Track Writer Agent. Plan and write the full ACE-Step temporal script for this one track.\n"
        f"TRACK COUNTER: you are writing track {index + 1} of {total}. Complete this track only.\n\n"
        f"ORIGINAL_PROMPT_SIGNAL:\n{_clip_text(concept, 1200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{_clip_text(retrieved_context or '[]', 2200)}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"PREVIOUS_TRACK_SUMMARIES:\n{json.dumps(previous_summaries, ensure_ascii=True, indent=2)}\n\n"
        f"ACE_STEP_TRACK_CONTRACT_COMPACT:\n{json.dumps(compact_contract, ensure_ascii=True, indent=2)}\n\n"
        "The full resolved ACE-Step prompt template is stored in the local debug log; use this compact contract for the response.\n"
        "Write full lyrics with section tags, enough unique short lines for the lyric plan, a repeatable hook, and no metadata prose. "
        "Prefer lyrics_lines: an array with one section tag or lyric line per item, so the JSON remains valid and complete. "
        "Return one JSON object with description, tag_list, tags/caption, lyrics, hook_promise, performance_brief, quality_checks, and counters."
    )


def _track_finalizer_prompt(
    *,
    concept: str,
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    writer_payload: dict[str, Any],
    track_prompt_template: str,
    lyric_plan: dict[str, Any] | None = None,
    index: int,
    total: int,
    retrieved_context: str = "",
) -> str:
    compact_contract = _compact_track_agent_contract(blueprint, lyric_plan or {}, include_schema=True)
    return (
        "You are the AceJAM Track Finalizer Agent. Normalize the writer output into one ACE-Step-ready track JSON object.\n"
        f"TRACK COUNTER: you are finalizing track {index + 1} of {total}.\n\n"
        f"ORIGINAL_PROMPT_SIGNAL:\n{_clip_text(concept, 1200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{_clip_text(retrieved_context or '[]', 2200)}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"WRITER_OUTPUT_JSON:\n{json.dumps(writer_payload, ensure_ascii=True, indent=2)}\n\n"
        f"ACE_STEP_TRACK_CONTRACT_COMPACT:\n{json.dumps(compact_contract, ensure_ascii=True, indent=2)}\n\n"
        "The full resolved ACE-Step prompt template is stored in the local debug log; use this compact contract for the response.\n"
        "Return the final track JSON with all required metadata and full lyrics. "
        "Prefer lyrics_lines for the full script, one line per array item. "
        "Preserve locked fields exactly. Caption must be sound tags only. Lyrics must be actual song lines only."
    )


def _track_source_evidence(blueprint: dict[str, Any], limit: int = 1600, *, include_producer: bool = True) -> str:
    fields = {
        "track_number": blueprint.get("track_number"),
        "title": blueprint.get("title") or blueprint.get("locked_title"),
        "bpm": blueprint.get("bpm"),
        "key_scale": blueprint.get("key_scale"),
        "style": blueprint.get("style"),
        "vibe": blueprint.get("vibe"),
        "narrative": blueprint.get("narrative"),
        "description": blueprint.get("description"),
        "required_phrases": blueprint.get("required_phrases") or [],
        "source_excerpt": blueprint.get("source_excerpt") or "",
    }
    if include_producer:
        fields["producer_credit"] = blueprint.get("producer_credit")
    elif blueprint.get("producer_credit"):
        fields["producer_credit_policy"] = "metadata_only_do_not_use_in_lyrics"
    return _clip_text(json.dumps(_debug_jsonable(fields), ensure_ascii=False, indent=2), limit)


def _compact_lyric_blueprint_for_agent(blueprint: dict[str, Any]) -> dict[str, Any]:
    result = dict(_compact_blueprint_for_agent(blueprint))
    if result.pop("producer_credit", None):
        result["producer_credit_policy"] = "metadata_only_do_not_use_in_lyrics"
    return result


def _track_settings_prompt(
    *,
    concept: str,
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    lyric_plan: dict[str, Any],
    language: str,
    index: int,
    total: int,
    retrieved_context: str = "",
) -> str:
    duration = int(parse_duration_seconds(blueprint.get("duration") or lyric_plan.get("duration") or 180, 180))
    track_evidence = _track_source_evidence(blueprint, include_producer=False)
    return (
        "You are the AceJAM Track Settings Agent. Create ONLY the ACE-Step sound/settings package for this one track.\n"
        f"TRACK COUNTER: settings for track {index + 1} of {total}. No lyrics in this response.\n\n"
        f"TRACK_SOURCE_EVIDENCE_ONLY:\n{track_evidence}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_lyric_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"LYRIC_LENGTH_PLAN_FOR_CONTEXT_ONLY:\n{json.dumps({key: lyric_plan.get(key) for key in ('duration', 'sections', 'target_words', 'min_words', 'target_lines', 'min_lines')}, ensure_ascii=True, indent=2)}\n\n"
        f"FULL_TAG_LIBRARY_COMPACT:\n{json.dumps(_agent_tag_library_summary(), ensure_ascii=True, indent=2)}\n\n"
        "CAPTION RULE: caption is only comma-separated sound traits. It must cover: primary genre, drum groove, "
        "low-end/bass, melodic identity, vocal delivery, arrangement movement, texture/space, and mix/master. "
        "Do not include lyrics, section tags, BPM, key, duration, model, seed, JSON, or album story prose in caption.\n\n"
        "OUTPUT_SCHEMA:\n"
        '{"description":"","caption":"","tags":"","tag_list":[],"bpm":95,"key_scale":"A minor",'
        f'"time_signature":"4","duration":{duration},"language":"{language}","vocal_language":"{language}",'
        '"genre_profile":"","hook_promise":"","performance_brief":"","negative_control":"",'
        '"caption_dimensions_covered":[],"quality_checks":{"caption_lyrics_consistent":true}}\n'
        "Return strict JSON only. No lyrics."
    )


def _agent_tag_items(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"[,;\n]+", value)
    else:
        items = []
    result: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("tag") or item.get("name") or item.get("value") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _int_setting(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    match = re.search(r"-?\d+", str(value or ""))
    return int(match.group(0)) if match else int(default)


def _micro_setting_specs(*, duration: float, language: str) -> list[dict[str, Any]]:
    duration_int = int(parse_duration_seconds(duration or 180, 180))
    return [
        {
            "agent": "Track BPM Agent",
            "field": "bpm",
            "accept": ["bpm"],
            "schema": {"bpm": DEFAULT_BPM},
            "instruction": (
                "Choose only the BPM. If the user or scaffold locked BPM, return that exact integer. "
                "Otherwise choose a professional BPM that matches style, groove, and duration."
            ),
        },
        {
            "agent": "Track Key Agent",
            "field": "key_scale",
            "accept": ["key_scale", "key"],
            "schema": {"key_scale": DEFAULT_KEY_SCALE},
            "instruction": (
                "Choose only the ACE-Step key_scale. If locked, return it exactly. "
                "Use values like A minor, C major, F# minor, Bb major."
            ),
        },
        {
            "agent": "Track Time Signature Agent",
            "field": "time_signature",
            "accept": ["time_signature", "timesignature"],
            "schema": {"time_signature": "4"},
            "instruction": "Choose only the time signature. Default to 4 unless the brief clearly requires another supported meter.",
        },
        {
            "agent": "Track Duration Agent",
            "field": "duration",
            "accept": ["duration"],
            "schema": {"duration": duration_int},
            "instruction": (
                "Return only the locked duration in seconds. Do not shorten a full song into a demo. "
                f"The job duration is {duration_int} seconds."
            ),
        },
        {
            "agent": "Track Language Agent",
            "field": "language",
            "accept": ["language", "vocal_language"],
            "schema": {"language": language, "vocal_language": language},
            "instruction": "Return only language and vocal_language codes. Use the selected language unless the track is instrumental.",
        },
        {
            "agent": "Track Tag List Agent",
            "field": "tag_list",
            "accept": ["tag_list", "tags", "caption_dimensions_covered"],
            "schema": {
                "tag_list": [],
                "tags": "",
                "caption_dimensions_covered": ACE_STEP_CAPTION_DIMENSIONS,
            },
            "instruction": (
                "Choose only compact sonic tags. Cover every producer-grade dimension: primary genre, drum groove, "
                "low-end/bass, melodic identity, vocal delivery, arrangement movement, texture/space, and mix/master. "
                "Do not write a sentence and do not include lyrics."
            ),
            "include_tag_library": True,
        },
        {
            "agent": "Track Caption Agent",
            "field": "caption",
            "accept": ["caption", "tags"],
            "schema": {"caption": ""},
            "instruction": (
                "Write only the final ACE-Step caption under 512 chars. It must be comma-separated sound traits "
                "derived from the tag list. No lyrics, no section tags, no BPM/key/duration/model/seed, no story prose."
            ),
        },
        {
            "agent": "Track Description Agent",
            "field": "description",
            "accept": ["description"],
            "schema": {"description": ""},
            "instruction": "Write only a short internal description of the track concept. No lyrics and no metadata list.",
        },
        {
            "agent": "Track Hook Agent",
            "field": "hook_promise",
            "accept": ["hook_promise"],
            "schema": {"hook_promise": ""},
            "instruction": "Write only the central hook promise in one short sentence. Do not write hook lyrics yet.",
        },
        {
            "agent": "Track Performance Agent",
            "field": "performance_brief",
            "accept": ["performance_brief", "negative_control", "genre_profile"],
            "schema": {"performance_brief": "", "negative_control": "", "genre_profile": ""},
            "instruction": (
                "Write only a compact performance/mix brief plus negative_control and genre_profile. "
                "No lyrics, no runtime switches, no ACE-Step model settings."
            ),
        },
    ]


def _track_micro_setting_prompt(
    *,
    spec: dict[str, Any],
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    lyric_plan: dict[str, Any],
    language: str,
    index: int,
    total: int,
    prior_settings: dict[str, Any],
    retrieved_context: str = "",
) -> str:
    schema = spec.get("schema") or {}
    field = str(spec.get("field") or "")
    creative_field = field in {
        "tag_list",
        "caption",
        "description",
        "hook_promise",
        "performance_brief",
    }
    retrieved_block = _clip_text(retrieved_context or "[]", 700) if creative_field else "[]"
    bible_brief = {
        "album_title": album_bible.get("album_title"),
        "arc": _clip_text(album_bible.get("arc") or "", 180),
        "sonic_palette": _clip_text(album_bible.get("sonic_palette") or "", 220),
        "recurring_motifs": [
            _clip_text(item, 90)
            for item in (album_bible.get("recurring_motifs") or [])[:3]
            if str(item).strip()
        ],
    }
    tag_library = (
        f"\nFULL_TAG_LIBRARY_COMPACT:\n{json.dumps(_agent_tag_library_summary(), ensure_ascii=True, separators=(',', ':'))}\n"
        if spec.get("include_tag_library")
        else ""
    )
    return (
        f"You are {spec.get('agent')}. Decide exactly ONE micro-setting for an ACE-Step album track.\n"
        f"TRACK COUNTER: track {index + 1} of {total}. MICRO_SETTING: {field}.\n"
        "Do not plan the whole song. Do not write lyrics. Do not output extra fields beyond the schema.\n\n"
        f"TRACK_SOURCE_EVIDENCE_ONLY:\n{_track_source_evidence(blueprint, include_producer=False)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_block}\n\n"
        f"ALBUM_BIBLE_BRIEF:\n{json.dumps(bible_brief, ensure_ascii=True, separators=(',', ':'))}\n\n"
        f"PRIOR_MICRO_SETTINGS:\n{json.dumps(_debug_jsonable(prior_settings), ensure_ascii=True, separators=(',', ':'))}\n\n"
        f"LYRIC_PLAN_CONTEXT:\n{json.dumps({key: lyric_plan.get(key) for key in ('duration', 'target_words', 'min_words', 'target_lines', 'min_lines')}, ensure_ascii=True, separators=(',', ':'))}\n"
        f"{tag_library}\n"
        f"INSTRUCTION:\n{spec.get('instruction')}\n\n"
        f"OUTPUT_BLOCKS:\n{_agent_block_template(f'track_micro_{field}_payload')}\n"
        "Return delimiter blocks only."
    )


def _normalize_micro_settings_payload(
    settings: dict[str, Any],
    *,
    blueprint: dict[str, Any],
    language: str,
    duration: float,
) -> dict[str, Any]:
    result = dict(settings or {})
    fallback_bpm = _int_setting(blueprint.get("bpm") or DEFAULT_BPM, DEFAULT_BPM)
    result["bpm"] = max(30, min(300, _int_setting(result.get("bpm") or fallback_bpm, fallback_bpm)))
    key_scale = str(result.get("key_scale") or result.get("key") or blueprint.get("key_scale") or DEFAULT_KEY_SCALE).strip()
    result["key_scale"] = key_scale or DEFAULT_KEY_SCALE
    time_signature = str(result.get("time_signature") or result.get("timesignature") or blueprint.get("time_signature") or "4").strip()
    result["time_signature"] = time_signature or "4"
    result["duration"] = int(parse_duration_seconds(blueprint.get("duration") or result.get("duration") or duration, duration or 180))
    result["language"] = str(result.get("language") or blueprint.get("language") or language or "en").strip() or "en"
    result["vocal_language"] = str(result.get("vocal_language") or result.get("language") or language or "en").strip() or "en"

    tag_list = _agent_tag_items(result.get("tag_list"))
    if not tag_list:
        tag_list = _agent_tag_items(result.get("tags"))
    if not tag_list:
        tag_list = _agent_tag_items(blueprint.get("tag_list") or blueprint.get("tags"))
    if len(tag_list) < 7:
        for item in _baseline_caption_tags({**dict(blueprint or {}), **result}, ""):
            if item not in tag_list:
                tag_list.append(item)
    result["tag_list"] = tag_list
    result["tags"] = ", ".join(tag_list)

    caption = str(result.get("caption") or result.get("tags") or "").strip()
    result["caption"] = _clip_text(caption, 508)
    result["description"] = str(result.get("description") or blueprint.get("description") or blueprint.get("narrative") or "").strip()
    result["hook_promise"] = str(result.get("hook_promise") or blueprint.get("hook_promise") or "").strip()
    result["performance_brief"] = str(result.get("performance_brief") or blueprint.get("performance_brief") or "").strip()
    result["negative_control"] = str(result.get("negative_control") or "").strip()
    result["genre_profile"] = str(result.get("genre_profile") or blueprint.get("style") or "").strip()
    covered = result.get("caption_dimensions_covered")
    if not isinstance(covered, list):
        covered = []
    result["caption_dimensions_covered"] = [str(item) for item in covered if str(item).strip()]
    result["agent_micro_settings_flow"] = True
    return result


def _micro_setting_fallback_payload(
    field: str,
    *,
    blueprint: dict[str, Any],
    settings: dict[str, Any],
    language: str,
    duration: float,
) -> dict[str, Any]:
    merged = {**dict(blueprint or {}), **dict(settings or {})}
    tags = _agent_tag_items(merged.get("tag_list") or merged.get("tags"))
    if len(tags) < 7:
        for item in _baseline_caption_tags(merged, ""):
            if item not in tags:
                tags.append(item)
    caption = ", ".join(tags[:9])
    required = [str(item).strip() for item in (blueprint.get("required_phrases") or []) if str(item).strip()]
    narrative = str(blueprint.get("narrative") or blueprint.get("description") or "").strip()
    style = str(blueprint.get("style") or "").strip()
    vibe = str(blueprint.get("vibe") or "").strip()
    if field == "bpm":
        return {"bpm": _int_setting(blueprint.get("bpm") or settings.get("bpm") or DEFAULT_BPM, DEFAULT_BPM)}
    if field == "key_scale":
        return {"key_scale": str(blueprint.get("key_scale") or settings.get("key_scale") or DEFAULT_KEY_SCALE)}
    if field == "time_signature":
        return {"time_signature": str(blueprint.get("time_signature") or settings.get("time_signature") or "4")}
    if field == "duration":
        return {"duration": int(parse_duration_seconds(blueprint.get("duration") or settings.get("duration") or duration, duration or 180))}
    if field == "language":
        return {"language": str(language or settings.get("language") or "en"), "vocal_language": str(language or settings.get("vocal_language") or "en")}
    if field == "tag_list":
        return {
            "tag_list": tags[:12],
            "caption_dimensions_covered": ACE_STEP_CAPTION_DIMENSIONS,
        }
    if field == "caption":
        return {
            "caption": _clip_text(caption, 508),
            "caption_dimensions_covered": ACE_STEP_CAPTION_DIMENSIONS,
        }
    if field == "description":
        return {"description": _clip_text(narrative or f"{style}. {vibe}".strip(), 360)}
    if field == "hook_promise":
        return {"hook_promise": _clip_text(required[0] if required else narrative or vibe, 180)}
    if field == "performance_brief":
        return {"performance_brief": _clip_text(f"{style}; {vibe}; clear lead vocal, tight phrasing, radio-ready delivery", 260)}
    return {field: merged.get(field)}


def _call_track_micro_settings_agents(
    *,
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    lyric_plan: dict[str, Any],
    language: str,
    index: int,
    total: int,
    duration: float,
    planner_provider: str,
    planner_model: str,
    logs: list[str],
    opts: dict[str, Any],
    agent_stats: dict[str, Any],
    context_store: Any = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "bpm": blueprint.get("bpm") or DEFAULT_BPM,
        "key_scale": blueprint.get("key_scale") or DEFAULT_KEY_SCALE,
        "time_signature": blueprint.get("time_signature") or "4",
        "duration": int(parse_duration_seconds(blueprint.get("duration") or duration, duration or 180)),
        "language": language,
        "vocal_language": language,
    }
    agent_names: list[str] = []
    title = str(blueprint.get("title") or blueprint.get("locked_title") or f"track {index + 1}")
    for spec in _micro_setting_specs(duration=duration, language=language):
        agent_name = str(spec["agent"])
        field = str(spec["field"])
        agent_names.append(agent_name)
        if context_store is not None:
            retrieved_context = context_store.block(
                f"track {index + 1} micro setting {field} {title} {blueprint.get('style') or ''} {blueprint.get('vibe') or ''}",
                kinds=["contract_track", "track_blueprint", "album_bible", "track_summary"],
                track_number=index + 1,
                label=f"track_{index + 1}_micro_{field}",
            )
        else:
            retrieved_context = ""
        logs.append(f"Micro setting call: {agent_name} for track {index + 1}.")
        try:
            payload = _agent_json_call(
                agent_name=agent_name,
                provider=planner_provider,
                model_name=planner_model,
                user_prompt=_track_micro_setting_prompt(
                    spec=spec,
                    album_bible=album_bible,
                    blueprint=blueprint,
                    lyric_plan=lyric_plan,
                    language=language,
                    index=index,
                    total=total,
                    prior_settings=settings,
                    retrieved_context=retrieved_context,
                ),
                logs=logs,
                debug_options=opts,
                schema_name=f"track_micro_{field}_payload",
                extra_system=(
                    "Micro-agent mode: decide only the requested field. "
                    "No lyrics, no full track plan, no runtime switches, no markdown."
                ),
                max_retries=0,
            )
        except Exception as exc:
            logs.append(
                f"Micro setting fallback: {agent_name} for track {index + 1} "
                f"({type(exc).__name__}: {_monitor_preview(exc, 180)})."
            )
            payload = _micro_setting_fallback_payload(field, blueprint=blueprint, settings=settings, language=language, duration=duration)
            _append_album_debug_jsonl(
                opts,
                "04_micro_settings.jsonl",
                {
                    "track_number": index + 1,
                    "title": title,
                    "agent": agent_name,
                    "field": field,
                    "fallback_error": f"{type(exc).__name__}: {exc}",
                    "payload": payload,
                    "settings_after": settings,
                },
            )
            agent_stats.setdefault("agent_rounds", []).append({
                "agent": agent_name,
                "track_number": index + 1,
                "field": field,
                "status": "fallback",
                "error": f"{type(exc).__name__}: {exc}",
            })
        for key in spec.get("accept") or [field]:
            if payload.get(key) not in (None, "", []):
                canonical = "key_scale" if key == "key" else "time_signature" if key == "timesignature" else key
                settings[canonical] = payload.get(key)
        if field not in settings and payload.get("value") not in (None, "", []):
            settings[field] = payload.get("value")
        _append_album_debug_jsonl(
            opts,
            "04_micro_settings.jsonl",
            {
                "track_number": index + 1,
                "title": title,
                "agent": agent_name,
                "field": field,
                "payload": payload,
                "settings_after": settings,
            },
        )
        agent_stats.setdefault("agent_rounds", []).append({
            "agent": agent_name,
            "track_number": index + 1,
            "field": field,
            "status": "completed",
        })
    normalized = _normalize_micro_settings_payload(
        settings,
        blueprint=blueprint,
        language=language,
        duration=duration,
    )
    normalized["micro_setting_agents"] = agent_names
    _append_album_debug_jsonl(
        opts,
        "04_micro_settings.jsonl",
        {
            "track_number": index + 1,
            "title": title,
            "agent": "Micro Settings Aggregator",
            "field": "all",
            "payload": normalized,
        },
    )
    return normalized


def _lyric_section_groups(sections: list[Any], max_parts: int | None = None) -> list[list[str]]:
    clean = [str(section or "").strip().strip("[]") for section in sections if str(section or "").strip()]
    if not clean:
        clean = ["Intro", "Verse 1", "Chorus", "Verse 2", "Bridge", "Final Chorus", "Outro"]
    part_count = max(1, min(len(clean), int(max_parts or ACEJAM_AGENT_LYRIC_PARTS)))
    group_size = max(1, (len(clean) + part_count - 1) // part_count)
    return [clean[index:index + group_size] for index in range(0, len(clean), group_size)]


def _lyric_part_targets(lyric_plan: dict[str, Any], groups: list[list[str]], part_index: int) -> dict[str, int]:
    group_count = max(1, len(groups))
    sections = groups[part_index] if 0 <= part_index < len(groups) else []
    target_words = int(lyric_plan.get("target_words") or lyric_plan.get("min_words") or 160)
    min_words = int(lyric_plan.get("min_words") or 0)
    target_lines = int(lyric_plan.get("target_lines") or lyric_plan.get("min_lines") or 32)
    min_lines = int(lyric_plan.get("min_lines") or 0)
    safe_chars = int(lyric_plan.get("safe_lyrics_char_target") or ACE_STEP_LYRICS_SOFT_TARGET_MAX)
    max_chars_total = min(int(lyric_plan.get("max_lyrics_chars") or ACE_STEP_LYRICS_CHAR_LIMIT), safe_chars)
    per_part_chars = max(260, int(round(max_chars_total / group_count)))
    section_floor = max(2, len(sections) * 3)
    return {
        "target_words": max(12, int(round(target_words / group_count))),
        "min_words": max(0, int(round(min_words / group_count))),
        "target_lines": max(section_floor, int(round(target_lines / group_count))),
        "min_lines": max(len(sections) * 2, int(round(min_lines / group_count))),
        "target_chars": max(220, int(per_part_chars * 0.82)),
        "max_chars": per_part_chars,
    }


def _required_phrases_for_part(blueprint: dict[str, Any], part_index: int, part_count: int) -> list[str]:
    phrases: list[str] = []
    for key in ("required_phrases", "required_lyrics"):
        value = blueprint.get(key)
        if isinstance(value, str):
            phrases.extend(line.strip() for line in value.splitlines() if line.strip())
        elif isinstance(value, list):
            phrases.extend(str(item or "").strip() for item in value if str(item or "").strip())
    if not phrases:
        return []
    total = max(1, int(part_count or 1))
    return [phrase for idx, phrase in enumerate(phrases) if idx % total == part_index]


def _track_lyrics_part_prompt(
    *,
    concept: str,
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    settings_payload: dict[str, Any],
    lyric_plan: dict[str, Any],
    section_group: list[str],
    part_index: int,
    part_count: int,
    previous_parts: list[dict[str, Any]],
    language: str,
    track_index: int,
    total_tracks: int,
    retrieved_context: str = "",
) -> str:
    target_groups = [[] for _ in range(max(1, part_count))]
    if 0 <= part_index < len(target_groups):
        target_groups[part_index] = section_group
    targets = _lyric_part_targets(lyric_plan, target_groups, part_index)
    required_phrases = _required_phrases_for_part(blueprint, part_index, part_count)
    previous_brief = [
        {
            "part_index": item.get("part_index"),
            "sections": item.get("sections"),
            "last_lines": (item.get("lyrics_lines") or [])[-8:],
            "hook_lines": item.get("hook_lines") or [],
        }
        for item in previous_parts[-2:]
        if isinstance(item, dict)
    ]
    track_evidence = _track_source_evidence(blueprint, include_producer=False)
    bible_brief = {
        "album_title": album_bible.get("album_title"),
        "arc": _clip_text(album_bible.get("arc") or "", 180),
        "sonic_palette": _clip_text(album_bible.get("sonic_palette") or "", 220),
        "recurring_motifs": [
            _clip_text(item, 90)
            for item in (album_bible.get("recurring_motifs") or [])[:3]
            if str(item).strip()
        ],
    }
    blueprint_brief = _compact_lyric_blueprint_for_agent(blueprint)
    settings_brief = {
        key: settings_payload.get(key)
        for key in ("caption", "hook_promise", "performance_brief", "language", "vocal_language")
    }
    return (
        "You are the AceJAM Track Lyrics Agent. Write ONLY this small lyric part; do not output settings/caption.\n"
        f"TRACK COUNTER: track {track_index + 1} of {total_tracks}. LYRIC PART: {part_index + 1} of {part_count}.\n\n"
        f"TRACK_SOURCE_EVIDENCE_ONLY:\n{track_evidence}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{_clip_text(retrieved_context or '[]', 650)}\n\n"
        f"ALBUM_BIBLE_BRIEF:\n{json.dumps(bible_brief, ensure_ascii=True, separators=(',', ':'))}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(blueprint_brief, ensure_ascii=True, separators=(',', ':'))}\n\n"
        f"TRACK_SETTINGS_CONTEXT:\n{json.dumps(settings_brief, ensure_ascii=True, separators=(',', ':'))}\n\n"
        f"WRITE_THESE_SECTIONS_ONLY:\n{json.dumps([f'[{section}]' for section in section_group], ensure_ascii=True)}\n\n"
        f"PART_TARGETS:\n{json.dumps(targets, ensure_ascii=True, separators=(',', ':'))}\n\n"
        f"REQUIRED_PHRASES_FOR_THIS_PART:\n{json.dumps(required_phrases, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"PREVIOUS_LYRIC_PARTS_CONTEXT:\n{json.dumps(previous_brief, ensure_ascii=True, separators=(',', ':'))}\n\n"
        f"LANGUAGE: {language}. Use the correct script and natural rhythm for this language.\n\n"
        "LYRIC RULES:\n"
        "- Start every requested section with its bracket tag.\n"
        "- Write actual performable lyric lines only: 3-8 words per line where possible.\n"
        "- Stay under PART_TARGETS.max_chars for this part. If you are running out of budget, write fewer complete lines; never end mid-line.\n"
        "- Rap lines need breath-control, cadence, internal rhyme, and bar momentum.\n"
        "- Hooks/choruses must be short, repeatable, and connected to the title/hook promise.\n"
        "- Include every REQUIRED_PHRASE exactly if provided for this part.\n"
        "- Producer credits and real-person names from metadata are not lyrics; never write them as sung or rapped lines.\n"
        "- No caption, no metadata, no BPM/key/duration, no prose explanation, no placeholders, no markdown.\n\n"
        "OUTPUT_SCHEMA:\n"
        '{"part_index":1,"sections":[],"lyrics_lines":[],"required_phrases_used":[],"hook_lines":[],'
        '"word_count":0,"line_count":0,"char_count":0,"quality_checks":{"short_lines":true,"under_char_budget":true,"no_placeholders":true}}\n'
        "Return strict JSON only."
    )


def _lyrics_part_fallback_payload(
    *,
    blueprint: dict[str, Any],
    settings_payload: dict[str, Any],
    lyric_plan: dict[str, Any],
    section_group: list[str],
    part_index: int,
    part_count: int,
    language: str,
) -> dict[str, Any]:
    target_groups = [[] for _ in range(max(1, part_count))]
    if 0 <= part_index < len(target_groups):
        target_groups[part_index] = section_group
    targets = _lyric_part_targets(lyric_plan, target_groups, part_index)
    required = _required_phrases_for_part(blueprint, part_index, part_count)
    title = str(blueprint.get("title") or blueprint.get("locked_title") or "the song").strip()
    hook = str(settings_payload.get("hook_promise") or (required[0] if required else title)).strip()
    vibe = str(blueprint.get("vibe") or blueprint.get("narrative") or "").strip()
    min_lines = max(1, int(targets.get("min_lines") or 0))
    lines: list[str] = []
    phrase_index = 0
    for section in section_group or ["Verse"]:
        section_tag = _section_tag_line(section)
        lines.append(section_tag)
        section_line_target = max(2, min_lines // max(1, len(section_group or [section])))
        for idx in range(section_line_target):
            if phrase_index < len(required):
                lines.append(required[phrase_index])
                phrase_index += 1
                continue
            if re.search(r"chorus|hook|refrain", section_tag, re.I):
                lines.append(_clip_text(hook, 90) or title)
            elif language.lower().startswith("de"):
                lines.append(_clip_text(f"{title} leuchtet hell durch die Nacht", 90))
                if len(lines) < min_lines + len(section_group):
                    lines.append(_clip_text("Wir singen zusammen, warm und klar", 90))
            else:
                lines.append(_clip_text(f"{title} glows bright through the night", 90))
                if len(lines) < min_lines + len(section_group):
                    lines.append(_clip_text("We sing together, warm and clear", 90))
    while phrase_index < len(required):
        lines.append(required[phrase_index])
        phrase_index += 1
    while len([line for line in lines if not line.startswith("[")]) < min_lines:
        lines.append(_clip_text(vibe or hook or title, 90))
    stats = lyric_stats("\n".join(lines))
    return {
        "part_index": part_index + 1,
        "sections": [_section_tag_line(section) for section in section_group],
        "lyrics_lines": lines,
        "required_phrases_used": required,
        "hook_lines": [line for line in lines if hook and hook.lower() in line.lower()][:3],
        "word_count": stats.get("word_count"),
        "line_count": stats.get("line_count"),
        "quality_checks": {"fallback": True, "short_lines": True, "no_placeholders": True},
    }


def _fit_lyric_lines_to_char_budget(lines: list[Any], max_chars: int) -> tuple[list[str], bool]:
    budget = max(120, int(max_chars or 0))
    fitted: list[str] = []
    changed = False
    for raw in lines:
        line = str(raw or "").rstrip()
        if not line:
            continue
        candidate = "\n".join([*fitted, line]).strip()
        if len(candidate) > budget:
            changed = True
            continue
        fitted.append(line)
    if not fitted and lines:
        for raw in lines:
            line = str(raw or "").rstrip()
            if line.startswith("[") and line.endswith("]") and len(line) <= budget:
                return [line], True
    return fitted, changed


def _enforce_lyric_part_budget(
    payload: dict[str, Any],
    lyric_plan: dict[str, Any],
    section_groups: list[list[str]],
    part_index: int,
) -> dict[str, Any]:
    result = dict(payload or {})
    targets = _lyric_part_targets(lyric_plan, section_groups, part_index)
    max_chars = int(targets.get("max_chars") or 0)
    lines = _agent_payload_lines(result)
    if not lines:
        return result
    joined = "\n".join(lines).strip()
    if max_chars and len(joined) > max_chars:
        fitted, changed = _fit_lyric_lines_to_char_budget(lines, max_chars)
        if changed:
            result["lyrics_lines"] = fitted
            result["lyrics"] = "\n".join(fitted).strip()
            result.setdefault("quality_checks", {})
            if isinstance(result["quality_checks"], dict):
                result["quality_checks"]["budget_repaired"] = True
    stats = lyric_stats(str(result.get("lyrics") or "\n".join(result.get("lyrics_lines") or [])))
    result["word_count"] = int(stats.get("word_count") or 0)
    result["line_count"] = int(stats.get("line_count") or 0)
    result["char_count"] = int(stats.get("char_count") or 0)
    return result


def _track_lyrics_continuation_prompt(
    *,
    blueprint: dict[str, Any],
    settings_payload: dict[str, Any],
    lyric_plan: dict[str, Any],
    current_lyrics: str,
    language: str,
    index: int,
    total: int,
    missing_words: int,
    missing_lines: int,
) -> str:
    stats = lyric_stats(current_lyrics)
    current_tail = "\n".join(str(current_lyrics or "").splitlines()[-28:])
    return (
        "You are the AceJAM Track Lyric Continuation Agent. Add only the missing lyric material for the SAME track.\n"
        f"TRACK COUNTER: track {index + 1} of {total}. Do not rewrite existing lyrics.\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_lyric_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"TRACK_SETTINGS_CONTEXT:\n{json.dumps({key: settings_payload.get(key) for key in ('caption', 'hook_promise', 'performance_brief', 'language')}, ensure_ascii=True, indent=2)}\n\n"
        f"LYRIC_PLAN:\n{json.dumps({key: lyric_plan.get(key) for key in ('sections', 'target_words', 'min_words', 'target_lines', 'min_lines', 'max_lyrics_chars')}, ensure_ascii=True, indent=2)}\n\n"
        f"CURRENT_STATS:\n{json.dumps({'word_count': stats.get('word_count'), 'line_count': stats.get('line_count'), 'missing_words': missing_words, 'missing_lines': missing_lines}, ensure_ascii=True, indent=2)}\n\n"
        f"CURRENT_LYRICS_TAIL:\n{_clip_text(current_tail, 1800)}\n\n"
        f"LANGUAGE: {language}\n\n"
        "Write a natural bridge/final chorus/outro continuation that closes the song. "
        "Use short performable lines, no filler, no metadata, no producer-credit names, no caption, no placeholders. "
        "Do not repeat a line more than twice.\n\n"
        "OUTPUT_SCHEMA:\n"
        '{"lyrics_lines":["[Bridge - extension]","..."],"word_count":0,"line_count":0,"quality_checks":{"no_filler":true}}\n'
        "Return strict JSON only."
    )


def _agent_memory_requested(opts: dict[str, Any]) -> bool:
    for key in ("agent_memory_enabled", "memory_enabled", "use_agent_memory"):
        if key in (opts or {}):
            value = opts.get(key)
            if isinstance(value, str):
                return value.strip().lower() not in {"0", "false", "no", "off"}
            return bool(value)
    return ACEJAM_AGENT_MEMORY_DEFAULT


def _recover_album_concept(concept: Any, options: dict[str, Any] | None = None, input_tracks: list[dict[str, Any]] | None = None) -> str:
    opts = options or {}
    parts: list[str] = []
    primary_prompt = next(
        (
            str(value).strip()
            for value in (opts.get("raw_user_prompt"), opts.get("user_prompt"), opts.get("prompt"))
            if isinstance(value, str) and value.strip()
        ),
        "",
    )
    candidate_values = [primary_prompt] if primary_prompt else [concept, opts.get("concept")]
    candidate_values.extend([opts.get("album_title"), opts.get("album_name")])
    for value in candidate_values:
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if not primary_prompt and len(parts) <= 1:
        track_candidates = input_tracks or opts.get("editable_plan_tracks") or opts.get("tracks") or opts.get("planned_tracks") or []
    else:
        track_candidates = []
    if isinstance(track_candidates, list):
        for idx, item in enumerate(track_candidates[:30]):
            if not isinstance(item, dict):
                continue
            text = " ".join(
                str(item.get(k) or "")
                for k in ("style", "vibe", "narrative", "description")
                if item.get(k)
            )
            title = str(item.get("title") or "").strip()
            fields = []
            if item.get("style"):
                fields.append(f"Style: {item.get('style')}")
            if text.strip():
                fields.append(text.strip())
            if title or fields:
                lines = [f'Track {idx + 1}: "{title}"'] if title else [f"Track {idx + 1}:"]
                lines.extend(fields)
                parts.append("\n".join(lines))
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in str(part or "").splitlines()
            if line.strip()
        ).strip()
        key = re.sub(r"\s+", " ", clean).casefold()
        if clean and key not in seen:
            deduped.append(clean)
            seen.add(key)
    return "\n".join(deduped).strip()


class AlbumAgentPromptLibrary:
    def __init__(self, options: dict[str, Any], language: str) -> None:
        self.options = options or {}
        self.language = language or "en"
        self.md_path = ACEJAM_PROMPT_KIT_MD_PATH
        self.md_available = self.md_path.is_file()

    def system_rules(self) -> str:
        language_info = language_preset(self.language)
        genre_prompt = str(self.options.get("album_agent_genre_prompt") or self.options.get("genre_prompt") or "").strip()
        mood = str(self.options.get("album_agent_mood_vibe") or "").strip()
        vocal = str(self.options.get("album_agent_vocal_type") or "").strip()
        audience = str(self.options.get("album_agent_audience") or "").strip()
        genre_modules = infer_genre_modules(" ".join([genre_prompt, mood]), max_modules=2)
        genre_bits: list[str] = []
        for module in genre_modules:
            if not isinstance(module, dict):
                continue
            slug = str(module.get("slug") or "").strip()
            caption_dna = ", ".join(str(item) for item in (module.get("caption_dna") or [])[:3])
            section_bias = ", ".join(str(item) for item in (module.get("section_bias") or [])[:3])
            genre_bits.append(f"{slug}: {caption_dna}; sections {section_bias}".strip(": ;"))
        # Inject the full prompt_kit album-mode reference: tag taxonomy, authoring
        # rules, producer-format cookbook, rap-mode cookbook, songwriter craft,
        # anti-patterns and worked examples. This is the same block prompt-assistant
        # routes inject for non-album modes; album agents previously ran without it
        # which left producer requests, craft moves and bar-floor enforcement weak.
        rich_kit_block = prompt_kit_system_block("album")
        return (
            "ACEJAM PROMPT-FIRST V2 RULES\n"
            "One small decision per call. Answer the schema only.\n"
            "ACE-Step: caption<512 sound-only; lyrics<4096 temporal script; metadata separate; complete agent lyrics bypass ACE LM rewrite.\n"
            "Lyrics: section tags exact, complete lines, clear section separation, 6-10 syllables as a guide, breaks for long tracks.\n"
            "Rap verses: minimum 16 bars per [Verse - rap] section on tracks >=120s. Multisyllabic mosaic rhymes stacked in begin/middle/end of bars; slant-dominant with perfect-rhyme landings on emphasis. Pack 8-15 syllables per bar.\n"
            "Never put BPM/key/duration/model/seed/title/story/producer/person names in caption or lyrics unless user explicitly wrote a sung line.\n"
            "Producer references: never put producer names in caption. Use the Producer-Format Cookbook below to translate to genre+era+drum+timbre stacks.\n"
            "Anti-pattern guard: forbid AI-cliche image bank (neon dreams, fire inside, shattered dreams, endless night, empty streets, embers, whispers, silhouettes, echoes, we rise, let it burn, chasing the night). Forbid telling-not-showing labels and generic POV.\n"
            f"Language={language_info.get('code') or self.language} {language_info.get('name') or ''}; genre_prompt={genre_prompt or 'not specified'}; "
            f"mood={mood or 'follow concept'}; vocal={vocal or 'choose'}; audience={audience or 'streaming release'}; "
            f"genre_hints={' | '.join(genre_bits) if genre_bits else 'none'}.\n\n"
            "================================================================\n"
            "ACE-STEP REFERENCE BLOCK (full tag library, authoring rules, producer/rap/songwriter cookbooks, anti-patterns, worked examples):\n"
            "================================================================\n"
            f"{rich_kit_block}\n"
        )


def _director_section_tags(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("section_map") or payload.get("sections") or payload.get("section_tags") or []
    if isinstance(raw, str):
        raw = [item.strip() for item in re.split(r"[,;\n]+", raw) if item.strip()]
    tags: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            text = str(item.get("tag") or item.get("section") or item.get("name") or "").strip()
        else:
            text = str(item or "").strip()
        if not text:
            continue
        tag = text if text.startswith("[") else f"[{text.strip('[]')}]"
        if tag not in tags:
            tags.append(tag)
    return tags


def _director_section_groups(section_tags: list[str]) -> list[list[str]]:
    tags = list(section_tags or [])
    if not tags:
        return []
    if len(tags) <= 3:
        return [tags]
    return [group for group in (tags[:3], tags[3:6], tags[6:]) if group]


def _director_section_line_minimums(section_tags: list[str], *, duration: float, genre_hint: str = "") -> dict[str, int]:
    rap = bool(re.search(r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b", genre_hint or "", re.I))
    dur = int(float(duration or 0))
    long_form = dur >= 210
    minimums: dict[str, int] = {}
    for tag in section_tags or []:
        label = str(tag or "")
        lower = label.lower()
        if rap and "verse" in lower:
            # 16-bar rap-verse floor for full songs (120s+) so they get
            # real verse density matching the 3-verses template. Shorter
            # tracks scale: 90-119s → 8 lines, 60-89s → 4 lines (skit
            # verse), <60s → 3 lines (snippet/instrumental). Keeps short
            # smoke fixtures usable while raising the floor for full songs.
            if dur >= 120:
                minimums[label] = 16
            elif dur >= 90:
                minimums[label] = 8
            elif dur >= 60:
                minimums[label] = 4
            else:
                minimums[label] = 3
        elif rap and re.search(r"beat\s*switch", lower):
            # Beat-switch is a transition section — 2-3 lines is enough.
            # Was 4-6 which made agents pad with low-quality lines.
            minimums[label] = 3 if long_form else 2
        elif rap and re.search(r"bridge", lower):
            minimums[label] = 6 if long_form else 4 if dur >= 120 else 2
        elif re.search(r"chorus|hook|refrain", lower):
            minimums[label] = 3 if long_form else 2
        elif re.search(r"intro|outro", lower):
            minimums[label] = 2
        elif rap:
            minimums[label] = 3
    return minimums


def _compact_json(value: Any, limit: int | None = None) -> str:
    text = json.dumps(_debug_jsonable(value), ensure_ascii=False, separators=(",", ":"))
    return _clip_text(text, int(limit)) if limit else text


def _clip_context_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        limits = {
            "required_lyrics": 700,
            "lyrics": 900,
            "description": 320,
            "narrative": 320,
            "style": 260,
            "vibe": 260,
            "tags": 420,
            "caption": 420,
            "performance_brief": 360,
            "negative_control": 220,
            "genre_profile": 220,
        }
        return _clip_text(value, limits.get(key, 240))
    if isinstance(value, list):
        limit = 14 if key in {"lyrics_lines"} else 10
        return [_clip_context_value(key, item) for item in value[:limit]]
    if isinstance(value, dict):
        return {str(k): _clip_context_value(str(k), v) for k, v in list(value.items())[:12]}
    return value


def _director_payload_lines(payload: dict[str, Any]) -> list[str]:
    lines = _agent_payload_lines(payload)
    return [line for line in lines if str(line or "").strip()]


def _director_section_blocks_from_lines(lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in [str(item or "").strip() for item in lines if str(item or "").strip()]:
        if re.fullmatch(r"\[[^\]]+\]", line):
            current = {"tag": line, "key": _section_key_for_director(line), "lines": []}
            blocks.append(current)
            continue
        if current is None:
            current = {"tag": "[untagged]", "key": "untagged", "lines": []}
            blocks.append(current)
        current["lines"].append(line)
    return blocks


def _director_replace_section_blocks(
    original_lines: list[str],
    replacement_lines: list[str],
    target_sections: list[str],
) -> tuple[list[str], bool, list[str]]:
    issues: list[str] = []
    target_keys = {_section_key_for_director(tag) for tag in target_sections}
    target_by_key = {_section_key_for_director(tag): tag for tag in target_sections}
    replacement_blocks = _director_section_blocks_from_lines(replacement_lines)
    replacement_by_key: dict[str, list[str]] = {}
    for block in replacement_blocks:
        key = str(block.get("key") or "")
        if key == "untagged":
            issues.append("replacement_lyrics_before_first_section_tag")
            continue
        if key not in target_keys:
            issues.append(f"replacement_extra_section:{block.get('tag')}")
            continue
        if key in replacement_by_key:
            issues.append(f"replacement_duplicate_section:{block.get('tag')}")
            continue
        replacement_by_key[key] = [target_by_key[key], *[str(line) for line in block.get("lines") or [] if str(line).strip()]]
    missing = [tag for tag in target_sections if _section_key_for_director(tag) not in replacement_by_key]
    if missing:
        issues.append("replacement_missing_sections:" + ",".join(missing))
    if issues:
        return original_lines, False, issues
    merged: list[str] = []
    pos = 0
    replaced = False
    while pos < len(original_lines):
        line = str(original_lines[pos] or "").strip()
        if re.fullmatch(r"\[[^\]]+\]", line) and _section_key_for_director(line) in target_keys:
            key = _section_key_for_director(line)
            merged.extend(replacement_by_key[key])
            replaced = True
            pos += 1
            while pos < len(original_lines) and not re.fullmatch(r"\[[^\]]+\]", str(original_lines[pos] or "").strip()):
                pos += 1
            continue
        merged.append(line)
        pos += 1
    return merged, replaced, []


def _section_key_for_director(tag: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(tag or "").lower())


def _caption_forbidden_markers(caption: str, track: dict[str, Any] | None = None) -> list[str]:
    text = str(caption or "")
    issues: list[str] = []
    if CAPTION_METADATA_RE.search(text):
        issues.append("metadata_or_credit_in_caption")
    track_data = track or {}
    for field in ("producer_credit", "artist_name", "title"):
        value = str(track_data.get(field) or "").strip()
        if value and value.lower() in text.lower():
            issues.append(f"{field}_in_caption")
    if re.search(r"\b(?:verse|chorus|hook|lyrics?|narrative|story)\s*[:=-]", text, re.I):
        issues.append("lyric_or_story_marker_in_caption")
    return sorted(set(issues))


def _final_payload_preserve_sources(current: dict[str, Any], final_payload: dict[str, Any]) -> dict[str, Any]:
    """Merge final assembler output without letting it rewrite source-of-truth agent fields."""
    track = {**current}
    for key, value in (final_payload or {}).items():
        if key in {
            "caption",
            "tags",
            "tag_list",
            "lyrics",
            "lyrics_lines",
            "bpm",
            "key_scale",
            "time_signature",
            "duration",
            "language",
            "vocal_language",
        } and track.get(key) not in (None, "", []):
            continue
        track[key] = value
    return track


def _director_build_final_payload(current: dict[str, Any], language: str) -> dict[str, Any]:
    return {
        "track_number": current.get("track_number"),
        "title": current.get("title") or current.get("locked_title") or "",
        "description": current.get("description") or "",
        "caption": current.get("caption") or current.get("tags") or "",
        "tags": current.get("tags") or current.get("caption") or "",
        "tag_list": current.get("tag_list") or [],
        "lyrics_lines": current.get("lyrics_lines") or [],
        "bpm": current.get("bpm"),
        "key_scale": current.get("key_scale"),
        "time_signature": current.get("time_signature"),
        "duration": current.get("duration"),
        "language": current.get("language") or current.get("vocal_language") or language,
        "performance_brief": current.get("performance_brief") or "",
        "genre_profile": current.get("genre_profile") or "",
        "genre_intent_contract": current.get("genre_intent_contract") or {},
        "genre_adherence": current.get("genre_adherence") or {},
        "quality_checks": {"deterministic_final_payload": True},
    }


def _validate_lyrics_part_payload(
    payload: dict[str, Any],
    *,
    expected_sections: list[str],
    forbidden_sections: list[str],
    expected_part_index: int,
) -> list[str]:
    issues: list[str] = []
    if int(payload.get("part_index") or 0) != expected_part_index:
        issues.append(f"wrong_part_index:{payload.get('part_index')}!={expected_part_index}")
    actual_sections = _director_section_tags({"section_map": payload.get("sections") or []})
    if [_section_key_for_director(item) for item in actual_sections] != [_section_key_for_director(item) for item in expected_sections]:
        issues.append("sections_mismatch")
    lines = _director_payload_lines(payload)
    line_tags = re.findall(r"\[[^\]]+\]", "\n".join(lines))
    expected_keys = [_section_key_for_director(item) for item in expected_sections]
    seen_keys = [_section_key_for_director(item) for item in line_tags]
    section_line_tags = [line for line in lines if re.fullmatch(r"\[[^\]]+\]", str(line or "").strip())]
    section_line_keys = [_section_key_for_director(item) for item in section_line_tags]
    extra_tags = [tag for tag in line_tags if _section_key_for_director(tag) not in expected_keys]
    missing_tags = [tag for tag in expected_sections if _section_key_for_director(tag) not in seen_keys]
    duplicate_tags = sorted({tag for tag in line_tags if seen_keys.count(_section_key_for_director(tag)) > 1})
    forbidden_hits = [tag for tag in line_tags if _section_key_for_director(tag) in {_section_key_for_director(item) for item in forbidden_sections}]
    first_section_index = next((idx for idx, line in enumerate(lines) if re.fullmatch(r"\[[^\]]+\]", str(line or "").strip())), None)
    if first_section_index is not None and first_section_index > 0:
        issues.append("lyrics_before_first_section_tag")
    if section_line_keys and section_line_keys != expected_keys:
        issues.append("section_tag_order_mismatch")
    if extra_tags:
        issues.append("unexpected_section_tags:" + ",".join(extra_tags))
    if missing_tags:
        issues.append("missing_section_tags:" + ",".join(missing_tags))
    if duplicate_tags:
        issues.append("duplicate_section_tags:" + ",".join(duplicate_tags))
    if forbidden_hits:
        issues.append("forbidden_section_tags:" + ",".join(forbidden_hits))
    if not lines:
        issues.append("empty_lyrics_lines")
    # Per-line minimum content length on non-tag lines. Padding lines like
    # "yeah", "uh", or single-word fillers slip past min_lines counting.
    # 12 chars is the floor for a line that actually carries lyric content.
    non_tag_lines = [
        str(line or "").strip()
        for line in lines
        if str(line or "").strip() and not re.fullmatch(r"\[[^\]]+\]", str(line or "").strip())
    ]
    short_lines = [line for line in non_tag_lines if len(line) < 12]
    # Allow a small budget of short lines (ad-libs, one-word punctuation) but
    # flag if more than 20% of the lyric body is sub-12-char filler.
    if non_tag_lines and len(short_lines) > max(2, len(non_tag_lines) // 5):
        issues.append(
            f"lyrics_too_many_short_lines:{len(short_lines)}/{len(non_tag_lines)}_lines_under_12_chars"
        )
    # Cliche / telling-not-showing / generic-POV detection on the lyric body.
    lyric_text = "\n".join(non_tag_lines)
    cliche_hits = _scan_for_cliche_phrases(lyric_text)
    if cliche_hits:
        issues.append("lyrics_contain_cliche_phrases:" + ",".join(cliche_hits[:5]))
    return issues


def _raw_director_section_tags(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("section_map") or payload.get("sections") or payload.get("section_tags") or []
    if isinstance(raw, str):
        raw = [item.strip() for item in re.split(r"[,;\n]+", raw) if item.strip()]
    tags: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            text = str(item.get("tag") or item.get("section") or item.get("name") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            tags.append(text if text.startswith("[") else f"[{text.strip('[]')}]")
    return tags


def _validate_track_concept_payload(payload: dict[str, Any]) -> list[str]:
    """Track Concept Agent must populate ALL five fields. Empty values
    cascade into empty wizard cells (the user's complaint: 'Stijl', 'Vibe',
    'Narrative' staying blank). The repair-loop reads these issue codes and
    re-prompts the agent until they fill every slot."""
    issues: list[str] = []
    payload = payload or {}
    for key in ("title", "description", "style", "vibe", "narrative"):
        value = str(payload.get(key) or "").strip()
        if not value:
            issues.append(f"missing_{key}")
    # Style must have >=2 words: rejects single-word generic outputs like
    # "rap" or "pop" while accepting real stacks like "warm boom-bap" or
    # "modern trap with sub-808". Char count would either over-reject ("warm
    # boom-bap" is 13 chars) or under-reject ("hip-hop pop" is 11 chars).
    style = str(payload.get("style") or "").strip()
    if style and len(style.split()) < 2:
        issues.append(f"style_too_generic:{style!r}_needs_at_least_2_words")
    # Narrative must have >=4 words: rejects "a sad song" placeholders.
    narrative = str(payload.get("narrative") or "").strip()
    if narrative and len(narrative.split()) < 4:
        issues.append(f"narrative_too_short:{narrative!r}_needs_at_least_4_words")
    return issues


def _validate_tag_payload(payload: dict[str, Any], track: dict[str, Any] | None = None) -> list[str]:
    issues: list[str] = []
    tag_list = payload.get("tag_list") if isinstance(payload, dict) else None
    clean_tags = [str(item or "").strip() for item in tag_list] if isinstance(tag_list, list) else []
    clean_tags = [item for item in clean_tags if item]
    tags = str((payload or {}).get("tags") or "").strip()
    if not clean_tags:
        issues.append("missing_tag_list")
    if not tags:
        issues.append("missing_tags")
    combined = ", ".join([tags, *clean_tags]).strip(", ")
    if combined:
        for issue in _caption_forbidden_markers(combined, track):
            issues.append(issue.replace("_caption", "_tags"))
        if re.search(r"\[[^\]]+\]", combined):
            issues.append("section_tag_in_tags")
    return sorted(set(issues))


def _validate_bpm_payload(payload: dict[str, Any]) -> list[str]:
    value = (payload or {}).get("bpm")
    try:
        bpm = float(value)
    except Exception:
        return ["invalid_bpm"]
    if bpm < 40 or bpm > 220:
        return [f"bpm_out_of_range:{value}"]
    return []


def _validate_key_payload(payload: dict[str, Any]) -> list[str]:
    key_scale = str((payload or {}).get("key_scale") or "").strip()
    if not key_scale:
        return ["missing_key_scale"]
    if not re.search(r"\b[A-G](?:#|b|♯|♭)?\s+(?:major|minor)\b", key_scale, re.I):
        return [f"invalid_key_scale:{key_scale}"]
    return []


def _validate_time_signature_payload(payload: dict[str, Any]) -> list[str]:
    value = str((payload or {}).get("time_signature") or "").strip()
    if not value:
        return ["missing_time_signature"]
    if not re.fullmatch(r"(?:[2-9]|[2-9]/[2-9]|1[0-2]/[2-9])", value):
        return [f"invalid_time_signature:{value}"]
    return []


def _validate_duration_payload(payload: dict[str, Any]) -> list[str]:
    value = (payload or {}).get("duration")
    if value in (None, ""):
        return ["missing_duration"]
    try:
        if isinstance(value, (int, float)):
            seconds = float(value)
        else:
            text = str(value)
            if ":" in text or re.search(r"\b(?:m|min|minutes?|s|sec|seconds?)\b", text, re.I):
                seconds = float(parse_duration_seconds(text, -1))
            else:
                seconds = float(re.search(r"-?\d+(?:\.\d+)?", text).group(0))
    except Exception:
        return [f"invalid_duration:{value}"]
    if seconds < 10 or seconds > 600:
        return [f"duration_out_of_range:{value}"]
    return []


def _validate_section_map_payload(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    raw_tags = _raw_director_section_tags(payload if isinstance(payload, dict) else {})
    if not raw_tags:
        issues.append("missing_section_map")
        return issues
    keys = [_section_key_for_director(tag) for tag in raw_tags]
    duplicate_tags = sorted({tag for tag in raw_tags if keys.count(_section_key_for_director(tag)) > 1})
    if duplicate_tags:
        issues.append("duplicate_section_tags:" + ",".join(duplicate_tags))
    if not all(re.fullmatch(r"\[[^\[\]]+\]", str(tag or "").strip()) for tag in raw_tags):
        issues.append("section_tags_must_be_bracketed")
    if not any(re.search(r"chorus|hook|refrain", str(tag), re.I) for tag in raw_tags):
        issues.append("section_map_missing_hook")
    return issues


# Pre-compiled cliche-phrase regex used by hook + lyric validators.
# Substring matches force the agent to rewrite with concrete imagery instead
# of falling back to the generic AI image bank. Keep this list in sync with
# LYRIC_ANTI_PATTERNS["cliche_image_bank"] in prompt_kit.py.
_LYRIC_CLICHE_PHRASES = (
    "neon dreams", "fire inside", "shattered dreams", "endless night",
    "empty streets", "embers", "whispers in the dark", "silhouettes",
    "echoes of", "we rise", "let it burn", "chasing the night",
    "broken heart", "rising from the ashes", "stars aligned",
    "fade away", "into the void", "burning bright", "stolen kisses",
    "tears like rain", "frozen in time", "dancing in the dark",
    "running through my mind",
)
_LYRIC_TELLING_LABELS = (
    "i feel sad", "my heart is broken", "i'm in pain", "we're all in pain",
    "this is sad", "this is hard", "we suffer", "i'm hurting inside",
)
_LYRIC_GENERIC_POV = (
    "we all", "everyone feels", "the world is", "the people need",
    "society today", "this generation", "the youth of today",
)


def _scan_for_cliche_phrases(text: str) -> list[str]:
    """Return the cliche / telling-not-showing / generic-POV phrases that
    appear in the given text. Match is case-insensitive substring; the LLM
    sees the hits in the validator response and is expected to rewrite."""
    if not text:
        return []
    lowered = str(text).lower()
    hits: list[str] = []
    for phrase in _LYRIC_CLICHE_PHRASES:
        if phrase in lowered:
            hits.append(f"cliche:{phrase}")
    for phrase in _LYRIC_TELLING_LABELS:
        if phrase in lowered:
            hits.append(f"telling:{phrase}")
    for phrase in _LYRIC_GENERIC_POV:
        # Generic POV needs word-boundary so 'we all' doesn't match 'wear all-stars'
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            hits.append(f"generic_pov:{phrase}")
    return hits


def _validate_hook_payload(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    hook_lines = payload.get("hook_lines")
    if not isinstance(hook_lines, list):
        return ["hook_lines_must_be_array"]
    tagged = [str(line) for line in hook_lines if re.search(r"\[[^\]]+\]", str(line or ""))]
    if tagged:
        issues.append("hook_lines_must_not_contain_section_tags")
    non_empty = [str(line or "").strip() for line in hook_lines if str(line or "").strip()]
    if not non_empty:
        issues.append("hook_lines_empty")
    # Hook lines need actual content — under 8 chars is almost always padding
    # ("yeah", "uh", "ok"). The hum-test fails on filler lines.
    too_short = [line for line in non_empty if len(line) < 8]
    if too_short:
        issues.append(f"hook_lines_too_short:{len(too_short)}_lines_under_8_chars")
    # Hook promise should explain the song's emotional thesis in one line.
    hook_promise = str((payload or {}).get("hook_promise") or "").strip()
    if hook_promise and len(hook_promise) < 20:
        issues.append(f"hook_promise_too_short:{len(hook_promise)}_chars_min_20")
    # Cliche image bank check on hook text (the most visible part of the song).
    hook_text = " ".join(non_empty)
    cliches = _scan_for_cliche_phrases(hook_text)
    if cliches:
        issues.append("hook_contains_cliche_phrases:" + ",".join(cliches[:3]))
    return issues


def _validate_caption_payload(payload: dict[str, Any], track: dict[str, Any] | None = None) -> list[str]:
    issues: list[str] = []
    caption = str((payload or {}).get("caption") or "")
    if not caption.strip():
        issues.append("missing_caption")
    if len(caption) > 512:
        issues.append(f"caption_over_512:{len(caption)}")
    issues.extend(_caption_forbidden_markers(caption, track))
    if re.search(r"\[[^\]]+\]", caption):
        issues.append("section_tag_in_caption")
    # Caption six-dimension coverage check: caption that names "drums" without
    # specifying kick/snare/hat triad, or "sample" without source-genre, fails
    # the production-grade rule from ACE_STEP_AUTHORING_RULES.
    lowered_caption = caption.lower()
    if re.search(r"\bsample\b", lowered_caption) and not re.search(
        r"\b(?:soul sample|jazz sample|gospel chop|funk sample|film score|"
        r"chopped|replayed|sample chops|sample loop|sample chop)\b",
        lowered_caption,
    ):
        issues.append("caption_bare_sample_token_missing_source_genre")
    return sorted(set(issues))


def _validate_performance_payload(payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return ["performance_payload_not_object"]
    allowed = {"performance_brief", "negative_control", "genre_profile"}
    unexpected = sorted(str(key) for key in payload.keys() if str(key) not in allowed)
    issues: list[str] = []
    if unexpected:
        issues.append("unexpected_performance_keys:" + ",".join(unexpected))
    brief = str(payload.get("performance_brief") or "").strip()
    if not brief:
        issues.append("missing_performance_brief")
    elif len(brief) < 50:
        # A real vocal performance brief covers persona + cadence + ad-lib +
        # mix notes. Under 50 chars it is almost always "soulful, confident"
        # filler that tells the vocalist nothing.
        issues.append(f"performance_brief_too_short:{len(brief)}_chars_min_50")
    return issues


def _director_sound_tag_candidates(track: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    fallback_tags = [
            "hip-hop drums",
            "deep low end",
            "clear lead rap vocal",
            "tight rhythmic pocket",
            "cinematic strings",
            "brass swells",
            "dark tense atmosphere",
            "triumphant energy",
            "polished modern mix",
    ]
    if build_genre_intent_contract(track).get("family") == "rap":
        sources = (
            _baseline_caption_tags(track, ""),
            fallback_tags,
            track.get("tag_list"),
            track.get("tags"),
            track.get("caption"),
        )
    else:
        sources = (
            track.get("tag_list"),
            track.get("tags"),
            track.get("caption"),
            _baseline_caption_tags(track, ""),
            fallback_tags,
        )
    for source in sources:
        for item in _agent_tag_items(source):
            tag = re.sub(r"\s+", " ", str(item or "").strip(" ,.;:"))
            if not tag or tag in candidates:
                continue
            if len(tag) > 70:
                continue
            if re.search(r"\[[^\]]+\]", tag):
                continue
            if _caption_forbidden_markers(tag, track):
                continue
            if re.search(r"\blyrics?\b|\bstory\b|\bnarrative\b", tag, re.I):
                continue
            candidates.append(tag)
    return candidates


def _caption_fallback_payload(track: dict[str, Any]) -> dict[str, Any]:
    tags = _director_sound_tag_candidates(track)
    caption = ", ".join(tags[:14]).strip()
    while len(caption) > 508 and tags:
        tags = tags[:-1]
        caption = ", ".join(tags).strip()
    if not caption:
        caption = "hip-hop drums, deep low end, clear lead vocal, tight rhythmic pocket, polished modern mix"
    return {
        "caption": caption,
        "caption_dimensions_covered": ACE_STEP_CAPTION_DIMENSIONS,
    }


def _performance_fallback_payload(track: dict[str, Any]) -> dict[str, Any]:
    tags = _director_sound_tag_candidates(track)
    profile = ", ".join(tags[:10]) or "hip-hop drums, deep low end, clear lead vocal, polished modern mix"
    return {
        "performance_brief": (
            "Keep the lead vocal forward and intelligible, lock the delivery to the groove, "
            "and balance the low end with the melodic and cinematic layers."
        ),
        "negative_control": (
            "Avoid muddy bass, clipped vocals, excessive reverb, random syllables, prompt text, "
            "and effects that hide lyric articulation."
        ),
        "genre_profile": _clip_text(profile, 220),
    }


def _director_minimal_validate(track: dict[str, Any], section_tags: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    issues: list[str] = []
    caption = str(track.get("caption") or track.get("tags") or "")
    lyrics = str(track.get("lyrics") or "")
    title = str(track.get("title") or "").strip()
    if not title:
        issues.append("missing_title")
    if not caption.strip():
        issues.append("missing_caption")
    if len(caption) > 512:
        issues.append(f"caption_over_512:{len(caption)}")
    for issue in _caption_forbidden_markers(caption, track):
        issues.append(issue)
    if not lyrics.strip():
        issues.append("missing_lyrics")
    if len(lyrics) > ACE_STEP_LYRICS_CHAR_LIMIT:
        issues.append(f"lyrics_over_4096:{len(lyrics)}")
    present_keys = {_section_key_for_director(item) for item in re.findall(r"\[[^\]]+\]", lyrics)}
    expected_keys = {_section_key_for_director(item) for item in section_tags}
    section_markers = re.findall(r"\[[^\]]+\]", lyrics)
    section_counts: dict[str, int] = {}
    for marker in section_markers:
        key = _section_key_for_director(marker)
        section_counts[key] = section_counts.get(key, 0) + 1
    duplicate_sections = [
        marker for marker in section_tags
        if section_counts.get(_section_key_for_director(marker), 0) > 1
    ]
    if duplicate_sections:
        issues.append("duplicate_section_tags:" + ",".join(duplicate_sections))
    missing_sections = [tag for tag in section_tags if _section_key_for_director(tag) not in present_keys]
    if missing_sections:
        issues.append("section_map_mismatch:" + ",".join(missing_sections))
    if not any(re.search(r"chorus|hook|refrain", str(tag), re.I) for tag in section_tags):
        issues.append("section_map_missing_hook")
    if not any(re.search(r"chorus|hook|refrain", section, re.I) for section in re.findall(r"\[[^\]]+\]", lyrics)):
        issues.append("lyrics_missing_hook_section")
    lyric_duration_fit = _director_lyric_duration_fit(track, options)
    issues.extend(lyric_duration_fit.get("issues") or [])
    lyrics_quality = _director_lyrics_quality(track, options, {"status": "fail" if issues else "pass", "issues": issues})
    short_role = bool(lyrics_quality.get("is_short_role"))
    full_song = not short_role and float(lyrics_quality.get("duration") or 0) >= 150
    rap_context = bool(lyrics_quality.get("is_rap"))
    if full_song:
        verse_sections = [
            marker for marker in section_markers
            if re.search(r"\bverse\b", str(marker), re.I)
        ]
        bridge_sections = [
            marker for marker in section_markers
            if re.search(r"\bbridge\b", str(marker), re.I)
        ]
        hook_sections = [
            marker for marker in section_markers
            if re.search(r"chorus|hook|refrain", str(marker), re.I)
        ]
        if len(verse_sections) < 2:
            issues.append(f"weak_section_map:verses_{len(verse_sections)}/2")
        if not bridge_sections:
            issues.append("weak_section_map:missing_bridge")
        if len(hook_sections) < 2:
            issues.append(f"missing_hook:hook_passes_{len(hook_sections)}/2")
        if rap_context:
            rap_bar_counts = dict(lyrics_quality.get("rap_bar_counts") or {})
            if len(rap_bar_counts) < 2:
                issues.append(f"weak_section_map:rap_verses_{len(rap_bar_counts)}/2")
            for section, count in rap_bar_counts.items():
                if int(count or 0) < 16:
                    issues.append(f"rap_verses_underfilled:{section}={count}/16")
    density_plan = ((lyric_duration_fit.get("lyric_density_gate") or {}).get("plan") or {}) if isinstance(lyric_duration_fit.get("lyric_density_gate"), dict) else {}
    instrumental = str(lyrics or "").strip().lower() == "[instrumental]" or bool(track.get("instrumental"))
    lyric_craft = lyric_craft_gate(
        lyrics,
        track,
        options=options,
        plan=density_plan,
        duration=parse_duration_seconds(track.get("duration") or (options or {}).get("track_duration") or 180, 180),
        genre_hint=_director_track_genre_hint(track, options),
        instrumental=instrumental,
    )
    if lyric_craft.get("status") != "pass":
        for issue in lyric_craft.get("issues") or []:
            issue_id = str(issue.get("id") if isinstance(issue, dict) else issue)
            detail = str(issue.get("detail") if isinstance(issue, dict) else "").strip()
            issues.append(f"{issue_id}:{detail}" if detail else issue_id)
    genre_adherence = evaluate_genre_adherence(track, options)
    issues.extend(str(issue.get("id") or issue) for issue in (genre_adherence.get("issues") or []))
    producer_ready = producer_grade_readiness(track, options=options)
    for issue in producer_ready.get("issues") or []:
        issue_id = str(issue.get("id") if isinstance(issue, dict) else issue)
        detail = str(issue.get("detail") if isinstance(issue, dict) else "").strip()
        issues.append(f"{issue_id}:{detail}" if detail else issue_id)
    lyrics_quality = _director_lyrics_quality(track, options, {"status": "fail" if issues else "pass", "issues": issues})
    return {
        "version": ACEJAM_ALBUM_DIRECTOR_VERSION,
        "gate_passed": not issues,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "caption_chars": len(caption),
        "lyrics_chars": len(lyrics),
        "lyrics_word_count": int(lyric_duration_fit.get("word_count") or 0),
        "lyrics_line_count": int(lyric_duration_fit.get("line_count") or 0),
        "lyrics_quality": lyrics_quality,
        "lyric_duration_fit": lyric_duration_fit,
        "lyric_density_gate": lyric_duration_fit.get("lyric_density_gate") or {},
        "lyrical_craft_contract": lyric_craft.get("contract") or {},
        "lyric_craft_gate": lyric_craft,
        "lyric_craft_score": lyric_craft.get("score"),
        "lyric_craft_issues": lyric_craft.get("issue_ids") or [],
        "genre_intent_contract": genre_adherence.get("contract") or {},
        "genre_adherence": {key: value for key, value in genre_adherence.items() if key != "contract"},
        "producer_grade_sonic_contract": (producer_ready.get("sonic_dna_coverage") or {}).get("contract") or {},
        "sonic_dna_coverage": producer_ready.get("sonic_dna_coverage") or {},
        "producer_grade_readiness": producer_ready,
        "section_tags": section_tags,
    }


def _director_album_context(album_bible: dict[str, Any]) -> dict[str, Any]:
    intake = album_bible.get("intake") if isinstance(album_bible.get("intake"), dict) else {}
    return {
        "album_title": album_bible.get("album_title") or intake.get("album_title") or "",
        "one_sentence_concept": intake.get("one_sentence_concept") or "",
        "style_guardrails": intake.get("style_guardrails") or [],
        "track_roles": intake.get("track_roles") or [],
        "language": album_bible.get("language") or "",
        "genre_prompt": album_bible.get("genre_prompt") or "",
        "mood_vibe": album_bible.get("mood_vibe") or "",
        "vocal_type": album_bible.get("vocal_type") or "",
        "audience_platform": album_bible.get("audience_platform") or "",
    }


def _director_track_context(
    data: dict[str, Any],
    *,
    include_lyrics: bool = False,
    include_lyric_constraints: bool = False,
    include_producer: bool = False,
    fields: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    allowed = set(fields or {
        "track_number",
        "locked_title",
        "source_title",
        "title",
        "description",
        "style",
        "vibe",
        "narrative",
        "duration",
        "bpm",
        "key_scale",
        "time_signature",
        "language",
        "vocal_language",
        "tag_list",
        "tags",
        "caption",
        "caption_dimensions_covered",
        "section_map",
        "hook",
        "hook_promise",
        "performance_brief",
        "negative_control",
        "genre_profile",
    })
    if include_lyric_constraints:
        allowed.update({"required_phrases", "required_lyrics"})
    if include_lyrics:
        allowed.update({"lyrics_lines", "lyrics"})
    if include_producer:
        allowed.update({"producer_credit", "engineer_credit"})
    result: dict[str, Any] = {}
    for key in allowed:
        value = data.get(key)
        if value not in (None, "", []):
            result[key] = _clip_context_value(key, value)
    return result


class AceJamAlbumDirector:
    def __init__(
        self,
        *,
        concept: str,
        num_tracks: int,
        track_duration: float,
        planner_model: str,
        language: str,
        opts: dict[str, Any],
        planner_provider: str,
        embedding_provider: str,
        embedding_model: str,
        logs: list[str],
        contract: dict[str, Any],
        model_info: dict[str, Any],
        repair_lines_before: int,
    ) -> None:
        self.concept = concept
        self.num_tracks = max(1, int(num_tracks or 1))
        self.track_duration = float(parse_duration_seconds(track_duration, 180))
        self.planner_model = str(planner_model or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL).strip()
        self.language = language or "en"
        self.opts = opts
        self.planner_provider = normalize_provider(planner_provider or "ollama")
        self.embedding_provider = normalize_provider(embedding_provider or "ollama")
        self.embedding_model = str(embedding_model or DEFAULT_ALBUM_EMBEDDING_MODEL).strip()
        self.logs = logs
        self.contract = contract or {}
        self.model_info = model_info or {}
        self.repair_lines_before = repair_lines_before
        self.prompt_library = AlbumAgentPromptLibrary(opts, self.language)
        self.agent_rounds: list[dict[str, Any]] = []
        self.agent_repair_count = 0
        self.agent_debug_dir = str(opts.get("album_debug_dir") or "")
        self.planner_llm_settings = planner_llm_settings_from_payload(opts)
        self.agent_runtime = normalize_album_agent_engine(opts.get("agent_engine"))
        self.crewai_used = self.agent_runtime == CREWAI_MICRO_AGENT_ENGINE
        self.agent_runtime_label = album_agent_engine_label(self.agent_runtime)

    def run(self) -> dict[str, Any]:
        if not self.concept.strip():
            raise AceJamAgentError("Album concept is empty. Provide concept, user_prompt, album_title, or track hints before planning.")
        self.logs.append(f"Planning Engine: {self.agent_runtime_label} ({self.agent_runtime}).")
        self.logs.append(f"Agent engine: {self.agent_runtime_label} ({ACEJAM_ALBUM_DIRECTOR_VERSION}).")
        self.logs.append(
            "Album writer mode: "
            f"{self.opts.get('album_writer_mode') or ALBUM_WRITER_MODE_DEFAULT}; "
            f"max_track_repair_rounds={max(0, min(3, int(self.opts.get('max_track_repair_rounds') or ALBUM_TRACK_GATE_REPAIR_RETRIES)))}."
        )
        if self.agent_debug_dir:
            self.logs.append(f"Agent debug log dir: {self.agent_debug_dir}")
            self.logs.append(f"Agent raw prompts JSONL: {Path(self.agent_debug_dir) / '03_agent_prompts.jsonl'}")
            self.logs.append(f"Agent raw responses JSONL: {Path(self.agent_debug_dir) / '04_agent_responses.jsonl'}")
            self.logs.append(f"Agent track state JSONL: {Path(self.agent_debug_dir) / '05_track_state.jsonl'}")
            self.logs.append(f"Agent gate reports JSONL: {Path(self.agent_debug_dir) / '06_gate_reports.jsonl'}")
            self.logs.append(f"Agent final payloads JSONL: {Path(self.agent_debug_dir) / '07_final_payloads.jsonl'}")
        preflight = preflight_album_agent_llm(self.planner_provider, self.planner_model, self.planner_llm_settings)
        self.logs.append(f"{provider_label(self.planner_provider)} preflight: planner chat={preflight.get('chat_ok')}.")
        for warning in preflight.get("warnings") or []:
            self.logs.append(f"Local LLM preflight warning: {warning}")
        if not preflight.get("chat_ok"):
            raise AceJamAgentError("; ".join(preflight.get("errors") or ["planner preflight failed"]))
        _write_album_debug_json(
            self.opts,
            "02_contract.json",
            {
                "planning_engine": self.agent_runtime,
                "album_writer_mode": self.opts.get("album_writer_mode") or ALBUM_WRITER_MODE_DEFAULT,
                "agent_runtime": self.agent_runtime,
                "crewai_used": self.crewai_used,
                "director_version": ACEJAM_ALBUM_DIRECTOR_VERSION,
                "concept": self.concept,
                "user_album_contract": self.contract,
                "planner_llm_settings": self.planner_llm_settings,
                "input_contract_applied": bool(self.contract.get("applied")),
                "editable_plan_tracks_as_hints": self.opts.get("editable_plan_tracks") or [],
            },
        )
        intake = self._call(
            "Album Intake Agent",
            self._album_intake_prompt(),
            "album_intake_payload",
            max_retries=1,
        )
        album_title = str(intake.get("album_title") or self.contract.get("album_title") or self.opts.get("album_title") or "AceJAM Album").strip()
        album_bible = {
            "album_title": album_title,
            "concept": intake.get("one_sentence_concept") or self.concept,
            "language": self.language,
            "genre_prompt": self.opts.get("album_agent_genre_prompt") or self.opts.get("genre_prompt") or "",
            "mood_vibe": self.opts.get("album_agent_mood_vibe") or "",
            "vocal_type": self.opts.get("album_agent_vocal_type") or "",
            "audience_platform": self.opts.get("album_agent_audience") or "",
            "intake": intake,
        }
        tracks: list[dict[str, Any]] = []
        previous_summaries: list[dict[str, Any]] = []
        planning_failures: list[dict[str, Any]] = []
        for index in range(self.num_tracks):
            try:
                track = self._write_track(index, album_bible, previous_summaries)
                track["planning_status"] = "completed"
                tracks.append(track)
                previous_summaries.append(_track_summary_for_agent(track))
            except Exception as exc:
                failed_track = self._failed_track_payload(index, exc)
                tracks.append(failed_track)
                planning_failures.append(
                    {
                        "track_number": failed_track.get("track_number"),
                        "title": failed_track.get("title"),
                        "error": failed_track.get("planning_error"),
                        "debug_paths": failed_track.get("debug_paths") or {},
                    }
                )
                self.logs.append(
                    "Track planning failed but album continues: "
                    f"track {index + 1} {failed_track.get('title')}: {_monitor_preview(exc, 280)}"
                )
                _append_album_debug_jsonl(
                    self.opts,
                    "07_rejected_payloads.jsonl",
                    {
                        "track_number": index + 1,
                        "title": failed_track.get("title"),
                        "planning_status": "failed",
                        "planning_error": str(exc),
                        "debug_paths": failed_track.get("debug_paths") or {},
                    },
                )
        completed_tracks = [track for track in tracks if track.get("planning_status") != "failed"]
        if not completed_tracks:
            error = (
                f"Album planning failed: 0/{self.num_tracks} tracks produced. "
                f"Failed tracks: {json.dumps(planning_failures, ensure_ascii=False)}"
            )
            _write_album_debug_json(
                self.opts,
                "08_sequence_report.json",
                {
                    "version": "acejam-sequence-critic-2026-04-30",
                    "gate_passed": False,
                    "status": "fail",
                    "issues": [{"id": "all_tracks_failed_planning", "detail": error}],
                    "planning_failures": planning_failures,
                    "track_count": 0,
                    "expected_track_count": self.num_tracks,
                },
            )
            contract_repairs = len([line for line in self.logs if str(line).startswith("Contract repaired:")]) - self.repair_lines_before
            return {
                "tracks": tracks,
                "logs": self.logs,
                "success": False,
                "error": str(planning_failures[0].get("error") if planning_failures else error),
                "planning_engine": self.agent_runtime,
                "album_writer_mode": self.opts.get("album_writer_mode") or ALBUM_WRITER_MODE_DEFAULT,
                "custom_agents_used": True,
                "crewai_used": self.crewai_used,
                "toolbelt_fallback": False,
                "crewai_output_log_file": "",
                "agent_debug_dir": self.agent_debug_dir,
                "agent_rounds": self.agent_rounds,
                "agent_repair_count": self.agent_repair_count,
                "memory_enabled": False,
                "context_chunks": 0,
                "retrieval_rounds": 0,
                "planning_status": "failed",
                "planning_failed_count": len(planning_failures),
                "planning_failures": planning_failures,
                "failed_tracks": planning_failures,
                "prompt_kit_version": PROMPT_KIT_VERSION,
                "prompt_kit": prompt_kit_payload(),
                "toolkit": toolkit_payload(self.opts.get("installed_models")),
                "input_contract": contract_prompt_context(self.contract),
                "input_contract_applied": bool(self.contract.get("applied")),
                "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
                "blocked_unsafe_count": int(self.contract.get("blocked_unsafe_count") or 0),
                "contract_repair_count": max(0, contract_repairs),
            }
        sequence_report = _album_sequence_report(completed_tracks, self.contract, len(completed_tracks))
        if planning_failures:
            sequence_report["planning_failures"] = planning_failures
            sequence_report["planning_failed_count"] = len(planning_failures)
            sequence_report["partial_album"] = True
        _write_album_debug_json(self.opts, "08_sequence_report.json", sequence_report)
        if not sequence_report.get("gate_passed") and not planning_failures:
            reasons = "; ".join(f"{item.get('id')}: {item.get('detail')}" for item in (sequence_report.get("issues") or [])[:8])
            raise AceJamAgentError(f"Album sequence critic failed: {reasons or 'sequence gate failed'}")
        if not sequence_report.get("gate_passed") and planning_failures:
            reasons = "; ".join(f"{item.get('id')}: {item.get('detail')}" for item in (sequence_report.get("issues") or [])[:8])
            self.logs.append(f"Album sequence critic warning kept partial album: {reasons or 'sequence gate failed'}")
        _write_album_debug_json(
            self.opts,
            "debug_index.json",
            {
                "version": ACEJAM_ALBUM_DIRECTOR_VERSION,
                "planning_engine": self.agent_runtime,
                "album_writer_mode": self.opts.get("album_writer_mode") or ALBUM_WRITER_MODE_DEFAULT,
                "agent_runtime": self.agent_runtime,
                "agent_debug_dir": self.agent_debug_dir,
                "files": {
                    "prompts": str(Path(self.agent_debug_dir) / "03_agent_prompts.jsonl") if self.agent_debug_dir else "",
                    "responses": str(Path(self.agent_debug_dir) / "04_agent_responses.jsonl") if self.agent_debug_dir else "",
                    "track_state": str(Path(self.agent_debug_dir) / "05_track_state.jsonl") if self.agent_debug_dir else "",
                    "gate_reports": str(Path(self.agent_debug_dir) / "06_gate_reports.jsonl") if self.agent_debug_dir else "",
                    "final_payloads": str(Path(self.agent_debug_dir) / "07_final_payloads.jsonl") if self.agent_debug_dir else "",
                },
            },
        )
        output_label = "AceJAM Director" if self.agent_runtime == ACEJAM_AGENT_ENGINE else self.agent_runtime_label
        self.logs.append(
            f"{output_label} produced {len(completed_tracks)} direct ACE-Step track payload(s) "
            f"({len(completed_tracks)}/{self.num_tracks} planned)."
        )
        if planning_failures:
            failed_preview = ", ".join(
                f"{item.get('track_number')} {item.get('title')}" for item in planning_failures[:8]
            )
            self.logs.append(f"AceJAM Director partial plan: failed tracks: {failed_preview}.")
        contract_repairs = len([line for line in self.logs if str(line).startswith("Contract repaired:")]) - self.repair_lines_before
        return {
            "tracks": tracks,
            "logs": self.logs,
            "success": True,
            "planning_status": "partial" if planning_failures else "completed",
            "planning_failed_count": len(planning_failures),
            "planning_failures": planning_failures,
            "failed_tracks": planning_failures,
            "planning_engine": self.agent_runtime,
            "album_writer_mode": self.opts.get("album_writer_mode") or ALBUM_WRITER_MODE_DEFAULT,
            "custom_agents_used": True,
            "crewai_used": self.crewai_used,
            "toolbelt_fallback": False,
            "crewai_output_log_file": "",
            "agent_debug_dir": self.agent_debug_dir,
            "agent_rounds": self.agent_rounds,
            "agent_repair_count": self.agent_repair_count,
            "memory_enabled": False,
            "context_chunks": 0,
            "retrieval_rounds": 0,
            "sequence_report": sequence_report,
            "album_title": album_title,
            "album_bible": album_bible,
            "concept": album_bible.get("concept") or self.concept,
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "prompt_kit": prompt_kit_payload(),
            "toolkit": toolkit_payload(self.opts.get("installed_models")),
            "input_contract": contract_prompt_context(self.contract),
            "input_contract_applied": bool(self.contract.get("applied")),
            "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
            "blocked_unsafe_count": int(self.contract.get("blocked_unsafe_count") or 0),
            "contract_repair_count": max(0, contract_repairs),
            "toolkit_report": {
                "director_version": ACEJAM_ALBUM_DIRECTOR_VERSION,
                "model_advice": self.model_info,
                "album_bible": album_bible,
                "user_album_contract": self.contract,
                "custom_agents": {
                    "enabled": True,
                    "agent_runtime": self.agent_runtime,
                    "planner_model": self.planner_model,
                    "planner_provider": self.planner_provider,
                    "agent_debug_dir": self.agent_debug_dir,
                    "agent_repair_count": self.agent_repair_count,
                },
            },
        }

    def _call(self, agent_name: str, user_prompt: str, schema_name: str, max_retries: int | None = 0) -> dict[str, Any]:
        call_fn = _crewai_micro_block_call if self.crewai_used else _agent_json_call
        payload = call_fn(
            agent_name=agent_name,
            provider=self.planner_provider,
            model_name=self.planner_model,
            user_prompt=user_prompt,
            logs=self.logs,
            debug_options=self.opts,
            schema_name=schema_name,
            extra_system=self.prompt_library.system_rules(),
            max_retries=max_retries,
        )
        self.agent_rounds.append({"agent": agent_name, "status": "completed", "agent_runtime": self.agent_runtime})
        return payload

    def _call_until_valid(
        self,
        agent_name: str,
        user_prompt: str,
        schema_name: str,
        validator: Callable[[dict[str, Any]], list[str]],
        *,
        repair_context: str = "",
        max_repair_retries: int | None = None,
        json_max_retries: int | None = 1,
    ) -> dict[str, Any]:
        max_repairs = max(0, ACEJAM_AGENT_GATE_REPAIR_RETRIES if max_repair_retries is None else int(max_repair_retries))
        prompt = user_prompt
        last_issues: list[str] = []
        rejected_payload: dict[str, Any] = {}
        for repair_attempt in range(0, max_repairs + 1):
            try:
                payload = self._call(agent_name, prompt, schema_name, max_retries=json_max_retries)
            except AceJamAgentError as exc:
                last_issues = [f"block_parse_failed:{_monitor_preview(exc, 260)}"]
                if repair_attempt >= max_repairs:
                    break
                next_attempt = repair_attempt + 1
                self.agent_repair_count += 1
                issue_text = "; ".join(last_issues)
                self.logs.append(
                    f"Agent block validation retry: {agent_name} attempt {next_attempt}/{max_repairs}: {issue_text}"
                )
                _append_album_debug_jsonl(
                    self.opts,
                    "04_agent_responses.jsonl",
                    {
                        "agent": agent_name,
                        "agent_runtime": self.agent_runtime,
                        "block_validation": "failed",
                        "repair_attempt": next_attempt,
                        "max_repair_retries": max_repairs,
                        "validation_issues": last_issues,
                        "rejected_payload_preview": "",
                    },
                )
                prompt = (
                    "BLOCK REPAIR REQUIRED.\n"
                    "Your previous response was not valid delimiter-block output.\n"
                    f"VALIDATOR_ISSUES_EXACT:\n{json.dumps(last_issues, ensure_ascii=False)}\n\n"
                    "EXPECTED_BLOCK_SHAPE:\n"
                    f"{_agent_block_template(schema_name)}\n\n"
                    f"{repair_context.strip()}\n\n"
                    "Do not include thoughts, refusal text, markdown, analysis, JSON, or any text outside the blocks. "
                    "Return one corrected delimiter-block response only.\n\n"
                    "ORIGINAL_TASK:\n"
                    f"{user_prompt}"
                )
                continue
            issues = sorted(set(str(issue) for issue in (validator(payload) or []) if str(issue)))
            if not issues:
                return payload
            last_issues = issues
            rejected_payload = payload
            if repair_attempt >= max_repairs:
                break
            next_attempt = repair_attempt + 1
            self.agent_repair_count += 1
            issue_text = "; ".join(issues)
            self.logs.append(
                f"Agent semantic validation retry: {agent_name} attempt {next_attempt}/{max_repairs}: {issue_text}"
            )
            _append_album_debug_jsonl(
                self.opts,
                "04_agent_responses.jsonl",
                {
                    "agent": agent_name,
                    "agent_runtime": self.agent_runtime,
                    "semantic_validation": "failed",
                    "repair_attempt": next_attempt,
                    "max_repair_retries": max_repairs,
                    "validation_issues": issues,
                    "rejected_payload_preview": _monitor_preview(_compact_json(payload), 700),
                },
            )
            prompt = (
                "SEMANTIC REPAIR REQUIRED.\n"
                "Your previous delimiter-block response parsed correctly but failed AceJAM's content validator.\n"
                f"VALIDATOR_ISSUES_EXACT:\n{json.dumps(issues, ensure_ascii=False)}\n\n"
                "EXPECTED_BLOCK_SHAPE:\n"
                f"{_agent_block_template(schema_name)}\n\n"
                f"{repair_context.strip()}\n\n"
                "Do not include previous invalid sections/lines. Do not copy rejected content. "
                "Return corrected delimiter blocks only, no JSON, no prose, no markdown, no extra blocks.\n\n"
                "ORIGINAL_TASK:\n"
                f"{user_prompt}"
            )
        _append_album_debug_jsonl(
            self.opts,
            "04_agent_responses.jsonl",
            {
                "agent": agent_name,
                "agent_runtime": self.agent_runtime,
                "semantic_validation": "exhausted",
                "max_repair_retries": max_repairs,
                "validation_issues": last_issues,
                "rejected_payload_preview": _monitor_preview(_compact_json(rejected_payload), 700),
            },
        )
        debug_path = str(Path(self.agent_debug_dir) / "04_agent_responses.jsonl") if self.agent_debug_dir else ""
        suffix = f" Debug JSONL: {debug_path}" if debug_path else ""
        raise AceJamAgentError(
            f"{agent_name} failed semantic validation after {max_repairs} repair attempt(s): "
            f"{'; '.join(last_issues) or 'unknown validation failure'}.{suffix}"
        )

    def _validation_retry_call(
        self,
        agent_name: str,
        user_prompt: str,
        schema_name: str,
        issues: list[str],
    ) -> dict[str, Any]:
        prompt = (
            "Your previous delimiter-block response had this validation failure and cannot be rendered by ACE-Step:\n"
            f"{'; '.join(issues)}\n\n"
            "ANSWER AGAIN EXACTLY AS DELIMITER BLOCKS, with no JSON, extra blocks, or prose:\n"
            f"{_agent_block_template(schema_name)}\n\n"
            "Original compact task follows:\n"
            f"{user_prompt}"
        )
        return self._call(agent_name, prompt, schema_name, max_retries=0)

    def _debug_paths(self) -> dict[str, str]:
        if not self.agent_debug_dir:
            return {}
        root = Path(self.agent_debug_dir)
        return {
            "prompts": str(root / "03_agent_prompts.jsonl"),
            "responses": str(root / "04_agent_responses.jsonl"),
            "track_state": str(root / "05_track_state.jsonl"),
            "gate_reports": str(root / "06_gate_reports.jsonl"),
            "final_payloads": str(root / "07_final_payloads.jsonl"),
            "rejected_payloads": str(root / "07_rejected_payloads.jsonl"),
        }

    def _failed_track_payload(self, index: int, exc: Exception) -> dict[str, Any]:
        """Build a stub track payload when crew planning fails. The wizard
        still receives editable fields filled with album-bible / opts values
        instead of raw blanks, so the user can see what failed and finish
        the track manually rather than staring at a half-empty form."""
        slot = self._locked_track_slot(index)
        title = str(slot.get("title") or slot.get("locked_title") or f"Track {index + 1}").strip()
        error = str(exc)
        # Pull album-level direction from opts so the wizard's mood / genre /
        # vocal_type cells are never None on a failed track. The user can
        # overwrite them, but blanks make the failure look worse than it is.
        opts = self.opts or {}
        bible_obj = getattr(self, "album_bible", None)
        bible_mood = ""
        bible_concept = ""
        bible_genre_hint = ""
        if isinstance(bible_obj, dict):
            bible_mood = str(bible_obj.get("mood") or bible_obj.get("mood_vibe") or "").strip()
            bible_concept = str(bible_obj.get("concept") or bible_obj.get("one_sentence_concept") or "").strip()
            bible_genre_hint = str(bible_obj.get("genre_prompt") or bible_obj.get("genre_hint") or "").strip()
        producer_credit = str(slot.get("producer_credit") or "").strip()
        derived_style = ", ".join(
            dict.fromkeys(bit for bit in (bible_genre_hint, producer_credit) if bit)
        )[:160]
        return {
            **{key: value for key, value in slot.items() if value not in (None, "", [])},
            "track_number": index + 1,
            "title": title,
            "duration": parse_duration_seconds(slot.get("duration") or self.track_duration, self.track_duration),
            # Wizard-visible fallback values so failed tracks still look
            # "filled" — user edits replace these freely.
            "style": str(slot.get("style") or derived_style or "").strip(),
            "vibe": str(slot.get("vibe") or bible_mood or "").strip(),
            "narrative": str(slot.get("narrative") or bible_concept or "").strip(),
            "description": str(slot.get("description") or bible_concept or "").strip(),
            "mood": str(opts.get("album_mood") or opts.get("mood_vibe") or bible_mood or "").strip(),
            "genre": str(opts.get("album_genre") or opts.get("genre_prompt") or bible_genre_hint or "").strip(),
            "vocal_type": str(opts.get("album_vocal_type") or opts.get("vocal_type") or "").strip(),
            "planning_status": "failed",
            "planning_error": error,
            "error": error,
            "generated": False,
            "skip_render": True,
            "agent_complete_payload": False,
            "payload_gate_status": "planning_failed",
            "payload_gate_passed": False,
            "payload_gate_blocking_issues": [{"id": "planning_failed", "detail": error}],
            "payload_quality_gate": {
                "status": "planning_failed",
                "gate_passed": False,
                "issues": ["planning_failed"],
                "detail": error,
            },
            "agent_debug_dir": self.agent_debug_dir,
            "debug_paths": self._debug_paths(),
            "model_results": [],
            "audios": [],
        }

    def _album_intake_prompt(self) -> str:
        return (
            "Normalize the album brief for the rest of the agents. Do not write songs.\n\n"
            f"FULL_ALBUM_CONCEPT:\n{self.concept}\n\n"
            f"REQUESTED_TRACK_COUNT: {self.num_tracks}\n"
            f"REQUESTED_TRACK_DURATION_SECONDS: {int(self.track_duration)}\n"
            f"LANGUAGE: {self.language}\n"
            f"USER_GENRE_PROMPT: {self.opts.get('album_agent_genre_prompt') or self.opts.get('genre_prompt') or ''}\n"
            f"MOOD_VIBE: {self.opts.get('album_agent_mood_vibe') or ''}\n"
            f"VOCAL_TYPE: {self.opts.get('album_agent_vocal_type') or ''}\n"
            f"AUDIENCE_PLATFORM: {self.opts.get('album_agent_audience') or ''}\n"
            f"USER_ALBUM_CONTRACT:\n{json.dumps(contract_prompt_context(self.contract), ensure_ascii=False, indent=2)}\n\n"
            "OUTPUT_BLOCKS:\n"
            f"{_agent_block_template('album_intake_payload')}\n"
            "Return delimiter blocks only."
        )

    def _locked_track_slot(self, index: int) -> dict[str, Any]:
        locked = contract_track(self.contract, index + 1, index) or {}
        hints = self.opts.get("editable_plan_tracks") if isinstance(self.opts.get("editable_plan_tracks"), list) else []
        hint = _hint_by_track_number([item for item in hints if isinstance(item, dict)], index + 1)
        slot = {**hint}
        if locked:
            slot.update({k: v for k, v in locked.items() if v not in (None, "", [])})
        title = slot.get("locked_title") or slot.get("title") or ""
        if title:
            slot["title"] = title
        slot["track_number"] = index + 1
        slot["duration"] = parse_duration_seconds(slot.get("duration") or self.track_duration, self.track_duration)
        return slot

    def _base_track_context(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any] | None = None,
        *,
        include_full_concept: bool = False,
        include_lyrics: bool = False,
        include_lyric_constraints: bool = False,
        include_producer: bool = False,
        fields: set[str] | None = None,
    ) -> str:
        concept_label = "FULL_ALBUM_CONCEPT" if include_full_concept else "ALBUM_CONCEPT_SUMMARY"
        concept_text = self.concept if include_full_concept else str((album_bible.get("intake") or {}).get("one_sentence_concept") or album_bible.get("concept") or self.concept)
        concept_text = _clip_text(concept_text, 1400 if include_full_concept else 420)
        locked = _director_track_context(
            self._locked_track_slot(index),
            include_lyrics=False,
            include_lyric_constraints=include_lyric_constraints,
            include_producer=include_producer,
            fields=fields,
        )
        current_payload = _director_track_context(
            current or {},
            include_lyrics=include_lyrics,
            include_lyric_constraints=include_lyric_constraints,
            include_producer=include_producer,
            fields=fields,
        )
        # Album-level direction the wizard collects gets surfaced to every
        # per-track agent. Legacy keys (`album_agent_*`) plus the new ones the
        # bridge passes (`album_mood`, `album_vocal_type`, `album_genre`,
        # `mood`, `mood_vibe`, `vocal_type`, `genre`, `genre_prompt`) are all
        # consulted so agents see the user's intent regardless of which
        # entry-point shaped the request.
        opts = self.opts or {}
        genre_prompt = (
            opts.get("album_agent_genre_prompt")
            or opts.get("genre_prompt")
            or opts.get("album_genre")
            or opts.get("genre")
            or ""
        )
        mood_vibe = (
            opts.get("album_agent_mood_vibe")
            or opts.get("mood_vibe")
            or opts.get("album_mood")
            or opts.get("mood")
            or ""
        )
        vocal_type = (
            opts.get("album_agent_vocal_type")
            or opts.get("album_vocal_type")
            or opts.get("vocal_type")
            or opts.get("vocal_lead")
            or ""
        )
        audience = (
            opts.get("album_agent_audience")
            or opts.get("album_audience")
            or opts.get("audience_platform")
            or ""
        )
        return (
            f"{concept_label}:\n{concept_text}\n\n"
            f"TRACK_COUNTER: you are working on track {index + 1} of {self.num_tracks}.\n"
            f"LANGUAGE: {self.language}\n"
            f"USER_GENRE_PROMPT: {genre_prompt}\n"
            f"MOOD_VIBE: {mood_vibe}\n"
            f"VOCAL_TYPE: {vocal_type}\n"
            f"AUDIENCE_PLATFORM: {audience}\n"
            f"ALBUM_BIBLE_COMPACT:\n{_compact_json(_director_album_context(album_bible))}\n"
            f"LOCKED_TRACK_FIELDS:\n{_compact_json(locked)}\n"
            f"CURRENT_TRACK_STATE:\n{_compact_json(current_payload)}\n"
        )

    def _generate_tags(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any],
        sonic_fields: set[str],
        *,
        repair_issues: list[str] | None = None,
    ) -> dict[str, Any]:
        repair_note = ""
        genre_contract = build_genre_intent_contract(current, self.opts)
        current["genre_intent_contract"] = genre_contract
        producer_contract = build_producer_grade_sonic_contract(current, self.opts)
        current["producer_grade_sonic_contract"] = producer_contract
        if repair_issues:
            repair_note = (
                "\nTAG_REPAIR_ISSUES:\n"
                f"{json.dumps(repair_issues, ensure_ascii=False)}\n"
                "Write fresh sonic tags. Do not preserve invalid tag order or non-dominant genre emphasis.\n"
            )
        producer_note = (
            "\nPRODUCER_GRADE_SONIC_CONTRACT:\n"
            f"{json.dumps(producer_contract, ensure_ascii=False)}\n"
            "tag_list must cover every required dimension with concrete sound traits; avoid generic-only tags like polished modern mix by itself.\n"
        )
        rap_note = ""
        if genre_contract.get("family") == "rap":
            rap_note = (
                "\nSTRICT_RAP_LOCK:\n"
                "The first tags must make this unmistakably rap/hip-hop, not cinematic score. "
                "Include rap/hip-hop as primary genre, rap vocal delivery, hip-hop drums or rap groove, and 808/sub-bass/low-end. "
                "Cinematic/orchestral terms are allowed only as secondary color after rap anchors.\n"
            )
        tag_payload = self._call_until_valid(
            "Tag Agent",
            self._base_track_context(index, album_bible, current, fields=sonic_fields)
            + repair_note
            + producer_note
            + rap_note
            + "\nTASK:\nChoose the track Sonic DNA before any lyric writing. Output sonic tags only. "
            "Do not include BPM, key, time signature, duration, model, seed, producer/person names, track title, lyric phrases, or story prose. "
            f"caption_dimensions_covered must use only: {json.dumps(ACE_STEP_CAPTION_DIMENSIONS)}.\n"
            + f"OUTPUT_BLOCKS:\n{_agent_block_template('tag_agent_payload')}\n",
            "tag_agent_payload",
            lambda payload: _validate_tag_payload(payload, current)
            + _director_genre_validation_issues(payload, current, self.opts, include_lyrics=False)
            + _director_producer_grade_validation_issues(payload, current, self.opts),
            repair_context=(
                "tag_list and tags must be filled with sonic traits only. No BPM, key, duration, title, producer, person names, "
                "section tags, story prose, or lyric lines. If rap is requested, rap/hip-hop must be primary and front-loaded. "
                "Cover primary_genre, drum_groove, low_end_bass, melodic_identity, vocal_delivery, arrangement_movement, texture_space, and mix_master."
            ),
        )
        current.update(tag_payload)
        current["genre_intent_contract"] = build_genre_intent_contract(current, self.opts)
        current["genre_adherence"] = evaluate_genre_adherence(current, self.opts)
        current["producer_grade_sonic_contract"] = build_producer_grade_sonic_contract(current, self.opts)
        current["producer_grade_readiness"] = producer_grade_readiness(current, options=self.opts)
        current["sonic_dna_coverage"] = (current["producer_grade_readiness"].get("sonic_dna_coverage") or {})
        _append_album_debug_jsonl(self.opts, "05_track_state.jsonl", {"track_number": index + 1, "stage": "Tag Agent", "state": current})
        return tag_payload

    def _generate_section_map(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any],
        lyric_fields: set[str],
        *,
        repair_issues: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        requested_duration = parse_duration_seconds(current.get("duration") or self.track_duration, self.track_duration)
        long_track_note = (
            "For this long vocal track, include at least one [Break] or [Instrumental Break] so the lyrics stay complete under 4096 chars."
            if requested_duration >= 180
            else "Keep sections concise and complete."
        )
        genre_contract = build_genre_intent_contract(current, self.opts)
        rap_note = ""
        if genre_contract.get("family") == "rap":
            # Rap-lock: full songs (>=120s) MUST have 3 [Verse - rap] sections
            # so the 16-bar floor delivers a real verse pool. Shorter tracks
            # use the 2-verse template. Use the canonical "Verse N - rap"
            # naming so the lyric agent recognises them as rap-modifier slots.
            if requested_duration >= 120:
                rap_note = (
                    " Rap-lock for full songs (this is a 120s+ track): use exactly THREE rap verses. "
                    "Recommended layout: [Intro], [Verse 1 - rap], [Hook], [Verse 2 - rap], [Hook], [Bridge], "
                    "[Verse 3 - rap], [Final Hook], [Outro]. Three verses are mandatory — two-verse "
                    "structures cap density and waste the album slot. Avoid pop-style repeated [Pre-Chorus] sections."
                )
            else:
                rap_note = (
                    " Rap-lock: prefer [Intro], [Verse 1 - rap], [Hook], [Verse 2 - rap], [Bridge], "
                    "[Final Hook], [Outro]. Avoid pop-style repeated [Pre-Chorus] sections."
                )
        repair_note = ""
        if repair_issues:
            repair_note = (
                "\nFINAL_GATE_REPAIR_ISSUES:\n"
                f"{json.dumps(repair_issues, ensure_ascii=False)}\n"
                "Plan a clean section_map again. Do not preserve invalid missing-hook or duplicate patterns.\n"
            )
        def _section_validator(payload: dict[str, Any]) -> list[str]:
            issues = _validate_section_map_payload(payload)
            if genre_contract.get("family") == "rap":
                tags = _raw_director_section_tags(payload if isinstance(payload, dict) else {})
                verse_tags = [tag for tag in tags if re.search(r"verse", tag, re.I)]
                if not verse_tags:
                    issues.append("rap_section_map_missing_verse")
                if sum(1 for tag in tags if re.search(r"pre[-\s]?chorus", tag, re.I)) > 1:
                    issues.append("rap_section_map_pop_prechorus_overused")
                # Full rap songs (>=120s) require 3 verses to match the
                # 3-verses template + 16-bar floor. Reject 1-2 verse layouts
                # so the repair loop coaxes a third verse in.
                if requested_duration >= 120 and len(verse_tags) < 3:
                    issues.append(f"rap_full_song_needs_3_verses:got_{len(verse_tags)}")
            return issues
        section_payload = self._call_until_valid(
            "Section Map Agent",
            self._base_track_context(index, album_bible, current, include_lyric_constraints=True, fields=lyric_fields)
            + repair_note
            + "\nCreate the exact bracketed section tags for the lyrics. Include a chorus/hook/refrain for vocal tracks. "
            + long_track_note
            + rap_note
            + "\nThe lyric part agents must use these tags exactly; do not output lyrics here.\n"
            + f"OUTPUT_BLOCKS:\n{_agent_block_template('section_map_payload')}\n",
            "section_map_payload",
            _section_validator,
            repair_context=(
                "section_map must be a non-empty list of unique bracketed section tags. "
                "It must contain at least one chorus/hook/refrain tag. No lyrics, no duplicate tags. "
                "If rap is requested, keep verse/hook structure primary and avoid repeated pop pre-chorus sections."
            ),
        )
        section_tags = _director_section_tags(section_payload)
        if not section_tags:
            raise AceJamAgentError(f"Section Map Agent returned no sections for track {index + 1}.")
        current["section_map"] = section_tags
        _append_album_debug_jsonl(self.opts, "05_track_state.jsonl", {"track_number": index + 1, "stage": "Section Map Agent", "state": current})
        return section_payload, section_tags

    def _generate_hook(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any],
        lyric_fields: set[str],
        *,
        repair_issues: list[str] | None = None,
    ) -> dict[str, Any]:
        repair_note = ""
        if repair_issues:
            repair_note = (
                "\nREPAIR_ISSUES:\n"
                f"{json.dumps(repair_issues, ensure_ascii=False)}\n"
                "Write a fresh hook. Do not include previous invalid hook lines or bracket tags.\n"
            )
        craft_contract = build_lyrical_craft_contract(current, self.opts)
        current["lyrical_craft_contract"] = craft_contract
        hook_payload = self._call_until_valid(
            "Hook Agent",
            self._base_track_context(index, album_bible, current, include_lyric_constraints=True, fields=lyric_fields)
            + repair_note
            + "\nLYRICAL_CRAFT_CONTRACT:\n"
            + f"{json.dumps(craft_contract, ensure_ascii=False)}\n"
            + "\nWrite only the hook idea and exact hook lines for chorus/hook/refrain sections. "
            "The hook must be singable/rappable and must not mention BPM, key, producer, or metadata. "
            "Make the hook title-connected and emotionally specific: one clear promise, one concrete image, no motivational poster lines, no random imagery, no filler. "
            "For sung hooks, keep vowel-friendly short phrases. For rap hooks, keep chantable cadence and clean breath length. "
            "hook_lines must be plain lyric lines only, with no bracketed section tags.\n"
            + f"OUTPUT_BLOCKS:\n{_agent_block_template('hook_payload')}\n",
            "hook_payload",
            _validate_hook_payload,
            repair_context=(
                "hook_lines must be a non-empty array of plain lyric lines only. "
                "Do not include [Chorus], [Hook], [Final Chorus], or any bracketed tag."
            ),
        )
        current.update({"hook": hook_payload, "hook_promise": hook_payload.get("hook_promise") or current.get("hook_promise") or ""})
        _append_album_debug_jsonl(self.opts, "05_track_state.jsonl", {"track_number": index + 1, "stage": "Hook Agent", "state": current})
        return hook_payload

    def _generate_lyrics_parts(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any],
        lyric_fields: set[str],
        section_tags: list[str],
        hook_payload: dict[str, Any],
        *,
        repair_issues: list[str] | None = None,
    ) -> list[str]:
        lyric_lines: list[str] = []
        current["lyrics_lines"] = []
        current["lyrics"] = ""
        groups = _director_section_groups(section_tags)
        requested_duration = parse_duration_seconds(current.get("duration") or self.track_duration, self.track_duration)
        genre_hint = _director_track_genre_hint(current, self.opts)
        lyric_plan = lyric_length_plan(
            requested_duration,
            str(current.get("lyric_density") or self.opts.get("lyric_density") or "dense"),
            str(current.get("structure_preset") or self.opts.get("structure_preset") or "auto"),
            genre_hint,
        )
        craft_contract = build_lyrical_craft_contract(current, self.opts)
        current["lyrical_craft_contract"] = craft_contract
        min_words = int(lyric_plan.get("min_words") or 0)
        min_lines = _director_effective_min_lines(int(lyric_plan.get("min_lines") or 0), min_words)
        target_words = int(lyric_plan.get("target_words") or min_words)
        target_lines = int(lyric_plan.get("target_lines") or min_lines)
        part_target_words = max(12, int((target_words + max(1, len(groups)) - 1) / max(1, len(groups))))
        part_min_lines = max(2, int((min_lines + max(1, len(groups)) - 1) / max(1, len(groups))))
        safe_lyrics_budget = min(3600, ACE_STEP_LYRICS_CHAR_LIMIT - 320)
        if repair_issues and any(str(issue).startswith("lyrics_over_4096") for issue in repair_issues):
            safe_lyrics_budget = min(3000, ACE_STEP_LYRICS_CHAR_LIMIT - 900)
        part_budget = max(420, int(safe_lyrics_budget / max(1, len(groups))))
        whole_section_minimums = _director_section_line_minimums(
            section_tags,
            duration=requested_duration,
            genre_hint=genre_hint,
        )
        if whole_section_minimums:
            current["lyric_section_minimums"] = whole_section_minimums
        current["lyric_density_plan"] = {
            "duration": requested_duration,
            "density": lyric_plan.get("density"),
            "target_words": target_words,
            "min_words": min_words,
            "target_lines": target_lines,
            "min_lines": min_lines,
            "section_minimums": whole_section_minimums,
            "max_lyrics_chars": lyric_plan.get("max_lyrics_chars") or ACE_STEP_LYRICS_CHAR_LIMIT,
        }
        forbidden_sections: list[str] = []
        repair_note = ""
        if repair_issues:
                repair_note = (
                    "\nFINAL_GATE_REPAIR_ISSUES:\n"
                    f"{json.dumps(repair_issues, ensure_ascii=False)}\n"
                    "Regenerate clean lyrics_lines from the section map. Do not include previous invalid sections/lines. "
                    "Do not copy duplicated, extra, or rejected tags from earlier attempts. "
                    "If the issue says lyrics_under_length or lyrics_too_few_lines, expand with fresh clear bars and one breath per line.\n"
                )
        for part_index, group in enumerate(groups):
            part_section_minimums = {tag: whole_section_minimums[tag] for tag in group if tag in whole_section_minimums}
            lyric_prompt = (
                self._base_track_context(index, album_bible, current, include_lyric_constraints=True, fields=lyric_fields)
                + repair_note
                + f"\nWHOLE_SONG_LYRIC_LENGTH_PLAN:\n{json.dumps({'duration': requested_duration, 'density': lyric_plan.get('density'), 'target_words': target_words, 'min_words': min_words, 'target_lines': target_lines, 'min_lines': min_lines, 'max_lyrics_chars': lyric_plan.get('max_lyrics_chars') or ACE_STEP_LYRICS_CHAR_LIMIT}, ensure_ascii=False)}\n"
                + f"LYRICAL_CRAFT_CONTRACT:\n{json.dumps(craft_contract, ensure_ascii=False)}\n"
                + f"SECTION_LINE_MINIMUMS_FOR_THIS_PART:\n{json.dumps(part_section_minimums, ensure_ascii=False)}\n"
                + f"BARS_PER_SECTION_FLOOR:\n{json.dumps(lyric_plan.get('bars_per_section') or {}, ensure_ascii=False)}\n"
                + f"\nONLY_ALLOWED_SECTION_TAGS:\n{json.dumps(group, ensure_ascii=False)}\n"
                + f"FORBIDDEN_SECTION_TAGS_ALREADY_WRITTEN:\n{json.dumps(forbidden_sections, ensure_ascii=False)}\n"
                + f"HOOK_LINES_TO_USE_IN_CHORUS_OR_HOOK:\n{json.dumps(hook_payload.get('hook_lines') or [], ensure_ascii=False)}\n"
                + f"PART_INDEX_REQUIRED: {part_index + 1}\n"
                + f"PART_TARGET_WORDS_APPROX: {part_target_words}\n"
                + f"PART_MIN_VOCAL_LINES_APPROX: {part_min_lines}\n"
                + f"PART_TARGET_CHARS_MAX: {part_budget}\n"
                + f"WHOLE_SONG_SAFE_LYRICS_TARGET_CHARS_MAX: {safe_lyrics_budget}\n"
                + "Write lyrics_lines only. sections must equal ONLY_ALLOWED_SECTION_TAGS exactly; do not add any other section tag. "
                "Each allowed section tag must appear once in lyrics_lines, in the same order. "
                "Never write a forbidden previous section. Never copy earlier sections. "
                "Respect SECTION_LINE_MINIMUMS_FOR_THIS_PART with fresh content; verses must be long enough to carry a full vocal track. "
                "RAP VERSE FLOOR: every [Verse - rap] section is minimum 16 bars (~16 lines at 8-15 syllables/line; 1 bar = 4 beats). "
                "On tracks under 120 seconds, follow the BARS_PER_SECTION_FLOOR Verse_rap value as the practical floor. "
                "Write award-level lyric craft: stack multisyllabic mosaic rhymes (Eminem-style begin/middle/end of bar) with slant-dominant flow and perfect-rhyme landings on emphasis; concrete sensory imagery per line (Nas: trap doors, rooftop snipers, lobby kids); one coherent metaphor world; pat-pattison prosody match (stable=AABB perfect, unstable=ABBA slant). "
                "Every verse moves the story forward — new scene, new POV, time jump, escalation, or revelation. A verse that just restates the chorus is dead weight. "
                "Every hook simplifies into a memorable emotional promise that passes the hum-test (a stranger should grasp the song's thesis from chorus alone). "
                "For rap or hip-hop, pack 8-15 syllables per bar (push to ~20 only on emotional spikes); pocket beats acrobatics; triplets only at high-tension moments. "
                "Ad-libs in (parens) on the same line are punctuation, not decoration — use them to mark payoff lines, not every 4 bars. "
                "For pop, R&B, rock, country, soul, latin, afro, dancehall, or sung tracks, keep vowel-friendly singable line lengths and title-connected hooks. "
                "For rap or hip-hop, never write stage directions like Instrumental break, orchestra swells, strings fade, taiko drums hit, or production notes as lyrics. "
                "For [Break] or [Instrumental Break], write the allowed section tag and one short rap ad-lib or crowd-response line only. "
                "ANTI-PATTERN GUARD: reject your own draft and rewrite if it contains any of these cliché phrases: neon dreams, fire inside, shattered dreams, endless night, empty streets, embers, whispers, silhouettes, echoes of, we rise, let it burn, chasing the night, broken heart, rising from the ashes, stars aligned, fade away, into the void, burning bright, frozen in time. Reject telling-not-showing labels ('I feel sad', 'my heart is broken'), generic POV ('we all', 'the world', 'everyone'), and explanation lines ('in other words', 'what I mean is'). "
                "Do not include BPM, key, caption, explanation, producer names, metadata, markdown, or escaped newlines.\n"
                + f"OUTPUT_BLOCKS:\n{_agent_block_template(f'lyrics_part_{part_index + 1}_payload')}\n"
            )

            def _part_validator(part_payload: dict[str, Any], *, expected_group: list[str] = group, previous_sections: list[str] = list(forbidden_sections), expected_part: int = part_index + 1) -> list[str]:
                issues = _validate_lyrics_part_payload(
                    part_payload,
                    expected_sections=expected_group,
                    forbidden_sections=previous_sections,
                    expected_part_index=expected_part,
                )
                genre_issues = _director_genre_validation_issues(
                    {"lyrics_lines": _director_payload_lines(part_payload)},
                    current,
                    self.opts,
                    include_lyrics=True,
                )
                issues.extend(
                    issue for issue in genre_issues
                    if not str(issue).startswith("non_rap_arrangement_lyric_leakage")
                )
                char_count = len("\n".join(_director_payload_lines(part_payload)))
                if char_count > part_budget + 300:
                    issues.append(f"lyrics_part_over_budget:{char_count}>{part_budget}")
                part_lines = _director_payload_lines(part_payload)
                part_craft = lyric_craft_gate(
                    "\n".join(part_lines),
                    current,
                    options=self.opts,
                    plan=lyric_plan,
                    duration=requested_duration,
                    genre_hint=genre_hint,
                    partial=True,
                )
                for craft_issue in part_craft.get("issues") or []:
                    craft_id = str(craft_issue.get("id") if isinstance(craft_issue, dict) else craft_issue)
                    if craft_id in {
                        "lyric_craft_placeholder",
                        "lyric_craft_fallback_artifact",
                        "lyric_craft_metadata_leakage",
                        "lyric_craft_generic_ai_phrase",
                        "lyric_craft_adjective_stacking",
                        "lyric_craft_line_breathability",
                        "lyric_craft_mixed_metaphor",
                    }:
                        issues.append(craft_id)
                if part_section_minimums:
                    section_counts: dict[str, int] = {}
                    active_section = ""
                    for raw_line in _director_payload_lines(part_payload):
                        clean_line = str(raw_line or "").strip()
                        if re.fullmatch(r"\[[^\]]+\]", clean_line):
                            active_section = clean_line
                            section_counts.setdefault(active_section, 0)
                            continue
                        if active_section:
                            section_counts[active_section] = section_counts.get(active_section, 0) + 1
                    for section, minimum in part_section_minimums.items():
                        actual = section_counts.get(section, 0)
                        # 10% tolerance — 15/16 was failing the whole part for
                        # one missing line. The repair loop spent 8 attempts
                        # trying to coax that last line out and shipped a stub
                        # track instead. A single line short is close enough
                        # to ship; bigger gaps still trigger repair.
                        tolerance = max(1, int(minimum * 0.1))
                        if actual + tolerance < minimum:
                            issues.append(f"section_under_min_lines:{section}={actual}/{minimum}")
                return issues

            try:
                part_payload = self._call_until_valid(
                    f"Track Lyrics Agent Part {part_index + 1}",
                    lyric_prompt,
                    f"lyrics_part_{part_index + 1}_payload",
                    _part_validator,
                    repair_context=(
                        f"ONLY_ALLOWED_SECTION_TAGS={json.dumps(group, ensure_ascii=False)}. "
                        f"FORBIDDEN_SECTION_TAGS_ALREADY_WRITTEN={json.dumps(forbidden_sections, ensure_ascii=False)}. "
                        f"SECTION_LINE_MINIMUMS_FOR_THIS_PART={json.dumps(part_section_minimums, ensure_ascii=False)}. "
                        "sections must equal ONLY_ALLOWED_SECTION_TAGS exactly. Each allowed tag appears once. "
                        "Do not include previous invalid sections/lines. If rap is requested, write short rap bars only: "
                        "no Instrumental break text, no orchestral/stage-direction lines, no pop-ballad prose. "
                        "Fix craft issues with concrete nouns, coherent imagery, breathable lines, and no generic AI phrases."
                    ),
                )
            except AceJamAgentError as exc:
                part_payload = _lyrics_part_fallback_payload(
                    blueprint=current,
                    settings_payload={**current, **dict(hook_payload or {})},
                    lyric_plan=lyric_plan,
                    section_group=group,
                    part_index=part_index,
                    part_count=len(groups),
                    language=self.language,
                )
                fallback_issues = _part_validator(part_payload)
                self.agent_repair_count += 1
                self.logs.append(
                    f"Agent deterministic fallback: Track Lyrics Agent Part {part_index + 1} after planner failure: "
                    f"{_monitor_preview(exc, 260)}"
                )
                _append_album_debug_jsonl(
                    self.opts,
                    "04_agent_responses.jsonl",
                    {
                        "agent": f"Track Lyrics Agent Part {part_index + 1}",
                        "deterministic_fallback": True,
                        "source_error": str(exc),
                        "validation_issues": fallback_issues,
                        "payload_preview": _monitor_preview(_compact_json(part_payload), 900),
                    },
                )
                if fallback_issues:
                    raise AceJamAgentError(
                        f"Track Lyrics Agent Part {part_index + 1} fallback failed validation: "
                        f"{'; '.join(fallback_issues)}"
                    ) from exc
            lines = _director_payload_lines(part_payload)
            lyric_lines.extend(lines)
            forbidden_sections.extend(group)
            current["lyrics_lines"] = lyric_lines
            current["lyrics"] = "\n".join(lyric_lines).strip()
            _append_album_debug_jsonl(
                self.opts,
                "05_track_state.jsonl",
                {"track_number": index + 1, "stage": f"lyrics_part_{part_index + 1}", "state": current},
            )
        current["lyrics_lines"] = lyric_lines
        current["lyrics"] = "\n".join(lyric_lines).strip()
        expanded_lines, expanded = _expand_director_lyrics_lines_to_fit(
            lyric_lines,
            current,
            min_words=min_words,
            min_lines=min_lines,
            max_chars=safe_lyrics_budget,
        )
        if expanded:
            before_stats = lyric_stats("\n".join(lyric_lines))
            after_stats = lyric_stats("\n".join(expanded_lines))
            lyric_lines = expanded_lines
            current["lyrics_lines"] = lyric_lines
            current["lyrics"] = "\n".join(lyric_lines).strip()
            self.agent_repair_count += 1
            self.logs.append(
                "Agent deterministic lyrics length expansion: "
                f"track {index + 1} {before_stats.get('word_count')}/{min_words}_words "
                f"and {before_stats.get('line_count')}/{min_lines}_lines -> "
                f"{after_stats.get('word_count')}_words/{after_stats.get('line_count')}_lines"
            )
            _append_album_debug_jsonl(
                self.opts,
                "05_track_state.jsonl",
                {
                    "track_number": index + 1,
                    "stage": "lyrics_length_expansion",
                    "before": before_stats,
                    "after": after_stats,
                    "state": current,
                },
            )
        return lyric_lines

    def _generate_caption(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any],
        sonic_fields: set[str],
        *,
        repair_issues: list[str] | None = None,
    ) -> dict[str, Any]:
        caption_context = {
            key: current.get(key)
            for key in (
                "track_number",
                "title",
                "description",
                "style",
                "vibe",
                "bpm",
                "key_scale",
                "time_signature",
                "duration",
                "tag_list",
                "tags",
                "caption_dimensions_covered",
                "section_map",
            )
        }
        producer_contract = build_producer_grade_sonic_contract(current, self.opts)
        current["producer_grade_sonic_contract"] = producer_contract
        repair_note = ""
        if repair_issues:
            repair_note = (
                "\nCAPTION_REPAIR_ISSUES:\n"
                f"{json.dumps(repair_issues, ensure_ascii=False)}\n"
                "Write a fresh sound-only caption. Do not include previous invalid caption text.\n"
            )
        producer_note = (
            "\nPRODUCER_GRADE_SONIC_CONTRACT:\n"
            f"{json.dumps(producer_contract, ensure_ascii=False)}\n"
            "The caption must cover every required dimension with concrete beat, vocal, texture, arrangement, and mix information.\n"
        )
        try:
            caption_payload = self._call_until_valid(
                "Caption Agent",
                self._base_track_context(index, album_bible, caption_context, fields=sonic_fields | {"caption"})
                + repair_note
                + producer_note
                + "\nTASK:\nWrite the final compact ACE-Step caption from tag_list/Sonic DNA only. "
                "It must be a comma-separated sound prompt under 512 chars. "
                "Include primary genre, drum pocket, bass behavior, melodic/sample/riff identity, vocal delivery, arrangement movement, texture/space, and mix/master character. "
                "Do not include BPM, key, time signature, duration, model, seed, producer/person names, track title, story prose, or lyrics.\n"
                + f"OUTPUT_BLOCKS:\n{_agent_block_template('caption_agent_payload')}\n",
                "caption_agent_payload",
                lambda payload: _validate_caption_payload(payload, current)
                + _director_genre_validation_issues(payload, current, self.opts, include_lyrics=False)
                + _director_producer_grade_validation_issues(
                    payload,
                    {"style": current.get("style"), "vibe": current.get("vibe")},
                    self.opts,
                ),
                repair_context=(
                    "caption must be non-empty, under 512 chars, sound-only, and must not contain BPM, key, duration, "
                    "producer credits, person names, track title, story prose, section tags, or lyric leakage. "
                    "Cover primary_genre, drum_groove, low_end_bass, melodic_identity, vocal_delivery, arrangement_movement, "
                    "texture_space, and mix_master. If rap is requested, include rap/hip-hop as primary, rap vocal delivery, hip-hop drums/groove, and low-end."
                ),
            )
        except AceJamAgentError as exc:
            caption_payload = _caption_fallback_payload(current)
            fallback_issues = _validate_caption_payload(caption_payload, current) + _director_genre_validation_issues(
                caption_payload,
                current,
                self.opts,
                include_lyrics=False,
            ) + _director_producer_grade_validation_issues(
                caption_payload,
                {"style": current.get("style"), "vibe": current.get("vibe")},
                self.opts,
            )
            self.agent_repair_count += 1
            self.logs.append(
                f"Agent deterministic fallback: Caption Agent after planner failure: {_monitor_preview(exc, 260)}"
            )
            _append_album_debug_jsonl(
                self.opts,
                "04_agent_responses.jsonl",
                {
                    "agent": "Caption Agent",
                    "deterministic_fallback": True,
                    "source_error": str(exc),
                    "validation_issues": fallback_issues,
                    "payload_preview": _monitor_preview(_compact_json(caption_payload), 700),
                },
            )
            if fallback_issues:
                raise AceJamAgentError(
                    f"Caption Agent fallback failed validation: {'; '.join(fallback_issues)}"
                ) from exc
        current.update(caption_payload)
        current["producer_grade_readiness"] = producer_grade_readiness(current, options=self.opts)
        current["sonic_dna_coverage"] = (current["producer_grade_readiness"].get("sonic_dna_coverage") or {})
        _append_album_debug_jsonl(self.opts, "05_track_state.jsonl", {"track_number": index + 1, "stage": "Caption Agent", "state": current})
        return caption_payload

    def _generate_performance(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any],
        sonic_fields: set[str],
    ) -> dict[str, Any]:
        producer_contract = build_producer_grade_sonic_contract(current, self.opts)
        producer_note = (
            "\nPRODUCER_GRADE_SONIC_CONTRACT:\n"
            f"{json.dumps(producer_contract, ensure_ascii=False)}\n"
            "Performance guidance must reinforce all required dimensions without writing caption text or lyrics.\n"
        )
        def _performance_sonic_payload(payload: dict[str, Any]) -> dict[str, Any]:
            text = ", ".join(
                str(payload.get(field) or "").strip()
                for field in ("performance_brief", "negative_control", "genre_profile")
                if str(payload.get(field) or "").strip()
            )
            return {
                "caption": text,
                "style": current.get("style"),
                "vibe": current.get("vibe"),
            }

        try:
            performance_payload = self._call_until_valid(
                "Performance Agent",
                self._base_track_context(index, album_bible, current, fields=sonic_fields | {"caption", "hook_promise"})
                + producer_note
                + "\nTASK:\nWrite performance/mix guidance only for debug metadata. Do not write caption, tags, lyrics, BPM, key, or duration.\n"
                + f"OUTPUT_BLOCKS:\n{_agent_block_template('performance_agent_payload')}\n",
                "performance_agent_payload",
                lambda payload: _validate_performance_payload(payload)
                + _director_genre_validation_issues(payload, current, self.opts, include_lyrics=False)
                + _director_producer_grade_validation_issues(_performance_sonic_payload(payload), None, self.opts),
                repair_context=(
                    "Return only performance_brief, negative_control, and genre_profile. No caption, tags, lyrics, BPM, key, duration, or extra keys. "
                    "Cover producer-grade intent across drums, bass, melody/riff/sample, vocal delivery, movement, texture, and mix. "
                    "If rap is requested, genre_profile and performance_brief must preserve rap delivery and hip-hop groove as primary."
                ),
            )
        except AceJamAgentError as exc:
            performance_payload = _performance_fallback_payload(current)
            fallback_issues = _validate_performance_payload(performance_payload) + _director_genre_validation_issues(
                performance_payload,
                current,
                self.opts,
                include_lyrics=False,
            ) + _director_producer_grade_validation_issues(_performance_sonic_payload(performance_payload), None, self.opts)
            self.agent_repair_count += 1
            self.logs.append(
                f"Agent deterministic fallback: Performance Agent after planner failure: {_monitor_preview(exc, 260)}"
            )
            _append_album_debug_jsonl(
                self.opts,
                "04_agent_responses.jsonl",
                {
                    "agent": "Performance Agent",
                    "deterministic_fallback": True,
                    "source_error": str(exc),
                    "validation_issues": fallback_issues,
                    "payload_preview": _monitor_preview(_compact_json(performance_payload), 700),
                },
            )
            if fallback_issues:
                raise AceJamAgentError(
                    f"Performance Agent fallback failed validation: {'; '.join(fallback_issues)}"
                ) from exc
        current.update(performance_payload)
        current["producer_grade_readiness"] = producer_grade_readiness(current, options=self.opts)
        current["sonic_dna_coverage"] = (current["producer_grade_readiness"].get("sonic_dna_coverage") or {})
        _append_album_debug_jsonl(self.opts, "05_track_state.jsonl", {"track_number": index + 1, "stage": "Performance Agent", "state": current})
        return performance_payload

    def _sanitize_arrangement_lyric_leakage(
        self,
        index: int,
        current: dict[str, Any],
        gate: dict[str, Any],
    ) -> bool:
        genre_report = gate.get("genre_adherence") if isinstance(gate.get("genre_adherence"), dict) else {}
        stats = genre_report.get("stats") if isinstance(genre_report.get("stats"), dict) else {}
        scan = stats.get("arrangement_lyric_scan") if isinstance(stats.get("arrangement_lyric_scan"), list) else []
        blocked_items = [
            item for item in scan
            if isinstance(item, dict) and item.get("status") == "blocked" and str(item.get("line") or "").strip()
        ]
        if not blocked_items:
            return False
        blocked_lines = {str(item.get("line") or "").strip() for item in blocked_items}
        lines = _director_payload_lines(current)
        if not lines:
            lines = [line for line in str(current.get("lyrics") or "").splitlines() if str(line).strip()]
        safe_bars = [
            "Street truth cuts clear through the city smoke",
            "Pressure talks loud but the cadence stays clear",
            "Cold blocks know where the money went",
            "Every bar lands where the silence broke",
            "Paper trails crack when the bassline talks",
            "Loyalty stands tall when the lights get low",
        ]
        required_phrases = [
            str(item).strip()
            for item in (current.get("required_phrases") or current.get("required_lyrics") or [])
            if str(item).strip()
        ]
        replacements: list[dict[str, str]] = []
        safe_index = 0
        updated_lines: list[str] = []
        for line in lines:
            clean = str(line or "").strip()
            if clean in blocked_lines and not re.fullmatch(r"\[[^\]]+\]", clean):
                replacement = ""
                for phrase in required_phrases:
                    if phrase.casefold() in clean.casefold():
                        replacement = f"{phrase} while the block speaks clear"
                        break
                if not replacement:
                    replacement = safe_bars[safe_index % len(safe_bars)]
                    safe_index += 1
                updated_lines.append(replacement)
                replacements.append({"from": clean, "to": replacement})
            else:
                updated_lines.append(line)
        if not replacements:
            return False
        current["lyrics_lines"] = updated_lines
        current["lyrics"] = "\n".join(updated_lines).strip()
        self.agent_repair_count += 1
        self.logs.append(
            "Final gate deterministic lyric sanitizer: "
            f"track {index + 1} replaced {len(replacements)} arrangement-leakage line(s): "
            f"{_monitor_preview('; '.join(item['from'] for item in replacements), 240)}"
        )
        _append_album_debug_jsonl(
            self.opts,
            "06_gate_reports.jsonl",
            {
                "track_number": index + 1,
                "stage": "deterministic_lyric_sanitizer",
                "validation_issues": gate.get("issues") or [],
                "arrangement_lyric_scan": scan,
                "blocked_lines": [item.get("line") for item in blocked_items],
                "replacement_lines": replacements,
                "payload_preview": _monitor_preview(_compact_json({"lyrics_lines": updated_lines}), 900),
            },
        )
        return True

    def _repair_weak_lyric_sections(
        self,
        index: int,
        current: dict[str, Any],
        gate: dict[str, Any],
    ) -> bool:
        issue_texts = [str(issue) for issue in (gate.get("issues") or [])]
        repairable_names = {
            "lyrics_under_length",
            "lyrics_too_few_lines",
            "lyrics_under_hit_density",
            "lyrics_under_hit_line_density",
            "rap_verses_underfilled",
            "hook_underwritten",
        }
        if not any(any(text == name or text.startswith(name + ":") for name in repairable_names) for text in issue_texts):
            return False
        lines = _director_payload_lines(current)
        if not lines:
            lines = [line.strip() for line in str(current.get("lyrics") or "").splitlines() if line.strip()]
        if not any(re.fullmatch(r"\[[^\]]+\]", line) for line in lines):
            return False
        duration = parse_duration_seconds(current.get("duration") or self.track_duration, self.track_duration)
        genre_hint = _director_track_genre_hint(current, self.opts)
        density_gate = gate.get("lyric_density_gate") if isinstance(gate.get("lyric_density_gate"), dict) else {}
        plan = dict(density_gate.get("plan") or {})
        if not plan:
            plan = lyric_length_plan(
                duration,
                str(current.get("lyric_density") or self.opts.get("lyric_density") or "dense"),
                str(current.get("structure_preset") or self.opts.get("structure_preset") or "auto"),
                genre_hint,
            )
        min_words = max(int(plan.get("min_words") or 0), int((int(plan.get("target_words") or 0)) * 0.82))
        min_lines = max(
            _director_effective_min_lines(int(plan.get("min_lines") or 0), int(plan.get("min_words") or 0)),
            int((int(plan.get("target_lines") or 0)) * 0.82),
        )
        max_chars = min(3600, int(plan.get("max_lyrics_chars") or ACE_STEP_LYRICS_SAFE_HEADROOM), ACE_STEP_LYRICS_CHAR_LIMIT - 360)
        tags = [line for line in lines if re.fullmatch(r"\[[^\]]+\]", line)]
        minimums = current.get("lyric_section_minimums") if isinstance(current.get("lyric_section_minimums"), dict) else {}
        if not minimums:
            minimums = _director_section_line_minimums(tags, duration=duration, genre_hint=genre_hint)
        bars = _director_lyric_extension_lines(current)
        used = {line.casefold() for line in lines}
        added: list[dict[str, str]] = []
        bar_index = 0

        def _section_counts(source: list[str]) -> dict[str, int]:
            counts: dict[str, int] = {}
            active = ""
            for raw_line in source:
                clean = str(raw_line or "").strip()
                if re.fullmatch(r"\[[^\]]+\]", clean):
                    active = clean
                    counts.setdefault(active, 0)
                elif active:
                    counts[active] = counts.get(active, 0) + 1
            return counts

        def _next_bar() -> str:
            nonlocal bar_index
            for _ in range(len(bars) * 3):
                bar = bars[bar_index % len(bars)]
                bar_index += 1
                if bar.casefold() not in used:
                    used.add(bar.casefold())
                    return bar
            bar = f"Street truth rises in measure {bar_index}"
            bar_index += 1
            used.add(bar.casefold())
            return bar

        def _insert(section: str, line: str) -> bool:
            try:
                section_index = lines.index(section)
            except ValueError:
                return False
            insert_at = len(lines)
            for pos in range(section_index + 1, len(lines)):
                if re.fullmatch(r"\[[^\]]+\]", lines[pos]):
                    insert_at = pos
                    break
            candidate = lines[:insert_at] + [line] + lines[insert_at:]
            if len("\n".join(candidate)) > max_chars:
                return False
            lines[:] = candidate
            added.append({"section": section, "line": line})
            return True

        counts = _section_counts(lines)
        preferred_sections = [tag for tag in tags if re.search(r"verse", tag, re.I)] or [
            tag for tag in tags if not re.search(r"intro|outro|break|instrumental", tag, re.I)
        ] or tags
        for section, minimum in minimums.items():
            if section not in tags:
                continue
            while counts.get(section, 0) < int(minimum or 0):
                if not _insert(section, _next_bar()):
                    break
                counts = _section_counts(lines)
        stats = lyric_stats("\n".join(lines))
        round_index = 0
        while (
            (int(stats.get("word_count") or 0) < min_words or int(stats.get("line_count") or 0) < min_lines)
            and round_index < 96
        ):
            section = preferred_sections[round_index % len(preferred_sections)]
            if not _insert(section, _next_bar()):
                break
            stats = lyric_stats("\n".join(lines))
            round_index += 1
        if not added:
            return False
        current["lyrics_lines"] = lines
        current["lyrics"] = "\n".join(lines).strip()
        current["lyric_density_gate"] = lyric_density_gate(
            current["lyrics"],
            plan,
            duration=duration,
            genre_hint=genre_hint,
            instrumental=False,
        )
        self.agent_repair_count += 1
        self.logs.append(
            "Final gate deterministic lyric density repair: "
            f"track {index + 1} inserted {len(added)} targeted bar(s) into weak sections"
        )
        _append_album_debug_jsonl(
            self.opts,
            "06_gate_reports.jsonl",
            {
                "track_number": index + 1,
                "stage": "deterministic_lyric_density_repair",
                "validation_issues": issue_texts,
                "lyric_density_plan": plan,
                "section_minimums": minimums,
                "inserted_lines": added,
                "lyric_density_gate": current["lyric_density_gate"],
                "payload_preview": _monitor_preview(_compact_json({"lyrics_lines": lines}), 900),
            },
        )
        return True

    def _repair_lyric_craft(
        self,
        index: int,
        album_bible: dict[str, Any],
        current: dict[str, Any],
        gate: dict[str, Any],
        section_tags: list[str],
        hook_payload: dict[str, Any],
        lyric_fields: set[str],
    ) -> bool:
        craft_gate = gate.get("lyric_craft_gate") if isinstance(gate.get("lyric_craft_gate"), dict) else {}
        craft_issue_ids = [str(item) for item in (craft_gate.get("issue_ids") or [])]
        if not craft_issue_ids:
            return False
        repairable = {
            "lyric_craft_generic_ai_phrase",
            "lyric_craft_adjective_stacking",
            "lyric_craft_mixed_metaphor",
            "lyric_craft_hook_weak",
            "lyric_craft_no_concrete_scene",
            "lyric_craft_line_breathability",
            "lyric_craft_rhyme_chaos",
            "lyric_craft_section_blur",
        }
        if not any(issue in repairable for issue in craft_issue_ids):
            return False
        lines = _director_payload_lines(current)
        if not lines:
            lines = [line.strip() for line in str(current.get("lyrics") or "").splitlines() if line.strip()]
        if not lines or not any(re.fullmatch(r"\[[^\]]+\]", line) for line in lines):
            return False
        blocks = _director_section_blocks_from_lines(lines)
        existing_by_key = {_section_key_for_director(str(block.get("tag") or "")): str(block.get("tag") or "") for block in blocks}
        weak_sections: list[str] = []
        for item in craft_gate.get("weak_sections") or []:
            section = str(item.get("section") if isinstance(item, dict) else item).strip()
            key = _section_key_for_director(section)
            if key in existing_by_key and existing_by_key[key] not in weak_sections and key != "untagged":
                weak_sections.append(existing_by_key[key])
        if "lyric_craft_hook_weak" in craft_issue_ids:
            for block in blocks:
                tag = str(block.get("tag") or "")
                if re.search(r"chorus|hook|refrain", tag, re.I) and tag not in weak_sections:
                    weak_sections.append(tag)
        if not weak_sections:
            for block in blocks:
                tag = str(block.get("tag") or "")
                if re.search(r"verse|chorus|hook|bridge", tag, re.I) and tag not in weak_sections:
                    weak_sections.append(tag)
                if len(weak_sections) >= 2:
                    break
        target_sections = [tag for tag in section_tags if _section_key_for_director(tag) in {_section_key_for_director(item) for item in weak_sections}]
        if not target_sections:
            target_sections = weak_sections[:3]
        target_sections = target_sections[:3]
        if not target_sections:
            return False
        current_sections: list[str] = []
        target_keys = {_section_key_for_director(tag) for tag in target_sections}
        for block in blocks:
            tag = str(block.get("tag") or "")
            if _section_key_for_director(tag) not in target_keys:
                continue
            current_sections.append(tag)
            current_sections.extend(str(line) for line in block.get("lines") or [])
        duration = parse_duration_seconds(current.get("duration") or self.track_duration, self.track_duration)
        genre_hint = _director_track_genre_hint(current, self.opts)
        density_plan = ((gate.get("lyric_density_gate") or {}).get("plan") or {}) if isinstance(gate.get("lyric_density_gate"), dict) else {}
        craft_contract = build_lyrical_craft_contract(current, self.opts)
        repair_prompt = (
            self._base_track_context(index, album_bible, current, include_lyric_constraints=True, fields=lyric_fields)
            + "\nLYRIC_CRAFT_REPAIR_ISSUES:\n"
            + f"{json.dumps(craft_issue_ids, ensure_ascii=False)}\n"
            + "LYRICAL_CRAFT_CONTRACT:\n"
            + f"{json.dumps(craft_contract, ensure_ascii=False)}\n"
            + "TARGET_SECTIONS_TO_REWRITE:\n"
            + f"{json.dumps(target_sections, ensure_ascii=False)}\n"
            + "CURRENT_TARGET_SECTION_LINES:\n"
            + f"{json.dumps(current_sections, ensure_ascii=False)}\n"
            + "FULL_SECTION_MAP_DO_NOT_CHANGE:\n"
            + f"{json.dumps(section_tags, ensure_ascii=False)}\n"
            + "HOOK_LINES_TO_PRESERVE_OR_IMPROVE:\n"
            + f"{json.dumps(hook_payload.get('hook_lines') or [], ensure_ascii=False)}\n"
            + "Rewrite only the target sections. Keep each target bracket tag exactly once. "
            "Do not output any non-target section. Do not include previous weak lines unless they are genuinely strong. "
            "Replace generic phrases with concrete human details, coherent metaphor, breathable lines, and genre-correct flow. "
            "Never include BPM, key, caption, metadata, producer names, markdown, explanations, or stage directions.\n"
            + f"OUTPUT_BLOCKS:\n{_agent_block_template('lyric_craft_repair_payload')}\n"
        )

        def _repair_validator(payload: dict[str, Any]) -> list[str]:
            issues: list[str] = []
            actual_sections = _director_section_tags({"section_map": payload.get("sections") or []})
            if [_section_key_for_director(item) for item in actual_sections] != [_section_key_for_director(item) for item in target_sections]:
                issues.append("sections_mismatch")
            replacement_lines = _director_payload_lines(payload)
            _merged, _changed, merge_issues = _director_replace_section_blocks(lines, replacement_lines, target_sections)
            issues.extend(merge_issues)
            partial_craft = lyric_craft_gate(
                "\n".join(replacement_lines),
                current,
                options=self.opts,
                plan=density_plan,
                duration=duration,
                genre_hint=genre_hint,
                partial=True,
            )
            for craft_issue in partial_craft.get("issues") or []:
                craft_id = str(craft_issue.get("id") if isinstance(craft_issue, dict) else craft_issue)
                if craft_id in {
                    "lyric_craft_placeholder",
                    "lyric_craft_fallback_artifact",
                    "lyric_craft_metadata_leakage",
                    "lyric_craft_generic_ai_phrase",
                    "lyric_craft_adjective_stacking",
                    "lyric_craft_line_breathability",
                    "lyric_craft_mixed_metaphor",
                    "lyric_craft_low_score",
                }:
                    issues.append(craft_id)
            return sorted(set(issues))

        try:
            repair_payload = self._call_until_valid(
                "Lyric Craft Repair Agent",
                repair_prompt,
                "lyric_craft_repair_payload",
                _repair_validator,
                repair_context=(
                    f"Rewrite only TARGET_SECTIONS_TO_REWRITE={json.dumps(target_sections, ensure_ascii=False)}. "
                    "sections must match exactly. Use delimiter blocks only. Remove generic AI phrases, mixed metaphors, "
                    "overlong lines, weak hook wording, and filler while preserving the bracket tags."
                ),
            )
        except AceJamAgentError as exc:
            self.logs.append(
                f"Lyric Craft Repair Agent failed for track {index + 1}; falling back to lyric part regeneration: "
                f"{_monitor_preview(exc, 240)}"
            )
            _append_album_debug_jsonl(
                self.opts,
                "06_gate_reports.jsonl",
                {
                    "track_number": index + 1,
                    "stage": "lyric_craft_repair_failed",
                    "validation_issues": craft_issue_ids,
                    "target_sections": target_sections,
                    "source_error": str(exc),
                    "repair_strategy": "part_regeneration",
                },
            )
            return False
        replacement_lines = _director_payload_lines(repair_payload)
        merged, changed, merge_issues = _director_replace_section_blocks(lines, replacement_lines, target_sections)
        if not changed or merge_issues:
            return False
        current["lyrics_lines"] = merged
        current["lyrics"] = "\n".join(merged).strip()
        current["lyrical_craft_contract"] = craft_contract
        current["lyric_craft_gate"] = lyric_craft_gate(
            current["lyrics"],
            current,
            options=self.opts,
            plan=density_plan,
            duration=duration,
            genre_hint=genre_hint,
            partial=False,
        )
        self.agent_repair_count += 1
        self.logs.append(
            "Final gate lyric craft repair: "
            f"track {index + 1} rewrote {', '.join(target_sections)} for {', '.join(craft_issue_ids)}"
        )
        _append_album_debug_jsonl(
            self.opts,
            "06_gate_reports.jsonl",
            {
                "track_number": index + 1,
                "stage": "lyric_craft_repair",
                "validation_issues": craft_issue_ids,
                "weak_sections": craft_gate.get("weak_sections") or [],
                "target_sections": target_sections,
                "repair_strategy": "section_rewrite",
                "lyric_craft_gate": current["lyric_craft_gate"],
                "rejected_lyric_preview": _monitor_preview("\n".join(current_sections), 900),
                "payload_preview": _monitor_preview(_compact_json(repair_payload), 900),
            },
        )
        return True

    def _assemble_track(
        self,
        index: int,
        slot: dict[str, Any],
        current: dict[str, Any],
        section_tags: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_payload = _director_build_final_payload(current, self.language)
        self.logs.append(f"Final Payload Assembler: deterministic assembly for track {index + 1}; no LLM call.")
        self.agent_rounds.append({"agent": "Final Payload Assembler", "status": "deterministic"})
        _append_album_debug_jsonl(
            self.opts,
            "05_track_state.jsonl",
            {"track_number": index + 1, "stage": "Final Payload Assembler", "state": final_payload, "deterministic": True},
        )
        track = _final_payload_preserve_sources(current, final_payload)
        track["track_number"] = index + 1
        track["title"] = slot.get("title") or track.get("title") or f"Track {index + 1}"
        track["duration"] = parse_duration_seconds(slot.get("duration") or track.get("duration") or self.track_duration, self.track_duration)
        track["tags"] = track.get("caption") or ""
        final_lines = _director_payload_lines(track)
        if final_lines:
            track["lyrics_lines"] = final_lines
            track["lyrics"] = "\n".join(final_lines).strip()
        track["language"] = track.get("language") or self.language
        track["vocal_language"] = track.get("vocal_language") or track.get("language") or self.language
        track["lyric_density"] = str(self.opts.get("lyric_density") or current.get("lyric_density") or "dense")
        track["structure_preset"] = str(self.opts.get("structure_preset") or current.get("structure_preset") or "auto")
        selected_model = self.model_info.get("model") if self.model_info.get("model") != "per-model portfolio" else ALBUM_FINAL_MODEL
        track["song_model"] = selected_model or ALBUM_FINAL_MODEL
        track["quality_profile"] = self.opts.get("quality_profile") or DEFAULT_QUALITY_PROFILE
        track["ace_lm_model"] = "none"
        track["allow_supplied_lyrics_lm"] = False
        track["thinking"] = False
        track["use_format"] = False
        track["use_cot_metas"] = False
        track["use_cot_caption"] = False
        track["use_cot_lyrics"] = False
        track["use_cot_language"] = False
        genre_adherence = evaluate_genre_adherence(track, self.opts)
        track["genre_intent_contract"] = genre_adherence.get("contract") or {}
        track["genre_adherence"] = {key: value for key, value in genre_adherence.items() if key != "contract"}
        track["genre_validation_issues"] = genre_adherence.get("issue_ids") or []
        producer_readiness = producer_grade_readiness(track, options=self.opts)
        track["producer_grade_sonic_contract"] = (producer_readiness.get("sonic_dna_coverage") or {}).get("contract") or {}
        track["sonic_dna_coverage"] = producer_readiness.get("sonic_dna_coverage") or {}
        track["producer_grade_readiness"] = producer_readiness
        craft_gate = lyric_craft_gate(
            track.get("lyrics") or "",
            track,
            options=self.opts,
            duration=track.get("duration") or self.track_duration,
            genre_hint=_director_track_genre_hint(track, self.opts),
            instrumental=str(track.get("lyrics") or "").strip().lower() == "[instrumental]" or bool(track.get("instrumental")),
        )
        track["lyrical_craft_contract"] = craft_gate.get("contract") or {}
        track["lyric_craft_gate"] = craft_gate
        track["lyric_craft_score"] = craft_gate.get("score")
        track["lyric_craft_issues"] = craft_gate.get("issue_ids") or []
        track["agent_complete_payload"] = True
        track["agent_director_version"] = ACEJAM_ALBUM_DIRECTOR_VERSION
        # Fill remaining UI-visible fields so the album wizard does not show
        # blank slots for inference settings, negative tags, artist credit and
        # production-team metadata. The user can still edit any of these in
        # the wizard after the crew run finishes.
        bible_obj = getattr(self, "album_bible", None)
        bible_artist = ""
        bible_mood = ""
        bible_concept = ""
        bible_genre_hint = ""
        if isinstance(bible_obj, dict):
            bible_artist = str(bible_obj.get("artist_name") or "").strip()
            bible_mood = str(bible_obj.get("mood") or bible_obj.get("mood_vibe") or "").strip()
            bible_concept = str(bible_obj.get("concept") or bible_obj.get("one_sentence_concept") or "").strip()
            bible_genre_hint = str(bible_obj.get("genre_prompt") or bible_obj.get("genre_hint") or "").strip()
        # Deterministic safety net: if Track Concept Agent left style/vibe/
        # narrative empty even after the validator's repair loop, derive
        # them from album bible + caption/genre hints so the wizard is never
        # blank. The crew is the primary source; this is the floor.
        producer_credit = str(track.get("producer_credit") or "").strip()
        caption_text = str(track.get("caption") or "").strip()
        first_caption_token = caption_text.split(",")[0].strip() if caption_text else ""
        if not str(track.get("style") or "").strip():
            derived_style_parts = [
                bit
                for bit in (bible_genre_hint, producer_credit, first_caption_token)
                if bit
            ]
            track["style"] = ", ".join(dict.fromkeys(derived_style_parts))[:160]
        if not str(track.get("vibe") or "").strip():
            track["vibe"] = bible_mood or "(set in wizard)"
        if not str(track.get("narrative") or "").strip():
            track["narrative"] = bible_concept or str(track.get("description") or "")
        if not str(track.get("description") or "").strip():
            track["description"] = str(track.get("narrative") or bible_concept)
        # Track-level mood / genre / vocal_type — first prefer crew-derived
        # values, fall back to album-level option keys, finally to derived.
        track["mood"] = str(
            track.get("mood")
            or self.opts.get("album_mood")
            or self.opts.get("mood_vibe")
            or bible_mood
            or ""
        ).strip()
        track["genre"] = str(
            track.get("genre")
            or self.opts.get("album_genre")
            or self.opts.get("genre_prompt")
            or bible_genre_hint
            or first_caption_token
            or ""
        ).strip()
        track["vocal_type"] = str(
            track.get("vocal_type")
            or self.opts.get("album_vocal_type")
            or self.opts.get("vocal_type")
            or ""
        ).strip()
        artist_default = (
            slot.get("artist_name")
            or current.get("artist_name")
            or bible_artist
            or self.opts.get("artist_name")
            or ""
        )
        if not track.get("artist_name"):
            track["artist_name"] = str(artist_default).strip()
        producer_default = (
            slot.get("producer_credit")
            or current.get("producer_credit")
            or self.opts.get("producer_credit")
            or ""
        )
        if not track.get("producer_credit"):
            track["producer_credit"] = str(producer_default).strip()
        # Negative tags: combine album-level negatives with the cookbook default
        # control list. ACE-Step itself does not always read negative_tags but
        # the field is shown in the wizard and used by other AceJAM gates.
        existing_negatives = str(track.get("negative_tags") or self.opts.get("negative_tags") or "").strip()
        negative_tokens: list[str] = []
        if existing_negatives:
            negative_tokens = [item.strip() for item in re.split(r",\s*", existing_negatives) if item.strip()]
        for default_token in DEFAULT_NEGATIVE_CONTROL:
            if default_token not in negative_tokens:
                negative_tokens.append(default_token)
        track["negative_tags"] = ", ".join(negative_tokens)
        # Inference settings: pass through album-level config so wizard shows
        # the actual values that will be sent to ACE-Step rather than blanks.
        track["inference_steps"] = clamp_int(
            track.get("inference_steps") or self.opts.get("inference_steps") or ALBUM_FINAL_DOCS_BEST["inference_steps"],
            ALBUM_FINAL_DOCS_BEST["inference_steps"],
            1,
            200,
        )
        track["guidance_scale"] = float(
            track.get("guidance_scale") or self.opts.get("guidance_scale") or ALBUM_FINAL_DOCS_BEST["guidance_scale"]
        )
        track["shift"] = float(
            track.get("shift") or self.opts.get("shift") or ALBUM_FINAL_DOCS_BEST["shift"]
        )
        track["infer_method"] = str(
            track.get("infer_method") or self.opts.get("infer_method") or ALBUM_FINAL_DOCS_BEST.get("infer_method") or "ode"
        )
        track["audio_format"] = str(
            track.get("audio_format") or self.opts.get("audio_format") or ALBUM_FINAL_DOCS_BEST.get("audio_format") or "wav32"
        )
        track["seed"] = str(track.get("seed") or "-1")
        track["use_random_seed"] = bool(track.get("use_random_seed", True))
        track["auto_score"] = bool(track.get("auto_score", False))
        track["auto_lrc"] = bool(track.get("auto_lrc", False))
        track["return_audio_codes"] = bool(track.get("return_audio_codes", True))
        track["save_to_library"] = bool(track.get("save_to_library", True))
        # Production team metadata: surface which agent did what so the user
        # can see the crew makeup in the wizard. Each role maps to a tiny
        # signature; the user can edit credits freely.
        if not track.get("production_team"):
            track["production_team"] = {
                "executive_producer": "AceJAM Director Agent",
                "artist_performer": str(track.get("artist_name") or ""),
                "songwriter": "AceJAM Track Concept + Lyrics Agents",
                "rhyme_metaphor_editor": "AceJAM Lyric Craft Gate",
                "beat_producer": str(track.get("producer_credit") or "AceJAM Sonic Tags Agent"),
                "ace_step_prompt_engineer": "AceJAM Caption Polisher",
                "studio_engineer": "AceJAM Final Payload Assembler",
                "ar_quality_gate": "AceJAM Payload + Lyric Craft Gates",
            }
        gate = _director_minimal_validate(track, section_tags, self.opts)
        track["payload_gate_status"] = gate["status"]
        track["payload_quality_gate"] = gate
        track["lyrics_quality"] = gate.get("lyrics_quality") or _director_lyrics_quality(track, self.opts, gate)
        track["debug_paths"] = self._debug_paths()
        track["album_writer_mode"] = str(self.opts.get("album_writer_mode") or ALBUM_WRITER_MODE_DEFAULT)
        # Quality report: mirror the gate scores so the wizard can show
        # the user how each track scored before they hit render. Runs after
        # the gate so warnings + lyrics_quality are available.
        if not track.get("quality_report"):
            track["quality_report"] = {
                "hit_angle": str(track.get("description") or "")[:240],
                "hook": (track.get("hook_promise") or current.get("hook_promise") or ""),
                "metaphor_world": str(track.get("style") or current.get("style") or "")[:240],
                "rhyme_flow": str((craft_gate or {}).get("summary") or "")[:240],
                "energy_curve": str(track.get("vibe") or current.get("vibe") or "")[:240],
                "lyric_word_target": int((track.get("lyrics_quality") or {}).get("target_words") or 0),
                "section_plan": list(section_tags or []),
                "warnings": list((gate or {}).get("warnings") or (track.get("lyric_craft_issues") or [])),
            }
        _append_album_debug_jsonl(self.opts, "06_gate_reports.jsonl", {"track_number": index + 1, "title": track.get("title"), "gate": gate})
        _print_agent_io(self.opts, f"track_{index + 1}_gate_report", gate)
        return track, gate

    def _write_track(self, index: int, album_bible: dict[str, Any], previous_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        slot = self._locked_track_slot(index)
        concept_fields = {
            "track_number", "locked_title", "source_title", "title", "description", "style",
            "vibe", "narrative", "duration", "bpm", "key_scale", "time_signature",
            "language", "vocal_language", "required_phrases", "required_lyrics",
        }
        sonic_fields = {
            "track_number", "locked_title", "source_title", "title", "description", "style",
            "vibe", "narrative", "duration", "bpm", "key_scale", "time_signature",
            "language", "vocal_language", "tag_list", "tags", "caption_dimensions_covered",
        }
        lyric_fields = {
            "track_number", "locked_title", "source_title", "title", "description", "style",
            "vibe", "narrative", "duration", "bpm", "key_scale", "time_signature",
            "language", "vocal_language", "tag_list", "tags", "hook_promise",
            "required_phrases", "required_lyrics",
        }
        concept_payload = self._call_until_valid(
            "Track Concept Agent",
            self._base_track_context(index, album_bible, slot, include_lyric_constraints=True, fields=concept_fields)
            + f"\nPREVIOUS_TRACK_SUMMARIES:\n{json.dumps(previous_summaries, ensure_ascii=False, indent=2)}\n"
            + f"OUTPUT_BLOCKS:\n{_agent_block_template('track_concept_payload')}\n",
            "track_concept_payload",
            _validate_track_concept_payload,
            repair_context="title, description, and style must be non-empty. Preserve any locked user title.",
        )
        current = {**slot, **concept_payload, "track_number": index + 1}
        current["title"] = slot.get("title") or current.get("title") or f"Track {index + 1}"
        current["genre_intent_contract"] = build_genre_intent_contract(current, self.opts)

        self._generate_tags(index, album_bible, current, sonic_fields)

        for agent_name, schema, instruction, validator, repair_context in [
            (
                "BPM Agent",
                '{"bpm":95}',
                "Choose the BPM from concept, locked fields, genre prompt, Sonic DNA tags, and track role. Do not write lyrics.",
                _validate_bpm_payload,
                "bpm must be a numeric tempo between 40 and 220.",
            ),
            (
                "Key Agent",
                '{"key_scale":"A minor"}',
                "Choose the ACE-Step key_scale from concept, mood, Sonic DNA, and BPM. Do not write lyrics.",
                _validate_key_payload,
                "key_scale must be a non-empty musical key like A minor or C major.",
            ),
            (
                "Time Signature Agent",
                '{"time_signature":"4"}',
                "Choose the supported time signature from concept, groove, and BPM. Do not write lyrics.",
                _validate_time_signature_payload,
                "time_signature must be non-empty and use a compact value like 4, 3, or 6/8.",
            ),
            (
                "Duration Agent",
                f'{{"duration":{int(self.track_duration)}}}',
                "Choose duration seconds; keep the requested full-song duration unless locked otherwise. Do not write lyrics.",
                _validate_duration_payload,
                "duration must be plausible seconds or m:ss between 10 and 600 seconds.",
            ),
        ]:
            schema_name = agent_name.lower().replace(" ", "_") + "_payload"
            payload = self._call_until_valid(
                agent_name,
                self._base_track_context(index, album_bible, current, fields=sonic_fields)
                + f"\nTASK:\n{instruction}\n"
                + f"OUTPUT_BLOCKS:\n{_agent_block_template(schema_name)}\n",
                schema_name,
                validator,
                repair_context=repair_context,
            )
            current.update(payload)
            _append_album_debug_jsonl(
                self.opts,
                "05_track_state.jsonl",
                {"track_number": index + 1, "stage": agent_name, "state": current},
            )
        current["duration"] = parse_duration_seconds(slot.get("duration") or current.get("duration") or self.track_duration, self.track_duration)
        current["lyric_density"] = str(self.opts.get("lyric_density") or current.get("lyric_density") or "dense")
        current["structure_preset"] = str(self.opts.get("structure_preset") or current.get("structure_preset") or "auto")

        _section_payload, section_tags = self._generate_section_map(index, album_bible, current, lyric_fields)
        hook_payload = self._generate_hook(index, album_bible, current, lyric_fields)
        self._generate_lyrics_parts(index, album_bible, current, lyric_fields, section_tags, hook_payload)
        self._generate_caption(index, album_bible, current, sonic_fields)
        self._generate_performance(index, album_bible, current, sonic_fields)

        track, gate = self._assemble_track(index, slot, current, section_tags)
        issue_history: list[dict[str, Any]] = []
        requested_repair_rounds = int(self.opts.get("max_track_repair_rounds") or ALBUM_TRACK_GATE_REPAIR_RETRIES)
        max_gate_repairs = max(0, min(3, requested_repair_rounds, ACEJAM_AGENT_GATE_REPAIR_RETRIES))

        def _has_issue(issues: list[str], *names: str) -> bool:
            return any(
                str(issue) == name or str(issue).startswith(name + ":")
                for issue in issues
                for name in names
            )

        for repair_attempt in range(1, max_gate_repairs + 1):
            if gate.get("gate_passed"):
                break
            issues = [str(issue) for issue in (gate.get("issues") or [])]
            issue_history.append({"attempt": repair_attempt, "issues": issues})
            issue_text = "; ".join(issues)
            self.agent_repair_count += 1
            self.logs.append(
                f"Final gate repair retry: track {index + 1} attempt {repair_attempt}/{max_gate_repairs}: {issue_text}"
            )
            _append_album_debug_jsonl(
                self.opts,
                "06_gate_reports.jsonl",
                {
                    "track_number": index + 1,
                    "title": track.get("title"),
                    "final_gate_repair_attempt": repair_attempt,
                    "repair_attempt": repair_attempt,
                    "validation_issues": issues,
                    "rejected_payload_preview": _monitor_preview(_compact_json(track), 900),
                },
            )
            if (
                _has_issue(issues, "non_rap_arrangement_lyric_leakage")
                and all(_has_issue([issue], "non_rap_arrangement_lyric_leakage") for issue in issues)
                and self._sanitize_arrangement_lyric_leakage(index, current, gate)
            ):
                track, gate = self._assemble_track(index, slot, current, section_tags)
                if gate.get("gate_passed"):
                    break
                issues = [str(issue) for issue in (gate.get("issues") or [])]
                if not _has_issue(issues, "non_rap_arrangement_lyric_leakage"):
                    continue
            if (
                _has_issue(
                    issues,
                    "lyrics_under_length",
                    "lyrics_too_few_lines",
                    "lyrics_under_hit_density",
                    "lyrics_under_hit_line_density",
                    "rap_verses_underfilled",
                    "hook_underwritten",
                )
                and self._repair_weak_lyric_sections(index, current, gate)
            ):
                track, gate = self._assemble_track(index, slot, current, section_tags)
                if gate.get("gate_passed"):
                    break
                issues = [str(issue) for issue in (gate.get("issues") or [])]
            craft_issue_names = {
                "lyric_craft_generic_ai_phrase",
                "lyric_craft_adjective_stacking",
                "lyric_craft_mixed_metaphor",
                "lyric_craft_hook_weak",
                "lyric_craft_no_concrete_scene",
                "lyric_craft_line_breathability",
                "lyric_craft_rhyme_chaos",
                "lyric_craft_section_blur",
                "lyric_craft_low_score",
            }
            if _has_issue(issues, "lyric_craft_hook_weak"):
                hook_payload = self._generate_hook(index, album_bible, current, lyric_fields, repair_issues=issues)
            if _has_issue(issues, *sorted(craft_issue_names)) and self._repair_lyric_craft(
                index,
                album_bible,
                current,
                gate,
                section_tags,
                hook_payload,
                lyric_fields,
            ):
                track, gate = self._assemble_track(index, slot, current, section_tags)
                if gate.get("gate_passed"):
                    break
                issues = [str(issue) for issue in (gate.get("issues") or [])]
            needs_section_hook_lyrics = _has_issue(issues, "section_map_missing_hook", "lyrics_missing_hook_section")
            needs_lyrics = needs_section_hook_lyrics or _has_issue(
                issues,
                "duplicate_section_tags",
                "section_map_mismatch",
                "missing_lyrics",
                "lyrics_over_4096",
                "lyrics_under_length",
                "lyrics_too_few_lines",
                "lyrics_under_hit_density",
                "lyrics_under_hit_line_density",
                "rap_verses_underfilled",
                "hook_underwritten",
                "lyric_unique_line_ratio_low",
                "fallback_reprise_overuse",
                "non_rap_arrangement_lyric_leakage",
                "rap_lines_not_bar_like",
                *sorted(craft_issue_names),
            )
            needs_tags = _has_issue(
                issues,
                "genre_intent_missing_rap_core",
                "genre_intent_missing_rap_vocal",
                "genre_intent_missing_rap_groove",
                "genre_intent_missing_low_end",
                "rap_not_dominant",
                "orchestral_overdominant",
                "producer_grade_missing_primary_genre",
                "producer_grade_missing_drum_groove",
                "producer_grade_missing_low_end_bass",
                "producer_grade_missing_melodic_identity",
                "producer_grade_missing_vocal_delivery",
                "producer_grade_missing_arrangement_movement",
                "producer_grade_missing_texture_space",
                "producer_grade_missing_mix_master",
                "producer_grade_sonic_dna_unrepaired",
            )
            needs_caption = _has_issue(
                issues,
                "missing_caption",
                "caption_over_512",
                "metadata_or_credit_in_caption",
                "producer_credit_in_caption",
                "artist_name_in_caption",
                "title_in_caption",
                "lyric_or_story_marker_in_caption",
                "section_tag_in_caption",
            ) or needs_tags
            needs_performance = needs_tags
            if not any([needs_section_hook_lyrics, needs_lyrics, needs_caption, needs_tags, needs_performance]):
                needs_caption = True
                needs_lyrics = True
            if needs_tags:
                self._generate_tags(index, album_bible, current, sonic_fields, repair_issues=issues)
            if needs_section_hook_lyrics:
                _section_payload, section_tags = self._generate_section_map(
                    index,
                    album_bible,
                    current,
                    lyric_fields,
                    repair_issues=issues,
                )
                hook_payload = self._generate_hook(index, album_bible, current, lyric_fields, repair_issues=issues)
                self._generate_lyrics_parts(
                    index,
                    album_bible,
                    current,
                    lyric_fields,
                    section_tags,
                    hook_payload,
                    repair_issues=issues,
                )
            elif needs_lyrics:
                self._generate_lyrics_parts(
                    index,
                    album_bible,
                    current,
                    lyric_fields,
                    section_tags,
                    hook_payload,
                    repair_issues=issues,
                )
            if needs_caption:
                self._generate_caption(index, album_bible, current, sonic_fields, repair_issues=issues)
            if needs_performance:
                self._generate_performance(index, album_bible, current, sonic_fields)
            track, gate = self._assemble_track(index, slot, current, section_tags)

        if not gate.get("gate_passed"):
            final_issues = [str(issue) for issue in (gate.get("issues") or [])]
            issue_history.append({"attempt": "final", "issues": final_issues})
            debug_paths = {
                "responses": str(Path(self.agent_debug_dir) / "04_agent_responses.jsonl") if self.agent_debug_dir else "",
                "gate_reports": str(Path(self.agent_debug_dir) / "06_gate_reports.jsonl") if self.agent_debug_dir else "",
                "rejected_payloads": str(Path(self.agent_debug_dir) / "07_rejected_payloads.jsonl") if self.agent_debug_dir else "",
            }
            _append_album_debug_jsonl(
                self.opts,
                "07_rejected_payloads.jsonl",
                {
                    "track_number": index + 1,
                    "title": track.get("title"),
                    "payload": track,
                    "gate": gate,
                    "issue_history": issue_history,
                    "debug_paths": debug_paths,
                },
            )
            _print_agent_io(self.opts, f"track_{index + 1}_rejected_payload", track)
            raise AceJamAgentError(
                f"AlbumPayloadQualityGate failed for track {index + 1} after {max_gate_repairs} repair attempt(s): "
                f"{json.dumps(issue_history, ensure_ascii=False)}. Debug paths: {json.dumps(debug_paths, ensure_ascii=False)}"
            )
        _append_album_debug_jsonl(self.opts, "07_final_payloads.jsonl", {"track_number": index + 1, "title": track.get("title"), "payload": track})
        _print_agent_io(self.opts, f"track_{index + 1}_final_payload", track)
        return track


# The active album runtime starts here. Legacy large CrewAI/RAG planner paths are
# kept only for import compatibility; plan_album selects AceJAM Direct or CrewAI Micro.

def plan_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
    language: str = "en",
    embedding_model: str = DEFAULT_ALBUM_EMBEDDING_MODEL,
    options: dict[str, Any] | None = None,
    use_crewai: bool = True,
    input_tracks: list[dict[str, Any]] | None = None,
    planner_provider: str = "ollama",
    embedding_provider: str = "ollama",
    log_callback: Callable[[str], None] | None = None,
    crewai_output_log_file: str | None = None,
) -> dict[str, Any]:
    """Compatibility facade for AceJAM Direct and CrewAI Micro album planning."""
    logs: list[str] = _AlbumPlanLogs(log_callback)
    requested_engine_raw = (options or {}).get("agent_engine")
    requested_engine_text = str(requested_engine_raw or "").strip().lower()
    requested_engine = normalize_album_agent_engine(requested_engine_raw)
    recovered_concept = _recover_album_concept(concept, options, input_tracks)
    if not recovered_concept:
        message = "Album concept is empty. Provide concept, user_prompt, album_title, or track hints before planning."
        logs.append(f"ERROR: {message}")
        return {
            "tracks": [],
            "logs": logs,
            "success": False,
            "error": message,
            "planning_engine": requested_engine,
            "custom_agents_used": True,
            "crewai_used": requested_engine == CREWAI_MICRO_AGENT_ENGINE,
            "toolbelt_fallback": False,
            "crewai_output_log_file": "",
            "agent_debug_dir": str((options or {}).get("album_debug_dir") or ""),
        }
    opts = _coerce_options(recovered_concept, num_tracks, track_duration, language, options)
    opts["concept"] = recovered_concept
    opts["sanitized_concept"] = recovered_concept
    opts["genre_hint"] = _album_genre_hint(opts)
    opts["agent_engine"] = requested_engine
    opts["album_writer_mode"] = str(opts.get("album_writer_mode") or ALBUM_WRITER_MODE_DEFAULT).strip() or ALBUM_WRITER_MODE_DEFAULT
    opts["max_track_repair_rounds"] = max(0, min(3, int(opts.get("max_track_repair_rounds") or ALBUM_TRACK_GATE_REPAIR_RETRIES)))
    opts["print_agent_io"] = _truthy(opts.get("print_agent_io"), ACEJAM_PRINT_AGENT_IO_DEFAULT)
    planner_settings = planner_llm_settings_from_payload(opts)
    opts.update(planner_settings)
    if input_tracks:
        opts["editable_plan_tracks"] = [dict(item) for item in input_tracks if isinstance(item, dict)]
        logs.append(
            f"Editable album plan received with {len(opts['editable_plan_tracks'])} track hint(s); "
            "the selected planner will still prompt every setting and lyrics part before render."
        )
    if requested_engine_text == "legacy_crewai":
        logs.append("legacy_crewai alias selected: using CrewAI Micro Tasks; the old large CrewAI flow is not active.")
    if not use_crewai and requested_engine == ACEJAM_AGENT_ENGINE:
        logs.append("Legacy use_crewai/toolbelt flag ignored: AceJAM Direct is selected.")
    if crewai_output_log_file and requested_engine == ACEJAM_AGENT_ENGINE:
        logs.append("Legacy CrewAI output log requested but ignored: AceJAM Direct writes agent JSONL logs.")
    elif crewai_output_log_file and requested_engine == CREWAI_MICRO_AGENT_ENGINE:
        logs.append("CrewAI Micro Tasks writes standard AceJAM debug JSONL; legacy large CrewAI logs are not used.")

    lang_name = LANG_NAMES.get(language, language)
    logs.append(f"Concept preview: {_monitor_preview(recovered_concept, 220)}")
    logs.append(f"Language: {lang_name}")
    logs.append(f"Prompt Kit: {PROMPT_KIT_VERSION}")
    logs.append(
        "Prompt Kit routing: "
        f"language_preset={language_preset(language).get('code')}; "
        f"genre_modules={','.join(module.get('slug', '') for module in infer_genre_modules(recovered_concept, max_modules=2))}."
    )
    contract = opts.get("user_album_contract") if isinstance(opts.get("user_album_contract"), dict) else {}
    repair_lines_before = len([line for line in logs if str(line).startswith("Contract repaired:")])
    if contract.get("applied"):
        locked_titles = [str(item.get("locked_title") or "").strip() for item in contract.get("tracks") or [] if str(item.get("locked_title") or "").strip()]
        logs.append(
            "Input Contract: applied; "
            f"album_title={_monitor_preview(contract.get('album_title') or 'untitled', 80)}; "
            f"locked_tracks={len(locked_titles)}; "
            f"blocked_unsafe={int(contract.get('blocked_unsafe_count') or 0)}."
        )
    logs.append(f"Tracks: {num_tracks} x {int(track_duration)}s")
    planner_provider = normalize_provider(planner_provider or opts.get("planner_lm_provider") or "ollama")
    embedding_provider = normalize_provider(embedding_provider or opts.get("embedding_lm_provider") or "ollama")
    ollama_model = str(ollama_model or opts.get("planner_model") or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL).strip()
    embedding_model = str(embedding_model or DEFAULT_ALBUM_EMBEDDING_MODEL).strip()
    logs.append(f"Local AI Writer/Planner: {provider_label(planner_provider)} ({ollama_model}) for lyrics, tags, BPM, key and captions.")
    logs.append(f"Planning Engine: {album_agent_engine_label(requested_engine)} ({requested_engine}).")
    if requested_engine == CREWAI_MICRO_AGENT_ENGINE:
        logs.append("================================================================")
        logs.append("CREWAI MICRO TASKS ENGINE ACTIVE")
        logs.append("Each track field is filled by a real CrewAI Agent/Task.")
        logs.append(f"Agents per track: Track Concept, BPM, Key, Time Signature, Duration, Sonic Tags, Section Map, Hook, Lyrics Parts, Caption Polisher, Performance, Final Payload Assembler.")
        logs.append(f"Watch the log for 'CrewAI Micro Agent call:' lines — that is each agent invoking crewai.Agent + crewai.Task.")
        logs.append("Knowledge injected per agent: ACE-Step tag library, Producer-Format Cookbook (17 entries incl. Dre G-funk + Chronic 2001 + Pete Rock + Havoc + Stoupe), Rap-Mode Cookbook, Songwriter Craft Cookbook (Eminem/2Pac/Kendrick/Nas signatures), Lyric Anti-Patterns, Worked Examples, 16-bar rap verse floor.")
        logs.append("================================================================")
    else:
        logs.append("(Tip: switch agent_engine to 'crewai_micro' to run multi-agent CrewAI flow with full visibility.)")
    logs.append("ACE-Step optional lyric/metadata LM: off for album-agent payloads; ACE-Step Audio Models render the final music.")
    logs.append(f"Album memory embedding: {provider_label(embedding_provider)} ({embedding_model}); hidden unless memory/debug is enabled and not used by the selected micro/direct director.")
    logs.append(
        "Album planner runtime: "
        f"block_retries={ACEJAM_AGENT_BLOCK_RETRIES}, "
        f"planner_preset={planner_settings.get('planner_creativity_preset')}, "
        f"temperature={planner_settings.get('planner_temperature')}, "
        f"top_p={planner_settings.get('planner_top_p')}, "
        f"top_k={planner_settings.get('planner_top_k')}, "
        f"repeat_penalty={planner_settings.get('planner_repeat_penalty')}, "
        f"max_tokens={planner_settings.get('planner_max_tokens')}, "
        f"context={planner_settings.get('planner_context_length')}, "
        f"timeout={planner_settings.get('planner_timeout')}, "
        f"print_agent_io={opts['print_agent_io']}, "
        f"planner_thinking={_truthy(opts.get('planner_thinking'), False)}."
    )
    model_info = choose_song_model(
        set(opts.get("installed_models") or []),
        str(opts.get("song_model_strategy") or "best_installed"),
        str(opts.get("requested_song_model") or "auto"),
    )
    if str(opts.get("song_model_strategy")) == "all_models_album":
        model_info = {
            "ok": True,
            "model": "per-model portfolio",
            "strategy": "all_models_album",
            "reason": "Album renders will be produced once per track for every selected ACE-Step album portfolio model.",
            "album_models": album_model_portfolio(opts.get("installed_models")),
            "multi_album": True,
        }
    if not model_info.get("ok"):
        error = str(model_info.get("error") or "No album model resolved")
        logs.append(f"ERROR: {error}")
        return {
            "tracks": [],
            "logs": logs,
            "success": False,
            "error": error,
            "planning_engine": requested_engine,
            "custom_agents_used": True,
            "crewai_used": requested_engine == CREWAI_MICRO_AGENT_ENGINE,
            "toolbelt_fallback": False,
            "crewai_output_log_file": "",
            "agent_debug_dir": str(opts.get("album_debug_dir") or ""),
        }
    try:
        return AceJamAlbumDirector(
            concept=recovered_concept,
            num_tracks=num_tracks,
            track_duration=track_duration,
            planner_model=ollama_model,
            language=language,
            opts=opts,
            planner_provider=planner_provider,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            logs=logs,
            contract=contract,
            model_info=model_info,
            repair_lines_before=repair_lines_before,
        ).run()
    except Exception as exc:
        agent_error = str(exc)
        failure_label = "AceJAM Director" if requested_engine == ACEJAM_AGENT_ENGINE else album_agent_engine_label(requested_engine)
        logs.append(f"{failure_label} planning failed loudly: {agent_error}")
        _write_album_debug_json(
            opts,
            "07_agent_failure.json",
            {
                "error": agent_error,
                "error_type": type(exc).__name__,
                "planning_engine": requested_engine,
                "director_version": ACEJAM_ALBUM_DIRECTOR_VERSION,
            },
        )
        contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
        return {
            "tracks": [],
            "logs": logs,
            "success": False,
            "planning_engine": requested_engine,
            "custom_agents_used": True,
            "crewai_used": requested_engine == CREWAI_MICRO_AGENT_ENGINE,
            "toolbelt_fallback": False,
            "crewai_error": "",
            "agent_error": agent_error,
            "error": agent_error or "AceJAM Director planning failed",
            "crewai_output_log_file": "",
            "agent_debug_dir": str(opts.get("album_debug_dir") or ""),
            "agent_rounds": [],
            "agent_repair_count": 0,
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "prompt_kit": prompt_kit_payload(),
            "toolkit": toolkit_payload(opts.get("installed_models")),
            "input_contract": contract_prompt_context(contract),
            "input_contract_applied": bool(contract.get("applied")),
            "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
            "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
            "contract_repair_count": max(0, contract_repairs),
            "toolkit_report": {"agent_error": agent_error, "director_version": ACEJAM_ALBUM_DIRECTOR_VERSION},
        }


def generate_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
    language: str = "en",
    embedding_model: str = DEFAULT_ALBUM_EMBEDDING_MODEL,
    options: dict[str, Any] | None = None,
    planner_provider: str = "ollama",
    embedding_provider: str = "ollama",
) -> dict[str, Any]:
    return plan_album(
        concept=concept,
        num_tracks=num_tracks,
        track_duration=track_duration,
        ollama_model=ollama_model,
        language=language,
        embedding_model=embedding_model,
        options=options,
        use_crewai=True,
        planner_provider=planner_provider,
        embedding_provider=embedding_provider,
    )
