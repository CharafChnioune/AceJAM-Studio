"""
Album generation using AceJAM's direct local-agent planner plus deterministic tools.

The album runtime no longer depends on CrewAI. It parses the user's album
contract, calls the selected local planner directly, runs deterministic
ACE-Step gates, and only returns tracks that are ready for audio rendering.
Legacy CrewAI constructors remain in this module for import compatibility.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Tuple

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
from album_quality_gate import evaluate_album_payload_quality
from local_llm import (
    chat_completion as local_llm_chat_completion,
    embed as local_llm_embed,
    lmstudio_api_base_url,
    lmstudio_load_model,
    lmstudio_model_catalog,
    normalize_provider,
    ollama_host,
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
    PROMPT_KIT_VERSION,
    infer_genre_modules,
    is_sparse_lyric_genre,
    kit_metadata_defaults,
    language_preset,
    prompt_kit_payload,
    section_map_for,
)
from studio_core import DEFAULT_BPM, DEFAULT_KEY_SCALE, DEFAULT_QUALITY_PROFILE, docs_best_model_settings, normalize_quality_profile
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
    "nomic-embed-text:latest",
    "mxbai-embed-large:latest",
]
CREWAI_LLM_TIMEOUT_SECONDS = int(os.environ.get("ACEJAM_CREWAI_LLM_TIMEOUT_SECONDS", "86400"))
CREWAI_EMPTY_RESPONSE_RETRIES = int(os.environ.get("ACEJAM_CREWAI_EMPTY_RESPONSE_RETRIES", "1"))
CREWAI_EMPTY_RESPONSE_RETRY_DELAY = float(os.environ.get("ACEJAM_CREWAI_EMPTY_RESPONSE_RETRY_DELAY", "8"))
CREWAI_AGENT_MAX_ITER = int(os.environ.get("ACEJAM_CREWAI_AGENT_MAX_ITER", "80"))
CREWAI_AGENT_MAX_RETRY_LIMIT = int(os.environ.get("ACEJAM_CREWAI_AGENT_MAX_RETRY_LIMIT", "8"))
CREWAI_TASK_MAX_RETRIES = int(os.environ.get("ACEJAM_CREWAI_TASK_MAX_RETRIES", "8"))
CREWAI_LLM_MAX_TOKENS = int(os.environ.get("ACEJAM_CREWAI_LLM_MAX_TOKENS", "12000"))
CREWAI_LLM_CONTEXT_WINDOW = int(os.environ.get("ACEJAM_CREWAI_LLM_CONTEXT_WINDOW", "32768"))
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
ACEJAM_AGENT_JSON_RETRIES = int(os.environ.get("ACEJAM_AGENT_JSON_RETRIES", "2"))
ACEJAM_AGENT_EMPTY_RETRIES = int(os.environ.get("ACEJAM_AGENT_EMPTY_RETRIES", "1"))
ACEJAM_AGENT_GATE_REPAIR_RETRIES = int(os.environ.get("ACEJAM_AGENT_GATE_REPAIR_RETRIES", "2"))
ACEJAM_AGENT_TEMPERATURE = float(os.environ.get("ACEJAM_AGENT_TEMPERATURE", "0.25"))
ACEJAM_AGENT_TOP_P = float(os.environ.get("ACEJAM_AGENT_TOP_P", "0.9"))
ACEJAM_AGENT_MEMORY_DEFAULT = os.environ.get("ACEJAM_AGENT_MEMORY_DEFAULT", "1").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_RETRIEVAL_TOP_K = int(os.environ.get("ACEJAM_AGENT_RETRIEVAL_TOP_K", "5"))
ACEJAM_AGENT_CONTEXT_CHUNK_CHARS = int(os.environ.get("ACEJAM_AGENT_CONTEXT_CHUNK_CHARS", "1400"))
ACEJAM_AGENT_OLLAMA_JSON_FORMAT = os.environ.get("ACEJAM_AGENT_OLLAMA_JSON_FORMAT", "0").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_SPLIT_TRACK_FLOW = os.environ.get("ACEJAM_AGENT_SPLIT_TRACK_FLOW", "1").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_MICRO_SETTINGS_FLOW = os.environ.get("ACEJAM_AGENT_MICRO_SETTINGS_FLOW", "1").lower() in {"1", "true", "yes"}
ACEJAM_AGENT_LYRIC_PARTS = max(1, int(os.environ.get("ACEJAM_AGENT_LYRIC_PARTS", "4")))

ACE_STEP_PAYLOAD_CONTRACT_VERSION = "ace-step-track-payload-contract-2026-04-29"

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
    match = re.search(r"\[(?:Intro|Verse|Pre-Chorus|Chorus|Bridge|Final Chorus|Outro|Ad-libs?)[^\]]*\]", text, flags=re.I)
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
            "required_dimensions": [
                "genre_style",
                "rhythm_groove",
                "instrumentation",
                "vocal_style",
                "mood_atmosphere",
                "arrangement_energy",
                "mix_production",
            ],
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


def preflight_album_agent_llm(planner_provider: str, planner_model: str) -> dict[str, Any]:
    provider_name = normalize_provider(planner_provider)
    model = str(planner_model or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL).strip()
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
                if (not loaded) or (CREWAI_LMSTUDIO_PIN_CONTEXT and loaded_context != CREWAI_LLM_CONTEXT_WINDOW):
                    lmstudio_load_model(
                        model,
                        kind="chat",
                        context_length=CREWAI_LLM_CONTEXT_WINDOW if CREWAI_LMSTUDIO_PIN_CONTEXT else None,
                    )
                    warnings.append(
                        f"Loaded LM Studio chat model {model} for AceJAM Agents"
                        + (f" with context_length={CREWAI_LLM_CONTEXT_WINDOW}." if CREWAI_LMSTUDIO_PIN_CONTEXT else ".")
                    )
        test = local_llm_test_model(provider_name, model, "chat")
        chat_ok = bool(test.get("success"))
        if not chat_ok:
            errors.append(str(test.get("error") or "chat preflight returned no success flag"))
    except Exception as exc:
        errors.append(f"Planner model preflight failed: {exc}")

    return {
        "ok": not errors and chat_ok,
        "planner_provider": provider_name,
        "planner_model": model,
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
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass
    sanitized = _escape_json_string_control_chars(text)
    if sanitized != text:
        try:
            parsed = json.loads(sanitized)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except json.JSONDecodeError:
            pass
    if text.lstrip().startswith(("{", "[")):
        decoder = json.JSONDecoder()
        for candidate in (text, sanitized):
            try:
                parsed, _end = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Crew result did not contain a valid JSON object")
    decoder = json.JSONDecoder()
    for candidate in (text, sanitized):
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


def _coerce_agent_lyrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize agent-friendly lyric arrays into the ACE-Step lyrics string."""
    if not isinstance(payload, dict):
        return payload
    result = dict(payload)

    def _stringify_line(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("line", "text", "tag", "section"):
                if value.get(key) not in (None, ""):
                    return str(value.get(key) or "").strip()
            return ""
        return str(value or "").strip()

    lines_value = result.get("lyrics_lines") or result.get("lyric_lines") or result.get("script_lines")
    if isinstance(lines_value, list):
        lines = [_stringify_line(line) for line in lines_value]
        lines = [line for line in lines if line]
        if lines:
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
                title = str(section.get("section") or section.get("name") or section.get("tag") or "").strip()
                if title:
                    lines.append(title if title.startswith("[") else f"[{title}]")
                section_lines = section.get("lines") or section.get("lyrics") or []
                if isinstance(section_lines, str):
                    lines.extend(line.strip() for line in section_lines.splitlines() if line.strip())
                elif isinstance(section_lines, list):
                    lines.extend(_stringify_line(line) for line in section_lines if _stringify_line(line))
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
        "[spoken word], [raspy vocal], [whispered], [falsetto], [powerful belting], [harmonies], [call and response], [ad-lib], "
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
        "Default generation profile is chart_master: use 64-step SFT/Base final-render settings, wav32 output, "
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
        goal=f"Define an original artist persona, brand identity, cadence, delivery, ad-libs, and vocal performance tags in {lang_name}",
        backstory=(
            f"{shared_rules}\n\n"
            "You develop the artist identity for this project: stage presence, vocal character, signature ad-libs, "
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
            "([Verse], [Pre-Chorus], [Chorus], [Bridge], [Final Chorus], [Ad-libs], [Outro]), "
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
            "persona, cadence, vocal character, ad-libs, delivery tags, and hook performance style. "
            f"{lang_guidance}\n"
            "Use VocalPerformanceTool and RhymeFlowTool when artist references or flow goals appear. "
            "Define: vocal type (male/female/group), vocal texture (raspy, smooth, breathy, powerful), "
            "delivery mode (rap, sing, spoken word, falsetto), ad-lib style, and hook delivery approach."
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
            "- tag_coverage confirms genre, rhythm/groove, instrumentation, vocal style, mood, arrangement energy, and mix/production\n"
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
        "Album Bible Agent": 4500,
        "Track Settings Agent": 2600,
        "Track BPM Agent": 700,
        "Track Key Agent": 700,
        "Track Time Signature Agent": 700,
        "Track Duration Agent": 700,
        "Track Language Agent": 900,
        "Track Tag List Agent": 1800,
        "Track Caption Agent": 1600,
        "Track Description Agent": 1500,
        "Track Hook Agent": 1200,
        "Track Performance Agent": 1400,
        "Track Writer Agent": 12000,
        "Track Finalizer Agent": 9000,
        "Quality Repair Agent": 10000,
        "Track Lyric Continuation Agent": 3200,
    }
    if str(agent_name or "").startswith("Track Lyrics Agent"):
        return max(1200, int(os.environ.get("ACEJAM_AGENT_MAX_TOKENS_TRACK_LYRICS_AGENT", "3600")))
    env_key = "ACEJAM_AGENT_MAX_TOKENS_" + re.sub(r"[^A-Z0-9]+", "_", str(agent_name or "agent").upper()).strip("_")
    fallback = CREWAI_LMSTUDIO_MAX_TOKENS if provider_name == "lmstudio" else CREWAI_LLM_NUM_PREDICT
    return max(512, int(os.environ.get(env_key, defaults.get(str(agent_name), min(4500, int(fallback))))))


def _agent_llm_options(provider: str, agent_name: str = "") -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    completion_cap = _agent_completion_cap(agent_name, provider_name)
    options: dict[str, Any] = {
        "temperature": ACEJAM_AGENT_TEMPERATURE,
        "top_p": ACEJAM_AGENT_TOP_P,
    }
    if provider_name == "ollama":
        options["num_ctx"] = CREWAI_LLM_CONTEXT_WINDOW
        options["num_predict"] = completion_cap
    else:
        options["max_tokens"] = completion_cap
    return options


def _agent_json_instruction(schema_name: str) -> str:
    return (
        f"Return strict JSON only for {schema_name}. No markdown fences, no commentary, no thoughts. "
        "If you need to repair, output the corrected JSON object directly. "
        "Do not rename locked user titles, producers, BPM, style, vibe, or narrative. "
        "For long lyrics, prefer a lyrics_lines array with one section tag or lyric line per item; "
        "do not place raw line breaks inside JSON string values."
    )


def _agent_system_prompt(agent_name: str) -> str:
    return (
        f"You are {agent_name}, part of AceJAM's custom ACE-Step album agent system.\n"
        "You are ACE-Step Multilingual Hit Architect for album tracks: prompt engineer, lyric editor, topline writer, "
        "arranger, and payload finalizer for ACE-Step 1.5 text-to-music.\n\n"
        "ABSOLUTE SOURCE OF TRUTH:\n"
        "- The user's album contract is hard metadata. Do not rename, reorder, translate, replace, or reinterpret locked "
        "album title, track title, producer credit, BPM, key, style, vibe, narrative, required phrases, or language.\n"
        "- If a locked field conflicts with your taste, keep the locked field and make the rest of the payload support it.\n"
        "- Real producer/artist names are credits or broad technique labels only; do not imitate living artists directly.\n\n"
        "ACE-STEP NON-NEGOTIABLES:\n"
        "- caption / tags: global sound only. Max 512 chars. Use compact comma-separated production traits: genre, groove, "
        "instruments, vocal type, mood, arrangement energy, mix. No lyrics, no section tags, no BPM/key/duration/model/seed, "
        "no JSON, no prompt prose, no album story paragraphs.\n"
        "- lyrics: temporal script only. Max 4096 chars. Use concise section tags and actual performable lyric lines. "
        "No metadata blocks, no analysis, no markdown commentary, no placeholders, no escaped literal \\n.\n"
        "- Metadata lives in fields only: bpm, key_scale, time_signature, duration, language/vocal_language.\n"
        "- Caption and lyrics must agree. Do not stack many genres; choose one primary and one secondary at most.\n"
        "- Use short rap-able or singable lines: normally 3-8 words per line. Split overlong bars at breath points.\n"
        "- Full 210-270s vocal tracks need real coverage: intro, verses, pre/hook/chorus, bridge or break, final chorus, outro. "
        "Do not produce short demo lyrics for a full song.\n\n"
        "HIT-WRITING GATES:\n"
        "- Every song needs one central emotional promise and one coherent image world.\n"
        "- Verses add new concrete information; chorus simplifies and intensifies the hook.\n"
        "- Rap requires cadence, breath control, internal rhyme, and bar momentum; not prose chopped into lines.\n"
        "- Hooks must be memorable after one listen, vowel-friendly, and repeatable.\n"
        "- Remove generic AI filler, mixed metaphors, empty slogans, and robotic repair lines. Never append filler just to hit a count.\n\n"
        "OUTPUT DISCIPLINE:\n"
        "- Return strict JSON only, no markdown fences, no commentary, no thoughts.\n"
        "- Prefer lyrics_lines as an array with one section tag or lyric line per item; AceJAM will join it.\n"
        "- Fill counters honestly: lyrics_word_count, lyrics_line_count, lyrics_char_count, section_count, hook_count, "
        "caption_dimensions_covered.\n"
        "- Do not output ACE-Step runtime LM switches such as ace_lm_model, use_format, thinking, or use_cot_*; AceJAM controls those.\n"
        "- Reject and repair your own output before final JSON if the caption leaks metadata/prose, the hook is weak, "
        "lyrics are short, or tag dimensions are missing."
    )


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
    if provider_name == "lmstudio" and CREWAI_LMSTUDIO_DISABLE_THINKING and CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE:
        user_content = f"{CREWAI_LMSTUDIO_NO_THINK_DIRECTIVE}\n\n{user_content}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    _append_album_debug_jsonl(
        debug_options,
        "03_agent_prompts.jsonl",
        {
            "agent": agent_name,
            "provider": provider_name,
            "model": model_name,
            "attempt": attempt,
            "messages": messages,
            "options": options,
        },
    )
    logs.append(f"AceJAM Agent call: {agent_name} attempt {attempt} via {provider_label(provider_name)}.")
    started = time.perf_counter()
    raw = local_llm_chat_completion(provider_name, model_name, messages, options=options, json_format=json_format)
    elapsed = round(time.perf_counter() - started, 3)
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
            "response": text,
        },
    )
    logs.append(f"AceJAM Agent response: {agent_name} {len(text)} chars in {elapsed}s.")
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
    retries = ACEJAM_AGENT_JSON_RETRIES if max_retries is None else int(max_retries)
    attempts = max(1, retries + ACEJAM_AGENT_EMPTY_RETRIES + 1)
    options = _agent_llm_options(provider, agent_name)
    system_prompt = _agent_system_prompt(agent_name)
    if extra_system:
        system_prompt += "\n" + str(extra_system).strip()
    system_prompt += "\n" + _agent_json_instruction(schema_name)
    prompt = user_prompt
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            use_json_format = normalize_provider(provider) != "ollama" or ACEJAM_AGENT_OLLAMA_JSON_FORMAT
            if normalize_provider(provider) == "ollama" and attempt > 1 and last_error == "empty response":
                use_json_format = False
                logs.append(f"AceJAM Agent retry: {agent_name} without Ollama JSON transport mode after empty response.")
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
                json_format=use_json_format,
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
                "Return the requested strict JSON object now."
            )
            continue
        if not raw.strip():
            last_error = "empty response"
            if attempt >= attempts:
                break
            logs.append(f"AceJAM Agent empty response: {agent_name}; retrying without tool-loop assumptions.")
            prompt = (
                f"{user_prompt}\n\nRECOVERY: Your previous response was empty. "
                "Return the requested strict JSON object only. Do not call tools."
            )
            continue
        try:
            payload = _json_object_from_text(raw)
            if not isinstance(payload, dict):
                raise ValueError("JSON root was not an object")
            return _coerce_agent_lyrics_payload(payload)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _append_album_debug_jsonl(
                debug_options,
                "04_agent_responses.jsonl",
                {
                    "agent": agent_name,
                    "attempt": attempt,
                    "parse_error": last_error,
                    "response_preview": _monitor_preview(raw, 500),
                },
            )
            if attempt >= attempts:
                break
            logs.append(f"AceJAM Agent JSON repair: {agent_name}; {last_error}.")
            prompt = (
                f"{user_prompt}\n\nJSON REPAIR: The previous response failed to parse as strict JSON: {last_error}. "
                "Return exactly one valid JSON object matching the requested schema. No markdown. "
                "If lyrics are long, use lyrics_lines as an array of strings instead of one multiline JSON string."
            )
    raise AceJamAgentError(f"{agent_name} failed to produce valid JSON after {attempts} attempt(s): {last_error}")


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


def _set_track_stats(track: dict[str, Any]) -> dict[str, Any]:
    stats = lyric_stats(str(track.get("lyrics") or ""))
    track["lyrics_word_count"] = int(stats.get("word_count") or 0)
    track["lyrics_line_count"] = int(stats.get("line_count") or 0)
    track["lyrics_char_count"] = int(stats.get("char_count") or 0)
    track["section_count"] = int(stats.get("section_count") or 0)
    track["hook_count"] = sum(
        1 for section in stats.get("sections") or [] if re.search(r"chorus|hook|refrain", str(section), re.I)
    )
    return track


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
    return text if text.startswith("[") else f"[{text.strip('[]')}]"


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


def _quality_repair_prompt(
    *,
    concept: str,
    album_bible: dict[str, Any],
    blueprint: dict[str, Any],
    payload: dict[str, Any],
    report: dict[str, Any],
    track_prompt_template: str,
    lyric_plan: dict[str, Any] | None = None,
    index: int,
    total: int,
    retrieved_context: str = "",
) -> str:
    compact_contract = _compact_track_agent_contract(blueprint, lyric_plan or {}, include_schema=True)
    return (
        "You are the AceJAM Quality Repair Agent. Repair the SAME track; do not invent a new song.\n"
        f"TRACK COUNTER: you are repairing track {index + 1} of {total}.\n\n"
        f"FULL_ORIGINAL_ALBUM_PROMPT_EXCERPT:\n{_clip_text(concept, 4200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"CURRENT_TRACK_JSON:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        f"QUALITY_GATE_REPORT:\n{json.dumps(_public_gate_report(report), ensure_ascii=True, indent=2)}\n\n"
        f"ACE_STEP_TRACK_CONTRACT_COMPACT:\n{json.dumps(compact_contract, ensure_ascii=True, indent=2)}\n\n"
        "The full resolved ACE-Step prompt template is stored in the local debug log; use this compact contract for the response.\n\n"
        "Return one repaired final track JSON object. Preserve locked fields exactly. "
        "Fix every blocking issue: enough lyric lines/words/sections, hook, caption dimensions, no leakage, no placeholders. "
        "Prefer lyrics_lines as an array so long lyrics cannot truncate or break JSON."
    )


def _gate_agent_track(
    *,
    track: dict[str, Any],
    blueprint: dict[str, Any],
    album_bible: dict[str, Any],
    concept: str,
    opts: dict[str, Any],
    contract: dict[str, Any],
    index: int,
    total: int,
    planner_provider: str,
    planner_model: str,
    logs: list[str],
    track_prompt_template: str,
    agent_stats: dict[str, Any],
    retrieved_context: str = "",
) -> dict[str, Any]:
    current = apply_user_album_contract_to_track(track, contract, index, logs)
    for repair_index in range(ACEJAM_AGENT_GATE_REPAIR_RETRIES + 1):
        current = _set_track_stats(dict(current))
        if not current.get("caption") and current.get("tags"):
            current["caption"] = current.get("tags")
        report = evaluate_album_payload_quality(current, options=_agent_gate_options(opts, current), repair=True)
        _append_album_debug_jsonl(
            opts,
            "05_track_gate_reports.jsonl",
            {
                "track_number": current.get("track_number") or index + 1,
                "title": current.get("title") or blueprint.get("title"),
                "attempt": repair_index + 1,
                "status": report.get("status"),
                "gate_passed": bool(report.get("gate_passed")),
                "report": report,
            },
        )
        repaired = dict(report.get("repaired_payload") or current)
        repaired = apply_user_album_contract_to_track(repaired, contract, index, logs)
        repaired = _set_track_stats(repaired)
        if report.get("gate_passed"):
            status = str(report.get("status") or "pass")
            if status == "auto_repair":
                agent_stats["agent_repair_count"] = int(agent_stats.get("agent_repair_count") or 0) + 1
                logs.append(f"AceJAM payload gate auto-repaired track {index + 1}: {_monitor_preview(repaired.get('title'), 90)}")
            public_report = _public_gate_report(report)
            repaired["payload_gate_status"] = status
            repaired["payload_quality_gate"] = public_report
            repaired["tag_coverage"] = report.get("tag_coverage") or {}
            repaired["caption_integrity"] = report.get("caption_integrity") or {}
            repaired["lyric_duration_fit"] = report.get("lyric_duration_fit") or {}
            repaired["repair_actions"] = report.get("repair_actions") or repaired.get("repair_actions") or []
            tool_report = dict(repaired.get("tool_report") or {})
            tool_report.update(
                {
                    "payload_quality_gate": public_report,
                    "payload_gate_status": status,
                    "tag_coverage": repaired["tag_coverage"],
                    "caption_integrity": repaired["caption_integrity"],
                    "lyric_duration_fit": repaired["lyric_duration_fit"],
                }
            )
            repaired["tool_report"] = tool_report
            return repaired
        if repair_index >= ACEJAM_AGENT_GATE_REPAIR_RETRIES:
            reasons = "; ".join(
                f"{issue.get('id')}: {issue.get('detail')}"
                for issue in (report.get("blocking_issues") or report.get("issues") or [])[:8]
            )
            raise AceJamAgentError(f"AlbumPayloadQualityGate failed for track {index + 1}: {reasons or report.get('status')}")
        agent_stats["agent_repair_count"] = int(agent_stats.get("agent_repair_count") or 0) + 1
        logs.append(f"Quality Repair Agent: track {index + 1} needs repair ({_monitor_preview(_track_gate_retry_message(report), 260)}).")
        repair_payload = _agent_json_call(
            agent_name="Quality Repair Agent",
            provider=planner_provider,
            model_name=planner_model,
            user_prompt=_quality_repair_prompt(
                concept=concept,
                album_bible=album_bible,
                blueprint=blueprint,
                payload=repaired,
                report=report,
                track_prompt_template=track_prompt_template,
                lyric_plan=(report.get("lyric_duration_fit") or {}).get("plan") or {},
                index=index,
                total=total,
                retrieved_context=retrieved_context,
            ),
            logs=logs,
            debug_options=opts,
            schema_name="repaired_track_payload",
            extra_system="Repair only the concrete gate failures. Keep the same title and production brief.",
        )
        agent_stats.setdefault("agent_rounds", []).append({
            "agent": "Quality Repair Agent",
            "track_number": index + 1,
            "status": "completed",
            "repair_attempt": repair_index + 1,
        })
        current = {**repaired, **repair_payload}
    raise AceJamAgentError(f"AlbumPayloadQualityGate failed for track {index + 1}")


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
    context = _compact_agent_tool_context(opts, track_duration, num_tracks, concept)
    return (
        "You are the AceJAM Album Bible Agent. Build compact album-level creative DNA for an ACE-Step album.\n"
        "Do not write full lyrics in this stage. Do not decide the final track count; AceJAM builds an exact N-track scaffold deterministically. "
        "You may return optional track blueprint hints, but missing hints are fine and must not stop the album.\n"
        "Preserve user locked titles/order/producers/BPM/style/vibe/narrative when you mention them.\n\n"
        f"FULL_ORIGINAL_ALBUM_PROMPT_EXCERPT:\n{_clip_text(concept, 5200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"USER_ALBUM_CONTRACT:\n{json.dumps(contract_prompt_context(contract), ensure_ascii=True, indent=2)}\n\n"
        f"ALBUM_TOOL_CONTEXT:\n{json.dumps(context, ensure_ascii=True, indent=2)}\n\n"
        f"FULL_TAG_LIBRARY_COMPACT:\n{json.dumps(_agent_tag_library_summary(), ensure_ascii=True, indent=2)}\n\n"
        f"MODEL_ADVICE:\n{json.dumps(model_info, ensure_ascii=True, indent=2)}\n\n"
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
        "caption_dimensions": [
            "genre_style",
            "rhythm_groove",
            "instrumentation",
            "vocal_style",
            "mood_atmosphere",
            "arrangement_energy",
            "mix_production",
        ],
        "tag_examples": {
            "genre_style": TAG_TAXONOMY.get("genre_style", [])[:14],
            "rhythm_groove": TAG_TAXONOMY.get("speed_rhythm", [])[:10] + ["boom-bap drums", "trap hi-hats", "handclaps"],
            "instrumentation": TAG_TAXONOMY.get("instruments", [])[:18],
            "vocal_style": TAG_TAXONOMY.get("vocal_character", [])[:12],
            "mood_atmosphere": TAG_TAXONOMY.get("mood_atmosphere", [])[:14],
            "arrangement_energy": TAG_TAXONOMY.get("structure_hints", [])[:10],
            "mix_production": TAG_TAXONOMY.get("production_style", [])[:12],
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
    for index in range(max(0, int(num_tracks or 0))):
        track_number = index + 1
        role = _album_arc_role(index, num_tracks)
        hint = _hint_by_track_number(hints, track_number)
        hint_duration = hint.get("duration")
        if hint_duration not in (None, "", []) and parse_duration_seconds(hint_duration, requested_duration) != requested_duration:
            logs.append(
                f"Ignored agent duration hint for track {track_number}: "
                f"{parse_duration_seconds(hint_duration, requested_duration)}s; job duration is {requested_duration}s."
            )
        slot: dict[str, Any] = {
            "track_number": track_number,
            "title": "",
            "duration": requested_duration,
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
            slot["duration"] = requested_duration
            slot["scaffold_source"] = "bible_hint"
        editable_hint = _hint_by_track_number(editable_hints, track_number)
        if editable_hint:
            editable_duration = editable_hint.get("duration")
            if editable_duration not in (None, "", []) and parse_duration_seconds(editable_duration, requested_duration) != requested_duration:
                logs.append(
                    f"Ignored editable duration hint for track {track_number}: "
                    f"{parse_duration_seconds(editable_duration, requested_duration)}s; job duration is {requested_duration}s."
                )
            for key, value in editable_hint.items():
                if key in {"lyrics", "lyrics_lines"}:
                    continue
                if value not in (None, "", []):
                    slot[key] = value
            slot["duration"] = requested_duration
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
    payload_duration = payload.get("duration")
    if payload_duration not in (None, "", []) and parse_duration_seconds(payload_duration, scaffold_duration) != scaffold_duration:
        logs.append(
            f"Ignored blueprint duration hint for track {merged['track_number']}: "
            f"{parse_duration_seconds(payload_duration, scaffold_duration)}s; scaffold duration is {scaffold_duration}s."
        )
    merged["duration"] = scaffold_duration
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
        f"FULL_ORIGINAL_ALBUM_PROMPT_EXCERPT:\n{_clip_text(concept, 4200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"USER_ALBUM_CONTRACT:\n{json.dumps(contract_prompt_context(contract), ensure_ascii=True, indent=2)}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"SCAFFOLD_SLOT:\n{json.dumps(_compact_blueprint_for_agent(scaffold), ensure_ascii=True, indent=2)}\n\n"
        f"FULL_TAG_LIBRARY_COMPACT:\n{json.dumps(_agent_tag_library_summary(), ensure_ascii=True, indent=2)}\n\n"
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
        f"FULL_ORIGINAL_ALBUM_PROMPT_EXCERPT:\n{_clip_text(concept, 4200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
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
        f"FULL_ORIGINAL_ALBUM_PROMPT_EXCERPT:\n{_clip_text(concept, 4200)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"WRITER_OUTPUT_JSON:\n{json.dumps(writer_payload, ensure_ascii=True, indent=2)}\n\n"
        f"ACE_STEP_TRACK_CONTRACT_COMPACT:\n{json.dumps(compact_contract, ensure_ascii=True, indent=2)}\n\n"
        "The full resolved ACE-Step prompt template is stored in the local debug log; use this compact contract for the response.\n"
        "Return the final track JSON with all required metadata and full lyrics. "
        "Prefer lyrics_lines for the full script, one line per array item. "
        "Preserve locked fields exactly. Caption must be sound tags only. Lyrics must be actual song lines only."
    )


def _track_source_evidence(blueprint: dict[str, Any], limit: int = 1600) -> str:
    fields = {
        "track_number": blueprint.get("track_number"),
        "title": blueprint.get("title") or blueprint.get("locked_title"),
        "producer_credit": blueprint.get("producer_credit"),
        "bpm": blueprint.get("bpm"),
        "key_scale": blueprint.get("key_scale"),
        "style": blueprint.get("style"),
        "vibe": blueprint.get("vibe"),
        "narrative": blueprint.get("narrative"),
        "description": blueprint.get("description"),
        "required_phrases": blueprint.get("required_phrases") or [],
        "source_excerpt": blueprint.get("source_excerpt") or "",
    }
    return _clip_text(json.dumps(_debug_jsonable(fields), ensure_ascii=False, indent=2), limit)


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
    track_evidence = _track_source_evidence(blueprint)
    return (
        "You are the AceJAM Track Settings Agent. Create ONLY the ACE-Step sound/settings package for this one track.\n"
        f"TRACK COUNTER: settings for track {index + 1} of {total}. No lyrics in this response.\n\n"
        f"TRACK_SOURCE_EVIDENCE_ONLY:\n{track_evidence}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"LYRIC_LENGTH_PLAN_FOR_CONTEXT_ONLY:\n{json.dumps({key: lyric_plan.get(key) for key in ('duration', 'sections', 'target_words', 'min_words', 'target_lines', 'min_lines')}, ensure_ascii=True, indent=2)}\n\n"
        f"FULL_TAG_LIBRARY_COMPACT:\n{json.dumps(_agent_tag_library_summary(), ensure_ascii=True, indent=2)}\n\n"
        "CAPTION RULE: caption is only comma-separated sound traits. It must cover: genre/style, rhythm/groove, "
        "instrumentation, vocal style, mood/atmosphere, arrangement energy, mix/production. "
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
                "caption_dimensions_covered": [
                    "genre_style",
                    "rhythm_groove",
                    "instrumentation",
                    "vocal_style",
                    "mood_atmosphere",
                    "arrangement_energy",
                    "mix_production",
                ],
            },
            "instruction": (
                "Choose only compact sonic tags. Cover every caption dimension: genre/style, rhythm/groove, "
                "instrumentation, vocal style, mood/atmosphere, arrangement energy, and mix/production. "
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
    tag_library = (
        f"\nFULL_TAG_LIBRARY_COMPACT:\n{json.dumps(_agent_tag_library_summary(), ensure_ascii=True, indent=2)}\n"
        if spec.get("include_tag_library")
        else ""
    )
    return (
        f"You are {spec.get('agent')}. Decide exactly ONE micro-setting for an ACE-Step album track.\n"
        f"TRACK COUNTER: track {index + 1} of {total}. MICRO_SETTING: {field}.\n"
        "Do not plan the whole song. Do not write lyrics. Do not output extra fields beyond the schema.\n\n"
        f"TRACK_SOURCE_EVIDENCE_ONLY:\n{_track_source_evidence(blueprint)}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"PRIOR_MICRO_SETTINGS:\n{json.dumps(_debug_jsonable(prior_settings), ensure_ascii=True, indent=2)}\n\n"
        f"LYRIC_PLAN_CONTEXT:\n{json.dumps({key: lyric_plan.get(key) for key in ('duration', 'target_words', 'min_words', 'target_lines', 'min_lines')}, ensure_ascii=True, indent=2)}\n"
        f"{tag_library}\n"
        f"INSTRUCTION:\n{spec.get('instruction')}\n\n"
        f"OUTPUT_SCHEMA:\n{json.dumps(schema, ensure_ascii=True)}\n"
        "Return strict JSON only."
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
            max_retries=1,
        )
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
    section_floor = max(2, len(sections) * 3)
    return {
        "target_words": max(12, int(round(target_words / group_count))),
        "min_words": max(0, int(round(min_words / group_count))),
        "target_lines": max(section_floor, int(round(target_lines / group_count))),
        "min_lines": max(len(sections) * 2, int(round(min_lines / group_count))),
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
    track_evidence = _track_source_evidence(blueprint)
    return (
        "You are the AceJAM Track Lyrics Agent. Write ONLY this small lyric part; do not output settings/caption.\n"
        f"TRACK COUNTER: track {track_index + 1} of {total_tracks}. LYRIC PART: {part_index + 1} of {part_count}.\n\n"
        f"TRACK_SOURCE_EVIDENCE_ONLY:\n{track_evidence}\n\n"
        f"RETRIEVED_CONTEXT_CHUNKS:\n{retrieved_context or '[]'}\n\n"
        f"ALBUM_BIBLE_SUMMARY:\n{json.dumps(_compact_album_bible_for_agent(album_bible), ensure_ascii=True, indent=2)}\n\n"
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"TRACK_SETTINGS_CONTEXT:\n{json.dumps({key: settings_payload.get(key) for key in ('caption', 'genre_profile', 'hook_promise', 'performance_brief', 'language', 'vocal_language')}, ensure_ascii=True, indent=2)}\n\n"
        f"WRITE_THESE_SECTIONS_ONLY:\n{json.dumps([f'[{section}]' for section in section_group], ensure_ascii=True)}\n\n"
        f"PART_TARGETS:\n{json.dumps(targets, ensure_ascii=True, indent=2)}\n\n"
        f"REQUIRED_PHRASES_FOR_THIS_PART:\n{json.dumps(required_phrases, ensure_ascii=False, indent=2)}\n\n"
        f"PREVIOUS_LYRIC_PARTS_CONTEXT:\n{json.dumps(previous_brief, ensure_ascii=True, indent=2)}\n\n"
        f"LANGUAGE: {language}. Use the correct script and natural rhythm for this language.\n\n"
        "LYRIC RULES:\n"
        "- Start every requested section with its bracket tag.\n"
        "- Write actual performable lyric lines only: 3-8 words per line where possible.\n"
        "- Rap lines need breath-control, cadence, internal rhyme, and bar momentum.\n"
        "- Hooks/choruses must be short, repeatable, and connected to the title/hook promise.\n"
        "- Include every REQUIRED_PHRASE exactly if provided for this part.\n"
        "- No caption, no metadata, no BPM/key/duration, no prose explanation, no placeholders, no markdown.\n\n"
        "OUTPUT_SCHEMA:\n"
        '{"part_index":1,"sections":[],"lyrics_lines":[],"required_phrases_used":[],"hook_lines":[],'
        '"word_count":0,"line_count":0,"quality_checks":{"short_lines":true,"no_placeholders":true}}\n'
        "Return strict JSON only."
    )


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
        f"LOCKED_TRACK_BLUEPRINT:\n{json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True, indent=2)}\n\n"
        f"TRACK_SETTINGS_CONTEXT:\n{json.dumps({key: settings_payload.get(key) for key in ('caption', 'hook_promise', 'performance_brief', 'language')}, ensure_ascii=True, indent=2)}\n\n"
        f"LYRIC_PLAN:\n{json.dumps({key: lyric_plan.get(key) for key in ('sections', 'target_words', 'min_words', 'target_lines', 'min_lines', 'max_lyrics_chars')}, ensure_ascii=True, indent=2)}\n\n"
        f"CURRENT_STATS:\n{json.dumps({'word_count': stats.get('word_count'), 'line_count': stats.get('line_count'), 'missing_words': missing_words, 'missing_lines': missing_lines}, ensure_ascii=True, indent=2)}\n\n"
        f"CURRENT_LYRICS_TAIL:\n{_clip_text(current_tail, 1800)}\n\n"
        f"LANGUAGE: {language}\n\n"
        "Write a natural bridge/final chorus/outro continuation that closes the song. "
        "Use short performable lines, no filler, no metadata, no caption, no placeholders. "
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


def _plan_album_with_acejam_agents(
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
) -> dict[str, Any]:
    opts = {
        **dict(opts or {}),
        "agent_engine": ACEJAM_AGENT_ENGINE,
        "strict_album_agents": True,
        "disable_auto_lyric_expansion": True,
    }
    agent_stats: dict[str, Any] = {"agent_rounds": [], "agent_repair_count": 0}
    agent_debug_dir = str(opts.get("album_debug_dir") or "")
    logs.append(f"Agent engine: AceJAM Agents ({ACEJAM_AGENT_ENGINE}).")
    if agent_debug_dir:
        logs.append(f"Agent debug log dir: {agent_debug_dir}")
        logs.append(f"Agent raw prompts JSONL: {Path(agent_debug_dir) / '03_agent_prompts.jsonl'}")
        logs.append(f"Agent raw responses JSONL: {Path(agent_debug_dir) / '04_agent_responses.jsonl'}")
        logs.append(f"Agent gate reports JSONL: {Path(agent_debug_dir) / '05_track_gate_reports.jsonl'}")
    logs.append("AceJAM Agents preflight starting.")
    preflight = preflight_album_local_llm(planner_provider, planner_model, embedding_provider, embedding_model)
    logs.append(
        f"{provider_label(planner_provider)} preflight: planner chat={preflight['chat_ok']}; "
        f"{provider_label(embedding_provider)} embedding={preflight.get('embed_ok')}."
    )
    for warning in preflight.get("warnings") or []:
        logs.append(f"Local LLM preflight warning: {warning}")
    if not preflight.get("chat_ok"):
        raise AceJamAgentError("; ".join(preflight.get("errors") or ["planner preflight failed"]))
    selected_embedding_model = str(preflight.get("embedding_model") or embedding_model or "").strip()
    memory_requested = _agent_memory_requested(opts)
    memory_enabled = bool(memory_requested and preflight.get("embed_ok") and selected_embedding_model)
    if memory_requested and not memory_enabled:
        logs.append(
            "Agent memory: off; embedding preflight failed or no embedding model was selected. "
            "Scaffolded planning will continue without retrieval."
        )
        for error in preflight.get("errors") or []:
            if "Embedding" in str(error) or "embedding" in str(error):
                logs.append(f"Agent memory warning: {error}")
    else:
        logs.append(f"Agent memory: {'on' if memory_enabled else 'off'}; job-scoped context store.")
    context_store = AlbumContextStore(
        options=opts,
        provider=embedding_provider,
        model=selected_embedding_model,
        enabled=memory_enabled,
        logs=logs,
    )
    context_store.add("original_prompt", concept, {"source": "user_prompt"})
    context_store.add("user_album_contract", json.dumps(contract_prompt_context(contract), ensure_ascii=True), {"source": "parsed_contract"})
    for item in contract.get("tracks") or []:
        context_store.add(
            "contract_track",
            json.dumps(item, ensure_ascii=True),
            {"track_number": item.get("track_number"), "title": item.get("locked_title")},
        )
    editable_plan_tracks = [item for item in (opts.get("editable_plan_tracks") or []) if isinstance(item, dict)]
    if editable_plan_tracks:
        logs.append(
            f"Editable plan scaffold: {len(editable_plan_tracks)} track(s) will be used as hints, "
            "not as final ACE-Step payloads."
        )
        for item in editable_plan_tracks:
            context_store.add(
                "editable_plan_track",
                json.dumps(_compact_blueprint_for_agent(item), ensure_ascii=True),
                {"track_number": item.get("track_number"), "title": item.get("title")},
            )

    _write_album_debug_json(
        opts,
        "02_contract.json",
        {
            "user_album_contract": contract,
            "input_contract_applied": bool(contract.get("applied")),
            "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
            "planning_engine": ACEJAM_AGENT_ENGINE,
            "memory_enabled": memory_enabled,
            "embedding_provider": normalize_provider(embedding_provider),
            "embedding_model": selected_embedding_model,
            "editable_plan_tracks": editable_plan_tracks,
        },
    )

    logs.append(f"Planning album bible with AceJAM Agents and {provider_label(planner_provider)} model {planner_model}...")
    bible_context = context_store.block(
        f"album bible {contract.get('album_title') or ''} {language} {num_tracks} tracks",
        label="album_bible",
    )
    bible_agent_error = ""
    try:
        bible_payload = _agent_json_call(
            agent_name="Album Bible Agent",
            provider=planner_provider,
            model_name=planner_model,
            user_prompt=_album_bible_agent_prompt(
                concept=concept,
                num_tracks=num_tracks,
                track_duration=track_duration,
                language=language,
                opts=opts,
                model_info=model_info,
                contract=contract,
                retrieved_context=bible_context,
            ),
            logs=logs,
            debug_options=opts,
            schema_name="album_bible_payload",
            extra_system="Do not write lyrics in the bible stage. Tracks are blueprints only.",
        )
        agent_stats["agent_rounds"].append({"agent": "Album Bible Agent", "status": "completed"})
    except Exception as exc:
        bible_agent_error = f"{type(exc).__name__}: {exc}"
        logs.append(
            "Album Bible Agent failed explicitly; continuing with deterministic album-bible scaffold "
            f"because the N-track scaffold is authoritative. Error: {_monitor_preview(bible_agent_error, 220)}"
        )
        agent_stats["agent_rounds"].append({
            "agent": "Album Bible Agent",
            "status": "failed_optional",
            "error": bible_agent_error,
        })
        bible_payload = {
            "album_bible": _deterministic_album_bible(concept, contract, language, num_tracks),
            "tracks": [],
            "album_bible_agent_error": bible_agent_error,
        }
    album_bible = bible_payload.get("album_bible") if isinstance(bible_payload.get("album_bible"), dict) else {}
    if not album_bible:
        album_bible = _deterministic_album_bible(concept, contract, language, num_tracks)
    context_store.add("album_bible", json.dumps(album_bible, ensure_ascii=True), {"source": "Album Bible Agent"})
    hint_count = len([item for item in (bible_payload.get("tracks") or []) if isinstance(item, dict)])
    if hint_count != num_tracks:
        logs.append(f"Bible returned {hint_count} optional blueprint hint(s); scaffold requires {num_tracks}.")
    scaffold = _build_album_track_scaffold(
        concept=concept,
        num_tracks=num_tracks,
        track_duration=track_duration,
        language=language,
        opts=opts,
        contract=contract,
        bible_payload=bible_payload,
        logs=logs,
    )
    _write_album_debug_json(opts, "04_album_bible.json", {"album_bible": album_bible, "optional_hints": bible_payload.get("tracks") or [], "scaffold": scaffold})
    blueprints: list[dict[str, Any]] = []
    for index, slot in enumerate(scaffold):
        title = str(slot.get("title") or f"Track {index + 1}")
        blueprint_context = context_store.block(
            f"track {index + 1} of {num_tracks} {title} {slot.get('style') or ''} {slot.get('vibe') or ''}",
            label=f"track_{index + 1}_blueprint",
        )
        logs.append(f"Planning track blueprint {index + 1}/{num_tracks}: {_monitor_preview(title, 90)}")
        try:
            blueprint_payload = _agent_json_call(
                agent_name="Track Blueprint Agent",
                provider=planner_provider,
                model_name=planner_model,
                user_prompt=_track_blueprint_prompt(
                    concept=concept,
                    album_bible=album_bible,
                    scaffold=slot,
                    contract=contract,
                    language=language,
                    index=index,
                    total=num_tracks,
                    retrieved_context=blueprint_context,
                ),
                logs=logs,
                debug_options=opts,
                schema_name="track_blueprint_payload",
                extra_system="Plan metadata only. Do not write lyrics. Preserve locked fields exactly.",
            )
            agent_stats["agent_rounds"].append({"agent": "Track Blueprint Agent", "track_number": index + 1, "status": "completed"})
        except Exception as exc:
            blueprint_error = f"{type(exc).__name__}: {exc}"
            logs.append(
                f"Track Blueprint Agent failed explicitly for track {index + 1}; "
                f"using deterministic scaffold slot. Error: {_monitor_preview(blueprint_error, 220)}"
            )
            agent_stats["agent_rounds"].append({
                "agent": "Track Blueprint Agent",
                "track_number": index + 1,
                "status": "failed_optional",
                "error": blueprint_error,
            })
            blueprint_payload = {
                **dict(slot),
                "track_blueprint_agent_error": blueprint_error,
            }
        blueprint = _merge_blueprint_payload(slot, blueprint_payload, contract, index, logs)
        blueprint["track_number"] = int(blueprint.get("track_number") or index + 1)
        blueprint["duration"] = parse_duration_seconds(blueprint.get("duration") or track_duration, track_duration)
        context_store.add("track_blueprint", json.dumps(_compact_blueprint_for_agent(blueprint), ensure_ascii=True), {
            "track_number": index + 1,
            "title": blueprint.get("title"),
        })
        blueprints.append(blueprint)
    blueprints = normalize_album_tracks(blueprints, opts)
    blueprints = apply_user_album_contract_to_tracks(blueprints, contract, logs)
    logs.append(f"AceJAM Agents planned {len(blueprints)} scaffolded track blueprint(s).")

    produced_tracks: list[dict[str, Any]] = []
    for index, blueprint in enumerate(blueprints):
        title = str(blueprint.get("title") or f"Track {index + 1}")
        duration = parse_duration_seconds(blueprint.get("duration") or track_duration, track_duration)
        density = str(opts.get("lyric_density") or "dense")
        structure_preset = str(opts.get("structure_preset") or "auto")
        lyric_plan = lyric_length_plan(
            duration,
            density,
            structure_preset,
            " ".join(str(blueprint.get(key) or "") for key in ("tags", "style", "vibe", "narrative", "description")),
        )
        payload_contract = _ace_step_track_payload_contract(lyric_plan, language, blueprint, opts)
        track_prompt_template = render_track_prompt_template(
            user_album_contract=contract_prompt_context(contract),
            ace_step_payload_contract=payload_contract,
            lyric_length_plan=lyric_plan,
            language_preset=language_preset(language),
            blueprint=blueprint,
            album_bible=album_bible,
        )
        _append_album_debug_jsonl(
            opts,
            "03_resolved_track_templates.jsonl",
            {
                "track_number": index + 1,
                "title": title,
                "template_version": ACE_STEP_TRACK_PROMPT_TEMPLATE_VERSION,
                "template": track_prompt_template,
            },
        )
        logs.append(f"Writing track {index + 1}/{num_tracks} with AceJAM Agents: {_monitor_preview(title, 90)}")
        previous_summaries = [_track_summary_for_agent(track) for track in produced_tracks]
        if ACEJAM_AGENT_SPLIT_TRACK_FLOW:
            if ACEJAM_AGENT_MICRO_SETTINGS_FLOW:
                logs.append(
                    f"Micro track flow: separate AI calls for BPM, key, time, duration, language, tags, "
                    f"caption, description, hook, and performance for track {index + 1}; "
                    "then lyric parts. No monolithic writer/finalizer prompt."
                )
                settings_payload = _call_track_micro_settings_agents(
                    album_bible=album_bible,
                    blueprint=blueprint,
                    lyric_plan=lyric_plan,
                    language=language,
                    index=index,
                    total=num_tracks,
                    duration=duration,
                    planner_provider=planner_provider,
                    planner_model=planner_model,
                    logs=logs,
                    opts=opts,
                    agent_stats=agent_stats,
                    context_store=context_store,
                )
            else:
                logs.append(
                    f"Split track flow: settings call + lyric parts for track {index + 1}; "
                    "no monolithic writer/finalizer prompt."
                )
                settings_payload = _agent_json_call(
                    agent_name="Track Settings Agent",
                    provider=planner_provider,
                    model_name=planner_model,
                    user_prompt=_track_settings_prompt(
                        concept=concept,
                        album_bible=album_bible,
                        blueprint=blueprint,
                        lyric_plan=lyric_plan,
                        language=language,
                        index=index,
                        total=num_tracks,
                        retrieved_context=context_store.block(
                            f"track {index + 1} settings caption metadata {title} {blueprint.get('style') or ''}",
                            kinds=["contract_track", "track_blueprint", "album_bible", "track_summary"],
                            track_number=index + 1,
                            label=f"track_{index + 1}_settings",
                        ),
                    ),
                    logs=logs,
                    debug_options=opts,
                    schema_name="track_settings_payload",
                    extra_system="No lyrics. Settings/caption/metadata only.",
                )
                agent_stats["agent_rounds"].append({"agent": "Track Settings Agent", "track_number": index + 1, "status": "completed"})
            section_groups = _lyric_section_groups(lyric_plan.get("sections") or [], ACEJAM_AGENT_LYRIC_PARTS)
            lyric_part_payloads: list[dict[str, Any]] = []
            for part_index, section_group in enumerate(section_groups):
                logs.append(
                    f"Writing lyrics part {part_index + 1}/{len(section_groups)} for track {index + 1}: "
                    f"{', '.join('[' + section + ']' for section in section_group)}"
                )
                part_payload = _agent_json_call(
                    agent_name=f"Track Lyrics Agent Part {part_index + 1}",
                    provider=planner_provider,
                    model_name=planner_model,
                    user_prompt=_track_lyrics_part_prompt(
                        concept=concept,
                        album_bible=album_bible,
                        blueprint=blueprint,
                        settings_payload=settings_payload,
                        lyric_plan=lyric_plan,
                        section_group=section_group,
                        part_index=part_index,
                        part_count=len(section_groups),
                        previous_parts=lyric_part_payloads,
                        language=language,
                        track_index=index,
                        total_tracks=num_tracks,
                        retrieved_context=context_store.block(
                            f"track {index + 1} lyrics part {part_index + 1} {title} {' '.join(section_group)}",
                            kinds=["contract_track", "track_blueprint", "album_bible", "track_summary"],
                            track_number=index + 1,
                            label=f"track_{index + 1}_lyrics_part_{part_index + 1}",
                        ),
                    ),
                    logs=logs,
                    debug_options=opts,
                    schema_name=f"track_lyrics_part_{part_index + 1}_payload",
                    extra_system="Lyrics only. Do not output settings, caption, or metadata.",
                )
                part_payload = _coerce_agent_lyrics_payload(part_payload)
                lyric_part_payloads.append(part_payload)
                _append_album_debug_jsonl(
                    opts,
                    "04_lyric_parts.jsonl",
                    {
                        "track_number": index + 1,
                        "title": title,
                        "part_index": part_index + 1,
                        "sections": section_group,
                        "payload": part_payload,
                    },
                )
                agent_stats["agent_rounds"].append({
                    "agent": "Track Lyrics Agent",
                    "track_number": index + 1,
                    "part_index": part_index + 1,
                    "status": "completed",
                })
            merged = _assemble_split_agent_track(
                blueprint=blueprint,
                settings_payload=settings_payload,
                lyric_part_payloads=lyric_part_payloads,
                section_groups=section_groups,
                language=language,
                duration=duration,
            )
            stats = lyric_stats(str(merged.get("lyrics") or ""))
            missing_words = max(0, int(lyric_plan.get("min_words") or 0) - int(stats.get("word_count") or 0))
            missing_lines = max(0, int(lyric_plan.get("min_lines") or 0) - int(stats.get("line_count") or 0))
            if missing_words or missing_lines:
                logs.append(
                    f"Track Lyric Continuation Agent: track {index + 1} needs small lyric continuation "
                    f"({missing_words} words, {missing_lines} lines)."
                )
                continuation_payload = _agent_json_call(
                    agent_name="Track Lyric Continuation Agent",
                    provider=planner_provider,
                    model_name=planner_model,
                    user_prompt=_track_lyrics_continuation_prompt(
                        blueprint=blueprint,
                        settings_payload=settings_payload,
                        lyric_plan=lyric_plan,
                        current_lyrics=str(merged.get("lyrics") or ""),
                        language=language,
                        index=index,
                        total=num_tracks,
                        missing_words=missing_words,
                        missing_lines=missing_lines,
                    ),
                    logs=logs,
                    debug_options=opts,
                    schema_name="track_lyrics_continuation_payload",
                    extra_system="Add only missing lyrics. No settings/caption/metadata.",
                    max_retries=1,
                )
                continuation_lines = _agent_payload_lines(continuation_payload)
                if continuation_lines:
                    merged["lyrics"] = (str(merged.get("lyrics") or "").rstrip() + "\n" + "\n".join(continuation_lines)).strip()
                    merged["lyrics_lines"] = [*list(merged.get("lyrics_lines") or []), *continuation_lines]
                    merged = _set_track_stats(merged)
                agent_stats["agent_rounds"].append({
                    "agent": "Track Lyric Continuation Agent",
                    "track_number": index + 1,
                    "status": "completed",
                })
        else:
            retrieved_context = context_store.block(
                f"track {index + 1} of {num_tracks} {title} lyrics tags caption {blueprint.get('style') or ''} {blueprint.get('narrative') or ''}",
                label=f"track_{index + 1}_writer",
            )
            writer_payload = _agent_json_call(
                agent_name="Track Writer Agent",
                provider=planner_provider,
                model_name=planner_model,
                user_prompt=_track_writer_prompt(
                    concept=concept,
                    album_bible=album_bible,
                    blueprint=blueprint,
                    previous_summaries=previous_summaries,
                    track_prompt_template=track_prompt_template,
                    lyric_plan=lyric_plan,
                    index=index,
                    total=num_tracks,
                    retrieved_context=retrieved_context,
                ),
                logs=logs,
                debug_options=opts,
                schema_name="track_writer_payload",
                extra_system="Write complete lyrics; do not summarize the song.",
            )
            agent_stats["agent_rounds"].append({"agent": "Track Writer Agent", "track_number": index + 1, "status": "completed"})
            finalizer_payload = _agent_json_call(
                agent_name="Track Finalizer Agent",
                provider=planner_provider,
                model_name=planner_model,
                user_prompt=_track_finalizer_prompt(
                    concept=concept,
                    album_bible=album_bible,
                    blueprint=blueprint,
                    writer_payload=writer_payload,
                    track_prompt_template=track_prompt_template,
                    lyric_plan=lyric_plan,
                    index=index,
                    total=num_tracks,
                    retrieved_context=context_store.block(
                        f"track {index + 1} finalizer {title} final ACE-Step payload",
                        label=f"track_{index + 1}_finalizer",
                    ),
                ),
                logs=logs,
                debug_options=opts,
                schema_name="final_track_payload",
                extra_system="Normalize only; preserve the writer lyrics unless repairing JSON structure.",
            )
            agent_stats["agent_rounds"].append({"agent": "Track Finalizer Agent", "track_number": index + 1, "status": "completed"})
            merged = {**blueprint, **writer_payload, **finalizer_payload}
        merged["track_number"] = int(blueprint.get("track_number") or index + 1)
        merged["duration"] = duration
        merged.setdefault("language", language)
        merged = apply_user_album_contract_to_track(merged, contract, index, logs)
        normalized = normalize_album_tracks([merged], opts)[0]
        gated = _gate_agent_track(
            track=normalized,
            blueprint=blueprint,
            album_bible=album_bible,
            concept=concept,
            opts=opts,
            contract=contract,
            index=index,
            total=num_tracks,
            planner_provider=planner_provider,
            planner_model=planner_model,
            logs=logs,
            track_prompt_template=track_prompt_template,
            agent_stats=agent_stats,
            retrieved_context=context_store.block(
                f"track {index + 1} quality repair {title} payload gate",
                kinds=["contract_track", "track_blueprint", "album_bible", "track_summary"],
                track_number=index + 1,
                label=f"track_{index + 1}_quality_repair",
            ),
        )
        _append_album_debug_jsonl(
            opts,
            "06_final_payloads.jsonl",
            {
                "track_number": gated.get("track_number"),
                "title": gated.get("title"),
                "payload_gate_status": gated.get("payload_gate_status"),
                "payload": gated,
            },
        )
        record_text, record_meta = _compact_track_memory_record(gated, include_lyrics_excerpt=False)
        context_store.add("track_summary", record_text, {"track_number": index + 1, **record_meta})
        produced_tracks.append(gated)

    tracks = []
    for index, track in enumerate(produced_tracks[:num_tracks]):
        tracks.append(_set_track_stats(apply_user_album_contract_to_track(track, contract, index, logs)))
    sequence_report = _album_sequence_report(tracks, contract, num_tracks)
    _write_album_debug_json(opts, "08_sequence_report.json", sequence_report)
    context_store.add("sequence_report", json.dumps(sequence_report, ensure_ascii=True), {"status": sequence_report.get("status")})
    if not sequence_report.get("gate_passed"):
        reasons = "; ".join(f"{item.get('id')}: {item.get('detail')}" for item in (sequence_report.get("issues") or [])[:8])
        raise AceJamAgentError(f"Album sequence critic failed: {reasons or 'sequence gate failed'}")
    logs.append(f"AceJAM Agents produced {len(tracks)} ACE-Step-ready track payload(s).")
    _write_album_debug_json(
        opts,
        "debug_index.json",
        {
            "version": "album-debug-index-acejam-agents-2026-04-29",
            "planning_engine": ACEJAM_AGENT_ENGINE,
            "agent_debug_dir": agent_debug_dir,
            "context_store": {
                "enabled": context_store.enabled,
                "provider": context_store.provider,
                "model": context_store.model,
                "chunk_count": context_store.chunk_count,
                "retrieval_rounds": context_store.retrieval_rounds,
                "index": str(context_store.root / "index.json") if context_store.root else "",
            },
            "sequence_report": sequence_report,
            "tracks": [
                {
                    "track_number": track.get("track_number"),
                    "title": track.get("title"),
                    "payload_gate_status": track.get("payload_gate_status"),
                    "lyrics_line_count": track.get("lyrics_line_count"),
                    "lyrics_word_count": track.get("lyrics_word_count"),
                }
                for track in tracks
            ],
        },
    )
    contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
    return {
        "tracks": tracks,
        "logs": logs,
        "success": True,
        "planning_engine": ACEJAM_AGENT_ENGINE,
        "custom_agents_used": True,
        "crewai_used": False,
        "toolbelt_fallback": False,
        "crewai_output_log_file": "",
        "agent_debug_dir": agent_debug_dir,
        "agent_rounds": agent_stats.get("agent_rounds") or [],
        "agent_repair_count": int(agent_stats.get("agent_repair_count") or 0),
        "album_bible_agent_error": bible_agent_error,
        "memory_enabled": context_store.enabled,
        "context_chunks": context_store.chunk_count,
        "retrieval_rounds": context_store.retrieval_rounds,
        "agent_context_store": str(context_store.root) if context_store.root else "",
        "context_store_index": str(context_store.root / "index.json") if context_store.root else "",
        "sequence_repair_count": int(sequence_report.get("repair_count") or 0),
        "sequence_report": sequence_report,
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "prompt_kit": prompt_kit_payload(),
        "toolkit": toolkit_payload(opts.get("installed_models")),
        "input_contract": contract_prompt_context(contract),
        "input_contract_applied": bool(contract.get("applied")),
        "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
        "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
        "contract_repair_count": max(0, contract_repairs),
        "toolkit_report": {
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "prompt_kit": prompt_kit_payload(),
            "model_advice": model_info,
            "artist_reference_notes": opts.get("artist_reference_notes", []),
            "album_bible": album_bible,
            "user_album_contract": contract,
            "input_contract_applied": bool(contract.get("applied")),
            "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
            "contract_repair_count": max(0, contract_repairs),
            "custom_agents": {
                "enabled": True,
                "planner_model": planner_model,
                "planner_provider": planner_provider,
                "agent_debug_dir": agent_debug_dir,
                "agent_repair_count": int(agent_stats.get("agent_repair_count") or 0),
                "album_bible_agent_error": bible_agent_error,
                "context_chunks": context_store.chunk_count,
                "retrieval_rounds": context_store.retrieval_rounds,
                "sequence_repair_count": int(sequence_report.get("repair_count") or 0),
            },
            "memory": {
                "enabled": context_store.enabled,
                "provider": context_store.provider,
                "embedding_model": context_store.model,
                "context_chunks": context_store.chunk_count,
                "retrieval_rounds": context_store.retrieval_rounds,
                "context_store": str(context_store.root) if context_store.root else "",
                "disabled_reason": context_store.disabled_reason,
            },
        },
    }


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
    logs: list[str] = _AlbumPlanLogs(log_callback)
    opts = _coerce_options(concept, num_tracks, track_duration, language, options)
    opts["genre_hint"] = _album_genre_hint(opts)
    lang_name = LANG_NAMES.get(language, language)
    logs.append(f"Concept preview: {_monitor_preview(opts['sanitized_concept'], 220)}")
    logs.append(f"Language: {lang_name}")
    logs.append(f"Prompt Kit: {PROMPT_KIT_VERSION}")
    logs.append(
        "Prompt Kit routing: "
        f"language_preset={language_preset(language).get('code')}; "
        f"genre_modules={','.join(module.get('slug', '') for module in infer_genre_modules(opts['sanitized_concept'], max_modules=2))}."
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
    planner_provider = normalize_provider(planner_provider or "ollama")
    embedding_provider = normalize_provider(embedding_provider or planner_provider)
    ollama_model = str(ollama_model or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL).strip()
    embedding_model = str(embedding_model or DEFAULT_ALBUM_EMBEDDING_MODEL).strip()
    logs.append(f"Planner LM: {provider_label(planner_provider)} ({ollama_model}); ACE-Step LM is not used for album agents.")
    logs.append(f"Embedding: {provider_label(embedding_provider)} ({embedding_model}).")
    completion_cap = CREWAI_LMSTUDIO_MAX_TOKENS if planner_provider == "lmstudio" else CREWAI_LLM_NUM_PREDICT
    logs.append(
        "AceJAM Agents runtime: "
        f"timeout={CREWAI_LLM_TIMEOUT_SECONDS}s, "
        f"num_ctx={CREWAI_LLM_CONTEXT_WINDOW}, "
        f"completion_cap={completion_cap}, "
        f"json_retries={ACEJAM_AGENT_JSON_RETRIES}, "
        f"gate_repair_retries={ACEJAM_AGENT_GATE_REPAIR_RETRIES}, "
        f"temperature={ACEJAM_AGENT_TEMPERATURE}."
    )
    logs.append(f"Song model strategy: {opts.get('song_model_strategy')}")
    if opts.get("artist_reference_notes"):
        logs.extend(str(note) for note in opts["artist_reference_notes"])
    agent_engine = str(opts.get("agent_engine") or ACEJAM_AGENT_ENGINE).strip().lower()

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
            "reason": "Album renders will be produced once per track for every ACE-Step album portfolio model.",
            "album_models": album_model_portfolio(opts.get("installed_models")),
            "multi_album": True,
        }
    if not model_info.get("ok"):
        if str(opts.get("song_model_strategy")) == "xl_sft_final" and model_info.get("model") == ALBUM_FINAL_MODEL:
            logs.append(f"Final model download required before generation: {model_info.get('error')}")
        else:
            logs.append(f"ERROR: {model_info.get('error')}")
            return {
                "tracks": [],
                "logs": logs,
                "success": False,
                "error": model_info.get("error"),
                "planning_engine": "none",
                "custom_agents_used": False,
                "crewai_used": False,
                "toolbelt_fallback": False,
                "crewai_output_log_file": str(crewai_output_log_file or ""),
                "prompt_kit_version": PROMPT_KIT_VERSION,
                "prompt_kit": prompt_kit_payload(),
                "toolkit": toolkit_payload(opts.get("installed_models")),
                "input_contract": contract_prompt_context(contract),
                "input_contract_applied": bool(contract.get("applied")),
                "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
                "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
                "contract_repair_count": 0,
            }

    # Backward-compat: if pre-planned tracks are provided while AI planning is
    # explicitly disabled, normalize and return. In the normal album generator,
    # editable tracks are only scaffold hints; AceJAM Agents still plan settings
    # and lyrics through the micro-call flow before audio render.
    if input_tracks:
        input_tracks = apply_user_album_contract_to_tracks(input_tracks, contract, logs)
        editable_tracks = normalize_album_tracks(input_tracks, opts)
        if not use_crewai or agent_engine == "editable_plan":
            logs.append(f"Using editable album plan with {len(editable_tracks)} tracks.")
            contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
            return {
                "tracks": editable_tracks, "logs": logs, "success": True,
                "planning_engine": "editable_plan", "custom_agents_used": False, "crewai_used": False, "toolbelt_fallback": False,
                "crewai_output_log_file": str(crewai_output_log_file or ""),
                "prompt_kit_version": PROMPT_KIT_VERSION, "prompt_kit": prompt_kit_payload(),
                "toolkit": toolkit_payload(opts.get("installed_models")),
                "input_contract": contract_prompt_context(contract),
                "input_contract_applied": bool(contract.get("applied")),
                "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
                "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
                "contract_repair_count": max(0, contract_repairs),
            }
        opts["editable_plan_tracks"] = editable_tracks
        logs.append(
            f"Editable album plan received with {len(editable_tracks)} track(s); "
            "AceJAM Agents will re-plan settings and lyrics before render."
        )

    if not use_crewai:
        fallback = build_album_plan(concept, num_tracks, track_duration, opts)
        logs.append(f"Toolbelt fallback planned {len(fallback['tracks'])} tracks.")
        contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
        return {
            "tracks": fallback["tracks"],
            "logs": logs,
            "success": True,
            "planning_engine": "toolbelt",
            "custom_agents_used": False,
            "crewai_used": False,
            "toolbelt_fallback": False,
            "crewai_output_log_file": str(crewai_output_log_file or ""),
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "prompt_kit": prompt_kit_payload(),
            "toolkit": toolkit_payload(opts.get("installed_models")),
            "input_contract": contract_prompt_context(contract),
            "input_contract_applied": bool(contract.get("applied")),
            "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
            "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
            "contract_repair_count": max(0, contract_repairs),
            "toolkit_report": fallback.get("toolkit_report", {}),
        }

    if agent_engine not in {"legacy_crewai", "crewai"}:
        try:
            return _plan_album_with_acejam_agents(
                concept=concept,
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
            )
        except Exception as exc:
            agent_error = str(exc)
            logs.append(f"AceJAM Agents planning failed loudly; deterministic toolbelt fallback was not used: {agent_error}")
            _write_album_debug_json(
                opts,
                "07_agent_failure.json",
                {
                    "error": agent_error,
                    "error_type": type(exc).__name__,
                    "planning_engine": ACEJAM_AGENT_ENGINE,
                    "toolbelt_fallback": False,
                },
            )
            contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
            return {
                "tracks": [],
                "logs": logs,
                "success": False,
                "planning_engine": ACEJAM_AGENT_ENGINE,
                "custom_agents_used": True,
                "crewai_used": False,
                "toolbelt_fallback": False,
                "crewai_error": "",
                "agent_error": agent_error,
                "error": agent_error or "AceJAM Agents planning failed",
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
                "toolkit_report": {"agent_error": agent_error, "toolbelt_fallback": False},
            }

    # --- Single professional CrewAI production crew ---
    crewai_error = ""
    try:
        crewai_log_file = str(crewai_output_log_file or "").strip()
        if crewai_log_file:
            Path(crewai_log_file).parent.mkdir(parents=True, exist_ok=True)
            logs.append(f"CrewAI output log file: {crewai_log_file}")
        step_callback = _crewai_step_callback(logs)
        task_callback = _crewai_task_callback(logs)

        # Preflight
        logs.append("CrewAI preflight starting.")
        preflight = preflight_album_local_llm(planner_provider, ollama_model, embedding_provider, embedding_model)
        logs.append(f"CrewAI memory: enabled at {preflight['memory_dir']} with local providers only.")
        logs.append(f"{provider_label(planner_provider)} preflight: planner chat={preflight['chat_ok']}, embedding={preflight['embed_ok']}.")
        for warning in preflight.get("warnings") or []:
            logs.append(f"Local LLM preflight warning: {warning}")
        if not preflight["ok"]:
            raise RuntimeError("; ".join(preflight["errors"]))
        embedding_model = str(preflight.get("embedding_model") or embedding_model)
        album_memory_writer = _make_album_memory_writer(ollama_model, embedding_model, planner_provider, embedding_provider)
        logs.append(f"CrewAI memory embedder selected: {embedding_model}.")
        logs.append(
            "CrewAI agent memory is read-only; AceJAM writes compact memory records only "
            f"(max {CREWAI_MEMORY_CONTENT_LIMIT} chars each)."
        )

        # Stage 1: compact album bible and locked track blueprints.
        logs.append(f"Planning compact album bible with CrewAI and {provider_label(planner_provider)} model {ollama_model}...")
        bible_crew = create_album_bible_crew(
            concept, num_tracks, track_duration, ollama_model, language, embedding_model,
            opts, planner_provider, embedding_provider,
            step_callback=step_callback, task_callback=task_callback,
            output_log_file=crewai_log_file or None,
        )
        bible_result = _kickoff_crewai_compact(bible_crew, logs, "album bible crew", crewai_log_file or None)
        fallback_plan = None
        try:
            bible_payload = _task_output_json_dict(bible_result)
            if _is_empty_response_payload(bible_payload):
                raise CrewAIEmptyResponseError(str(bible_payload.get("error") or "album bible crew returned an empty response marker"))
            raw_bible = bible_payload.get("album_bible")
            album_bible = raw_bible if isinstance(raw_bible, dict) else {"concept": str(raw_bible or opts["sanitized_concept"])}
            blueprints = [item for item in (bible_payload.get("tracks") or []) if isinstance(item, dict)]
        except Exception as parse_exc:
            if isinstance(parse_exc, CrewAIEmptyResponseError):
                raise
            logs.append(f"CrewAI bible JSON parse repair: {_monitor_preview(parse_exc, 320)}")
            fallback_plan = build_album_plan(concept, num_tracks, track_duration, opts)
            album_bible = {
                "concept": opts["sanitized_concept"],
                "arc": "deterministic repair after bible JSON parse issue",
                "motifs": [],
            }
            blueprints = list(fallback_plan.get("tracks") or [])

        blueprints = [item for item in blueprints if isinstance(item, dict)][:num_tracks]
        if len(blueprints) < num_tracks:
            fallback_plan = fallback_plan or build_album_plan(concept, num_tracks, track_duration, opts)
            seen_numbers = {int(item.get("track_number") or idx + 1) for idx, item in enumerate(blueprints)}
            before = len(blueprints)
            for fallback_track in fallback_plan.get("tracks") or []:
                number = int(fallback_track.get("track_number") or 0)
                if number not in seen_numbers:
                    blueprints.append(fallback_track)
                    seen_numbers.add(number)
                if len(blueprints) >= num_tracks:
                    break
            logs.append(f"Supplemented {len(blueprints) - before} bible blueprint(s) from deterministic toolbelt.")

        blueprints = apply_user_album_contract_to_tracks(blueprints[:num_tracks], contract, logs)
        blueprints = normalize_album_tracks(blueprints, opts)
        for index, blueprint in enumerate(blueprints):
            blueprint["track_number"] = int(blueprint.get("track_number") or index + 1)
            blueprint["duration"] = parse_duration_seconds(track_duration, track_duration)
        logs.append(f"CrewAI compact bible planned {len(blueprints)} track blueprint(s).")

        # Stage 2: produce each track with compact context.
        produced_tracks: list[dict[str, Any]] = []
        for index, blueprint in enumerate(blueprints):
            title = str(blueprint.get("title") or f"Track {index + 1}")
            logs.append(f"Producing track {index + 1}/{num_tracks} with compact context: {_monitor_preview(title, 90)}")
            track_crew = create_track_production_crew(
                album_bible, blueprint, num_tracks, track_duration, ollama_model, language, embedding_model,
                opts, planner_provider, embedding_provider,
                step_callback=step_callback, task_callback=task_callback,
                output_log_file=crewai_log_file or None,
            )
            track_result = _kickoff_crewai_compact(track_crew, logs, f"track {index + 1} production crew", crewai_log_file or None)
            try:
                track_payload = _task_output_json_dict(track_result)
                if _is_empty_response_payload(track_payload):
                    raise CrewAIEmptyResponseError(str(track_payload.get("error") or f"track {index + 1} crew returned an empty response marker"))
            except Exception as parse_exc:
                if isinstance(parse_exc, CrewAIEmptyResponseError):
                    raise
                logs.append(f"CrewAI track JSON parse repair: {_monitor_preview(parse_exc, 320)}")
                raw_text = _task_output_raw_text(track_result)
                lyrics = _lyric_like_text(raw_text)
                track_payload = {**blueprint, "lyrics": lyrics}
                logs.append(f"CrewAI JSON repair used production text for track {index + 1}.")
            if not isinstance(track_payload, dict):
                track_payload = {"lyrics": str(track_payload), "tool_notes": "Non-dict crew output coerced to lyrics."}
            preferred_lyrics, used_production_lyrics = _prefer_production_lyrics(
                track_payload.get("lyrics"),
                track_result,
                duration=parse_duration_seconds(blueprint.get("duration") or track_duration, track_duration),
                density=str(opts.get("lyric_density") or "balanced"),
                structure_preset=str(opts.get("structure_preset") or "auto"),
                genre_hint=" ".join(
                    str(blueprint.get(key) or track_payload.get(key) or "")
                    for key in ("tags", "description", "style", "vibe", "narrative")
                ),
            )
            if used_production_lyrics:
                track_payload["lyrics"] = preferred_lyrics
                notes = str(track_payload.get("tool_notes") or "").strip()
                track_payload["tool_notes"] = " ".join(
                    part for part in [notes, "CrewAI production lyrics preserved over shortened finalizer lyrics."] if part
                )
                logs.append(f"CrewAI production lyrics preserved for track {index + 1}.")
            merged = {**blueprint, **track_payload}
            merged["track_number"] = int(blueprint.get("track_number") or index + 1)
            merged["duration"] = parse_duration_seconds(blueprint.get("duration") or track_duration, track_duration)
            _append_album_debug_jsonl(
                opts,
                "04_track_final_json.jsonl",
                {
                    "track_number": merged.get("track_number"),
                    "title": merged.get("title"),
                    "payload": merged,
                },
            )
            produced_tracks.append(merged)

        produced_tracks = apply_user_album_contract_to_tracks(produced_tracks[:num_tracks], contract, logs)
        tracks = normalize_album_tracks(produced_tracks, opts)
        for index, track in enumerate(tracks):
            track["duration"] = parse_duration_seconds(blueprints[index].get("duration") if index < len(blueprints) else track_duration, track_duration)
            stats = lyric_stats(str(track.get("lyrics") or ""))
            track["lyrics_word_count"] = int(stats.get("word_count") or 0)
            track["lyrics_line_count"] = int(stats.get("line_count") or 0)
            track["lyrics_char_count"] = int(stats.get("char_count") or 0)
            track["section_count"] = int(stats.get("section_count") or 0)
            track["hook_count"] = sum(
                1 for section in stats.get("sections") or [] if re.search(r"chorus|hook|refrain", str(section), re.I)
            )

        # Write to memory
        _remember_album_bible(album_memory_writer, album_bible, tracks, logs)
        for track in tracks:
            _remember_track(album_memory_writer, track, logs)

        logs.append(f"CrewAI produced {len(tracks)} duration-ready track(s) with compact per-track production.")
        contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
        return {
            "tracks": tracks,
            "logs": logs,
            "success": True,
            "planning_engine": "crewai",
            "crewai_used": True,
            "toolbelt_fallback": False,
            "crewai_output_log_file": crewai_log_file,
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "prompt_kit": prompt_kit_payload(),
            "toolkit": toolkit_payload(opts.get("installed_models")),
            "input_contract": contract_prompt_context(contract),
            "input_contract_applied": bool(contract.get("applied")),
            "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
            "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
            "contract_repair_count": max(0, contract_repairs),
            "toolkit_report": {
                "prompt_kit_version": PROMPT_KIT_VERSION,
                "prompt_kit": prompt_kit_payload(),
                "model_advice": model_info,
                "artist_reference_notes": opts.get("artist_reference_notes", []),
                "album_bible": album_bible,
                "user_album_contract": contract,
                "input_contract_applied": bool(contract.get("applied")),
                "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
                "contract_repair_count": max(0, contract_repairs),
                "memory": {
                    "enabled": True,
                    "read_only_agents": True,
                    "backend_compact_writer": True,
                    "planner_model": ollama_model,
                    "planner_provider": planner_provider,
                    "embedding_model": embedding_model,
                    "embedding_provider": embedding_provider,
                    "memory_dir": str(CREWAI_MEMORY_DIR),
                    "legacy_memory_dir_untouched": str(CREWAI_LEGACY_MEMORY_DIR),
                    "record_limit_chars": CREWAI_MEMORY_CONTENT_LIMIT,
                },
            },
        }
    except Exception as exc:
        crewai_error = str(exc)
        logs.append(f"CrewAI planning failed loudly; deterministic toolbelt fallback was not used: {crewai_error}")
        _write_album_debug_json(
            opts,
            "04_crewai_failure.json",
            {
                "error": crewai_error,
                "error_type": type(exc).__name__,
                "planning_engine": "crewai",
                "toolbelt_fallback": False,
            },
        )

    contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
    return {
        "tracks": [],
        "logs": logs,
        "success": False,
        "planning_engine": "crewai",
        "crewai_used": True,
        "toolbelt_fallback": False,
        "crewai_error": crewai_error,
        "error": crewai_error or "CrewAI planning failed",
        "crewai_output_log_file": str(crewai_output_log_file or ""),
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "prompt_kit": prompt_kit_payload(),
        "toolkit": toolkit_payload(opts.get("installed_models")),
        "input_contract": contract_prompt_context(contract),
        "input_contract_applied": bool(contract.get("applied")),
        "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
        "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
        "contract_repair_count": max(0, contract_repairs),
        "toolkit_report": {"crewai_error": crewai_error, "toolbelt_fallback": False},
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
