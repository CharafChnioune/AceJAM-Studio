"""
Album generation using CrewAI agents with Ollama plus AceJAM songwriting tools.

The CrewAI layer is backed by deterministic post-processing tools. If an LLM
returns weak or malformed JSON, the same toolbelt repairs the plan so album
generation still has usable tags, lyrics, metadata, and model advice.
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
from typing import Any, Callable

# CrewAI telemetry attempts to attach custom Memory objects as OpenTelemetry
# attributes. AceJAM uses local compact monitor logs instead.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from pydantic import BaseModel, ConfigDict, Field

from local_llm import (
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
from studio_core import DEFAULT_QUALITY_PROFILE, docs_best_model_settings, normalize_quality_profile
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
        return text[match.start():]
    return text


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
    if text.lstrip().startswith(("{", "[")):
        raise ValueError("Crew result did not contain a valid JSON object")
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            parsed, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Crew result did not contain a valid JSON object")


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
    matched_genres = infer_genre_modules(opts["sanitized_concept"], max_modules=2)
    length_plan = lyric_length_plan(
        track_duration,
        str(opts.get("lyric_density") or "dense"),
        str(opts.get("structure_preset") or "auto"),
        opts["sanitized_concept"],
    )
    tool_context = json.dumps(
        {
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "language_preset": lang_preset,
            "genre_modules": matched_genres,
            "section_map": section_map_for(track_duration, opts["sanitized_concept"], instrumental=is_sparse_lyric_genre(opts["sanitized_concept"])),
            "lyric_length_plan": length_plan,
            "album_model_portfolio": album_model_portfolio(opts.get("installed_models")),
            "user_album_contract": contract_prompt_context(opts.get("user_album_contract")),
        },
        ensure_ascii=True,
    )
    tools = make_crewai_tools(opts)
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
    lyric_plan = lyric_length_plan(
        blueprint.get("duration") or track_duration,
        str(opts.get("lyric_density") or "dense"),
        str(opts.get("structure_preset") or "auto"),
        " ".join(str(blueprint.get(key) or "") for key in ("description", "tags", "style", "vibe", "narrative")),
    )
    tools = make_crewai_tools(opts)
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
        "settings_policy": "Use AceStepSettingsPolicyTool, ChartMasterProfileTool, AceStepCoverageAuditTool, EffectiveSettingsTool, TaskApplicabilityTool, and ModelCompatibilityTool. Keep unsupported/reserved/read-only settings out of active payloads.",
    }
    task_produce = _crew_task(
        description=(
            f"Produce exactly one track from this compact blueprint: {_compact_json(blueprint, 3600)}\n"
            f"Album bible: {_compact_json(album_bible, 1800)}\n"
            f"Production context: {_compact_json(compact_context, 5200)}\n"
            "Write complete lyrics unless the blueprint is explicitly instrumental. "
            "Stay on this single track; do not rewrite album-wide plans."
        ),
        expected_output="Complete production notes and lyrics for one track.",
        agent=producer,
    )
    task_json = _crew_task(
        description=(
            "Return strict JSON object only. Required fields: track_number, artist_name, title, description, tags, "
            "lyrics, bpm, key_scale, time_signature, language, duration, song_model, seed, inference_steps, "
            "guidance_scale, shift, infer_method, sampler_mode, audio_format, auto_score, auto_lrc, "
            "return_audio_codes, save_to_library, tool_notes, production_team, model_render_notes, "
            "quality_profile, prompt_kit_version, settings_policy_version, settings_compliance, quality_checks, contract_compliance, "
            "tag_coverage, lyric_duration_fit, caption_integrity, payload_gate_status, repair_actions. "
            "Preserve the blueprint title and locked fields exactly. No markdown fences."
        ),
        expected_output="Strict JSON object for exactly one produced track.",
        agent=finalizer,
        context=[task_produce],
        output_json=_output_json_for_provider(TrackProductionPayloadModel, planner_provider),
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
        "CrewAI runtime: "
        f"timeout={CREWAI_LLM_TIMEOUT_SECONDS}s, "
        f"num_ctx={CREWAI_LLM_CONTEXT_WINDOW}, "
        f"completion_cap={completion_cap}, "
        f"agent_iter={CREWAI_AGENT_MAX_ITER}, "
        f"agent_retries={CREWAI_AGENT_MAX_RETRY_LIMIT}, "
        f"task_retries={CREWAI_TASK_MAX_RETRIES}, "
        f"respect_context_window={CREWAI_RESPECT_CONTEXT_WINDOW}."
    )
    logs.append(f"Song model strategy: {opts.get('song_model_strategy')}")
    if opts.get("artist_reference_notes"):
        logs.extend(str(note) for note in opts["artist_reference_notes"])

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

    # Backward-compat: if pre-planned tracks are provided, normalize and return.
    if input_tracks:
        input_tracks = apply_user_album_contract_to_tracks(input_tracks, contract, logs)
        tracks = normalize_album_tracks(input_tracks, opts)
        logs.append(f"Using editable album plan with {len(tracks)} tracks.")
        contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
        return {
            "tracks": tracks, "logs": logs, "success": True,
            "planning_engine": "editable_plan", "crewai_used": False, "toolbelt_fallback": False,
            "crewai_output_log_file": str(crewai_output_log_file or ""),
            "prompt_kit_version": PROMPT_KIT_VERSION, "prompt_kit": prompt_kit_payload(),
            "toolkit": toolkit_payload(opts.get("installed_models")),
            "input_contract": contract_prompt_context(contract),
            "input_contract_applied": bool(contract.get("applied")),
            "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
            "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0),
            "contract_repair_count": max(0, contract_repairs),
        }

    if not use_crewai:
        fallback = build_album_plan(concept, num_tracks, track_duration, opts)
        logs.append(f"Toolbelt fallback planned {len(fallback['tracks'])} tracks.")
        contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
        return {
            "tracks": fallback["tracks"],
            "logs": logs,
            "success": True,
            "planning_engine": "toolbelt",
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
            raw_bible = bible_payload.get("album_bible")
            album_bible = raw_bible if isinstance(raw_bible, dict) else {"concept": str(raw_bible or opts["sanitized_concept"])}
            blueprints = [item for item in (bible_payload.get("tracks") or []) if isinstance(item, dict)]
        except Exception as parse_exc:
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
            except Exception as parse_exc:
                logs.append(f"CrewAI track JSON parse repair: {_monitor_preview(parse_exc, 320)}")
                raw_text = _task_output_raw_text(track_result)
                lyrics = _lyric_like_text(raw_text)
                track_payload = {**blueprint, "lyrics": lyrics}
                logs.append(f"CrewAI JSON repair used production text for track {index + 1}.")
            if not isinstance(track_payload, dict):
                track_payload = {"lyrics": str(track_payload), "tool_notes": "Non-dict crew output coerced to lyrics."}
            merged = {**blueprint, **track_payload}
            merged["track_number"] = int(blueprint.get("track_number") or index + 1)
            merged["duration"] = parse_duration_seconds(blueprint.get("duration") or track_duration, track_duration)
            produced_tracks.append(merged)

        produced_tracks = apply_user_album_contract_to_tracks(produced_tracks[:num_tracks], contract, logs)
        tracks = normalize_album_tracks(produced_tracks, opts)
        for index, track in enumerate(tracks):
            track["duration"] = parse_duration_seconds(blueprints[index].get("duration") if index < len(blueprints) else track_duration, track_duration)

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
        logs.append(f"CrewAI planning fell back to deterministic toolbelt: {crewai_error}")

    # Deterministic fallback
    fallback = build_album_plan(concept, num_tracks, track_duration, opts)
    logs.append(f"Toolbelt fallback planned {len(fallback['tracks'])} tracks.")
    contract_repairs = len([line for line in logs if str(line).startswith("Contract repaired:")]) - repair_lines_before
    return {
        "tracks": fallback["tracks"],
        "logs": logs,
        "success": True,
        "planning_engine": "toolbelt",
        "crewai_used": False,
        "toolbelt_fallback": True,
        "crewai_error": crewai_error,
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
