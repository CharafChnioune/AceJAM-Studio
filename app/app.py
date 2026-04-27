from __future__ import annotations

import base64
import gc
import hashlib
import ast
import json
import os
import re
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

for name in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(name, None)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
DATA_DIR = BASE_DIR / "data"
SONGS_DIR = DATA_DIR / "songs"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
ALBUMS_DIR = DATA_DIR / "albums"
LORA_DATASETS_DIR = DATA_DIR / "lora_datasets"
LORA_EXPORTS_DIR = DATA_DIR / "loras"
LORA_IMPORTS_DIR = DATA_DIR / "lora_imports"
OFFICIAL_ACE_STEP_DIR = BASE_DIR / "vendor" / "ACE-Step-1.5"
OFFICIAL_RUNNER_SCRIPT = BASE_DIR / "official_runner.py"
PINOKIO_START_LOG = BASE_DIR.parent / "logs" / "api" / "start.js" / "latest"
APP_UI_VERSION = "acejam-v0.5-bpm-key-audit-2026-04-27"
PAYLOAD_CONTRACT_VERSION = "2026-04-26"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL = "charaf/qwen3.6-27b-abliterated-mlx:mxfp4-instruct-general"
DEFAULT_ALBUM_EMBEDDING_MODEL = "nomic-embed-text:latest"
ALBUM_EMBEDDING_FALLBACK_MODELS = [
    DEFAULT_ALBUM_EMBEDDING_MODEL,
    "mxbai-embed-large:latest",
    "charaf/qwen3-vl-embedding-8b:latest",
]
ALBUM_JOB_KEEP_LIMIT = 50
ACE_LM_ABLITERATED_DIR = MODEL_CACHE_DIR / "ace_lm_abliterated"
ACE_LM_PREFERRED_MODEL = "acestep-5Hz-lm-4B"
_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
ACE_LM_BACKEND_DEFAULT = "mlx" if _IS_APPLE_SILICON else "pt"
ACE_LM_PRIVATE_UPLOAD_CONFIRM = "PRIVATE_HF_UPLOAD"
ACE_LM_CLEANUP_CONFIRM = "DELETE_ORIGINAL_ACE_LM_AFTER_SMOKE"
ACE_LM_SMOKE_CONFIRM = "ACE_LM_SMOKE_PASSED"
OBLITERATUS_REPO_URL = "https://github.com/elder-plinius/OBLITERATUS"

MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ACE_LM_ABLITERATED_DIR.mkdir(parents=True, exist_ok=True)
SONGS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ALBUMS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LORA_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_MODULES_CACHE", str(MODEL_CACHE_DIR / "hf_modules"))
os.environ.setdefault("MPLCONFIGDIR", str(MODEL_CACHE_DIR / "matplotlib"))

NANO_VLLM_DIR = BASE_DIR / "acestep" / "third_parts" / "nano-vllm"
if NANO_VLLM_DIR.exists():
    sys.path.insert(0, str(NANO_VLLM_DIR))

import torch
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from gradio import Server

from local_llm import (
    chat_completion as local_llm_chat_completion,
    lmstudio_download_model,
    lmstudio_download_status,
    lmstudio_load_model,
    lmstudio_model_catalog,
    lmstudio_unload_model,
    normalize_provider,
    provider_label,
    test_model as local_llm_test_model,
)
from acestep.constants import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    TASK_TYPES,
    TRACK_NAMES,
    VALID_LANGUAGES,
    VALID_TIME_SIGNATURES,
)
from acestep.handler import AceStepHandler
from lora_trainer import AceTrainingManager
from local_composer import LocalComposer
from songwriting_toolkit import (
    ALBUM_FINAL_MODEL,
    ALBUM_MODEL_PORTFOLIO,
    ALBUM_MODEL_PORTFOLIO_MODELS,
    MODEL_STRATEGIES,
    album_model_portfolio,
    album_models_for_strategy,
    choose_song_model,
    derive_artist_name,
    normalize_album_tracks,
    normalize_artist_name,
    parse_duration_seconds,
    split_terms,
    toolkit_payload,
)
from prompt_kit import (
    PROMPT_KIT_METADATA_FIELDS,
    PROMPT_KIT_VERSION,
    infer_genre_modules,
    is_sparse_lyric_genre,
    kit_metadata_defaults,
    prompt_kit_payload,
    prompt_kit_system_block,
    section_map_for,
)
from user_album_contract import (
    USER_ALBUM_CONTRACT_VERSION,
    apply_user_album_contract_to_tracks,
    contract_prompt_context,
    extract_user_album_contract,
    tracks_from_user_album_contract,
)
from studio_core import (
    ACE_STEP_CAPTION_CHAR_LIMIT,
    ACE_STEP_LYRICS_CHAR_LIMIT,
    ACE_STEP_LM_MODELS,
    PRO_AUDIO_TARGETS,
    PRO_QUALITY_AUDIT_VERSION,
    ALLOWED_AUDIO_EXTENSIONS,
    DOCS_BEST_AUDIO_FORMAT,
    DOCS_BEST_DEFAULT_LM_MODEL,
    DOCS_BEST_LM_DEFAULTS,
    DOCS_BEST_SOURCE_TASK_LM_SKIPS,
    DOCS_BEST_TURBO_HIGH_CAP_STEPS,
    DEFAULT_BPM,
    DEFAULT_KEY_SCALE,
    DEFAULT_QUALITY_PROFILE,
    KNOWN_ACE_STEP_MODELS,
    MAX_BATCH_SIZE,
    OFFICIAL_ACE_STEP_MANIFEST,
    OFFICIAL_UNRELEASED_MODELS,
    ace_step_settings_compliance,
    ace_step_settings_registry,
    build_task_instruction,
    clamp_float,
    clamp_int,
    apply_ace_step_text_budget,
    docs_best_model_settings,
    docs_best_quality_policy,
    ensure_task_supported,
    get_param,
    hit_readiness_report,
    lm_model_profiles_for_models,
    model_label,
    model_profiles_for_models,
    normalize_generation_text_fields,
    normalize_audio_format,
    normalize_key_scale,
    normalize_quality_profile,
    normalize_task_type,
    normalize_track_names,
    needs_vocal_lyrics,
    official_fields_used,
    official_manifest,
    ordered_models,
    parse_bool,
    parse_timesteps,
    pro_quality_policy,
    recommended_lm_model,
    recommended_song_model,
    quality_profile_model_settings,
    runtime_planner_report,
    safe_filename,
    safe_id,
    studio_ui_schema,
    supported_tasks_for_model,
    VALID_KEY_SCALES,
)


def _cleanup_accelerator_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _default_acestep_checkpoint() -> str:
    override = os.environ.get("ACE_STEP_MODEL", "").strip()
    if override:
        return override
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "acestep-v15-turbo"
    return "acestep-v15-xl-turbo"


def _song_model_label(name: str) -> str:
    return model_label(name)


def _app_ui_hash() -> str:
    try:
        return hashlib.sha256((BASE_DIR / "index.html").read_bytes()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _backend_code_hash() -> str:
    digest = hashlib.sha256()
    for filename in ["app.py", "studio_core.py", "official_runner.py", "songwriting_toolkit.py", "album_crew.py", "lora_trainer.py"]:
        path = BASE_DIR / filename
        if path.is_file():
            digest.update(filename.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


ACE_LM_DISABLED_DEFAULTS: dict[str, Any] = {
    "ace_lm_model": "none",
    "lm_model": "none",
    "lm_model_path": "none",
    "thinking": False,
    "sample_mode": False,
    "sample_query": "",
    "use_format": False,
    "allow_lm_batch": False,
    "lm_batch_chunk_size": 8,
    "lm_backend": ACE_LM_BACKEND_DEFAULT,
    "lm_cfg_scale": 2.0,
    "lm_negative_prompt": "NO USER INPUT",
    "lm_repetition_penalty": 1.0,
    "lm_temperature": 0.85,
    "lm_top_k": 0,
    "lm_top_p": 0.9,
    "use_cot_caption": False,
    "use_cot_language": False,
    "use_cot_lyrics": False,
    "use_cot_metas": False,
    "use_constrained_decoding": True,
    "constrained_decoding_debug": False,
}

ACE_LM_QUALITY_DEFAULTS: dict[str, Any] = {
    "ace_lm_model": DOCS_BEST_LM_DEFAULTS["ace_lm_model"],
    "lm_model": DOCS_BEST_LM_DEFAULTS["ace_lm_model"],
    "lm_model_path": DOCS_BEST_LM_DEFAULTS["ace_lm_model"],
    "thinking": DOCS_BEST_LM_DEFAULTS["thinking"],
    "sample_mode": False,
    "sample_query": "",
    "use_format": DOCS_BEST_LM_DEFAULTS["use_format"],
    "allow_lm_batch": False,
    "lm_batch_chunk_size": 8,
    "lm_backend": DOCS_BEST_LM_DEFAULTS["lm_backend"],
    "lm_cfg_scale": DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"],
    "lm_negative_prompt": "NO USER INPUT",
    "lm_repetition_penalty": 1.0,
    "lm_temperature": DOCS_BEST_LM_DEFAULTS["lm_temperature"],
    "lm_top_k": DOCS_BEST_LM_DEFAULTS["lm_top_k"],
    "lm_top_p": DOCS_BEST_LM_DEFAULTS["lm_top_p"],
    "use_cot_caption": DOCS_BEST_LM_DEFAULTS["use_cot_caption"],
    "use_cot_language": DOCS_BEST_LM_DEFAULTS["use_cot_language"],
    "use_cot_lyrics": DOCS_BEST_LM_DEFAULTS["use_cot_lyrics"],
    "use_cot_metas": DOCS_BEST_LM_DEFAULTS["use_cot_metas"],
    "use_constrained_decoding": DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"],
    "constrained_decoding_debug": False,
}

ACE_LM_TRIGGER_FIELDS = {
    "allow_lm_batch",
    "constrained_decoding_debug",
    "lm_batch_chunk_size",
    "lm_cfg_scale",
    "lm_negative_prompt",
    "lm_repetition_penalty",
    "lm_temperature",
    "lm_top_k",
    "lm_top_p",
    "sample_mode",
    "sample_query",
    "thinking",
    "use_constrained_decoding",
    "use_cot_caption",
    "use_cot_language",
    "use_cot_lyrics",
    "use_cot_metas",
    "use_format",
}


def _requested_ace_lm_model(payload: dict[str, Any]) -> str:
    value = str(get_param(payload or {}, "ace_lm_model", "") or "").strip()
    if value:
        lowered = value.lower()
        if lowered in {"off", "false", "0", "disabled"}:
            return "none"
        if lowered == "auto":
            return ACE_LM_PREFERRED_MODEL
        return value
    return ACE_LM_PREFERRED_MODEL


def _normalize_lm_backend(value: Any) -> str:
    backend = str(value or ACE_LM_BACKEND_DEFAULT).strip().lower()
    if backend == "auto":
        return ACE_LM_BACKEND_DEFAULT
    if backend == "mlx" and not _IS_APPLE_SILICON:
        return ACE_LM_BACKEND_DEFAULT
    if backend == "pt" and _IS_APPLE_SILICON and not parse_bool(os.environ.get("ACEJAM_ALLOW_PT_LM_BACKEND_ON_APPLE"), False):
        return ACE_LM_BACKEND_DEFAULT
    return backend if backend in {"pt", "vllm", "mlx"} else ACE_LM_BACKEND_DEFAULT


def _disable_acestep_mlx_backends(handler_cls: Any) -> None:
    def _disabled_mlx_backends(self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False
        self.mlx_vae = None
        self.use_mlx_vae = False
        return "Disabled by AceJAM (PyTorch/MPS)", "Disabled by AceJAM (PyTorch/MPS)"

    handler_cls._initialize_mlx_backends = _disabled_mlx_backends


def _apply_studio_lm_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep local planner metadata, while allowing explicit ACE-Step LM native controls."""
    cleaned = dict(payload or {})
    provider = normalize_provider(cleaned.get("planner_lm_provider") or cleaned.get("planner_provider") or "ollama")
    cleaned["planner_lm_provider"] = provider
    if provider == "ollama":
        cleaned.setdefault("planner_ollama_model", str(cleaned.get("planner_model") or cleaned.get("ollama_model") or "").strip())
    cleaned.setdefault("planner_model", str(cleaned.get("planner_model") or cleaned.get("planner_ollama_model") or cleaned.get("ollama_model") or "").strip())
    if "sample_query" not in cleaned:
        sample_query = str(get_param(cleaned, "sample_query", "") or "").strip()
        if sample_query:
            cleaned["sample_query"] = sample_query
    if "use_format" not in cleaned:
        use_format = get_param(cleaned, "use_format", None)
        if use_format not in [None, ""]:
            cleaned["use_format"] = use_format
    requested = _requested_ace_lm_model(cleaned)
    if requested == "none" and _explicit_ace_lm_controls(cleaned):
        requested = ACE_LM_PREFERRED_MODEL
    if requested == "none":
        cleaned["ace_lm_model"] = "none"
        cleaned["lm_model"] = "none"
        cleaned["lm_model_path"] = "none"
        cleaned["use_official_lm"] = False
        cleaned["lm_backend"] = _normalize_lm_backend(cleaned.get("lm_backend"))
        return cleaned
    cleaned["ace_lm_model"] = requested
    cleaned["lm_model"] = requested
    cleaned["lm_model_path"] = requested
    cleaned["lm_backend"] = _normalize_lm_backend(cleaned.get("lm_backend"))
    cleaned["use_official_lm"] = True
    return cleaned


def _explicit_ace_lm_controls(payload: dict[str, Any]) -> list[str]:
    used: list[str] = []
    for field in sorted(ACE_LM_TRIGGER_FIELDS):
        if field not in payload:
            continue
        value = payload.get(field)
        if field in ACE_LM_DISABLED_DEFAULTS:
            default = ACE_LM_DISABLED_DEFAULTS[field]
            if isinstance(default, bool):
                if parse_bool(value, default) and not default:
                    used.append(field)
                continue
            if not _studio_default_value(value, default):
                used.append(field)
        elif str(value or "").strip():
            used.append(field)
    return used


def _quality_lm_controls_enabled(payload: dict[str, Any], task_type: str) -> bool:
    if _requested_ace_lm_model(payload) == "none":
        return False
    if task_type in DOCS_BEST_SOURCE_TASK_LM_SKIPS:
        return False
    supplied_lyrics = str(payload.get("lyrics") or "").strip()
    has_supplied_vocal_lyrics = bool(supplied_lyrics and supplied_lyrics.lower() != "[instrumental]")
    default_enabled = not has_supplied_vocal_lyrics
    return any(
        [
            parse_bool(payload.get("thinking"), default_enabled and ACE_LM_QUALITY_DEFAULTS["thinking"]),
            parse_bool(get_param(payload, "use_format"), default_enabled and ACE_LM_QUALITY_DEFAULTS["use_format"]),
            parse_bool(payload.get("use_cot_lyrics"), default_enabled and ACE_LM_QUALITY_DEFAULTS["use_cot_lyrics"]),
            parse_bool(payload.get("sample_mode"), False),
            bool(str(get_param(payload, "sample_query", "") or "").strip()),
        ]
    )


def _active_official_fields(payload: dict[str, Any], task_type: str, existing: list[str]) -> list[str]:
    fields = list(existing)
    if _quality_lm_controls_enabled(payload, task_type):
        for field in ["thinking", "use_format", "use_cot_lyrics", "use_cot_caption", "use_cot_language", "use_cot_metas"]:
            if parse_bool(payload.get(field), ACE_LM_QUALITY_DEFAULTS.get(field, True)) and field not in fields:
                fields.append(field)
    return sorted(fields)


def _studio_default_value(value: Any, default: Any) -> bool:
    if isinstance(default, bool):
        return parse_bool(value, default) == default
    if isinstance(default, int) and not isinstance(default, bool):
        return clamp_int(value, default, -1000000, 1000000) == default
    if isinstance(default, float):
        return abs(clamp_float(value, default, -1000000.0, 1000000.0) - default) < 1e-9
    return str(value or "").strip() == str(default or "").strip()


def _disable_studio_ace_lm(payload: dict[str, Any]) -> dict[str, Any]:
    """Explicit opt-out path for users who disable official ACE-Step LM controls."""
    cleaned = dict(payload or {})
    cleaned["planner_lm_provider"] = normalize_provider(cleaned.get("planner_lm_provider") or cleaned.get("planner_provider") or "ollama")
    cleaned["ace_lm_model"] = "none"
    cleaned["lm_model"] = "none"
    cleaned["lm_model_path"] = "none"
    cleaned["use_official_lm"] = False
    for field, default in ACE_LM_DISABLED_DEFAULTS.items():
        if field in cleaned:
            cleaned[field] = default
    return cleaned


def _quality_default_steps(song_model: str, quality_profile: str | None = None) -> int:
    return int(quality_profile_model_settings(song_model, quality_profile or DEFAULT_QUALITY_PROFILE)["inference_steps"])


PROMPT_ASSISTANT_MODES: dict[str, dict[str, str]] = {
    "simple": {"label": "Simple", "file": "promptsimple.md", "description": "Fast prompt-to-song fields."},
    "custom": {"label": "Custom", "file": "promptcustom.md", "description": "Full Custom Studio song payload."},
    "song": {"label": "Song", "file": "promptsong.md", "description": "Backwards-compatible full song prompt."},
    "cover": {"label": "Cover / Remix", "file": "promptcover.md", "description": "Cover or remix direction from source audio notes."},
    "repaint": {"label": "Repaint", "file": "promptrepaint.md", "description": "Replace a section of an existing audio result."},
    "extract": {"label": "Extract", "file": "promptextract.md", "description": "Stem extraction plan."},
    "lego": {"label": "Lego", "file": "promptlego.md", "description": "Stem/layer reconstruction plan."},
    "complete": {"label": "Complete", "file": "promptcomplete.md", "description": "Finish an incomplete arrangement."},
    "album": {"label": "Album", "file": "promptalbum.md", "description": "Production-team album plan."},
    "news": {"label": "News to Song", "file": "promptnieuws.md", "description": "Turn news into a safe, postable song."},
    "improve": {"label": "Improve Lyrics", "file": "promptverbeter.md", "description": "Improve lyrics and optionally create AceJAM fields."},
    "trainer": {"label": "Trainer / LoRA", "file": "prompttrainer.md", "description": "Dataset labels and training metadata."},
}

PROMPT_ASSISTANT_ALIASES = {
    "lora": "trainer",
    "trainer_lora": "trainer",
    "library": "custom",
    "news_to_song": "news",
    "lyrics": "improve",
    "settings": "custom",
}


def _prompt_assistant_mode(value: str) -> str:
    mode = str(value or "custom").strip().lower().replace("-", "_")
    mode = PROMPT_ASSISTANT_ALIASES.get(mode, mode)
    if mode not in PROMPT_ASSISTANT_MODES:
        raise ValueError(f"Unknown prompt assistant mode: {value}")
    return mode


def _prompt_assistant_path(mode: str) -> Path:
    info = PROMPT_ASSISTANT_MODES[_prompt_assistant_mode(mode)]
    path = (BASE_DIR.parent / info["file"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {info['file']}")
    if BASE_DIR.parent.resolve() not in path.parents:
        raise ValueError("Prompt file path escaped project root")
    return path


def _prompt_assistant_system_prompt(mode: str) -> str:
    text = _prompt_assistant_path(mode).read_text(encoding="utf-8")
    match = re.search(r"## System Prompt\s*```(?:text)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        base_prompt = match.group(1).strip()
        return f"{base_prompt}\n\n{prompt_kit_system_block(mode)}"
    match = re.search(r"```(?:text)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        base_prompt = match.group(1).strip()
        return f"{base_prompt}\n\n{prompt_kit_system_block(mode)}"
    return f"{text.strip()}\n\n{prompt_kit_system_block(mode)}"


def _balanced_json_object(raw: str, start: int = 0) -> dict[str, Any]:
    text = str(raw or "")
    first = text.find("{", max(0, start))
    if first < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(first, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _loads_json_lenient_object(text[first : index + 1])
    raise ValueError("JSON object was not closed")


def _loads_json_lenient_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        repaired: list[str] = []
        in_string = False
        escape = False
        for char in text:
            if in_string:
                if escape:
                    repaired.append(char)
                    escape = False
                    continue
                if char == "\\":
                    repaired.append(char)
                    escape = True
                    continue
                if char == '"':
                    repaired.append(char)
                    in_string = False
                    continue
                if char == "\n":
                    repaired.append("\\n")
                    continue
                if char == "\r":
                    continue
                if char == "\t":
                    repaired.append("\\t")
                    continue
                repaired.append(char)
                continue
            repaired.append(char)
            if char == '"':
                in_string = True
        payload = json.loads("".join(repaired))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def _extract_prompt_assistant_json(raw: str, mode: str) -> tuple[dict[str, Any], str]:
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return _loads_json_lenient_object(text), ""
    except Exception:
        pass
    markers = [
        "ACEJAM_PAYLOAD_JSON",
        "ACEJAM_ALBUM_SETTINGS_JSON",
        "ACEJAM_DATASET_JSON",
    ]
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            payload = _balanced_json_object(text, pos + len(marker))
            return payload, text[:pos].strip()
    return _balanced_json_object(text), text


def _prompt_mode_task_type(mode: str) -> str:
    if mode in {"cover", "repaint", "extract", "lego", "complete"}:
        return mode
    return "text2music"


def _prompt_mode_default_model(mode: str) -> str:
    return "acestep-v15-xl-base" if mode in {"extract", "lego", "complete"} else "acestep-v15-xl-sft"


PROMPT_KIT_POLISHED_MODES = {"simple", "custom", "song", "album", "news"}
PROMPT_KIT_SOURCE_AUDIO_MODES = {"cover", "repaint", "extract", "lego", "complete"}


def _prompt_payload_kit_hint(payload: dict[str, Any]) -> str:
    parts = [
        payload.get("genre_profile"),
        payload.get("genre_modules"),
        payload.get("tags"),
        payload.get("caption"),
        payload.get("description"),
        payload.get("concept"),
        payload.get("title"),
    ]
    return " ".join(json.dumps(part, ensure_ascii=True) if isinstance(part, (dict, list)) else str(part or "") for part in parts)


def _apply_prompt_kit_metadata(mode: str, payload: dict[str, Any]) -> None:
    language = (
        payload.get("target_language")
        or payload.get("vocal_language")
        or payload.get("language")
        or "en"
    )
    duration = payload.get("duration") or payload.get("track_duration") or 180
    hint = _prompt_payload_kit_hint(payload)
    instrumental = parse_bool(payload.get("instrumental"), False) or is_sparse_lyric_genre(hint)
    defaults = kit_metadata_defaults(
        mode=_prompt_mode_task_type(mode),
        language=language,
        genre_hint=hint,
        duration=parse_duration_seconds(duration, 180),
        instrumental=instrumental,
    )
    defaults["concept_summary"] = str(payload.get("concept_summary") or payload.get("concept") or payload.get("description") or payload.get("title") or "")[:300]
    defaults["ace_caption"] = str(payload.get("ace_caption") or payload.get("caption") or payload.get("tags") or "")
    defaults["lyrics"] = str(payload.get("lyrics") or "")
    defaults["metadata"] = {
        "duration": parse_duration_seconds(duration, 180),
        "bpm": payload.get("bpm"),
        "key_scale": payload.get("key_scale"),
        "time_signature": payload.get("time_signature"),
        "vocal_language": payload.get("vocal_language") or language,
    }
    defaults["generation_settings"] = {
        key: payload.get(key)
        for key in ["song_model", "seed", "inference_steps", "guidance_scale", "shift", "infer_method", "sampler_mode", "audio_format"]
        if payload.get(key) not in (None, "")
    }
    for field in PROMPT_KIT_METADATA_FIELDS:
        if field == "copy_paste_block":
            payload.setdefault(field, "")
        elif field in defaults:
            payload.setdefault(field, defaults[field])
    payload.setdefault("prompt_kit_version", PROMPT_KIT_VERSION)
    payload.setdefault("vocal_language", defaults.get("vocal_language") or language)
    payload.setdefault("section_map", section_map_for(parse_duration_seconds(duration, 180), hint, instrumental=instrumental))
    payload.setdefault("genre_modules", [module.get("slug") for module in infer_genre_modules(hint, max_modules=2)])
    if mode in PROMPT_KIT_SOURCE_AUDIO_MODES:
        payload["source_audio_mode"] = payload.get("source_audio_mode") or "source_locked"
    if mode in PROMPT_KIT_POLISHED_MODES:
        payload["use_format"] = False


def _normalize_prompt_assistant_payload(mode: str, payload: dict[str, Any], body: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(payload or {})
    warnings: list[str] = []
    mode = _prompt_assistant_mode(mode)
    planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or normalized.get("planner_lm_provider") or "ollama")
    planner_model = str(
        body.get("planner_model")
        or body.get("planner_ollama_model")
        or body.get("ollama_model")
        or normalized.get("planner_model")
        or normalized.get("planner_ollama_model")
        or ""
    ).strip()
    normalized["ace_lm_model"] = _requested_ace_lm_model(normalized)
    normalized["planner_lm_provider"] = planner_provider
    normalized["planner_model"] = planner_model
    if planner_provider == "ollama":
        normalized["planner_ollama_model"] = planner_model
    else:
        normalized.pop("planner_ollama_model", None)
    if normalized["ace_lm_model"] != "none":
        if parse_bool(normalized.get("auto_score"), False) or parse_bool(normalized.get("auto_lrc"), False):
            warnings.append("Auto score/LRC were turned off because official ACE-Step LM generation cannot use the in-process tensor cache.")
        normalized["auto_score"] = False
        normalized["auto_lrc"] = False
    quality_profile = normalize_quality_profile(normalized.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    normalized["quality_profile"] = quality_profile

    if mode == "album":
        normalized.setdefault("song_model_strategy", "all_models_album")
        normalized.setdefault("final_song_model", "all_models_album")
        contract_source = "\n\n".join(
            part for part in [
                str(body.get("user_prompt") or body.get("prompt") or ""),
                str(normalized.get("album_title") or normalized.get("album_name") or ""),
                str(normalized.get("concept") or ""),
                json.dumps(normalized.get("tracks") or [], ensure_ascii=True, default=str)[:6000],
            ]
            if part
        )
        user_album_contract = normalized.get("user_album_contract")
        if not isinstance(user_album_contract, dict):
            user_album_contract = extract_user_album_contract(
                contract_source,
                int(normalized.get("num_tracks") or body.get("num_tracks") or 0) or None,
                str(normalized.get("language") or normalized.get("target_language") or body.get("language") or "en"),
                normalized,
            )
        if user_album_contract.get("applied"):
            normalized["user_album_contract"] = user_album_contract
            normalized["input_contract"] = contract_prompt_context(user_album_contract)
            normalized["input_contract_applied"] = True
            normalized["input_contract_version"] = USER_ALBUM_CONTRACT_VERSION
            normalized["blocked_unsafe_count"] = int(user_album_contract.get("blocked_unsafe_count") or 0)
            if user_album_contract.get("album_title"):
                normalized["album_title"] = user_album_contract.get("album_title")
        album_defaults = quality_profile_model_settings(ALBUM_FINAL_MODEL, quality_profile)
        normalized["audio_format"] = album_defaults["audio_format"]
        normalized["inference_steps"] = album_defaults["inference_steps"]
        normalized["guidance_scale"] = album_defaults["guidance_scale"]
        normalized["shift"] = album_defaults["shift"]
        normalized["infer_method"] = album_defaults["infer_method"]
        normalized["sampler_mode"] = album_defaults["sampler_mode"]
        for field, value in DOCS_BEST_LM_DEFAULTS.items():
            if field != "ace_lm_model":
                normalized[field] = value
        normalized["thinking"] = False
        normalized["use_format"] = False
        normalized["use_cot_lyrics"] = False
        normalized.setdefault("track_variants", 1)
        normalized.setdefault("save_to_library", True)
        tracks = normalized.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            contract_tracks = tracks_from_user_album_contract(user_album_contract)
            normalized["tracks"] = contract_tracks if contract_tracks else []
            if not contract_tracks:
                warnings.append("Album prompt did not return tracks; use Plan Album or ask again with more detail.")
        else:
            normalized["tracks"] = apply_user_album_contract_to_tracks(tracks, user_album_contract)
        for track in normalized.get("tracks") or []:
            if isinstance(track, dict):
                track.setdefault(
                    "artist_name",
                    derive_artist_name(
                        track.get("title") or "",
                        " ".join(str(item or "") for item in [normalized.get("concept"), track.get("description")]),
                        track.get("tags") or track.get("caption") or "",
                        int(track.get("track_number") or 1) - 1,
                    ),
                )
                track["ace_lm_model"] = normalized["ace_lm_model"]
                track["quality_profile"] = quality_profile
                track["thinking"] = False
                track["use_format"] = False
                track["use_cot_lyrics"] = False
                track["audio_format"] = normalized["audio_format"]
                track["inference_steps"] = normalized["inference_steps"]
                track["guidance_scale"] = normalized["guidance_scale"]
                track["shift"] = normalized["shift"]
                track["sampler_mode"] = normalized["sampler_mode"]
                track["auto_score"] = False
                track["auto_lrc"] = False
                track.setdefault("language", str(normalized.get("language") or normalized.get("target_language") or "en"))
                _apply_prompt_kit_metadata("album", track)
        _apply_prompt_kit_metadata("album", normalized)
        normalized["use_format"] = False
        return normalized, warnings

    if mode == "trainer":
        normalized.setdefault("adapter_type", "lora")
        normalized.setdefault("generation_reference_defaults", {})
        if isinstance(normalized["generation_reference_defaults"], dict):
            normalized["generation_reference_defaults"].setdefault("ace_lm_model", DOCS_BEST_DEFAULT_LM_MODEL)
            normalized["generation_reference_defaults"].setdefault("planner_lm_provider", planner_provider)
            normalized["generation_reference_defaults"].setdefault("planner_model", planner_model)
            if planner_provider == "ollama":
                normalized["generation_reference_defaults"].setdefault("planner_ollama_model", planner_model)
            normalized["generation_reference_defaults"].setdefault("thinking", DOCS_BEST_LM_DEFAULTS["thinking"])
            normalized["generation_reference_defaults"].setdefault("use_format", DOCS_BEST_LM_DEFAULTS["use_format"])
            normalized["generation_reference_defaults"].setdefault("use_cot_lyrics", DOCS_BEST_LM_DEFAULTS["use_cot_lyrics"])
            normalized["generation_reference_defaults"].setdefault("prompt_kit_version", PROMPT_KIT_VERSION)
        _apply_prompt_kit_metadata("trainer", normalized)
        return normalized, warnings

    normalized.setdefault("task_type", _prompt_mode_task_type(mode))
    normalized.setdefault("song_model", _prompt_mode_default_model(mode))
    normalized.setdefault("title", "Untitled")
    normalized["artist_name"] = normalize_artist_name(
        normalized.get("artist_name") or normalized.get("artist"),
        derive_artist_name(
            normalized.get("title") or "",
            normalized.get("description") or normalized.get("caption") or "",
            normalized.get("tags") or "",
        ),
    )
    tags = normalized.get("tags")
    if isinstance(tags, list):
        tags_text = ", ".join(str(item).strip() for item in tags if str(item).strip())
        normalized["tags"] = tags_text
    else:
        tags_text = str(tags or "").strip()
    caption = str(normalized.get("caption") or "").strip()
    if not caption and tags_text:
        normalized["caption"] = tags_text
    if not tags_text and caption:
        normalized["tags"] = caption
    normalized.setdefault(
        "negative_tags",
        "muddy mix, generic lyrics, weak hook, empty lyrics, off-key vocal, unclear vocal, noisy artifacts, flat drums, contradictory style",
    )
    instrumental = parse_bool(normalized.get("instrumental"), False)
    normalized["instrumental"] = instrumental
    if instrumental and not str(normalized.get("lyrics") or "").strip():
        normalized["lyrics"] = "[Instrumental]"
    normalized.setdefault("lyrics", "")
    normalized.setdefault("duration", 180)
    normalized.setdefault("bpm", DEFAULT_BPM)
    normalized.setdefault("key_scale", DEFAULT_KEY_SCALE)
    normalized.setdefault("time_signature", "4")
    normalized.setdefault("vocal_language", "en")
    normalized.setdefault("seed", "-1")
    normalized.setdefault("use_random_seed", True)
    mode_defaults = quality_profile_model_settings(str(normalized.get("song_model") or ""), quality_profile)
    normalized["inference_steps"] = mode_defaults["inference_steps"]
    normalized["guidance_scale"] = mode_defaults["guidance_scale"]
    normalized["shift"] = mode_defaults["shift"]
    normalized["infer_method"] = mode_defaults["infer_method"]
    normalized["sampler_mode"] = mode_defaults["sampler_mode"]
    normalized["audio_format"] = mode_defaults["audio_format"]
    for field, value in DOCS_BEST_LM_DEFAULTS.items():
        if field != "ace_lm_model":
            normalized[field] = value
    normalized["thinking"] = False
    normalized["use_format"] = False
    normalized["use_cot_lyrics"] = False
    _apply_prompt_kit_metadata(mode, normalized)
    normalized.setdefault("save_to_library", True)
    if mode in {"cover", "repaint", "extract", "lego", "complete"} and not (
        normalized.get("src_audio_id") or normalized.get("src_result_id") or normalized.get("audio_code_string")
    ):
        warnings.append(f"{mode} needs source audio selected/uploaded in AceJAM before generation.")
    return normalized, warnings


def _run_prompt_assistant_local(
    system_prompt: str,
    user_prompt: str,
    planner_provider: str,
    planner_model: str,
    current_payload: dict[str, Any],
) -> str:
    provider = normalize_provider(planner_provider)
    model = str(planner_model or "").strip()
    if provider == "ollama":
        if not model:
            listed = json.loads(ollama_models())
            models = listed.get("chat_models") or listed.get("models") or []
            if models:
                first = models[0]
                model = str(first.get("name") or first.get("model") if isinstance(first, dict) else first).strip()
        if not model:
            raise RuntimeError("No Ollama model selected or available.")
        _ensure_ollama_model_or_start_pull(model, context="AI Fill", kind="chat")
    else:
        model = _resolve_local_llm_model_selection(provider, model, "chat", "AI Fill")
    user_content = (
        f"USER REQUEST:\n{str(user_prompt or '').strip()}\n\n"
        "CURRENT ACEJAM UI PAYLOAD JSON:\n"
        f"{json.dumps(_jsonable(current_payload or {}), ensure_ascii=False, indent=2)}\n\n"
        "Return the exact sections requested by the system prompt. Keep JSON valid."
    )
    try:
        return local_llm_chat_completion(
            provider,
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            options={"temperature": 0.55, "top_p": 0.92, "top_k": 40},
        )
    except Exception as exc:
        if provider == "ollama" and _ollama_error_is_missing_model(exc):
            job = _start_ollama_pull(model, reason="AI Fill", kind="chat")
            raise OllamaPullStarted(model, job, f"{model} is missing; pull started for AI Fill.") from exc
        raise


def _run_prompt_assistant_ollama(system_prompt: str, user_prompt: str, ollama_model: str, current_payload: dict[str, Any]) -> str:
    return _run_prompt_assistant_local(system_prompt, user_prompt, "ollama", ollama_model, current_payload)


def _download_job_active(model_name: str) -> bool:
    jobs = globals().get("_model_download_jobs", {})
    job = jobs.get(model_name) if isinstance(jobs, dict) else None
    return bool(job and job.get("state") in {"queued", "running"})


def _checkpoint_dir_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else {}
            shards = sorted({str(name) for name in weight_map.values()}) if isinstance(weight_map, dict) else []
            if shards:
                return all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards)
        except Exception:
            return False
    return any(
        child.is_file() and child.stat().st_size > 0 and child.suffix in {".safetensors", ".bin", ".pt"}
        for child in path.iterdir()
    )


def _available_acestep_models() -> list[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    available = set(KNOWN_ACE_STEP_MODELS)
    if checkpoint_dir.exists():
        for child in checkpoint_dir.iterdir():
            if child.is_dir() and child.name.startswith("acestep-v15-"):
                available.add(child.name)
    return ordered_models(list(available))


def _installed_acestep_models() -> set[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    if not checkpoint_dir.exists():
        return set()
    return {
        child.name
        for child in checkpoint_dir.iterdir()
        if child.name.startswith("acestep-v15-") and not _download_job_active(child.name) and _checkpoint_dir_ready(child)
    }


def _installed_lm_models() -> set[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    installed = {"auto", "none"}
    if checkpoint_dir.exists():
        installed.update(
            child.name
            for child in checkpoint_dir.iterdir()
            if child.name.startswith("acestep-5Hz-lm-") and not _download_job_active(child.name) and _checkpoint_dir_ready(child)
        )
    return installed


def _normalize_song_model(requested: str | None) -> str:
    value = (requested or "").strip()
    if not value or value == "auto":
        return _default_acestep_checkpoint()
    if value.startswith("acestep-v15-"):
        return value
    return _default_acestep_checkpoint()


def _song_model_for_quality_profile(requested: str | None, quality_profile: str | None, task_type: str = "text2music") -> str:
    value = (requested or "").strip()
    profile = normalize_quality_profile(quality_profile)
    if value and value != "auto":
        return _normalize_song_model(value)
    installed = _installed_acestep_models()
    if normalize_task_type(task_type) in {"extract", "lego", "complete"}:
        for candidate in ["acestep-v15-xl-base", "acestep-v15-base"]:
            if candidate in installed:
                return candidate
        return "acestep-v15-xl-base"
    if profile in {"chart_master", "balanced_pro"}:
        return recommended_song_model(installed)
    if profile == "preview_fast":
        for candidate in ["acestep-v15-xl-turbo", "acestep-v15-turbo"]:
            if candidate in installed:
                return candidate
    return _default_acestep_checkpoint()


def _log_block(label: str, text: str) -> None:
    print(f"[{label}] ---")
    cleaned = (text or "").rstrip()
    print(cleaned if cleaned else "<empty>")
    print(f"[/{label}] ---")


def _get_storage_path() -> str:
    storage_root = MODEL_CACHE_DIR
    checkpoint_dir = storage_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = _default_acestep_checkpoint()

    try:
        from huggingface_hub import snapshot_download

        target = checkpoint_dir / checkpoint_name
        if not target.exists():
            cached = Path(snapshot_download(f"ACE-Step/{checkpoint_name}", local_files_only=True))
            try:
                target.symlink_to(cached, target_is_directory=True)
                print(f"[startup] Linked {checkpoint_name} -> {cached}")
            except FileExistsError:
                pass
            except OSError as exc:
                print(f"[startup] Could not link {checkpoint_name}: {exc}")

        shared_cache = Path(snapshot_download("ACE-Step/Ace-Step1.5", local_files_only=True))
        for child in shared_cache.iterdir():
            dst = checkpoint_dir / child.name
            if dst.exists() or not child.is_dir():
                continue
            try:
                dst.symlink_to(child, target_is_directory=True)
                print(f"[startup] Linked {child.name} -> {child}")
            except OSError as exc:
                print(f"[startup] Could not link {child.name}: {exc}")
    except Exception as exc:
        print(f"[startup] Cache warm links skipped: {exc}")

    return str(storage_root)


STORAGE_PATH = _get_storage_path()
print(f"[startup] Model storage: {STORAGE_PATH}")
ACE_STEP_CHECKPOINT = _default_acestep_checkpoint()
print(f"[startup] ACE-Step checkpoint: {ACE_STEP_CHECKPOINT}")

if not _IS_APPLE_SILICON:
    _disable_acestep_mlx_backends(AceStepHandler)
else:
    print("[startup] Apple Silicon detected: MLX backends enabled for DiT and VAE")
handler = AceStepHandler(persistent_storage_path=STORAGE_PATH)
handler_lock = threading.Lock()
ACTIVE_ACE_STEP_MODEL = ACE_STEP_CHECKPOINT


def _release_handler_state() -> None:
    handler.model = None
    handler.config = None
    handler.vae = None
    handler.text_encoder = None
    handler.text_tokenizer = None
    handler.silence_latent = None
    gc.collect()
    _cleanup_accelerator_memory()


def _release_models_for_training() -> None:
    with handler_lock:
        _release_handler_state()


def _activate_trained_adapter(adapter_path: Path, scale: float = 1.0) -> dict[str, Any]:
    adapter_path = adapter_path.expanduser().resolve()
    metadata_path = adapter_path / "acejam_adapter.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    try:
        with handler_lock:
            if metadata.get("song_model"):
                _ensure_song_model(str(metadata.get("song_model")))
            status_msg = handler.load_lora(str(adapter_path))
            if status_msg.startswith("❌"):
                return {"success": False, "status": status_msg, "path": str(adapter_path)}
            scale_msg = handler.set_lora_scale(float(scale))
            use_msg = handler.set_use_lora(True)
            return {
                "success": not str(use_msg).startswith("❌"),
                "path": str(adapter_path),
                "status": status_msg,
                "scale_status": scale_msg,
                "use_status": use_msg,
                **handler.get_lora_status(),
            }
    except Exception as exc:
        return {"success": False, "error": str(exc), "path": str(adapter_path)}


def _unload_llm_models_for_generation() -> None:
    """Unload Ollama and LM Studio models to free unified memory for audio generation."""
    # Unload all Ollama models by sending keep_alive=0
    try:
        import ollama as _ollama_client
        client = _ollama_client.Client(host=ollama_host())
        response = client.list()
        for model in response.models:
            try:
                client.generate(model=model.model, prompt="", keep_alive=0)
            except Exception:
                pass
        print("[generate_album] Unloaded Ollama models to free memory.", flush=True)
    except Exception as exc:
        print(f"[generate_album] Ollama unload skipped: {exc}", flush=True)
    # Unload LM Studio models
    try:
        catalog = lmstudio_model_catalog()
        for model in catalog.get("loaded_models", []):
            try:
                lmstudio_unload_model(str(model))
            except Exception:
                pass
        if catalog.get("loaded_models"):
            print("[generate_album] Unloaded LM Studio models to free memory.", flush=True)
    except Exception:
        pass


training_manager = AceTrainingManager(
    base_dir=BASE_DIR,
    data_dir=DATA_DIR,
    model_cache_dir=MODEL_CACHE_DIR,
    release_models=_release_models_for_training,
    adapter_ready=_activate_trained_adapter,
)


def _ensure_training_idle() -> None:
    active_job = training_manager.active_job()
    if active_job:
        raise RuntimeError(
            f"ACE-Step trainer is busy with {active_job['kind']} job {active_job['id']}. "
            "Wait for it to finish or stop it before generation."
        )


def _safe_lora_upload_relative_path(filename: str) -> Path:
    text = str(filename or "").replace("\\", "/").strip().lstrip("/")
    parts = []
    for part in Path(text).parts:
        if part in {"", ".", ".."}:
            continue
        if ":" in part:
            continue
        parts.append(part)
    if not parts:
        parts = [f"upload-{uuid.uuid4().hex[:8]}"]
    return Path(*parts)


def _lora_adapter_request(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload.get("lora_adapter_path") or payload.get("lora_path") or payload.get("lora_name_or_path") or "").strip()
    name = str(payload.get("lora_adapter_name") or payload.get("adapter_name") or "").strip()
    use_lora = parse_bool(payload.get("use_lora"), bool(path) or parse_bool(payload.get("lora_use"), False))
    scale = clamp_float(payload.get("lora_scale", payload.get("lora_weight")), 1.0, 0.0, 1.0)
    model_variant = str(payload.get("adapter_model_variant") or "").strip()
    return {
        "use_lora": use_lora,
        "lora_adapter_path": path,
        "lora_adapter_name": name,
        "lora_scale": scale,
        "adapter_model_variant": model_variant,
    }


def _apply_lora_request(params: dict[str, Any]) -> dict[str, Any]:
    if not params.get("use_lora"):
        try:
            status = handler.set_use_lora(False)
            return {"success": not status.startswith("❌"), "status": status, **handler.get_lora_status()}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
    adapter_path = str(params.get("lora_adapter_path") or "").strip()
    if not adapter_path:
        raise RuntimeError("Use adapter is enabled, but no adapter was selected.")
    status_msg = handler.load_lora(adapter_path)
    if status_msg.startswith("❌"):
        raise RuntimeError(status_msg)
    scale_msg = handler.set_lora_scale(float(params.get("lora_scale") or 1.0))
    use_msg = handler.set_use_lora(True)
    if str(use_msg).startswith("❌"):
        raise RuntimeError(use_msg)
    return {
        "success": True,
        "path": adapter_path,
        "status": status_msg,
        "scale_status": scale_msg,
        "use_status": use_msg,
        **handler.get_lora_status(),
    }


def _initialize_acestep_handler(config_path: str) -> tuple[str, bool]:
    return handler.initialize_service(
        project_root=str(BASE_DIR),
        config_path=config_path,
        device="auto",
        use_flash_attention=handler.is_flash_attention_available(),
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )


def _ensure_song_model(requested: str | None) -> str:
    global ACTIVE_ACE_STEP_MODEL

    target_model = _normalize_song_model(requested)
    if handler.model is not None and ACTIVE_ACE_STEP_MODEL == target_model:
        return ACTIVE_ACE_STEP_MODEL

    previous_model = ACTIVE_ACE_STEP_MODEL
    if handler.model is None:
        print(f"[song-model] initializing {target_model}")
    else:
        print(f"[song-model] switching {previous_model} -> {target_model}")

    _release_handler_state()
    status, ready = _initialize_acestep_handler(target_model)
    if ready:
        ACTIVE_ACE_STEP_MODEL = target_model
        print(f"[song-model] active={ACTIVE_ACE_STEP_MODEL}")
        print(status)
        return ACTIVE_ACE_STEP_MODEL

    print(f"[song-model] failed to load {target_model}")
    print(status)
    if previous_model != target_model:
        print(f"[song-model] restoring previous model {previous_model}")
        _release_handler_state()
        restore_status, restore_ready = _initialize_acestep_handler(previous_model)
        if restore_ready:
            ACTIVE_ACE_STEP_MODEL = previous_model
            print(f"[song-model] restored active={ACTIVE_ACE_STEP_MODEL}")
            print(restore_status)
        else:
            print("[song-model] restore failed")
            print(restore_status)

    raise RuntimeError(f"failed to initialize ACE-Step model: {target_model}")


if os.environ.get("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "").strip() == "1":
    status, ready = ("Skipped ACE-Step model initialization for tests.", True)
else:
    status, ready = _initialize_acestep_handler(ACE_STEP_CHECKPOINT)
print(f"[startup] Handler ready={ready} status={status}")

composer = LocalComposer()


def _language_for_generation(language: str) -> str:
    value = (language or "unknown").strip().lower()
    if value == "instrumental":
        return "unknown"
    if value in VALID_LANGUAGES:
        return value
    return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "shape"):
        return {"shape": list(value.shape)}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _audio_tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    data = tensor.cpu().float().numpy()
    if data.ndim == 2:
        data = data.T
        if data.shape[1] == 1:
            data = data[:, 0]
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak > 1e-4:
        data = (data / peak * 0.95).astype(np.float32)
    return data


def _write_audio_file(audio_dict: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = _audio_tensor_to_array(audio_dict["tensor"])
    sf.write(str(out_path), data, int(audio_dict["sample_rate"]))


def _encode_audio_file(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:audio/{path.suffix.lstrip('.') or 'wav'};base64,{encoded}"


def _song_public_url(song_id: str, filename: str) -> str:
    return f"/media/songs/{song_id}/{filename}"


def _result_public_url(result_id: str, filename: str) -> str:
    return f"/media/results/{result_id}/{filename}"


def _model_slug(model_name: str) -> str:
    value = re.sub(r"^acestep-v15-", "", str(model_name or "model")).replace("_", "-")
    return safe_filename(value, "model")


def _artist_name_from_payload(payload: dict[str, Any], *, title: str = "", index: int = 0) -> str:
    album_meta = payload.get("album_metadata") if isinstance(payload.get("album_metadata"), dict) else {}
    raw = payload.get("artist_name") or payload.get("artist") or album_meta.get("artist_name") or album_meta.get("artist")
    fallback = derive_artist_name(
        title or payload.get("title") or album_meta.get("title") or "",
        " ".join(
            str(item or "")
            for item in [
                payload.get("description"),
                payload.get("caption"),
                payload.get("tags"),
                payload.get("global_caption"),
                album_meta.get("album_concept"),
            ]
        ),
        " ".join(str(item or "") for item in [payload.get("tag_list"), album_meta.get("tag_list")]),
        index,
    )
    return normalize_artist_name(raw, fallback)


def _artist_title_display(artist_name: str, title: str) -> str:
    artist = normalize_artist_name(artist_name, "AceJAM")
    clean_title = str(title or "Untitled").strip() or "Untitled"
    return f"{artist} - {clean_title}"


def _numbered_audio_filename(
    title: str,
    model_name: str,
    audio_format: str,
    *,
    artist_name: str = "",
    track_number: Any = None,
    variant: Any = None,
) -> str:
    try:
        track_no = int(track_number or 0)
    except (TypeError, ValueError):
        track_no = 0
    try:
        variant_no = int(variant or 1)
    except (TypeError, ValueError):
        variant_no = 1
    prefix = f"{track_no:02d}-" if track_no > 0 else ""
    artist_slug = safe_filename(normalize_artist_name(artist_name, "AceJAM"), "AceJAM")[:48]
    title_slug = safe_filename(str(title or "track"), "track")
    model_slug = _model_slug(model_name)
    ext = normalize_audio_format(audio_format or "wav")
    stem = f"{prefix}{artist_slug}--{title_slug}--{model_slug}--v{max(1, variant_no)}"
    if len(stem) > 180:
        title_slug = title_slug[: max(20, 180 - len(prefix) - len(artist_slug) - len(model_slug) - 10)].strip("-._") or "track"
        stem = f"{prefix}{artist_slug}--{title_slug}--{model_slug}--v{max(1, variant_no)}"
    return f"{stem}.{ext}"


def _preferred_audio_filename(params: dict[str, Any], model_name: str, index: int) -> str:
    album_meta = params.get("album_metadata") if isinstance(params.get("album_metadata"), dict) else {}
    if album_meta:
        raw_variant = album_meta.get("track_variant")
        try:
            variant = int(raw_variant)
        except (TypeError, ValueError):
            variant = index + 1
        return _numbered_audio_filename(
            params.get("title") or "track",
            album_meta.get("album_model") or model_name,
            params.get("audio_format") or "wav",
            artist_name=params.get("artist_name") or album_meta.get("artist_name") or "",
            track_number=album_meta.get("track_number"),
            variant=variant,
        )
    if params.get("preferred_filename"):
        stem = safe_filename(str(params.get("preferred_filename")), f"take-{index + 1}")
        return f"{stem}.{normalize_audio_format(params.get('audio_format') or 'wav')}"
    return _numbered_audio_filename(
        params.get("title") or "track",
        model_name,
        params.get("audio_format") or "wav",
        artist_name=params.get("artist_name") or "",
        variant=index + 1,
    )


def _save_song_entry(meta: dict[str, Any], audio_source: Path) -> dict[str, Any]:
    song_id = meta.get("id") or uuid.uuid4().hex[:12]
    song_dir = SONGS_DIR / song_id
    song_dir.mkdir(parents=True, exist_ok=True)

    extension = audio_source.suffix or ".wav"
    preferred_file = str(meta.get("preferred_audio_file") or "").strip()
    artist_name = normalize_artist_name(
        meta.get("artist_name") or meta.get("artist"),
        derive_artist_name(meta.get("title") or "", meta.get("description") or "", meta.get("tags") or ""),
    )
    if preferred_file:
        preferred_path = Path(preferred_file)
        if preferred_path.suffix:
            audio_file = f"{safe_filename(preferred_path.stem, song_id)}{preferred_path.suffix}"
        else:
            audio_file = f"{safe_filename(preferred_file, song_id)}{extension}"
    else:
        audio_file = _numbered_audio_filename(
            str(meta.get("title") or "track"),
            str(meta.get("song_model") or meta.get("album_model") or "model"),
            extension.lstrip(".") or "wav",
            artist_name=artist_name,
            track_number=meta.get("track_number"),
            variant=meta.get("track_variant") if str(meta.get("track_variant") or "").isdigit() else 1,
        )
    shutil.copyfile(audio_source, song_dir / audio_file)

    saved_meta = dict(meta)
    saved_meta.pop("preferred_audio_file", None)
    saved_meta["artist_name"] = artist_name
    saved_meta.update(
        {
            "id": song_id,
            "audio_file": audio_file,
            "created_at": saved_meta.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }
    )
    (song_dir / "meta.json").write_text(json.dumps(_jsonable(saved_meta), indent=2), encoding="utf-8")
    entry = _decorate_song(saved_meta)
    _feed_songs.insert(0, entry)
    return entry


def _run_inference(
    prompt: str,
    lyrics: str,
    audio_duration: float,
    infer_steps: int,
    seed: int,
    language: str,
    song_model: str | None = None,
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    guidance_scale: float | None = None,
) -> tuple[str, str]:
    _ensure_training_idle()
    use_random_seed = seed < 0
    with handler_lock:
        active_song_model = _ensure_song_model(song_model)
        is_turbo = "turbo" in active_song_model
        model_defaults = docs_best_model_settings(active_song_model)
        model_shift = float(model_defaults["shift"])
        if infer_steps <= 0:
            infer_steps = int(model_defaults["inference_steps"])
        if is_turbo:
            infer_steps = min(infer_steps, DOCS_BEST_TURBO_HIGH_CAP_STEPS)
        effective_guidance = float(guidance_scale if guidance_scale and guidance_scale > 0 else model_defaults["guidance_scale"])
        result = handler.generate_music(
            captions=prompt,
            lyrics=lyrics,
            audio_duration=audio_duration,
            inference_steps=infer_steps,
            guidance_scale=effective_guidance,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            use_random_seed=use_random_seed,
            seed=None if use_random_seed else seed,
            infer_method="ode",
            shift=model_shift,
            use_adg=False,
            vocal_language=_language_for_generation(language),
            batch_size=1,
        )

    if not result.get("success"):
        raise RuntimeError(result.get("error", "generation failed"))

    out_path = Path(tempfile.mkdtemp()) / "output.wav"
    _write_audio_file(result["audios"][0], out_path)
    return str(out_path), active_song_model


def _song_public_url(song_id: str, filename: str) -> str:
    return f"/media/songs/{song_id}/{filename}"


def _decorate_song(meta: dict) -> dict:
    entry = dict(meta)
    if not entry.get("artist_name"):
        entry["artist_name"] = derive_artist_name(entry.get("title") or "", entry.get("description") or "", entry.get("tags") or "")
    audio_file = entry.get("audio_file")
    if audio_file:
        entry["audio_url"] = _song_public_url(entry["id"], audio_file)
    thumb_file = entry.get("thumb_file")
    if thumb_file:
        entry["thumb_url"] = _song_public_url(entry["id"], thumb_file)
    return entry


def _load_feed_from_disk() -> list[dict]:
    songs: list[dict] = []
    if not SONGS_DIR.exists():
        return songs

    for song_dir in SONGS_DIR.iterdir():
        meta_path = song_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            songs.append(_decorate_song(meta))
        except Exception:
            continue

    songs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    print(f"[feed] Loaded {len(songs)} saved songs")
    return songs


_feed_songs = _load_feed_from_disk()
_result_extra_cache: dict[str, dict[str, Any]] = {}
_model_download_jobs: dict[str, dict[str, Any]] = {}
_model_download_lock = threading.Lock()
_model_download_runner_lock = threading.Lock()
_ollama_pull_jobs: dict[str, dict[str, Any]] = {}
_ollama_pull_lock = threading.Lock()
_album_jobs: dict[str, dict[str, Any]] = {}
_album_jobs_lock = threading.Lock()
_api_generation_tasks: dict[str, dict[str, Any]] = {}
_api_generation_tasks_lock = threading.Lock()


class ModelDownloadStarted(RuntimeError):
    def __init__(self, model_name: str, job: dict[str, Any], message: str):
        super().__init__(message)
        self.model_name = model_name
        self.job = job
        self.message = message


class OllamaPullStarted(RuntimeError):
    def __init__(self, model_name: str, job: dict[str, Any], message: str):
        super().__init__(message)
        self.model_name = model_name
        self.job = job
        self.message = message


def _downloadable_model_names() -> set[str]:
    return (set(KNOWN_ACE_STEP_MODELS) - set(OFFICIAL_UNRELEASED_MODELS)) | {
        name for name in ACE_STEP_LM_MODELS if name not in {"auto", "none"}
    }


def _ace_lm_checkpoint_dir(model_name: str) -> Path:
    return MODEL_CACHE_DIR / "checkpoints" / model_name


def _folder_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _ace_lm_abliterated_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for root in [MODEL_CACHE_DIR / "checkpoints", ACE_LM_ABLITERATED_DIR]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if "acestep-5hz-lm" not in name.lower() or "abliter" not in name.lower():
                continue
            candidates.append(
                {
                    "name": name,
                    "path": str(child),
                    "ready": _checkpoint_dir_ready(child),
                    "runner_usable": child.parent == (MODEL_CACHE_DIR / "checkpoints"),
                    "size_bytes": _folder_size(child),
                    "smoke_passed": (child / "acejam_smoke_passed.json").is_file(),
                    "metadata_file": str(child / "acejam_abliteration.json"),
                }
            )
    return sorted(candidates, key=lambda item: (not item["ready"], item["name"]))


def _ace_lm_cleanup_preview() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for model_name in ACE_STEP_LM_MODELS:
        if model_name in {"auto", "none"}:
            continue
        path = _ace_lm_checkpoint_dir(model_name)
        if path.is_dir():
            candidates.append(
                {
                    "model": model_name,
                    "path": str(path),
                    "size_bytes": _folder_size(path),
                    "downloadable": model_name in _downloadable_model_names(),
                }
            )
    abliterated = _ace_lm_abliterated_candidates()
    preferred_ready = any(
        item["ready"] and item["runner_usable"] and item["smoke_passed"] and ACE_LM_PREFERRED_MODEL.lower() in item["name"].lower()
        for item in abliterated
    )
    return {
        "safe_to_cleanup": preferred_ready,
        "requires_confirm": ACE_LM_CLEANUP_CONFIRM,
        "reason": "Cleanup is locked until an abliterated 4B ACE-Step LM has a passing local smoke marker.",
        "delete_candidates": candidates,
        "keep_candidates": abliterated,
        "total_delete_bytes": sum(int(item["size_bytes"]) for item in candidates),
    }


def _ace_lm_status_payload() -> dict[str, Any]:
    installed_lms = sorted(_installed_lm_models())
    original_lms = [name for name in ACE_STEP_LM_MODELS if name not in {"auto", "none"}]
    abliterated = _ace_lm_abliterated_candidates()
    selected = next(
        (
            item["name"]
            for item in abliterated
            if item["ready"] and item["runner_usable"] and ACE_LM_PREFERRED_MODEL.lower() in item["name"].lower()
        ),
        "",
    )
    return {
        "policy": "hybrid",
        "planner": "ollama",
        "ace_lm_usage": ["sample_mode", "format_sample", "understand_music", "CoT/metas", "official LM controls"],
        "preferred_model": ACE_LM_PREFERRED_MODEL,
        "recommended_lm_model": recommended_lm_model(set(installed_lms)),
        "official_models": [
            {
                "model": name,
                "installed": name in installed_lms,
                "downloadable": name in _downloadable_model_names(),
                "path": str(_ace_lm_checkpoint_dir(name)),
            }
            for name in original_lms
        ],
        "abliterated_models": abliterated,
        "selected_abliterated_model": selected,
        "obliteratus": {
            "repo": OBLITERATUS_REPO_URL,
            "installed": shutil.which("obliteratus") is not None,
            "license": "AGPL-3.0",
            "local_only_until_uploaded_private": True,
        },
        "upload": {
            "policy": "private_gated",
            "requires_confirm": ACE_LM_PRIVATE_UPLOAD_CONFIRM,
            "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        },
        "cleanup_preview": _ace_lm_cleanup_preview(),
        "workflow": [
            "Install/download official ACE-Step 5Hz LM 4B",
            "Run OBLITERATUS locally outside git-tracked files",
            "Copy passing abliterated model into app/model_cache/checkpoints/acestep-5Hz-lm-4B-abliterated",
            "Run create_sample/format_sample/understand_music smoke tests",
            "Write acejam_smoke_passed.json",
            "Optionally private-upload with HF_TOKEN and PRIVATE_HF_UPLOAD",
            "Only then delete ordinary ACE-Step LM checkpoints if desired",
        ],
    }


def _ace_lm_cleanup_originals(confirm: str) -> dict[str, Any]:
    preview = _ace_lm_cleanup_preview()
    if confirm != ACE_LM_CLEANUP_CONFIRM:
        raise HTTPException(status_code=400, detail=f"confirm must be {ACE_LM_CLEANUP_CONFIRM}")
    if not preview["safe_to_cleanup"]:
        raise HTTPException(status_code=409, detail=preview["reason"])
    deleted: list[dict[str, Any]] = []
    for item in preview["delete_candidates"]:
        path = Path(item["path"])
        if path.is_dir() and item.get("downloadable"):
            shutil.rmtree(path, ignore_errors=True)
            deleted.append(item)
    return {"success": True, "deleted": deleted, "status": _ace_lm_status_payload()}


def _ace_lm_private_upload(body: dict[str, Any]) -> dict[str, Any]:
    confirm = str(body.get("confirm") or "")
    repo_id = str(body.get("repo_id") or "").strip()
    raw_model_path = str(body.get("model_path") or "").strip()
    model_path = Path(raw_model_path).expanduser()
    if raw_model_path and not model_path.is_absolute():
        for candidate in [BASE_DIR / raw_model_path, BASE_DIR.parent / raw_model_path, Path.cwd() / raw_model_path]:
            if candidate.is_dir():
                model_path = candidate
                break
    if confirm != ACE_LM_PRIVATE_UPLOAD_CONFIRM:
        raise HTTPException(status_code=400, detail=f"confirm must be {ACE_LM_PRIVATE_UPLOAD_CONFIRM}")
    if not repo_id or "/" not in repo_id:
        raise HTTPException(status_code=400, detail="repo_id must be like username/model-name")
    if not parse_bool(body.get("license_confirmed"), False):
        raise HTTPException(status_code=400, detail="Confirm license/provenance review before private upload")
    if not model_path.is_dir():
        raise HTTPException(status_code=400, detail="model_path must point to a local abliterated ACE-Step LM folder")
    if "abliter" not in model_path.name.lower():
        raise HTTPException(status_code=400, detail="Refusing upload: model_path name must clearly indicate abliterated/experimental")
    if not (model_path / "acejam_smoke_passed.json").is_file():
        raise HTTPException(status_code=409, detail="Run and record a local compatibility smoke pass before upload")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise HTTPException(status_code=401, detail="Set HF_TOKEN or HUGGINGFACE_TOKEN before private upload")
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"huggingface_hub is not installed: {exc}") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_path),
        commit_message="Upload private AceJAM ACE-Step LM experiment",
    )
    return {"success": True, "repo_id": repo_id, "private": True, "model_path": str(model_path)}


def _ace_lm_mark_smoke_passed(body: dict[str, Any]) -> dict[str, Any]:
    confirm = str(body.get("confirm") or "")
    raw_model_path = str(body.get("model_path") or "").strip()
    model_path = Path(raw_model_path).expanduser()
    if raw_model_path and not model_path.is_absolute():
        for candidate in [BASE_DIR / raw_model_path, BASE_DIR.parent / raw_model_path, Path.cwd() / raw_model_path]:
            if candidate.is_dir():
                model_path = candidate
                break
    if confirm != ACE_LM_SMOKE_CONFIRM:
        raise HTTPException(status_code=400, detail=f"confirm must be {ACE_LM_SMOKE_CONFIRM}")
    if not model_path.is_dir() or not _checkpoint_dir_ready(model_path):
        raise HTTPException(status_code=400, detail="model_path must be a ready ACE-Step LM checkpoint folder")
    if "abliter" not in model_path.name.lower() or ACE_LM_PREFERRED_MODEL.lower() not in model_path.name.lower():
        raise HTTPException(status_code=400, detail="Smoke marker is only allowed for an abliterated ACE-Step 4B LM folder")
    marker = {
        "model_path": str(model_path),
        "marked_at": datetime.now(timezone.utc).isoformat(),
        "checks": body.get("checks") or ["create_sample", "format_sample", "understand_music"],
        "source": "AceJAM manual smoke gate",
    }
    (model_path / "acejam_smoke_passed.json").write_text(json.dumps(_jsonable(marker), indent=2), encoding="utf-8")
    return {"success": True, "marker": marker, "status": _ace_lm_status_payload()}


def _is_model_installed(model_name: str, ignore_active_job: bool = False) -> bool:
    if not ignore_active_job and _download_job_active(model_name):
        return False
    checkpoint_path = MODEL_CACHE_DIR / "checkpoints" / model_name
    if model_name.startswith("acestep-v15-"):
        return _checkpoint_dir_ready(checkpoint_path)
    if model_name.startswith("acestep-5Hz-lm-"):
        return model_name in {"auto", "none"} or _checkpoint_dir_ready(checkpoint_path)
    return False


def _model_download_job(model_name: str) -> dict[str, Any]:
    job = _model_download_jobs.get(model_name)
    if job:
        return dict(job)
    return {
        "id": "",
        "model_name": model_name,
        "state": "installed" if _is_model_installed(model_name) else "missing",
        "message": "Already installed" if _is_model_installed(model_name) else "Not installed",
        "started_at": None,
        "finished_at": None,
        "error": "",
    }


def _set_model_download_job(model_name: str, **updates: Any) -> dict[str, Any]:
    with _model_download_lock:
        job = _model_download_jobs.setdefault(
            model_name,
            {
                "id": uuid.uuid4().hex[:12],
                "model_name": model_name,
                "state": "queued",
                "message": "Queued",
                "started_at": None,
                "finished_at": None,
                "error": "",
            },
        )
        job.update(_jsonable(updates))
        return dict(job)


def _download_model_worker(model_name: str) -> None:
    with _model_download_runner_lock:
        _set_model_download_job(
            model_name,
            state="running",
            message=f"Downloading {model_name}...",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            error="",
        )
        try:
            checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            handler._ensure_model_downloaded(model_name, str(checkpoint_dir))
            if model_name.startswith("acestep-v15-") and model_name != "acestep-v15-turbo":
                if not (checkpoint_dir / "acestep-v15-turbo").exists():
                    _set_model_download_job(model_name, message="Downloading shared ACE-Step components...")
                    handler._ensure_model_downloaded("acestep-v15-turbo", str(checkpoint_dir))
            if not _is_model_installed(model_name, ignore_active_job=True):
                raise RuntimeError(f"{model_name} download finished but the checkpoint folder was not found.")
            _set_model_download_job(
                model_name,
                state="succeeded",
                message=f"{model_name} installed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="",
            )
        except Exception as exc:
            _set_model_download_job(
                model_name,
                state="failed",
                message=f"{model_name} download failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )


def _start_model_download(model_name: str) -> dict[str, Any]:
    model_name = str(model_name or "").strip()
    if model_name not in _downloadable_model_names():
        raise ValueError(f"{model_name or 'model'} is not a known downloadable ACE-Step model.")
    if _is_model_installed(model_name):
        return _set_model_download_job(
            model_name,
            state="succeeded",
            message=f"{model_name} already installed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error="",
        )
    existing = _model_download_jobs.get(model_name)
    if existing and existing.get("state") in {"queued", "running"}:
        return dict(existing)
    job = _set_model_download_job(
        model_name,
        id=uuid.uuid4().hex[:12],
        state="queued",
        message=f"Queued download for {model_name}",
        started_at=None,
        finished_at=None,
        error="",
    )
    thread = threading.Thread(target=_download_model_worker, args=(model_name,), daemon=True)
    thread.start()
    return job


def _start_model_download_or_raise(model_name: str, context: str = "generation") -> None:
    job = _start_model_download(model_name)
    raise ModelDownloadStarted(
        model_name,
        job,
        f"{model_name} is not installed yet. AceJAM started the download for {context}. "
        "Wait until the model is installed, then press Generate again.",
    )


def _download_started_payload(model_name: str, job: dict[str, Any], logs: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    message = (
        f"{model_name} is not installed yet. AceJAM started downloading it. "
        "Generate will be available when the download finishes."
    )
    payload = {
        "success": False,
        "download_started": True,
        "download_model": model_name,
        "download_job": _jsonable(job),
        "message": message,
        "error": "",
        "logs": list(logs or []) + [message],
    }
    payload.update(_jsonable(extra))
    return payload


def _album_missing_download_payload(models: list[str], logs: list[str], **extra: Any) -> dict[str, Any]:
    unique_models = [model for index, model in enumerate(models) if model and model not in models[:index]]
    jobs: dict[str, dict[str, Any]] = {}
    for model in unique_models:
        jobs[model] = _start_model_download(model)
    primary = unique_models[0]
    payload = _download_started_payload(primary, jobs[primary], logs, **extra)
    payload.update(
        {
            "download_models": unique_models,
            "download_jobs": _jsonable(jobs),
            "message": f"AceJAM started downloading {len(unique_models)} missing album model(s). Album generation will resume after install.",
        }
    )
    payload["logs"] = list(payload.get("logs") or []) + [
        f"Queued album model downloads: {', '.join(unique_models)}"
    ]
    return payload


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_HOST).strip() or OLLAMA_DEFAULT_HOST


def _ollama_client():
    import ollama

    return ollama.Client(host=_ollama_host())


def _ollama_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _ollama_model_name(value: Any) -> str:
    return str(_ollama_attr(value, "model", _ollama_attr(value, "name", "")) or "").strip()


def _ollama_model_size(value: Any) -> int:
    try:
        return int(_ollama_attr(value, "size", 0) or 0)
    except Exception:
        return 0


def _is_embedding_model_name(name: str) -> bool:
    return bool(re.search(r"(embed|embedding|bge|e5|gte|nomic|jina|snowflake|mxbai|arctic)", name or "", re.IGNORECASE))


def _ollama_job_snapshot(job: dict[str, Any] | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    if job is not None:
        return _jsonable(dict(job))
    with _ollama_pull_lock:
        return [_jsonable(dict(item)) for item in _ollama_pull_jobs.values()]


def _ollama_pull_job(job_id_or_model: str) -> dict[str, Any] | None:
    token = str(job_id_or_model or "").strip()
    if not token:
        return None
    with _ollama_pull_lock:
        if token in _ollama_pull_jobs:
            return dict(_ollama_pull_jobs[token])
        for job in _ollama_pull_jobs.values():
            if job.get("id") == token or job.get("model") == token:
                return dict(job)
    return None


def _ollama_model_catalog() -> dict[str, Any]:
    host = _ollama_host()
    try:
        client = _ollama_client()
        response = client.list()
        raw_models = list(_ollama_attr(response, "models", []) or [])
        details: list[dict[str, Any]] = []
        for item in raw_models:
            name = _ollama_model_name(item)
            if not name:
                continue
            size = _ollama_model_size(item)
            modified_at = _ollama_attr(item, "modified_at", "")
            digest = _ollama_attr(item, "digest", "")
            model_details = _ollama_attr(item, "details", {}) or {}
            details.append(
                {
                    "name": name,
                    "model": name,
                    "size": size,
                    "size_gb": round(size / 1e9, 2) if size else 0,
                    "modified_at": str(modified_at or ""),
                    "digest": str(digest or ""),
                    "family": str(_ollama_attr(model_details, "family", "") or (model_details.get("family", "") if isinstance(model_details, dict) else "")),
                    "parameter_size": str(_ollama_attr(model_details, "parameter_size", "") or (model_details.get("parameter_size", "") if isinstance(model_details, dict) else "")),
                    "quantization_level": str(_ollama_attr(model_details, "quantization_level", "") or (model_details.get("quantization_level", "") if isinstance(model_details, dict) else "")),
                    "kind": "embedding" if _is_embedding_model_name(name) else "chat",
                }
            )
        model_names = [item["name"] for item in details]
        embedding_models = [item["name"] for item in details if item["kind"] == "embedding"]
        chat_models = [name for name in model_names if name not in set(embedding_models)]
        running_models: list[str] = []
        if hasattr(client, "ps"):
            try:
                running_response = client.ps()
                running_raw = list(_ollama_attr(running_response, "models", []) or [])
                running_models = [
                    name
                    for name in (_ollama_model_name(item) for item in running_raw)
                    if name
                ]
            except Exception:
                running_models = []
        return {
            "success": True,
            "ready": True,
            "ollama_host": host,
            "models": model_names,
            "chat_models": chat_models,
            "embedding_models": embedding_models,
            "details": details,
            "running_models": running_models,
            "pull_jobs": _ollama_job_snapshot(),
            "planner_provider": "ollama",
            "embedding_provider": "ollama",
            "error": "",
        }
    except Exception as exc:
        print(f"[ollama_models ERROR] {exc}")
        return {
            "success": False,
            "ready": False,
            "ollama_host": host,
            "models": [],
            "chat_models": [],
            "embedding_models": [],
            "details": [],
            "running_models": [],
            "pull_jobs": _ollama_job_snapshot(),
            "planner_provider": "ollama",
            "embedding_provider": "ollama",
            "error": f"Ollama is not reachable at {host}: {exc}",
        }


def _ollama_model_installed(model_name: str) -> bool:
    model = str(model_name or "").strip()
    if not model:
        return False
    return model in set(_ollama_model_catalog().get("models") or [])


def _ollama_error_is_missing_model(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc).lower()
    return status_code == 404 or "model" in text and ("not found" in text or "try pulling" in text or "pull model" in text)


def _set_ollama_pull_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with _ollama_pull_lock:
        job = _ollama_pull_jobs.setdefault(
            job_id,
            {
                "id": job_id,
                "model": "",
                "kind": "chat",
                "state": "queued",
                "progress": 0,
                "status": "Queued",
                "logs": [],
                "started_at": None,
                "finished_at": None,
                "error": "",
            },
        )
        if "logs" in updates:
            old_logs = list(job.get("logs") or [])
            new_logs = updates.pop("logs")
            if isinstance(new_logs, list):
                job["logs"] = (old_logs + [str(item) for item in new_logs])[-60:]
            elif new_logs:
                job["logs"] = (old_logs + [str(new_logs)])[-60:]
        job.update(_jsonable(updates))
        return dict(job)


def _ollama_pull_worker(job_id: str, model_name: str) -> None:
    _set_ollama_pull_job(
        job_id,
        state="running",
        status=f"Pulling {model_name}",
        progress=0,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        error="",
        logs=[f"Pulling {model_name} from Ollama"],
    )
    try:
        client = _ollama_client()
        last_status = ""
        for progress in client.pull(model_name, stream=True):
            status_text = str(_ollama_attr(progress, "status", "") or "")
            completed = _ollama_attr(progress, "completed", None)
            total = _ollama_attr(progress, "total", None)
            digest = str(_ollama_attr(progress, "digest", "") or "")
            percent = None
            try:
                if completed is not None and total:
                    percent = max(0, min(100, round(float(completed) / float(total) * 100, 1)))
            except Exception:
                percent = None
            update = {
                "status": status_text or last_status or f"Pulling {model_name}",
                "digest": digest,
            }
            if percent is not None:
                update["progress"] = percent
            log_line = status_text
            if percent is not None:
                log_line = f"{status_text} {percent}%".strip()
            if digest:
                log_line = f"{log_line} {digest[:12]}".strip()
            if log_line and log_line != last_status:
                update["logs"] = [log_line]
                last_status = log_line
            _set_ollama_pull_job(job_id, **update)
        _set_ollama_pull_job(
            job_id,
            state="succeeded",
            progress=100,
            status=f"{model_name} installed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error="",
            logs=[f"{model_name} installed"],
        )
    except Exception as exc:
        _set_ollama_pull_job(
            job_id,
            state="failed",
            status=f"{model_name} pull failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
            logs=[f"ERROR: {exc}"],
        )


def _start_ollama_pull(model_name: str, reason: str = "", kind: str = "chat") -> dict[str, Any]:
    model = str(model_name or "").strip()
    if not model:
        raise ValueError("Ollama model name is required.")
    for job in _ollama_job_snapshot():
        if isinstance(job, dict) and job.get("model") == model and job.get("state") in {"queued", "running"}:
            return job
    if _ollama_model_installed(model):
        job_id = uuid.uuid4().hex[:12]
        return _set_ollama_pull_job(
            job_id,
            model=model,
            kind=kind,
            reason=reason,
            state="succeeded",
            progress=100,
            status=f"{model} already installed",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            error="",
            logs=[f"{model} already installed"],
        )
    job_id = uuid.uuid4().hex[:12]
    job = _set_ollama_pull_job(
        job_id,
        model=model,
        kind=kind,
        reason=reason,
        state="queued",
        progress=0,
        status=f"Queued pull for {model}",
        started_at=None,
        finished_at=None,
        error="",
        logs=[f"Queued pull for {model}"],
    )
    threading.Thread(target=_ollama_pull_worker, args=(job_id, model), daemon=True).start()
    return job


def _ensure_ollama_model_or_start_pull(model_name: str, context: str = "Ollama", kind: str = "chat") -> None:
    model = str(model_name or "").strip()
    if not model:
        return
    catalog = _ollama_model_catalog()
    if not catalog.get("ready"):
        raise RuntimeError(catalog.get("error") or "Ollama is not running.")
    if model in set(catalog.get("models") or []):
        return
    job = _start_ollama_pull(model, reason=context, kind=kind)
    raise OllamaPullStarted(
        model,
        job,
        f"{model} is not installed in Ollama. AceJAM started pulling it for {context}.",
    )


def _resolve_ollama_model_selection(model_name: str, kind: str, context: str) -> str:
    model = str(model_name or "").strip()
    if model:
        _ensure_ollama_model_or_start_pull(model, context=context, kind=kind)
        return model
    catalog = _ollama_model_catalog()
    if not catalog.get("ready"):
        raise RuntimeError(catalog.get("error") or "Ollama is not running.")
    key = "embedding_models" if kind == "embedding" else "chat_models"
    models = [str(item) for item in (catalog.get(key) or []) if str(item).strip()]
    preferred_models = ALBUM_EMBEDDING_FALLBACK_MODELS if kind == "embedding" else [DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL]
    installed = set(models)
    for preferred in preferred_models:
        if preferred in installed:
            return preferred
    if not models:
        raise RuntimeError(f"No local Ollama {kind} model installed. Pull one in Settings > Ollama Models.")
    return models[0]


def _resolve_local_llm_model_selection(provider: str, model_name: str, kind: str, context: str) -> str:
    provider_name = normalize_provider(provider)
    if provider_name == "ollama":
        return _resolve_ollama_model_selection(model_name, kind, context)
    model = str(model_name or "").strip()
    catalog = lmstudio_model_catalog()
    if not catalog.get("ready"):
        raise RuntimeError(catalog.get("error") or "LM Studio is not running. Start LM Studio local server, then refresh.")
    key = "embedding_models" if kind == "embedding" else "chat_models"
    models = [str(item) for item in (catalog.get(key) or []) if str(item).strip()]
    if model:
        if model not in set(catalog.get("models") or []):
            raise RuntimeError(f"{model} is not downloaded/available in LM Studio. Download or load it in Settings > Local LLM Models.")
        return model
    if not models:
        raise RuntimeError(f"No local LM Studio {kind} model available. Download/load one in LM Studio, then refresh AceJAM.")
    return models[0]


def _ollama_pull_started_payload(model_name: str, job: dict[str, Any], context: str = "Ollama", **extra: Any) -> dict[str, Any]:
    message = f"{model_name} is not installed in Ollama. AceJAM started pulling it for {context}."
    payload = {
        "success": False,
        "ollama_pull_started": True,
        "ollama_model": model_name,
        "ollama_pull_job": _jsonable(job),
        "message": message,
        "error": "",
        "logs": [message],
    }
    payload.update(_jsonable(extra))
    return payload


def _wait_for_model_download(model_name: str, timeout: float = 3600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _model_download_job(model_name)
        if job.get("state") == "succeeded" or _is_model_installed(model_name):
            return
        if job.get("state") == "failed":
            raise RuntimeError(job.get("error") or f"{model_name} download failed")
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {model_name} download")


def _run_advanced_generation_with_download_retry(payload: dict[str, Any], attempts: int = 2) -> dict[str, Any]:
    last_exc: ModelDownloadStarted | None = None
    for _ in range(max(1, attempts)):
        try:
            return _run_advanced_generation(payload)
        except ModelDownloadStarted as exc:
            last_exc = exc
            _wait_for_model_download(exc.model_name)
    assert last_exc is not None
    raise RuntimeError(last_exc.message)


def _album_download_candidate(model_info: dict[str, Any], album_options: dict[str, Any]) -> str:
    requested = str(album_options.get("requested_song_model") or "").strip()
    if requested and requested != "auto" and requested in _downloadable_model_names():
        return requested
    model = str(model_info.get("model") or "").strip()
    if model in _downloadable_model_names():
        return model
    strategy = str(album_options.get("song_model_strategy") or "best_installed")
    for candidate in MODEL_STRATEGIES.get(strategy, MODEL_STRATEGIES["best_installed"]).get("order", []):
        if candidate in _downloadable_model_names():
            return candidate
    return ""


def _resolve_child(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    target = root.joinpath(*parts).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise HTTPException(status_code=404, detail="File not found")
    return target


def _resolve_upload_file(upload_id: str | None) -> Path | None:
    if not upload_id:
        return None
    upload_dir = _resolve_child(UPLOADS_DIR, safe_id(upload_id))
    if not upload_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload not found")
    for item in upload_dir.iterdir():
        if item.is_file() and item.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
            return item
    raise HTTPException(status_code=404, detail="Upload has no audio file")


def _result_meta_path(result_id: str) -> Path:
    return _resolve_child(RESULTS_DIR, safe_id(result_id), "result.json")


def _load_result_meta(result_id: str) -> dict[str, Any]:
    meta_path = _result_meta_path(result_id)
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_result_audio(result_id: str | None, audio_id: str | None = None) -> Path | None:
    if not result_id:
        return None
    meta = _load_result_meta(result_id)
    selected = None
    for audio in meta.get("audios", []):
        if audio_id and audio.get("id") == audio_id:
            selected = audio
            break
        if selected is None:
            selected = audio
    if not selected:
        raise HTTPException(status_code=404, detail="Result has no audio")
    return _resolve_child(RESULTS_DIR, safe_id(result_id), selected["filename"])


def _resolve_direct_audio_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")
    if path.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio file: {path.suffix}")
    return path.resolve()


def _resolve_audio_reference(payload: dict[str, Any], upload_key: str, result_key: str) -> Path | None:
    direct_key = "reference_audio_path" if upload_key.startswith("reference") else "src_audio_path"
    legacy_key = "reference_audio" if upload_key.startswith("reference") else "src_audio"
    direct = _resolve_direct_audio_path(get_param(payload, direct_key, payload.get(legacy_key)))
    if direct is not None:
        return direct
    upload_path = _resolve_upload_file(payload.get(upload_key))
    if upload_path is not None:
        return upload_path
    return _resolve_result_audio(payload.get(result_key), payload.get(f"{result_key}_audio_id"))


def _model_capabilities() -> dict[str, Any]:
    installed = _installed_acestep_models()
    return {
        model: {
            "label": _song_model_label(model),
            "tasks": supported_tasks_for_model(model),
            "installed": model in installed,
        }
        for model in _available_acestep_models()
    }


def _official_runner_status() -> dict[str, Any]:
    missing = []
    if not OFFICIAL_ACE_STEP_DIR.exists():
        missing.append("app/vendor/ACE-Step-1.5")
    if not OFFICIAL_RUNNER_SCRIPT.exists():
        missing.append("app/official_runner.py")
    return {
        "available": not missing,
        "vendor_path": str(OFFICIAL_ACE_STEP_DIR),
        "runner_path": str(OFFICIAL_RUNNER_SCRIPT),
        "missing": missing,
        "routing_note": "Used when Custom enables official-only ACE-Step 1.5 controls.",
    }


def _songwriting_toolkit_payload() -> dict[str, Any]:
    return toolkit_payload(_installed_acestep_models())


def _json_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return split_terms(stripped)
    return [value]


def _album_options_from_payload(payload: dict[str, Any], song_model: str = "auto") -> dict[str, Any]:
    strategy = str(payload.get("song_model_strategy") or "all_models_album")
    quality_profile = normalize_quality_profile(payload.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    payload_song_model = str(payload.get("requested_song_model") or payload.get("song_model") or "").strip()
    selected_song_model = str(song_model or "").strip()
    requested_song_model = (
        selected_song_model
        if selected_song_model and selected_song_model != "auto"
        else payload_song_model
    )
    if strategy != "selected":
        requested_song_model = "auto"
    installed_models = sorted(_installed_acestep_models())
    default_model = ALBUM_FINAL_MODEL if strategy != "selected" else (requested_song_model or ALBUM_FINAL_MODEL)
    model_defaults = quality_profile_model_settings(default_model, quality_profile)
    contract_source = "\n\n".join(
        part for part in [
            str(payload.get("user_prompt") or payload.get("prompt") or ""),
            str(payload.get("album_title") or payload.get("album_name") or ""),
            str(payload.get("concept") or ""),
            json.dumps(payload.get("tracks") or payload.get("planned_tracks") or [], ensure_ascii=True, default=str)[:6000],
        ]
        if part
    )
    user_album_contract = payload.get("user_album_contract")
    if not isinstance(user_album_contract, dict):
        user_album_contract = extract_user_album_contract(
            contract_source,
            int(payload.get("num_tracks") or 0) or None,
            str(payload.get("language") or payload.get("target_language") or "en"),
            payload,
        )
    return {
        "requested_song_model": requested_song_model or "auto",
        "song_model_strategy": strategy,
        "final_song_model": ALBUM_FINAL_MODEL,
        "prompt_kit_version": str(payload.get("prompt_kit_version") or PROMPT_KIT_VERSION),
        "planner_lm_provider": normalize_provider(payload.get("planner_lm_provider") or payload.get("planner_provider") or "ollama"),
        "planner_model": str(payload.get("planner_model") or payload.get("planner_ollama_model") or payload.get("ollama_model") or ""),
        "planner_ollama_model": str(payload.get("planner_ollama_model") or payload.get("ollama_model") or ""),
        "embedding_lm_provider": normalize_provider(payload.get("embedding_lm_provider") or payload.get("embedding_provider") or payload.get("planner_lm_provider") or "ollama"),
        "ace_lm_model": _requested_ace_lm_model(payload),
        "album_model_portfolio": album_model_portfolio(installed_models),
        "quality_target": str(payload.get("quality_target") or "hit"),
        "quality_profile": quality_profile,
        "tag_packs": _json_list(payload.get("tag_packs")),
        "custom_tags": payload.get("custom_tags") or "",
        "negative_tags": payload.get("negative_tags") or "",
        "lyric_density": str(payload.get("lyric_density") or "dense"),
        "rhyme_density": clamp_float(payload.get("rhyme_density"), 0.8, 0.0, 1.0),
        "metaphor_density": clamp_float(payload.get("metaphor_density"), 0.7, 0.0, 1.0),
        "hook_intensity": clamp_float(payload.get("hook_intensity"), 0.85, 0.0, 1.0),
        "structure_preset": str(payload.get("structure_preset") or "auto"),
        "bpm_strategy": str(payload.get("bpm_strategy") or "varied"),
        "key_strategy": str(payload.get("key_strategy") or "related"),
        "inspiration_queries": payload.get("inspiration_queries") or "",
        "use_web_inspiration": parse_bool(payload.get("use_web_inspiration"), False),
        "track_variants": clamp_int(payload.get("track_variants"), 1, 1, MAX_BATCH_SIZE),
        "seed": str(payload.get("seed") or "-1"),
        "inference_steps": clamp_int(payload.get("inference_steps"), model_defaults["inference_steps"], 1, 200),
        "guidance_scale": clamp_float(payload.get("guidance_scale"), model_defaults["guidance_scale"], 1.0, 15.0),
        "shift": clamp_float(payload.get("shift"), model_defaults["shift"], 1.0, 5.0),
        "infer_method": str(payload.get("infer_method") or model_defaults["infer_method"]),
        "sampler_mode": str(payload.get("sampler_mode") or model_defaults["sampler_mode"]),
        "audio_format": str(payload.get("audio_format") or model_defaults["audio_format"]),
        "thinking": parse_bool(payload.get("thinking"), DOCS_BEST_LM_DEFAULTS["thinking"]),
        "use_format": parse_bool(payload.get("use_format"), False),
        "lm_temperature": clamp_float(payload.get("lm_temperature"), DOCS_BEST_LM_DEFAULTS["lm_temperature"], 0.0, 2.0),
        "lm_cfg_scale": clamp_float(payload.get("lm_cfg_scale"), DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"], 0.0, 10.0),
        "lm_top_k": clamp_int(payload.get("lm_top_k"), DOCS_BEST_LM_DEFAULTS["lm_top_k"], 0, 200),
        "lm_top_p": clamp_float(payload.get("lm_top_p"), DOCS_BEST_LM_DEFAULTS["lm_top_p"], 0.0, 1.0),
        "use_cot_metas": parse_bool(payload.get("use_cot_metas"), DOCS_BEST_LM_DEFAULTS["use_cot_metas"]),
        "use_cot_caption": parse_bool(payload.get("use_cot_caption"), DOCS_BEST_LM_DEFAULTS["use_cot_caption"]),
        "use_cot_lyrics": parse_bool(payload.get("use_cot_lyrics"), DOCS_BEST_LM_DEFAULTS["use_cot_lyrics"]),
        "use_cot_language": parse_bool(payload.get("use_cot_language"), DOCS_BEST_LM_DEFAULTS["use_cot_language"]),
        "use_constrained_decoding": parse_bool(payload.get("use_constrained_decoding"), DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"]),
        "auto_score": parse_bool(payload.get("auto_score"), False),
        "auto_lrc": parse_bool(payload.get("auto_lrc"), False),
        "return_audio_codes": parse_bool(payload.get("return_audio_codes"), False),
        "save_to_library": parse_bool(payload.get("save_to_library"), True),
        "installed_models": installed_models,
        "global_caption": str(payload.get("global_caption") or ""),
        "album_title": str(payload.get("album_title") or user_album_contract.get("album_title") or ""),
        "user_album_contract": user_album_contract,
        "input_contract_applied": bool(user_album_contract.get("applied")),
        "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
        "blocked_unsafe_count": int(user_album_contract.get("blocked_unsafe_count") or 0),
    }


def _merge_nested_generation_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload or {})
    nested: dict[str, Any] = {}
    for key in ["metas", "metadata", "user_metadata", "param_obj"]:
        value = merged.get(key)
        if isinstance(value, dict):
            nested.update(value)
        elif isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    nested.update(parsed)
            except json.JSONDecodeError:
                pass
    for source, target in {
        "bpm": "bpm",
        "duration": "duration",
        "key": "key_scale",
        "key_scale": "key_scale",
        "keyscale": "key_scale",
        "time_signature": "time_signature",
        "timesignature": "time_signature",
        "language": "vocal_language",
        "vocal_language": "vocal_language",
    }.items():
        if target not in merged or merged.get(target) in [None, ""]:
            if nested.get(source) not in [None, ""]:
                merged[target] = nested[source]
    return merged


def _payload_has_any(payload: dict[str, Any], keys: list[str]) -> bool:
    return any(key in payload for key in keys)


def _bpm_from_payload(payload: dict[str, Any]) -> int | None:
    if not _payload_has_any(payload, ["bpm"]):
        return DEFAULT_BPM
    value = payload.get("bpm")
    if value is None or str(value).strip().lower() in {"", "auto", "none", "n/a", "na"}:
        return None
    return clamp_int(value, DEFAULT_BPM, BPM_MIN, BPM_MAX)


def _key_scale_from_payload(payload: dict[str, Any]) -> str:
    key_fields = ["key_scale", "keyscale", "keyScale", "key"]
    if not _payload_has_any(payload, key_fields):
        return DEFAULT_KEY_SCALE
    return normalize_key_scale(get_param(payload, "key_scale", ""))


def _time_signature_from_payload(payload: dict[str, Any]) -> str:
    time_fields = ["time_signature", "timesignature", "timeSignature"]
    if not _payload_has_any(payload, time_fields):
        return "4"
    value = str(get_param(payload, "time_signature", "") or "").strip()
    if value.lower() in {"auto", "none", "n/a", "na"}:
        return ""
    if value:
        try:
            if int(float(value)) in VALID_TIME_SIGNATURES:
                return str(int(float(value)))
        except ValueError:
            return ""
    return ""


def _merge_song_album_metadata(song_id: str, extra: dict[str, Any]) -> None:
    if not song_id:
        return
    song_dir = SONGS_DIR / safe_id(song_id)
    meta_path = song_dir / "meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(_jsonable(extra))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for index, song in enumerate(_feed_songs):
        if song.get("id") == song_id:
            _feed_songs[index] = _decorate_song(meta)
            break


def _album_manifest_path(album_id: str) -> Path:
    return _resolve_child(ALBUMS_DIR, safe_id(album_id), "album.json")


def _write_album_manifest(album_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    album_dir = _resolve_child(ALBUMS_DIR, safe_id(album_id))
    album_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["album_id"] = album_id
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _album_manifest_path(album_id).write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def _load_album_manifest(album_id: str) -> dict[str, Any]:
    path = _album_manifest_path(album_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Album not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _audio_path_from_item(audio: dict[str, Any]) -> Path | None:
    result_id = audio.get("result_id")
    filename = audio.get("filename")
    if result_id and filename:
        path = _resolve_child(RESULTS_DIR, safe_id(str(result_id)), str(filename))
        if path.is_file():
            return path
    song_id = audio.get("song_id")
    library_url = str(audio.get("library_url") or audio.get("audio_url") or "")
    if song_id and "/media/songs/" in library_url:
        filename = library_url.rsplit("/", 1)[-1]
        path = _resolve_child(SONGS_DIR, safe_id(str(song_id)), filename)
        if path.is_file():
            return path
    return None


def _add_album_audio_to_zip(zipf: zipfile.ZipFile, track: dict[str, Any], audio: dict[str, Any], prefix: str = "") -> None:
    path = _audio_path_from_item(audio)
    if not path:
        return
    filename = str(audio.get("filename") or path.name)
    arcname = f"{prefix}{filename}" if prefix else filename
    zipf.write(path, arcname)


def _build_album_zip(album_id: str) -> Path:
    manifest = _load_album_manifest(album_id)
    zip_path = _resolve_child(ALBUMS_DIR, safe_id(album_id), f"{safe_filename(album_id, 'album')}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("album.json", json.dumps(_jsonable(manifest), indent=2))
        for track in manifest.get("tracks", []):
            for audio in track.get("audios", []):
                _add_album_audio_to_zip(zipf, track, audio)
    return zip_path


def _build_album_family_zip(family_id: str) -> Path:
    family_manifest = _load_album_manifest(family_id)
    zip_path = _resolve_child(ALBUMS_DIR, safe_id(family_id), f"{safe_filename(family_id, 'album-family')}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("album_family.json", json.dumps(_jsonable(family_manifest), indent=2))
        for model_album in family_manifest.get("model_albums", []):
            album_id = str(model_album.get("album_id") or "")
            if not album_id:
                continue
            try:
                manifest = _load_album_manifest(album_id)
            except Exception:
                continue
            folder = safe_filename(str(model_album.get("album_model") or album_id), "album")
            zipf.writestr(f"{folder}/album.json", json.dumps(_jsonable(manifest), indent=2))
            for track in manifest.get("tracks", []):
                for audio in track.get("audios", []):
                    _add_album_audio_to_zip(zipf, track, audio, prefix=f"{folder}/")
    return zip_path


DELETE_GENERATED_CONFIRM = "DELETE_GENERATED_OUTPUTS"
DELETE_ALBUM_CONFIRM = "DELETE_ALBUM"
DELETE_ALBUM_FAMILY_CONFIRM = "DELETE_ALBUM_FAMILY"
SONG_PORTFOLIO_STRATEGY = "all_models_song"


def _count_generated_outputs() -> dict[str, Any]:
    def count_dirs(root: Path) -> int:
        return sum(1 for item in root.iterdir() if item.is_dir()) if root.exists() else 0

    def count_audio(root: Path) -> int:
        if not root.exists():
            return 0
        return sum(
            1
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower().lstrip(".") in {"wav", "flac", "ogg", "mp3", "opus", "aac"}
        )

    return {
        "songs": count_dirs(SONGS_DIR),
        "results": count_dirs(RESULTS_DIR),
        "albums": count_dirs(ALBUMS_DIR),
        "audio_files": count_audio(SONGS_DIR) + count_audio(RESULTS_DIR),
        "scope": ["songs", "results", "albums"],
        "preserved": ["uploads", "lora_datasets", "loras", "training"],
        "confirm": DELETE_GENERATED_CONFIRM,
    }


def _delete_result_ids(result_ids: set[str]) -> int:
    deleted = 0
    for result_id in {safe_id(str(item)) for item in result_ids if item}:
        result_dir = _resolve_child(RESULTS_DIR, result_id)
        if result_dir.is_dir():
            shutil.rmtree(result_dir, ignore_errors=True)
            deleted += 1
            _result_extra_cache.pop(result_id, None)
    return deleted


def _delete_songs_matching(*, album_id: str | None = None, family_id: str | None = None) -> dict[str, Any]:
    deleted_songs = 0
    result_ids: set[str] = set()
    album_id = str(album_id or "")
    family_id = str(family_id or "")
    if SONGS_DIR.exists():
        for song_dir in list(SONGS_DIR.iterdir()):
            if not song_dir.is_dir():
                continue
            meta_path = song_dir / "meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            except Exception:
                meta = {}
            matches_album = bool(album_id and str(meta.get("album_id") or "") == album_id)
            matches_family = bool(family_id and str(meta.get("album_family_id") or "") == family_id)
            if matches_album or matches_family:
                if meta.get("result_id"):
                    result_ids.add(str(meta["result_id"]))
                shutil.rmtree(song_dir, ignore_errors=True)
                deleted_songs += 1
    if album_id:
        _feed_songs[:] = [song for song in _feed_songs if str(song.get("album_id") or "") != album_id]
    if family_id:
        _feed_songs[:] = [song for song in _feed_songs if str(song.get("album_family_id") or "") != family_id]
    return {"songs": deleted_songs, "results": _delete_result_ids(result_ids)}


def _delete_album(album_id: str, confirm: str) -> dict[str, Any]:
    if confirm != DELETE_ALBUM_CONFIRM:
        raise HTTPException(status_code=400, detail=f"confirm must be {DELETE_ALBUM_CONFIRM}")
    album_id = str(album_id or "").strip()
    if not album_id:
        raise HTTPException(status_code=400, detail="album_id is required")
    album_dir = _resolve_child(ALBUMS_DIR, safe_id(album_id))
    if not album_dir.is_dir():
        raise HTTPException(status_code=404, detail="Album not found")
    result_ids: set[str] = set()
    try:
        manifest = _load_album_manifest(album_id)
        for track in manifest.get("tracks", []):
            for audio in track.get("audios", []):
                if audio.get("result_id"):
                    result_ids.add(str(audio["result_id"]))
    except Exception:
        pass
    song_delete = _delete_songs_matching(album_id=album_id)
    result_ids.update(str(item) for item in song_delete.get("result_ids", []) if item)
    deleted_results = int(song_delete.get("results") or 0) + _delete_result_ids(result_ids)
    shutil.rmtree(album_dir, ignore_errors=True)
    return {
        "success": True,
        "album_id": album_id,
        "deleted": {"albums": 1, "songs": song_delete["songs"], "results": deleted_results},
        "remaining": _count_generated_outputs(),
    }


def _delete_album_family(family_id: str, confirm: str) -> dict[str, Any]:
    if confirm != DELETE_ALBUM_FAMILY_CONFIRM:
        raise HTTPException(status_code=400, detail=f"confirm must be {DELETE_ALBUM_FAMILY_CONFIRM}")
    family_id = str(family_id or "").strip()
    if not family_id:
        raise HTTPException(status_code=400, detail="family_id is required")
    family_dir = _resolve_child(ALBUMS_DIR, safe_id(family_id))
    if not family_dir.is_dir():
        raise HTTPException(status_code=404, detail="Album family not found")
    album_ids: set[str] = set()
    result_ids: set[str] = set()
    try:
        manifest = _load_album_manifest(family_id)
        for model_album in manifest.get("model_albums", []):
            if model_album.get("album_id"):
                album_ids.add(str(model_album["album_id"]))
            for track in model_album.get("tracks", []):
                for audio in track.get("audios", []):
                    if audio.get("result_id"):
                        result_ids.add(str(audio["result_id"]))
    except Exception:
        pass
    deleted_album_dirs = 0
    for album_id in album_ids:
        album_dir = _resolve_child(ALBUMS_DIR, safe_id(album_id))
        if album_dir.is_dir():
            shutil.rmtree(album_dir, ignore_errors=True)
            deleted_album_dirs += 1
    song_delete = _delete_songs_matching(family_id=family_id)
    deleted_results = int(song_delete.get("results") or 0) + _delete_result_ids(result_ids)
    shutil.rmtree(family_dir, ignore_errors=True)
    return {
        "success": True,
        "album_family_id": family_id,
        "deleted": {"album_families": 1, "albums": deleted_album_dirs, "songs": song_delete["songs"], "results": deleted_results},
        "remaining": _count_generated_outputs(),
    }


def _delete_generated_outputs(confirm: str) -> dict[str, Any]:
    if confirm != DELETE_GENERATED_CONFIRM:
        raise HTTPException(status_code=400, detail=f"confirm must be {DELETE_GENERATED_CONFIRM}")
    before = _count_generated_outputs()
    for root in [SONGS_DIR, RESULTS_DIR, ALBUMS_DIR]:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
    _feed_songs.clear()
    _result_extra_cache.clear()
    with _api_generation_tasks_lock:
        _api_generation_tasks.clear()
    return {"success": True, "deleted": before, "remaining": _count_generated_outputs()}


def _parse_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _merge_nested_generation_metadata(payload)
    task_type = normalize_task_type(payload.get("task_type"))
    payload = normalize_generation_text_fields(payload, task_type=task_type)
    overflow_policy = str(payload.get("lyrics_overflow_policy") or "auto_fit").strip().lower() or "auto_fit"
    exact_lyrics = parse_bool(payload.get("exact_lyrics") or payload.get("locked_lyrics"), False)
    raw_lyrics_len = len(str(payload.get("lyrics") or ""))
    if raw_lyrics_len > ACE_STEP_LYRICS_CHAR_LIMIT and (exact_lyrics or overflow_policy in {"error", "strict", "fail"}):
        raise ValueError(
            f"ACE-Step accepts max {ACE_STEP_LYRICS_CHAR_LIMIT} lyrics characters per render; "
            f"got {raw_lyrics_len}. Use auto_fit or split the song into parts."
        )
    payload = apply_ace_step_text_budget(payload, task_type=task_type)
    payload = _apply_studio_lm_policy(payload)
    quality_profile = normalize_quality_profile(payload.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    song_model = _song_model_for_quality_profile(get_param(payload, "song_model", payload.get("song_model")), quality_profile, task_type)
    ensure_task_supported(song_model, task_type)
    if song_model in OFFICIAL_UNRELEASED_MODELS:
        raise ValueError(f"{song_model} is official but unreleased; it cannot be downloaded or used yet.")
    if song_model not in _installed_acestep_models():
        if song_model in _downloadable_model_names():
            _start_model_download_or_raise(song_model, context=f"{task_type} generation")
        raise ValueError(f"{song_model} is not installed and is not in the known ACE-Step download list.")
    is_album_track = bool(payload.get("album_metadata") or payload.get("album_id") or payload.get("track_variant"))
    requested_batch_size = payload.get("batch_size")
    default_batch_size = 3 if quality_profile == "chart_master" and task_type == "text2music" and not is_album_track else 1
    batch_size = clamp_int(requested_batch_size, default_batch_size, 1, MAX_BATCH_SIZE)
    duration = clamp_float(get_param(payload, "duration"), 60.0, DURATION_MIN, DURATION_MAX)
    model_defaults = quality_profile_model_settings(song_model, quality_profile)
    is_turbo = "turbo" in song_model
    raw_steps = payload.get("inference_steps", payload.get("infer_step"))
    if raw_steps in [None, "", "auto"]:
        default_steps = _quality_default_steps(song_model, quality_profile)
    else:
        try:
            default_steps = int(raw_steps)
        except (TypeError, ValueError):
            default_steps = _quality_default_steps(song_model, quality_profile)
    inference_steps = clamp_int(default_steps, default_steps, 1, 200)
    if is_turbo and inference_steps > DOCS_BEST_TURBO_HIGH_CAP_STEPS:
        inference_steps = min(inference_steps, DOCS_BEST_TURBO_HIGH_CAP_STEPS)

    bpm = _bpm_from_payload(payload)
    time_signature = _time_signature_from_payload(payload)

    requested_lm_model = _requested_ace_lm_model(payload)
    official_used = _active_official_fields(payload, task_type, official_fields_used(payload))
    use_official = bool(official_used) or _quality_lm_controls_enabled(payload, task_type)
    requested_format = str(payload.get("audio_format") or (model_defaults["audio_format"] if use_official else "wav")).strip().lower().lstrip(".")
    if use_official and requested_format == "ogg":
        raise ValueError("OGG is only available in the fast AceJAM runner. Use wav/flac/mp3/opus/aac/wav32 with official ACE-Step controls.")
    vocal_language = _language_for_generation(str(get_param(payload, "vocal_language", "unknown") or "unknown"))
    track_names = normalize_track_names(payload.get("track_names") or payload.get("track_name"))
    instruction = str(payload.get("instruction") or "").strip() or build_task_instruction(task_type, track_names)

    if task_type in {"cover", "repaint", "extract", "lego", "complete"}:
        has_source = bool(payload.get("src_audio_id") or payload.get("src_result_id") or payload.get("audio_code_string"))
        if not has_source:
            raise ValueError(f"{task_type} requires source audio, a source result, or audio codes")
    if task_type in {"extract", "lego"} and not track_names:
        raise ValueError(f"{task_type} requires a track name")
    if task_type == "complete" and not track_names:
        raise ValueError("complete requires one or more track names")

    lyrics_text = str(payload.get("lyrics") or "")
    instrumental = parse_bool(payload.get("instrumental"), False)
    sample_mode = parse_bool(payload.get("sample_mode"), False)
    sample_query = str(get_param(payload, "sample_query", "") or "").strip()
    lm_controls = _explicit_ace_lm_controls(payload)
    supplied_vocal_lyrics = bool(lyrics_text.strip() and lyrics_text.strip().lower() != "[instrumental]")
    lm_quality_defaults = requested_lm_model != "none" and task_type not in DOCS_BEST_SOURCE_TASK_LM_SKIPS and not supplied_vocal_lyrics
    if lm_controls and requested_lm_model == "none":
        raise ValueError(
            "ACE-Step LM controls require ace_lm_model. "
            f"Set ace_lm_model to auto/acestep-5Hz-lm-4B or disable: {', '.join(lm_controls)}."
        )
    if needs_vocal_lyrics(
        task_type=task_type,
        instrumental=instrumental,
        lyrics=lyrics_text,
        sample_mode=sample_mode,
        sample_query=sample_query,
    ):
        warnings = " ".join(str(item) for item in payload.get("payload_warnings", []) if item)
        suffix = f" Payload warnings: {warnings}" if warnings else ""
        raise ValueError(
            "Text2Music vocal generation needs lyrics. Add lyrics, use Write Lyrics/Format, or enable Instrumental."
            + suffix
        )

    title = str(payload.get("title") or "").strip() or "Untitled"
    artist_name = _artist_name_from_payload(payload, title=title)
    lora_request = _lora_adapter_request(payload)
    if lora_request["use_lora"] and not lora_request["lora_adapter_path"]:
        raise ValueError("Use adapter is enabled but no LoRA adapter was selected")
    parsed = {
        "ui_mode": str(payload.get("ui_mode") or task_type),
        "quality_profile": quality_profile,
        "task_type": task_type,
        "caption": str(payload.get("caption") or ""),
        "global_caption": str(payload.get("global_caption") or ""),
        "lyrics": lyrics_text,
        "caption_source": str(payload.get("caption_source") or "caption"),
        "lyrics_source": str(payload.get("lyrics_source") or "lyrics"),
        "tag_list": list(payload.get("tag_list") or []),
        "payload_warnings": list(payload.get("payload_warnings") or []),
        "ace_step_text_budget": dict(payload.get("ace_step_text_budget") or {}),
        "instrumental": instrumental,
        "duration": duration,
        "bpm": bpm,
        "key_scale": _key_scale_from_payload(payload),
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "batch_size": batch_size,
        "seed": str(payload.get("seeds") or payload.get("seed") or "-1"),
        "use_random_seed": parse_bool(payload.get("use_random_seed"), str(payload.get("seeds") or payload.get("seed") or "-1").strip() in {"", "-1"}),
        "song_model": song_model,
        "ace_lm_model": requested_lm_model,
        "planner_lm_provider": normalize_provider(payload.get("planner_lm_provider") or payload.get("planner_provider") or "ollama"),
        "planner_model": str(payload.get("planner_model") or payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        "planner_ollama_model": str(payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        "reference_audio": _resolve_audio_reference(payload, "reference_audio_id", "reference_result_id"),
        "src_audio": _resolve_audio_reference(payload, "src_audio_id", "src_result_id"),
        "audio_code_string": str(get_param(payload, "audio_code_string", "") or ""),
        "repainting_start": clamp_float(payload.get("repainting_start"), 0.0, -DURATION_MAX, DURATION_MAX),
        "repainting_end": None if payload.get("repainting_end") in [None, "", "end"] else clamp_float(payload.get("repainting_end"), -1.0, -1.0, DURATION_MAX),
        "instruction": instruction,
        "audio_cover_strength": clamp_float(get_param(payload, "audio_cover_strength", 1.0), 1.0, 0.0, 1.0),
        "cover_noise_strength": clamp_float(payload.get("cover_noise_strength"), 0.0, 0.0, 1.0),
        "inference_steps": inference_steps,
        "guidance_scale": clamp_float(payload.get("guidance_scale"), model_defaults["guidance_scale"], 1.0, 15.0),
        "shift": clamp_float(payload.get("shift"), model_defaults["shift"], 1.0, 5.0),
        "infer_method": "sde" if str(payload.get("infer_method") or model_defaults["infer_method"]).lower() == "sde" else "ode",
        "sampler_mode": "euler" if str(payload.get("sampler_mode") or model_defaults["sampler_mode"]).lower() == "euler" else "heun",
        "velocity_norm_threshold": clamp_float(payload.get("velocity_norm_threshold"), 0.0, 0.0, 20.0),
        "velocity_ema_factor": clamp_float(payload.get("velocity_ema_factor"), 0.0, 0.0, 1.0),
        "use_adg": parse_bool(payload.get("use_adg"), bool(model_defaults.get("use_adg", False))),
        "cfg_interval_start": clamp_float(payload.get("cfg_interval_start"), 0.0, 0.0, 1.0),
        "cfg_interval_end": clamp_float(payload.get("cfg_interval_end"), 1.0, 0.0, 1.0),
        "timesteps": parse_timesteps(payload.get("timesteps")),
        "audio_format": normalize_audio_format(payload.get("audio_format") or (model_defaults["audio_format"] if use_official else "wav"), allow_official=use_official),
        "mp3_bitrate": str(payload.get("mp3_bitrate") or "128k").strip() or "128k",
        "mp3_sample_rate": clamp_int(payload.get("mp3_sample_rate"), 48000, 16000, 48000),
        "auto_score": parse_bool(payload.get("auto_score"), False),
        "auto_lrc": parse_bool(payload.get("auto_lrc"), False),
        "return_audio_codes": parse_bool(payload.get("return_audio_codes"), False),
        "save_to_library": parse_bool(payload.get("save_to_library"), False),
        "title": title,
        "artist_name": artist_name,
        "description": str(payload.get("description") or "").strip(),
        **lora_request,
        "album_metadata": payload.get("album_metadata") if isinstance(payload.get("album_metadata"), dict) else {},
        "track_names": track_names,
        "thinking": parse_bool(payload.get("thinking"), lm_quality_defaults),
        "sample_mode": sample_mode,
        "sample_query": sample_query,
        "use_format": parse_bool(get_param(payload, "use_format"), lm_quality_defaults),
        "lm_temperature": clamp_float(payload.get("lm_temperature"), DOCS_BEST_LM_DEFAULTS["lm_temperature"] if lm_quality_defaults else 0.85, 0.0, 2.0),
        "lm_cfg_scale": clamp_float(payload.get("lm_cfg_scale"), DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"] if lm_quality_defaults else 2.0, 0.0, 10.0),
        "lm_repetition_penalty": clamp_float(payload.get("lm_repetition_penalty") or payload.get("repetition_penalty"), 1.0, 0.1, 4.0),
        "lm_top_k": clamp_int(payload.get("lm_top_k"), DOCS_BEST_LM_DEFAULTS["lm_top_k"] if lm_quality_defaults else 0, 0, 200),
        "lm_top_p": clamp_float(payload.get("lm_top_p"), DOCS_BEST_LM_DEFAULTS["lm_top_p"] if lm_quality_defaults else 0.9, 0.0, 1.0),
        "lm_negative_prompt": str(payload.get("lm_negative_prompt") or "NO USER INPUT"),
        "lm_backend": _normalize_lm_backend(payload.get("lm_backend")),
        "use_cot_metas": parse_bool(payload.get("use_cot_metas"), lm_quality_defaults),
        "use_cot_caption": parse_bool(payload.get("use_cot_caption"), lm_quality_defaults),
        "use_cot_lyrics": parse_bool(payload.get("use_cot_lyrics"), False),
        "use_cot_language": parse_bool(payload.get("use_cot_language"), lm_quality_defaults),
        "allow_lm_batch": parse_bool(payload.get("allow_lm_batch"), False),
        "lm_batch_chunk_size": clamp_int(payload.get("lm_batch_chunk_size"), 8, 1, 64),
        "use_constrained_decoding": parse_bool(payload.get("use_constrained_decoding"), True),
        "constrained_decoding_debug": parse_bool(payload.get("constrained_decoding_debug"), False),
        "chunk_mask_mode": "explicit" if str(payload.get("chunk_mask_mode")).lower() == "explicit" else "auto",
        "repaint_latent_crossfade_frames": clamp_int(payload.get("repaint_latent_crossfade_frames"), 10, 0, 250),
        "repaint_wav_crossfade_sec": clamp_float(payload.get("repaint_wav_crossfade_sec"), 0.0, 0.0, 20.0),
        "repaint_mode": str(payload.get("repaint_mode") or "balanced").strip().lower()
        if str(payload.get("repaint_mode") or "balanced").strip().lower() in {"conservative", "balanced", "aggressive"}
        else "balanced",
        "repaint_strength": clamp_float(payload.get("repaint_strength"), 0.5, 0.0, 1.0),
        "enable_normalization": parse_bool(payload.get("enable_normalization"), True),
        "normalization_db": clamp_float(payload.get("normalization_db"), -1.0, -24.0, 0.0),
        "fade_in_duration": clamp_float(payload.get("fade_in_duration"), 0.0, 0.0, 20.0),
        "fade_out_duration": clamp_float(payload.get("fade_out_duration"), 0.0, 0.0, 20.0),
        "latent_shift": clamp_float(payload.get("latent_shift"), 0.0, -2.0, 2.0),
        "latent_rescale": clamp_float(payload.get("latent_rescale"), 1.0, 0.1, 3.0),
        "device": str(payload.get("device") or "auto").strip() or "auto",
        "dtype": str(payload.get("dtype") or "auto").strip() or "auto",
        "use_flash_attention": payload.get("use_flash_attention", "auto"),
        "compile_model": parse_bool(payload.get("compile_model"), False),
        "offload_to_cpu": parse_bool(payload.get("offload_to_cpu"), False),
        "offload_dit_to_cpu": parse_bool(payload.get("offload_dit_to_cpu"), False),
        "lm_device": str(payload.get("lm_device") or "auto").strip() or "auto",
        "lm_dtype": str(payload.get("lm_dtype") or "auto").strip() or "auto",
        "lm_offload_to_cpu": parse_bool(payload.get("lm_offload_to_cpu"), False),
        "official_fields": official_used,
        "requires_official_runner": use_official,
        "runner_plan": "official" if use_official else "fast",
    }
    settings_compliance = ace_step_settings_compliance(
        parsed,
        task_type=task_type,
        song_model=song_model,
        runner_plan=parsed["runner_plan"],
    )
    parsed["settings_policy_version"] = settings_compliance["version"]
    parsed["settings_compliance"] = settings_compliance
    _attach_pro_preflight(parsed)
    return parsed


def _slice_batch_tensor(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > index:
        return value[index : index + 1]
    return value


def _extra_for_index(extra: dict[str, Any], index: int) -> dict[str, Any]:
    return {key: _slice_batch_tensor(value, index) for key, value in extra.items()}


def _calculate_lrc(extra: dict[str, Any], duration: float, language: str, inference_steps: int, seed: int) -> dict[str, Any]:
    required = ["pred_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents", "lyric_token_idss"]
    if any(extra.get(key) is None for key in required):
        return {"success": False, "error": "LRC tensors are unavailable for this result"}
    with handler_lock:
        return handler.get_lyric_timestamp(
            pred_latent=extra["pred_latents"],
            encoder_hidden_states=extra["encoder_hidden_states"],
            encoder_attention_mask=extra["encoder_attention_mask"],
            context_latents=extra["context_latents"],
            lyric_token_ids=extra["lyric_token_idss"],
            total_duration_seconds=duration,
            vocal_language=language,
            inference_steps=inference_steps,
            seed=seed,
        )


def _calculate_score(extra: dict[str, Any], language: str, inference_steps: int, seed: int) -> dict[str, Any]:
    required = ["pred_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents", "lyric_token_idss"]
    if any(extra.get(key) is None for key in required):
        return {"success": False, "error": "Score tensors are unavailable for this result"}
    with handler_lock:
        return handler.get_lyric_score(
            pred_latent=extra["pred_latents"],
            encoder_hidden_states=extra["encoder_hidden_states"],
            encoder_attention_mask=extra["encoder_attention_mask"],
            context_latents=extra["context_latents"],
            lyric_token_ids=extra["lyric_token_idss"],
            vocal_language=language,
            inference_steps=inference_steps,
            seed=seed,
        )


def _concrete_lm_model(requested: str) -> str | None:
    value = (requested or "auto").strip()
    if value == "none":
        return None
    installed = _installed_lm_models()
    if value == "auto":
        for candidate in sorted(installed):
            lowered = candidate.lower()
            if "acestep-5hz-lm-4b" in lowered and "abliter" in lowered:
                return candidate
        for candidate in ["acestep-5Hz-lm-4B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B"]:
            if candidate in installed:
                return candidate
        return None
    return value if value in installed else None


def _concrete_lm_model_or_download(requested: str, context: str) -> str:
    value = (requested or "auto").strip() or "auto"
    lm_model = _concrete_lm_model(value)
    if lm_model:
        return lm_model
    download_target = ACE_LM_PREFERRED_MODEL if value == "auto" else value
    if download_target in _downloadable_model_names():
        _start_model_download_or_raise(download_target, context=context)
    raise RuntimeError(
        "Official ACE-Step LM controls require a locally installed 5Hz LM model. "
        "Choose or install acestep-5Hz-lm-0.6B/1.7B/4B."
    )


def _requires_lm(params: dict[str, Any]) -> bool:
    if str(params.get("ace_lm_model") or "none").strip() == "none":
        return False
    if params["task_type"] in DOCS_BEST_SOURCE_TASK_LM_SKIPS:
        return False
    lm_control_fields = {
        "allow_lm_batch",
        "constrained_decoding_debug",
        "lm_batch_chunk_size",
        "lm_cfg_scale",
        "lm_negative_prompt",
        "lm_temperature",
        "lm_top_k",
        "lm_top_p",
        "use_constrained_decoding",
        "use_cot_caption",
        "use_cot_language",
        "use_cot_lyrics",
        "use_cot_metas",
    }
    return any(
        [
            params["thinking"],
            params["sample_mode"],
            bool(params["sample_query"]),
            params["use_format"],
            params["use_cot_lyrics"],
            bool(lm_control_fields.intersection(params.get("official_fields", []))),
        ]
    )


def _set_field_error(errors: dict[str, str], field: str, message: str) -> None:
    if field in errors:
        errors[field] = f"{errors[field]}; {message}"
    else:
        errors[field] = message


def _audio_validation_status(
    payload: dict[str, Any],
    *,
    upload_key: str,
    result_key: str,
    audio_codes_key: str | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "present": False,
        "ok": False,
        "kind": "",
        "id": "",
        "result_id": "",
        "filename": "",
        "error": "",
    }
    audio_codes = str(payload.get(audio_codes_key) or "").strip() if audio_codes_key else ""
    if audio_codes:
        status.update({"present": True, "ok": True, "kind": "audio_codes"})
        return status

    direct_key = "reference_audio_path" if upload_key.startswith("reference") else "src_audio_path"
    legacy_key = "reference_audio" if upload_key.startswith("reference") else "src_audio"
    direct_text = str(get_param(payload, direct_key, payload.get(legacy_key)) or "").strip()
    if direct_text:
        status.update({"present": True, "kind": "path", "id": direct_text, "filename": Path(direct_text).name})
        try:
            direct = _resolve_direct_audio_path(direct_text)
            status.update({"ok": True, "filename": direct.name})
        except Exception as exc:
            status["error"] = str(exc)
        return status

    upload_id = str(payload.get(upload_key) or "").strip()
    result_id = str(payload.get(result_key) or "").strip()
    if upload_id:
        status.update({"present": True, "kind": "upload", "id": upload_id})
        try:
            path = _resolve_upload_file(upload_id)
            status.update({"ok": True, "filename": path.name if path else ""})
        except HTTPException as exc:
            status["error"] = str(exc.detail)
        except Exception as exc:
            status["error"] = str(exc)
        return status

    if result_id:
        status.update({"present": True, "kind": "result", "result_id": result_id})
        try:
            path = _resolve_result_audio(result_id, payload.get(f"{result_key}_audio_id"))
            status.update({"ok": True, "filename": path.name if path else ""})
        except HTTPException as exc:
            status["error"] = str(exc.detail)
        except Exception as exc:
            status["error"] = str(exc)
        return status

    return status


def _lm_validation_status(payload: dict[str, Any], requires_lm: bool) -> dict[str, Any]:
    requested = str(payload.get("ace_lm_model") or payload.get("lm_model") or ACE_LM_PREFERRED_MODEL).strip() or ACE_LM_PREFERRED_MODEL
    status: dict[str, Any] = {
        "requested": requested,
        "requires_lm": requires_lm,
        "model": requested,
        "installed": requested in {"auto", "none"},
        "download_required": False,
        "downloading": False,
        "download_job": None,
    }
    if not requires_lm or requested == "none":
        status["installed"] = True
        return status

    installed_lms = _installed_lm_models()
    if requested == "auto":
        concrete = _concrete_lm_model("auto")
        download_target = concrete or ACE_LM_PREFERRED_MODEL
    else:
        concrete = requested if requested in installed_lms else None
        download_target = requested

    status["model"] = concrete or download_target
    status["installed"] = bool(concrete)
    if not concrete:
        status["download_required"] = download_target in _downloadable_model_names()
        status["download_job"] = _model_download_job(download_target)
        status["downloading"] = bool(status["download_job"] and status["download_job"].get("state") in {"queued", "running"})
    return status


def _preview_generation_payload(payload: dict[str, Any], task_type: str, song_model: str, official_used: list[str]) -> dict[str, Any]:
    track_names = normalize_track_names(payload.get("track_names") or payload.get("track_name"))
    quality_profile = normalize_quality_profile(payload.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    requested_lm_model = _requested_ace_lm_model(payload)
    official_used = _active_official_fields(payload, task_type, official_used)
    use_official = bool(official_used) or _quality_lm_controls_enabled(payload, task_type)
    model_defaults = quality_profile_model_settings(song_model, quality_profile)
    requested_format = str(payload.get("audio_format") or (model_defaults["audio_format"] if use_official else "wav")).strip().lower().lstrip(".")
    time_signature = _time_signature_from_payload(payload)
    bpm = _bpm_from_payload(payload)
    is_turbo = "turbo" in song_model
    raw_steps = payload.get("inference_steps", payload.get("infer_step"))
    if raw_steps in [None, "", "auto"]:
        default_steps = _quality_default_steps(song_model, quality_profile)
    else:
        default_steps = int(raw_steps)
    inference_steps = clamp_int(default_steps, default_steps, 1, 200)
    if is_turbo and inference_steps > DOCS_BEST_TURBO_HIGH_CAP_STEPS:
        inference_steps = DOCS_BEST_TURBO_HIGH_CAP_STEPS
    title = str(payload.get("title") or "").strip() or "Untitled"
    artist_name = _artist_name_from_payload(payload, title=title)
    return {
        "ui_mode": str(payload.get("ui_mode") or task_type),
        "quality_profile": quality_profile,
        "task_type": task_type,
        "title": title,
        "artist_name": artist_name,
        "caption": str(payload.get("caption") or ""),
        "global_caption": str(payload.get("global_caption") or ""),
        "lyrics": str(payload.get("lyrics") or ""),
        "caption_source": str(payload.get("caption_source") or "caption"),
        "lyrics_source": str(payload.get("lyrics_source") or "lyrics"),
        "tag_list": list(payload.get("tag_list") or []),
        "payload_warnings": list(payload.get("payload_warnings") or []),
        "ace_step_text_budget": dict(payload.get("ace_step_text_budget") or {}),
        "instrumental": parse_bool(payload.get("instrumental"), False),
        "duration": clamp_float(get_param(payload, "duration"), 60.0, DURATION_MIN, DURATION_MAX),
        "bpm": bpm,
        "key_scale": _key_scale_from_payload(payload),
        "time_signature": time_signature,
        "vocal_language": _language_for_generation(str(get_param(payload, "vocal_language", "unknown") or "unknown")),
        "batch_size": clamp_int(payload.get("batch_size"), 3 if quality_profile == "chart_master" and task_type == "text2music" else 1, 1, MAX_BATCH_SIZE),
        "seed": str(payload.get("seeds") or payload.get("seed") or "-1"),
        "use_random_seed": parse_bool(payload.get("use_random_seed"), str(payload.get("seeds") or payload.get("seed") or "-1").strip() in {"", "-1"}),
        "song_model": song_model,
        "ace_lm_model": _requested_ace_lm_model(payload),
        "planner_lm_provider": normalize_provider(payload.get("planner_lm_provider") or payload.get("planner_provider") or "ollama"),
        "planner_model": str(payload.get("planner_model") or payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        "planner_ollama_model": str(payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        "audio_code_string": str(get_param(payload, "audio_code_string", "") or ""),
        "track_names": track_names,
        "track_name": track_names[0] if track_names else "",
        "reference_audio_id": str(payload.get("reference_audio_id") or ""),
        "reference_result_id": str(payload.get("reference_result_id") or ""),
        "src_audio_id": str(payload.get("src_audio_id") or ""),
        "src_result_id": str(payload.get("src_result_id") or ""),
        "inference_steps": inference_steps,
        "guidance_scale": clamp_float(payload.get("guidance_scale"), model_defaults["guidance_scale"], 1.0, 15.0),
        "shift": clamp_float(payload.get("shift"), model_defaults["shift"], 1.0, 5.0),
        "infer_method": "sde" if str(payload.get("infer_method") or model_defaults["infer_method"]).lower() == "sde" else "ode",
        "sampler_mode": "euler" if str(payload.get("sampler_mode") or model_defaults["sampler_mode"]).lower() == "euler" else "heun",
        "audio_format": normalize_audio_format(payload.get("audio_format") or (model_defaults["audio_format"] if use_official else "wav"), allow_official=use_official)
        if not (use_official and requested_format == "ogg")
        else requested_format,
        "auto_score": parse_bool(payload.get("auto_score"), False),
        "auto_lrc": parse_bool(payload.get("auto_lrc"), False),
        "return_audio_codes": parse_bool(payload.get("return_audio_codes"), False),
        "save_to_library": parse_bool(payload.get("save_to_library"), False),
        "official_fields": official_used,
        "requires_official_runner": use_official,
        "runner_plan": "official" if use_official else "fast",
    }


def _validate_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_payload = _merge_nested_generation_metadata(dict(payload or {}))
    task_type = normalize_task_type(raw_payload.get("task_type"))
    normalized_text = normalize_generation_text_fields(raw_payload, task_type=task_type)
    overflow_policy = str(normalized_text.get("lyrics_overflow_policy") or "auto_fit").strip().lower() or "auto_fit"
    exact_lyrics = parse_bool(normalized_text.get("exact_lyrics") or normalized_text.get("locked_lyrics"), False)
    raw_lyrics_len = len(str(normalized_text.get("lyrics") or ""))
    text_budget_error = ""
    if raw_lyrics_len > ACE_STEP_LYRICS_CHAR_LIMIT and (exact_lyrics or overflow_policy in {"error", "strict", "fail"}):
        text_budget_error = (
            f"ACE-Step accepts max {ACE_STEP_LYRICS_CHAR_LIMIT} lyrics characters per render; "
            f"got {raw_lyrics_len}. Use auto_fit or split the song into parts."
        )
    normalized_text = apply_ace_step_text_budget(normalized_text, task_type=task_type)
    normalized_text = _apply_studio_lm_policy(normalized_text)
    quality_profile = normalize_quality_profile(normalized_text.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    song_model = _song_model_for_quality_profile(get_param(normalized_text, "song_model", normalized_text.get("song_model")), quality_profile, task_type)
    official_used = _active_official_fields(normalized_text, task_type, official_fields_used(normalized_text))
    field_errors: dict[str, str] = {}
    try:
        preview = _preview_generation_payload(normalized_text, task_type, song_model, official_used)
    except ValueError as exc:
        _set_field_error(field_errors, "key_scale", str(exc))
        preview_payload = dict(normalized_text)
        preview_payload["key_scale"] = "auto"
        preview = _preview_generation_payload(preview_payload, task_type, song_model, official_used)
    if text_budget_error:
        _set_field_error(field_errors, "lyrics", text_budget_error)

    supported_tasks = supported_tasks_for_model(song_model)
    task_supported = task_type in supported_tasks
    if not task_supported:
        _set_field_error(
            field_errors,
            "song_model",
            f"{song_model} cannot run {task_type}. Supported tasks: {', '.join(supported_tasks)}.",
        )

    installed = song_model in _installed_acestep_models()
    downloadable = song_model in _downloadable_model_names()
    download_job = _model_download_job(song_model)
    downloading = bool(download_job and download_job.get("state") in {"queued", "running"})
    if song_model in OFFICIAL_UNRELEASED_MODELS:
        _set_field_error(field_errors, "song_model", f"{song_model} is official but unreleased; it is not downloadable yet.")
    elif not installed and not downloadable:
        _set_field_error(field_errors, "song_model", f"{song_model} is not installed and cannot be auto-downloaded.")

    requested_format = str(
        normalized_text.get("audio_format")
        or (quality_profile_model_settings(song_model, quality_profile)["audio_format"] if _quality_lm_controls_enabled(normalized_text, task_type) else "wav")
    ).strip().lower().lstrip(".")
    if official_used and requested_format == "ogg":
        _set_field_error(
            field_errors,
            "audio_format",
            "OGG is only available in the fast AceJAM runner. Use wav/flac/mp3/opus/aac/wav32 with official ACE-Step controls.",
        )

    source_status = _audio_validation_status(
        normalized_text,
        upload_key="src_audio_id",
        result_key="src_result_id",
        audio_codes_key="audio_code_string",
    )
    reference_status = _audio_validation_status(
        normalized_text,
        upload_key="reference_audio_id",
        result_key="reference_result_id",
    )
    if task_type in {"cover", "repaint", "extract", "lego", "complete"} and not source_status["present"]:
        _set_field_error(field_errors, "source", f"{task_type} requires source audio, a source result, or audio codes.")
    if source_status["present"] and not source_status["ok"]:
        _set_field_error(field_errors, "source", source_status.get("error") or "Source audio could not be resolved.")
    if reference_status["present"] and not reference_status["ok"]:
        _set_field_error(field_errors, "reference", reference_status.get("error") or "Reference audio could not be resolved.")

    track_names = normalize_track_names(normalized_text.get("track_names") or normalized_text.get("track_name"))
    if task_type in {"extract", "lego"} and not track_names:
        _set_field_error(field_errors, "track_name", f"{task_type} requires a track/stem name.")
    if task_type == "complete" and not track_names:
        _set_field_error(field_errors, "track_names", "complete requires one or more track/stem names.")

    instrumental = parse_bool(normalized_text.get("instrumental"), False)
    sample_mode = parse_bool(normalized_text.get("sample_mode"), False)
    sample_query = str(get_param(normalized_text, "sample_query", "") or "").strip()
    requested_lm_model = _requested_ace_lm_model(normalized_text)
    lm_controls = _explicit_ace_lm_controls(normalized_text)
    if lm_controls and requested_lm_model == "none":
        _set_field_error(
            field_errors,
            "ace_lm_model",
            "ACE-Step LM controls require ace_lm_model set to auto or an installed 5Hz LM. "
            f"Disable these controls or select an LM: {', '.join(lm_controls)}.",
        )
    if needs_vocal_lyrics(
        task_type=task_type,
        instrumental=instrumental,
        lyrics=str(normalized_text.get("lyrics") or ""),
        sample_mode=sample_mode,
        sample_query=sample_query,
    ):
        _set_field_error(
            field_errors,
            "lyrics",
            "Text2Music vocal generation needs lyrics. Use Write Lyrics/Format, paste lyrics, or enable Instrumental.",
        )

    requires_lm = bool(
        requested_lm_model != "none"
        and task_type not in DOCS_BEST_SOURCE_TASK_LM_SKIPS
        and (lm_controls or _quality_lm_controls_enabled(normalized_text, task_type))
    )
    lm_status = _lm_validation_status(normalized_text, requires_lm)

    parse_error = ""
    normalized_payload = preview
    if not field_errors and installed:
        try:
            parsed = _parse_generation_payload(dict(normalized_text))
            normalized_payload = {
                key: _jsonable(value)
                for key, value in parsed.items()
                if key not in {"reference_audio", "src_audio"}
            }
            normalized_payload.update(
                {
                    "reference_audio_id": str(normalized_text.get("reference_audio_id") or ""),
                    "reference_result_id": str(normalized_text.get("reference_result_id") or ""),
                    "src_audio_id": str(normalized_text.get("src_audio_id") or ""),
                    "src_result_id": str(normalized_text.get("src_result_id") or ""),
                }
            )
            requires_lm = _requires_lm(parsed)
            lm_status = _lm_validation_status(parsed, requires_lm)
        except ModelDownloadStarted:
            pass
        except Exception as exc:
            parse_error = str(exc)
            _set_field_error(field_errors, "payload", parse_error)

    valid = not field_errors
    settings_compliance = ace_step_settings_compliance(
        normalized_payload,
        task_type=task_type,
        song_model=song_model,
        runner_plan=normalized_payload.get("runner_plan") or ("official" if official_used else "fast"),
    )
    normalized_payload["settings_policy_version"] = settings_compliance["version"]
    normalized_payload["settings_compliance"] = settings_compliance
    _attach_pro_preflight(normalized_payload)
    return {
        "success": True,
        "valid": valid,
        "normalized_payload": _jsonable(normalized_payload),
        "field_errors": field_errors,
        "payload_warnings": list(normalized_payload.get("payload_warnings") or normalized_text.get("payload_warnings") or []),
        "tag_list": list(normalized_payload.get("tag_list") or normalized_text.get("tag_list") or []),
        "runner_plan": normalized_payload.get("runner_plan") or ("official" if official_used else "fast"),
        "official_fields": list(normalized_payload.get("official_fields") or official_used),
        "settings_policy_version": settings_compliance["version"],
        "settings_compliance": _jsonable(settings_compliance),
        "hit_readiness": _jsonable(normalized_payload.get("hit_readiness") or {}),
        "effective_settings": _jsonable(normalized_payload.get("effective_settings") or {}),
        "settings_coverage": _jsonable(normalized_payload.get("settings_coverage") or {}),
        "runtime_planner": _jsonable(normalized_payload.get("runtime_planner") or {}),
        "model": {
            "name": song_model,
            "installed": installed,
            "downloadable": downloadable,
            "download_required": bool(not installed and downloadable),
            "downloading": downloading,
            "download_job": _jsonable(download_job),
            "task_supported": task_supported,
            "supported_tasks": supported_tasks,
        },
        "lm_model": _jsonable(lm_status),
        "source_status": _jsonable(source_status),
        "reference_status": _jsonable(reference_status),
        "compatibility": {
            "task_type": task_type,
            "task_supported": task_supported,
            "model_installed": installed,
            "download_required": bool(not installed and downloadable),
            "official_runner_available": _official_runner_status().get("available"),
        },
        "parse_error": parse_error,
        "contract_version": PAYLOAD_CONTRACT_VERSION,
    }


def _official_request_payload(params: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    needs_lm = _requires_lm(params)
    lm_model = _concrete_lm_model(params["ace_lm_model"]) if needs_lm else None
    runtime_caption = str(params.get("caption") or "")
    runtime_lyrics = "[Instrumental]" if params["instrumental"] else str(params.get("lyrics") or "")
    if len(runtime_caption) > ACE_STEP_CAPTION_CHAR_LIMIT:
        raise RuntimeError(
            f"Official ACE-Step runner blocked an over-budget caption: "
            f"{len(runtime_caption)}/{ACE_STEP_CAPTION_CHAR_LIMIT} chars."
        )
    if not params["instrumental"] and len(runtime_lyrics) > ACE_STEP_LYRICS_CHAR_LIMIT:
        raise RuntimeError(
            f"Official ACE-Step runner blocked over-budget lyrics: "
            f"{len(runtime_lyrics)}/{ACE_STEP_LYRICS_CHAR_LIMIT} chars. "
            "Use auto_fit or split the song into parts."
        )
    if needs_lm and not lm_model:
        requested_lm = params["ace_lm_model"]
        download_target = ACE_LM_PREFERRED_MODEL if requested_lm == "auto" else requested_lm
        if download_target in _downloadable_model_names():
            _start_model_download_or_raise(download_target, context="official ACE-Step LM controls")
        raise RuntimeError(
            "Official ACE-Step LM controls require a locally installed 5Hz LM model. "
            "Choose an installed LM or install acestep-5Hz-lm-0.6B/1.7B/4B first."
        )
    if params["auto_lrc"] or params["auto_score"]:
        raise RuntimeError(
            "Auto score and Auto LRC need AceJAM's in-process tensor cache. "
            "Disable official-only controls or turn off Auto score/LRC for this run."
        )

    return {
        "base_dir": str(BASE_DIR),
        "vendor_dir": str(OFFICIAL_ACE_STEP_DIR),
        "model_cache_dir": str(MODEL_CACHE_DIR),
        "checkpoint_dir": str(MODEL_CACHE_DIR / "checkpoints"),
        "save_dir": str(save_dir),
        "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "song_model": params["song_model"],
        "lm_model": lm_model,
        "requires_lm": needs_lm,
        "use_lora": params.get("use_lora", False),
        "lora_adapter_path": params.get("lora_adapter_path", ""),
        "lora_adapter_name": params.get("lora_adapter_name", ""),
        "lora_scale": params.get("lora_scale", 1.0),
        "adapter_model_variant": params.get("adapter_model_variant", ""),
        "device": params["device"],
        "dtype": params["dtype"],
        "use_flash_attention": params["use_flash_attention"],
        "compile_model": params["compile_model"],
        "offload_to_cpu": params["offload_to_cpu"],
        "offload_dit_to_cpu": params["offload_dit_to_cpu"],
        "lm_device": params["lm_device"],
        "lm_dtype": params["lm_dtype"],
        "lm_offload_to_cpu": params["lm_offload_to_cpu"],
        "params": {
            "task_type": params["task_type"],
            "instruction": params["instruction"],
            "reference_audio": str(params["reference_audio"]) if params["reference_audio"] else None,
            "src_audio": str(params["src_audio"]) if params["src_audio"] else None,
            "audio_codes": params["audio_code_string"],
            "caption": runtime_caption,
            "global_caption": params["global_caption"],
            "lyrics": runtime_lyrics,
            "instrumental": params["instrumental"],
            "vocal_language": params["vocal_language"],
            "bpm": params["bpm"],
            "keyscale": params["key_scale"],
            "timesignature": params["time_signature"],
            "duration": params["duration"],
            "enable_normalization": params["enable_normalization"],
            "normalization_db": params["normalization_db"],
            "fade_in_duration": params["fade_in_duration"],
            "fade_out_duration": params["fade_out_duration"],
            "latent_shift": params["latent_shift"],
            "latent_rescale": params["latent_rescale"],
            "inference_steps": params["inference_steps"],
            "seed": -1,
            "guidance_scale": params["guidance_scale"],
            "use_adg": params["use_adg"],
            "cfg_interval_start": params["cfg_interval_start"],
            "cfg_interval_end": params["cfg_interval_end"],
            "shift": params["shift"],
            "infer_method": params["infer_method"],
            "sampler_mode": params["sampler_mode"],
            "velocity_norm_threshold": params["velocity_norm_threshold"],
            "velocity_ema_factor": params["velocity_ema_factor"],
            "timesteps": params["timesteps"],
            "repainting_start": params["repainting_start"],
            "repainting_end": params["repainting_end"],
            "chunk_mask_mode": params["chunk_mask_mode"],
            "repaint_latent_crossfade_frames": params["repaint_latent_crossfade_frames"],
            "repaint_wav_crossfade_sec": params["repaint_wav_crossfade_sec"],
            "repaint_mode": params["repaint_mode"],
            "repaint_strength": params["repaint_strength"],
            "audio_cover_strength": params["audio_cover_strength"],
            "cover_noise_strength": params["cover_noise_strength"],
            "thinking": params["thinking"],
            "lm_temperature": params["lm_temperature"],
            "lm_cfg_scale": params["lm_cfg_scale"],
            "lm_top_k": params["lm_top_k"],
            "lm_top_p": params["lm_top_p"],
            "lm_negative_prompt": params["lm_negative_prompt"],
            "use_cot_metas": params["use_cot_metas"],
            "use_cot_caption": params["use_cot_caption"],
            "use_cot_lyrics": params["use_cot_lyrics"],
            "use_cot_language": params["use_cot_language"],
            "use_constrained_decoding": params["use_constrained_decoding"],
            "sample_mode": params["sample_mode"],
            "sample_query": params["sample_query"],
            "use_format": params["use_format"],
        },
        "lm_backend": params["lm_backend"],
        "config": {
            "batch_size": params["batch_size"],
            "allow_lm_batch": params["allow_lm_batch"],
            "use_random_seed": params["use_random_seed"],
            "seeds": None if params["use_random_seed"] or params["seed"].strip() in {"", "-1"} else params["seed"],
            "lm_batch_chunk_size": params["lm_batch_chunk_size"],
            "constrained_decoding_debug": params["constrained_decoding_debug"],
            "audio_format": params["audio_format"],
            "mp3_bitrate": params["mp3_bitrate"],
            "mp3_sample_rate": params["mp3_sample_rate"],
        },
    }


def _generation_metadata_audit(params: dict[str, Any], official_request: dict[str, Any] | None = None) -> dict[str, Any]:
    request_params = dict((official_request or {}).get("params") or {})
    effective = request_params or {
        "bpm": params.get("bpm"),
        "keyscale": params.get("key_scale"),
        "timesignature": params.get("time_signature"),
        "duration": params.get("duration"),
    }
    required = {
        "bpm": effective.get("bpm"),
        "keyscale": effective.get("keyscale"),
        "duration": effective.get("duration"),
        "timesignature": effective.get("timesignature"),
    }
    missing = [key for key, value in required.items() if value in [None, ""]]
    return {
        "metadata_present": not missing,
        "missing": missing,
        "bpm": {"value": effective.get("bpm"), "present": effective.get("bpm") not in [None, ""]},
        "keyscale": {"value": effective.get("keyscale"), "present": effective.get("keyscale") not in [None, ""]},
        "duration": {"value": effective.get("duration"), "present": effective.get("duration") not in [None, ""]},
        "timesignature": {"value": effective.get("timesignature"), "present": effective.get("timesignature") not in [None, ""]},
        "song_model": params.get("song_model"),
        "lm_backend": params.get("lm_backend"),
        "inference_steps": params.get("inference_steps"),
        "guidance_scale": params.get("guidance_scale"),
        "shift": params.get("shift"),
        "audio_format": params.get("audio_format"),
        "take_count": params.get("batch_size"),
        "use_lora": params.get("use_lora"),
        "lora_adapter_path": params.get("lora_adapter_path"),
        "lora_scale": params.get("lora_scale"),
        "source_lyrics_char_count": (params.get("ace_step_text_budget") or {}).get("source_lyrics_char_count"),
        "runtime_lyrics_char_count": (params.get("ace_step_text_budget") or {}).get("runtime_lyrics_char_count"),
        "lyrics_overflow_action": (params.get("ace_step_text_budget") or {}).get("lyrics_overflow_action"),
    }


def _effective_settings_summary(params: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "quality_profile",
        "task_type",
        "song_model",
        "ace_lm_model",
        "lm_backend",
        "duration",
        "bpm",
        "key_scale",
        "time_signature",
        "batch_size",
        "inference_steps",
        "guidance_scale",
        "shift",
        "infer_method",
        "sampler_mode",
        "use_adg",
        "timesteps",
        "audio_format",
        "use_lora",
        "lora_adapter_name",
        "lora_adapter_path",
        "lora_scale",
        "adapter_model_variant",
        "runner_plan",
    ]
    return {
        "version": PRO_QUALITY_AUDIT_VERSION,
        "fields": {field: _jsonable(params.get(field)) for field in fields if field in params},
        "text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "settings_policy_version": params.get("settings_policy_version"),
        "settings_compliance": _jsonable(params.get("settings_compliance") or {}),
    }


def _attach_pro_preflight(params: dict[str, Any]) -> dict[str, Any]:
    params["hit_readiness"] = hit_readiness_report(
        params,
        task_type=params.get("task_type"),
        song_model=str(params.get("song_model") or ""),
        runner_plan=str(params.get("runner_plan") or ""),
    )
    params["runtime_planner"] = runtime_planner_report(
        params,
        task_type=params.get("task_type"),
        song_model=str(params.get("song_model") or ""),
        quality_profile=str(params.get("quality_profile") or DEFAULT_QUALITY_PROFILE),
    )
    registry = ace_step_settings_registry()
    params["settings_coverage"] = _jsonable(registry.get("coverage") or {})
    params["effective_settings"] = _effective_settings_summary(params)
    return params


def _estimate_bpm_from_audio(samples: np.ndarray, sample_rate: int) -> float | None:
    if sample_rate <= 0 or samples.size < sample_rate * 4:
        return None
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    frame = 2048
    hop = 512
    if mono.size < frame * 4:
        return None
    usable = mono[: mono.size - (mono.size % hop)]
    if usable.size < frame * 4:
        return None
    starts = range(0, usable.size - frame + 1, hop)
    rms = np.array([float(np.sqrt(np.mean(np.square(usable[start : start + frame])))) for start in starts], dtype=np.float32)
    if rms.size < 16 or float(rms.max(initial=0.0)) <= 1e-5:
        return None
    novelty = np.maximum(np.diff(rms), 0.0)
    if novelty.size < 8 or float(novelty.max(initial=0.0)) <= 1e-5:
        return None
    novelty = novelty - float(novelty.mean())
    ac = np.correlate(novelty, novelty, mode="full")[novelty.size - 1 :]
    if ac.size < 4 or float(ac[0]) <= 1e-8:
        return None
    min_bpm, max_bpm = 60.0, 180.0
    min_lag = max(1, int(round((60.0 / max_bpm) * sample_rate / hop)))
    max_lag = min(ac.size - 1, int(round((60.0 / min_bpm) * sample_rate / hop)))
    if max_lag <= min_lag:
        return None
    region = ac[min_lag : max_lag + 1]
    best_offset = int(np.argmax(region))
    best_lag = min_lag + best_offset
    strength = float(region[best_offset] / max(ac[0], 1e-8))
    if strength < 0.08:
        return None
    return round(float(60.0 * sample_rate / (hop * best_lag)), 1)


def _audio_quality_audit(path: Path, params: dict[str, Any], *, seed: str = "") -> dict[str, Any]:
    requested_duration = float(params.get("duration") or 0.0)
    targets = PRO_AUDIO_TARGETS
    try:
        data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        if data.size == 0:
            raise ValueError("audio file is empty")
        mono = data.mean(axis=1)
        abs_mono = np.abs(mono)
        peak = float(abs_mono.max(initial=0.0))
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        rms_db = round(20.0 * np.log10(max(rms, 1e-12)), 2)
        duration = round(float(len(mono) / sample_rate), 3) if sample_rate else 0.0
        clip_percent = round(float(np.mean(abs_mono >= float(targets["clip_linear_threshold"])) * 100.0), 5)
        threshold = max(0.001, peak * 0.01)
        non_silent = np.flatnonzero(abs_mono > threshold)
        if non_silent.size:
            leading_silence = round(float(non_silent[0] / sample_rate), 3)
            trailing_silence = round(float((len(mono) - non_silent[-1] - 1) / sample_rate), 3)
        else:
            leading_silence = duration
            trailing_silence = duration
        duration_delta = round(duration - requested_duration, 3) if requested_duration else 0.0
        duration_tolerance = max(float(targets["duration_tolerance_seconds"]), requested_duration * float(targets["duration_tolerance_ratio"]))
        estimated_bpm = _estimate_bpm_from_audio(mono, int(sample_rate))
        issues: list[str] = []
        if requested_duration and abs(duration_delta) > duration_tolerance:
            issues.append("duration_mismatch")
        if peak < float(targets["peak_linear_min"]):
            issues.append("low_peak")
        if peak > float(targets["peak_linear_max"]):
            issues.append("near_clip_peak")
        if clip_percent > float(targets["clip_percent_max"]):
            issues.append("clipping")
        if max(leading_silence, trailing_silence) > float(targets["silence_edge_seconds_warn"]):
            issues.append("edge_silence")
        return {
            "version": PRO_QUALITY_AUDIT_VERSION,
            "status": "pass" if not issues else "warn",
            "path": str(path),
            "filename": path.name,
            "format": path.suffix.lower().lstrip("."),
            "sample_rate": int(sample_rate),
            "channels": int(data.shape[1]),
            "duration_seconds": duration,
            "requested_duration_seconds": requested_duration,
            "duration_delta_seconds": duration_delta,
            "duration_tolerance_seconds": round(duration_tolerance, 3),
            "peak": round(peak, 6),
            "peak_dbfs": round(20.0 * np.log10(max(peak, 1e-12)), 2),
            "rms_dbfs": rms_db,
            "clip_percent": clip_percent,
            "leading_silence_seconds": leading_silence,
            "trailing_silence_seconds": trailing_silence,
            "estimated_bpm": estimated_bpm,
            "estimated_keyscale": None,
            "key_analysis_status": "not_analyzed",
            "seed": seed,
            "issues": issues,
        }
    except Exception as exc:
        return {
            "version": PRO_QUALITY_AUDIT_VERSION,
            "status": "warn",
            "path": str(path),
            "filename": path.name,
            "error": str(exc),
            "issues": ["audio_analysis_failed"],
        }


def _metadata_adherence(params: dict[str, Any], metadata_audit: dict[str, Any], audio_audit: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, status: str, requested: Any, actual: Any, detail: str = "") -> None:
        checks.append({"id": check_id, "status": status, "requested": requested, "actual": actual, "detail": detail})

    requested_duration = float(params.get("duration") or 0.0)
    actual_duration = audio_audit.get("duration_seconds")
    tolerance = float(audio_audit.get("duration_tolerance_seconds") or max(2.0, requested_duration * 0.05))
    duration_ok = bool(actual_duration is not None and (not requested_duration or abs(float(actual_duration) - requested_duration) <= tolerance))
    add("duration", "pass" if duration_ok else "warn", requested_duration, actual_duration, f"tolerance {round(tolerance, 3)}s")

    requested_bpm = params.get("bpm")
    estimated_bpm = audio_audit.get("estimated_bpm")
    if requested_bpm in [None, ""]:
        add("bpm", "warn", requested_bpm, estimated_bpm, "BPM is Auto/missing")
    elif estimated_bpm is None:
        add("bpm", "not_analyzed", requested_bpm, estimated_bpm, "BPM analyzer could not infer a stable tempo")
    else:
        delta = abs(float(estimated_bpm) - float(requested_bpm))
        add("bpm", "pass" if delta <= float(PRO_AUDIO_TARGETS["bpm_tolerance"]) else "warn", requested_bpm, estimated_bpm, f"delta {round(delta, 2)} BPM")

    requested_key = params.get("key_scale")
    add(
        "key_scale",
        "not_analyzed" if requested_key else "warn",
        requested_key,
        audio_audit.get("estimated_keyscale"),
        "Key estimation is not bundled; requested key is still passed to ACE-Step",
    )
    add("time_signature", "pass" if params.get("time_signature") not in [None, ""] else "warn", params.get("time_signature"), None, "conditioning metadata")
    add("request_metadata", "pass" if metadata_audit.get("metadata_present") else "warn", "present", metadata_audit.get("missing"), "")
    statuses = [check["status"] for check in checks]
    if any(status == "warn" for status in statuses):
        overall = "warn"
    elif any(status == "not_analyzed" for status in statuses):
        overall = "pass_with_unanalyzed_fields"
    else:
        overall = "pass"
    return {"version": PRO_QUALITY_AUDIT_VERSION, "status": overall, "checks": checks}


def _score_take(audio: dict[str, Any], hit_readiness: dict[str, Any], metadata_audit: dict[str, Any]) -> tuple[int, list[str], str]:
    score = 100
    reasons: list[str] = []
    audio_audit = audio.get("audio_quality_audit") or {}
    adherence = audio.get("metadata_adherence") or {}
    if audio_audit.get("status") != "pass":
        issues = list(audio_audit.get("issues") or [])
        score -= min(35, 8 * len(issues))
        reasons.extend(issues[:4])
    if adherence.get("status") == "warn":
        score -= 10
        reasons.append("metadata_adherence_warn")
    if not metadata_audit.get("metadata_present"):
        score -= 12
        reasons.append("request_metadata_missing")
    readiness_status = hit_readiness.get("status")
    if readiness_status == "warn":
        score -= 8
        reasons.append("hit_readiness_warn")
    elif readiness_status == "review":
        score -= 20
        reasons.append("hit_readiness_review")
    if audio_audit.get("peak") and 0.45 <= float(audio_audit.get("peak") or 0) <= float(PRO_AUDIO_TARGETS["peak_linear_max"]):
        reasons.append("strong_peak_headroom")
    if not reasons:
        reasons.append("clean_audio_and_metadata")
    score = max(0, min(100, int(score)))
    status = "pass" if score >= 85 else "warn" if score >= 70 else "rerender_suggested"
    return score, reasons, status


def _build_pro_quality_audit(
    params: dict[str, Any],
    audios: list[dict[str, Any]],
    metadata_audit: dict[str, Any],
    hit_readiness: dict[str, Any],
) -> dict[str, Any]:
    take_scores: list[dict[str, Any]] = []
    for audio in audios:
        score, reasons, status = _score_take(audio, hit_readiness, metadata_audit)
        audio["pro_quality_score"] = score
        audio["pro_quality_status"] = status
        take_scores.append(
            {
                "audio_id": audio.get("id"),
                "filename": audio.get("filename"),
                "score": score,
                "status": status,
                "reasons": reasons,
            }
        )
    recommended = max(take_scores, key=lambda item: item["score"], default=None)
    if recommended:
        for audio in audios:
            audio["is_recommended_take"] = audio.get("id") == recommended.get("audio_id")
    suggestions: list[str] = []
    if recommended and recommended["score"] < 85:
        suggestions.append("Regenerate with a new seed or review prompt/lyrics before final release.")
    if not metadata_audit.get("metadata_present"):
        suggestions.append("Keep BPM/key/time signature locked before rendering.")
    if hit_readiness.get("status") != "pass":
        suggestions.append("Improve hook clarity, sections, or runtime text budget before the next take.")
    status = "pass"
    if recommended and recommended["score"] < 70:
        status = "rerender_suggested"
    elif suggestions:
        status = "warn"
    return {
        "version": PRO_QUALITY_AUDIT_VERSION,
        "status": status,
        "quality_profile": params.get("quality_profile"),
        "single_song_takes": params.get("batch_size"),
        "recommended_take": recommended,
        "take_scores": take_scores,
        "hit_readiness": _jsonable(hit_readiness),
        "metadata_audit": _jsonable(metadata_audit),
        "audio_targets": dict(PRO_AUDIO_TARGETS),
        "rerender_suggestions": suggestions,
    }


def _redact_official_runner_log_line(line: str) -> str:
    """Keep official ACE-Step subprocess logs compact in Pinokio terminals."""
    if "formatted_prompt_with_cot=" in line:
        return re.sub(
            r"formatted_prompt_with_cot=.*",
            "formatted_prompt_with_cot=[redacted by AceJAM: prompt/audio-code payload]",
            line,
        )
    if "Debug output text:" in line and "<|audio_code_" in line:
        return re.sub(
            r"Debug output text:.*",
            "Debug output text: [redacted by AceJAM: audio-code payload]",
            line,
        )
    if "<|audio_code_" in line:
        return re.sub(r"(?:<\|audio_code_\d+\|>){3,}", "<|audio_code_REDACTED|>", line)
    if len(line) > 1600:
        ending = "\n" if line.endswith("\n") else ""
        return f"{line[:1600].rstrip()} ... [truncated by AceJAM]{ending}"
    return line


def _redact_official_runner_stream_line(line: str, state: dict[str, Any]) -> str:
    if "conditioning_text:_prepare_text_conditioning_inputs" in line:
        state["conditioning_block"] = True
        if "text_prompt:" in line:
            return re.sub(r"text_prompt:.*", "text_prompt: [redacted by AceJAM: conditioning prompt]", line)
        if "lyrics_text:" in line:
            return re.sub(r"lyrics_text:.*", "lyrics_text: [redacted by AceJAM: conditioning lyrics]", line)
        return ""
    if state.get("conditioning_block"):
        if re.match(r"^\d{4}-\d{2}-\d{2}\s", line):
            state["conditioning_block"] = False
            return _redact_official_runner_log_line(line)
        return ""
    return _redact_official_runner_log_line(line)


def _run_official_runner_request(request_payload: dict[str, Any], work_dir: Path, timeout: int = 3600) -> dict[str, Any]:
    if not OFFICIAL_ACE_STEP_DIR.exists():
        raise RuntimeError("Official ACE-Step runner requires app/vendor/ACE-Step-1.5. Run Install/Update first.")
    if not OFFICIAL_RUNNER_SCRIPT.exists():
        raise RuntimeError("Official ACE-Step runner script is missing.")

    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "official_request.json"
    response_path = work_dir / "official_response.json"
    request_path.write_text(json.dumps(_jsonable(request_payload), indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OFFICIAL_ACE_STEP_DIR)
    env["HF_HOME"] = str(MODEL_CACHE_DIR / "huggingface")
    env["HF_MODULES_CACHE"] = str(MODEL_CACHE_DIR / "hf_modules")
    env["XDG_CACHE_HOME"] = str(MODEL_CACHE_DIR / "xdg")
    env["ACESTEP_DISABLE_TQDM"] = "1"
    env["LOGURU_LEVEL"] = os.environ.get("ACEJAM_OFFICIAL_LOGURU_LEVEL", "INFO")

    stdout_path = work_dir / "official_stdout.log"
    stderr_path = work_dir / "official_stderr.log"
    print(f"[official_runner] starting action={request_payload.get('action') or 'generate'} work_dir={work_dir}", flush=True)

    def stream_pipe(pipe: Any, log_path: Path) -> None:
        redaction_state: dict[str, Any] = {}
        with log_path.open("w", encoding="utf-8") as log_file:
            while True:
                try:
                    line = pipe.readline()
                except ValueError:
                    break
                if not line:
                    break
                line = _redact_official_runner_stream_line(line, redaction_state)
                if not line:
                    continue
                log_file.write(line)
                log_file.flush()
                print(line, end="", flush=True)

    process = subprocess.Popen(
        [sys.executable, str(OFFICIAL_RUNNER_SCRIPT), str(request_path), str(response_path)],
        cwd=str(OFFICIAL_ACE_STEP_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    readers = [
        threading.Thread(target=stream_pipe, args=(process.stdout, stdout_path), daemon=True),
        threading.Thread(target=stream_pipe, args=(process.stderr, stderr_path), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        returncode = process.wait(timeout=30)
        raise RuntimeError(f"Official ACE-Step runner timed out after {timeout}s") from exc
    finally:
        for pipe in [process.stdout, process.stderr]:
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass
        for reader in readers:
            reader.join(timeout=2)
    print(f"[official_runner] finished action={request_payload.get('action') or 'generate'} returncode={returncode}", flush=True)
    if returncode != 0:
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.is_file() else ""
        tail_lines = (stderr_text or stdout_text).splitlines()[-8:]
        tail = "\n".join(line[:600] for line in tail_lines)
        if returncode < 0:
            signal_name = "SIGKILL" if returncode == -9 else f"signal {-returncode}"
            budget = request_payload.get("ace_step_text_budget") or {}
            details = (
                f"Official ACE-Step runner was killed by the OS ({signal_name}), likely memory pressure. "
                f"Runtime lyrics {budget.get('runtime_lyrics_char_count', 'unknown')}/"
                f"{budget.get('lyrics_char_limit', ACE_STEP_LYRICS_CHAR_LIMIT)} chars; "
                f"LM backend {request_payload.get('lm_backend') or 'auto'}; "
                f"LM required {bool(request_payload.get('requires_lm'))}."
            )
            raise RuntimeError(f"{details}\nLast official log lines:\n{tail}")
        raise RuntimeError(f"Official ACE-Step runner failed (exit {returncode}): {tail or returncode}")
    if not response_path.is_file():
        raise RuntimeError("Official ACE-Step runner did not write a response file")
    return json.loads(response_path.read_text(encoding="utf-8"))


def _official_aux_params(body: dict[str, Any]) -> dict[str, Any]:
    params = {
        "caption": str(body.get("caption") or body.get("prompt") or body.get("description") or ""),
        "lyrics": str(body.get("lyrics") or ""),
        "sample_query": str(get_param(body, "sample_query", body.get("query") or body.get("prompt") or body.get("description") or "") or ""),
        "instrumental": parse_bool(body.get("instrumental"), False),
        "vocal_language": _language_for_generation(str(get_param(body, "vocal_language", body.get("language") or "unknown") or "unknown")),
        "bpm": _bpm_from_payload(body),
        "keyscale": _key_scale_from_payload(body),
        "timesignature": _time_signature_from_payload(body),
        "duration": clamp_float(get_param(body, "duration", body.get("audio_duration")), 60.0, DURATION_MIN, DURATION_MAX),
        "lm_temperature": clamp_float(body.get("lm_temperature") or body.get("temperature"), DOCS_BEST_LM_DEFAULTS["lm_temperature"], 0.0, 2.0),
        "lm_top_k": clamp_int(body.get("lm_top_k") or body.get("top_k"), DOCS_BEST_LM_DEFAULTS["lm_top_k"], 0, 200),
        "lm_top_p": clamp_float(body.get("lm_top_p") or body.get("top_p"), DOCS_BEST_LM_DEFAULTS["lm_top_p"], 0.0, 1.0),
        "repetition_penalty": clamp_float(body.get("lm_repetition_penalty") or body.get("repetition_penalty"), 1.0, 0.1, 4.0),
        "use_constrained_decoding": parse_bool(body.get("use_constrained_decoding"), True),
        "constrained_decoding_debug": parse_bool(body.get("constrained_decoding_debug"), False),
    }
    return apply_ace_step_text_budget(params, task_type="text2music")


def _run_official_lm_aux(action: str, body: dict[str, Any], *, audio_codes: str = "") -> dict[str, Any]:
    lm_model = _concrete_lm_model_or_download(
        str(get_param(body, "ace_lm_model", "auto") or "auto"),
        context=f"official ACE-Step {action}",
    )
    params = _official_aux_params(body)
    if audio_codes:
        params["audio_codes"] = audio_codes
    if isinstance(body.get("param_obj"), str) and body.get("param_obj"):
        try:
            params["user_metadata"] = json.loads(str(body["param_obj"]))
        except json.JSONDecodeError:
            params["user_metadata"] = {}
    else:
        params["user_metadata"] = {
            key: value
            for key, value in {
                "bpm": params.get("bpm"),
                "keyscale": params.get("keyscale"),
                "timesignature": params.get("timesignature"),
                "duration": params.get("duration"),
                "language": params.get("vocal_language"),
            }.items()
            if value not in [None, "", "unknown"]
        }

    aux_id = uuid.uuid4().hex[:12]
    work_dir = RESULTS_DIR / f"official-{action}-{aux_id}"
    raw = _run_official_runner_request(
        {
            "action": action,
            "base_dir": str(BASE_DIR),
            "vendor_dir": str(OFFICIAL_ACE_STEP_DIR),
            "model_cache_dir": str(MODEL_CACHE_DIR),
            "checkpoint_dir": str(MODEL_CACHE_DIR / "checkpoints"),
            "save_dir": str(work_dir),
            "lm_model": lm_model,
            "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
            "lm_backend": _normalize_lm_backend(body.get("lm_backend")),
            "lm_device": str(body.get("lm_device") or "auto"),
            "lm_dtype": str(body.get("lm_dtype") or "auto"),
            "lm_offload_to_cpu": parse_bool(body.get("lm_offload_to_cpu"), False),
            "params": params,
        },
        work_dir,
    )
    if not raw.get("success", False):
        raise RuntimeError(raw.get("error") or raw.get("status_message") or f"{action} failed")
    raw["engine"] = "official"
    raw["ace_lm_model"] = lm_model
    raw["tags"] = raw.get("caption", "")
    raw["key_scale"] = raw.get("keyscale", "")
    raw["time_signature"] = raw.get("timesignature", "")
    raw["language"] = raw.get("language") or raw.get("vocal_language") or params.get("vocal_language")
    raw["vocal_language"] = raw["language"]
    raw.setdefault("title", "ACE-Step Sample")
    return raw


def _copy_official_audio(
    result_dir: Path,
    audio: dict[str, Any],
    index: int,
    requested_format: str,
    preferred_filename: str = "",
) -> tuple[Path, str]:
    source = Path(str(audio.get("path") or ""))
    if not source.is_file():
        raise RuntimeError("Official ACE-Step runner did not return an audio file")
    ext = source.suffix.lstrip(".") or ("wav" if requested_format == "wav32" else requested_format)
    filename = preferred_filename or f"take-{index + 1}.{ext}"
    if not filename.endswith(f".{ext}"):
        filename = f"{safe_filename(filename, f'take-{index + 1}')}.{ext}"
    target = result_dir / filename
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target, filename


def _run_official_generation(params: dict[str, Any]) -> dict[str, Any]:
    result_id = uuid.uuid4().hex[:12]
    result_dir = RESULTS_DIR / result_id
    official_dir = result_dir / "official"
    result_dir.mkdir(parents=True, exist_ok=True)
    official_dir.mkdir(parents=True, exist_ok=True)

    with handler_lock:
        _release_handler_state()

    official_request = _official_request_payload(params, official_dir)
    metadata_audit = _generation_metadata_audit(params, official_request)
    hit_readiness = params.get("hit_readiness") or hit_readiness_report(
        params,
        task_type=params.get("task_type"),
        song_model=params.get("song_model"),
        runner_plan=params.get("runner_plan"),
    )
    official = _run_official_runner_request(official_request, result_dir)
    if not official.get("success"):
        raise RuntimeError(official.get("error") or "Official ACE-Step generation failed")
    official_lora_status = official.get("lora_status") or {
        "active": bool(params.get("use_lora")),
        "path": params.get("lora_adapter_path", ""),
        "scale": params.get("lora_scale", 1.0),
    }

    audios: list[dict[str, Any]] = []
    for index, audio in enumerate(official.get("audios", [])):
        preferred_filename = _preferred_audio_filename(params, params["song_model"], index)
        path, filename = _copy_official_audio(result_dir, audio, index, params["audio_format"], preferred_filename)
        audio_id = f"take-{index + 1}"
        audio_params = audio.get("params") or {}
        seed_text = str(audio_params.get("seed") or params["seed"] or "-1")
        audio_audit = _audio_quality_audit(path, params, seed=seed_text)
        adherence = _metadata_adherence(params, metadata_audit, audio_audit)
        item = {
            "id": audio_id,
            "result_id": result_id,
            "filename": filename,
            "audio_url": _result_public_url(result_id, filename),
            "download_url": _result_public_url(result_id, filename),
            "artist_name": params["artist_name"],
            "title": params["title"] if len(official.get("audios", [])) == 1 else f"{params['title']} {index + 1}",
            "seed": seed_text,
            "sample_rate": int(audio.get("sample_rate") or 48000),
            "runner": "official",
            "payload_warnings": params["payload_warnings"],
            "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
            "generation_metadata_audit": metadata_audit,
            "audio_quality_audit": _jsonable(audio_audit),
            "metadata_adherence": _jsonable(adherence),
            "hit_readiness": _jsonable(hit_readiness),
            "lora_adapter": {
                "use_lora": params.get("use_lora", False),
                "path": params.get("lora_adapter_path", ""),
                "name": params.get("lora_adapter_name", ""),
                "scale": params.get("lora_scale", 1.0),
                "adapter_model_variant": params.get("adapter_model_variant", ""),
                "status": _jsonable(official_lora_status),
            },
        }
        if params["return_audio_codes"] and audio_params.get("audio_codes"):
            item["audio_codes"] = audio_params["audio_codes"]
        if params["save_to_library"]:
            entry = _save_song_entry(
                {
                    "artist_name": params["artist_name"],
                    "title": item["title"],
                    "description": params["description"],
                    "tags": params["caption"],
                    "tag_list": params["tag_list"],
                    "lyrics": "[Instrumental]" if params["instrumental"] else params["lyrics"],
                    "caption_source": params["caption_source"],
                    "lyrics_source": params["lyrics_source"],
                    "payload_warnings": params["payload_warnings"],
                    "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
                    "generation_metadata_audit": _jsonable(metadata_audit),
                    "audio_quality_audit": _jsonable(audio_audit),
                    "metadata_adherence": _jsonable(adherence),
                    "hit_readiness": _jsonable(hit_readiness),
                    "lora_adapter": {
                        "use_lora": params.get("use_lora", False),
                        "path": params.get("lora_adapter_path", ""),
                        "name": params.get("lora_adapter_name", ""),
                        "scale": params.get("lora_scale", 1.0),
                        "adapter_model_variant": params.get("adapter_model_variant", ""),
                        "status": _jsonable(official_lora_status),
                    },
                    "runner_plan": params["runner_plan"],
                    "ui_mode": params["ui_mode"],
                    "bpm": params["bpm"],
                    "key_scale": params["key_scale"],
                    "time_signature": params["time_signature"],
                    "language": params["vocal_language"],
                    "duration": params["duration"],
                    "task_type": params["task_type"],
                    "song_model": params["song_model"],
                    "ace_lm_model": params["ace_lm_model"],
                    "seed": seed_text,
                    "parameters": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
                    "album": _jsonable(params["album_metadata"]),
                    "album_concept": params["album_metadata"].get("album_concept"),
                    "album_id": params["album_metadata"].get("album_id"),
                    "track_number": params["album_metadata"].get("track_number"),
                    "track_variant": params["album_metadata"].get("track_variant"),
                    "result_id": result_id,
                    "runner": "official",
                    "album_family_id": params["album_metadata"].get("album_family_id"),
                    "album_model": params["album_metadata"].get("album_model"),
                    "album_model_label": params["album_metadata"].get("album_model_label"),
                    "preferred_audio_file": filename,
                },
                path,
            )
            item["song_id"] = entry["id"]
            item["library_url"] = entry["audio_url"]
        audios.append(item)

    pro_quality_audit = _build_pro_quality_audit(params, audios, metadata_audit, hit_readiness)
    recommended_take = pro_quality_audit.get("recommended_take")
    meta = {
        "id": result_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_song_model": params["song_model"],
        "success": True,
        "runner": "official",
        "runner_plan": params["runner_plan"],
        "ui_mode": params["ui_mode"],
        "title": params["title"],
        "artist_name": params["artist_name"],
        "quality_profile": params["quality_profile"],
        "task_type": params["task_type"],
        "song_model": params["song_model"],
        "bpm": params["bpm"],
        "key_scale": params["key_scale"],
        "time_signature": params["time_signature"],
        "duration": params["duration"],
        "batch_size": params["batch_size"],
        "inference_steps": params["inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "shift": params["shift"],
        "infer_method": params["infer_method"],
        "sampler_mode": params["sampler_mode"],
        "use_adg": params["use_adg"],
        "use_lora": params["use_lora"],
        "lora_adapter_path": params["lora_adapter_path"],
        "lora_adapter_name": params["lora_adapter_name"],
        "lora_scale": params["lora_scale"],
        "adapter_model_variant": params["adapter_model_variant"],
        "lora_adapter": _jsonable(official_lora_status),
        "audio_format": params["audio_format"],
        "lm_backend": params["lm_backend"],
        "thinking": params["thinking"],
        "use_format": params["use_format"],
        "use_cot_metas": params["use_cot_metas"],
        "use_cot_caption": params["use_cot_caption"],
        "use_cot_language": params["use_cot_language"],
        "use_cot_lyrics": params["use_cot_lyrics"],
        "tags": params["caption"],
        "tag_list": params["tag_list"],
        "lyrics": "[Instrumental]" if params["instrumental"] else params["lyrics"],
        "payload_warnings": params["payload_warnings"],
        "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "official_features": params["official_fields"],
        "generation_metadata_audit": _jsonable(metadata_audit),
        "hit_readiness": _jsonable(hit_readiness),
        "effective_settings": _jsonable(params.get("effective_settings") or _effective_settings_summary(params)),
        "settings_coverage": _jsonable(params.get("settings_coverage") or ace_step_settings_registry().get("coverage") or {}),
        "runtime_planner": _jsonable(params.get("runtime_planner") or runtime_planner_report(params)),
        "pro_quality_audit": _jsonable(pro_quality_audit),
        "recommended_take": _jsonable(recommended_take),
        "rerender_suggestions": _jsonable(pro_quality_audit.get("rerender_suggestions") or []),
        "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
        "time_costs": _jsonable(official.get("time_costs", {})),
        "lm_metadata": _jsonable(official.get("lm_metadata")),
        "audios": audios,
    }
    (result_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "success": True,
        "result_id": result_id,
        "active_song_model": params["song_model"],
        "runner": "official",
        "official_features": params["official_fields"],
        "audios": audios,
        "params": meta["params"],
        "bpm": params["bpm"],
        "key_scale": params["key_scale"],
        "time_signature": params["time_signature"],
        "duration": params["duration"],
        "song_model": params["song_model"],
        "quality_profile": params["quality_profile"],
        "batch_size": params["batch_size"],
        "inference_steps": params["inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "shift": params["shift"],
        "audio_format": params["audio_format"],
        "lm_backend": params["lm_backend"],
        "generation_metadata_audit": metadata_audit,
        "hit_readiness": hit_readiness,
        "effective_settings": meta["effective_settings"],
        "settings_coverage": meta["settings_coverage"],
        "runtime_planner": meta["runtime_planner"],
        "pro_quality_audit": pro_quality_audit,
        "recommended_take": recommended_take,
        "rerender_suggestions": pro_quality_audit.get("rerender_suggestions") or [],
        "payload_warnings": params["payload_warnings"],
        "ace_step_text_budget": meta["ace_step_text_budget"],
        "use_lora": params["use_lora"],
        "lora_adapter_path": params["lora_adapter_path"],
        "lora_adapter_name": params["lora_adapter_name"],
        "lora_scale": params["lora_scale"],
        "adapter_model_variant": params["adapter_model_variant"],
        "lora_adapter": _jsonable(official_lora_status),
    }


def _run_advanced_generation(raw_payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_training_idle()
    params = _parse_generation_payload(raw_payload)
    if params["instrumental"] and not params["lyrics"].strip():
        params["lyrics"] = "[Instrumental]"
    if params["requires_official_runner"]:
        return _run_official_generation(params)
    use_random_seed = bool(params["use_random_seed"])
    with handler_lock:
        active_song_model = _ensure_song_model(params["song_model"])
        lora_status = _apply_lora_request(params)
        result = handler.generate_music(
            captions=params["caption"],
            lyrics="[Instrumental]" if params["instrumental"] else params["lyrics"],
            bpm=params["bpm"],
            key_scale=params["key_scale"],
            time_signature=params["time_signature"],
            vocal_language=params["vocal_language"],
            inference_steps=params["inference_steps"],
            guidance_scale=params["guidance_scale"],
            use_random_seed=use_random_seed,
            seed=None if use_random_seed else params["seed"],
            reference_audio=str(params["reference_audio"]) if params["reference_audio"] else None,
            audio_duration=params["duration"],
            batch_size=params["batch_size"],
            src_audio=str(params["src_audio"]) if params["src_audio"] else None,
            audio_code_string=params["audio_code_string"],
            repainting_start=params["repainting_start"],
            repainting_end=params["repainting_end"],
            instruction=params["instruction"],
            audio_cover_strength=params["audio_cover_strength"],
            task_type=params["task_type"],
            use_adg=params["use_adg"],
            cfg_interval_start=params["cfg_interval_start"],
            cfg_interval_end=params["cfg_interval_end"],
            shift=params["shift"],
            infer_method=params["infer_method"],
            timesteps=params["timesteps"],
        )

    if not result.get("success"):
        raise RuntimeError(result.get("error", "generation failed"))

    result_id = uuid.uuid4().hex[:12]
    result_dir = RESULTS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)
    extra = result.get("extra_outputs") or {}
    metadata_audit = _generation_metadata_audit(params)
    hit_readiness = params.get("hit_readiness") or hit_readiness_report(
        params,
        task_type=params.get("task_type"),
        song_model=params.get("song_model"),
        runner_plan=params.get("runner_plan"),
    )
    seed_values = [item.strip() for item in str(extra.get("seed_value") or params["seed"]).split(",")]
    audios = []

    for index, audio_dict in enumerate(result.get("audios", [])):
        audio_id = f"take-{index + 1}"
        filename = _preferred_audio_filename(params, active_song_model, index)
        path = result_dir / filename
        _write_audio_file(audio_dict, path)
        item_extra = _extra_for_index(extra, index)
        seed_text = seed_values[index] if index < len(seed_values) else (seed_values[0] if seed_values else "42")
        try:
            seed_int = int(seed_text)
        except (TypeError, ValueError):
            seed_int = 42
        audio_audit = _audio_quality_audit(path, params, seed=seed_text)
        adherence = _metadata_adherence(params, metadata_audit, audio_audit)

        item = {
            "id": audio_id,
            "result_id": result_id,
            "filename": filename,
            "audio_url": _result_public_url(result_id, filename),
            "download_url": _result_public_url(result_id, filename),
            "artist_name": params["artist_name"],
            "title": params["title"] if len(result.get("audios", [])) == 1 else f"{params['title']} {index + 1}",
            "seed": seed_text,
            "sample_rate": int(audio_dict["sample_rate"]),
            "runner": "fast",
            "payload_warnings": params["payload_warnings"],
            "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
            "generation_metadata_audit": metadata_audit,
            "audio_quality_audit": _jsonable(audio_audit),
            "metadata_adherence": _jsonable(adherence),
            "hit_readiness": _jsonable(hit_readiness),
            "lora_adapter": _jsonable(lora_status),
        }
        if params["auto_lrc"]:
            item["lrc"] = _calculate_lrc(item_extra, params["duration"], params["vocal_language"], params["inference_steps"], seed_int)
        if params["auto_score"]:
            item["score"] = _calculate_score(item_extra, params["vocal_language"], params["inference_steps"], seed_int)
        if params["return_audio_codes"]:
            with handler_lock:
                item["audio_codes"] = handler.convert_src_audio_to_codes(str(path))
        if params["save_to_library"]:
            entry = _save_song_entry(
                {
                    "artist_name": params["artist_name"],
                    "title": item["title"],
                    "description": params["description"],
                    "tags": params["caption"],
                    "tag_list": params["tag_list"],
                    "lyrics": params["lyrics"],
                    "caption_source": params["caption_source"],
                    "lyrics_source": params["lyrics_source"],
                    "payload_warnings": params["payload_warnings"],
                    "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
                    "generation_metadata_audit": _jsonable(metadata_audit),
                    "audio_quality_audit": _jsonable(audio_audit),
                    "metadata_adherence": _jsonable(adherence),
                    "hit_readiness": _jsonable(hit_readiness),
                    "lora_adapter": _jsonable(lora_status),
                    "runner_plan": params["runner_plan"],
                    "ui_mode": params["ui_mode"],
                    "bpm": params["bpm"],
                    "key_scale": params["key_scale"],
                    "time_signature": params["time_signature"],
                    "language": params["vocal_language"],
                    "duration": params["duration"],
                    "task_type": params["task_type"],
                    "song_model": active_song_model,
                    "seed": seed_text,
                    "parameters": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
                    "album": _jsonable(params["album_metadata"]),
                    "album_concept": params["album_metadata"].get("album_concept"),
                    "album_id": params["album_metadata"].get("album_id"),
                    "track_number": params["album_metadata"].get("track_number"),
                    "track_variant": params["album_metadata"].get("track_variant"),
                    "score": item.get("score"),
                    "lrc": item.get("lrc"),
                    "result_id": result_id,
                    "album_family_id": params["album_metadata"].get("album_family_id"),
                    "album_model": params["album_metadata"].get("album_model"),
                    "album_model_label": params["album_metadata"].get("album_model_label"),
                    "preferred_audio_file": filename,
                },
                path,
            )
            item["song_id"] = entry["id"]
            item["library_url"] = entry["audio_url"]
        audios.append(item)

    pro_quality_audit = _build_pro_quality_audit(params, audios, metadata_audit, hit_readiness)
    recommended_take = pro_quality_audit.get("recommended_take")
    meta = {
        "id": result_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_song_model": active_song_model,
        "success": True,
        "runner": "fast",
        "runner_plan": params["runner_plan"],
        "ui_mode": params["ui_mode"],
        "title": params["title"],
        "artist_name": params["artist_name"],
        "quality_profile": params["quality_profile"],
        "task_type": params["task_type"],
        "song_model": active_song_model,
        "bpm": params["bpm"],
        "key_scale": params["key_scale"],
        "time_signature": params["time_signature"],
        "duration": params["duration"],
        "batch_size": params["batch_size"],
        "inference_steps": params["inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "shift": params["shift"],
        "infer_method": params["infer_method"],
        "sampler_mode": params["sampler_mode"],
        "use_adg": params["use_adg"],
        "use_lora": params["use_lora"],
        "lora_adapter_path": params["lora_adapter_path"],
        "lora_adapter_name": params["lora_adapter_name"],
        "lora_scale": params["lora_scale"],
        "adapter_model_variant": params["adapter_model_variant"],
        "lora_adapter": _jsonable(lora_status),
        "audio_format": params["audio_format"],
        "tags": params["caption"],
        "tag_list": params["tag_list"],
        "lyrics": params["lyrics"],
        "payload_warnings": params["payload_warnings"],
        "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "generation_metadata_audit": _jsonable(metadata_audit),
        "hit_readiness": _jsonable(hit_readiness),
        "effective_settings": _jsonable(params.get("effective_settings") or _effective_settings_summary(params)),
        "settings_coverage": _jsonable(params.get("settings_coverage") or ace_step_settings_registry().get("coverage") or {}),
        "runtime_planner": _jsonable(params.get("runtime_planner") or runtime_planner_report(params)),
        "pro_quality_audit": _jsonable(pro_quality_audit),
        "recommended_take": _jsonable(recommended_take),
        "rerender_suggestions": _jsonable(pro_quality_audit.get("rerender_suggestions") or []),
        "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
        "time_costs": _jsonable(extra.get("time_costs", {})),
        "audios": audios,
    }
    (result_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _result_extra_cache[result_id] = extra
    while len(_result_extra_cache) > 8:
        _result_extra_cache.pop(next(iter(_result_extra_cache)))

    return {
        "success": True,
        "result_id": result_id,
        "active_song_model": active_song_model,
        "audios": audios,
        "params": meta["params"],
        "runner": "fast",
        "bpm": params["bpm"],
        "key_scale": params["key_scale"],
        "time_signature": params["time_signature"],
        "duration": params["duration"],
        "song_model": active_song_model,
        "quality_profile": params["quality_profile"],
        "batch_size": params["batch_size"],
        "inference_steps": params["inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "shift": params["shift"],
        "audio_format": params["audio_format"],
        "generation_metadata_audit": metadata_audit,
        "hit_readiness": hit_readiness,
        "effective_settings": meta["effective_settings"],
        "settings_coverage": meta["settings_coverage"],
        "runtime_planner": meta["runtime_planner"],
        "pro_quality_audit": pro_quality_audit,
        "recommended_take": recommended_take,
        "rerender_suggestions": pro_quality_audit.get("rerender_suggestions") or [],
        "payload_warnings": params["payload_warnings"],
        "ace_step_text_budget": meta["ace_step_text_budget"],
        "time_costs": meta["time_costs"],
    }


def _portfolio_generation_payload(raw_payload: dict[str, Any], model_item: dict[str, Any], family_id: str) -> dict[str, Any]:
    model_name = str(model_item.get("model") or "")
    quality_profile = normalize_quality_profile(raw_payload.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    model_defaults = quality_profile_model_settings(model_name, quality_profile)
    payload = dict(raw_payload or {})
    has_vocal_lyrics = bool(str(payload.get("lyrics") or "").strip() and str(payload.get("lyrics") or "").strip().lower() != "[instrumental]")
    payload.update(
        {
            "task_type": "text2music",
            "quality_profile": quality_profile,
            "song_model": model_name,
            "batch_size": 1,
            "inference_steps": int(model_item.get("default_steps") or _quality_default_steps(model_name, quality_profile)),
            "guidance_scale": float(model_item.get("default_guidance_scale") or model_defaults["guidance_scale"]),
            "shift": float(model_item.get("default_shift") or model_defaults["shift"]),
            "ace_lm_model": _requested_ace_lm_model(raw_payload),
            "planner_lm_provider": normalize_provider(raw_payload.get("planner_lm_provider") or raw_payload.get("planner_provider") or "ollama"),
            "planner_model": str(raw_payload.get("planner_model") or raw_payload.get("planner_ollama_model") or raw_payload.get("ollama_model") or ""),
            "render_strategy": SONG_PORTFOLIO_STRATEGY,
            "thinking": parse_bool(raw_payload.get("thinking"), DOCS_BEST_LM_DEFAULTS["thinking"] if not has_vocal_lyrics else False),
            "use_format": parse_bool(raw_payload.get("use_format"), DOCS_BEST_LM_DEFAULTS["use_format"] if not has_vocal_lyrics else False),
            "use_cot_metas": parse_bool(raw_payload.get("use_cot_metas"), DOCS_BEST_LM_DEFAULTS["use_cot_metas"] if not has_vocal_lyrics else False),
            "use_cot_caption": parse_bool(raw_payload.get("use_cot_caption"), DOCS_BEST_LM_DEFAULTS["use_cot_caption"] if not has_vocal_lyrics else False),
            "use_cot_lyrics": parse_bool(raw_payload.get("use_cot_lyrics"), DOCS_BEST_LM_DEFAULTS["use_cot_lyrics"] if not has_vocal_lyrics else False),
            "use_cot_language": parse_bool(raw_payload.get("use_cot_language"), DOCS_BEST_LM_DEFAULTS["use_cot_language"] if not has_vocal_lyrics else False),
            "use_constrained_decoding": parse_bool(raw_payload.get("use_constrained_decoding"), DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"]),
            "lm_temperature": clamp_float(raw_payload.get("lm_temperature"), DOCS_BEST_LM_DEFAULTS["lm_temperature"], 0.0, 2.0),
            "lm_cfg_scale": clamp_float(raw_payload.get("lm_cfg_scale"), DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"], 0.0, 10.0),
            "lm_top_p": clamp_float(raw_payload.get("lm_top_p"), DOCS_BEST_LM_DEFAULTS["lm_top_p"], 0.0, 1.0),
            "lm_top_k": clamp_int(raw_payload.get("lm_top_k"), DOCS_BEST_LM_DEFAULTS["lm_top_k"], 0, 200),
        }
    )
    payload.setdefault("seed", "-1")
    payload.setdefault("audio_format", model_defaults["audio_format"])
    payload.setdefault("infer_method", model_defaults["infer_method"])
    payload.setdefault("sampler_mode", model_defaults["sampler_mode"])
    payload["artist_name"] = _artist_name_from_payload(payload, title=str(payload.get("title") or "Untitled"))
    album_metadata = payload.get("album_metadata") if isinstance(payload.get("album_metadata"), dict) else {}
    album_metadata = dict(album_metadata)
    album_metadata.update(
        {
            "render_strategy": SONG_PORTFOLIO_STRATEGY,
            "portfolio_family_id": family_id,
            "portfolio_model": model_name,
            "portfolio_model_label": model_item.get("label") or model_name,
            "portfolio_model_summary": model_item.get("summary") or "",
            "portfolio_index": int(model_item.get("index") or 0),
            "portfolio_model_slug": model_item.get("slug") or _model_slug(model_name),
            "source_payload": _jsonable(raw_payload),
            "album_model": model_name,
            "album_model_label": model_item.get("label") or model_name,
            "track_variant": 1,
        }
    )
    payload["album_metadata"] = album_metadata
    return payload


def _run_model_portfolio_generation(raw_payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_training_idle()
    raw_payload = dict(raw_payload or {})
    task_type = normalize_task_type(raw_payload.get("task_type") or "text2music")
    if task_type != "text2music":
        raise ValueError("Render all 7 models is only available for Simple/Custom text2music.")
    installed = _installed_acestep_models()
    portfolio = album_model_portfolio(installed)
    missing = [str(item["model"]) for item in portfolio if not item.get("installed")]
    family_id = raw_payload.get("portfolio_family_id") or f"songfam-{uuid.uuid4().hex[:10]}"
    logs = [
        "Render all 7 models requested.",
        f"Portfolio family: {family_id}",
        f"Models: {', '.join(ALBUM_MODEL_PORTFOLIO_MODELS)}",
    ]
    if missing:
        payload = _album_missing_download_payload(
            missing,
            logs,
            render_strategy=SONG_PORTFOLIO_STRATEGY,
            portfolio_family_id=family_id,
            portfolio_models=portfolio,
            source_payload=_jsonable(raw_payload),
        )
        payload["message"] = (
            f"AceJAM started downloading {len(missing)} missing model(s). "
            "The 7-model song render will resume after install."
        )
        return payload

    validation_payload = _portfolio_generation_payload(raw_payload, portfolio[0], str(family_id))
    validation = _validate_generation_payload(validation_payload)
    if not validation.get("valid", False):
        errors = validation.get("field_errors") or {}
        message = "; ".join(f"{field}: {value}" for field, value in errors.items()) or validation.get("error") or "Payload is invalid"
        raise ValueError(message)

    model_results: list[dict[str, Any]] = []
    audios: list[dict[str, Any]] = []
    success_count = 0
    for model_item in portfolio:
        model_name = str(model_item["model"])
        payload = _portfolio_generation_payload(raw_payload, model_item, str(family_id))
        try:
            result = _run_advanced_generation(payload)
            model_audios = []
            for audio in result.get("audios", []):
                enriched = dict(audio)
                enriched.update(
                    {
                        "render_strategy": SONG_PORTFOLIO_STRATEGY,
                        "portfolio_family_id": family_id,
                        "portfolio_model": model_name,
                        "portfolio_model_label": model_item.get("label") or model_name,
                        "portfolio_index": model_item.get("index"),
                        "album_model": model_name,
                        "album_model_label": model_item.get("label") or model_name,
                    }
                )
                model_audios.append(enriched)
                audios.append(enriched)
            model_results.append(
                {
                    "success": True,
                    "portfolio_model": model_name,
                    "portfolio_model_label": model_item.get("label") or model_name,
                    "portfolio_index": model_item.get("index"),
                    "result_id": result.get("result_id"),
                    "audios": model_audios,
                    "params": result.get("params") or payload,
                    "runner": result.get("runner"),
                }
            )
            success_count += 1
            logs.append(f"Rendered {model_item.get('label') or model_name}: {len(model_audios)} take(s).")
        except ModelDownloadStarted:
            raise
        except Exception as exc:
            model_results.append(
                {
                    "success": False,
                    "portfolio_model": model_name,
                    "portfolio_model_label": model_item.get("label") or model_name,
                    "portfolio_index": model_item.get("index"),
                    "error": str(exc),
                    "audios": [],
                }
            )
            logs.append(f"Failed {model_item.get('label') or model_name}: {exc}")

    success = success_count == len(portfolio)
    return {
        "success": success,
        "error": "" if success else f"Portfolio incomplete: {success_count}/{len(portfolio)} model renders succeeded.",
        "render_strategy": SONG_PORTFOLIO_STRATEGY,
        "portfolio_family_id": family_id,
        "portfolio_models": portfolio,
        "model_results": model_results,
        "audios": audios,
        "active_song_model": "all 7 models",
        "runner": "portfolio",
        "params": validation.get("normalized_payload") or validation_payload,
        "payload_warnings": validation.get("payload_warnings") or [],
        "logs": logs,
    }


app = Server(title="AceJAM")


@app.api(name="compose", concurrency_limit=1, time_limit=120)
def compose(
    description: str,
    audio_duration: float = 60.0,
    composer_profile: str = "auto",
    instrumental: bool = False,
    ollama_model: str = "",
    planner_lm_provider: str = "ollama",
    planner_model: str = "",
) -> str:
    """Compose song spec (title, tags, lyrics, etc.) without generating music."""
    provider = normalize_provider(planner_lm_provider)
    selected_model = str(planner_model or (ollama_model if provider == "ollama" else "")).strip()
    try:
        if selected_model:
            selected_model = _resolve_local_llm_model_selection(provider, selected_model, "chat", "songwriting")
        composed = composer.compose(
            description=description,
            audio_duration=audio_duration,
            profile=composer_profile,
            instrumental=instrumental,
            ollama_model=ollama_model or None,
            planner_lm_provider=provider,
            planner_model=selected_model or None,
        )
        return json.dumps(composed)
    except OllamaPullStarted:
        raise
    except Exception as exc:
        if provider == "ollama" and selected_model and _ollama_error_is_missing_model(exc):
            job = _start_ollama_pull(selected_model, reason="songwriting", kind="chat")
            raise OllamaPullStarted(selected_model, job, f"{selected_model} is missing; pull started for songwriting.") from exc
        print(f"[compose ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise


@app.api(name="create", concurrency_limit=1, time_limit=420)
def create(
    description: str,
    audio_duration: float = 60.0,
    seed: int = -1,
    community: bool = False,
    composer_profile: str = "auto",
    song_model: str = "auto",
    instrumental: bool = False,
) -> str:
    started_at = time.perf_counter()
    try:
        print(
            "[create] "
            f"request duration={audio_duration} "
            f"seed={seed} "
            f"community={community} "
            f"composer_profile={composer_profile} "
            f"song_model={song_model} "
            f"instrumental={instrumental}"
        )
        _log_block("create.description", description)
        compose_started_at = time.perf_counter()
        composed = composer.compose(
            description=description,
            audio_duration=audio_duration,
            profile=composer_profile,
            instrumental=instrumental,
        )
        compose_elapsed = time.perf_counter() - compose_started_at
        resolved_profile = composed.get("composer_profile", composer_profile)
        print(
            "[create] "
            f"profile={resolved_profile} "
            f"model={composed.get('composer_model', 'unknown')} "
            f"title={composed['title']} "
            f"language={composed['language']} "
            f"bpm={composed['bpm']} "
            f"tags={composed['tags'][:80]} "
            f"compose_time={compose_elapsed:.2f}s"
        )
        _log_block("create.generated_lyrics", composed["lyrics"])
        _cleanup_accelerator_memory()
        create_defaults = docs_best_model_settings(_normalize_song_model(song_model))

        print(
            "[create->acestep] "
            f"requested_song_model={song_model} "
            f"audio_duration={audio_duration} "
            f"infer_steps={create_defaults['inference_steps']} "
            f"seed={seed} "
            f"language={composed['language']} "
            f"bpm={composed['bpm']} "
            f"key_scale={composed.get('key_scale', '')} "
            f"time_signature={composed.get('time_signature', '')}"
        )
        _log_block("create.acestep_prompt", composed["tags"])
        _log_block("create.acestep_lyrics", composed["lyrics"])
        inference_started_at = time.perf_counter()
        wav_path, active_song_model = _run_inference(
            prompt=composed["tags"],
            lyrics=composed["lyrics"],
            audio_duration=audio_duration,
            infer_steps=int(create_defaults["inference_steps"]),
            seed=seed,
            language=composed["language"],
            song_model=song_model,
            bpm=composed["bpm"],
            key_scale=composed.get("key_scale", ""),
            time_signature=composed.get("time_signature", ""),
            guidance_scale=float(create_defaults["guidance_scale"]),
        )
        inference_elapsed = time.perf_counter() - inference_started_at
        total_elapsed = time.perf_counter() - started_at
        print(
            "[create timing] "
            f"compose={compose_elapsed:.2f}s "
            f"generate={inference_elapsed:.2f}s "
            f"total={total_elapsed:.2f}s"
        )
        wav_bytes = Path(wav_path).read_bytes()
        audio_b64 = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode()}"

        result = {
            "audio": audio_b64,
            "artist_name": composed.get("artist_name") or derive_artist_name(composed["title"], description, composed["tags"]),
            "title": composed["title"],
            "tags": composed["tags"],
            "lyrics": composed["lyrics"],
            "bpm": composed["bpm"],
            "key_scale": composed.get("key_scale", ""),
            "time_signature": composed.get("time_signature", ""),
            "language": composed["language"],
            "composer_profile": resolved_profile,
            "composer_model": composed.get("composer_model", "unknown"),
            "song_model": active_song_model,
        }

        if community:
            song_id = uuid.uuid4().hex[:12]
            song_dir = SONGS_DIR / song_id
            song_dir.mkdir(parents=True, exist_ok=True)

            audio_file = _numbered_audio_filename(
                composed["title"],
                active_song_model,
                "wav",
                artist_name=result["artist_name"],
                variant=1,
            )
            (song_dir / audio_file).write_bytes(wav_bytes)

            meta = {
                "id": song_id,
                "artist_name": result["artist_name"],
                "title": composed["title"],
                "description": description,
                "tags": composed["tags"],
                "lyrics": composed["lyrics"],
                "bpm": composed["bpm"],
                "key_scale": composed.get("key_scale", ""),
                "time_signature": composed.get("time_signature", ""),
                "language": composed["language"],
                "duration": audio_duration,
                "audio_file": audio_file,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (song_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            entry = _decorate_song(meta)
            _feed_songs.insert(0, entry)
            result["song_id"] = song_id
            result["community_url"] = entry["audio_url"]

        return json.dumps(result)
    except Exception as exc:
        print(f"[create ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
    finally:
        _cleanup_accelerator_memory()


@app.api(name="generate", concurrency_limit=1, time_limit=240)
def generate(
    prompt: str,
    lyrics: str,
    audio_duration: float = 60.0,
    infer_step: int = 0,
    guidance_scale: float = 0.0,
    seed: int = -1,
    song_model: str = "auto",
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    lora_name_or_path: str = "",
    lora_weight: float = 0.8,
) -> str:
    try:
        repaired = normalize_generation_text_fields(
            {
                "caption": prompt,
                "lyrics": lyrics,
                "instrumental": str(lyrics or "").strip().lower() == "[instrumental]",
            },
            task_type="text2music",
        )
        prompt = repaired["caption"]
        lyrics = repaired["lyrics"]
        if lora_name_or_path.strip():
            with handler_lock:
                status_msg = handler.load_lora(lora_name_or_path.strip())
                if status_msg.startswith("❌"):
                    raise RuntimeError(status_msg)
                scale_msg = handler.set_lora_scale(clamp_float(lora_weight, 0.8, 0.0, 1.0))
                if str(scale_msg).startswith("❌"):
                    raise RuntimeError(scale_msg)
                use_msg = handler.set_use_lora(True)
                if str(use_msg).startswith("❌"):
                    raise RuntimeError(use_msg)
        wav_path, _ = _run_inference(
            prompt, lyrics, audio_duration, infer_step, seed, "en",
            song_model=song_model,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            guidance_scale=guidance_scale,
        )
        encoded = base64.b64encode(Path(wav_path).read_bytes()).decode()
        return f"data:audio/wav;base64,{encoded}"
    except Exception as exc:
        print(f"[generate ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
    finally:
        _cleanup_accelerator_memory()


@app.api(name="delete_song", concurrency_limit=4)
def delete_song(song_id: str) -> str:
    song_dir = SONGS_DIR / safe_id(song_id)
    if not song_dir.exists():
        return json.dumps({"success": False, "error": "not found"})
    # Remove from in-memory feed
    _feed_songs[:] = [s for s in _feed_songs if s.get("id") != song_id]
    # Remove from disk
    shutil.rmtree(song_dir, ignore_errors=True)
    print(f"[delete] removed song {song_id}")
    return json.dumps({"success": True})


@app.api(name="community", concurrency_limit=4)
def community() -> str:
    return json.dumps(_feed_songs[:50])


@app.api(name="config", concurrency_limit=8)
def config() -> str:
    available_models = _available_acestep_models()
    installed_models = _installed_acestep_models()
    installed_lms = _installed_lm_models()
    model_profiles = model_profiles_for_models(available_models, installed_models)
    for model_name, profile in model_profiles.items():
        profile.update(_model_runtime_status(model_name))
    lm_models = list(dict.fromkeys([*ACE_STEP_LM_MODELS, *sorted(name for name in installed_lms if name not in {"auto", "none"})]))
    lm_profiles = lm_model_profiles_for_models(lm_models, installed_lms)
    for lm_name, profile in lm_profiles.items():
        if lm_name not in {"auto", "none"}:
            job = _model_download_job(lm_name)
            profile.update(
                {
                    "downloadable": lm_name in _downloadable_model_names(),
                    "downloading": bool(job and job.get("state") in {"queued", "running"}),
                    "download_job": _jsonable(job),
                    "status": "installed" if lm_name in installed_lms else "download_required",
                }
            )
    return json.dumps(
        {
            "app_version": APP_UI_VERSION,
            "ui_hash": _app_ui_hash(),
            "backend_hash": _backend_code_hash(),
            "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
            "prompt_kit_version": PROMPT_KIT_VERSION,
            "prompt_kit": prompt_kit_payload(),
            "active_song_model": ACTIVE_ACE_STEP_MODEL,
            "default_song_model": _default_acestep_checkpoint(),
            "default_bpm": DEFAULT_BPM,
            "default_key_scale": DEFAULT_KEY_SCALE,
            "valid_keyscales": VALID_KEY_SCALES,
            "default_planner_lm_provider": "ollama",
            "default_album_planner_ollama_model": DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            "default_album_embedding_model": DEFAULT_ALBUM_EMBEDDING_MODEL,
            "default_album_planner_model": DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            "default_album_embedding_provider": "ollama",
            "local_llm": {
                "default_provider": "ollama",
                "ollama_host": _ollama_host(),
                "lmstudio_host": lmstudio_model_catalog().get("host", ""),
            },
            "recommended_song_model": recommended_song_model(installed_models),
            "preferred_lm_model": ACE_LM_PREFERRED_MODEL,
            "recommended_lm_model": recommended_lm_model(installed_lms),
            "available_song_models": available_models,
            "installed_song_models": sorted(installed_models),
            "installed_lm_models": sorted(installed_lms),
            "model_labels": {name: model_profiles[name]["label"] for name in available_models},
            "model_profiles": model_profiles,
            "lm_model_profiles": lm_profiles,
            "model_capabilities": _model_capabilities(),
            "model_downloads": {name: _model_download_job(name) for name in sorted(_downloadable_model_names())},
            "official_runner": _official_runner_status(),
            "official_ace_step_manifest": official_manifest(),
            "official_parity": _official_parity_payload()["manifest"],
            "ui_schema": studio_ui_schema(),
            "songwriting_toolkit": _songwriting_toolkit_payload(),
            "task_types": TASK_TYPES,
            "track_names": TRACK_NAMES,
            "valid_languages": VALID_LANGUAGES,
            "valid_time_signatures": VALID_TIME_SIGNATURES,
            "lm_models": lm_models,
            "ace_lm": _ace_lm_status_payload(),
            "lora": handler.get_lora_status(),
            "trainer": training_manager.status(),
        }
    )


@app.api(name="ollama_models", concurrency_limit=8)
def ollama_models() -> str:
    """List available Ollama models using the official ollama library."""
    return json.dumps(_ollama_model_catalog())


@app.api(name="generate_album", concurrency_limit=1, time_limit=3600)
def generate_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
    language: str = "en",
    song_model: str = "auto",
    embedding_model: str = DEFAULT_ALBUM_EMBEDDING_MODEL,
    ace_lm_model: str = ACE_LM_PREFERRED_MODEL,
    request_json: str = "",
    planner_lm_provider: str = "ollama",
    embedding_lm_provider: str = "ollama",
) -> str:
    """Plan album with tools/CrewAI, then generate through the advanced engine."""
    logs: list[str] = []
    try:
        request_payload = json.loads(request_json or "{}")
        if "ace_lm_model" not in request_payload and ace_lm_model:
            request_payload["ace_lm_model"] = ace_lm_model
        ace_lm_model = _requested_ace_lm_model(request_payload)
        request_payload["ace_lm_model"] = ace_lm_model
        album_job_id = str(request_payload.get("album_job_id") or "")
        planner_lm_provider = normalize_provider(request_payload.get("planner_lm_provider") or planner_lm_provider or "ollama")
        embedding_lm_provider = normalize_provider(request_payload.get("embedding_lm_provider") or request_payload.get("embedding_provider") or embedding_lm_provider or planner_lm_provider)
        if not ollama_model:
            ollama_model = DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL if planner_lm_provider == "ollama" else ""
        if not embedding_model:
            embedding_model = DEFAULT_ALBUM_EMBEDDING_MODEL
        track_duration = parse_duration_seconds(request_payload.get("track_duration") or request_payload.get("duration") or track_duration, track_duration)
        album_options = _album_options_from_payload(request_payload, song_model=song_model)
        planned_tracks = _json_list(request_payload.get("tracks") or request_payload.get("planned_tracks"))
        strategy = str(album_options.get("song_model_strategy") or "all_models_album")
        album_models = album_models_for_strategy(
            strategy,
            _installed_acestep_models(),
            str(album_options.get("requested_song_model") or request_payload.get("song_model") or song_model or ""),
        )
        if not album_models:
            raise RuntimeError(f"No album models resolved for strategy {strategy}.")
        missing_models = [item["model"] for item in album_models if item.get("model") not in _installed_acestep_models()]
        if missing_models:
            downloadable_missing = [model for model in missing_models if model in _downloadable_model_names()]
            if len(downloadable_missing) != len(missing_models):
                blocked = sorted(set(missing_models) - set(downloadable_missing))
                raise RuntimeError(f"Album model(s) cannot be downloaded: {', '.join(blocked)}")
            logs.append(f"Album strategy {strategy} needs {len(album_models)} model album(s).")
            logs.append(f"Missing model(s): {', '.join(downloadable_missing)}")
            return json.dumps(
                _album_missing_download_payload(
                    downloadable_missing,
                    logs,
                    tracks=planned_tracks,
                    album_model_strategy=strategy,
                    album_model_portfolio=album_models,
                )
            )

        from album_crew import plan_album as _plan_album

        logs.append("Phase 1: Planning album with Hit Album Agent tools...")
        _album_job_log(album_job_id, "Phase 1: Planning album with Hit Album Agent tools.", status="Planning album", progress=3)
        result = _plan_album(
            concept=concept,
            num_tracks=num_tracks,
            track_duration=track_duration,
            ollama_model=ollama_model,
            language=language,
            embedding_model=embedding_model,
            options=album_options,
            use_crewai=not planned_tracks,
            input_tracks=planned_tracks if planned_tracks else None,
            planner_provider=planner_lm_provider,
            embedding_provider=embedding_lm_provider,
        )
        tracks = result.get("tracks", [])
        logs.extend(result.get("logs", []))
        actual_memory = ((result.get("toolkit_report") or {}).get("memory") or {}) if isinstance(result.get("toolkit_report"), dict) else {}
        if actual_memory.get("embedding_model"):
            embedding_model = str(actual_memory.get("embedding_model") or embedding_model)
        if album_job_id:
            _set_album_job(
                album_job_id,
                logs=result.get("logs", []),
                status="Album plan ready",
                progress=8,
                planner_model=ollama_model,
                planner_provider=planner_lm_provider,
                embedding_model=embedding_model,
                embedding_provider=embedding_lm_provider,
                planning_engine=str(result.get("planning_engine") or ""),
                crewai_used=bool(result.get("crewai_used")),
                toolbelt_fallback=bool(result.get("toolbelt_fallback")),
                input_contract=result.get("input_contract") or contract_prompt_context(album_options.get("user_album_contract")),
                input_contract_applied=bool(result.get("input_contract_applied") or album_options.get("input_contract_applied")),
                input_contract_version=str(result.get("input_contract_version") or USER_ALBUM_CONTRACT_VERSION),
                blocked_unsafe_count=int(result.get("blocked_unsafe_count") or album_options.get("blocked_unsafe_count") or 0),
                contract_repair_count=int(result.get("contract_repair_count") or 0),
            )

        if not result.get("success", True) or not tracks or "error" in tracks[0]:
            logs.append("ERROR: Album planning failed")
            return json.dumps({"tracks": tracks, "logs": logs, "success": False, "error": result.get("error") or "Planning failed"})

        logs.append(f"Phase 1 complete: {len(tracks)} tracks planned")
        logs.append(
            f"Planner LM: {provider_label(planner_lm_provider)} ({ollama_model}); ACE-Step LM "
            f"{ace_lm_model if ace_lm_model != 'none' else 'off unless official LM controls are enabled'}."
        )
        logs.append(f"Album model policy: {len(album_models)} full model album(s), {len(tracks) * len(album_models)} total render(s)")
        logs.append("---")
        # Free unified memory: unload LLM models before heavy audio generation
        _unload_llm_models_for_generation()
        logs.append("LLM models unloaded to maximize memory for audio generation.")
        logs.append("Phase 2: Generating every track through the album model portfolio...")
        _album_job_log(
            album_job_id,
            "Phase 2: Generating every track through the album model portfolio.",
            status="Generating model albums",
            progress=10,
            expected_count=len(tracks) * len(album_models),
        )

        album_family_id = uuid.uuid4().hex[:12]
        album_family_title = safe_filename(concept[:48] or "album", "album")
        generated_audios: list[dict[str, Any]] = []
        model_albums: list[dict[str, Any]] = []
        default_variants = clamp_int(album_options.get("track_variants"), 1, 1, MAX_BATCH_SIZE)
        for base_track in tracks:
            base_track["model_results"] = []
            base_track["audios"] = []

        for model_index, model_item in enumerate(album_models, start=1):
            track_model = str(model_item["model"])
            album_model_slug = safe_filename(str(model_item.get("slug") or _model_slug(track_model)), _model_slug(track_model))
            album_id = f"{album_family_id}-{album_model_slug}"
            logs.append(f"Model album {model_index}/{len(album_models)}: {model_item.get('label') or track_model} ({track_model})")
            _album_job_log(
                album_job_id,
                f"Model album {model_index}/{len(album_models)}: {model_item.get('label') or track_model} ({track_model})",
                current_model_album=track_model,
                status=f"Generating {model_item.get('label') or track_model}",
            )
            album_tracks: list[dict[str, Any]] = []
            album_audios: list[dict[str, Any]] = []
            album_success_count = 0

            if track_model not in _installed_acestep_models():
                if track_model in _downloadable_model_names():
                    logs.append(f"Model album {track_model} is missing. Starting download instead of falling back.")
                    return json.dumps(
                        _album_missing_download_payload(
                            [track_model],
                            logs,
                            tracks=tracks,
                            album_family_id=album_family_id,
                            album_id=album_id,
                            album_model_strategy=strategy,
                            album_model_portfolio=album_models,
                        )
                    )
                raise RuntimeError(f"{track_model} is not installed and is not in the known ACE-Step download list.")

            for i, base_track in enumerate(tracks):
                track = dict(base_track)
                track.pop("model_results", None)
                track.pop("audios", None)
                track_title = track.get("title", f"Track {i+1}")
                track_artist = normalize_artist_name(
                    track.get("artist_name") or track.get("artist"),
                    derive_artist_name(track_title, concept, track.get("tags") or track.get("caption") or "", i),
                )
                track["artist_name"] = track_artist
                track["song_model"] = track_model
                track["album_model"] = track_model
                track["album_model_label"] = model_item.get("label") or track_model
                track["final_model_policy"] = {
                    "model": track_model,
                    "model_label": model_item.get("label") or track_model,
                    "locked": True,
                    "strategy": strategy,
                    "reason": "This model-specific album render uses the fixed portfolio model without fallback.",
                }
                raw_model_settings = track.get("model_render_settings") or track.get("per_model_settings") or {}
                model_render_settings = {}
                if isinstance(raw_model_settings, dict):
                    model_render_settings = raw_model_settings.get(track_model) or raw_model_settings.get(album_model_slug) or {}
                    if not isinstance(model_render_settings, dict):
                        model_render_settings = {}
                variants = clamp_int(track.get("track_variants", default_variants), default_variants, 1, MAX_BATCH_SIZE)
                logs.append(f"  Track {i+1}/{len(tracks)}: {track_title} ({variants} variant{'s' if variants != 1 else ''})")
                print(f"[generate_album] Generating {album_id} track {i+1}/{len(tracks)}: {track_title}")
                _album_job_log(
                    album_job_id,
                    f"Generating {album_id} track {i+1}/{len(tracks)}: {track_title}",
                    current_model_album=track_model,
                    current_track=f"{i + 1}/{len(tracks)} {track_title}",
                    status=f"{model_item.get('label') or track_model}: track {i + 1}/{len(tracks)}",
                    progress=10 + int(((model_index - 1) * len(tracks) + i) / max(1, len(album_models) * len(tracks)) * 85),
                )

                try:
                    _cleanup_accelerator_memory()
                    track_lm_model = str(track.get("ace_lm_model") or request_payload.get("ace_lm_model") or ACE_LM_PREFERRED_MODEL)
                    track_lm_enabled = _requested_ace_lm_model({"ace_lm_model": track_lm_model}) != "none"
                    track_has_vocal_lyrics = bool(str(track.get("lyrics") or "").strip() and str(track.get("lyrics") or "").strip().lower() != "[instrumental]")
                    quality_profile = normalize_quality_profile(track.get("quality_profile") or request_payload.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
                    model_defaults = quality_profile_model_settings(track_model, quality_profile)
                    generation_payload = {
                        "task_type": "text2music",
                        "ui_mode": "album",
                        "quality_profile": quality_profile,
                        "artist_name": track_artist,
                        "title": track_title,
                        "description": track.get("description", ""),
                        "caption": track.get("tags") or track.get("caption") or "",
                        "lyrics": track.get("lyrics", ""),
                        "duration": track.get("duration") or track_duration,
                        "bpm": track.get("bpm") or request_payload.get("bpm") or DEFAULT_BPM,
                        "key_scale": track.get("key_scale") or request_payload.get("key_scale") or DEFAULT_KEY_SCALE,
                        "time_signature": track.get("time_signature") or request_payload.get("time_signature") or "4",
                        "vocal_language": track.get("language") or request_payload.get("vocal_language") or language,
                        "batch_size": variants,
                        "seed": str(track.get("seed") or request_payload.get("seed") or request_payload.get("seeds") or "-1"),
                        "song_model": track_model,
                        "ace_lm_model": track_lm_model,
                        "global_caption": request_payload.get("global_caption") or concept,
                        "inference_steps": clamp_int(
                            model_render_settings.get("inference_steps", model_item.get("default_steps")),
                            int(model_item.get("default_steps") or _quality_default_steps(track_model, quality_profile)),
                            1,
                            200,
                        ),
                        "guidance_scale": clamp_float(
                            model_render_settings.get("guidance_scale", model_item.get("default_guidance_scale")),
                            float(model_item.get("default_guidance_scale") or model_defaults["guidance_scale"]),
                            1.0,
                            15.0,
                        ),
                        "shift": clamp_float(
                            model_render_settings.get("shift", model_item.get("default_shift")),
                            float(model_item.get("default_shift") or model_defaults["shift"]),
                            1.0,
                            5.0,
                        ),
                        "infer_method": str(model_render_settings.get("infer_method") or track.get("infer_method") or request_payload.get("infer_method") or model_defaults["infer_method"]),
                        "sampler_mode": str(model_render_settings.get("sampler_mode") or track.get("sampler_mode") or request_payload.get("sampler_mode") or model_defaults["sampler_mode"]),
                        "use_adg": parse_bool(model_render_settings.get("use_adg", track.get("use_adg", request_payload.get("use_adg"))), bool(model_defaults.get("use_adg", False))),
                        "cfg_interval_start": clamp_float(model_render_settings.get("cfg_interval_start", track.get("cfg_interval_start", request_payload.get("cfg_interval_start"))), 0.0, 0.0, 1.0),
                        "cfg_interval_end": clamp_float(model_render_settings.get("cfg_interval_end", track.get("cfg_interval_end", request_payload.get("cfg_interval_end"))), 1.0, 0.0, 1.0),
                        "timesteps": model_render_settings.get("timesteps") or track.get("timesteps") or request_payload.get("timesteps") or "",
                        "audio_format": str(track.get("audio_format") or request_payload.get("audio_format") or model_defaults["audio_format"]),
                        "mp3_bitrate": str(track.get("mp3_bitrate") or request_payload.get("mp3_bitrate") or "128k"),
                        "mp3_sample_rate": track.get("mp3_sample_rate") or request_payload.get("mp3_sample_rate") or 48000,
                        "auto_score": parse_bool(track.get("auto_score", request_payload.get("auto_score")), False),
                        "auto_lrc": parse_bool(track.get("auto_lrc", request_payload.get("auto_lrc")), False),
                        "return_audio_codes": parse_bool(track.get("return_audio_codes", request_payload.get("return_audio_codes")), False),
                        "save_to_library": parse_bool(track.get("save_to_library", request_payload.get("save_to_library")), True),
                        "thinking": parse_bool(track.get("thinking", False if track_has_vocal_lyrics else request_payload.get("thinking")), DOCS_BEST_LM_DEFAULTS["thinking"] if track_lm_enabled and not track_has_vocal_lyrics else False),
                        "use_format": parse_bool(track.get("use_format", False if track_has_vocal_lyrics else request_payload.get("use_format")), DOCS_BEST_LM_DEFAULTS["use_format"] if track_lm_enabled and not track_has_vocal_lyrics else False),
                        "lm_temperature": clamp_float(track.get("lm_temperature", request_payload.get("lm_temperature")), DOCS_BEST_LM_DEFAULTS["lm_temperature"] if track_lm_enabled else 0.85, 0.0, 2.0),
                        "lm_cfg_scale": clamp_float(track.get("lm_cfg_scale", request_payload.get("lm_cfg_scale")), DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"] if track_lm_enabled else 2.0, 0.0, 10.0),
                        "lm_top_k": clamp_int(track.get("lm_top_k", request_payload.get("lm_top_k")), DOCS_BEST_LM_DEFAULTS["lm_top_k"] if track_lm_enabled else 0, 0, 200),
                        "lm_top_p": clamp_float(track.get("lm_top_p", request_payload.get("lm_top_p")), DOCS_BEST_LM_DEFAULTS["lm_top_p"] if track_lm_enabled else 0.9, 0.0, 1.0),
                        "lm_repetition_penalty": clamp_float(track.get("lm_repetition_penalty", request_payload.get("lm_repetition_penalty")), 1.0, 0.1, 4.0),
                        "use_cot_metas": parse_bool(track.get("use_cot_metas", request_payload.get("use_cot_metas")), DOCS_BEST_LM_DEFAULTS["use_cot_metas"] if track_lm_enabled and not track_has_vocal_lyrics else False),
                        "use_cot_caption": parse_bool(track.get("use_cot_caption", request_payload.get("use_cot_caption")), DOCS_BEST_LM_DEFAULTS["use_cot_caption"] if track_lm_enabled and not track_has_vocal_lyrics else False),
                        "use_cot_lyrics": parse_bool(track.get("use_cot_lyrics", False if track_has_vocal_lyrics else request_payload.get("use_cot_lyrics")), DOCS_BEST_LM_DEFAULTS["use_cot_lyrics"] if track_lm_enabled and not track_has_vocal_lyrics else False),
                        "use_cot_language": parse_bool(track.get("use_cot_language", request_payload.get("use_cot_language")), DOCS_BEST_LM_DEFAULTS["use_cot_language"] if track_lm_enabled and not track_has_vocal_lyrics else False),
                        "use_constrained_decoding": parse_bool(track.get("use_constrained_decoding", request_payload.get("use_constrained_decoding")), DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"]),
                        "album_metadata": {
                            "album_family_id": album_family_id,
                            "album_family_title": album_family_title,
                            "album_id": album_id,
                            "artist_name": track_artist,
                            "album_model": track_model,
                            "album_model_slug": album_model_slug,
                            "album_model_label": model_item.get("label") or track_model,
                            "album_model_summary": model_item.get("summary") or "",
                            "album_concept": concept,
                            "album_options": _jsonable(album_options),
                            "album_model_portfolio": _jsonable(album_models),
                            "album_toolkit_report": _jsonable(result.get("toolkit_report", {})),
                            "track_number": track.get("track_number", i + 1),
                            "track_variant": "batch",
                            "tool_report": _jsonable(track.get("tool_report", {})),
                            "production_team": _jsonable(track.get("production_team", {})),
                            "model_render_settings": _jsonable(model_render_settings),
                            "final_model_policy": _jsonable(track.get("final_model_policy", {})),
                            "tag_list": track.get("tag_list", []),
                        },
                    }
                    payload_validation = _validate_generation_payload(generation_payload)
                    track["payload_validation"] = payload_validation
                    track["payload_warnings"] = payload_validation.get("payload_warnings", [])
                    if not payload_validation.get("valid"):
                        raise ValueError(f"Invalid track payload: {payload_validation.get('field_errors')}")
                    generation_result = _run_advanced_generation(generation_payload)
                    if not generation_result.get("success"):
                        raise RuntimeError(generation_result.get("error") or "Track generation failed")

                    track["result_id"] = generation_result.get("result_id")
                    track["active_song_model"] = generation_result.get("active_song_model")
                    track["audios"] = generation_result.get("audios", [])
                    track["payload_warnings"] = generation_result.get("payload_warnings", [])
                    track["runner"] = generation_result.get("runner")
                    track["generation_params"] = generation_result.get("params", {})
                    track["album_id"] = album_id
                    track["album_family_id"] = album_family_id
                    if track["audios"]:
                        first_audio = track["audios"][0]
                        track["song_id"] = first_audio.get("song_id")
                        track["audio_url"] = first_audio.get("audio_url") or first_audio.get("library_url")
                    for audio_index, audio in enumerate(track["audios"]):
                        audio["artist_name"] = track_artist
                        audio["album_id"] = album_id
                        audio["album_family_id"] = album_family_id
                        audio["album_model"] = track_model
                        audio["album_model_label"] = model_item.get("label") or track_model
                        if audio.get("song_id"):
                            _merge_song_album_metadata(
                                audio["song_id"],
                                {
                                    "artist_name": track_artist,
                                    "album_concept": concept,
                                    "album_family_id": album_family_id,
                                    "album_id": album_id,
                                    "album_model": track_model,
                                    "album_model_label": model_item.get("label") or track_model,
                                    "track_number": track.get("track_number", i + 1),
                                    "track_variant": audio_index + 1,
                                    "album_toolkit_report": result.get("toolkit_report", {}),
                                    "tool_report": track.get("tool_report", {}),
                                    "production_team": track.get("production_team", {}),
                                    "final_model_policy": track.get("final_model_policy", {}),
                                    "tag_list": track.get("tag_list", []),
                                },
                            )
                    generated_audios.extend(track["audios"])
                    album_audios.extend(track["audios"])
                    track["generated"] = True
                    album_success_count += 1
                    base_track["model_results"].append(_jsonable(track))
                    base_track["audios"].extend(_jsonable(track["audios"]))

                    logs.append(f"    Done: {track_title} -> {track.get('result_id')}")
                    print(f"[generate_album] {album_id} track {i+1} generated: {track.get('result_id')}")
                    _album_job_log(
                        album_job_id,
                        f"Generated {album_id} track {i+1}: {track_title}",
                        generated_count=len(generated_audios),
                        progress=10 + int(((model_index - 1) * len(tracks) + i + 1) / max(1, len(album_models) * len(tracks)) * 85),
                    )

                except Exception as track_exc:
                    track["generated"] = False
                    track["error"] = str(track_exc)
                    base_track["model_results"].append(_jsonable(track))
                    logs.append(f"    FAILED: {track_exc}")
                    print(f"[generate_album] {album_id} track {i+1} failed: {track_exc}")
                    _album_job_log(
                        album_job_id,
                        f"FAILED {album_id} track {i+1}: {track_exc}",
                        errors=[str(track_exc)],
                    )
                finally:
                    _cleanup_accelerator_memory()

                album_tracks.append(track)

            album_success = album_success_count == len(tracks)
            album_status = "completed" if album_success else "incomplete"
            album_manifest = _write_album_manifest(
                album_id,
                {
                    "album_family_id": album_family_id,
                    "album_concept": concept,
                    "album_model": track_model,
                    "album_model_label": model_item.get("label") or track_model,
                    "album_model_summary": model_item.get("summary") or "",
                    "album_status": album_status,
                    "track_count": len(tracks),
                    "generated_count": album_success_count,
                    "tracks": album_tracks,
                    "audios": album_audios,
                    "toolkit_report": result.get("toolkit_report", {}),
                    "album_options": album_options,
                    "download_url": f"/api/albums/{album_id}/download",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            model_albums.append(
                {
                    "album_id": album_id,
                    "album_family_id": album_family_id,
                    "album_model": track_model,
                    "album_model_label": model_item.get("label") or track_model,
                    "album_model_summary": model_item.get("summary") or "",
                    "album_status": album_status,
                    "track_count": len(tracks),
                    "generated_count": album_success_count,
                    "audios": album_audios,
                    "tracks": album_tracks,
                    "download_url": f"/api/albums/{album_id}/download",
                    "manifest": album_manifest,
                }
            )
            logs.append(f"  Model album {album_status}: {album_success_count}/{len(tracks)} tracks.")

        generated_count = sum(int(album.get("generated_count") or 0) for album in model_albums)
        expected_count = len(tracks) * len(album_models)
        album_success = generated_count == expected_count
        album_status = "completed" if album_success else "incomplete"
        family_manifest = _write_album_manifest(
            album_family_id,
            {
                "album_family_id": album_family_id,
                "album_concept": concept,
                "album_status": album_status,
                "strategy": strategy,
                "track_count": len(tracks),
                "model_count": len(album_models),
                "expected_renders": expected_count,
                "generated_count": generated_count,
                "model_albums": model_albums,
                "album_model_portfolio": album_models,
                "toolkit_report": result.get("toolkit_report", {}),
                "album_options": album_options,
                "download_url": f"/api/album-families/{album_family_id}/download",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logs.append("---")
        logs.append(f"Album family {album_status}: {generated_count}/{expected_count} track/model renders generated.")
        _album_job_log(
            album_job_id,
            f"Album family {album_status}: {generated_count}/{expected_count} track/model renders generated.",
            generated_count=generated_count,
            expected_count=expected_count,
            album_family_id=album_family_id,
            download_url=f"/api/album-families/{album_family_id}/download",
            progress=100 if album_success else 98,
        )
        if not album_success:
            logs.append("Album family marked incomplete because every requested track/model render must succeed.")

        return json.dumps({
            "tracks": tracks,
            "audios": generated_audios,
            "album_id": model_albums[0]["album_id"] if model_albums else album_family_id,
            "album_family_id": album_family_id,
            "model_albums": model_albums,
            "album_model_portfolio": album_models,
            "album_status": album_status,
            "expected_renders": expected_count,
            "generated_count": generated_count,
            "final_song_model": "all_models_album" if strategy == "all_models_album" else (album_models[0]["model"] if album_models else ALBUM_FINAL_MODEL),
            "family_download_url": f"/api/album-families/{album_family_id}/download",
            "manifest": family_manifest,
            "toolkit": result.get("toolkit", _songwriting_toolkit_payload()),
            "toolkit_report": result.get("toolkit_report", {}),
            "planner_model": ollama_model,
            "planner_provider": planner_lm_provider,
            "embedding_model": embedding_model,
            "embedding_provider": embedding_lm_provider,
            "logs": logs,
            "success": album_success,
            "error": "" if album_success else f"Album incomplete: {generated_count}/{expected_count} track/model renders generated.",
        })
    except ModelDownloadStarted as exc:
        print(f"[generate_album DOWNLOAD] {exc.message}")
        logs.append(exc.message)
        return json.dumps(_download_started_payload(exc.model_name, exc.job, logs))
    except Exception as exc:
        print(f"[generate_album ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        logs.append(f"ERROR: {exc}")
        return json.dumps({"tracks": [], "logs": logs, "success": False, "error": str(exc)})


@app.api(name="generate_advanced", concurrency_limit=1, time_limit=3600)
def generate_advanced(request_json: str) -> str:
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(request_json or "{}")
        return json.dumps(_run_advanced_generation(payload))
    except ModelDownloadStarted as exc:
        print(f"[generate_advanced DOWNLOAD] {exc.message}")
        return json.dumps(_download_started_payload(exc.model_name, exc.job))
    except Exception as exc:
        print(f"[generate_advanced ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return json.dumps({"success": False, "error": str(exc), "validation": _validate_generation_payload(payload)})
    finally:
        _cleanup_accelerator_memory()


@app.api(name="generate_portfolio", concurrency_limit=1, time_limit=7200)
def generate_portfolio(request_json: str) -> str:
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(request_json or "{}")
        return json.dumps(_run_model_portfolio_generation(payload))
    except Exception as exc:
        print(f"[generate_portfolio ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return json.dumps({"success": False, "error": str(exc), "validation": _validate_generation_payload(payload)})
    finally:
        _cleanup_accelerator_memory()


def _api_timestamp() -> int:
    return int(time.time() * 1000)


def _official_api_response(data: Any = None, *, error: str | None = None, code: int = 200) -> dict[str, Any]:
    return {"data": data, "code": code, "error": error, "timestamp": _api_timestamp(), "extra": None}


async def _require_official_api_key(request: Request) -> None:
    expected = os.environ.get("ACESTEP_API_KEY", "").strip()
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    supplied = request.headers.get("x-api-key") or request.query_params.get("api_key") or bearer
    if supplied != expected and request.method.upper() not in {"GET", "HEAD"}:
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.json()
                if isinstance(body, dict):
                    supplied = body.get("ai_token") or body.get("api_key") or supplied
            elif "form" in content_type or "multipart" in content_type:
                form = await request.form()
                supplied = form.get("ai_token") or form.get("api_key") or supplied
        except Exception:
            pass
    if supplied != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing ACE-Step API key")


def _job_stats() -> dict[str, Any]:
    with _api_generation_tasks_lock:
        tasks = list(_api_generation_tasks.values())
    counts = {
        "total": len(tasks),
        "queued": sum(1 for item in tasks if item.get("state") == "queued"),
        "running": sum(1 for item in tasks if item.get("state") == "running"),
        "succeeded": sum(1 for item in tasks if item.get("state") == "succeeded"),
        "failed": sum(1 for item in tasks if item.get("state") == "failed"),
    }
    return {
        "jobs": counts,
        "queue_size": counts["queued"],
        "queue_maxsize": 200,
        "avg_job_seconds": 0.0,
        "active_downloads": len([job for job in _model_download_jobs.values() if job.get("state") in {"queued", "running"}]),
        "active_training_job": training_manager.active_job(),
    }


def _runtime_progress_snapshot() -> dict[str, Any]:
    if not PINOKIO_START_LOG.is_file():
        return {"success": True, "available": False, "log_path": str(PINOKIO_START_LOG), "lines": []}
    try:
        with PINOKIO_START_LOG.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 24000))
            text = handle.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"success": True, "available": False, "error": str(exc), "log_path": str(PINOKIO_START_LOG), "lines": []}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if (
            "[official_runner]" in line
            or "[generate_album]" in line
            or "[generate_advanced]" in line
            or "it/s]" in line
            or "s/it]" in line
            or "%|" in line
            or "loading 5Hz LM" in line
            or "initialize_service" in line
        )
    ]
    latest = interesting[-1] if interesting else (lines[-1] if lines else "")
    progress: dict[str, Any] = {
        "success": True,
        "available": True,
        "log_path": str(PINOKIO_START_LOG),
        "backend": _runtime_backend_label(),
        "latest": latest,
        "lines": interesting[-20:],
    }
    match = re.search(r"(\d{1,3})%\|.*?\|\s*(\d+)/(\d+).*?\[([^<\]]+)<([^,\]]+),\s*([^\]]+)\]", latest)
    if match:
        progress.update(
            {
                "percent": int(match.group(1)),
                "step": int(match.group(2)),
                "total_steps": int(match.group(3)),
                "elapsed": match.group(4).strip(),
                "eta": match.group(5).strip(),
                "speed": match.group(6).strip(),
            }
        )
    return progress


def _lora_dataset_health(files: list[dict[str, Any]] | None) -> dict[str, Any]:
    items = [item for item in (files or []) if isinstance(item, dict)]
    count = len(items)
    labeled = sum(1 for item in items if str(item.get("caption") or "").strip())
    lyrics_ready = sum(1 for item in items if str(item.get("lyrics") or "").strip())
    durations = [clamp_float(item.get("duration"), 0.0, 0.0, 600.0) for item in items]
    durations = [value for value in durations if value > 0]
    total_duration = round(sum(durations), 2)
    avg_duration = round(total_duration / len(durations), 2) if durations else 0.0
    languages = sorted({str(item.get("language") or "unknown") for item in items})
    score = 0
    checks = []

    def add(check_id: str, ok: bool, detail: str, points: int) -> None:
        nonlocal score
        if ok:
            score += points
        checks.append({"id": check_id, "status": "pass" if ok else "warn", "detail": detail, "points": points})

    add("sample_count", count >= 8, f"{count} audio sample(s)", 25)
    add("caption_labels", count > 0 and labeled == count, f"{labeled}/{count} captioned", 25)
    add("lyrics_labels", count > 0 and lyrics_ready == count, f"{lyrics_ready}/{count} lyrics/instrumental labels", 15)
    add("duration_total", total_duration >= 300, f"{total_duration}s total", 20)
    add("duration_shape", not durations or 5 <= avg_duration <= 300, f"{avg_duration}s average", 10)
    add("language_metadata", bool(languages), ", ".join(languages[:6]) or "unknown", 5)
    status = "ready" if score >= 85 else "usable" if score >= 65 else "needs_work"
    return {
        "version": PRO_QUALITY_AUDIT_VERSION,
        "status": status,
        "score": min(100, score),
        "sample_count": count,
        "labeled_count": labeled,
        "lyrics_ready_count": lyrics_ready,
        "total_duration_seconds": total_duration,
        "average_duration_seconds": avg_duration,
        "languages": languages,
        "checks": checks,
        "audition_plan": {
            "adapter_scales": [0.3, 0.6, 0.8, 1.0],
            "compare_to_baseline": True,
            "recommended_seed_count": 2,
        },
    }


def _model_runtime_status(name: str) -> dict[str, Any]:
    download_job = _model_download_job(name)
    unreleased = name in OFFICIAL_UNRELEASED_MODELS
    installed = name in _installed_acestep_models()
    status = {
        "installed": installed,
        "downloadable": name in _downloadable_model_names(),
        "downloading": bool(download_job and download_job.get("state") in {"queued", "running"}),
        "download_job": _jsonable(download_job),
        "status": "unreleased" if unreleased else ("installed" if installed else "download_required"),
    }
    if unreleased:
        status["error"] = f"{name} is official but unreleased; it cannot be downloaded yet."
    return status


def _vendor_dataclass_fields(class_name: str) -> list[str]:
    inference_path = OFFICIAL_ACE_STEP_DIR / "acestep" / "inference.py"
    try:
        tree = ast.parse(inference_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                fields: list[str] = []
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields.append(item.target.id)
                return sorted(fields)
    except Exception as exc:
        print(f"[parity] unable to inspect {class_name} from {inference_path}: {exc}")
    try:
        from acestep import inference as inference_module

        cls = getattr(inference_module, class_name)
        fields = getattr(cls, "__dataclass_fields__", None) or {}
        return sorted(fields)
    except Exception as exc:
        print(f"[parity] fallback import failed for {class_name}: {exc}")
        return []


def _runtime_backend_label() -> str:
    if torch.cuda.is_available():
        return "PyTorch (cuda)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "PyTorch (mps)"
    return "PyTorch (cpu)"


def _schema_parity(manifest: dict[str, Any]) -> dict[str, Any]:
    vendor_params = _vendor_dataclass_fields("GenerationParams")
    vendor_config = _vendor_dataclass_fields("GenerationConfig")
    manifest_params = sorted(manifest.get("generation_params", {}))
    manifest_config = sorted(manifest.get("generation_config", {}))
    return {
        "generation_params": {
            "vendor_fields": vendor_params,
            "manifest_fields": manifest_params,
            "unsupported_by_vendor": sorted(set(manifest_params) - set(vendor_params)),
            "missing_in_manifest": sorted(set(vendor_params) - set(manifest_params)),
        },
        "generation_config": {
            "vendor_fields": vendor_config,
            "manifest_fields": manifest_config,
            "unsupported_by_vendor": sorted(set(manifest_config) - set(vendor_config)),
            "missing_in_manifest": sorted(set(vendor_config) - set(manifest_config)),
        },
    }


def _official_parity_payload(request: Request | None = None) -> dict[str, Any]:
    manifest = official_manifest()
    installed_models = _installed_acestep_models()
    installed_lms = _installed_lm_models()
    recommended_actions: list[str] = []
    for name, meta in manifest.get("dit_models", {}).items():
        meta.update(_model_runtime_status(name))
        meta["tasks"] = supported_tasks_for_model(name)
    for name, meta in manifest.get("lm_models", {}).items():
        job = _model_download_job(name)
        installed = name in installed_lms
        meta.update(
            {
                "installed": installed,
                "downloadable": name in _downloadable_model_names(),
                "downloading": bool(job and job.get("state") in {"queued", "running"}),
                "download_job": _jsonable(job),
                "status": "installed" if installed else "download_required",
            }
        )
    schema_parity = _schema_parity(manifest)
    if schema_parity["generation_params"]["unsupported_by_vendor"]:
        recommended_actions.append("AceJAM will drop unsupported GenerationParams fields before calling the official runner.")
    if DOCS_BEST_DEFAULT_LM_MODEL not in installed_lms:
        recommended_actions.append(f"Install {DOCS_BEST_DEFAULT_LM_MODEL}; it is the Docs-best default for official LM controls.")
    manifest["runtime"] = {
        "app_version": APP_UI_VERSION,
        "ui_hash": _app_ui_hash(),
        "backend_hash": _backend_code_hash(),
        "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
        "active_song_model": ACTIVE_ACE_STEP_MODEL,
        "installed_song_models": sorted(installed_models),
        "installed_lm_models": sorted(installed_lms),
        "backend": _runtime_backend_label(),
        "official_runner": _official_runner_status(),
        "trainer": training_manager.status(),
        "stats": _job_stats(),
        "server_url": str(request.base_url).rstrip("/") if request is not None else "",
        "api_key_enabled": bool(os.environ.get("ACESTEP_API_KEY", "").strip()),
    }
    manifest["quality_policy"] = docs_best_quality_policy()
    manifest["settings_registry"] = studio_ui_schema().get("ace_step_settings_registry", {})
    manifest["schema_parity"] = schema_parity
    manifest["lm_task_policy"] = docs_best_quality_policy()["lm_task_policy"]
    manifest["recommended_actions"] = recommended_actions
    return {"success": True, "manifest": manifest}


async def _request_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        return dict(body or {}) if isinstance(body, dict) else {}
    form = await request.form()
    payload: dict[str, Any] = {}
    for key, value in form.items():
        if isinstance(value, str):
            payload[key] = value
            continue
        filename = getattr(value, "filename", "")
        read = getattr(value, "read", None)
        if filename and callable(read):
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_AUDIO_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported audio file: {suffix}")
            upload_id = uuid.uuid4().hex[:12]
            upload_dir = UPLOADS_DIR / upload_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            target = upload_dir / f"{safe_filename(filename)}{suffix}"
            target.write_bytes(await read())
            if key in {"src_audio", "source_audio", "src_audio_path"}:
                payload["src_audio_id"] = upload_id
            elif key in {"reference_audio", "reference", "reference_audio_path"}:
                payload["reference_audio_id"] = upload_id
            else:
                payload[key] = upload_id
    return payload


def _set_api_generation_task(task_id: str, **updates: Any) -> dict[str, Any]:
    with _api_generation_tasks_lock:
        task = _api_generation_tasks.setdefault(
            task_id,
            {
                "task_id": task_id,
                "status": 0,
                "state": "queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "payload": {},
                "result": None,
                "error": "",
            },
        )
        task.update(_jsonable(updates))
        task["updated_at"] = datetime.now(timezone.utc).isoformat()
        return dict(task)


def _generation_task_worker(task_id: str, payload: dict[str, Any]) -> None:
    _set_api_generation_task(task_id, state="running", status=0)
    try:
        result = _run_advanced_generation_with_download_retry(payload)
        _set_api_generation_task(task_id, state="succeeded", status=1, result=result, error="")
    except Exception as exc:
        _set_api_generation_task(task_id, state="failed", status=2, result=None, error=str(exc))
    finally:
        _cleanup_accelerator_memory()


def _submit_api_generation_task(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    _set_api_generation_task(task_id, payload=payload)
    thread = threading.Thread(target=_generation_task_worker, args=(task_id, payload), daemon=True)
    thread.start()
    return {"task_id": task_id, "status": 0}


def _official_query_item(task_id: str) -> dict[str, Any]:
    with _api_generation_tasks_lock:
        task = dict(_api_generation_tasks.get(task_id) or {})
    if not task:
        return {"task_id": task_id, "status": 2, "result": "", "error": "Task not found"}
    status_code = int(task.get("status") or 0)
    if status_code == 1:
        result = task.get("result") or {}
        result_payload = []
        for audio in result.get("audios", []):
            params = result.get("params") or {}
            result_payload.append(
                {
                    "file": audio.get("download_url") or audio.get("audio_url") or "",
                    "wave": "",
                    "status": 1,
                    "create_time": int(time.time()),
                    "env": "acejam",
                    "prompt": params.get("caption") or "",
                    "lyrics": params.get("lyrics") or "",
                    "metas": {
                        "bpm": params.get("bpm"),
                        "duration": params.get("duration"),
                        "genres": params.get("caption") or "",
                        "keyscale": params.get("key_scale") or "",
                        "timesignature": params.get("time_signature") or "",
                    },
                    "generation_info": f"AceJAM {result.get('runner', 'fast')} generation",
                    "seed_value": audio.get("seed") or "",
                    "lm_model": params.get("ace_lm_model") or "",
                    "dit_model": result.get("active_song_model") or params.get("song_model") or "",
                    "result_id": audio.get("result_id") or result.get("result_id") or "",
                    "audio_id": audio.get("id") or "",
                }
            )
        return {"task_id": task_id, "status": 1, "result": json.dumps(result_payload), "error": None}
    if status_code == 2:
        return {"task_id": task_id, "status": 2, "result": "", "error": task.get("error") or "Task failed"}
    return {"task_id": task_id, "status": 0, "result": "", "error": None}


def _task_ids_from_payload(body: dict[str, Any]) -> list[str]:
    value = body.get("task_id_list") or body.get("task_ids") or body.get("task_id")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return []


def _runtime_status(request: Request | None = None) -> dict[str, Any]:
    installed_models = _installed_acestep_models()
    installed_lms = _installed_lm_models()
    active_downloads = {
        name: job
        for name, job in _model_download_jobs.items()
        if job.get("state") in {"queued", "running"}
    }
    base_url = str(request.base_url).rstrip("/") if request is not None else ""
    return {
        "success": True,
        "app_version": APP_UI_VERSION,
        "ui_hash": _app_ui_hash(),
        "backend_hash": _backend_code_hash(),
        "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
        "server_url": base_url,
        "entrypoint": "http",
        "file_url_supported": False,
        "message": "Open AceJAM through the Pinokio Web UI HTTP URL, not file://app/index.html.",
        "active_song_model": ACTIVE_ACE_STEP_MODEL,
        "installed_song_model_count": len(installed_models),
        "installed_lm_model_count": len([name for name in installed_lms if name not in {"auto", "none"}]),
        "installed_song_models": sorted(installed_models),
        "installed_lm_models": sorted(installed_lms),
        "active_downloads": active_downloads,
        "active_training_job": training_manager.active_job(),
        "active_album_jobs": [
            job for job in _album_job_snapshot() if isinstance(job, dict) and job.get("state") in {"queued", "running"}
        ],
        "ollama": json.loads(ollama_models()),
        "default_album_planner_ollama_model": DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
        "default_album_embedding_model": DEFAULT_ALBUM_EMBEDDING_MODEL,
        "official_runner": _official_runner_status(),
        "pro_quality_policy": pro_quality_policy(),
        "ace_lm": _ace_lm_status_payload(),
        "trainer": training_manager.status(),
    }


def _album_job_snapshot(job_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    with _album_jobs_lock:
        if job_id:
            job = _album_jobs.get(job_id)
            return _jsonable(dict(job)) if job else {}
        return [_jsonable(dict(job)) for job in _album_jobs.values()]


def _set_album_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with _album_jobs_lock:
        job = _album_jobs.setdefault(
            job_id,
            {
                "id": job_id,
                "state": "queued",
                "status": "Queued",
                "progress": 0,
                "logs": [],
                "errors": [],
                "result": None,
                "payload": {},
                "planner_model": "",
                "embedding_model": "",
                "memory_enabled": False,
                "current_model_album": "",
                "current_track": "",
                "generated_count": 0,
                "expected_count": 0,
                "album_id": "",
                "album_family_id": "",
                "download_url": "",
                "started_at": None,
                "finished_at": None,
            },
        )
        if "logs" in updates:
            old_logs = list(job.get("logs") or [])
            new_logs = updates.pop("logs")
            if isinstance(new_logs, list):
                job["logs"] = (old_logs + [str(item) for item in new_logs])[-500:]
            elif new_logs:
                job["logs"] = (old_logs + [str(new_logs)])[-500:]
        if "errors" in updates:
            old_errors = list(job.get("errors") or [])
            new_errors = updates.pop("errors")
            if isinstance(new_errors, list):
                job["errors"] = (old_errors + [str(item) for item in new_errors])[-100:]
            elif new_errors:
                job["errors"] = (old_errors + [str(new_errors)])[-100:]
        job.update(_jsonable(updates))
        if len(_album_jobs) > ALBUM_JOB_KEEP_LIMIT:
            removable = sorted(
                _album_jobs.values(),
                key=lambda item: str(item.get("finished_at") or item.get("started_at") or ""),
            )
            for old in removable[: max(0, len(_album_jobs) - ALBUM_JOB_KEEP_LIMIT)]:
                if old.get("state") not in {"queued", "running"}:
                    _album_jobs.pop(str(old.get("id")), None)
        return dict(job)


def _album_job_log(job_id: str, line: str, **updates: Any) -> None:
    if not job_id:
        return
    _set_album_job(job_id, logs=[line], **updates)


def _album_job_worker(job_id: str, body: dict[str, Any]) -> None:
    planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
    embedding_provider = normalize_provider(body.get("embedding_lm_provider") or body.get("embedding_provider") or planner_provider)
    planner_model = str(body.get("planner_model") or body.get("ollama_model") or body.get("planner_ollama_model") or (DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL if planner_provider == "ollama" else ""))
    embedding_model = str(body.get("embedding_model") or DEFAULT_ALBUM_EMBEDDING_MODEL)
    started = datetime.now(timezone.utc).isoformat()
    _set_album_job(
        job_id,
        state="running",
        status="Running album production team",
        progress=1,
        started_at=started,
        finished_at=None,
        payload=body,
        planner_model=planner_model,
        planner_provider=planner_provider,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        memory_enabled=True,
        logs=[
            f"Album job {job_id} started.",
            f"Planner: {provider_label(planner_provider)} ({planner_model})",
            f"Embedding: {provider_label(embedding_provider)} ({embedding_model})",
            "CrewAI memory: enabled, local provider only.",
        ],
    )
    try:
        track_duration = parse_duration_seconds(body.get("track_duration") or body.get("duration") or 180.0, 180.0)
        num_tracks = clamp_int(body.get("num_tracks"), 7, 1, 40)
        strategy = str(body.get("song_model_strategy") or "all_models_album")
        expected_models = album_models_for_strategy(
            strategy,
            _installed_acestep_models(),
            str(body.get("requested_song_model") or body.get("song_model") or ""),
        )
        expected_count = len(expected_models) * num_tracks
        _set_album_job(job_id, expected_count=expected_count, status="Planning album", progress=2)
        request_body = dict(body)
        request_body["album_job_id"] = job_id
        request_body["planner_lm_provider"] = planner_provider
        request_body["embedding_lm_provider"] = embedding_provider
        request_body["planner_model"] = planner_model
        request_body["ollama_model"] = planner_model if planner_provider == "ollama" else body.get("ollama_model", "")
        request_body["embedding_model"] = embedding_model
        request_body["track_duration"] = track_duration
        raw = generate_album(
            concept=str(request_body.get("concept") or ""),
            num_tracks=num_tracks,
            track_duration=track_duration,
            ollama_model=planner_model,
            language=str(request_body.get("language") or "en"),
            song_model=str(request_body.get("song_model") or "auto"),
            embedding_model=embedding_model,
            ace_lm_model=str(request_body.get("ace_lm_model") or ACE_LM_PREFERRED_MODEL),
            request_json=json.dumps(request_body),
            planner_lm_provider=planner_provider,
            embedding_lm_provider=embedding_provider,
        )
        result = json.loads(raw or "{}")
        planner_model = str(result.get("planner_model") or planner_model)
        embedding_model = str(result.get("embedding_model") or embedding_model)
        generated_count = int(result.get("generated_count") or len(result.get("audios") or []))
        album_family_id = str(result.get("album_family_id") or "")
        download_url = str(result.get("family_download_url") or (f"/api/album-families/{album_family_id}/download" if album_family_id else ""))
        state = "succeeded" if result.get("success") else "failed"
        _set_album_job(
            job_id,
            state=state,
            status="Album completed" if state == "succeeded" else "Album failed or incomplete",
            progress=100 if state == "succeeded" else max(5, min(99, int(generated_count / max(1, expected_count) * 100))),
            result=result,
            logs=result.get("logs") or [],
            errors=[] if state == "succeeded" else [result.get("error") or "Album incomplete"],
            generated_count=generated_count,
            expected_count=int(result.get("expected_renders") or expected_count),
            planner_model=planner_model,
            embedding_model=embedding_model,
            album_id=str(result.get("album_id") or ""),
            album_family_id=album_family_id,
            download_url=download_url,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _set_album_job(
            job_id,
            state="failed",
            status="Album job failed",
            progress=100,
            errors=[str(exc)],
            logs=[traceback.format_exc()],
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        _cleanup_accelerator_memory()


def _run_album_plan_from_payload(body: dict[str, Any], log_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    concept = str(body.get("concept") or "")
    num_tracks = int(body.get("num_tracks") or 5)
    track_duration = parse_duration_seconds(body.get("track_duration") or body.get("duration") or 180.0, 180.0)
    language = str(body.get("language") or "en")
    song_model = str(body.get("song_model") or "auto")
    planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
    embedding_provider = normalize_provider(body.get("embedding_lm_provider") or body.get("embedding_provider") or planner_provider)
    planner_model = _resolve_local_llm_model_selection(
        planner_provider,
        str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or ""),
        "chat",
        "album planning",
    )
    embedding_model = _resolve_local_llm_model_selection(
        embedding_provider,
        str(body.get("embedding_model") or ""),
        "embedding",
        "album embeddings",
    )
    options = _album_options_from_payload({**body, "ollama_model": planner_model, "planner_model": planner_model}, song_model=song_model)
    if not concept and isinstance(options.get("user_album_contract"), dict):
        concept = str(options["user_album_contract"].get("concept") or "")
    from album_crew import plan_album as _plan_album

    result = _plan_album(
        concept=concept,
        num_tracks=num_tracks,
        track_duration=track_duration,
        ollama_model=planner_model,
        language=language,
        embedding_model=embedding_model,
        options=options,
        use_crewai=not parse_bool(body.get("toolbelt_only"), False),
        input_tracks=_json_list(body.get("tracks")) or None,
        planner_provider=planner_provider,
        embedding_provider=embedding_provider,
        log_callback=log_callback,
        crewai_output_log_file=str(body.get("crewai_output_log_file") or ""),
    )
    result["planner_model"] = planner_model
    result["planner_provider"] = planner_provider
    result["embedding_model"] = str(((result.get("toolkit_report") or {}).get("memory") or {}).get("embedding_model") or embedding_model) if isinstance(result.get("toolkit_report"), dict) else embedding_model
    result["embedding_provider"] = embedding_provider
    return result


def _album_plan_job_worker(job_id: str, body: dict[str, Any]) -> None:
    planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
    embedding_provider = normalize_provider(body.get("embedding_lm_provider") or body.get("embedding_provider") or planner_provider)
    planner_model = str(body.get("planner_model") or body.get("ollama_model") or body.get("planner_ollama_model") or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL)
    embedding_model = str(body.get("embedding_model") or DEFAULT_ALBUM_EMBEDDING_MODEL)
    toolbelt_only = parse_bool(body.get("toolbelt_only"), False)
    requested_engine = "deterministic toolbelt" if toolbelt_only else "CrewAI"
    crewai_output_log_file = str(body.get("crewai_output_log_file") or "")
    user_album_contract = extract_user_album_contract(
        "\n\n".join(
            part for part in [
                str(body.get("user_prompt") or body.get("prompt") or ""),
                str(body.get("album_title") or body.get("album_name") or ""),
                str(body.get("concept") or ""),
                json.dumps(body.get("tracks") or [], ensure_ascii=True, default=str)[:6000],
            ]
            if part
        ),
        int(body.get("num_tracks") or 0) or None,
        str(body.get("language") or "en"),
        body,
    )
    if not toolbelt_only and not crewai_output_log_file:
        from album_crew import crewai_output_log_path as _crewai_output_log_path

        crewai_output_log_file = str(_crewai_output_log_path(job_id))
        body = {**body, "crewai_output_log_file": crewai_output_log_file}
    start_logs = [
        f"Album plan job {job_id} started.",
        f"Planning engine requested: {requested_engine}.",
        f"Prompt Kit: {PROMPT_KIT_VERSION}.",
        f"Planner: {provider_label(planner_provider)} ({planner_model})",
        f"Embedding: {provider_label(embedding_provider)} ({embedding_model})",
    ]
    if user_album_contract.get("applied"):
        start_logs.append(
            "Input Contract: applied; "
            f"locked_tracks={len(user_album_contract.get('tracks') or [])}; "
            f"blocked_unsafe={int(user_album_contract.get('blocked_unsafe_count') or 0)}"
        )
    if crewai_output_log_file:
        start_logs.append(f"CrewAI output log file: {crewai_output_log_file}")
    _set_album_job(
        job_id,
        state="running",
        job_type="album_plan",
        status="Planning album",
        progress=5,
        payload=body,
        planner_model=planner_model,
        planner_provider=planner_provider,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        crewai_output_log_file=crewai_output_log_file,
        prompt_kit_version=PROMPT_KIT_VERSION,
        input_contract=contract_prompt_context(user_album_contract),
        input_contract_applied=bool(user_album_contract.get("applied")),
        input_contract_version=USER_ALBUM_CONTRACT_VERSION,
        blocked_unsafe_count=int(user_album_contract.get("blocked_unsafe_count") or 0),
        memory_enabled=not toolbelt_only,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        logs=start_logs,
    )
    try:
        def _stream_plan_log(line: str) -> None:
            compact = str(line).replace("\n", " ")[:700]
            updates: dict[str, Any] = {}
            lower = compact.lower()
            if "planning compact album bible with crewai" in lower:
                updates.update(status="CrewAI album bible running", progress=15)
            elif "crewai compact bible planned" in lower:
                updates.update(status="CrewAI track blueprints ready", progress=45)
            elif "producing track" in lower:
                updates.update(status="CrewAI track production running", progress=55)
            elif "crewai produced" in lower:
                updates.update(status="CrewAI plan ready", progress=95)
            elif "fell back to deterministic toolbelt" in lower or "toolbelt fallback planned" in lower:
                updates.update(status="Deterministic toolbelt fallback", progress=90)
            print(f"[album_plan_job][{job_id}] {compact}", file=sys.__stdout__, flush=True)
            _album_job_log(job_id, compact, **updates)

        result = _run_album_plan_from_payload(body, log_callback=_stream_plan_log)
        tracks = result.get("tracks") or []
        success = bool(result.get("success", True)) and bool(tracks)
        _set_album_job(
            job_id,
            state="succeeded" if success else "failed",
            status="Album plan ready" if success else "Album plan failed",
            progress=100,
            result=result,
            logs=result.get("logs") or [],
            errors=[] if success else [result.get("error") or "Album planning failed"],
            planner_model=str(result.get("planner_model") or planner_model),
            embedding_model=str(result.get("embedding_model") or embedding_model),
            planning_engine=str(result.get("planning_engine") or requested_engine.lower()),
            crewai_used=bool(result.get("crewai_used")),
            toolbelt_fallback=bool(result.get("toolbelt_fallback")),
            crewai_output_log_file=str(result.get("crewai_output_log_file") or crewai_output_log_file),
            prompt_kit_version=str(result.get("prompt_kit_version") or PROMPT_KIT_VERSION),
            input_contract=result.get("input_contract") or contract_prompt_context(user_album_contract),
            input_contract_applied=bool(result.get("input_contract_applied") or user_album_contract.get("applied")),
            input_contract_version=str(result.get("input_contract_version") or USER_ALBUM_CONTRACT_VERSION),
            blocked_unsafe_count=int(result.get("blocked_unsafe_count") or user_album_contract.get("blocked_unsafe_count") or 0),
            contract_repair_count=int(result.get("contract_repair_count") or 0),
            expected_count=len(tracks),
            planned_count=len(tracks),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _set_album_job(
            job_id,
            state="failed",
            job_type="album_plan",
            status="Album plan failed",
            progress=100,
            errors=[str(exc)],
            logs=[traceback.format_exc()],
            crewai_output_log_file=crewai_output_log_file,
            prompt_kit_version=PROMPT_KIT_VERSION,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        _cleanup_accelerator_memory()


@app.post("/api/config")
async def api_config():
    return JSONResponse(json.loads(config()))


@app.get("/api/config")
async def api_config_get():
    return JSONResponse(json.loads(config()))


@app.get("/api/status")
async def api_status(request: Request):
    return JSONResponse(_runtime_status(request))


@app.get("/api/runtime/progress")
async def api_runtime_progress():
    return JSONResponse(_runtime_progress_snapshot())


@app.get("/api/ace-step/parity")
async def api_ace_step_parity(request: Request):
    return JSONResponse(_official_parity_payload(request))


@app.get("/api/ace-lm/status")
async def api_ace_lm_status():
    return JSONResponse(_ace_lm_status_payload())


@app.post("/api/ace-lm/cleanup-preview")
async def api_ace_lm_cleanup_preview():
    return JSONResponse(_ace_lm_cleanup_preview())


@app.post("/api/ace-lm/cleanup-originals")
async def api_ace_lm_cleanup_originals(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(_ace_lm_cleanup_originals(str((body or {}).get("confirm") or "")))


@app.post("/api/ace-lm/upload-private")
async def api_ace_lm_upload_private(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(_ace_lm_private_upload(body if isinstance(body, dict) else {}))


@app.post("/api/ace-lm/mark-smoke-passed")
async def api_ace_lm_mark_smoke_passed(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(_ace_lm_mark_smoke_passed(body if isinstance(body, dict) else {}))


@app.post("/api/payload/validate")
async def api_payload_validate(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return JSONResponse(_validate_generation_payload(payload if isinstance(payload, dict) else {}))


@app.get("/health")
async def health(request: Request):
    return JSONResponse(_official_api_response({"status": "ok", **_runtime_status(request)}))


@app.get("/v1/models")
async def v1_models(request: Request):
    await _require_official_api_key(request)
    models = [
        {
            "name": name,
            "is_default": name == _default_acestep_checkpoint(),
            "is_loaded": name == ACTIVE_ACE_STEP_MODEL,
            "tasks": supported_tasks_for_model(name),
            **_model_runtime_status(name),
        }
        for name in _available_acestep_models()
    ]
    return JSONResponse(_official_api_response({"models": models, "default_model": _default_acestep_checkpoint()}))


@app.post("/v1/init")
async def v1_init(request: Request):
    await _require_official_api_key(request)
    try:
        body = await _request_payload(request)
        slot = clamp_int(body.get("slot"), 1, 1, 3)
        slot_env = "" if slot == 1 else os.environ.get(f"ACESTEP_CONFIG_PATH{slot}", "").strip()
        if slot != 1 and not slot_env:
            return JSONResponse(
                _official_api_response(None, error=f"Slot {slot} requires ACESTEP_CONFIG_PATH{slot} before startup.", code=400),
                status_code=400,
            )
        model_name = _normalize_song_model(str(body.get("model") or body.get("song_model") or _default_acestep_checkpoint()))
        if model_name in OFFICIAL_UNRELEASED_MODELS:
            return JSONResponse(_official_api_response(None, error=f"{model_name} is unreleased.", code=400), status_code=400)
        if model_name not in _installed_acestep_models():
            if model_name in _downloadable_model_names():
                job = _start_model_download(model_name)
                return JSONResponse(_official_api_response({"download_started": True, "download_model": model_name, "download_job": job}))
            return JSONResponse(_official_api_response(None, error=f"{model_name} is not installed.", code=400), status_code=400)
        with handler_lock:
            loaded_model = _ensure_song_model(model_name)
        loaded_lm = None
        if parse_bool(body.get("init_llm"), False):
            loaded_lm = _concrete_lm_model_or_download(str(body.get("lm_model_path") or body.get("ace_lm_model") or "auto"), "v1 init")
        return JSONResponse(
            _official_api_response(
                {
                    "message": "Model initialization completed",
                    "slot": slot,
                    "loaded_model": loaded_model,
                    "loaded_lm_model": loaded_lm,
                    "llm_initialized": bool(loaded_lm),
                    "llm_lazy": bool(loaded_lm),
                    "models": [
                        {"name": item, "is_default": item == _default_acestep_checkpoint(), "is_loaded": item == ACTIVE_ACE_STEP_MODEL}
                        for item in _available_acestep_models()
                    ],
                    "lm_models": sorted(_installed_lm_models()),
                }
            )
        )
    except ModelDownloadStarted as exc:
        return JSONResponse(_official_api_response(_download_started_payload(exc.model_name, exc.job)))
    except Exception as exc:
        return JSONResponse(_official_api_response(None, error=str(exc), code=400), status_code=400)


@app.get("/v1/stats")
async def v1_stats(request: Request):
    await _require_official_api_key(request)
    return JSONResponse(_official_api_response(_job_stats()))


@app.get("/v1/audio")
async def v1_audio(request: Request):
    await _require_official_api_key(request)
    raw_path = str(request.query_params.get("path") or "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="path is required")
    if raw_path.startswith("/media/results/"):
        _, _, result_id, filename = raw_path.split("/", 4)[1:]
        return FileResponse(_resolve_child(RESULTS_DIR, safe_id(result_id), filename))
    if raw_path.startswith("/media/songs/"):
        _, _, song_id, filename = raw_path.split("/", 4)[1:]
        return FileResponse(_resolve_child(SONGS_DIR, safe_id(song_id), filename))
    target = Path(raw_path).expanduser().resolve()
    data_root = DATA_DIR.resolve()
    if data_root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/api/songwriting_toolkit")
async def api_songwriting_toolkit():
    return JSONResponse(_songwriting_toolkit_payload())


@app.get("/api/models/downloads")
async def api_model_downloads():
    return JSONResponse({name: _model_download_job(name) for name in sorted(_downloadable_model_names())})


@app.get("/api/models/download/{model_name}")
async def api_model_download_status(model_name: str):
    if model_name not in _downloadable_model_names():
        return JSONResponse({"success": False, "error": f"{model_name} is not downloadable"}, status_code=404)
    return JSONResponse({"success": True, "job": _model_download_job(model_name), "installed": _is_model_installed(model_name)})


@app.post("/api/models/download")
async def api_model_download(request: Request):
    try:
        body = await request.json()
        model_name = str(body.get("model_name") or body.get("model") or "").strip()
        job = _start_model_download(model_name)
        return JSONResponse({"success": True, "job": job, "installed": _is_model_installed(model_name)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/album/plan")
async def api_album_plan(request: Request):
    try:
        body = await request.json()
        result = _run_album_plan_from_payload(body)
        return JSONResponse(result, status_code=200 if result.get("success", True) else 400)
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "album planning"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "logs": [str(exc)]}, status_code=400)


@app.post("/api/album/plan/jobs")
async def api_create_album_plan_job(request: Request):
    try:
        body = await request.json()
        planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
        embedding_provider = normalize_provider(body.get("embedding_lm_provider") or body.get("embedding_provider") or planner_provider)
        planner_model = _resolve_local_llm_model_selection(
            planner_provider,
            str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or ""),
            "chat",
            "album planning",
        )
        embedding_model = _resolve_local_llm_model_selection(
            embedding_provider,
            str(body.get("embedding_model") or ""),
            "embedding",
            "album embeddings",
        )
        job_id = uuid.uuid4().hex[:12]
        request_body = {
            **body,
            "planner_lm_provider": planner_provider,
            "embedding_lm_provider": embedding_provider,
            "planner_model": planner_model,
            "ollama_model": planner_model if planner_provider == "ollama" else body.get("ollama_model", ""),
            "embedding_model": embedding_model,
        }
        _set_album_job(
            job_id,
            state="queued",
            job_type="album_plan",
            status="Queued album planning job",
            progress=0,
            payload=request_body,
            planner_model=planner_model,
            planner_provider=planner_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            memory_enabled=not parse_bool(body.get("toolbelt_only"), False),
            expected_count=int(body.get("num_tracks") or 5),
            logs=[f"Queued album plan job {job_id}."],
        )
        threading.Thread(target=_album_plan_job_worker, args=(job_id, request_body), daemon=True).start()
        return JSONResponse({"success": True, "job_id": job_id, "job": _album_job_snapshot(job_id)})
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "album planning"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "logs": [str(exc)]}, status_code=400)


@app.get("/api/community")
async def api_community():
    return JSONResponse(_feed_songs[:100])


@app.get("/api/ollama_models")
async def api_ollama_models():
    return JSONResponse(json.loads(ollama_models()))


@app.get("/api/ollama/status")
async def api_ollama_status():
    data = _ollama_model_catalog()
    return JSONResponse(
        {
            "success": data.get("success", False),
            "ready": data.get("ready", False),
            "ollama_host": data.get("ollama_host", _ollama_host()),
            "model_count": len(data.get("models") or []),
            "chat_model_count": len(data.get("chat_models") or []),
            "embedding_model_count": len(data.get("embedding_models") or []),
            "running_models": data.get("running_models") or [],
            "pull_jobs": data.get("pull_jobs") or [],
            "error": data.get("error") or "",
        }
    )


@app.get("/api/ollama/models")
async def api_ollama_models_rich():
    return JSONResponse(_ollama_model_catalog())


@app.get("/api/local-llm/providers")
async def api_local_llm_providers():
    return JSONResponse(
        {
            "success": True,
            "default_provider": "ollama",
            "providers": [
                {"id": "ollama", "label": "Ollama", "host": _ollama_host(), "ready": _ollama_model_catalog().get("ready", False)},
                {"id": "lmstudio", "label": "LM Studio", "host": lmstudio_model_catalog().get("host", ""), "ready": lmstudio_model_catalog().get("ready", False)},
            ],
        }
    )


@app.get("/api/local-llm/models")
async def api_local_llm_models(provider: str = "ollama"):
    provider_name = normalize_provider(provider)
    if provider_name == "ollama":
        data = _ollama_model_catalog()
        data["provider"] = "ollama"
        data["provider_label"] = "Ollama"
        data["host"] = data.get("ollama_host") or _ollama_host()
        return JSONResponse(data)
    return JSONResponse(lmstudio_model_catalog())


@app.post("/api/local-llm/test")
async def api_local_llm_test(request: Request):
    try:
        body = await request.json()
        provider = normalize_provider(body.get("provider") or body.get("planner_lm_provider") or "ollama")
        model = str(body.get("model") or body.get("model_name") or body.get("planner_model") or body.get("ollama_model") or "").strip()
        kind = str(body.get("kind") or "chat").strip().lower()
        if provider == "ollama":
            _ensure_ollama_model_or_start_pull(model, context=f"{kind} test", kind="embedding" if kind == "embedding" else "chat")
        return JSONResponse(local_llm_test_model(provider, model, kind))
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "local LLM test"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/local-llm/load")
async def api_local_llm_load(request: Request):
    try:
        body = await request.json()
        provider = normalize_provider(body.get("provider") or "lmstudio")
        model = str(body.get("model") or body.get("model_name") or "").strip()
        kind = str(body.get("kind") or "chat").strip().lower()
        if provider != "lmstudio":
            return JSONResponse({"success": True, "provider": provider, "model": model, "message": "Ollama loads models on demand."})
        context_length = body.get("context_length")
        return JSONResponse(lmstudio_load_model(model, kind=kind, context_length=int(context_length) if context_length else None))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/local-llm/unload")
async def api_local_llm_unload(request: Request):
    try:
        body = await request.json()
        provider = normalize_provider(body.get("provider") or "lmstudio")
        model = str(body.get("model") or body.get("model_name") or "").strip()
        if provider != "lmstudio":
            return JSONResponse({"success": True, "provider": provider, "model": model, "message": "Ollama unload is managed by Ollama."})
        return JSONResponse(lmstudio_unload_model(model))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/local-llm/download")
async def api_local_llm_download(request: Request):
    try:
        body = await request.json()
        provider = normalize_provider(body.get("provider") or "ollama")
        model_name = str(body.get("model") or body.get("model_name") or "").strip()
        kind = str(body.get("kind") or ("embedding" if _is_embedding_model_name(model_name) else "chat")).strip().lower()
        if provider == "ollama":
            job = _start_ollama_pull(model_name, reason=str(body.get("reason") or "manual"), kind=kind)
            return JSONResponse({"success": True, "provider": "ollama", "job": job})
        return JSONResponse(lmstudio_download_model(model_name, str(body.get("quantization") or ""), kind=kind))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/local-llm/download/{job_id}")
async def api_local_llm_download_status(job_id: str, provider: str = "lmstudio"):
    try:
        provider_name = normalize_provider(provider)
        if provider_name == "ollama":
            job = _ollama_pull_job(job_id)
            if not job:
                return JSONResponse({"success": False, "error": "Ollama pull job not found."}, status_code=404)
            return JSONResponse({"success": True, "provider": "ollama", "job": job})
        return JSONResponse(lmstudio_download_status(job_id))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/lmstudio/models")
async def api_lmstudio_models():
    return JSONResponse(lmstudio_model_catalog())


@app.post("/api/ollama/pull")
async def api_ollama_pull(request: Request):
    try:
        body = await request.json()
        model_name = str(body.get("model") or body.get("model_name") or "").strip()
        kind = str(body.get("kind") or ("embedding" if _is_embedding_model_name(model_name) else "chat"))
        job = _start_ollama_pull(model_name, reason=str(body.get("reason") or "manual"), kind=kind)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/ollama/pull/{job_id}")
async def api_ollama_pull_status(job_id: str):
    job = _ollama_pull_job(job_id)
    if not job:
        return JSONResponse({"success": False, "error": "Ollama pull job not found"}, status_code=404)
    return JSONResponse({"success": True, "job": job})


@app.post("/api/ollama/show")
async def api_ollama_show(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
        model_name = str(body.get("model") or body.get("model_name") or "").strip()
        _ensure_ollama_model_or_start_pull(model_name, context="model inspect", kind="embedding" if _is_embedding_model_name(model_name) else "chat")
        data = _ollama_client().show(model_name)
        return JSONResponse({"success": True, "model": model_name, "details": _jsonable(data)})
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "model inspect"))
    except Exception as exc:
        if _ollama_error_is_missing_model(exc):
            model_name = str(body.get("model") or body.get("model_name") or "").strip()
            job = _start_ollama_pull(model_name, reason="model inspect", kind="embedding" if _is_embedding_model_name(model_name) else "chat")
            return JSONResponse(_ollama_pull_started_payload(model_name, job, "model inspect"))
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/ollama/test")
async def api_ollama_test(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
        model_name = str(body.get("model") or body.get("model_name") or "").strip()
        kind = str(body.get("kind") or ("embedding" if _is_embedding_model_name(model_name) else "chat"))
        _ensure_ollama_model_or_start_pull(model_name, context=f"{kind} test", kind=kind)
        client = _ollama_client()
        if kind == "embedding":
            response = client.embed(model=model_name, input="AceJAM embedding test")
            embeddings = _ollama_attr(response, "embeddings", [])
            first = embeddings[0] if embeddings else []
            return JSONResponse(
                {
                    "success": True,
                    "model": model_name,
                    "kind": "embedding",
                    "embedding_count": len(embeddings),
                    "dimensions": len(first) if hasattr(first, "__len__") else 0,
                }
            )
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            think=False,
            options={"temperature": 0.1, "top_p": 0.8, "top_k": 20},
        )
        return JSONResponse({"success": True, "model": model_name, "kind": "chat", "response": str(response.message.content or "").strip()})
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "Ollama test"))
    except Exception as exc:
        model_name = str(body.get("model") or body.get("model_name") or "").strip()
        if model_name and _ollama_error_is_missing_model(exc):
            kind = str(body.get("kind") or ("embedding" if _is_embedding_model_name(model_name) else "chat"))
            job = _start_ollama_pull(model_name, reason=f"{kind} test", kind=kind)
            return JSONResponse(_ollama_pull_started_payload(model_name, job, "Ollama test"))
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/prompt-assistant/prompts")
async def api_prompt_assistant_prompts():
    prompts = []
    for mode, info in PROMPT_ASSISTANT_MODES.items():
        path = BASE_DIR.parent / info["file"]
        prompts.append(
            {
                "mode": mode,
                "label": info["label"],
                "description": info["description"],
                "file": info["file"],
                "available": path.is_file(),
            }
        )
    return JSONResponse({"success": True, "prompts": prompts})


@app.post("/api/prompt-assistant/run")
async def api_prompt_assistant_run(request: Request):
    raw_text = ""
    try:
        body = await request.json()
        mode = _prompt_assistant_mode(str(body.get("mode") or "custom"))
        user_prompt = str(body.get("user_prompt") or body.get("prompt") or "").strip()
        if not user_prompt:
            return JSONResponse({"success": False, "error": "Prompt is empty.", "raw_text": ""}, status_code=400)
        current_payload = body.get("current_payload") if isinstance(body.get("current_payload"), dict) else {}
        planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
        planner_model = str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or "").strip()
        system_prompt = _prompt_assistant_system_prompt(mode)
        raw_text = _run_prompt_assistant_local(system_prompt, user_prompt, planner_provider, planner_model, current_payload)
        parsed_payload, paste_blocks = _extract_prompt_assistant_json(raw_text, mode)
        payload, warnings = _normalize_prompt_assistant_payload(mode, parsed_payload, body)
        validation = None
        if mode not in {"album", "trainer"}:
            try:
                validation = _validate_generation_payload(payload)
            except Exception as validation_exc:
                warnings.append(str(validation_exc))
        return JSONResponse(
            {
                "success": True,
                "mode": mode,
                "prompt_kit_version": PROMPT_KIT_VERSION,
                "prompt_file": PROMPT_ASSISTANT_MODES[mode]["file"],
                "payload": _jsonable(payload),
                "paste_blocks": paste_blocks,
                "warnings": warnings,
                "validation": _jsonable(validation),
                "raw_text": raw_text,
            }
        )
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "AI Fill", raw_text=raw_text, warnings=[]))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "raw_text": raw_text, "warnings": []}, status_code=400)


@app.post("/api/delete_song")
async def api_delete_song(request: Request):
    body = await request.json()
    return JSONResponse(json.loads(delete_song(str(body.get("song_id") or ""))))


@app.get("/api/outputs/status")
async def api_outputs_status():
    return JSONResponse({"success": True, "generated": _count_generated_outputs()})


@app.post("/api/outputs/delete_generated")
async def api_delete_generated_outputs(request: Request):
    body = await request.json()
    return JSONResponse(_delete_generated_outputs(str(body.get("confirm") or "")))


@app.post("/api/albums/{album_id}/delete")
async def api_delete_album(album_id: str, request: Request):
    body = await request.json()
    return JSONResponse(_delete_album(album_id, str(body.get("confirm") or "")))


@app.post("/api/album-families/{family_id}/delete")
async def api_delete_album_family(family_id: str, request: Request):
    body = await request.json()
    return JSONResponse(_delete_album_family(family_id, str(body.get("confirm") or "")))


@app.post("/api/compose")
async def api_compose(request: Request):
    try:
        body = await request.json()
        provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
        planner_model = str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or "").strip()
        raw = compose(
            description=str(body.get("description") or ""),
            audio_duration=float(body.get("audio_duration") or body.get("duration") or 60.0),
            composer_profile=str(body.get("composer_profile") or "auto"),
            instrumental=parse_bool(body.get("instrumental"), False),
            ollama_model=str(body.get("ollama_model") or ""),
            planner_lm_provider=provider,
            planner_model=planner_model,
        )
        return JSONResponse(json.loads(raw))
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "compose"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/create_sample")
async def api_create_sample(request: Request):
    try:
        body = _apply_studio_lm_policy(await request.json())
        use_official = parse_bool(body.get("use_official_lm"), _requested_ace_lm_model(body) != "none")
        if use_official:
            return JSONResponse(_run_official_lm_aux("create_sample", body))
        raw = compose(
            description=str(body.get("query") or body.get("description") or body.get("caption") or ""),
            audio_duration=float(body.get("duration") or 60.0),
            composer_profile="auto",
            instrumental=parse_bool(body.get("instrumental"), False),
            ollama_model=str(body.get("ollama_model") or ""),
            planner_lm_provider=str(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama"),
            planner_model=str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or ""),
        )
        data = json.loads(raw)
        data["artist_name"] = normalize_artist_name(
            body.get("artist_name") or data.get("artist_name"),
            derive_artist_name(data.get("title") or "", body.get("description") or body.get("caption") or "", data.get("tags") or ""),
        )
        return JSONResponse({"success": True, "engine": normalize_provider(body.get("planner_lm_provider") or "ollama"), **data})
    except ModelDownloadStarted as exc:
        return JSONResponse(_download_started_payload(exc.model_name, exc.job))
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "create sample"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/format_sample")
async def api_format_sample(request: Request):
    try:
        body = _apply_studio_lm_policy(await request.json())
        use_official = parse_bool(body.get("use_official_lm"), _requested_ace_lm_model(body) != "none")
        if use_official:
            return JSONResponse(_run_official_lm_aux("format_sample", body))
        raw = compose(
            description=str(body.get("caption") or body.get("description") or "custom song"),
            audio_duration=float(body.get("duration") or 60.0),
            composer_profile="auto",
            instrumental=parse_bool(body.get("instrumental"), False),
            ollama_model=str(body.get("ollama_model") or ""),
            planner_lm_provider=str(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama"),
            planner_model=str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or ""),
        )
        data = json.loads(raw)
        data["artist_name"] = normalize_artist_name(
            body.get("artist_name") or data.get("artist_name"),
            derive_artist_name(data.get("title") or "", body.get("description") or body.get("caption") or "", data.get("tags") or ""),
        )
        if str(body.get("lyrics") or "").strip():
            data["lyrics"] = str(body["lyrics"]).strip()
        return JSONResponse({"success": True, "engine": normalize_provider(body.get("planner_lm_provider") or "ollama"), **data})
    except ModelDownloadStarted as exc:
        return JSONResponse(_download_started_payload(exc.model_name, exc.job))
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "format sample"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/create_random_sample")
async def create_random_sample(request: Request):
    await _require_official_api_key(request)
    try:
        body = await _request_payload(request)
        sample_type = str(body.get("sample_type") or "simple_mode").strip().lower()
        if sample_type == "custom_mode":
            data = {
                "caption": "cinematic synth pop, warm analog bass, bright lead vocal, polished radio mix",
                "lyrics": "[Verse]\nI found a spark in the city lights\n[Chorus]\nWe rise, we shine, we carry the night",
                "bpm": 118,
                "key_scale": "C major",
                "time_signature": "4",
                "duration": 180,
                "vocal_language": "en",
                "sample_type": "custom_mode",
            }
        else:
            data = {
                "caption": "upbeat pop song with guitar accompaniment, memorable hook, clean vocal production",
                "lyrics": "[Verse 1]\nSunlight on my face, I keep moving\n[Chorus]\nThis is our moment, this is our sound",
                "bpm": 120,
                "key_scale": "G major",
                "time_signature": "4",
                "duration": 180,
                "vocal_language": "en",
                "sample_type": "simple_mode",
            }
        return JSONResponse(_official_api_response(data))
    except Exception as exc:
        return JSONResponse(_official_api_response(None, error=str(exc), code=400), status_code=400)


@app.post("/format_input")
async def format_input(request: Request):
    await _require_official_api_key(request)
    try:
        body = await _request_payload(request)
        body.setdefault("caption", body.get("prompt") or "")
        body.setdefault("ace_lm_model", body.get("lm_model") or "auto")
        body.setdefault("use_official_lm", True)
        data = _run_official_lm_aux("format_sample", body)
        return JSONResponse(_official_api_response(data))
    except ModelDownloadStarted as exc:
        return JSONResponse(_official_api_response(_download_started_payload(exc.model_name, exc.job)))
    except Exception as exc:
        return JSONResponse(_official_api_response(None, error=str(exc), code=400), status_code=400)


@app.post("/release_task")
async def release_task(request: Request):
    await _require_official_api_key(request)
    try:
        body = await _request_payload(request)
        if "prompt" in body and "caption" not in body:
            body["caption"] = body["prompt"]
        if "model" in body and "song_model" not in body:
            body["song_model"] = body["model"]
        task = _submit_api_generation_task(body)
        return JSONResponse(_official_api_response(task))
    except Exception as exc:
        return JSONResponse(_official_api_response(None, error=str(exc), code=400), status_code=400)


@app.post("/query_result")
async def query_result(request: Request):
    await _require_official_api_key(request)
    body = await _request_payload(request)
    task_ids = _task_ids_from_payload(body)
    return JSONResponse(_official_api_response([_official_query_item(task_id) for task_id in task_ids]))


@app.post("/api/generate_advanced")
async def api_generate_advanced(request: Request):
    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
        return JSONResponse(_run_advanced_generation(payload))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[api_generate_advanced ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return JSONResponse(
            {"success": False, "error": str(exc), "validation": _validate_generation_payload(payload)},
            status_code=400,
        )
    finally:
        _cleanup_accelerator_memory()


@app.post("/api/generate_portfolio")
async def api_generate_portfolio(request: Request):
    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
        return JSONResponse(_run_model_portfolio_generation(payload))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[api_generate_portfolio ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return JSONResponse(
            {"success": False, "error": str(exc), "validation": _validate_generation_payload(payload)},
            status_code=400,
        )
    finally:
        _cleanup_accelerator_memory()


@app.post("/api/uploads")
async def api_upload_audio(file: UploadFile = File(...)):
    original_name = file.filename or "audio.wav"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        return JSONResponse({"success": False, "error": f"Unsupported audio file: {suffix}"}, status_code=400)
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOADS_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_filename(original_name)}{suffix}"
    target = upload_dir / filename
    target.write_bytes(await file.read())
    return JSONResponse({"success": True, "id": upload_id, "filename": filename, "url": f"/api/uploads/{upload_id}"})


@app.get("/api/uploads/{upload_id}")
async def api_get_upload(upload_id: str):
    path = _resolve_upload_file(upload_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return FileResponse(path)


@app.get("/api/results/{result_id}")
async def api_get_result(result_id: str):
    return JSONResponse(_load_result_meta(result_id))


@app.post("/api/audio-codes")
async def api_audio_codes(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        audio_path = _resolve_upload_file(body.get("upload_id")) or _resolve_result_audio(body.get("result_id"), body.get("audio_id"))
        if audio_path is None:
            return JSONResponse({"success": False, "error": "No upload_id or result_id supplied"}, status_code=400)
        with handler_lock:
            _ensure_song_model(body.get("song_model"))
            codes = handler.convert_src_audio_to_codes(str(audio_path))
        return JSONResponse({"success": True, "audio_codes": codes})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


def _update_result_item(result_id: str, audio_id: str, field: str, value: Any) -> None:
    meta_path = _result_meta_path(result_id)
    meta = _load_result_meta(result_id)
    for item in meta.get("audios", []):
        if item.get("id") == audio_id:
            item[field] = _jsonable(value)
            break
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@app.post("/api/lrc")
async def api_lrc(request: Request):
    body = await request.json()
    result_id = safe_id(str(body.get("result_id") or ""))
    audio_id = str(body.get("audio_id") or "take-1")
    meta = _load_result_meta(result_id)
    extra = _result_extra_cache.get(result_id)
    if not extra:
        return JSONResponse({"success": False, "error": "LRC cache expired; regenerate with Auto LRC enabled"}, status_code=400)
    index = max(0, int(audio_id.split("-")[-1]) - 1) if "-" in audio_id else 0
    params = meta.get("params", {})
    seed = int((meta.get("audios", [{}])[index].get("seed") or 42))
    lrc = _calculate_lrc(_extra_for_index(extra, index), float(params.get("duration") or 60), str(params.get("vocal_language") or "unknown"), int(params.get("inference_steps") or 8), seed)
    _update_result_item(result_id, audio_id, "lrc", lrc)
    return JSONResponse(lrc)


@app.post("/api/score")
async def api_score(request: Request):
    body = await request.json()
    result_id = safe_id(str(body.get("result_id") or ""))
    audio_id = str(body.get("audio_id") or "take-1")
    meta = _load_result_meta(result_id)
    extra = _result_extra_cache.get(result_id)
    if not extra:
        return JSONResponse({"success": False, "error": "Score cache expired; regenerate with Auto Score enabled"}, status_code=400)
    index = max(0, int(audio_id.split("-")[-1]) - 1) if "-" in audio_id else 0
    params = meta.get("params", {})
    seed = int((meta.get("audios", [{}])[index].get("seed") or 42))
    score = _calculate_score(_extra_for_index(extra, index), str(params.get("vocal_language") or "unknown"), int(params.get("inference_steps") or 8), seed)
    _update_result_item(result_id, audio_id, "score", score)
    return JSONResponse(score)


@app.get("/api/lora/status")
async def api_lora_status():
    return JSONResponse(
        {
            "success": True,
            **handler.get_lora_status(),
            "trainer": training_manager.status(),
            "adapters": training_manager.list_adapters(),
            "pro_audition_policy": _lora_dataset_health([])["audition_plan"],
        }
    )


@app.post("/api/lora/load")
async def api_lora_load(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        with handler_lock:
            status_msg = handler.load_lora(str(body.get("path") or ""))
        return JSONResponse({"success": not status_msg.startswith("❌"), "status": status_msg, **handler.get_lora_status()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/unload")
async def api_lora_unload():
    try:
        _ensure_training_idle()
        with handler_lock:
            status_msg = handler.unload_lora()
        return JSONResponse({"success": not status_msg.startswith("❌"), "status": status_msg, **handler.get_lora_status()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/use")
async def api_lora_use(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        with handler_lock:
            status_msg = handler.set_use_lora(parse_bool(body.get("use"), True))
        return JSONResponse({"success": not status_msg.startswith("❌"), "status": status_msg, **handler.get_lora_status()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/scale")
async def api_lora_scale(request: Request):
    try:
        body = await request.json()
        scale = clamp_float(body.get("scale", body.get("lora_scale")), 1.0, 0.0, 1.0)
        with handler_lock:
            status_msg = handler.set_lora_scale(scale)
        return JSONResponse({"success": not status_msg.startswith("❌"), "status": status_msg, "scale": scale, **handler.get_lora_status()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/dataset/import-folder")
async def api_lora_dataset_import_folder(
    files: list[UploadFile] = File(...),
    dataset_id: str = Form(""),
    trigger_tag: str = Form(""),
    language: str = Form("unknown"),
):
    try:
        if not files:
            return JSONResponse({"success": False, "error": "No files were selected"}, status_code=400)
        import_id = safe_id(dataset_id or f"dataset-{uuid.uuid4().hex[:8]}")
        target_root = training_manager.import_root_for(import_id)
        if target_root.exists():
            shutil.rmtree(target_root)
        target_root.mkdir(parents=True, exist_ok=True)
        copied = []
        skipped = []
        sidecar_suffixes = {".txt", ".json", ".csv"}
        for upload in files:
            rel_path = _safe_lora_upload_relative_path(upload.filename or "")
            suffix = rel_path.suffix.lower()
            if suffix not in ALLOWED_AUDIO_EXTENSIONS and suffix not in sidecar_suffixes:
                skipped.append(str(rel_path))
                continue
            dest = target_root / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(await upload.read())
            copied.append(str(rel_path))
        if not copied:
            return JSONResponse({"success": False, "error": "No supported audio or sidecar files were imported"}, status_code=400)
        data = training_manager.scan_dataset(target_root)
        labels_preview = training_manager.label_entries(
            data.get("files") or [],
            trigger_tag=trigger_tag,
            language=language,
            tag_position="prepend",
        )
        return JSONResponse(
            {
                "success": True,
                "dataset_id": import_id,
                "import_root": str(target_root),
                "copied_files": copied,
                "skipped_files": skipped,
                "files": labels_preview,
                "dataset_health": _lora_dataset_health(labels_preview),
                "scan": data,
            }
        )
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/dataset/scan")
async def api_lora_dataset_scan(request: Request):
    try:
        body = await request.json()
        data = training_manager.scan_dataset(Path(str(body.get("path") or "")))
        data["dataset_health"] = _lora_dataset_health(data.get("files") or [])
        return JSONResponse({"success": True, **data})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


_MB_ARTIST_CACHE: dict[str, dict[str, Any]] = {}
_MB_RATE_LIMIT_LAST = [0.0]


def _musicbrainz_search(endpoint: str, query: str, limit: int = 1) -> dict[str, Any]:
    """Query MusicBrainz API with rate limiting (1 req/sec)."""
    import time
    import urllib.parse
    import urllib.request

    now = time.time()
    wait = max(0, 1.1 - (now - _MB_RATE_LIMIT_LAST[0]))
    if wait > 0:
        time.sleep(wait)
    _MB_RATE_LIMIT_LAST[0] = time.time()

    encoded = urllib.parse.quote(query)
    url = f"https://musicbrainz.org/ws/2/{endpoint}/?query={encoded}&fmt=json&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "AceJAM/1.0 (training-dataset-labeler)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def _musicbrainz_artist_tags(artist_name: str) -> list[str]:
    """Get genre tags for an artist from MusicBrainz. Cached."""
    key = artist_name.lower().strip()
    if key in _MB_ARTIST_CACHE:
        return _MB_ARTIST_CACHE[key].get("tags", [])
    data = _musicbrainz_search("artist", artist_name, limit=1)
    artists = data.get("artists", [])
    if not artists:
        _MB_ARTIST_CACHE[key] = {"tags": []}
        return []
    tags = [t["name"] for t in artists[0].get("tags", []) if t.get("count", 0) >= 0]
    # Filter to music genre tags (skip nationality/decade tags)
    skip = {"american", "british", "english", "german", "french", "dutch", "canadian", "australian", "swedish", "korean", "japanese"}
    genre_tags_raw = [t for t in tags if t.lower() not in skip and not re.match(r"^\d{4}s?$", t)]
    # Deduplicate similar tags (e.g., "hip-hop" and "hip hop")
    seen_normalized: set[str] = set()
    genre_tags: list[str] = []
    for t in genre_tags_raw:
        norm = t.lower().replace("-", " ").replace("_", " ").strip()
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            genre_tags.append(t)
        if len(genre_tags) >= 6:
            break
    _MB_ARTIST_CACHE[key] = {"tags": genre_tags, "all_tags": tags}
    return genre_tags


def _musicbrainz_recording_info(artist: str, title: str) -> dict[str, Any]:
    """Get recording info (duration, album) from MusicBrainz."""
    query = f'artist:"{artist}" AND recording:"{title}"'
    data = _musicbrainz_search("recording", query, limit=1)
    recordings = data.get("recordings", [])
    if not recordings:
        return {}
    rec = recordings[0]
    releases = rec.get("releases", [])
    return {
        "title": rec.get("title", ""),
        "duration_ms": rec.get("length"),
        "album": releases[0].get("title", "") if releases else "",
        "year": (releases[0].get("date", "")[:4] if releases and releases[0].get("date") else ""),
        "tags": [t["name"] for t in rec.get("tags", [])],
    }


def _parse_artist_title(filename: str) -> tuple[str, str]:
    """Extract artist and title from filename like 'Artist - Title.wav' or 'Artist - 02 - Title.wav'."""
    stem = Path(filename).stem
    # Remove leading track numbers: "01 - ", "02.", "Track 3 -"
    stem = re.sub(r"^\d{1,3}[\s.\-_]+", "", stem).strip()
    for sep in [" - ", " – ", " — ", " _ "]:
        if sep in stem:
            parts = stem.split(sep)
            parts = [p.strip() for p in parts if p.strip()]
            # Remove pure track-number parts ("01", "02", "03")
            parts = [p for p in parts if not re.match(r"^\d{1,3}$", p)]
            if len(parts) >= 2:
                return parts[0], parts[-1]  # first = artist, last = title
            if len(parts) == 1:
                return "", parts[0]
    return "", stem.replace("-", " ").replace("_", " ").strip()


def _search_lyrics_online(artist: str, title: str) -> str:
    """Fetch lyrics from Genius. Returns lyrics text or empty string."""
    import urllib.parse
    import urllib.request

    if not artist or not title:
        return ""

    def _genius_slug(artist_name: str, song_title: str) -> str:
        slug = f"{artist_name} {song_title}".lower()
        slug = re.sub(r"[^a-z0-9\s]", "", slug)
        slug = re.sub(r"\s+", "-", slug.strip())
        return slug

    def _fetch_genius_page(url: str) -> str:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def _extract_genius_lyrics(html: str) -> str:
        blocks = re.findall(r'data-lyrics-container="true"[^>]*>(.*?)</div>', html, re.DOTALL)
        if not blocks:
            return ""
        text = "\n".join(blocks)
        text = re.sub(r"<br\s*/?>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&#x27;", "'").replace("&quot;", '"')
        # Clean Genius header junk (contributors, translations line)
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Skip contributor/translation header lines
            if re.match(r"^\d+\s*Contributor", stripped):
                continue
            if stripped.startswith("Translations") or re.match(r"^[A-Z][a-zà-ü]+(,\s*[A-Z])", stripped):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    try:
        slug = _genius_slug(artist, title)
        url = f"https://genius.com/{slug}-lyrics"
        html = _fetch_genius_page(url)
        lyrics = _extract_genius_lyrics(html)
        if lyrics and len(lyrics) > 50:
            return lyrics
    except Exception:
        pass

    # Fallback: try with simplified title (remove parentheses, features)
    try:
        clean_title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip()
        if clean_title != title:
            slug = _genius_slug(artist, clean_title)
            url = f"https://genius.com/{slug}-lyrics"
            html = _fetch_genius_page(url)
            lyrics = _extract_genius_lyrics(html)
            if lyrics and len(lyrics) > 50:
                return lyrics
    except Exception:
        pass

    return ""


def _detect_bpm_key(audio_path: str) -> tuple[int | None, str]:
    """Detect BPM and musical key from audio file using scipy. No librosa needed."""
    import numpy as np

    try:
        data, sr = sf.read(str(audio_path), dtype="float32")
        if len(data.shape) > 1:
            data = data.mean(axis=1)  # mono
        # Limit to first 30 seconds for speed
        data = data[: sr * 30]

        # BPM detection via onset envelope autocorrelation
        from scipy import signal
        # Create onset strength envelope
        hop = 512
        frame_len = 2048
        # Simple spectral flux
        spec = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(data, frame_len)[::hop]))
        flux = np.sum(np.maximum(0, np.diff(spec, axis=0)), axis=1)
        if len(flux) < 4:
            return None, ""
        # Autocorrelation for tempo
        corr = np.correlate(flux - flux.mean(), flux - flux.mean(), mode="full")
        corr = corr[len(corr) // 2:]
        # Find peaks in valid BPM range (60-200 BPM)
        min_lag = int(60 * sr / hop / 200)  # 200 BPM
        max_lag = int(60 * sr / hop / 60)   # 60 BPM
        if max_lag > len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            return None, ""
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return None, ""
        peak_idx = np.argmax(search) + min_lag
        bpm = round(60 * sr / hop / peak_idx)

        # Key detection via chroma energy
        chroma_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        # Simple chroma via FFT
        fft_data = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
        chroma = np.zeros(12)
        for i, name in enumerate(chroma_names):
            # Sum energy around each pitch class (across octaves)
            for octave in range(1, 8):
                freq = 440 * 2 ** ((i - 9) / 12 + octave - 4)
                idx = np.argmin(np.abs(freqs - freq))
                if idx < len(fft_data):
                    chroma[i] += fft_data[max(0, idx - 2):idx + 3].sum()
        # Dominant pitch class
        root = chroma_names[np.argmax(chroma)]
        # Simple major/minor: check if minor third is stronger than major third
        root_idx = np.argmax(chroma)
        minor_third = chroma[(root_idx + 3) % 12]
        major_third = chroma[(root_idx + 4) % 12]
        scale = "minor" if minor_third > major_third else "major"
        key = f"{root} {scale}"

        return bpm, key
    except Exception:
        return None, ""


def _build_smart_caption(artist: str, title: str, bpm: int | None, key: str, has_vocals: bool, genre_tags: list[str] | None = None) -> str:
    """Build an ACE-Step caption from metadata. No LLM needed."""
    parts = []
    # Genre tags first (most important for ACE-Step conditioning)
    if genre_tags:
        parts.extend(genre_tags[:3])
    # BPM and key
    if bpm:
        parts.append(f"{bpm} BPM")
    if key:
        parts.append(key)
    # Vocal info
    if has_vocals:
        parts.append("vocals")
    else:
        parts.append("instrumental")
    return ", ".join(parts) if parts else "music track"


def _smart_autolabel_file(audio_path: Path, filename: str, trigger_tag: str = "", tag_position: str = "prepend") -> dict[str, Any]:
    """Auto-label a single audio file using MusicBrainz + online lyrics + audio analysis. No LLM needed."""
    artist, title = _parse_artist_title(filename)

    # Duration from soundfile
    duration = 0.0
    try:
        info = sf.info(str(audio_path))
        duration = round(info.frames / info.samplerate, 2)
    except Exception:
        pass

    # MusicBrainz: get genre tags from artist
    genre_tags: list[str] = []
    mb_info: dict[str, Any] = {}
    if artist:
        genre_tags = _musicbrainz_artist_tags(artist)
    if artist and title:
        mb_info = _musicbrainz_recording_info(artist, title)
        if mb_info.get("duration_ms") and not duration:
            duration = round(mb_info["duration_ms"] / 1000, 2)

    # Audio analysis for BPM and key
    bpm, key = _detect_bpm_key(str(audio_path))

    # Search lyrics online
    lyrics = ""
    if artist and title:
        lyrics = _search_lyrics_online(artist, title)

    has_vocals = bool(lyrics and lyrics.strip() != "[Instrumental]")
    # Detect language from genre tags
    language = "en"
    if any("dutch" in t.lower() or "nederland" in t.lower() for t in genre_tags):
        language = "nl"
    elif any("french" in t.lower() for t in genre_tags):
        language = "fr"
    elif any("spanish" in t.lower() or "latin" in t.lower() for t in genre_tags):
        language = "es"
    elif any("arabic" in t.lower() for t in genre_tags):
        language = "ar"
    elif any("japanese" in t.lower() or "j-pop" in t.lower() for t in genre_tags):
        language = "ja"
    elif any("korean" in t.lower() or "k-pop" in t.lower() for t in genre_tags):
        language = "ko"

    caption = _build_smart_caption(artist, title, bpm, key, has_vocals, genre_tags)
    if trigger_tag:
        if tag_position == "prepend":
            caption = f"{trigger_tag}, {caption}"
        elif tag_position == "append":
            caption = f"{caption}, {trigger_tag}"

    return {
        "path": str(audio_path),
        "filename": filename,
        "caption": caption,
        "lyrics": lyrics or "[Instrumental]",
        "genre": ", ".join(genre_tags[:3]) if genre_tags else "",
        "bpm": bpm,
        "keyscale": key,
        "timesignature": "4",
        "language": language if has_vocals else "instrumental",
        "duration": duration,
        "is_instrumental": not has_vocals,
        "label_source": "smart_musicbrainz",
        "trigger_tag": trigger_tag,
        "tag_position": tag_position,
        "musicbrainz_tags": genre_tags,
        "musicbrainz_album": mb_info.get("album", ""),
        "musicbrainz_year": mb_info.get("year", ""),
    }


@app.post("/api/lora/dataset/autolabel")
async def api_lora_dataset_autolabel(request: Request):
    try:
        body = await request.json()
        files = body.get("files") or []
        mode = str(body.get("mode") or body.get("label_mode") or "smart").strip().lower()
        use_official = parse_bool(body.get("use_official_lm"), False) or mode == "official"
        trigger_tag = body.get("custom_tag") or body.get("trigger_tag") or ""
        tag_position = body.get("tag_position") or "prepend"
        labels = []

        # Smart mode: online lyrics + audio analysis (no model needed)
        if mode == "smart" and not use_official:
            for item in files[: int(body.get("limit") or 24)]:
                path = Path(str(item.get("path") if isinstance(item, dict) else item)).expanduser()
                if path.is_file():
                    label = _smart_autolabel_file(path, path.name, trigger_tag, tag_position)
                    # Preserve any existing metadata from the item
                    if isinstance(item, dict):
                        for key in ("caption", "lyrics", "genre", "bpm", "keyscale", "language"):
                            if item.get(key) and not label.get(key):
                                label[key] = item[key]
                    labels.append(label)
                else:
                    labels.append({"path": str(path), "filename": path.name, "error": "File not found", "label_source": "error"})
            return JSONResponse({"success": True, "labels": labels, "label_mode": "smart", "dataset_health": _lora_dataset_health(labels)})

        # Filename-only fallback mode
        if mode == "filename":
            for item in files[: int(body.get("limit") or 24)]:
                path = Path(str(item.get("path") if isinstance(item, dict) else item)).expanduser()
                duration = None
                try:
                    info = sf.info(str(path))
                    duration = round(info.frames / info.samplerate, 2)
                except Exception:
                    pass
                labels.append({
                    "path": str(path), "filename": path.name,
                    "caption": path.stem.replace("-", " ").replace("_", " "),
                    "lyrics": "[Instrumental]", "genre": "", "bpm": None, "keyscale": "",
                    "timesignature": "4", "language": "instrumental", "duration": duration,
                    "is_instrumental": True, "label_source": "filename_fallback",
                    "trigger_tag": trigger_tag, "tag_position": tag_position,
                })
            return JSONResponse({"success": True, "labels": labels, "label_mode": "filename", "dataset_health": _lora_dataset_health(labels)})

        # Official mode (original behavior, requires model)
        for item in files[: int(body.get("limit") or 24)]:
            path = Path(str(item.get("path") if isinstance(item, dict) else item)).expanduser()
            duration = None
            try:
                info = sf.info(str(path))
                duration = round(info.frames / info.samplerate, 2)
            except Exception:
                pass
            fallback = {
                "path": str(path),
                "filename": path.name,
                "caption": (item.get("caption") if isinstance(item, dict) else "") or path.stem.replace("-", " ").replace("_", " "),
                "lyrics": (item.get("lyrics") if isinstance(item, dict) else "") or "[Instrumental]",
                "genre": item.get("genre", "") if isinstance(item, dict) else "",
                "bpm": (item.get("bpm") if isinstance(item, dict) else None) or None,
                "keyscale": (item.get("keyscale") if isinstance(item, dict) else "") or "",
                "timesignature": (item.get("timesignature") if isinstance(item, dict) else "") or "4",
                "language": (item.get("language") if isinstance(item, dict) else "") or "instrumental",
                "duration": duration or (item.get("duration") if isinstance(item, dict) else 0),
                "is_instrumental": True,
                "label_source": "filename_duration_fallback",
                "trigger_tag": body.get("custom_tag") or body.get("trigger_tag") or "",
                "tag_position": body.get("tag_position") or "prepend",
            }
            if use_official and path.is_file():
                try:
                    with handler_lock:
                        _ensure_song_model(body.get("song_model"))
                        codes = handler.convert_src_audio_to_codes(str(path))
                    understood = _run_official_lm_aux("understand_music", body, audio_codes=codes)
                    fallback.update(
                        {
                            "caption": understood.get("caption") or fallback["caption"],
                            "lyrics": understood.get("lyrics") or fallback["lyrics"],
                            "bpm": understood.get("bpm") or fallback["bpm"],
                            "keyscale": understood.get("key_scale") or fallback["keyscale"],
                            "timesignature": understood.get("time_signature") or fallback["timesignature"],
                            "language": understood.get("language") or fallback["language"],
                            "is_instrumental": str(understood.get("lyrics") or "").strip().lower() == "[instrumental]",
                            "label_source": "official_ace_step_understand_music",
                            "official_understanding": True,
                            "ace_lm_model": understood.get("ace_lm_model"),
                        }
                    )
                except ModelDownloadStarted:
                    raise
                except Exception as official_exc:
                    fallback["official_error"] = str(official_exc)
            labels.append(fallback)
        return JSONResponse({"success": True, "labels": labels, "dataset_health": _lora_dataset_health(labels)})
    except ModelDownloadStarted as exc:
        return JSONResponse(_download_started_payload(exc.model_name, exc.job))


@app.post("/api/lora/one-click-train")
async def api_lora_one_click_train(request: Request):
    try:
        body = await request.json()
        job = training_manager.start_one_click_train(body)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/train")
async def api_lora_train(request: Request):
    try:
        body = await request.json()
        job = training_manager.start_train(body)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/v1/training/start")
async def v1_training_start(request: Request):
    await _require_official_api_key(request)
    try:
        body = await _request_payload(request)
        body["adapter_type"] = "lora"
        job = training_manager.start_train(body)
        return JSONResponse(_official_api_response({"job": job}))
    except Exception as exc:
        return JSONResponse(_official_api_response(None, error=str(exc), code=400), status_code=400)


@app.post("/v1/training/start_lokr")
async def v1_training_start_lokr(request: Request):
    await _require_official_api_key(request)
    try:
        body = await _request_payload(request)
        body["adapter_type"] = "lokr"
        job = training_manager.start_train(body)
        return JSONResponse(_official_api_response({"job": job}))
    except Exception as exc:
        return JSONResponse(_official_api_response(None, error=str(exc), code=400), status_code=400)


@app.post("/api/lora/dataset/save")
async def api_lora_dataset_save(request: Request):
    try:
        body = await request.json()
        entries = body.get("entries") or body.get("labels") or body.get("files") or []
        data = training_manager.save_dataset(
            entries,
            dataset_id=str(body.get("dataset_id") or ""),
            metadata=body.get("metadata") or {},
        )
        data["dataset_health"] = _lora_dataset_health(entries)
        return JSONResponse({"success": True, **data})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/preprocess")
async def api_lora_preprocess(request: Request):
    try:
        body = await request.json()
        job = training_manager.start_preprocess(body)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/estimate")
async def api_lora_estimate(request: Request):
    try:
        body = await request.json()
        job = training_manager.start_estimate(body)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/lora/jobs")
async def api_lora_jobs():
    return JSONResponse({"success": True, "jobs": training_manager.list_jobs()})


@app.get("/api/lora/jobs/{job_id}/log")
async def api_lora_job_log(job_id: str):
    try:
        return JSONResponse({"success": True, **training_manager.read_log(job_id)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)


@app.get("/api/lora/jobs/{job_id}")
async def api_lora_job(job_id: str):
    try:
        return JSONResponse({"success": True, "job": training_manager.get_job(job_id)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)


@app.post("/api/lora/jobs/{job_id}/stop")
async def api_lora_job_stop(job_id: str):
    try:
        return JSONResponse({"success": True, "job": training_manager.stop_job(job_id)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)


@app.get("/api/lora/adapters")
async def api_lora_adapters():
    return JSONResponse({"success": True, "adapters": training_manager.list_adapters()})


@app.post("/api/lora/export")
async def api_lora_export(request: Request):
    body = await request.json()
    try:
        source_text = str(body.get("source_path") or "").strip()
        if not source_text:
            return JSONResponse({"success": False, "error": "LoRA source path is required"}, status_code=400)
        data = training_manager.export_adapter(
            Path(source_text),
            str(body.get("name") or ""),
        )
        return JSONResponse(data)
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/audio-understand")
async def api_audio_understand(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        audio_path = _resolve_upload_file(body.get("upload_id")) or _resolve_result_audio(body.get("result_id"), body.get("audio_id"))
        if audio_path is None:
            return JSONResponse({"success": False, "error": "No audio supplied"}, status_code=400)
        info = sf.info(str(audio_path))
        with handler_lock:
            _ensure_song_model(body.get("song_model"))
            codes = handler.convert_src_audio_to_codes(str(audio_path))
        response = {
            "success": True,
            "duration": round(info.frames / info.samplerate, 2),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "audio_codes": codes,
            "official_understanding": False,
        }
        return JSONResponse(response)
    except ModelDownloadStarted as exc:
        return JSONResponse(_download_started_payload(exc.model_name, exc.job))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/generate_album")
async def api_generate_album(request: Request):
    try:
        body = await request.json()
        planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
        embedding_provider = normalize_provider(body.get("embedding_lm_provider") or body.get("embedding_provider") or planner_provider)
        planner_model = _resolve_local_llm_model_selection(
            planner_provider,
            str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or ""),
            "chat",
            "album generation",
        )
        embedding_model = _resolve_local_llm_model_selection(
            embedding_provider,
            str(body.get("embedding_model") or ""),
            "embedding",
            "album embeddings",
        )
        request_body = {
            **body,
            "planner_lm_provider": planner_provider,
            "embedding_lm_provider": embedding_provider,
            "planner_model": planner_model,
            "ollama_model": planner_model if planner_provider == "ollama" else body.get("ollama_model", ""),
            "embedding_model": embedding_model,
        }
        raw = generate_album(
            concept=str(body.get("concept") or ""),
            num_tracks=int(body.get("num_tracks") or 5),
            track_duration=float(body.get("track_duration") or body.get("duration") or 180.0),
            ollama_model=planner_model,
            language=str(body.get("language") or "en"),
            song_model=str(body.get("song_model") or "auto"),
            embedding_model=embedding_model,
            ace_lm_model=str(body.get("ace_lm_model") or ACE_LM_PREFERRED_MODEL),
            request_json=json.dumps(request_body),
            planner_lm_provider=planner_provider,
            embedding_lm_provider=embedding_provider,
        )
        return JSONResponse(json.loads(raw))
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "album generation"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "logs": [str(exc)]}, status_code=400)


@app.post("/api/album/jobs")
async def api_create_album_job(request: Request):
    try:
        body = await request.json()
        planner_provider = normalize_provider(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama")
        embedding_provider = normalize_provider(body.get("embedding_lm_provider") or body.get("embedding_provider") or planner_provider)
        planner_model = _resolve_local_llm_model_selection(
            planner_provider,
            str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or (DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL if planner_provider == "ollama" else "")),
            "chat",
            "album job planning",
        )
        embedding_model = _resolve_local_llm_model_selection(
            embedding_provider,
            str(body.get("embedding_model") or DEFAULT_ALBUM_EMBEDDING_MODEL),
            "embedding",
            "album job embeddings",
        )
        job_id = uuid.uuid4().hex[:12]
        request_body = {
            **body,
            "planner_lm_provider": planner_provider,
            "embedding_lm_provider": embedding_provider,
            "planner_model": planner_model,
            "ollama_model": planner_model if planner_provider == "ollama" else body.get("ollama_model", ""),
            "embedding_model": embedding_model,
        }
        _set_album_job(
            job_id,
            state="queued",
            status="Queued album production job",
            progress=0,
            payload=request_body,
            planner_model=planner_model,
            planner_provider=planner_provider,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            memory_enabled=True,
            logs=[f"Queued album job {job_id}."],
        )
        threading.Thread(target=_album_job_worker, args=(job_id, request_body), daemon=True).start()
        return JSONResponse({"success": True, "job_id": job_id, "job": _album_job_snapshot(job_id)})
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "album job"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "logs": [str(exc)]}, status_code=400)


@app.get("/api/album/jobs")
async def api_album_jobs_list():
    jobs = _album_job_snapshot(None)
    jobs.sort(key=lambda j: str(j.get("started_at") or j.get("finished_at") or ""), reverse=True)
    return JSONResponse({"success": True, "jobs": jobs})


@app.get("/api/album/jobs/{job_id}")
async def api_album_job_status(job_id: str):
    job = _album_job_snapshot(job_id)
    if not job:
        return JSONResponse({"success": False, "error": "Album job not found"}, status_code=404)
    return JSONResponse({"success": True, "job": job})


@app.get("/api/albums/{album_id}/download")
async def api_download_album(album_id: str):
    zip_path = _build_album_zip(album_id)
    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


@app.get("/api/album-families/{family_id}/download")
async def api_download_album_family(family_id: str):
    zip_path = _build_album_family_zip(family_id)
    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


@app.get("/media/songs/{song_id}/{filename}")
async def media(song_id: str, filename: str):
    songs_root = SONGS_DIR.resolve()
    song_dir = (SONGS_DIR / song_id).resolve()
    target = (song_dir / filename).resolve()
    if songs_root not in song_dir.parents or not song_dir.is_dir() or song_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/media/results/{result_id}/{filename}")
async def result_media(result_id: str, filename: str):
    target = _resolve_child(RESULTS_DIR, safe_id(result_id), filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


demo = app


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False)
