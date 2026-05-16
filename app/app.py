from __future__ import annotations

import base64
import asyncio
import builtins
import errno
import gc
import hashlib
import html as html_lib
import importlib
import ast
import json
import math
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
import urllib.request
import uuid
import zipfile
from urllib.parse import quote
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf


def _safe_print(*args: Any, **kwargs: Any) -> None:
    try:
        builtins.print(*args, **kwargs)
    except (BrokenPipeError, OSError) as exc:
        if isinstance(exc, BrokenPipeError) or getattr(exc, "errno", None) == errno.EPIPE:
            return
        raise


print = _safe_print

for name in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(name, None)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
DATA_DIR = BASE_DIR / "data"
SONGS_DIR = DATA_DIR / "songs"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
ALBUMS_DIR = DATA_DIR / "albums"
SONG_BATCHES_DIR = DATA_DIR / "song_batches"
LORA_BENCHMARKS_DIR = DATA_DIR / "lora_benchmarks"
ART_DIR = DATA_DIR / "art"
LORA_DATASETS_DIR = DATA_DIR / "lora_datasets"
LORA_EXPORTS_DIR = DATA_DIR / "loras"
LORA_IMPORTS_DIR = DATA_DIR / "lora_imports"
LOCAL_LLM_SETTINGS_PATH = DATA_DIR / "local_llm_settings.json"
OFFICIAL_ACE_STEP_DIR = BASE_DIR / "vendor" / "ACE-Step-1.5"
OFFICIAL_RUNNER_SCRIPT = BASE_DIR / "official_runner.py"
PINOKIO_START_LOG = BASE_DIR.parent / "logs" / "api" / "start.js" / "latest"
APP_UI_VERSION = "0.6"
PAYLOAD_CONTRACT_VERSION = "2026-04-26"
ACE_STEP_VENDOR_SYNC_CONFIRM = "SYNC_ACE_STEP_VENDOR_PATCH_PRESERVING"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL = "charaf/qwen3.6-27b-abliterated-mlx:mxfp4-instruct-general"
DEFAULT_ALBUM_EMBEDDING_MODEL = "charaf/qwen3-vl-embedding-8b:latest"
ALBUM_EMBEDDING_FALLBACK_MODELS = [
    DEFAULT_ALBUM_EMBEDDING_MODEL,
    "mxbai-embed-large:latest",
    "nomic-embed-text:latest",
]
ALBUM_JOB_KEEP_LIMIT = 50
GENERATION_JOB_KEEP_LIMIT = 50
SONG_BATCH_JOB_KEEP_LIMIT = 50
LORA_BENCHMARK_JOB_KEEP_LIMIT = 50
ACE_LM_ABLITERATED_DIR = MODEL_CACHE_DIR / "ace_lm_abliterated"
ACE_LM_PREFERRED_MODEL = "acestep-5Hz-lm-4B"
ACEJAM_BOOT_DOWNLOAD_OFFICIAL_HELPERS = _env_flag("ACEJAM_BOOT_DOWNLOAD_OFFICIAL_HELPERS", default=True)
ACEJAM_BOOT_DOWNLOAD_BEST_QUALITY_MODELS = _env_flag("ACEJAM_BOOT_DOWNLOAD_BEST_QUALITY_MODELS", default=True)
ACEJAM_BOOT_DOWNLOAD_ALL_OFFICIAL_MODELS = _env_flag("ACEJAM_BOOT_DOWNLOAD_ALL_OFFICIAL_MODELS", default=False)
ACEJAM_BOOT_DOWNLOAD_ENABLED = _env_flag("ACEJAM_BOOT_DOWNLOAD_ENABLED", default=True)
ACEJAM_BOOT_DOWNLOAD_DELAY_SECONDS = max(0, _env_int("ACEJAM_BOOT_DOWNLOAD_DELAY_SECONDS", 1))
_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
ACE_LM_BACKEND_DEFAULT = "mlx" if _IS_APPLE_SILICON else "pt"
ACE_LM_PRIVATE_UPLOAD_CONFIRM = "PRIVATE_HF_UPLOAD"
ACE_LM_CLEANUP_CONFIRM = "DELETE_ORIGINAL_ACE_LM_AFTER_SMOKE"
ACE_LM_SMOKE_CONFIRM = "ACE_LM_SMOKE_PASSED"
OBLITERATUS_REPO_URL = "https://github.com/elder-plinius/OBLITERATUS"
ACEJAM_PRINT_ACE_PAYLOAD = _env_flag(
    "ACEJAM_PRINT_ACE_PAYLOAD",
    default=True,
)
ACEJAM_PRINT_ACE_PAYLOAD_MAX_CHARS = max(0, _env_int("ACEJAM_PRINT_ACE_PAYLOAD_MAX_CHARS", 0))
ACEJAM_REDACT_OFFICIAL_LOG_TEXT = _env_flag("ACEJAM_REDACT_OFFICIAL_LOG_TEXT", default=False)
ACEJAM_OFFICIAL_RUNNER_TIMEOUT_SECONDS = max(3600, _env_int("ACEJAM_OFFICIAL_RUNNER_TIMEOUT_SECONDS", 10800))
ACEJAM_OFFICIAL_RUNNER_MAX_TIMEOUT_SECONDS = max(
    ACEJAM_OFFICIAL_RUNNER_TIMEOUT_SECONDS,
    _env_int("ACEJAM_OFFICIAL_RUNNER_MAX_TIMEOUT_SECONDS", 21600),
)
ACEJAM_GENERATE_ADVANCED_TIME_LIMIT_SECONDS = max(
    ACEJAM_OFFICIAL_RUNNER_TIMEOUT_SECONDS,
    _env_int("ACEJAM_GENERATE_ADVANCED_TIME_LIMIT_SECONDS", 10800),
)
ACEJAM_GENERATE_ALBUM_TIME_LIMIT_SECONDS = max(
    ACEJAM_GENERATE_ADVANCED_TIME_LIMIT_SECONDS,
    _env_int("ACEJAM_GENERATE_ALBUM_TIME_LIMIT_SECONDS", 21600),
)

_ALBUM_QUALITY_REQUIRED_EXPORTS = (
    "build_lyrical_craft_contract",
    "build_producer_grade_sonic_contract",
    "lyric_craft_gate",
    "lyric_density_gate",
    "producer_grade_readiness",
    "sonic_dna_coverage",
)


def _album_crew_has_recursion_bug(crew_module) -> bool:
    """Detect whether the in-memory album_crew module still carries the
    pre-fix `_agent_full_system_prompt` that called itself (RecursionError
    on first crew agent). Recognise the buggy version by reading the source
    of the function and checking for a self-call inside its body. Returns
    True when the function recurses, signalling we must reload the module
    so the live dev server picks up the fix without an app restart."""
    func = getattr(crew_module, "_agent_full_system_prompt", None)
    if func is None:
        return True  # missing function = stale module, force reload
    try:
        import inspect

        source = inspect.getsource(func)
    except (OSError, TypeError):
        return False
    # Buggy body had the function calling itself with the same kwargs.
    # Fixed body delegates to `_agent_system_prompt(agent_name)` instead.
    if "_agent_full_system_prompt(" in source and "_agent_system_prompt(" not in source:
        return True
    return False


def _ensure_album_agent_modules_current() -> None:
    """Reload album gate / crew modules so a live dev server picks up source
    fixes without an app restart. Each call invalidates the import caches,
    refreshes album_quality_gate when stale, and ALWAYS reloads album_crew
    so source-level edits (recursion fix, thinking-variant detect, persona
    tweaks, etc.) take effect on the next wizard fill instead of waiting
    for the next process restart. Reload cost is ~50-200ms — negligible
    against a multi-minute crew run."""
    importlib.invalidate_caches()
    gate_module = sys.modules.get("album_quality_gate")
    if gate_module is not None:
        missing = [name for name in _ALBUM_QUALITY_REQUIRED_EXPORTS if not hasattr(gate_module, name)]
        if missing:
            gate_module = importlib.reload(gate_module)
            still_missing = [name for name in _ALBUM_QUALITY_REQUIRED_EXPORTS if not hasattr(gate_module, name)]
            if still_missing:
                raise ImportError(
                    "album_quality_gate is missing required export(s): "
                    + ", ".join(still_missing)
                )
    crew_module = sys.modules.get("album_crew")
    if crew_module is not None:
        # Skip reload when running under pytest — pytest's mock.patch on
        # `album_crew.plan_album` does not survive an importlib.reload, so
        # forcing one would wipe every test's mock and turn unit tests
        # into real-network calls. Production / dev server runs always
        # reload so source-level fixes take effect on the next fill.
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            importlib.reload(crew_module)
ACEJAM_VOCAL_INTELLIGIBILITY_GATE = _env_flag("ACEJAM_VOCAL_INTELLIGIBILITY_GATE", default=True)
ACEJAM_VOCAL_INTELLIGIBILITY_ATTEMPTS = max(1, _env_int("ACEJAM_VOCAL_INTELLIGIBILITY_ATTEMPTS", 8))
ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE = _env_flag("ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE", default=True)
ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_AFTER = max(
    1,
    _env_int("ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_AFTER", 1),
)
ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_ATTEMPTS = max(
    1,
    _env_int("ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_ATTEMPTS", 2),
)
ACEJAM_VOCAL_INTELLIGIBILITY_RESCUE_MODELS = [
    item.strip()
    for item in os.environ.get(
        "ACEJAM_VOCAL_INTELLIGIBILITY_RESCUE_MODELS",
        "acestep-v15-turbo,acestep-v15-xl-turbo,acestep-v15-turbo-shift3,acestep-v15-sft",
    ).split(",")
    if item.strip()
]
ACEJAM_VOCAL_INTELLIGIBILITY_MIN_WORDS = max(1, _env_int("ACEJAM_VOCAL_INTELLIGIBILITY_MIN_WORDS", 8))
ACEJAM_VOCAL_INTELLIGIBILITY_MIN_KEYWORDS = max(0, _env_int("ACEJAM_VOCAL_INTELLIGIBILITY_MIN_KEYWORDS", 2))
ACEJAM_VOCAL_INTELLIGIBILITY_MIN_UNIQUE_WORDS = max(1, _env_int("ACEJAM_VOCAL_INTELLIGIBILITY_MIN_UNIQUE_WORDS", 5))
ACEJAM_VOCAL_INTELLIGIBILITY_MAX_FILLER_RATIO = min(
    1.0,
    max(0.0, _env_float("ACEJAM_VOCAL_INTELLIGIBILITY_MAX_FILLER_RATIO", 0.30)),
)
ACEJAM_VOCAL_INTELLIGIBILITY_MAX_REPEAT_RATIO = min(
    1.0,
    max(0.0, _env_float("ACEJAM_VOCAL_INTELLIGIBILITY_MAX_REPEAT_RATIO", 0.28)),
)
ACEJAM_VOCAL_ASR_TIMEOUT = max(30, _env_int("ACEJAM_VOCAL_ASR_TIMEOUT", 240))
ACEJAM_VOCAL_ASR_MODEL = os.environ.get("ACEJAM_VOCAL_ASR_MODEL", "").strip()
ACEJAM_VOCAL_ASR_DEVICE = os.environ.get("ACEJAM_VOCAL_ASR_DEVICE", "cpu").strip().lower() or "cpu"
ACEJAM_LORA_PREFLIGHT_DURATION_SECONDS = max(10, min(60, _env_int("ACEJAM_LORA_PREFLIGHT_DURATION_SECONDS", 30)))
ACEJAM_LORA_PREFLIGHT_SCALES = tuple(
    sorted(
        {
            max(0.0, min(1.0, _env_float(f"ACEJAM_LORA_PREFLIGHT_SCALE_{index}", default)))
            for index, default in enumerate((0.15, 0.30, 0.45), start=1)
        }
    )
)
ACEJAM_LORA_UNSAFE_QUALITY_STATUSES = {"not_generation_loadable"}
ACEJAM_LORA_REVIEW_QUALITY_STATUSES = {"needs_review"}

MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ACE_LM_ABLITERATED_DIR.mkdir(parents=True, exist_ok=True)
SONGS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ALBUMS_DIR.mkdir(parents=True, exist_ok=True)
SONG_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
LORA_BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server

from local_llm import (
    PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
    chat_completion as local_llm_chat_completion,
    chat_completion_response as local_llm_chat_completion_response,
    lmstudio_download_model,
    lmstudio_download_status,
    lmstudio_load_model,
    lmstudio_model_catalog,
    lmstudio_unload_model,
    normalize_provider,
    planner_llm_options_for_provider,
    planner_llm_settings_from_payload,
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
from lora_trainer import (
    DEFAULT_LORA_GENERATION_SCALE,
    EPOCH_AUDITION_CLARITY_CAPTION,
    AceTrainingManager,
    adapter_quality_metadata,
    apply_audio_style_conditioning,
    audio_style_profiles,
    epoch_audition_genre_options,
    fit_epoch_audition_lyrics,
    infer_adapter_model_metadata,
    is_missing_vocal_lyrics,
    model_from_variant,
    model_to_variant,
    normalize_training_song_model,
    safe_generation_trigger_tag,
    training_inference_defaults,
    vocal_dataset_health,
)
from mflux_manager import (
    MFLUX_ALLOWED_IMAGE_EXTENSIONS,
    MFLUX_RESULTS_DIR,
    MFLUX_UPLOADS_DIR,
    mflux_create_job,
    mflux_get_job,
    mflux_list_jobs,
    mflux_list_lora_adapters,
    mflux_models,
    mflux_public_upload_url,
    mflux_start_lora_training,
    mflux_status,
)
from mlx_video_manager import (
    MLX_VIDEO_ALLOWED_UPLOAD_EXTENSIONS,
    MLX_VIDEO_ATTACHMENTS_PATH,
    MLX_VIDEO_JOBS_DIR,
    MLX_VIDEO_RESULTS_DIR,
    MLX_VIDEO_UPLOADS_DIR,
    mlx_video_attach,
    mlx_video_create_job,
    mlx_video_get_job,
    mlx_video_list_attachments,
    mlx_video_list_jobs,
    mlx_video_list_loras,
    mlx_video_models,
    mlx_video_public_upload_url,
    mlx_video_register_model_dir,
    mlx_video_registered_model_dirs,
    mlx_video_status,
)
from local_composer import LocalComposer
from album_quality_gate import (
    ALBUM_PAYLOAD_GATE_VERSION,
    AlbumRunDebugLogger,
    build_album_global_sonic_caption,
    evaluate_album_payload_quality,
    evaluate_genre_adherence,
)
from songwriting_toolkit import (
    ALBUM_FINAL_MODEL,
    ALBUM_MODEL_PORTFOLIO,
    ALBUM_MODEL_PORTFOLIO_MODELS,
    MODEL_STRATEGIES,
    album_model_portfolio,
    album_models_for_strategy,
    choose_song_model,
    derive_artist_name,
    lyric_length_plan,
    lyric_stats,
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
    QUALITY_PROFILE_OFFICIAL_RAW,
    QUALITY_PROFILE_DOCS_DAILY,
    KNOWN_ACE_STEP_MODELS,
    MAX_BATCH_SIZE,
    OFFICIAL_ACE_STEP_MODEL_REGISTRY,
    OFFICIAL_ACE_STEP_MANIFEST,
    OFFICIAL_CORE_MODEL_ID,
    OFFICIAL_MAIN_MODEL_COMPONENTS,
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
    official_boot_model_ids,
    official_downloadable_model_ids,
    official_helper_model_ids,
    official_manifest,
    official_model_registry,
    official_model_repo_id,
    ordered_models,
    parse_bool,
    parse_timesteps,
    pro_quality_policy,
    recommended_lm_model,
    recommended_song_model,
    quality_profile_model_settings,
    diffusers_pipeline_dir_ready,
    diffusers_pipeline_missing_reasons,
    runtime_planner_report,
    safe_filename,
    safe_id,
    studio_ui_schema,
    strip_ace_step_lyrics_leakage,
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


_official_generation_runner_lock = threading.Lock()


def _mps_memory_snapshot(label: str = "") -> dict[str, Any]:
    mps = getattr(torch, "mps", None)
    backend = getattr(torch.backends, "mps", None)
    if not (mps and backend and backend.is_available()):
        return {"available": False, "label": label}

    def read_int(name: str) -> int | None:
        fn = getattr(mps, name, None)
        if not callable(fn):
            return None
        try:
            return int(fn())
        except Exception:
            return None

    snapshot: dict[str, Any] = {
        "available": True,
        "label": label,
        "current_allocated_bytes": read_int("current_allocated_memory"),
        "driver_allocated_bytes": read_int("driver_allocated_memory"),
        "recommended_max_bytes": read_int("recommended_max_memory"),
    }
    for key in ["current_allocated_bytes", "driver_allocated_bytes", "recommended_max_bytes"]:
        value = snapshot.get(key)
        if isinstance(value, int):
            snapshot[key.replace("_bytes", "_gib")] = round(value / (1024 ** 3), 3)
    return snapshot


def _is_mps_oom_error(error: Any) -> bool:
    text = str(error or "").lower()
    return "mps backend out of memory" in text or ("mps" in text and "out of memory" in text)


def _is_acestep_generation_timeout_error(error: Any) -> bool:
    text = str(error or "").lower()
    return (
        "music generation timed out" in text
        or "service_generate exceeded" in text
        or "official ace-step runner timed out" in text
        or ("generation" in text and "timed out" in text)
    )


def _prepare_audio_generation_memory(context: str = "", *, release_handler: bool = False) -> dict[str, Any]:
    before = _mps_memory_snapshot(f"{context}:before_cleanup")
    _unload_llm_models_for_generation()
    if release_handler:
        with handler_lock:
            _release_handler_state()
    else:
        gc.collect()
        _cleanup_accelerator_memory()
    after = _mps_memory_snapshot(f"{context}:after_cleanup")
    return {"context": context, "before": before, "after": after}


_accelerator_cleanup_lock = threading.Lock()
_accelerator_cleanup_active = False


def _schedule_accelerator_cleanup(context: str = "") -> None:
    global _accelerator_cleanup_active
    with _accelerator_cleanup_lock:
        if _accelerator_cleanup_active:
            return
        _accelerator_cleanup_active = True

    def worker() -> None:
        global _accelerator_cleanup_active
        try:
            _cleanup_accelerator_memory()
        except Exception as exc:
            print(f"[cleanup] accelerator cleanup skipped after {context or 'request'}: {exc}", flush=True)
        finally:
            with _accelerator_cleanup_lock:
                _accelerator_cleanup_active = False

    threading.Thread(target=worker, name="acejam-accelerator-cleanup", daemon=True).start()


def _default_acestep_checkpoint() -> str:
    override = os.environ.get("ACE_STEP_MODEL", "").strip()
    if override:
        return override
    return "acestep-v15-xl-sft"


def _song_model_label(name: str) -> str:
    return model_label(name)


def _app_ui_hash() -> str:
    try:
        return hashlib.sha256(
            (BASE_DIR / "web" / "dist" / "index.html").read_bytes()
        ).hexdigest()[:16]
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
        if lowered in {"none", "off", "false", "0", "disabled"}:
            return "none"
        if lowered == "auto":
            return ACE_LM_PREFERRED_MODEL
        return value
    return ACE_LM_PREFERRED_MODEL


def _writer_provider_from_payload(payload: dict[str, Any] | None, default: str = "ollama") -> str:
    source = payload if isinstance(payload, dict) else {}
    global_default = default
    if "planner_lm_provider" not in source and "planner_provider" not in source:
        try:
            global_default = str(_load_local_llm_settings().get("provider") or default)
        except Exception:
            global_default = default
    return normalize_provider(source.get("planner_lm_provider") or source.get("planner_provider") or global_default)


def _album_planner_provider_from_payload(payload: dict[str, Any] | None, default: str = "ollama") -> str:
    provider = _writer_provider_from_payload(payload, default)
    return "ollama" if provider == "ace_step_lm" else provider


def _embedding_provider_from_payload(payload: dict[str, Any] | None, default: str = "ollama") -> str:
    source = payload if isinstance(payload, dict) else {}
    global_default = default
    if "embedding_lm_provider" not in source and "embedding_provider" not in source:
        try:
            global_default = str(_load_local_llm_settings().get("embedding_provider") or default)
        except Exception:
            global_default = default
    provider = normalize_provider(source.get("embedding_lm_provider") or source.get("embedding_provider") or global_default)
    return "ollama" if provider == "ace_step_lm" else provider


def _album_ace_lm_disabled_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = dict(payload or {})
    original_provider = _writer_provider_from_payload(cleaned)
    cleaned["planner_lm_provider"] = _album_planner_provider_from_payload(cleaned)
    if original_provider == "ace_step_lm":
        fallback_model = str(
            cleaned.get("planner_ollama_model")
            or cleaned.get("ollama_model")
            or globals().get("DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL", "")
            or ""
        ).strip()
        cleaned["planner_model"] = fallback_model
        cleaned["planner_ollama_model"] = fallback_model
        cleaned["ollama_model"] = fallback_model
    cleaned["ace_lm_model"] = "none"
    cleaned["lm_model"] = "none"
    cleaned["lm_model_path"] = "none"
    cleaned["use_official_lm"] = False
    cleaned["thinking"] = False
    cleaned["sample_mode"] = False
    cleaned["sample_query"] = ""
    cleaned["use_format"] = False
    cleaned["use_cot_metas"] = False
    cleaned["use_cot_caption"] = False
    cleaned["use_cot_lyrics"] = False
    cleaned["use_cot_language"] = False
    cleaned["allow_supplied_lyrics_lm"] = False
    cleaned["album_use_ace_lm_for_supplied_lyrics"] = False
    cleaned["album_allow_ace_lm_rewrite"] = False
    return cleaned


def _normalize_lm_backend(value: Any) -> str:
    backend = str(value or ACE_LM_BACKEND_DEFAULT).strip().lower()
    if backend == "auto":
        return ACE_LM_BACKEND_DEFAULT
    if backend == "mlx" and not _IS_APPLE_SILICON:
        return ACE_LM_BACKEND_DEFAULT
    if backend == "pt" and _IS_APPLE_SILICON and not parse_bool(os.environ.get("ACEJAM_ALLOW_PT_LM_BACKEND_ON_APPLE"), False):
        return ACE_LM_BACKEND_DEFAULT
    return backend if backend in {"pt", "vllm", "mlx"} else ACE_LM_BACKEND_DEFAULT


def _default_audio_backend() -> str:
    return "mps_torch"


ACE_AUDIO_BACKEND_DEFAULT = _default_audio_backend()


def _truthy_backend_flag(value: Any) -> bool | None:
    if value in [None, ""]:
        return None
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return None
    if text in {"true", "1", "yes", "on", "mlx"}:
        return True
    if text in {"false", "0", "no", "off", "pt", "torch", "mps", "mps_torch", "pytorch"}:
        return False
    return None


def _normalize_audio_backend(value: Any = None, use_mlx_dit: Any = None) -> str:
    """Normalize the user-facing audio runtime choice.

    `lm_backend` controls ACE-Step's language model path. Audio DiT/VAE runtime
    needs a separate switch so the UI can default to PyTorch/MPS quality while
    keeping native MLX available as an explicit fast/experimental choice.
    """
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"mlx", "native_mlx", "mlx_dit", "mlx_audio"}:
        return "mlx" if _IS_APPLE_SILICON else "mps_torch"
    if raw in {"mps", "mps_torch", "torch", "pytorch", "pt"}:
        return "mps_torch"
    flag = _truthy_backend_flag(use_mlx_dit)
    if flag is True:
        return "mlx" if _IS_APPLE_SILICON else "mps_torch"
    if flag is False:
        return "mps_torch"
    return _default_audio_backend()


def _audio_backend_uses_mlx(params: dict[str, Any] | None) -> bool:
    if not isinstance(params, dict):
        return False
    return _normalize_audio_backend(params.get("audio_backend"), params.get("use_mlx_dit")) == "mlx"


def _apply_audio_backend_defaults(params: dict[str, Any], *, source: str) -> dict[str, Any]:
    backend = _normalize_audio_backend(params.get("audio_backend"), params.get("use_mlx_dit"))
    changed: list[str] = []
    if params.get("audio_backend") != backend:
        params["audio_backend"] = backend
        changed.append(f"audio_backend={backend}")
    desired_mlx = backend == "mlx"
    if params.get("use_mlx_dit") is not desired_mlx:
        params["use_mlx_dit"] = desired_mlx
        changed.append(f"use_mlx_dit={str(desired_mlx).lower()}")
    if _IS_APPLE_SILICON and str(params.get("device") or "auto").strip().lower() in {"", "auto"}:
        params["device"] = "mps"
        changed.append("device=mps")
    if _IS_APPLE_SILICON and str(params.get("dtype") or "auto").strip().lower() in {"", "auto"}:
        params["dtype"] = "float32"
        changed.append("dtype=float32")
    if changed:
        warnings = list(params.get("payload_warnings") or [])
        warning = f"audio_backend_defaults:{source}:{','.join(changed)}"
        if warning not in warnings:
            warnings.append(warning)
        params["payload_warnings"] = warnings
    return params


def _finalize_official_audio_backend_request(params: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Make the user-facing audio backend choice authoritative for the runner."""
    backend = _normalize_audio_backend(params.get("audio_backend"), params.get("use_mlx_dit"))
    use_mlx_dit = backend == "mlx"
    params["audio_backend"] = backend
    params["use_mlx_dit"] = use_mlx_dit
    request["audio_backend"] = backend
    request["use_mlx_dit"] = use_mlx_dit
    request["requested_audio_backend"] = backend
    request["requested_use_mlx_dit"] = use_mlx_dit
    request["audio_backend_contract"] = {
        "requested_audio_backend": backend,
        "requested_use_mlx_dit": use_mlx_dit,
        "enforced_at": "official_request",
    }
    if backend == "mlx" and request.get("use_mlx_dit") is not True:
        raise RuntimeError("MLX audio backend was requested but the official request was not configured for MLX DiT.")
    return request


VOCAL_CLARITY_CAPTION_TRAITS = [
    "clear intelligible English rap vocal",
    "dry upfront lead vocal",
    "crisp consonants",
    "tight spoken-word delivery",
    "minimal vocal effects",
    "radio-ready hook",
]

ACEJAM_AUTO_VOCAL_CLARITY_RECOVERY = _env_flag("ACEJAM_AUTO_VOCAL_CLARITY_RECOVERY", default=True)


def _supplied_vocal_lyrics(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if parse_bool(payload.get("instrumental"), False):
        return False
    lyrics = str(payload.get("lyrics") or "").strip()
    return bool(lyrics and lyrics.lower() != "[instrumental]")


def _explicit_vocal_clarity_recovery(payload: dict[str, Any] | None) -> bool | None:
    if not isinstance(payload, dict):
        return None
    for key in ("vocal_clarity_recovery", "album_vocal_clarity_recovery"):
        if key in payload and payload.get(key) not in [None, ""]:
            return parse_bool(payload.get(key), False)
    return None


def _album_or_agent_vocal_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    ui_mode = str(payload.get("ui_mode") or "").strip().lower()
    return bool(
        ui_mode == "album"
        or payload.get("album_metadata")
        or payload.get("album_id")
        or payload.get("agent_complete_payload")
        or payload.get("album_vocal_clarity_recovery_auto")
    )


def _vocal_clarity_recovery_enabled(payload: dict[str, Any]) -> bool:
    explicit = _explicit_vocal_clarity_recovery(payload)
    if explicit is not None:
        return explicit
    if not ACEJAM_AUTO_VOCAL_CLARITY_RECOVERY:
        return False
    task_type = str((payload or {}).get("task_type") or "text2music").strip().lower()
    if task_type != "text2music":
        return False
    return _album_or_agent_vocal_payload(payload) and _supplied_vocal_lyrics(payload)


def _caption_with_vocal_clarity_traits(caption: Any) -> str:
    existing = str(caption or "").strip()
    terms = split_terms(existing)
    lowered = {term.lower() for term in terms}
    for trait in VOCAL_CLARITY_CAPTION_TRAITS:
        if trait.lower() in lowered:
            continue
        candidate_terms = [*terms, trait]
        candidate = ", ".join(candidate_terms).strip(", ")
        if len(candidate) <= ACE_STEP_CAPTION_CHAR_LIMIT:
            terms.append(trait)
            lowered.add(trait.lower())
    return ", ".join(terms).strip(", ") or ", ".join(VOCAL_CLARITY_CAPTION_TRAITS[:3])


def _disable_acestep_mlx_backends(handler_cls: Any) -> None:
    def _disabled_mlx_backends(self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False
        self.mlx_vae = None
        self.use_mlx_vae = False
        return "Disabled by MLX Media (PyTorch/MPS)", "Disabled by MLX Media (PyTorch/MPS)"

    handler_cls._initialize_mlx_backends = _disabled_mlx_backends


def _apply_studio_lm_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Force the writer onto Ollama/LM Studio and keep the ACE-Step 5Hz LM rewriter off."""
    cleaned = dict(payload or {})
    requested_provider = _writer_provider_from_payload(cleaned)
    if requested_provider == "ace_step_lm":
        try:
            fallback = normalize_provider(_load_local_llm_settings().get("provider") or "ollama")
        except Exception:
            fallback = "ollama"
        if fallback == "ace_step_lm":
            fallback = "ollama"
        provider = fallback
    else:
        provider = requested_provider
    cleaned["planner_lm_provider"] = provider
    cleaned.pop("planner_ollama_model", None) if provider == "lmstudio" else None
    if provider == "ollama":
        cleaned.setdefault("planner_ollama_model", str(cleaned.get("planner_model") or cleaned.get("ollama_model") or "").strip())
    cleaned.setdefault(
        "planner_model",
        str(cleaned.get("planner_model") or cleaned.get("planner_ollama_model") or cleaned.get("ollama_model") or "").strip(),
    )
    if "sample_query" not in cleaned:
        sample_query = str(get_param(cleaned, "sample_query", "") or "").strip()
        if sample_query:
            cleaned["sample_query"] = sample_query
    if "use_format" not in cleaned:
        use_format = get_param(cleaned, "use_format", None)
        if use_format not in [None, ""]:
            cleaned["use_format"] = use_format
    cleaned.update(planner_llm_settings_from_payload(cleaned))
    for field, default in ACE_LM_DISABLED_DEFAULTS.items():
        cleaned[field] = default
    cleaned["use_official_lm"] = False
    cleaned["allow_supplied_lyrics_lm"] = False
    cleaned["album_use_ace_lm_for_supplied_lyrics"] = False
    cleaned["album_allow_ace_lm_rewrite"] = False
    cleaned["sample_mode"] = False
    cleaned["sample_query"] = ""
    cleaned["lm_backend"] = _normalize_lm_backend(cleaned.get("lm_backend"))
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
    if has_supplied_vocal_lyrics and not parse_bool(payload.get("allow_supplied_lyrics_lm"), False):
        return False
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


DIRECT_LYRICS_LM_FIELDS = {
    "allow_lm_batch",
    "constrained_decoding_debug",
    "lm_batch_chunk_size",
    "lm_cfg_scale",
    "lm_negative_prompt",
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


def _active_official_fields(payload: dict[str, Any], task_type: str, existing: list[str]) -> list[str]:
    fields = list(existing)
    supplied_lyrics = str(payload.get("lyrics") or "").strip()
    direct_lyrics_render = (
        task_type == "text2music"
        and bool(supplied_lyrics and supplied_lyrics.lower() != "[instrumental]")
        and not parse_bool(payload.get("allow_supplied_lyrics_lm"), False)
    )
    if direct_lyrics_render:
        fields = [field for field in fields if field not in DIRECT_LYRICS_LM_FIELDS]
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


def _docs_correct_render_steps(song_model: str, quality_profile: str | None, raw_steps: Any) -> int:
    profile = normalize_quality_profile(quality_profile)
    model_defaults = quality_profile_model_settings(song_model, profile)
    default_steps = int(model_defaults["inference_steps"])
    return default_steps


def _docs_correct_render_shift(song_model: str, quality_profile: str | None, raw_shift: Any) -> float:
    profile = normalize_quality_profile(quality_profile)
    model_defaults = quality_profile_model_settings(song_model, profile)
    default_shift = float(model_defaults["shift"])
    if "turbo" not in str(song_model or "").lower() and profile != QUALITY_PROFILE_OFFICIAL_RAW:
        return default_shift
    return clamp_float(raw_shift, default_shift, 1.0, 5.0)


def _docs_correct_render_guidance(song_model: str, quality_profile: str | None, raw_guidance: Any) -> float:
    profile = normalize_quality_profile(quality_profile)
    model_defaults = quality_profile_model_settings(song_model, profile)
    default_guidance = float(model_defaults["guidance_scale"])
    if profile != QUALITY_PROFILE_OFFICIAL_RAW:
        return default_guidance
    return clamp_float(raw_guidance, default_guidance, 1.0, 15.0)


def _enforce_model_correct_render_settings(params: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Keep stale UI drafts from sending Turbo inference schedules to SFT/Base models."""
    song_model = str(params.get("song_model") or "").strip()
    quality_profile = normalize_quality_profile(params.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    if not song_model:
        return params
    before_steps = params.get("inference_steps")
    before_shift = params.get("shift")
    before_guidance = params.get("guidance_scale")
    corrected_steps = _docs_correct_render_steps(song_model, quality_profile, before_steps)
    corrected_shift = _docs_correct_render_shift(song_model, quality_profile, before_shift)
    corrected_guidance = _docs_correct_render_guidance(song_model, quality_profile, before_guidance)
    params["inference_steps"] = corrected_steps
    params["shift"] = corrected_shift
    params["guidance_scale"] = corrected_guidance
    if before_steps != corrected_steps or before_shift != corrected_shift or before_guidance != corrected_guidance:
        warnings = list(params.get("payload_warnings") or [])
        warning = (
            "model_corrected_render_settings:"
            f"{source}:{song_model}:steps={before_steps}->{corrected_steps},"
            f"shift={before_shift}->{corrected_shift},"
            f"guidance={before_guidance}->{corrected_guidance}"
        )
        if warning not in warnings:
            warnings.append(warning)
        params["payload_warnings"] = warnings
    return params


def _apply_mac_mlx_xl_repetition_guard(params: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Avoid the upstream-reported Mac XL repetition trap for MLX-backed text2music.

    ACE-Step issue #1191 reports severe Mac XL Turbo/SFT artifacts when DCW and
    LM-code conditioning strength are both active. Keep this as an app-side guard
    until upstream lands a real model/runtime fix. MPS/Torch is the default
    quality backend; this only protects explicit MLX selections.
    """
    if not parse_bool(os.environ.get("ACEJAM_MAC_MLX_XL_REPETITION_GUARD"), True):
        return params
    if not _IS_APPLE_SILICON:
        return params
    if normalize_task_type(params.get("task_type")) != "text2music":
        return params
    song_model = str(params.get("song_model") or "").strip().lower()
    if "xl" not in song_model or not any(part in song_model for part in ["turbo", "sft", "base"]):
        return params
    if not _audio_backend_uses_mlx(params):
        return params

    before_dcw = parse_bool(params.get("dcw_enabled"), True)
    before_cover = clamp_float(params.get("audio_cover_strength"), 1.0, 0.0, 1.0)
    changed = before_dcw or before_cover != 0.0
    params["dcw_enabled"] = False
    params["audio_cover_strength"] = 0.0
    if changed:
        warnings = list(params.get("payload_warnings") or [])
        warning = (
            "mac_mlx_xl_repetition_guard:"
            f"{source}:dcw_enabled={before_dcw}->False,"
            f"audio_cover_strength={before_cover:g}->0"
        )
        if warning not in warnings:
            warnings.append(warning)
        params["payload_warnings"] = warnings
    return params


def _mps_long_lora_memory_guard_required(params: dict[str, Any]) -> bool:
    if not _IS_APPLE_SILICON:
        return False
    if normalize_task_type(params.get("task_type")) != "text2music":
        return False
    if _audio_backend_uses_mlx(params):
        return False
    if str(params.get("device") or "auto").strip().lower() not in {"auto", "mps", "metal"}:
        return False
    song_model = str(params.get("song_model") or "").strip().lower()
    if "xl" not in song_model or "turbo" in song_model:
        return False
    if not parse_bool(params.get("use_lora"), False):
        return False
    return clamp_float(params.get("duration"), 60.0, DURATION_MIN, DURATION_MAX) >= 240.0


def _apply_mps_long_lora_memory_guard(params: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Reduce avoidable MPS memory pressure for full-song XL-SFT/Base LoRA renders.

    The model, steps, shift, backend and user LoRA scale stay untouched. DCW is
    the only automatic change here because it is an optional correction pass and
    materially increases long-render memory pressure on PyTorch/MPS.
    """
    if not parse_bool(os.environ.get("ACEJAM_MPS_LONG_LORA_MEMORY_GUARD"), True):
        return params
    if not _mps_long_lora_memory_guard_required(params):
        return params
    before_dcw = parse_bool(params.get("dcw_enabled"), True)
    if not before_dcw:
        return params
    params["dcw_enabled"] = False
    warnings = list(params.get("payload_warnings") or [])
    warning = (
        "mps_long_lora_memory_guard:"
        f"{source}:duration={params.get('duration')},"
        f"song_model={params.get('song_model')},dcw_enabled=True->False"
    )
    if warning not in warnings:
        warnings.append(warning)
    params["payload_warnings"] = warnings
    repairs = list(params.get("repair_actions") or [])
    repairs.append(
        {
            "type": "mps_long_lora_memory_guard",
            "source": source,
            "duration": params.get("duration"),
            "song_model": params.get("song_model"),
            "use_lora": True,
            "dcw_enabled": False,
        }
    )
    params["repair_actions"] = repairs
    return params


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
    "image": {"label": "Image Studio", "file": "promptimage.md", "description": "MFLUX image generation/edit prompt and settings."},
    "video": {"label": "Video Studio", "file": "promptvideo.md", "description": "MLX Video prompt, source intent and render settings."},
    "improve": {"label": "Improve Lyrics", "file": "promptverbeter.md", "description": "Improve lyrics and optionally create MLX Media fields."},
    "trainer": {"label": "Trainer / LoRA", "file": "prompttrainer.md", "description": "Dataset labels and training metadata."},
}
PROMPT_ASSISTANT_DISABLED_MODES = {"trainer"}

PROMPT_ASSISTANT_ALIASES = {
    "lora": "trainer",
    "trainer_lora": "trainer",
    "library": "custom",
    "news_to_song": "news",
    "lyrics": "improve",
    "settings": "custom",
    "image_studio": "image",
    "art": "image",
    "video_studio": "video",
    "music_video": "video",
}


class PromptAssistantStageError(RuntimeError):
    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


def _prompt_assistant_mode(value: str) -> str:
    mode = str(value or "custom").strip().lower().replace("-", "_")
    mode = PROMPT_ASSISTANT_ALIASES.get(mode, mode)
    if mode not in PROMPT_ASSISTANT_MODES:
        raise ValueError(f"Unknown prompt assistant mode: {value}")
    return mode


def _prompt_assistant_path(mode: str) -> Path:
    info = PROMPT_ASSISTANT_MODES[_prompt_assistant_mode(mode)]
    prompts_root = (BASE_DIR / "prompts").resolve()
    path = (prompts_root / info["file"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: app/prompts/{info['file']}")
    if prompts_root not in path.parents:
        raise ValueError("Prompt file path escaped app/prompts/")
    return path


def _prompt_assistant_system_prompt(mode: str) -> str:
    # Album mode is handled by the CrewAI Micro Tasks director in
    # _run_prompt_assistant_album_crew before this function is reached. The
    # legacy single-Ollama-call album system prompt was deleted because each
    # wizard field is now filled by a specialised crew agent (Topline Hook
    # Writer, Tier-1 Lyric Writer, Sonic Tags Engineer, etc.).
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


def _prompt_assistant_structured_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "payload": {"type": "object", "additionalProperties": True},
            "paste_blocks": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["payload"],
    }


def _unwrap_prompt_assistant_structured_payload(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str], Any]:
    if isinstance(parsed.get("payload"), dict):
        warnings = [str(item) for item in parsed.get("warnings") or [] if str(item).strip()]
        return dict(parsed["payload"]), warnings, parsed.get("paste_blocks")
    return parsed, [], None


def _server_paste_blocks_from_payload(payload: dict[str, Any], mode: str) -> str:
    title = str(payload.get("title") or payload.get("album_title") or "Untitled").strip()
    caption = str(payload.get("caption") or payload.get("tags") or payload.get("concept") or "").strip()
    negative = str(payload.get("negative_tags") or "").strip()
    lyrics = str(payload.get("lyrics") or "").strip()
    lines = [
        f"Title: {title}",
        f"Caption / Tags: {caption}",
    ]
    if negative:
        lines.append(f"Negative Tags: {negative}")
    if lyrics:
        lines.append(f"Lyrics: {lyrics}")
    for label, key in [("BPM", "bpm"), ("Key", "key_scale"), ("Time Signature", "time_signature"), ("Duration", "duration")]:
        if payload.get(key) not in [None, ""]:
            lines.append(f"{label}: {payload.get(key)}")
    if payload.get("song_model"):
        lines.append(f"Model / Settings: {payload.get('song_model')}")
    if mode == "album" and isinstance(payload.get("tracks"), list):
        lines.append(f"Album tracks: {len(payload.get('tracks') or [])}")
    return "\n".join(lines).strip()


def _compact_text_for_prompt(value: Any, limit: int = 1400) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    head = text[: max(200, limit // 2)].rstrip()
    tail = text[-max(200, limit // 3) :].lstrip()
    return f"{head}\n...[truncated {len(text) - len(head) - len(tail)} chars already present in the UI]...\n{tail}"


def _compact_prompt_assistant_current_payload(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    keys = [
        "ai_fill_stage",
        "previous_ai_payload",
        "ui_mode",
        "task_type",
        "artist_name",
        "title",
        "caption",
        "tags",
        "negative_tags",
        "duration",
        "bpm",
        "key_scale",
        "time_signature",
        "vocal_language",
        "language",
        "instrumental",
        "song_model",
        "quality_profile",
        "song_intent",
        "ace_lm_model",
        "planner_lm_provider",
        "planner_model",
        "lyrics",
        "concept",
        "num_tracks",
        "track_duration",
        "duration_mode",
        "tracks",
        "album_agent_genre_prompt",
        "album_agent_mood_vibe",
        "album_agent_vocal_type",
        "agent_engine",
    ]
    compact: dict[str, Any] = {}
    for key in keys:
        if key not in source:
            continue
        value = source.get(key)
        if key == "lyrics":
            compact[key] = _compact_text_for_prompt(value, 1600)
        elif isinstance(value, (dict, list)):
            text = json.dumps(_jsonable(value), ensure_ascii=False)
            compact[key] = _jsonable(value) if len(text) <= 2400 else _compact_text_for_prompt(text, 2400)
        else:
            compact[key] = value
    compact["assistant_mode"] = mode
    return compact


def _prompt_assistant_structured_system_prompt(system_prompt: str, mode: str) -> str:
    schema = json.dumps(_prompt_assistant_structured_schema(), ensure_ascii=False)
    return (
        f"{system_prompt}\n\n"
        "STRICT STRUCTURED OUTPUT OVERRIDE:\n"
        "Return one valid JSON object only. Do not return ACEJAM_PASTE_BLOCKS markers, markdown, or prose.\n"
        "The top-level object must match this schema and place the MLX Media payload inside `payload`.\n"
        "Use `warnings` only for short user-facing caveats. MLX Media will build paste blocks server-side.\n"
        f"JSON schema:\n{schema}\n"
    )


def _prompt_assistant_user_content(user_prompt: str, current_payload: dict[str, Any], mode: str) -> str:
    compact_payload = _compact_prompt_assistant_current_payload(current_payload, mode)
    return (
        f"USER REQUEST:\n{str(user_prompt or '').strip()}\n\n"
        "CURRENT ACEJAM UI PAYLOAD JSON, compacted:\n"
        f"{json.dumps(_jsonable(compact_payload), ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Return JSON only with top-level key `payload`. Keep lyrics complete when the user supplied lyrics. "
        "When making or changing a song, include `song_intent` so the Song Intent Builder UI can show the filled "
        "genres, subgenres, style tags, rhythm, instruments, vocals, arrangement, production, negatives, task mode, "
        "model strategy, and personalization choices."
    )


def _prompt_assistant_stage_specs(mode: str) -> list[tuple[str, str]]:
    mode = _prompt_assistant_mode(mode)
    if mode == "album":
        return []
    if mode in PROMPT_KIT_SOURCE_AUDIO_MODES:
        return [
            (
                "source_intent",
                "Choose the source-audio task intent and visible Song Intent Builder fields. Return song_intent with "
                "genre/style/rhythm/instrument/vocal/structure/production/negative arrays plus source_audio_mode, "
                "track_name or track_classes when relevant.",
            ),
            (
                "source_render",
                "Finalize the source-audio payload. Preserve previous_ai_payload, add task_type, song_model, source "
                "strength/range/stem settings, caption, negative_tags, and warnings if source audio is required.",
            ),
        ]
    return [
        (
            "song_intent",
            "Choose every visible Song Intent Builder field first. Return payload.song_intent with scalar fields "
            "genre_family, subgenre, mood, energy, vocal_type, language, drum_groove, bass_low_end, melodic_identity, "
            "texture_space, mix_master, task_mode, model_strategy, source_audio_mode, plus arrays genre_modules, "
            "style_tags, rhythm_tags, instrument_tags, vocal_tags, structure_tags, production_tags, negative_tags. "
            "Also return top-level caption/tags/negative_tags. Do not write full lyrics yet.",
        ),
        (
            "song_writing",
            "Use previous_ai_payload as locked sonic direction. Return title, artist_name, complete lyrics, duration, "
            "bpm, key_scale, time_signature, vocal_language, and repeat the same song_intent so the song stays coherent.",
        ),
        (
            "song_render",
            "Finalize the MLX Media payload. Preserve previous lyrics and song_intent; add task_type, song_model, "
            "quality_profile/model_strategy, inference-safe defaults, seed if useful, and any source or personalization "
            "fields requested by the user.",
        ),
    ]


def _prompt_assistant_stage_system_prompt(system_prompt: str, mode: str, stage_id: str, instruction: str, index: int, total: int) -> str:
    return (
        f"{system_prompt}\n\n"
        f"MULTI-PASS AI FILL STAGE {index + 1}/{total}: {stage_id}\n"
        f"{instruction}\n"
        "The compact current payload may contain `previous_ai_payload`; treat it as the canonical prior decision set. "
        "Do not contradict earlier genre, mood, instrumentation, language, title, or lyric choices unless the user asked for a change. "
        "Return only the fields this stage can improve, but include enough repeated context that later stages remain consistent."
    )


def _merge_prompt_stage_payload(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (update or {}).items():
        if key in {"previous_ai_payload", "ai_fill_stage"}:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list):
            if not value:
                continue
            existing = merged.get(key)
            if isinstance(existing, list):
                merged[key] = _dedupe_prompt_values(existing + value, limit=200)
            else:
                merged[key] = value
            continue
        if isinstance(value, dict):
            if not value:
                continue
            existing = merged.get(key)
            merged[key] = _merge_prompt_stage_payload(existing if isinstance(existing, dict) else {}, value)
            continue
        merged[key] = value
    return merged


def _dedupe_prompt_values(values: Any, limit: int = 32) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values if isinstance(values, list) else [values]:
        if value is None:
            continue
        if isinstance(value, list):
            for item in _dedupe_prompt_values(value, limit=limit):
                key = json.dumps(_jsonable(item), ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    result.append(item)
            continue
        if not isinstance(value, (dict, list)) and not str(value).strip():
            continue
        key = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
        if len(result) >= limit:
            break
    return result


@lru_cache(maxsize=1)
def _intent_schema_groups() -> dict[str, list[Any]]:
    try:
        groups = toolkit_payload().get("song_intent_schema", {}).get("groups", {})
    except Exception:
        groups = {}
    return groups if isinstance(groups, dict) else {}


def _intent_option_value(option: Any) -> str:
    if isinstance(option, dict):
        return str(option.get("value") or "").strip()
    return str(option or "").strip()


def _intent_option_terms(option: Any) -> list[str]:
    if isinstance(option, dict):
        terms: list[Any] = [
            option.get("value"),
            option.get("label"),
            option.get("description"),
            option.get("aliases"),
            option.get("tags"),
        ]
    else:
        terms = [option]
    flat: list[str] = []
    for term in terms:
        if isinstance(term, list):
            flat.extend(str(item).strip() for item in term if str(item).strip())
        elif str(term or "").strip():
            flat.append(str(term).strip())
    return _dedupe_prompt_values(flat, limit=40)


def _intent_match_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _match_intent_group_terms(text: str, group: str, *, limit: int = 8) -> list[str]:
    haystack = f" {_intent_match_key(text)} "
    if not haystack.strip():
        return []
    matches: list[str] = []
    for option in _intent_schema_groups().get(group, []) or []:
        option_value = _intent_option_value(option)
        if not option_value:
            continue
        for term in _intent_option_terms(option):
            key = _intent_match_key(term)
            if key and (f" {key} " in haystack or (len(key) > 5 and key in haystack)):
                matches.append(option_value)
                break
        if len(matches) >= limit:
            break
    return _dedupe_prompt_values(matches, limit=limit)


def _prompt_values_from_any(value: Any) -> list[str]:
    if isinstance(value, list):
        return _dedupe_prompt_values([item for entry in value for item in _prompt_values_from_any(entry)], limit=200)
    if isinstance(value, dict):
        return _dedupe_prompt_values([item for entry in value.values() for item in _prompt_values_from_any(entry)], limit=200)
    return split_terms(value)


def _first_prompt_value(*values: Any) -> str:
    for value in values:
        for item in _prompt_values_from_any(value):
            text = str(item or "").strip()
            if text:
                return text
    return ""


def _prompt_hint_text(payload: dict[str, Any], song_intent: dict[str, Any] | None = None) -> str:
    parts: list[Any] = [
        payload.get("caption"),
        payload.get("tags"),
        payload.get("negative_tags"),
        payload.get("genre_profile"),
        payload.get("genre_modules"),
        payload.get("description"),
        payload.get("concept"),
        payload.get("title"),
    ]
    if isinstance(song_intent, dict):
        parts.append(song_intent)
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, (dict, list)):
            text_parts.append(json.dumps(_jsonable(part), ensure_ascii=False))
        elif str(part or "").strip():
            text_parts.append(str(part).strip())
    return " ".join(text_parts)


def _ensure_song_intent_payload(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    existing = payload.get("song_intent") if isinstance(payload.get("song_intent"), dict) else {}
    song_intent = dict(existing or {})
    hint = _prompt_hint_text(payload, song_intent)
    caption = str(song_intent.get("caption") or payload.get("caption") or payload.get("tags") or "").strip()
    inferred_modules = [module.get("slug") for module in infer_genre_modules(hint or caption, max_modules=3)]
    genre_modules = _dedupe_prompt_values(
        [
            song_intent.get("genre_modules"),
            payload.get("genre_modules"),
            song_intent.get("genre_family"),
            payload.get("genre_family"),
            inferred_modules,
        ],
        limit=8,
    )
    style_tags = _dedupe_prompt_values(
        [
            song_intent.get("style_tags"),
            song_intent.get("subgenre"),
            payload.get("subgenre"),
            _match_intent_group_terms(hint, "genre_style", limit=8),
            _match_intent_group_terms(hint, "era_reference", limit=4),
        ],
        limit=20,
    )
    rhythm_tags = _dedupe_prompt_values(
        [
            song_intent.get("rhythm_tags"),
            song_intent.get("energy"),
            song_intent.get("drum_groove"),
            payload.get("energy"),
            payload.get("drum_groove"),
            payload.get("drums"),
            _match_intent_group_terms(hint, "speed_rhythm", limit=6),
            _match_intent_group_terms(hint, "drums_groove", limit=8),
        ],
        limit=20,
    )
    instrument_tags = _dedupe_prompt_values(
        [
            song_intent.get("instrument_tags"),
            song_intent.get("bass_low_end"),
            song_intent.get("melodic_identity"),
            payload.get("bass_low_end"),
            payload.get("bass"),
            payload.get("melodic_identity"),
            payload.get("melodic_element"),
            _match_intent_group_terms(hint, "bass_low_end", limit=6),
            _match_intent_group_terms(hint, "melodic_identity", limit=8),
            _match_intent_group_terms(hint, "instruments", limit=10),
            _match_intent_group_terms(hint, "stems", limit=6),
        ],
        limit=28,
    )
    vocal_tags = _dedupe_prompt_values(
        [
            song_intent.get("vocal_tags"),
            song_intent.get("vocal_type"),
            payload.get("vocal_type"),
            payload.get("vocal_delivery"),
            _match_intent_group_terms(hint, "vocal_character", limit=8),
        ],
        limit=16,
    )
    structure_tags = _dedupe_prompt_values(
        [
            song_intent.get("structure_tags"),
            payload.get("structure_tags"),
            _match_intent_group_terms(hint, "structure_hints", limit=8),
            _match_intent_group_terms(hint, "lyric_meta_tags", limit=8),
        ],
        limit=20,
    )
    production_tags = _dedupe_prompt_values(
        [
            song_intent.get("production_tags"),
            song_intent.get("texture_space"),
            song_intent.get("mix_master"),
            payload.get("texture_space"),
            payload.get("texture"),
            payload.get("mix_master"),
            payload.get("mix"),
            _match_intent_group_terms(hint, "timbre_texture", limit=8),
            _match_intent_group_terms(hint, "production_style", limit=8),
        ],
        limit=20,
    )
    negative_tags = _dedupe_prompt_values(
        [
            song_intent.get("negative_tags"),
            payload.get("negative_tags"),
            _match_intent_group_terms(str(payload.get("negative_tags") or ""), "negative_control", limit=12),
        ],
        limit=24,
    )
    custom_tags = _dedupe_prompt_values([song_intent.get("custom_tags"), payload.get("custom_tags")], limit=24)
    song_intent.update(
        {
            "genre_family": _first_prompt_value(song_intent.get("genre_family"), payload.get("genre_family"), genre_modules),
            "subgenre": _first_prompt_value(song_intent.get("subgenre"), payload.get("subgenre"), style_tags),
            "mood": _first_prompt_value(song_intent.get("mood"), payload.get("mood"), _match_intent_group_terms(hint, "mood_atmosphere", limit=4)),
            "energy": _first_prompt_value(song_intent.get("energy"), payload.get("energy"), rhythm_tags),
            "vocal_type": _first_prompt_value(song_intent.get("vocal_type"), payload.get("vocal_type"), payload.get("vocal_delivery"), vocal_tags),
            "language": _first_prompt_value(song_intent.get("language"), payload.get("vocal_language"), payload.get("language")) or "en",
            "drum_groove": _first_prompt_value(song_intent.get("drum_groove"), payload.get("drum_groove"), payload.get("drums"), rhythm_tags),
            "bass_low_end": _first_prompt_value(song_intent.get("bass_low_end"), payload.get("bass_low_end"), payload.get("bass"), instrument_tags),
            "melodic_identity": _first_prompt_value(song_intent.get("melodic_identity"), payload.get("melodic_identity"), payload.get("melodic_element"), instrument_tags),
            "texture_space": _first_prompt_value(song_intent.get("texture_space"), payload.get("texture_space"), payload.get("texture"), production_tags),
            "mix_master": _first_prompt_value(song_intent.get("mix_master"), payload.get("mix_master"), payload.get("mix"), production_tags),
            "custom_tags": custom_tags,
            "genre_modules": genre_modules,
            "style_tags": style_tags,
            "rhythm_tags": rhythm_tags,
            "instrument_tags": instrument_tags,
            "vocal_tags": vocal_tags,
            "structure_tags": structure_tags,
            "production_tags": production_tags,
            "negative_tags": negative_tags,
            "task_mode": _first_prompt_value(song_intent.get("task_mode"), payload.get("task_type")) or _prompt_mode_task_type(mode),
            "model_strategy": _first_prompt_value(song_intent.get("model_strategy"), payload.get("quality_profile")) or "auto",
            "source_audio_mode": _first_prompt_value(song_intent.get("source_audio_mode"), payload.get("source_audio_mode")) or ("src_audio" if mode in PROMPT_KIT_SOURCE_AUDIO_MODES else "none"),
            "track_name": _first_prompt_value(song_intent.get("track_name"), payload.get("track_name")),
            "track_classes": _dedupe_prompt_values([song_intent.get("track_classes"), payload.get("track_classes"), payload.get("track_names")], limit=12),
            "caption": caption,
        }
    )
    if isinstance(song_intent.get("ace_step_controls"), dict):
        ace_controls = dict(song_intent["ace_step_controls"])
    else:
        ace_controls = {}
    if payload.get("song_model"):
        ace_controls.setdefault("render_model", payload.get("song_model"))
    if payload.get("ace_lm_model"):
        ace_controls.setdefault("lm_model", payload.get("ace_lm_model"))
    if ace_controls:
        song_intent["ace_step_controls"] = ace_controls
    if not caption:
        caption = ", ".join(_dedupe_prompt_values([style_tags, rhythm_tags, instrument_tags, vocal_tags, production_tags], limit=18))
        song_intent["caption"] = caption
    if caption:
        payload.setdefault("caption", caption)
        payload.setdefault("tags", caption)
    payload["song_intent"] = song_intent
    return payload


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


ALBUM_DURATION_MODE_AI = "ai_per_track"
ALBUM_DURATION_MODE_FIXED = "fixed"


def _normalize_album_duration_mode(value: Any) -> str:
    return ALBUM_DURATION_MODE_FIXED if str(value or "").strip().lower() == ALBUM_DURATION_MODE_FIXED else ALBUM_DURATION_MODE_AI


def _album_track_role_hint(track: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("role", "album_arc_role", "title", "description", "narrative", "caption", "tags"):
        value = track.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif value not in (None, "", []):
            parts.append(str(value))
    return " ".join(parts).lower()


def _album_default_duration_for_role(track: dict[str, Any], fallback: float) -> float:
    text = _album_track_role_hint(track)
    if any(token in text for token in ("intro", "outro", "skit", "interlude", "breather")):
        default = 90.0
    elif any(token in text for token in ("extended", "epic", "cinematic")):
        default = 270.0
    elif any(token in text for token in ("single", "full_song", "full song", "opener", "climax", "closer")):
        default = 210.0
    else:
        default = fallback or 180.0
    return clamp_float(default, 180.0, 30.0, 600.0)


def _clean_album_caption_metadata(value: Any) -> str:
    terms: list[str] = []
    for raw in split_terms(value):
        term = str(raw or "").strip()
        if not term:
            continue
        if re.search(
            r"\b(?:\d{2,3}\s*bpm|bpm\s*[:=]|\d+\/\d+|"
            r"[A-G](?:#|b|♯|♭)?\s+(?:major|minor)|"
            r"time\s*signature|duration|seconds?|minutes?)\b",
            term,
            re.I,
        ):
            continue
        key = term.lower()
        if key not in {existing.lower() for existing in terms}:
            terms.append(term)
    return ", ".join(terms)[:ACE_STEP_CAPTION_CHAR_LIMIT].strip(" ,.")


def _normalize_album_track_durations(
    tracks: list[Any],
    fallback_duration: float,
    duration_mode: str,
) -> list[dict[str, Any]]:
    fallback = parse_duration_seconds(fallback_duration or 180.0, 180.0)
    fixed = _normalize_album_duration_mode(duration_mode) == ALBUM_DURATION_MODE_FIXED
    normalized_tracks: list[dict[str, Any]] = []
    for index, raw in enumerate(tracks or []):
        if not isinstance(raw, dict):
            continue
        track = dict(raw)
        track.setdefault("track_number", index + 1)
        explicit = track.get("duration")
        if fixed:
            duration = fallback
            track["duration_source"] = "fixed_duration_mode"
        elif explicit not in (None, "", []):
            duration = parse_duration_seconds(explicit, fallback)
            track["duration_source"] = track.get("duration_source") or "ai_per_track"
        else:
            duration = _album_default_duration_for_role(track, fallback)
            track["duration_source"] = "role_default"
        track["duration"] = clamp_float(duration, fallback, 30.0, 600.0)
        caption = _clean_album_caption_metadata(track.get("caption") or track.get("tags") or "")
        if caption:
            track["caption"] = caption
            if track.get("tags"):
                track["tags"] = caption
        normalized_tracks.append(track)
    return normalized_tracks


def _normalize_prompt_assistant_payload(mode: str, payload: dict[str, Any], body: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(payload or {})
    warnings: list[str] = []
    mode = _prompt_assistant_mode(mode)
    normalized.setdefault("ui_mode", mode)
    planner_provider = _writer_provider_from_payload({**normalized, **body})
    planner_model = str(
        body.get("planner_model")
        or body.get("planner_ollama_model")
        or body.get("ollama_model")
        or normalized.get("planner_model")
        or normalized.get("planner_ollama_model")
        or ""
    ).strip()
    body_ace_lm = str(body.get("ace_lm_model") or "").strip()
    if planner_provider == "ace_step_lm":
        raw_ace_lm = str(get_param(normalized, "ace_lm_model", "") or body_ace_lm or "").strip()
        normalized["ace_lm_model"] = _requested_ace_lm_model({"ace_lm_model": raw_ace_lm or "auto"})
        planner_model = planner_model or normalized["ace_lm_model"]
    else:
        normalized["ace_lm_model"] = _requested_ace_lm_model({"ace_lm_model": body_ace_lm}) if body_ace_lm else "none"
    normalized["planner_lm_provider"] = planner_provider
    normalized["planner_model"] = planner_model
    normalized.update(
        planner_llm_settings_from_payload(
            {**normalized, **body},
            default_max_tokens=2048,
            default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
        )
    )
    if planner_provider == "ollama":
        normalized["planner_ollama_model"] = planner_model
    else:
        normalized.pop("planner_ollama_model", None)
    if parse_bool(normalized.get("auto_score"), False) or parse_bool(normalized.get("auto_lrc"), False):
        reason = "because official ACE-Step LM generation cannot use the in-process tensor cache" if normalized["ace_lm_model"] != "none" else "because AI Fill should not enable post-render automation without user review"
        warnings.append(f"Auto score/LRC were turned off {reason}.")
    normalized["auto_score"] = False
    normalized["auto_lrc"] = False
    quality_profile = _default_quality_profile_for_payload(normalized, normalized.get("task_type"))
    normalized["quality_profile"] = quality_profile

    if mode == "image":
        action = str(normalized.get("action") or "generate").strip().lower()
        if action not in {"generate", "edit", "inpaint", "upscale", "depth"}:
            warnings.append(f"Unsupported image action '{action}' was changed to generate.")
            action = "generate"
        normalized["action"] = action
        normalized.setdefault("prompt", normalized.get("description") or normalized.get("image_prompt") or "")
        normalized.setdefault("model_id", "qwen-image")
        normalized["width"] = clamp_int(normalized.get("width"), 1024, 256, 2048)
        normalized["height"] = clamp_int(normalized.get("height"), 1024, 256, 2048)
        normalized["steps"] = clamp_int(normalized.get("steps"), 30, 1, 80)
        normalized.setdefault("seed", -1)
        normalized["strength"] = clamp_float(normalized.get("strength"), 0.55, 0.0, 1.0)
        normalized["upscale_factor"] = clamp_int(normalized.get("upscale_factor"), 2, 2, 4)
        if not isinstance(normalized.get("lora_adapters"), list):
            normalized["lora_adapters"] = []
        normalized.setdefault("negative_prompt", "text, watermark, logo, distorted hands, low quality")
        normalized.setdefault("prompt_kit_version", PROMPT_KIT_VERSION)
        normalized.setdefault("planner_lm_provider", planner_provider)
        normalized.setdefault("planner_model", planner_model)
        return normalized, warnings

    if mode == "video":
        action = str(normalized.get("action") or "t2v").strip().lower()
        if action not in {"t2v", "i2v", "a2v", "song_video", "final"}:
            warnings.append(f"Unsupported video action '{action}' was changed to t2v.")
            action = "t2v"
        normalized["action"] = action
        normalized.setdefault("prompt", normalized.get("description") or normalized.get("video_prompt") or "")
        normalized.setdefault("model_id", "ltx2-fast-draft")
        normalized["width"] = clamp_int(normalized.get("width"), 512, 256, 1280)
        normalized["height"] = clamp_int(normalized.get("height"), 320, 192, 768)
        normalized["num_frames"] = clamp_int(normalized.get("num_frames") or normalized.get("frames"), 33, 9, 161)
        normalized["fps"] = clamp_int(normalized.get("fps"), 24, 8, 60)
        normalized["steps"] = clamp_int(normalized.get("steps"), 8, 1, 80)
        normalized.setdefault("seed", -1)
        normalized.setdefault("guide_scale", "")
        normalized.setdefault("shift", "")
        normalized["enhance_prompt"] = parse_bool(normalized.get("enhance_prompt"), False)
        normalized["spatial_upscaler"] = str(normalized.get("spatial_upscaler") or "").strip()
        normalized["tiling"] = parse_bool(normalized.get("tiling"), False)
        normalized["audio_policy"] = "replace_with_source" if action == "song_video" else str(normalized.get("audio_policy") or "none")
        normalized["mux_audio"] = action == "song_video"
        if not isinstance(normalized.get("lora_adapters"), list):
            normalized["lora_adapters"] = []
        normalized.setdefault("prompt_kit_version", PROMPT_KIT_VERSION)
        normalized.setdefault("planner_lm_provider", planner_provider)
        normalized.setdefault("planner_model", planner_model)
        return normalized, warnings

    if mode == "album":
        if planner_provider == "ace_step_lm":
            warnings.append("Album agents ignore ACE-Step 5Hz LM; switch Writer/Planner to Ollama or LM Studio for album planning.")
            planner_provider = "ollama"
            normalized["planner_lm_provider"] = "ollama"
            normalized["planner_model"] = str(body.get("ollama_model") or body.get("planner_ollama_model") or normalized.get("planner_ollama_model") or "")
        normalized = _album_ace_lm_disabled_payload(normalized)
        normalized.setdefault("song_model_strategy", "single_model_album")
        normalized.setdefault("song_model", ALBUM_FINAL_MODEL)
        normalized.setdefault("final_song_model", ALBUM_FINAL_MODEL)
        contract_source = _album_contract_source_from_payload(normalized, body)
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
            contract_count = int(user_album_contract.get("track_count") or 0)
            contract_tracks = tracks_from_user_album_contract(user_album_contract)
            normalized["num_tracks"] = max(
                int(normalized.get("num_tracks") or body.get("num_tracks") or 0),
                contract_count,
                len(contract_tracks),
                1,
            )
            primary_prompt = next(
                (
                    str(normalized.get(key) or body.get(key) or "").strip()
                    for key in ("raw_user_prompt", "user_prompt", "prompt")
                    if str(normalized.get(key) or body.get(key) or "").strip()
                ),
                "",
            )
            if contract_tracks and primary_prompt:
                normalized["tracks"] = contract_tracks
        album_defaults = quality_profile_model_settings(ALBUM_FINAL_MODEL, quality_profile)
        normalized["audio_format"] = album_defaults["audio_format"]
        normalized["inference_steps"] = album_defaults["inference_steps"]
        normalized["guidance_scale"] = album_defaults["guidance_scale"]
        normalized["shift"] = album_defaults["shift"]
        normalized["infer_method"] = album_defaults["infer_method"]
        normalized["sampler_mode"] = album_defaults["sampler_mode"]
        for field, value in DOCS_BEST_LM_DEFAULTS.items():
            if field != "ace_lm_model" and field not in ACE_LM_DISABLED_DEFAULTS:
                normalized[field] = value
        normalized = _album_ace_lm_disabled_payload(normalized)
        normalized["thinking"] = False
        normalized["use_format"] = False
        normalized["use_cot_lyrics"] = False
        normalized["album_writer_mode"] = "per_track_writer_loop"
        normalized["max_track_repair_rounds"] = clamp_int(
            normalized.get("max_track_repair_rounds") or body.get("max_track_repair_rounds"),
            3,
            0,
            3,
        )
        duration_mode = _normalize_album_duration_mode(normalized.get("duration_mode") or body.get("duration_mode"))
        fallback_duration = parse_duration_seconds(
            normalized.get("track_duration") or body.get("track_duration") or normalized.get("duration") or body.get("duration") or 180,
            180,
        )
        normalized["duration_mode"] = duration_mode
        normalized["track_duration"] = clamp_float(fallback_duration, 180.0, 30.0, 600.0)
        normalized.setdefault("track_variants", 1)
        normalized.setdefault("save_to_library", True)
        tracks = normalized.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            contract_tracks = tracks_from_user_album_contract(user_album_contract)
            normalized["tracks"] = _normalize_album_track_durations(contract_tracks, normalized["track_duration"], duration_mode) if contract_tracks else []
            if not contract_tracks:
                warnings.append("Album prompt did not return tracks; use Plan Album or ask again with more detail.")
        else:
            contracted_tracks = apply_user_album_contract_to_tracks(tracks, user_album_contract)
            normalized["tracks"] = _normalize_album_track_durations(contracted_tracks, normalized["track_duration"], duration_mode)
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
                track["ace_lm_model"] = "none"
                track["quality_profile"] = quality_profile
                track["thinking"] = False
                track["use_format"] = False
                track["use_cot_lyrics"] = False
                track["use_cot_metas"] = False
                track["use_cot_caption"] = False
                track["use_cot_language"] = False
                track["use_official_lm"] = False
                track["allow_supplied_lyrics_lm"] = False
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
    song_intent = normalized.get("song_intent") if isinstance(normalized.get("song_intent"), dict) else {}
    if song_intent:
        intent_caption = str(song_intent.get("caption") or "").strip()
        if intent_caption:
            normalized["caption"] = intent_caption
        else:
            song_intent["caption"] = str(normalized.get("caption") or "").strip()
    else:
        custom_tags = normalized.get("custom_tags")
        if isinstance(custom_tags, str):
            custom_tag_list = [item.strip() for item in re.split(r"[,;|]", custom_tags) if item.strip()]
        elif isinstance(custom_tags, list):
            custom_tag_list = [str(item).strip() for item in custom_tags if str(item).strip()]
        else:
            custom_tag_list = []
        normalized["song_intent"] = {
            "genre_family": str(normalized.get("genre_family") or "").strip(),
            "subgenre": str(normalized.get("subgenre") or "").strip(),
            "mood": str(normalized.get("mood") or "").strip(),
            "energy": str(normalized.get("energy") or "").strip(),
            "vocal_type": str(normalized.get("vocal_type") or normalized.get("vocal_delivery") or "").strip(),
            "drum_groove": str(normalized.get("drum_groove") or normalized.get("drums") or "").strip(),
            "bass_low_end": str(normalized.get("bass_low_end") or normalized.get("bass") or "").strip(),
            "melodic_identity": str(normalized.get("melodic_identity") or normalized.get("melodic_element") or "").strip(),
            "texture_space": str(normalized.get("texture_space") or normalized.get("texture") or "").strip(),
            "mix_master": str(normalized.get("mix_master") or normalized.get("mix") or "").strip(),
            "custom_tags": custom_tag_list,
            "caption": str(normalized.get("caption") or tags_text or "").strip(),
        }
    normalized.setdefault(
        "negative_tags",
        "muddy mix, generic lyrics, weak hook, empty lyrics, off-key vocal, unclear vocal, noisy artifacts, flat drums, contradictory style",
    )
    instrumental = parse_bool(normalized.get("instrumental"), False)
    normalized["instrumental"] = instrumental
    if instrumental and not str(normalized.get("lyrics") or "").strip():
        normalized["lyrics"] = "[Instrumental]"
    normalized.setdefault("lyrics", "")
    normalized.setdefault("duration", "auto")
    normalized.setdefault("bpm", "auto")
    normalized.setdefault("key_scale", "auto")
    normalized.setdefault("time_signature", "auto")
    normalized.setdefault("vocal_language", "unknown")
    normalized["metadata_locks"] = _effective_metadata_locks(normalized)
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
    _ensure_song_intent_payload(normalized, mode)
    normalized.setdefault("save_to_library", True)
    if mode in {"cover", "repaint", "extract", "lego", "complete"} and not (
        normalized.get("src_audio_id") or normalized.get("src_result_id") or normalized.get("audio_code_string")
    ):
        warnings.append(f"{mode} needs source audio selected/uploaded in MLX Media before generation.")
    return normalized, warnings


def _run_prompt_assistant_local(
    system_prompt: str,
    user_prompt: str,
    planner_provider: str,
    planner_model: str,
    current_payload: dict[str, Any],
    planner_llm_settings: dict[str, Any] | None = None,
    *,
    mode: str = "custom",
) -> str:
    provider = normalize_provider(planner_provider)
    model = str(planner_model or "").strip()
    settings = dict(planner_llm_settings or {})
    if not settings:
        settings = _load_local_llm_settings()
    if not model:
        model = str(settings.get("chat_model") or "").strip()
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
    schema = _prompt_assistant_structured_schema()
    structured_system = _prompt_assistant_structured_system_prompt(system_prompt, mode)
    user_content = _prompt_assistant_user_content(user_prompt, current_payload or {}, mode)
    messages = [
        {"role": "system", "content": structured_system},
        {"role": "user", "content": user_content},
    ]
    last_raw = ""
    last_error = ""
    try:
        for attempt in range(2):
            option_payload = {**settings, **(planner_llm_settings or {})}
            if attempt > 0:
                option_payload.update(
                    {
                        "planner_creativity_preset": "stable",
                        "planner_temperature": min(clamp_float(option_payload.get("planner_temperature"), 0.45, 0.0, 2.0), 0.2),
                        "planner_top_p": min(clamp_float(option_payload.get("planner_top_p"), 0.92, 0.0, 1.0), 0.85),
                        "planner_top_k": min(clamp_int(option_payload.get("planner_top_k"), 40, 0, 200), 20),
                        "planner_repeat_penalty": max(clamp_float(option_payload.get("planner_repeat_penalty"), 1.1, 0.8, 2.0), 1.15),
                        "planner_max_tokens": 8192,
                        "planner_context_length": 32768,
                    }
                )
                compact_current = _compact_prompt_assistant_current_payload(current_payload or {}, mode)
                compact_current.pop("lyrics", None)
                messages = [
                    {"role": "system", "content": structured_system},
                    {
                        "role": "user",
                        "content": (
                            f"USER REQUEST:\n{_compact_text_for_prompt(user_prompt, 1800)}\n\n"
                            "CURRENT ACEJAM UI PAYLOAD JSON, extra-short retry context:\n"
                            f"{json.dumps(_jsonable(compact_current), ensure_ascii=False, separators=(',', ':'))}\n\n"
                            "Return one closed JSON object only with top-level key `payload`."
                        ),
                    },
                ]
            response = local_llm_chat_completion_response(
                provider,
                model,
                messages,
                options=planner_llm_options_for_provider(
                    provider,
                    option_payload,
                    default_max_tokens=8192,
                    default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
                ),
                json_schema=schema,
            )
            last_raw = str(response.get("content") or "")
            if response.get("truncated") or str(response.get("done_reason") or "").lower() == "length":
                last_error = "AI Fill response was truncated by the local LLM."
                continue
            try:
                parsed, _ = _extract_prompt_assistant_json(last_raw, mode)
                payload, _, _ = _unwrap_prompt_assistant_structured_payload(parsed)
                if isinstance(payload, dict) and payload:
                    return last_raw
                last_error = "AI Fill returned an empty structured payload."
            except Exception as parse_exc:
                last_error = str(parse_exc)
                continue
        if "truncated" in last_error.lower() or "not closed" in last_error.lower() or "length" in last_error.lower():
            raise RuntimeError("AI Fill response was truncated; increase Settings > Max output or shorten lyrics.") from None
        preview = last_raw[:600].replace("\n", " ")
        raise RuntimeError(f"AI Fill returned invalid JSON after structured retry: {last_error or 'unknown parse error'}. Preview: {preview}")
    except Exception as exc:
        if provider == "ollama" and _ollama_error_is_missing_model(exc):
            job = _start_ollama_pull(model, reason="AI Fill", kind="chat")
            raise OllamaPullStarted(model, job, f"{model} is missing; pull started for AI Fill.") from exc
        raise


def _run_prompt_assistant_ollama(system_prompt: str, user_prompt: str, ollama_model: str, current_payload: dict[str, Any]) -> str:
    return _run_prompt_assistant_local(system_prompt, user_prompt, "ollama", ollama_model, current_payload)


def _run_prompt_assistant_local_staged(
    system_prompt: str,
    user_prompt: str,
    planner_provider: str,
    planner_model: str,
    current_payload: dict[str, Any],
    planner_llm_settings: dict[str, Any] | None = None,
    *,
    mode: str = "custom",
) -> str:
    stages = _prompt_assistant_stage_specs(mode)
    if not stages:
        return _run_prompt_assistant_local(
            system_prompt,
            user_prompt,
            planner_provider,
            planner_model,
            current_payload,
            planner_llm_settings,
            mode=mode,
        )
    combined_payload = dict(current_payload or {})
    warnings: list[str] = []
    stage_payloads: dict[str, Any] = {}
    total = len(stages)
    for index, (stage_id, instruction) in enumerate(stages):
        stage_current = dict(current_payload or {})
        stage_current["ai_fill_stage"] = stage_id
        stage_current["previous_ai_payload"] = _compact_prompt_assistant_current_payload(combined_payload, mode)
        raw = _run_prompt_assistant_local(
            _prompt_assistant_stage_system_prompt(system_prompt, mode, stage_id, instruction, index, total),
            user_prompt,
            planner_provider,
            planner_model,
            stage_current,
            planner_llm_settings,
            mode=mode,
        )
        try:
            parsed, _ = _extract_prompt_assistant_json(raw, mode)
            stage_payload, stage_warnings, _ = _unwrap_prompt_assistant_structured_payload(parsed)
        except Exception as exc:
            raise PromptAssistantStageError(f"AI Fill stage `{stage_id}` returned invalid JSON: {exc}", raw) from exc
        if not isinstance(stage_payload, dict) or not stage_payload:
            raise PromptAssistantStageError(f"AI Fill stage `{stage_id}` returned an empty payload.", raw)
        combined_payload = _merge_prompt_stage_payload(combined_payload, stage_payload)
        if mode != "album":
            _ensure_song_intent_payload(combined_payload, mode)
        warnings.extend(stage_warnings)
        stage_payloads[stage_id] = _compact_prompt_assistant_current_payload(combined_payload, mode)
    return json.dumps(
        {
            "payload": combined_payload,
            "warnings": _dedupe_prompt_values(warnings, limit=20),
            "stage_payloads": stage_payloads,
            "multi_pass": True,
        },
        ensure_ascii=False,
    )


def _album_wizard_track_dict(track: Any) -> dict[str, Any]:
    """Lift a single album-crew track payload into the JSON-clean dict the
    wizard UI expects. Empty strings beat None so the wizard textareas
    render editable rather than null."""
    if not isinstance(track, dict):
        return {}
    role = str(track.get("role") or "").strip() or "single"
    return {
        "track_number": int(track.get("track_number") or 0),
        "title": str(track.get("title") or "").strip(),
        "role": role,
        "duration": float(track.get("duration") or 0) or 180.0,
        "artist_name": str(track.get("artist_name") or "").strip(),
        "producer_credit": str(track.get("producer_credit") or "").strip(),
        "caption": str(track.get("caption") or track.get("tags") or "").strip(),
        "tags": str(track.get("tags") or track.get("caption") or "").strip(),
        "tag_list": list(track.get("tag_list") or []),
        "negative_tags": str(track.get("negative_tags") or "").strip(),
        "lyrics": str(track.get("lyrics") or "").strip(),
        "lyrics_lines": list(track.get("lyrics_lines") or []),
        "bpm": int(track.get("bpm") or 0) or 120,
        "key_scale": str(track.get("key_scale") or "C major"),
        "time_signature": str(track.get("time_signature") or "4"),
        "vocal_language": str(track.get("vocal_language") or track.get("language") or "en"),
        "instrumental": bool(track.get("instrumental")),
        "song_model": str(track.get("song_model") or "acestep-v15-xl-sft"),
        "quality_profile": str(track.get("quality_profile") or "chart_master"),
        "ace_lm_model": "none",
        "inference_steps": int(track.get("inference_steps") or 50),
        "guidance_scale": float(track.get("guidance_scale") or 8.0),
        "shift": float(track.get("shift") or 1.0),
        "infer_method": str(track.get("infer_method") or "ode"),
        "audio_format": str(track.get("audio_format") or "wav32"),
        "seed": str(track.get("seed") or "-1"),
        "use_random_seed": bool(track.get("use_random_seed", True)),
        "auto_score": bool(track.get("auto_score", False)),
        "auto_lrc": bool(track.get("auto_lrc", False)),
        "return_audio_codes": bool(track.get("return_audio_codes", True)),
        "save_to_library": bool(track.get("save_to_library", True)),
        "production_team": dict(track.get("production_team") or {}),
        "quality_report": dict(track.get("quality_report") or {}),
        "section_plan": list(
            (track.get("quality_report") or {}).get("section_plan")
            or track.get("section_plan")
            or []
        ),
        "hook_promise": str(track.get("hook_promise") or ""),
        "hook_lines": list(track.get("hook_lines") or []),
        "style": str(track.get("style") or ""),
        "style_profile": str(track.get("style_profile") or ""),
        "vibe": str(track.get("vibe") or ""),
        "narrative": str(track.get("narrative") or track.get("description") or ""),
        "description": str(track.get("description") or ""),
        "genre_profile": str(track.get("genre_profile") or ""),
        "genre_direction": str(track.get("genre_direction") or track.get("genre_prompt") or ""),
        "caption_tags": str(track.get("caption_tags") or ""),
        "album_tags": str(track.get("album_tags") or ""),
        "negative_control": str(track.get("negative_control") or ""),
        # Album-level fields the wizard form exposes per track. The crew's
        # _assemble_track populates these from album bible / opts when the
        # per-track agents leave them empty, so the wizard never blanks.
        "mood": str(track.get("mood") or ""),
        "genre": str(track.get("genre") or ""),
        "vocal_type": str(track.get("vocal_type") or ""),
    }


def _album_crew_stdout_log(line: str) -> None:
    """Forward album-crew log lines to stdout with a visible prefix so the
    user can see CrewAI running in real time. CrewAI agents call litellm
    directly (not AceJAM's chat_ollama wrapper), so without this forwarder
    the user only sees the preflight ping and then silence. With it, every
    crew agent call surfaces as `[ALBUM_FILL_CREW] ...` in the terminal."""
    try:
        print(f"[ALBUM_FILL_CREW] {line}", flush=True)
    except Exception:
        pass


def _derive_album_level_fields(
    raw_tracks: list[Any],
    current_payload: dict[str, Any],
    album_bible: dict[str, Any],
) -> dict[str, str]:
    """Resolve the five top-level wizard fields the AlbumWizard hydrate()
    expects. Source priority per field: user's locked input > album_bible
    intake outputs > first-track derived > empty string. Empty strings are
    intentional so the wizard textareas render editable rather than null."""
    payload = current_payload if isinstance(current_payload, dict) else {}
    bible = album_bible if isinstance(album_bible, dict) else {}
    intake = bible.get("intake") if isinstance(bible.get("intake"), dict) else {}
    first_track = next(
        (track for track in (raw_tracks or []) if isinstance(track, dict)),
        {},
    )

    def _first_str(*candidates: Any) -> str:
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)):
                joined = ", ".join(str(item).strip() for item in candidate if str(item).strip())
                if joined:
                    return joined
                continue
            text = str(candidate).strip()
            if text:
                return text
        return ""

    album_mood = _first_str(
        payload.get("album_mood"),
        payload.get("mood_vibe"),
        payload.get("mood"),
        bible.get("mood_vibe"),
        first_track.get("mood"),
        # The Track Concept Agent emits a rich `vibe` line ("Menacing
        # swagger, gritty urban realism, triumphant survival energy") which
        # is exactly what the Sfeer wizard cell wants when the user did not
        # lock an album_mood up front.
        first_track.get("vibe"),
    )
    vocal_type_caption = str(first_track.get("caption") or "").lower()
    vocal_inferred = ""
    if vocal_type_caption:
        # Crew captions consistently say things like "male rap vocal",
        # "female lead", "mixed choir". Pick out the matching span so the
        # wizard's Vocals cell never blanks just because the user did not
        # lock vocal_type up front.
        for keyword in (
            "male rap lead", "female rap lead",
            "male rap vocal", "female rap vocal",
            "male lead vocal", "female lead vocal",
            "male lead", "female lead",
            "male vocal", "female vocal",
            "mixed choir", "choir vocal",
            "rap lead", "rap vocal",
        ):
            if keyword in vocal_type_caption:
                vocal_inferred = keyword
                break
    vocal_type = _first_str(
        payload.get("vocal_type"),
        payload.get("album_vocal_type"),
        payload.get("vocal_lead"),
        bible.get("vocal_type"),
        first_track.get("vocal_type"),
        vocal_inferred,
    )
    genre_prompt = _first_str(
        payload.get("genre_prompt"),
        payload.get("album_genre"),
        payload.get("genre"),
        bible.get("genre_prompt"),
        first_track.get("genre"),
        first_track.get("style"),
    )
    custom_tags = _first_str(
        payload.get("custom_tags"),
        payload.get("tags"),
        first_track.get("caption"),
        first_track.get("tags"),
    )
    style_profile = _first_str(
        payload.get("style_profile"),
        first_track.get("style_profile"),
        first_track.get("genre_profile"),
        first_track.get("style"),
        first_track.get("genre"),
        bible.get("genre_prompt"),
    ) or "auto"
    negative_tags = _first_str(
        payload.get("negative_tags"),
        first_track.get("negative_tags"),
        first_track.get("negative_control"),
    )
    style_guardrails = intake.get("style_guardrails") if isinstance(intake.get("style_guardrails"), list) else []
    motif_words = bible.get("recurring_motifs") if isinstance(bible.get("recurring_motifs"), list) else []

    return {
        "album_mood": album_mood,
        "vocal_type": vocal_type,
        "genre_prompt": genre_prompt,
        "custom_tags": custom_tags,
        "style_profile": style_profile,
        "negative_tags": negative_tags,
        "style_guardrails": [str(item).strip() for item in style_guardrails if str(item).strip()],
        "motif_words": [str(item).strip() for item in motif_words if str(item).strip()],
    }


def _run_prompt_assistant_album_crew(
    body: dict[str, Any],
    user_prompt: str,
    current_payload: dict[str, Any],
    planner_provider: str,
    planner_model: str,
) -> dict[str, Any]:
    """Album wizard AI Fill — the ONLY path for filling the album wizard.
    Routes the request through the CrewAI Micro Tasks director (`plan_album`)
    so each track field gets filled by a specialised CrewAI agent (Topline
    Hook Writer, Tier-1 Lyric Writer, Sonic Tags Engineer, etc.). The legacy
    single-Ollama-call planner is gone; this is the only album-fill code path.

    The user can edit every wizard field after fill; ACE-Step audio render
    happens separately when the user clicks Generate (track-by-track,
    outside the crew). The `track_variants` field controls how many
    variations per track the crew produces — same lever the Custom mode
    uses, defaulting to 1 unless the user requests more."""
    # Force module-current check BEFORE importing plan_album so a live dev
    # server picks up source-level fixes (e.g. the _agent_full_system_prompt
    # recursion fix) without requiring an app restart.
    _ensure_album_agent_modules_current()
    from album_crew import plan_album as _plan_album

    _album_crew_stdout_log("=" * 60)
    _album_crew_stdout_log("Starting CrewAI album wizard fill")
    _album_crew_stdout_log("=" * 60)

    concept_text = (user_prompt or "").strip() or str(current_payload.get("concept") or "").strip()
    if not concept_text:
        _album_crew_stdout_log("REJECTED: empty concept")
        return {
            "success": False,
            "error": "Album concept is empty. Paste an album idea or fill the concept field before AI Fill.",
            "payload": {},
            "warnings": [],
            "raw_text": "",
        }

    current_payload = _album_prepare_contract_request_body(
        {
            **(current_payload or {}),
            **(body or {}),
            "concept": concept_text,
            "user_prompt": user_prompt,
            "prompt": user_prompt,
        },
        fallback_tracks=7,
    )

    try:
        num_tracks = int(current_payload.get("num_tracks") or body.get("num_tracks") or 7)
    except (TypeError, ValueError):
        num_tracks = 7
    num_tracks = max(1, min(20, num_tracks))
    try:
        track_duration = float(current_payload.get("track_duration") or body.get("track_duration") or 180.0)
    except (TypeError, ValueError):
        track_duration = 180.0
    language = str(current_payload.get("language") or body.get("language") or "en").strip() or "en"
    # Variations per track — same lever the Custom mode exposes via batch_size.
    # Wizard accepts either "track_variants" (album-native) or "batch_size" (custom-style).
    try:
        track_variants = int(
            current_payload.get("track_variants")
            or current_payload.get("batch_size")
            or body.get("track_variants")
            or body.get("batch_size")
            or 1
        )
    except (TypeError, ValueError):
        track_variants = 1
    track_variants = max(1, min(8, track_variants))
    global_llm_settings = _load_local_llm_settings()
    embedding_payload = {
        **global_llm_settings,
        **(body if isinstance(body, dict) else {}),
        **(current_payload if isinstance(current_payload, dict) else {}),
    }
    embedding_provider = _embedding_provider_from_payload(
        embedding_payload,
        str(global_llm_settings.get("embedding_provider") or "ollama"),
    )
    embedding_model = str(
        current_payload.get("embedding_model")
        or body.get("embedding_model")
        or global_llm_settings.get("embedding_model")
        or DEFAULT_ALBUM_EMBEDDING_MODEL
    ).strip() or DEFAULT_ALBUM_EMBEDDING_MODEL

    # Build options through the canonical _album_options_from_payload helper
    # so the crew receives installed_models, model portfolio, render defaults,
    # and other context the AceJamAlbumDirector relies on. Bypassing this
    # helper was the root cause of the AI-Fill failure: the crew would try
    # to reach for installed_models / song_model recommendations and KeyError
    # without them. Reusing the helper guarantees plan_album sees the same
    # options shape as /api/album/plan does.
    bridge_payload: dict[str, Any] = {}
    if isinstance(current_payload, dict):
        bridge_payload.update({k: v for k, v in current_payload.items() if k not in {"tracks"}})
    bridge_payload["concept"] = concept_text
    bridge_payload["user_prompt"] = user_prompt
    bridge_payload["num_tracks"] = num_tracks
    bridge_payload["track_duration"] = track_duration
    bridge_payload["language"] = language
    bridge_payload["track_variants"] = track_variants
    bridge_payload["batch_size"] = max(1, min(track_variants, 4))
    bridge_payload["planner_lm_provider"] = planner_provider
    bridge_payload["planner_model"] = planner_model
    bridge_payload["planner_ollama_model"] = planner_model
    bridge_payload["ollama_model"] = planner_model
    bridge_payload["embedding_provider"] = embedding_provider
    bridge_payload["embedding_lm_provider"] = embedding_provider
    bridge_payload["embedding_model"] = embedding_model
    bridge_payload.setdefault("song_model_strategy", "single_model_album")
    bridge_payload.setdefault("final_song_model", "acestep-v15-xl-sft")
    bridge_payload.setdefault("song_model", "acestep-v15-xl-sft")
    bridge_payload.setdefault("duration_mode", "ai_per_track")
    bridge_payload.setdefault("album_writer_mode", "per_track_writer_loop")
    bridge_payload.setdefault("quality_profile", "chart_master")
    # Lift album-level form fields the wizard collects (mood, vocal_type,
    # genre) so the AceJamAlbumDirector can inject them into every per-track
    # agent context. Without this passthrough Track Concept / Tag / Hook /
    # Lyric agents cannot see the user's album-wide direction and the
    # corresponding wizard cells stay blank.
    for album_field in (
        "album_mood", "mood", "mood_vibe",
        "album_vocal_type", "vocal_type", "vocal_lead",
        "album_genre", "genre", "genre_prompt",
        "album_audience", "audience_platform",
        "album_title", "album_concept",
    ):
        if album_field in current_payload and not bridge_payload.get(album_field):
            bridge_payload[album_field] = current_payload[album_field]

    try:
        options = _album_options_from_payload(bridge_payload, song_model=bridge_payload.get("song_model") or "auto")
    except Exception as opts_exc:
        # If the option-builder itself fails (e.g. missing installed models
        # mapping), fall back to the minimum required option set so the wizard
        # still gets a useful error rather than a 500.
        return {
            "success": False,
            "error": f"Album option setup failed: {type(opts_exc).__name__}: {opts_exc}",
            "payload": {},
            "warnings": [],
            "raw_text": "",
        }

    # Force CrewAI Micro engine and copy the variation lever into the options
    # so AceJamAlbumDirector sees it.
    options["agent_engine"] = "crewai_micro"
    options.setdefault("album_writer_mode", "per_track_writer_loop")
    options["track_variants"] = track_variants
    options["batch_size"] = bridge_payload["batch_size"]
    options["planner_lm_provider"] = planner_provider
    options["planner_model"] = planner_model
    options["planner_ollama_model"] = planner_model
    options["embedding_lm_provider"] = embedding_provider
    options["embedding_provider"] = embedding_provider
    options["embedding_model"] = embedding_model
    options["concept"] = concept_text
    options["user_prompt"] = user_prompt

    input_tracks = _json_list(body.get("tracks") or current_payload.get("tracks")) or None

    _album_crew_stdout_log(f"Concept: {concept_text[:160]}")
    _album_crew_stdout_log(
        f"num_tracks={num_tracks}, duration={int(track_duration)}s, variants={track_variants}, "
        f"language={language}, planner={planner_provider}/{planner_model}, "
        f"embedding={embedding_provider}/{embedding_model}"
    )
    if input_tracks:
        _album_crew_stdout_log(f"Locked input tracks from user paste: {len(input_tracks)}")
    _album_crew_stdout_log("Calling album_crew.plan_album with agent_engine=crewai_micro ...")

    logs_buffer: list[str] = []

    def _log_to_buffer_and_stdout(line: str) -> None:
        logs_buffer.append(str(line))
        _album_crew_stdout_log(str(line))

    try:
        result = _plan_album(
            concept=concept_text,
            num_tracks=num_tracks,
            track_duration=track_duration,
            ollama_model=planner_model,
            language=language,
            embedding_model=embedding_model,
            options=options,
            use_crewai=True,
            input_tracks=input_tracks,
            planner_provider=planner_provider,
            embedding_provider=embedding_provider,
            log_callback=_log_to_buffer_and_stdout,
        )
    except Exception as plan_exc:
        # Surface the actual crew failure to the wizard instead of a generic
        # 500. The traceback signals which agent failed (model missing,
        # Ollama down, schema error, etc.) so the user can act on it.
        import traceback

        tb_text = traceback.format_exc(limit=4)
        _album_crew_stdout_log(f"CRASHED: {type(plan_exc).__name__}: {plan_exc}")
        for tb_line in tb_text.splitlines():
            _album_crew_stdout_log(tb_line)
        return {
            "success": False,
            "error": f"CrewAI album fill crashed: {type(plan_exc).__name__}: {plan_exc}",
            "payload": {
                "concept": concept_text,
                "num_tracks": num_tracks,
                "language": language,
                "track_duration": int(track_duration),
                "track_variants": track_variants,
                "agent_engine": "crewai_micro",
                "tracks": [],
            },
            "warnings": [str(line) for line in logs_buffer[-20:]] + [tb_text],
            "raw_text": "",
        }

    raw_tracks = result.get("tracks") or []
    _album_crew_stdout_log(
        f"Crew finished. success={result.get('success', True)}, "
        f"tracks={len(raw_tracks)}, planning_engine={result.get('planning_engine')}, "
        f"crewai_used={result.get('crewai_used')}"
    )
    if not raw_tracks:
        _album_crew_stdout_log("WARNING: crew returned 0 tracks — wizard will show empty.")
        if result.get("error"):
            _album_crew_stdout_log(f"Crew error: {result.get('error')}")
    wizard_tracks = [_album_wizard_track_dict(track) for track in raw_tracks if isinstance(track, dict)]
    # Backfill mood / genre / vocal_type from the underlying raw track + bridge
    # opts in case _album_wizard_track_dict was loaded before those keys were
    # added (Python only hot-reloads album_crew, not app.py — module restart
    # required to pick up the helper change). This belt-and-braces line keeps
    # the wizard fields populated regardless of when the helper was edited.
    for wizard_track, raw_track in zip(wizard_tracks, raw_tracks):
        if isinstance(raw_track, dict):
            wizard_track.setdefault(
                "mood",
                str(
                    raw_track.get("mood")
                    or current_payload.get("album_mood")
                    or current_payload.get("mood")
                    or ""
                ),
            )
            wizard_track.setdefault(
                "genre",
                str(
                    raw_track.get("genre")
                    or current_payload.get("album_genre")
                    or current_payload.get("genre")
                    or ""
                ),
            )
            wizard_track.setdefault(
                "vocal_type",
                str(
                    raw_track.get("vocal_type")
                    or current_payload.get("album_vocal_type")
                    or current_payload.get("vocal_type")
                    or ""
                ),
            )
            # Force-overwrite if the existing value is None/empty AND we have a
            # non-empty source — `setdefault` only fills missing keys, not
            # null/empty ones present from the stale helper.
            for field, sources in (
                ("mood", (raw_track.get("mood"), current_payload.get("album_mood"), current_payload.get("mood"))),
                ("genre", (raw_track.get("genre"), current_payload.get("album_genre"), current_payload.get("genre"))),
                ("vocal_type", (raw_track.get("vocal_type"), current_payload.get("album_vocal_type"), current_payload.get("vocal_type"))),
                ("style_profile", (raw_track.get("style_profile"), raw_track.get("genre_profile"), current_payload.get("style_profile"))),
                ("caption_tags", (raw_track.get("caption_tags"), raw_track.get("tags"), raw_track.get("caption"))),
                ("album_tags", (raw_track.get("album_tags"), current_payload.get("custom_tags"), raw_track.get("caption_tags"))),
                ("negative_control", (raw_track.get("negative_control"), raw_track.get("negative_tags"), current_payload.get("negative_tags"))),
            ):
                if not str(wizard_track.get(field) or "").strip():
                    for candidate in sources:
                        if candidate:
                            wizard_track[field] = str(candidate)
                            break

    album_bible = result.get("album_bible") if isinstance(result.get("album_bible"), dict) else {}
    toolkit_report = result.get("toolkit_report") if isinstance(result.get("toolkit_report"), dict) else {}
    actual_memory = toolkit_report.get("memory") if isinstance(toolkit_report.get("memory"), dict) else {}
    if actual_memory.get("embedding_model"):
        embedding_model = str(actual_memory.get("embedding_model") or embedding_model)
    if actual_memory.get("embedding_provider"):
        embedding_provider = _embedding_provider_from_payload(
            {"embedding_provider": actual_memory.get("embedding_provider")},
            embedding_provider,
        )
    album_level = _derive_album_level_fields(raw_tracks, current_payload, album_bible)
    _album_crew_stdout_log(
        "Wizard hydrate fields: "
        f"album_mood={album_level['album_mood'][:60]!r}, "
        f"vocal_type={album_level['vocal_type'][:60]!r}, "
        f"genre_prompt={album_level['genre_prompt'][:60]!r}, "
        f"custom_tags={album_level['custom_tags'][:60]!r}, "
        f"style_profile={album_level['style_profile'][:60]!r}"
    )

    payload: dict[str, Any] = {
        "concept": str(result.get("concept") or concept_text),
        "album_title": str(result.get("album_title") or ""),
        "num_tracks": int(result.get("num_tracks") or len(wizard_tracks) or num_tracks),
        "language": language,
        "duration_mode": str(options.get("duration_mode") or "ai_per_track"),
        "track_duration": int(track_duration),
        "album_writer_mode": str(options.get("album_writer_mode") or "per_track_writer_loop"),
        "quality_profile": str(options.get("quality_profile") or "chart_master"),
        "song_model_strategy": str(options.get("song_model_strategy") or "single_model_album"),
        "final_song_model": str(options.get("final_song_model") or "acestep-v15-xl-sft"),
        "track_variants": track_variants,
        "batch_size": bridge_payload["batch_size"],
        "planner_lm_provider": planner_provider,
        "planner_model": planner_model,
        "embedding_provider": embedding_provider,
        "embedding_lm_provider": embedding_provider,
        "embedding_model": embedding_model,
        "memory_enabled": bool(actual_memory.get("enabled") or result.get("memory_enabled")),
        "context_chunks": int(actual_memory.get("context_chunks") or result.get("context_chunks") or 0),
        "retrieval_rounds": int(actual_memory.get("retrieval_rounds") or result.get("retrieval_rounds") or 0),
        "agent_context_store": str(
            actual_memory.get("context_store")
            or result.get("agent_context_store")
            or result.get("context_store_index")
            or ""
        ),
        "ace_step_text_encoder": "Qwen3-Embedding-0.6B",
        "ace_lm_model": "none",
        "use_official_lm": False,
        "vocal_language": language,
        "tracks": wizard_tracks,
        "negative_tags": album_level["negative_tags"] or "muddy mix, weak hook, unclear vocal, noisy artifacts, flat drums, boring arrangement, generic lyrics, contradictory style",
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "max_track_repair_rounds": int(options.get("max_track_repair_rounds") or 3),
        "agent_engine": "crewai_micro",
        "planning_engine": "crewai_micro",
        "crewai_used": True,
        # Album-level wizard fields the AlbumWizard hydrate() reads from the
        # response. Derived from user input first, album bible second, first
        # track third — never None so the wizard textareas render editable.
        "album_mood": album_level["album_mood"],
        "vocal_type": album_level["vocal_type"],
        "genre_prompt": album_level["genre_prompt"],
        "custom_tags": album_level["custom_tags"],
        "style_profile": album_level["style_profile"],
        "style_guardrails": album_level["style_guardrails"],
        "motif_words": album_level["motif_words"],
    }

    warnings: list[str] = []
    if not bool(result.get("success", True)):
        warnings.append(str(result.get("error") or "Album crew planning did not complete; partial results may be incomplete."))
    for entry in result.get("warnings") or []:
        warnings.append(str(entry))
    # Surface the crew log lines so the wizard can show progress / which
    # agents ran. Trimmed to keep the response payload compact.
    if logs_buffer:
        warnings.extend(str(line) for line in logs_buffer[-12:])

    return {
        "success": bool(result.get("success", True) and wizard_tracks),
        "payload": payload,
        "warnings": warnings,
        "raw_text": json.dumps({"payload": payload}, ensure_ascii=False, default=str),
    }


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
        child.is_file() and child.stat().st_size > 0 and child.suffix in {".safetensors", ".bin", ".pt", ".ckpt"}
        for child in path.iterdir()
    )


ACE_STEP_SHARED_RUNTIME_COMPONENTS = ("vae", "Qwen3-Embedding-0.6B")


def _diffusers_pipeline_status_reason(path: Path) -> str:
    reasons = diffusers_pipeline_missing_reasons(path)
    if not reasons:
        return "ready"
    return f"{path.name} is missing Diffusers pipeline files: {', '.join(reasons)}"


def _checkpoint_status_reason(path: Path) -> str:
    if not path.exists():
        return f"{path.name} is missing"
    if not path.is_dir():
        return f"{path.name} is not a checkpoint directory"
    if (path / "model_index.json").is_file():
        return _diffusers_pipeline_status_reason(path)
    if not (path / "config.json").is_file():
        return f"{path.name} is missing config.json"
    if not _checkpoint_dir_ready(path):
        return f"{path.name} is missing usable weight files"
    return "ready"


def _song_model_runtime_ready(model_name: str) -> bool:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    if not _checkpoint_dir_ready(checkpoint_dir / model_name):
        return False
    return all(_checkpoint_dir_ready(checkpoint_dir / component) for component in ACE_STEP_SHARED_RUNTIME_COMPONENTS)


def _song_model_runtime_missing_reasons(model_name: str) -> list[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    checks = [model_name, *ACE_STEP_SHARED_RUNTIME_COMPONENTS]
    return [
        _checkpoint_status_reason(checkpoint_dir / name)
        for name in checks
        if not _checkpoint_dir_ready(checkpoint_dir / name)
    ]


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
        if child.name.startswith("acestep-v15-") and not _download_job_active(child.name) and _song_model_runtime_ready(child.name)
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
    if profile == QUALITY_PROFILE_DOCS_DAILY:
        for candidate in ["acestep-v15-xl-turbo", "acestep-v15-turbo"]:
            if candidate in installed:
                return candidate
        return "acestep-v15-xl-turbo"
    if profile == "chart_master":
        for candidate in ["acestep-v15-xl-sft", "acestep-v15-sft"]:
            if candidate in installed:
                return candidate
        return "acestep-v15-xl-sft"
    if profile == "balanced_pro":
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


def _activate_trained_adapter(adapter_path: Path, scale: float = DEFAULT_LORA_GENERATION_SCALE) -> dict[str, Any]:
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
        client = _ollama_client.Client(host=_ollama_host())
        response = client.list()
        for model in response.models:
            try:
                client.generate(model=model.model, prompt="", keep_alive=0)
            except Exception:
                pass
        print("[generation] Unloaded Ollama models to free unified memory.", flush=True)
    except Exception as exc:
        print(f"[generation] Ollama unload skipped: {exc}", flush=True)
    # Unload LM Studio models
    try:
        catalog = lmstudio_model_catalog()
        for model in catalog.get("loaded_models", []):
            try:
                lmstudio_unload_model(str(model))
            except Exception:
                pass
        if catalog.get("loaded_models"):
            print("[generation] Unloaded LM Studio models to free unified memory.", flush=True)
    except Exception:
        pass


EPOCH_AUDITION_INFERENCE_STEPS = 50


def _apply_verified_vocal_audio_backend_defaults(params: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Backwards-compatible wrapper for old callers.

    Earlier builds forced vocal SFT/Base renders onto PyTorch/MPS here. The UI
    now exposes the audio runtime explicitly, with MLX as default and MPS/Torch
    as the fallback.
    """
    return _apply_audio_backend_defaults(params, source=source)


def _lora_epoch_audition_song_model(request: dict[str, Any]) -> str:
    variant = model_to_variant(str(request.get("model_variant") or request.get("song_model") or ""))
    return model_from_variant(variant, normalize_training_song_model(str(request.get("song_model") or "")))


def _run_lora_epoch_audition(request: dict[str, Any]) -> dict[str, Any]:
    epoch = int(request.get("epoch") or 0)
    trigger = str(request.get("trigger_tag") or "LoRA").strip() or "LoRA"
    duration = clamp_int(request.get("duration"), 30, 10, 60)
    request_fit = request.get("lyrics_fit") if isinstance(request.get("lyrics_fit"), dict) else {}
    style_profile = str(request.get("style_profile") or request.get("genre_profile") or "")
    source_lyrics = str(request.get("lyrics") or "")
    if request_fit.get("timed_structure"):
        runtime_lyrics = source_lyrics
        lyrics_fit = dict(request_fit)
    else:
        style_seed = apply_audio_style_conditioning(
            {
                "style_profile": style_profile or "auto",
                "caption": str(request.get("caption") or ""),
                "lyrics": source_lyrics,
            }
        )
        runtime_lyrics, lyrics_fit = fit_epoch_audition_lyrics(str(style_seed.get("lyrics") or source_lyrics), duration=duration)
    vocal_language = str(request.get("vocal_language") or request.get("language") or "en").strip() or "en"
    seed_value = request.get("seed")
    if seed_value in (None, ""):
        seed_value = "42"
    song_model = _lora_epoch_audition_song_model(request)
    defaults = training_inference_defaults(song_model)
    inference_steps = int(defaults["num_inference_steps"])
    shift = float(defaults["training_shift"])
    backend_defaults = _apply_verified_vocal_audio_backend_defaults(
        {"song_model": song_model, "task_type": "text2music", "instrumental": False},
        source="lora_epoch_audition",
    )
    raw_caption = str(request.get("caption") or trigger).strip()
    if duration != 20:
        raw_caption = re.sub(r"\b20-second LoRA audition\b", f"{duration}-second LoRA audition", raw_caption)
    clarity_caption = EPOCH_AUDITION_CLARITY_CAPTION
    if "clear intelligible vocal" in raw_caption.lower():
        caption = raw_caption
    else:
        caption = f"{raw_caption}, {clarity_caption}" if raw_caption else clarity_caption
    raw_payload = {
        "task_type": "text2music",
        "ui_mode": "lora_epoch_audition",
        "artist_name": "MLX Media LoRA",
        "title": f"{trigger} epoch {epoch}",
        "caption": caption,
        "lyrics": runtime_lyrics,
        "style_profile": style_profile,
        "language": vocal_language,
        "vocal_language": vocal_language,
        "lyric_duration_fit": lyrics_fit,
        "duration": duration,
        "bpm": request.get("bpm"),
        "keyscale": str(request.get("keyscale") or ""),
        "timesignature": str(request.get("timesignature") or ""),
        "song_model": song_model,
        "seed": str(seed_value),
        "use_random_seed": False,
        "device": backend_defaults.get("device", "auto"),
        "dtype": backend_defaults.get("dtype", "auto"),
        "audio_backend": backend_defaults.get("audio_backend", _default_audio_backend()),
        "use_mlx_dit": backend_defaults.get("use_mlx_dit", "auto"),
        "inference_steps": inference_steps,
        "shift": shift,
        "batch_size": 1,
        "use_lora": True,
        "lora_adapter_path": str(request.get("checkpoint_path") or ""),
        "lora_adapter_name": str(request.get("lora_adapter_name") or f"{trigger} epoch {epoch}"),
        "lora_scale": request.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE),
        "adapter_model_variant": str(request.get("model_variant") or ""),
        "save_to_library": False,
        "ace_lm_model": "none",
        "allow_supplied_lyrics_lm": False,
        "thinking": False,
        "sample_mode": False,
        "sample_query": "",
        "use_format": False,
        "use_cot_metas": False,
        "use_cot_caption": False,
        "use_cot_lyrics": False,
        "use_cot_language": False,
        "vocal_intelligibility_gate": True,
        "lora_preflight_required": True,
        "auto_score": False,
        "auto_lrc": False,
        "audio_format": "wav",
        "payload_warnings": backend_defaults.get("payload_warnings", []),
    }
    params = _parse_generation_payload(raw_payload)
    params["lora_preflight_required"] = True
    preflight_result = _run_lora_preflight_verifier(params)
    if preflight_result is not None:
        audios = list(preflight_result.get("audios") or [])
        first_audio = audios[0] if audios else {}
        preflight = preflight_result.get("lora_preflight") if isinstance(preflight_result.get("lora_preflight"), dict) else {}
        gate = preflight_result.get("vocal_intelligibility_gate") if isinstance(preflight_result.get("vocal_intelligibility_gate"), dict) else {}
        return {
            "success": False,
            "error": str(preflight_result.get("error") or "LoRA epoch audition preflight failed."),
            "result_id": str(preflight_result.get("result_id") or first_audio.get("result_id") or ""),
            "audio_url": str(first_audio.get("audio_url") or ""),
            "audios": audios,
            "lyrics_fit": lyrics_fit,
            "vocal_intelligibility_gate": gate,
            "lora_preflight": preflight,
            "transcript_preview": _vocal_gate_transcript_preview(gate),
            "song_model": params.get("song_model"),
            "inference_steps": params.get("inference_steps"),
            "shift": params.get("shift"),
            "lora_scale": params.get("lora_scale"),
            "style_profile": params.get("style_profile"),
            "style_caption_tags": params.get("style_caption_tags"),
            "style_lyric_tags_applied": params.get("style_lyric_tags_applied"),
            "style_conditioning_audit": params.get("style_conditioning_audit"),
        }
    result = _run_advanced_generation_once(params)
    gate = _apply_vocal_intelligibility_gate_to_result(result, params, attempt=1, max_attempts=1)
    _annotate_generation_attempt_result(
        result,
        params,
        role="primary",
        gate=gate,
        requested_params=params,
        failure_reason=_attempt_failure_reason(result, gate),
    )
    adapter_path = str(params.get("lora_adapter_path") or "")
    if adapter_path:
        if result.get("success") and gate.get("passed"):
            _update_lora_adapter_quality_metadata(
                adapter_path,
                quality_status="verified",
                reason=f"Epoch {epoch} audition passed the vocal intelligibility gate.",
                recommended_lora_scale=float(params.get("lora_scale") or DEFAULT_LORA_GENERATION_SCALE),
                audition={
                    "status": "succeeded",
                    "type": "epoch_audition",
                    "epoch": epoch,
                    "result_id": result.get("result_id"),
                    "lora_scale": params.get("lora_scale"),
                    "song_model": params.get("song_model"),
                    "inference_steps": params.get("inference_steps"),
                    "shift": params.get("shift"),
                    "vocal_intelligibility_gate": gate,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        else:
            _update_lora_adapter_quality_metadata(
                adapter_path,
                quality_status="failed_audition",
                reason=str(result.get("error") or _attempt_failure_reason(result, gate) or "Epoch audition failed the vocal intelligibility gate."),
                audition={
                    "status": "failed",
                    "type": "epoch_audition",
                    "epoch": epoch,
                    "result_id": result.get("result_id"),
                    "lora_scale": params.get("lora_scale"),
                    "song_model": params.get("song_model"),
                    "inference_steps": params.get("inference_steps"),
                    "shift": params.get("shift"),
                    "vocal_intelligibility_gate": gate,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    audios = list(result.get("audios") or [])
    first_audio = audios[0] if audios else {}
    return {
        "success": bool(result.get("success") and gate.get("passed")),
        "error": str(result.get("error") or ""),
        "result_id": str(result.get("result_id") or first_audio.get("result_id") or ""),
        "audio_url": str(first_audio.get("audio_url") or ""),
        "audios": audios,
        "lyrics_fit": lyrics_fit,
        "vocal_intelligibility_gate": gate,
        "lora_preflight": params.get("lora_preflight") if isinstance(params.get("lora_preflight"), dict) else {},
        "transcript_preview": _vocal_gate_transcript_preview(gate),
        "song_model": params.get("song_model"),
        "inference_steps": params.get("inference_steps"),
        "shift": params.get("shift"),
        "lora_scale": params.get("lora_scale"),
        "style_profile": params.get("style_profile"),
        "style_caption_tags": params.get("style_caption_tags"),
        "style_lyric_tags_applied": params.get("style_lyric_tags_applied"),
        "style_conditioning_audit": params.get("style_conditioning_audit"),
    }


def _training_understand_music(audio_path: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Run ACE-Step understand_music on a single audio file (called from training thread).

    Kept for legacy callers; the LoRA training pipeline now prefers the much
    cheaper online-lyrics path (`_training_lookup_online_lyrics`) because the
    LM transcribes incorrectly on dense music + adds minutes per file due to
    subprocess + model load.
    """
    with handler_lock:
        _ensure_song_model(body.get("song_model"))
        codes = handler.convert_src_audio_to_codes(str(audio_path))
    return _run_official_lm_aux("understand_music", body, audio_codes=codes)


# ---------------------------------------------------------------------------
# Online lyrics lookup (replaces transcribe-based labeling for LoRA training)
#
# Strategy:
#   1. Read ID3 tags via mutagen for {artist, title} (TPE1/TIT2) — most
#      reliable signal when present.
#   2. Fallback: parse the filename. Patterns supported:
#        "Artist - Title.ext"
#        "Artist - 01 - Title.ext"
#        "Artist - Album - 01 - Title.ext"
#      Numeric tokens are treated as track numbers and stripped.
#   3. Query https://api.lyrics.ovh/v1/{artist}/{title} (free, no key, no rate
#      limit issues for occasional batches). 404 → instrumental fallback.
#   4. Convert "Section: Author" / "Verse One" / etc into ACE-Step section
#      tags `[Verse 1]`, `[Chorus]`, `[Bridge]`. If no section markers exist
#      anywhere, paragraph-split and synthesize basic tags.
# ---------------------------------------------------------------------------

_LYRICS_OVH_BASE = "https://api.lyrics.ovh/v1"
_LYRICS_OVH_TIMEOUT = 12.0
_FILENAME_NUM_RE = re.compile(r"^\d{1,3}$")
_SECTION_HEADER_RE = re.compile(
    r"^\s*\[?(?P<kind>Intro|Verse|Chorus|Pre[-\s]?Chorus|Bridge|Hook|Refrain|Outro|Interlude|Breakdown|Drop|Build|Coda|Tag)"
    r"(?:\s*(?P<num>One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+))?"
    r"\s*\]?\s*(?::\s*(?P<author>[^\n]+))?\s*$",
    re.IGNORECASE,
)
_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_LYRICS_TITLE_ALIASES = {
    "bomb first": ["Bomb First My Second Reply"],
    "bomb first intro": ["Bomb First My Second Reply"],
    "hell razor": ["Hellrazor"],
    "nothin to lose": ["Nothing To Lose"],
    "only fear death": ["Only Fear Of Death"],
    "califonia love": ["California Love"],
}

_LYRICS_WORD_REPLACEMENTS = {
    "califonia": "california",
    "nothin": "nothing",
    "whatz": "whats",
}


def _extract_artist_title_from_id3(audio_path: Path) -> tuple[str, str]:
    try:
        from mutagen import File as MutagenFile  # type: ignore[import-not-found]
    except Exception:
        return ("", "")
    try:
        mf = MutagenFile(str(audio_path), easy=True)
        if mf is None:
            return ("", "")
        artist = ""
        title = ""
        try:
            artist = " / ".join(str(v) for v in (mf.get("artist") or [])).strip()
        except Exception:
            artist = ""
        try:
            title = " / ".join(str(v) for v in (mf.get("title") or [])).strip()
        except Exception:
            title = ""
        return (artist, title)
    except Exception:
        return ("", "")


def _extract_easy_audio_tags(audio_path: Path) -> dict[str, Any]:
    try:
        from mutagen import File as MutagenFile  # type: ignore[import-not-found]
    except Exception:
        return {}
    try:
        mf = MutagenFile(str(audio_path), easy=True)
    except Exception:
        return {}
    if mf is None:
        return {}

    def first_list(key: str) -> list[str]:
        try:
            values = mf.get(key) or []
        except Exception:
            values = []
        return [str(item).strip() for item in values if str(item).strip()]

    tags = {
        "artist": " / ".join(first_list("artist")),
        "title": " / ".join(first_list("title")),
        "album": " / ".join(first_list("album")),
        "date": " / ".join(first_list("date")),
        "genre": first_list("genre"),
    }
    return {key: value for key, value in tags.items() if value}


def _extract_artist_title_from_filename(audio_path: Path) -> tuple[str, str]:
    stem = audio_path.stem
    parts = [p.strip() for p in stem.split(" - ")]
    parts = [p for p in parts if p]
    if not parts:
        return ("", stem)
    if len(parts) == 1:
        return ("", parts[0])
    artist = parts[0]
    rest = [p for p in parts[1:] if not _FILENAME_NUM_RE.match(p)]
    title = rest[-1] if rest else parts[-1]
    return (artist, title)


def _lyrics_lookup_key(value: str) -> str:
    text = (value or "").lower()
    text = text.replace("&", " and ")
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\bfeat(?:uring)?\.?\b.*$", "", text)
    text = re.sub(r"\bft\.?\b.*$", "", text)
    text = re.sub(r"[\(\)\[\]]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_artist_prefix_from_title(artist: str, title: str) -> str:
    cleaned = (title or "").strip()
    if not artist or not cleaned:
        return cleaned
    artist_key = _lyrics_lookup_key(artist)
    title_key = _lyrics_lookup_key(cleaned)
    if not artist_key or not title_key.startswith(artist_key + " "):
        return cleaned
    # Handles bad ID3 titles like "2Pac - Ready 4 Whatever" while preserving
    # normal titles that merely mention the artist later in the text.
    patterns = [
        rf"^\s*{re.escape(artist)}\s*[-–—:]\s*",
        rf"^\s*{re.escape(artist)}\s+",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if stripped and stripped != cleaned:
            return stripped
    parts = cleaned.split(None, 1)
    return parts[1].strip() if len(parts) == 2 else cleaned


def _resolve_audio_artist_title(audio_path: Path) -> tuple[str, str]:
    artist, title = _extract_artist_title_from_id3(audio_path)
    if artist and title:
        return (artist, _strip_artist_prefix_from_title(artist, title))
    fa, ft = _extract_artist_title_from_filename(audio_path)
    resolved_artist = artist or fa
    resolved_title = title or ft
    return (resolved_artist, _strip_artist_prefix_from_title(resolved_artist, resolved_title))


def _training_read_existing_label_sidecars(audio_path: Path) -> dict[str, Any]:
    stem = audio_path.stem
    metadata_path = audio_path.with_name(f"{stem}.json")
    caption_path = audio_path.with_name(f"{stem}.caption.txt")
    lyrics_path = audio_path.with_name(f"{stem}.lyrics.txt")
    legacy_lyrics_path = audio_path.with_suffix(".txt")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                metadata.update(raw)
        except Exception:
            metadata = {}
    if caption_path.is_file() and not str(metadata.get("caption") or "").strip():
        try:
            metadata["caption"] = caption_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pass
    if not str(metadata.get("lyrics") or "").strip():
        for candidate in (lyrics_path, legacy_lyrics_path):
            if not candidate.is_file():
                continue
            try:
                metadata["lyrics"] = candidate.read_text(encoding="utf-8", errors="replace").strip()
                break
            except Exception:
                pass
    return metadata


def _training_split_genre_terms(*values: Any) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("genre", "genres", "style", "style_profile", "caption_tags", "musicbrainz_tags", "id3_genre"):
                add(value.get(key))
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                add(child)
            return
        for part in re.split(r"[,;/\n|]+", str(value or "")):
            cleaned = re.sub(r"\s+", " ", part).strip()
            if not cleaned:
                continue
            key = cleaned.lower().replace("-", " ")
            if key in seen:
                continue
            seen.add(key)
            terms.append(cleaned)

    for item in values:
        add(item)
    return terms


def _training_style_profile_from_terms(*values: Any) -> str:
    haystack = " ".join(_training_split_genre_terms(*values))
    haystack = re.sub(r"[^a-z0-9]+", " ", haystack.lower()).strip()
    if not haystack:
        return ""
    for profile in audio_style_profiles(include_auto=False):
        candidates = [
            profile.get("key"),
            profile.get("label"),
            profile.get("caption_tags"),
            *(profile.get("terms") or []),
        ]
        for candidate in candidates:
            token = re.sub(r"[^a-z0-9]+", " ", str(candidate or "").lower()).strip()
            if token and token in haystack:
                return str(profile.get("key") or "").strip()
    return ""


def _training_style_caption_tags(style_profile: str) -> str:
    wanted = str(style_profile or "").strip().lower()
    if not wanted:
        return ""
    for profile in audio_style_profiles(include_auto=False):
        if str(profile.get("key") or "").strip().lower() == wanted:
            return str(profile.get("caption_tags") or "").strip()
    return ""


def _training_compose_caption(
    *,
    artist: str = "",
    title: str = "",
    bpm: Any = None,
    keyscale: str = "",
    has_vocals: bool = True,
    genre_terms: list[str] | None = None,
    style_profile: str = "",
    caption_tags: str = "",
    fallback_caption: str = "",
) -> str:
    parts: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        for term in _training_split_genre_terms(value):
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(term)

    add(caption_tags)
    add(_training_style_caption_tags(style_profile))
    add(genre_terms or [])
    if bpm:
        add(f"{bpm} BPM")
    if keyscale:
        add(keyscale)
    add("vocals" if has_vocals else "instrumental")
    if not parts:
        add(fallback_caption or "music track")
    return ", ".join(parts)


def _normalize_training_genre_label_mode(value: Any) -> str:
    mode = str(value or "ai_auto").strip().lower().replace("-", "_")
    aliases = {
        "ai": "ai_auto",
        "auto": "ai_auto",
        "ai_per_track": "ai_auto",
        "manual": "manual_global",
        "global": "manual_global",
        "manual_genre": "manual_global",
        "preserve": "metadata_musicbrainz",
        "off": "metadata_musicbrainz",
        "existing": "metadata_musicbrainz",
        "metadata": "metadata_musicbrainz",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"ai_auto", "manual_global", "metadata_musicbrainz"}:
        mode = "ai_auto"
    return mode


def _training_ai_genre_label(
    audio_path: Path,
    *,
    body: dict[str, Any],
    artist: str,
    title: str,
    caption: str,
    lyrics: str,
    bpm: Any,
    keyscale: str,
) -> dict[str, Any]:
    settings = _load_local_llm_settings()
    provider = normalize_provider(
        body.get("genre_label_provider")
        or body.get("planner_lm_provider")
        or settings.get("provider")
        or "ollama"
    )
    if provider not in {"ollama", "lmstudio"}:
        provider = "ollama"
    requested_model = str(
        body.get("genre_label_model")
        or body.get("planner_model")
        or body.get("chat_model")
        or settings.get("chat_model")
        or ""
    ).strip()
    try:
        model = _resolve_local_llm_model_selection(provider, requested_model, "chat", "trainer genre labeling")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "genre": {"type": "string"},
                "style_profile": {"type": "string"},
                "caption_tags": {"type": "string"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["genre", "style_profile", "caption_tags", "confidence", "reason"],
        }
        style_keys = [str(item.get("key") or "") for item in audio_style_profiles(include_auto=False)]
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify one music training sample for ACE-Step LoRA training. "
                    "Use concise music genres and production tags. "
                    f"style_profile must be one of: {', '.join(style_keys)}. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "filename": audio_path.name,
                        "folder": str(audio_path.parent.name),
                        "artist": artist,
                        "title": title,
                        "existing_caption": caption,
                        "lyrics_preview": _compact_text_for_prompt(lyrics, 900),
                        "bpm": bpm,
                        "keyscale": keyscale,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = local_llm_chat_completion_response(
            provider,
            model,
            messages,
            options=planner_llm_options_for_provider(
                provider,
                {**settings, **body, "planner_temperature": 0.1, "planner_max_tokens": 512},
                default_max_tokens=512,
                default_timeout=180.0,
            ),
            json_schema=schema,
        )
        parsed = _loads_json_lenient_object(str(response.get("content") or ""))
        genre_terms = _training_split_genre_terms(parsed.get("genre"))
        profile = _training_style_profile_from_terms(parsed.get("style_profile"), genre_terms) or _training_style_profile_from_terms(parsed.get("caption_tags"))
        if not profile:
            profile = "pop"
        return {
            "genre": ", ".join(genre_terms[:6]),
            "style_profile": profile,
            "caption_tags": str(parsed.get("caption_tags") or "").strip(),
            "genre_label_source": "ai_local_llm",
            "genre_label_provider": provider,
            "genre_label_model": model,
            "genre_confidence": clamp_float(parsed.get("confidence"), 0.55, 0.0, 1.0),
            "genre_reason": str(parsed.get("reason") or "").strip(),
        }
    except OllamaPullStarted as exc:
        return {
            "genre_label_source": "ai_unavailable",
            "genre_label_error": exc.message,
            "genre_label_provider": provider,
            "genre_label_model": requested_model,
        }
    except Exception as exc:
        return {
            "genre_label_source": "ai_unavailable",
            "genre_label_error": str(exc),
            "genre_label_provider": provider,
            "genre_label_model": requested_model,
        }


def _training_genre_label(
    audio_path: Path,
    *,
    body: dict[str, Any],
    artist: str,
    title: str,
    metadata: dict[str, Any],
    id3_tags: dict[str, Any],
    lyrics: str,
    bpm: Any,
    keyscale: str,
) -> dict[str, Any]:
    mode = _normalize_training_genre_label_mode(body.get("genre_label_mode"))
    overwrite = parse_bool(body.get("overwrite_existing_labels"), False)
    caption_seed = str(metadata.get("caption") or "").strip() or " – ".join(bit for bit in [artist, title] if bit).strip() or audio_path.stem

    metadata_terms = _training_split_genre_terms(
        metadata.get("genre"),
        metadata.get("genres"),
    )
    metadata_style_terms = _training_split_genre_terms(
        metadata.get("style_profile"),
        metadata.get("caption_tags"),
    )
    if metadata_terms and not overwrite:
        profile = str(metadata.get("style_profile") or "").strip() or _training_style_profile_from_terms(metadata_terms, metadata_style_terms)
        return {
            "genre": ", ".join(metadata_terms[:6]),
            "style_profile": profile,
            "caption_tags": str(metadata.get("caption_tags") or _training_style_caption_tags(profile) or "").strip(),
            "genre_label_source": "metadata",
            "genre_confidence": clamp_float(metadata.get("genre_confidence"), 1.0, 0.0, 1.0),
            "genre_reason": "Existing sidecar metadata genre preserved.",
        }

    id3_terms = _training_split_genre_terms(id3_tags.get("genre"))
    if id3_terms:
        profile = _training_style_profile_from_terms(id3_terms)
        return {
            "genre": ", ".join(id3_terms[:6]),
            "style_profile": profile,
            "caption_tags": _training_style_caption_tags(profile),
            "id3_genre": ", ".join(id3_terms[:6]),
            "genre_label_source": "id3_metadata",
            "genre_confidence": 0.95,
            "genre_reason": "Genre read from audio metadata.",
        }

    mb_terms = _musicbrainz_artist_tags(artist) if artist else []
    if mb_terms:
        profile = _training_style_profile_from_terms(mb_terms)
        return {
            "genre": ", ".join(mb_terms[:6]),
            "style_profile": profile,
            "caption_tags": _training_style_caption_tags(profile),
            "musicbrainz_tags": mb_terms,
            "genre_label_source": "musicbrainz",
            "genre_confidence": 0.85,
            "genre_reason": "Genre inferred from MusicBrainz artist tags.",
        }

    if mode == "manual_global":
        manual_terms = _training_split_genre_terms(body.get("genre") or body.get("caption_tags"))
        profile = _training_style_profile_from_terms(manual_terms)
        return {
            "genre": ", ".join(manual_terms[:6]),
            "style_profile": profile,
            "caption_tags": _training_style_caption_tags(profile) or ", ".join(manual_terms[:6]),
            "genre_label_source": "manual_global",
            "genre_confidence": 0.75 if manual_terms else 0.0,
            "genre_reason": "Manual global genre used because metadata and MusicBrainz had no genre.",
        }

    if mode == "ai_auto":
        ai = _training_ai_genre_label(
            audio_path,
            body=body,
            artist=artist,
            title=title,
            caption=caption_seed,
            lyrics=lyrics,
            bpm=bpm,
            keyscale=keyscale,
        )
        if str(ai.get("genre") or "").strip() or str(ai.get("caption_tags") or "").strip():
            return ai

    return {
        "genre": "",
        "style_profile": "",
        "caption_tags": "",
        "genre_label_source": "filename_fallback",
        "genre_confidence": 0.0,
        "genre_reason": "No metadata, MusicBrainz, manual, or AI genre was available.",
    }


def _strip_paren_suffix(title: str) -> str:
    return re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*$", "", title or "").strip()


def _lyrics_title_candidates(title: str) -> list[str]:
    """Return likely public-lyrics title variants for messy filenames/ID3 tags."""
    original = (title or "").strip()
    if not original:
        return []
    candidates: list[str] = []

    def add(value: str) -> None:
        cleaned = re.sub(r"\s+", " ", (value or "").strip(" -–—:"))
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    add(original)
    bare = _strip_paren_suffix(original)
    add(bare)
    punctless = re.sub(r"[’'`]", "", bare or original).strip()
    add(punctless)

    key = _lyrics_lookup_key(bare or original)
    for alias_key, aliases in _LYRICS_TITLE_ALIASES.items():
        if key == alias_key:
            for alias in aliases:
                add(alias)

    replaced = key
    for source, target in _LYRICS_WORD_REPLACEMENTS.items():
        replaced = re.sub(rf"\b{re.escape(source)}\b", target, replaced)
    if replaced and replaced != key:
        add(replaced.title())

    if key == "only fear death":
        add("Only Fear Of Death")
    if key == "hell razor":
        add("Hellrazor")

    # Some lyrics sites collapse short compound titles into a single word.
    words = key.split()
    if len(words) == 2 and all(len(word) > 3 for word in words):
        add("".join(words).title())

    return candidates


def _fetch_lyrics_ovh(artist: str, title: str) -> str:
    if not artist or not title:
        return ""
    candidates: list[tuple[str, str]] = [(artist, item) for item in _lyrics_title_candidates(title)]
    for artist_q, title_q in candidates:
        try:
            url = f"{_LYRICS_OVH_BASE}/{quote(artist_q, safe='')}/{quote(title_q, safe='')}"
            req = urllib.request.Request(url, headers={"User-Agent": "AceJAM/1.0"})
            with urllib.request.urlopen(req, timeout=_LYRICS_OVH_TIMEOUT) as response:
                if response.status >= 400:
                    continue
                data = json.loads(response.read().decode("utf-8", errors="replace"))
            lyrics = str(data.get("lyrics") or "").strip()
            if lyrics:
                return lyrics
        except Exception:
            continue
    return ""


def _normalize_section_kind(kind: str) -> str:
    k = re.sub(r"[\s-]+", "", kind or "").lower()
    mapping = {
        "intro": "Intro",
        "verse": "Verse",
        "chorus": "Chorus",
        "prechorus": "Pre-Chorus",
        "bridge": "Bridge",
        "hook": "Hook",
        "refrain": "Refrain",
        "outro": "Outro",
        "interlude": "Interlude",
        "breakdown": "Breakdown",
        "drop": "Drop",
        "build": "Build",
        "coda": "Coda",
        "tag": "Tag",
    }
    return mapping.get(k, kind.title() if kind else "Verse")


def _apply_acestep_section_tags(raw_lyrics: str) -> str:
    """Convert lyric sources (Genius/lyrics.ovh style) into ACE-Step section tags.

    Recognises headers like "Verse One: Author", "[Chorus]", "Intro:", etc.
    If no section header is present anywhere, we synthesize tags by treating
    the first stanza as `[Intro]`, alternating `[Verse N]`/`[Chorus]` for the
    middle stanzas, and the last stanza as `[Outro]` so the trainer at least
    sees the canonical bracket structure.
    """
    lyrics = (raw_lyrics or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not lyrics:
        return ""

    paragraphs = [p.strip("\n") for p in re.split(r"\n\s*\n", lyrics) if p.strip()]
    has_header = any(_SECTION_HEADER_RE.match(p.split("\n", 1)[0] or "") for p in paragraphs)

    counters: dict[str, int] = {}

    # ACE-Step convention (per prompt_kit MASTER_RULES): Verse is numbered,
    # everything else is unnumbered. Multiple [Chorus] occurrences keep the
    # same tag.
    UNNUMBERED = {
        "Intro", "Outro", "Chorus", "Pre-Chorus", "Bridge", "Hook",
        "Refrain", "Interlude", "Breakdown", "Drop", "Build", "Coda", "Tag",
    }

    def next_tag(kind: str, explicit_num: str = "") -> str:
        normalized = _normalize_section_kind(kind)
        if normalized in UNNUMBERED:
            return f"[{normalized}]"
        num: int | None = None
        if explicit_num:
            stripped = explicit_num.strip().lower()
            if stripped.isdigit():
                num = int(stripped)
            elif stripped in _WORD_TO_INT:
                num = _WORD_TO_INT[stripped]
        if num is None:
            counters[normalized] = counters.get(normalized, 0) + 1
            num = counters[normalized]
        return f"[{normalized} {num}]"

    if has_header:
        out_parts: list[str] = []
        for paragraph in paragraphs:
            lines = paragraph.split("\n")
            head = lines[0]
            match = _SECTION_HEADER_RE.match(head)
            if match:
                tag = next_tag(match.group("kind") or "Verse", match.group("num") or "")
                body = "\n".join(lines[1:]).strip()
                out_parts.append(f"{tag}\n{body}".rstrip())
            else:
                out_parts.append(paragraph.rstrip())
        return "\n\n".join(part for part in out_parts if part).strip()

    if len(paragraphs) == 1:
        return f"[Verse 1]\n{paragraphs[0]}".strip()
    out_parts2: list[str] = []
    last_idx = len(paragraphs) - 1
    for idx, paragraph in enumerate(paragraphs):
        if idx == 0:
            tag = "[Intro]"
        elif idx == last_idx:
            tag = "[Outro]"
        else:
            # Alternate Verse → Chorus → Verse → Chorus
            tag = next_tag("Chorus") if idx % 2 == 0 else next_tag("Verse")
        out_parts2.append(f"{tag}\n{paragraph}".strip())
    return "\n\n".join(out_parts2).strip()


def _training_lookup_online_lyrics(audio_path: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Look up lyrics for an audio file online and return an understand_music
    -shaped dict that `write_label_sidecars` can consume.

    No ACE-Step LM, no audio-codes extraction — just sidecars/ID3,
    MusicBrainz, HTTP lyrics, lightweight audio analysis, and optional local
    Ollama/LM Studio genre fallback. Returns:
        caption       — "{artist} – {title}" derived from tags/filename
        lyrics        — section-tagged lyrics (or "[Instrumental]" if not found)
        bpm/key_scale — detected from audio when possible
        time_signature— "4" fallback
        language      — request language when known
        is_instrumental — true when lyrics lookup failed
        label_source  — online source used, or "online_lyrics_missing"
    """
    existing_metadata = _training_read_existing_label_sidecars(audio_path)
    id3_tags = _extract_easy_audio_tags(audio_path)
    artist, title = _resolve_audio_artist_title(audio_path)
    source = "online_lyrics_missing"
    online = ""
    if artist and title:
        online = _search_lyrics_online(artist, title)
        if online:
            source = "online_lyrics_genius"
        else:
            online = _fetch_lyrics_ovh(artist, title)
            if online:
                source = "online_lyrics_ovh"
    existing_lyrics = str(existing_metadata.get("lyrics") or "").strip()
    tagged = _apply_acestep_section_tags(online) if online else ""
    if not tagged.strip() and existing_lyrics and not is_missing_vocal_lyrics({"lyrics": existing_lyrics}):
        tagged = _apply_acestep_section_tags(existing_lyrics)
        source = str(existing_metadata.get("label_source") or "existing_sidecar")
    has_lyrics = bool(tagged.strip())
    bpm, keyscale = _detect_bpm_key(str(audio_path))
    bpm = bpm or existing_metadata.get("bpm")
    keyscale = keyscale or str(existing_metadata.get("key_scale") or existing_metadata.get("keyscale") or "")
    caption_bits = [bit for bit in [artist, title] if bit]
    caption_seed = str(existing_metadata.get("caption") or "").strip() or " – ".join(caption_bits).strip() or audio_path.stem
    lyrics_for_genre = tagged or existing_lyrics
    genre_label = _training_genre_label(
        audio_path,
        body=body,
        artist=artist,
        title=title,
        metadata=existing_metadata,
        id3_tags=id3_tags,
        lyrics=lyrics_for_genre,
        bpm=bpm,
        keyscale=keyscale,
    )
    genre_terms = _training_split_genre_terms(genre_label.get("genre"))
    style_profile = str(genre_label.get("style_profile") or "").strip()
    caption = _training_compose_caption(
        artist=artist,
        title=title,
        bpm=bpm,
        keyscale=keyscale,
        has_vocals=has_lyrics,
        genre_terms=genre_terms,
        style_profile=style_profile,
        caption_tags=str(genre_label.get("caption_tags") or ""),
        fallback_caption=caption_seed,
    )
    conditioned = apply_audio_style_conditioning(
        {
            "caption": caption,
            "lyrics": tagged or "[Instrumental]",
            "style_profile": style_profile or "auto",
        }
    )
    caption = str(conditioned.get("caption") or caption).strip()
    lyrics_out = str(conditioned.get("lyrics") or tagged or "[Instrumental]").strip()
    lyrics_status = "verified" if has_lyrics else "missing"
    requires_review = not has_lyrics
    return {
        "caption": caption,
        "lyrics": lyrics_out,
        "lyrics_status": lyrics_status,
        "requires_review": requires_review,
        "bpm": bpm,
        "key_scale": keyscale,
        "time_signature": "4",
        "language": str(existing_metadata.get("language") or body.get("language") or body.get("vocal_language") or "").strip() or "unknown",
        "is_instrumental": not has_lyrics,
        "label_source": source if has_lyrics else "online_lyrics_missing",
        "ace_lm_model": "",
        "online_artist": artist,
        "online_title": title,
        "genre": str(genre_label.get("genre") or "").strip(),
        "style_profile": style_profile,
        "genre_profile": style_profile,
        "caption_tags": str(genre_label.get("caption_tags") or "").strip(),
        "genre_label_source": str(genre_label.get("genre_label_source") or "").strip(),
        "genre_confidence": genre_label.get("genre_confidence"),
        "genre_reason": str(genre_label.get("genre_reason") or "").strip(),
        "genre_label_provider": str(genre_label.get("genre_label_provider") or "").strip(),
        "genre_label_model": str(genre_label.get("genre_label_model") or "").strip(),
        "genre_label_error": str(genre_label.get("genre_label_error") or "").strip(),
        "id3_genre": str(genre_label.get("id3_genre") or "").strip(),
        "musicbrainz_tags": genre_label.get("musicbrainz_tags") or [],
        "style_lyric_tags_applied": conditioned.get("style_lyric_tags_applied") or [],
        "style_conditioning_audit": conditioned.get("style_conditioning_audit") or {},
    }


def _training_write_label_sidecars(audio_path: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Write `<stem>.lyrics.txt` and `<stem>.json` sidecars next to audio_path."""
    stem = audio_path.stem
    lyrics_path = audio_path.with_name(f"{stem}.lyrics.txt")
    metadata_path = audio_path.with_name(f"{stem}.json")
    lyrics_text = str(payload.get("lyrics") or "").strip()
    missing = is_missing_vocal_lyrics({**payload, "lyrics": lyrics_text})
    raw_status = str(payload.get("lyrics_status") or "").strip().lower()
    lyrics_path.write_text(lyrics_text + "\n", encoding="utf-8")
    metadata = {
        "caption": str(payload.get("caption") or "").strip(),
        "lyrics": lyrics_text,
        "lyrics_status": raw_status or ("missing" if missing else "present"),
        "requires_review": parse_bool(payload.get("requires_review"), missing),
        "bpm": payload.get("bpm"),
        "keyscale": str(payload.get("key_scale") or payload.get("keyscale") or "").strip(),
        "timesignature": str(payload.get("time_signature") or payload.get("timesignature") or "").strip(),
        "language": str(payload.get("language") or payload.get("vocal_language") or "").strip(),
        "is_instrumental": lyrics_text.strip().lower() == "[instrumental]",
        "label_source": str(payload.get("label_source") or ("online_lyrics_missing" if missing else "online_lyrics")),
        "ace_lm_model": str(payload.get("ace_lm_model") or ""),
        "online_artist": str(payload.get("online_artist") or ""),
        "online_title": str(payload.get("online_title") or ""),
        "genre": str(payload.get("genre") or "").strip(),
        "style_profile": str(payload.get("style_profile") or payload.get("genre_profile") or "").strip(),
        "genre_profile": str(payload.get("genre_profile") or payload.get("style_profile") or "").strip(),
        "caption_tags": str(payload.get("caption_tags") or "").strip(),
        "genre_label_source": str(payload.get("genre_label_source") or "").strip(),
        "genre_confidence": payload.get("genre_confidence"),
        "genre_reason": str(payload.get("genre_reason") or "").strip(),
        "genre_label_provider": str(payload.get("genre_label_provider") or "").strip(),
        "genre_label_model": str(payload.get("genre_label_model") or "").strip(),
        "genre_label_error": str(payload.get("genre_label_error") or "").strip(),
        "id3_genre": str(payload.get("id3_genre") or "").strip(),
        "musicbrainz_tags": payload.get("musicbrainz_tags") or [],
        "style_lyric_tags_applied": payload.get("style_lyric_tags_applied") or [],
        "style_conditioning_audit": payload.get("style_conditioning_audit") or {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"lyrics_path": str(lyrics_path), "metadata_path": str(metadata_path)}


training_manager = AceTrainingManager(
    base_dir=BASE_DIR,
    data_dir=DATA_DIR,
    model_cache_dir=MODEL_CACHE_DIR,
    release_models=_release_models_for_training,
    adapter_ready=_activate_trained_adapter,
    audition_runner=_run_lora_epoch_audition,
    understand_music=_training_lookup_online_lyrics,
    write_label_sidecars=_training_write_label_sidecars,
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
    scale = clamp_float(payload.get("lora_scale", payload.get("lora_weight")), DEFAULT_LORA_GENERATION_SCALE, 0.0, 1.0)
    model_variant = str(payload.get("adapter_model_variant") or "").strip()
    adapter_song_model = str(payload.get("adapter_song_model") or "").strip()
    payload_trigger_tag = str(payload.get("lora_trigger_tag") or payload.get("lora_trigger") or "").strip()
    trigger_tag = payload_trigger_tag
    trigger_source = "payload" if payload_trigger_tag else ""
    trigger_aliases: list[Any] = []
    trigger_candidates: list[Any] = []
    adapter_metadata: dict[str, Any] = {}
    if path:
        adapter_path = Path(path).expanduser()
        adapter_metadata = infer_adapter_model_metadata(adapter_path) if adapter_path.exists() else {}
        model_variant = model_variant or str(adapter_metadata.get("model_variant") or "").strip()
        adapter_song_model = adapter_song_model or str(adapter_metadata.get("song_model") or "").strip()
        name = name or str(
            adapter_metadata.get("display_name")
            or adapter_metadata.get("trigger_tag_raw")
            or adapter_metadata.get("generation_trigger_tag")
            or adapter_metadata.get("trigger_tag")
            or ""
        ).strip()
        trigger_tag = trigger_tag or str(
            adapter_metadata.get("generation_trigger_tag")
            or adapter_metadata.get("trigger_tag")
            or ""
        ).strip()
        if not trigger_source and trigger_tag:
            trigger_source = str(adapter_metadata.get("trigger_source") or "metadata").strip() or "metadata"
        trigger_aliases = list(adapter_metadata.get("trigger_aliases") or [])
        trigger_candidates = list(adapter_metadata.get("trigger_candidates") or [])
    safe_trigger = safe_generation_trigger_tag(trigger_tag)
    trigger_toggle = payload.get("use_lora_trigger", payload.get("lora_use_trigger"))
    trigger_explicitly_disabled = (
        ("use_lora_trigger" in payload or "lora_use_trigger" in payload)
        and not parse_bool(trigger_toggle, True)
    )
    use_trigger = bool(
        use_lora
        and safe_trigger
        and not trigger_explicitly_disabled
    )
    return {
        "use_lora": use_lora,
        "lora_adapter_path": path,
        "lora_adapter_name": name,
        "use_lora_trigger": use_trigger,
        "lora_trigger_tag": safe_trigger if use_trigger else "",
        "lora_trigger_tag_candidate": safe_trigger,
        "lora_scale": scale,
        "adapter_model_variant": model_variant,
        "adapter_song_model": adapter_song_model,
        "adapter_metadata": adapter_metadata,
        "lora_trigger_source": trigger_source if use_trigger else "",
        "lora_trigger_aliases": trigger_aliases,
        "lora_trigger_candidates": trigger_candidates,
        "allow_unsafe_lora_for_benchmark": parse_bool(payload.get("allow_unsafe_lora_for_benchmark"), False),
    }


def _caption_contains_lora_trigger(caption: str, trigger_tag: str) -> bool:
    trigger = str(trigger_tag or "").strip()
    if not trigger:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(trigger)}(?![A-Za-z0-9])"
    return re.search(pattern, str(caption or ""), flags=re.IGNORECASE) is not None


def _lora_trigger_term_matches(term: str, trigger_tag: str) -> bool:
    trigger = safe_generation_trigger_tag(trigger_tag).strip().lower()
    if not trigger:
        return False
    cleaned = re.sub(r"\s+", " ", str(term or "").strip()).strip("[]").strip()
    if not cleaned:
        return False
    return cleaned.lower() == trigger or safe_generation_trigger_tag(cleaned).strip().lower() == trigger


def _remove_lora_trigger_from_caption(caption: str, trigger_tag: str) -> tuple[str, bool]:
    trigger = safe_generation_trigger_tag(trigger_tag)
    text = str(caption or "").strip()
    if not text or not trigger:
        return text, False
    terms = split_terms(text)
    if terms:
        filtered = [term for term in terms if not _lora_trigger_term_matches(term, trigger)]
        removed = len(filtered) != len(terms)
        return ", ".join(filtered).strip(), removed
    pattern = rf"(?:(?<=^)|(?<=[,;\n|]))\s*{re.escape(trigger)}\s*(?=([,;\n|]|$))"
    stripped = re.sub(pattern, "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*[,;|]\s*[,;|]\s*", ", ", stripped)
    stripped = re.sub(r"^[\s,;|]+|[\s,;|]+$", "", stripped).strip()
    return stripped, stripped != text


def _strip_lora_trigger_conditioning(
    params: dict[str, Any],
    trigger_tag: str | None = None,
    *,
    warning: str = "lora_trigger_stripped_for_no_lora",
) -> bool:
    trigger = safe_generation_trigger_tag(
        trigger_tag
        or params.get("lora_trigger_tag")
        or params.get("lora_trigger_tag_candidate")
    )
    if not trigger:
        return False
    caption, removed = _remove_lora_trigger_from_caption(str(params.get("caption") or ""), trigger)
    if removed:
        params["caption"] = caption
        params["tag_list"] = split_terms(caption)
        warnings = list(params.get("payload_warnings") or [])
        if warning not in warnings:
            warnings.append(warning)
        params["payload_warnings"] = warnings
    return removed


def _apply_lora_trigger_conditioning(params: dict[str, Any]) -> None:
    if not params.get("use_lora"):
        stripped = _strip_lora_trigger_conditioning(params)
        params["use_lora_trigger"] = False
        params["lora_trigger_tag"] = ""
        params["lora_trigger_applied"] = False
        params["lora_trigger_conditioning_audit"] = {
            "status": "disabled",
            "caption_only": True,
            "stripped_from_caption": stripped,
        }
        return

    trigger = safe_generation_trigger_tag(params.get("lora_trigger_tag"))
    use_trigger = bool(params.get("use_lora_trigger") and trigger)
    params["use_lora_trigger"] = use_trigger
    params["lora_trigger_tag"] = trigger if use_trigger else ""
    audit = {
        "status": "disabled",
        "caption_only": True,
        "trigger_tag": trigger,
        "trigger_source": str(params.get("lora_trigger_source") or ""),
        "trigger_aliases": list(params.get("lora_trigger_aliases") or []),
        "trigger_candidates": list(params.get("lora_trigger_candidates") or []),
        "applied": False,
        "already_present": False,
        "in_lyrics": False,
    }
    if not use_trigger:
        params["lora_trigger_applied"] = False
        params["lora_trigger_conditioning_audit"] = audit
        return

    caption = str(params.get("caption") or "").strip()
    already_present = _caption_contains_lora_trigger(caption, trigger)
    if not already_present:
        caption = f"{trigger}, {caption}" if caption else trigger
        params["caption"] = caption
        warnings = list(params.get("payload_warnings") or [])
        marker = "lora_trigger_tag_added_to_caption"
        if marker not in warnings:
            warnings.append(marker)
        params["payload_warnings"] = warnings
    params["tag_list"] = split_terms(caption)
    in_lyrics = _caption_contains_lora_trigger(str(params.get("lyrics") or ""), trigger)
    audit.update(
        {
            "status": "present" if already_present else "applied",
            "applied": not already_present,
            "already_present": already_present,
            "in_lyrics": in_lyrics,
        }
    )
    if in_lyrics:
        warnings = list(params.get("payload_warnings") or [])
        marker = "lora_trigger_tag_present_in_user_lyrics"
        if marker not in warnings:
            warnings.append(marker)
        params["payload_warnings"] = warnings
    params["lora_trigger_applied"] = True
    params["lora_trigger_conditioning_audit"] = audit


def _validate_lora_request_for_song_model(lora_request: dict[str, Any], song_model: str) -> None:
    if not lora_request.get("use_lora"):
        return
    requested = normalize_training_song_model(song_model)
    adapter_song_model = str(lora_request.get("adapter_song_model") or "").strip()
    adapter_variant = str(lora_request.get("adapter_model_variant") or "").strip()
    if not adapter_song_model and adapter_variant:
        adapter_song_model = model_from_variant(adapter_variant, "")
    if adapter_song_model:
        adapter_song_model = normalize_training_song_model(adapter_song_model)
    if adapter_song_model and requested and adapter_song_model != requested:
        raise ValueError(
            f"Selected LoRA was trained for {adapter_song_model}, but this render uses {requested}. "
            "Choose the matching ACE-Step model or select a different LoRA."
        )
    quality = adapter_quality_metadata(lora_request.get("adapter_metadata") or {}, adapter_type="lora")
    quality_status = str(quality.get("quality_status") or "").lower()
    if quality_status in ACEJAM_LORA_UNSAFE_QUALITY_STATUSES and not parse_bool(
        lora_request.get("allow_unsafe_lora_for_benchmark"), False
    ):
        reasons = "; ".join(str(item) for item in quality.get("quality_reasons") or [] if str(item))
        raise ValueError(
            "Selected LoRA is quarantined and cannot be used for generation"
            + (f": {reasons}" if reasons else ".")
        )


def _lora_quality_for_params(params: dict[str, Any]) -> dict[str, Any]:
    metadata = params.get("adapter_metadata") if isinstance(params.get("adapter_metadata"), dict) else {}
    if not metadata and params.get("lora_adapter_path"):
        metadata = infer_adapter_model_metadata(Path(str(params.get("lora_adapter_path"))))
    quality = adapter_quality_metadata(metadata, adapter_type=str(metadata.get("adapter_type") or "lora"))
    return {**quality, "metadata": metadata}


def _lora_adapter_metadata_path(adapter_path: str | Path) -> Path | None:
    raw_path = str(adapter_path or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_file():
        path = path.parent
    return path / "acejam_adapter.json"


def _update_lora_adapter_quality_metadata(
    adapter_path: str | Path,
    *,
    quality_status: str,
    reason: str,
    audition: dict[str, Any] | None = None,
    recommended_lora_scale: float | None = None,
) -> None:
    meta_path = _lora_adapter_metadata_path(adapter_path)
    if meta_path is None:
        return
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["quality_status"] = quality_status
    reasons = [str(item) for item in list(meta.get("quality_reasons") or []) if str(item).strip()]
    if reason:
        reasons.append(reason)
    meta["quality_reasons"] = list(dict.fromkeys(reasons))
    if recommended_lora_scale is not None:
        # Audit trail only: generation must use the explicit user-selected
        # lora_scale from the render payload, never a stored recommendation.
        meta.pop("recommended_lora_scale", None)
        meta["last_verified_lora_scale"] = round(float(recommended_lora_scale), 4)
    if audition:
        auditions = [dict(item) for item in list(meta.get("epoch_auditions") or []) if isinstance(item, dict)]
        auditions.append(dict(audition))
        meta["epoch_auditions"] = auditions[-20:]
        meta["audition_passed"] = quality_status in {"verified", "succeeded", "passed"}
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[lora_quality] failed to update {meta_path}: {exc}", flush=True)


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
    raw_scale = params.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE)
    scale_msg = handler.set_lora_scale(float(DEFAULT_LORA_GENERATION_SCALE if raw_scale in (None, "") else raw_scale))
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
    model_name = str(config_path or "").strip()
    if model_name.startswith("acestep-v15-") and not _song_model_runtime_ready(model_name):
        reasons = "; ".join(_song_model_runtime_missing_reasons(model_name))
        return (
            f"ACE-Step model {model_name} is not fully downloaded. Missing: {reasons}. "
            "Run Install to download all weights, or use the Models panel to download this model.",
            False,
        )
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


def _ace_payload_debug_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), ensure_ascii=False, indent=2)


def _ace_payload_debug_block(label: str, value: Any) -> str:
    text = _ace_payload_debug_text(value)
    original_len = len(text)
    if ACEJAM_PRINT_ACE_PAYLOAD_MAX_CHARS and original_len > ACEJAM_PRINT_ACE_PAYLOAD_MAX_CHARS:
        text = (
            text[:ACEJAM_PRINT_ACE_PAYLOAD_MAX_CHARS].rstrip()
            + f"\n[truncated by ACEJAM_PRINT_ACE_PAYLOAD_MAX_CHARS; original_chars={original_len}]"
        )
    return (
        f"[ace_step_payload][BEGIN {label} chars={original_len}]\n"
        f"{text}\n"
        f"[ace_step_payload][END {label}]"
    )


def _print_ace_step_terminal_payload(params: dict[str, Any], request: dict[str, Any], save_dir: Path) -> None:
    if not ACEJAM_PRINT_ACE_PAYLOAD:
        return
    request_params = dict(request.get("params") or {})
    album_meta = params.get("album_metadata") if isinstance(params.get("album_metadata"), dict) else {}
    runner_request_path = save_dir.parent / "official_request.json" if save_dir.name == "official" else save_dir / "official_request.json"
    summary = {
        "song_model": request.get("song_model"),
        "lm_model": request.get("lm_model") or "none",
        "lm_backend": request.get("lm_backend"),
        "requires_lm": request.get("requires_lm"),
        "task_type": request_params.get("task_type"),
        "title": album_meta.get("title") or params.get("title") or "",
        "track_number": album_meta.get("track_number"),
        "album_title": album_meta.get("album_title") or album_meta.get("album_name") or "",
        "save_dir": request.get("save_dir"),
        "official_request_path": str(runner_request_path),
        "terminal_payload_text_path": str(save_dir / "ace_step_terminal_payload.txt"),
        "terminal_payload_json_path": str(save_dir / "ace_step_terminal_payload.json"),
    }
    metadata_and_settings = {
        "metadata": {
            "duration": request_params.get("duration"),
            "bpm": request_params.get("bpm"),
            "keyscale": request_params.get("keyscale"),
            "timesignature": request_params.get("timesignature"),
            "vocal_language": request_params.get("vocal_language"),
            "instrumental": request_params.get("instrumental"),
        },
        "diffusion": {
            "inference_steps": request_params.get("inference_steps"),
            "guidance_scale": request_params.get("guidance_scale"),
            "shift": request_params.get("shift"),
            "infer_method": request_params.get("infer_method"),
            "sampler_mode": request_params.get("sampler_mode"),
            "timesteps": request_params.get("timesteps"),
            "seed": request_params.get("seed"),
            "use_adg": request_params.get("use_adg"),
        },
        "lm_controls": {
            "ace_lm_model": request.get("lm_model") or "none",
            "lm_backend": request.get("lm_backend"),
            "thinking": request_params.get("thinking"),
            "use_format": request_params.get("use_format"),
            "use_cot_metas": request_params.get("use_cot_metas"),
            "use_cot_caption": request_params.get("use_cot_caption"),
            "use_cot_lyrics": request_params.get("use_cot_lyrics"),
            "use_cot_language": request_params.get("use_cot_language"),
            "lm_temperature": request_params.get("lm_temperature"),
            "lm_cfg_scale": request_params.get("lm_cfg_scale"),
            "lm_top_k": request_params.get("lm_top_k"),
            "lm_top_p": request_params.get("lm_top_p"),
        },
        "output": dict(request.get("config") or {}),
        "adapter": {
            "use_lora": request.get("use_lora"),
            "lora_adapter_name": request.get("lora_adapter_name"),
            "lora_adapter_path": request.get("lora_adapter_path"),
            "lora_scale": request.get("lora_scale"),
            "adapter_model_variant": request.get("adapter_model_variant"),
        },
        "text_budget": request.get("ace_step_text_budget") or {},
    }
    request_block_label = "direct_handler_request_json" if request.get("runner") == "direct_handler" else "official_request_json"
    blocks = [
        "[ace_step_payload] FULL ACE-Step request dump enabled. Set ACEJAM_PRINT_ACE_PAYLOAD=0 to silence.",
        _ace_payload_debug_block("summary_json", summary),
        _ace_payload_debug_block("caption", request_params.get("caption") or ""),
        _ace_payload_debug_block("global_caption", request_params.get("global_caption") or ""),
        _ace_payload_debug_block("lyrics", request_params.get("lyrics") or ""),
        _ace_payload_debug_block("metadata_and_settings_json", metadata_and_settings),
        _ace_payload_debug_block(request_block_label, request),
    ]
    debug_text = "\n".join(blocks)
    print(debug_text, flush=True)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "ace_step_terminal_payload.txt").write_text(debug_text, encoding="utf-8")
        (save_dir / "ace_step_terminal_payload.json").write_text(
            json.dumps(
                _jsonable({
                    "summary": summary,
                    "metadata_and_settings": metadata_and_settings,
                    "ace_step_request": request,
                    "official_request": request,
                }),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[ace_step_payload][write_error] {type(exc).__name__}: {exc}", flush=True)


def _direct_ace_step_debug_request(params: dict[str, Any], save_dir: Path, active_song_model: str | None = None) -> dict[str, Any]:
    runtime_caption = str(params.get("caption") or params.get("prompt") or "")
    runtime_global_caption = str(params.get("global_caption") or "")
    runtime_lyrics = "[Instrumental]" if params.get("instrumental") else str(params.get("lyrics") or "")
    params_payload = {
        "task_type": params.get("task_type") or "text2music",
        "instruction": params.get("instruction") or "Fill the audio semantic mask based on the given conditions:",
        "reference_audio": str(params.get("reference_audio")) if params.get("reference_audio") else None,
        "src_audio": str(params.get("src_audio")) if params.get("src_audio") else None,
        "audio_codes": params.get("audio_code_string") or params.get("audio_codes") or "",
        "caption": runtime_caption,
        "global_caption": runtime_global_caption,
        "lyrics": runtime_lyrics,
        "instrumental": bool(params.get("instrumental")),
        "vocal_language": params.get("vocal_language") or _language_for_generation(str(params.get("language") or "")),
        "bpm": params.get("bpm"),
        "keyscale": params.get("key_scale") or params.get("keyscale") or "",
        "timesignature": params.get("time_signature") or params.get("timesignature") or "",
        "duration": params.get("duration") if "duration" in params else params.get("audio_duration"),
        "inference_steps": params.get("inference_steps") if "inference_steps" in params else params.get("infer_steps"),
        "seed": params.get("seed"),
        "guidance_scale": params.get("guidance_scale"),
        "use_adg": params.get("use_adg"),
        "cfg_interval_start": params.get("cfg_interval_start"),
        "cfg_interval_end": params.get("cfg_interval_end"),
        "shift": params.get("shift"),
        "infer_method": params.get("infer_method"),
        "sampler_mode": params.get("sampler_mode"),
        "timesteps": params.get("timesteps"),
        "repainting_start": params.get("repainting_start"),
        "repainting_end": params.get("repainting_end"),
        "audio_cover_strength": params.get("audio_cover_strength"),
        "thinking": params.get("thinking"),
        "lm_temperature": params.get("lm_temperature"),
        "lm_cfg_scale": params.get("lm_cfg_scale"),
        "lm_top_k": params.get("lm_top_k"),
        "lm_top_p": params.get("lm_top_p"),
        "use_cot_metas": params.get("use_cot_metas"),
        "use_cot_caption": params.get("use_cot_caption"),
        "use_cot_lyrics": params.get("use_cot_lyrics"),
        "use_cot_language": params.get("use_cot_language"),
        "use_format": params.get("use_format"),
    }
    return {
        "runner": "direct_handler",
        "base_dir": str(BASE_DIR),
        "model_cache_dir": str(MODEL_CACHE_DIR),
        "save_dir": str(save_dir),
        "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "song_model": active_song_model or params.get("song_model") or ACE_STEP_CHECKPOINT,
        "lm_model": None,
        "requires_lm": False,
        "use_lora": bool(params.get("use_lora", False)),
        "lora_adapter_path": params.get("lora_adapter_path", ""),
        "lora_adapter_name": params.get("lora_adapter_name", ""),
        "lora_scale": params.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE),
        "adapter_model_variant": params.get("adapter_model_variant", ""),
        "device": params.get("device", "handler"),
        "dtype": params.get("dtype", "handler"),
        "lm_backend": params.get("lm_backend", ACE_LM_BACKEND_DEFAULT),
        "audio_backend": params.get("audio_backend", _default_audio_backend()),
        "use_mlx_dit": params.get("use_mlx_dit", "auto"),
        "params": params_payload,
        "config": {
            "batch_size": params.get("batch_size", 1),
            "use_random_seed": params.get("use_random_seed"),
            "audio_format": params.get("audio_format", "wav"),
        },
    }


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


def _art_public_url(art_id: str, filename: str) -> str:
    return f"/media/art/{safe_id(art_id)}/{filename}"


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
    artist = normalize_artist_name(artist_name, "MLX Media")
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
    artist_slug = safe_filename(normalize_artist_name(artist_name, "MLX Media"), "MLX Media")[:48]
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
        debug_params = {
            "song_model": active_song_model,
            "caption": prompt,
            "global_caption": "",
            "lyrics": lyrics,
            "instrumental": False,
            "language": language,
            "vocal_language": _language_for_generation(language),
            "audio_duration": audio_duration,
            "duration": audio_duration,
            "infer_steps": infer_steps,
            "inference_steps": infer_steps,
            "guidance_scale": effective_guidance,
            "bpm": bpm,
            "key_scale": key_scale,
            "time_signature": time_signature,
            "use_random_seed": use_random_seed,
            "seed": -1 if use_random_seed else seed,
            "infer_method": "ode",
            "shift": model_shift,
            "use_adg": False,
            "batch_size": 1,
        }
        debug_save_dir = RESULTS_DIR / "_direct_handler_payloads" / uuid.uuid4().hex[:12]
        _print_ace_step_terminal_payload(
            debug_params,
            _direct_ace_step_debug_request(debug_params, debug_save_dir, active_song_model),
            debug_save_dir,
        )
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
_song_batch_jobs: dict[str, dict[str, Any]] = {}
_song_batch_jobs_lock = threading.Lock()
_lora_benchmark_jobs: dict[str, dict[str, Any]] = {}
_lora_benchmark_jobs_lock = threading.Lock()
_lora_autolabel_jobs: dict[str, dict[str, Any]] = {}
_lora_autolabel_jobs_lock = threading.Lock()


def _set_lora_autolabel_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with _lora_autolabel_jobs_lock:
        job = _lora_autolabel_jobs.setdefault(
            job_id,
            {
                "id": job_id,
                "state": "queued",
                "status": "Queued",
                "progress": 0,
                "processed": 0,
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "current_file": "",
                "logs": [],
                "errors": [],
                "labels": [],
                "dataset_id": "",
                "started_at": None,
                "finished_at": None,
            },
        )
        if "logs" in updates:
            new_logs = updates.pop("logs")
            if isinstance(new_logs, list):
                job["logs"] = (list(job.get("logs") or []) + [str(item) for item in new_logs])[-500:]
        for key, value in updates.items():
            job[key] = value
        return _jsonable(dict(job))


def _lora_autolabel_job_snapshot(job_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    with _lora_autolabel_jobs_lock:
        if job_id:
            job = _lora_autolabel_jobs.get(job_id)
            return _jsonable(dict(job)) if job else {}
        return [_jsonable(dict(job)) for job in _lora_autolabel_jobs.values()]


def _community_feed(limit: int = 100, *, refresh_disk: bool = True) -> list[dict[str, Any]]:
    if refresh_disk:
        disk_songs = _load_feed_from_disk()
        _feed_songs[:] = disk_songs
    return _feed_songs[: max(1, int(limit or 100))]


def _library_sort_key(item: dict[str, Any]) -> str:
    return str(item.get("created_at") or item.get("updated_at") or "")


def _library_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"all": len(items), "songs": 0, "results": 0, "videos": 0, "images": 0, "audio": 0}
    for item in items:
        kind = str(item.get("kind") or "")
        source = str(item.get("source") or "")
        if kind == "video":
            counts["videos"] += 1
        elif kind == "image":
            counts["images"] += 1
        else:
            counts["audio"] += 1
        if source == "song":
            counts["songs"] += 1
        elif source == "result":
            counts["results"] += 1
    return counts


def _library_safe_filename(value: str) -> str:
    filename = str(value or "").strip()
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid library filename")
    return filename


def _library_song_item(song: dict[str, Any]) -> dict[str, Any] | None:
    song_id = str(song.get("id") or song.get("song_id") or "").strip()
    if not song_id:
        return None
    audio_url = str(song.get("audio_url") or "").strip()
    return {
        "id": f"song:{song_id}",
        "kind": "audio",
        "source": "song",
        "deletable": True,
        "song_id": song_id,
        "result_id": str(song.get("result_id") or ""),
        "audio_id": str(song.get("audio_id") or ""),
        "title": song.get("title") or "Untitled",
        "artist_name": song.get("artist_name") or "—",
        "tags": song.get("tag_list") or song.get("tags") or [],
        "caption": song.get("caption") or song.get("tags") or "",
        "lyrics": song.get("lyrics") or "",
        "duration": song.get("duration"),
        "bpm": song.get("bpm"),
        "key_scale": song.get("key_scale") or song.get("keyscale"),
        "vocal_language": song.get("vocal_language") or song.get("language"),
        "song_model": song.get("song_model"),
        "created_at": song.get("created_at"),
        "audio_url": audio_url,
        "download_url": audio_url,
        "art": song.get("art") or song.get("single_art"),
        "raw": _jsonable(song),
    }


def _library_result_audio_items(song_ids: set[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not RESULTS_DIR.exists():
        return items
    for result_dir in sorted(RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not result_dir.is_dir():
            continue
        result_id = safe_id(result_dir.name)
        meta: dict[str, Any] = {}
        meta_path = result_dir / "result.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        audios = meta.get("audios") if isinstance(meta.get("audios"), list) else []
        if not audios:
            audios = [
                {
                    "id": path.stem,
                    "filename": path.name,
                    "title": meta.get("title") or path.stem,
                    "audio_url": _result_public_url(result_id, path.name),
                    "download_url": _result_public_url(result_id, path.name),
                }
                for path in sorted(result_dir.iterdir())
                if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
            ]
        for index, audio in enumerate(audios):
            if not isinstance(audio, dict):
                continue
            filename = str(audio.get("filename") or "").strip()
            if not filename:
                url = str(audio.get("audio_url") or audio.get("download_url") or "")
                if "/media/results/" in url:
                    filename = url.rsplit("/", 1)[-1]
            if not filename:
                continue
            try:
                filename = _library_safe_filename(filename)
            except HTTPException:
                continue
            try:
                audio_path = _resolve_child(RESULTS_DIR, result_id, filename)
            except HTTPException:
                continue
            if not audio_path.is_file():
                continue
            linked_song_id = str(audio.get("song_id") or "").strip()
            if linked_song_id and linked_song_id in song_ids:
                continue
            audio_id = str(audio.get("id") or f"take-{index + 1}")
            audio_url = str(audio.get("audio_url") or audio.get("download_url") or _result_public_url(result_id, filename))
            items.append(
                {
                    "id": f"result-audio:{result_id}:{audio_id}",
                    "kind": "audio",
                    "source": "result",
                    "deletable": True,
                    "song_id": linked_song_id,
                    "result_id": result_id,
                    "audio_id": audio_id,
                    "filename": filename,
                    "title": audio.get("title") or meta.get("title") or audio_path.stem,
                    "artist_name": audio.get("artist_name") or meta.get("artist_name") or "—",
                    "tags": meta.get("tag_list") or meta.get("tags") or [],
                    "caption": meta.get("caption") or meta.get("tags") or "",
                    "lyrics": meta.get("lyrics") or "",
                    "duration": meta.get("duration"),
                    "bpm": meta.get("bpm"),
                    "key_scale": meta.get("key_scale") or meta.get("keyscale"),
                    "vocal_language": meta.get("vocal_language") or meta.get("language"),
                    "song_model": meta.get("song_model") or meta.get("active_song_model"),
                    "created_at": meta.get("created_at") or datetime.fromtimestamp(audio_path.stat().st_mtime, timezone.utc).isoformat(),
                    "audio_url": audio_url,
                    "download_url": str(audio.get("download_url") or audio_url),
                    "art": audio.get("art") or meta.get("art") or meta.get("single_art"),
                    "use_lora": bool(meta.get("use_lora")),
                    "lora_scale": meta.get("lora_scale"),
                    "recommended": bool(audio.get("is_recommended_take") or str((meta.get("recommended_take") or {}).get("audio_id") or "") == audio_id),
                    "raw": _jsonable({**meta, "audio": audio}),
                }
            )
    return items


def _library_video_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not MLX_VIDEO_RESULTS_DIR.exists():
        return items
    for result_dir in sorted(MLX_VIDEO_RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not result_dir.is_dir():
            continue
        result_id = safe_id(result_dir.name)
        meta_path = result_dir / "mlx_video_result.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        except Exception:
            meta = {}
        filename = str(meta.get("filename") or "").strip()
        if not filename:
            candidates = sorted(result_dir.glob("*.mp4")) + sorted(result_dir.glob("*.mov")) + sorted(result_dir.glob("*.webm"))
            filename = candidates[-1].name if candidates else ""
        if not filename:
            continue
        try:
            video_path = _resolve_child(MLX_VIDEO_RESULTS_DIR, result_id, filename)
        except HTTPException:
            continue
        if not video_path.is_file():
            continue
        primary_url = str(meta.get("primary_video_url") or meta.get("video_url") or meta.get("url") or f"/media/mlx-video/{result_id}/{filename}")
        items.append(
            {
                "id": f"video:{result_id}",
                "kind": "video",
                "source": "mlx-video",
                "deletable": True,
                "result_id": result_id,
                "filename": filename,
                "title": meta.get("title") or meta.get("prompt") or "MLX video",
                "artist_name": meta.get("artist_name") or "—",
                "prompt": meta.get("prompt") or "",
                "created_at": meta.get("created_at") or datetime.fromtimestamp(video_path.stat().st_mtime, timezone.utc).isoformat(),
                "video_url": primary_url,
                "url": primary_url,
                "download_url": primary_url,
                "poster_url": meta.get("poster_url") or "",
                "raw_video_url": meta.get("raw_video_url") or "",
                "muxed_video_url": meta.get("muxed_video_url") or "",
                "model_label": meta.get("model_label") or meta.get("model_id") or "MLX video",
                "action": meta.get("action") or "",
                "target_type": (meta.get("attach_status") or {}).get("target_type") if isinstance(meta.get("attach_status"), dict) else "",
                "raw": _jsonable(meta),
            }
        )
    return items


def _library_image_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not MFLUX_RESULTS_DIR.exists():
        return items
    for result_dir in sorted(MFLUX_RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not result_dir.is_dir():
            continue
        result_id = safe_id(result_dir.name)
        meta_path = result_dir / "mflux_result.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        except Exception:
            meta = {}
        filename = str(meta.get("filename") or "").strip()
        if not filename:
            candidates = [
                path
                for path in sorted(result_dir.iterdir())
                if path.is_file() and path.suffix.lower() in MFLUX_ALLOWED_IMAGE_EXTENSIONS
            ]
            filename = candidates[-1].name if candidates else ""
        if not filename:
            continue
        try:
            image_path = _resolve_child(MFLUX_RESULTS_DIR, result_id, filename)
        except HTTPException:
            continue
        if not image_path.is_file():
            continue
        image_url = str(meta.get("image_url") or meta.get("url") or f"/media/mflux/{result_id}/{filename}")
        items.append(
            {
                "id": f"image:{result_id}",
                "kind": "image",
                "source": "mflux",
                "deletable": True,
                "result_id": result_id,
                "filename": filename,
                "title": meta.get("title") or meta.get("prompt") or image_path.stem,
                "artist_name": meta.get("artist_name") or "MFLUX",
                "prompt": meta.get("prompt") or "",
                "caption": meta.get("prompt") or "",
                "created_at": meta.get("created_at") or datetime.fromtimestamp(image_path.stat().st_mtime, timezone.utc).isoformat(),
                "image_url": image_url,
                "url": image_url,
                "download_url": image_url,
                "thumbnail_url": meta.get("thumbnail_url") or image_url,
                "width": meta.get("width"),
                "height": meta.get("height"),
                "model_label": meta.get("model_label") or meta.get("model_id") or "MFLUX",
                "action": meta.get("action") or "",
                "tags": meta.get("tags") or [],
                "raw": _jsonable(meta),
            }
        )
    return items


def _library_items(limit: int = 500) -> dict[str, Any]:
    songs = _community_feed(limit=max(limit, 100), refresh_disk=True)
    song_ids = {str(song.get("id") or song.get("song_id") or "") for song in songs if song.get("id") or song.get("song_id")}
    items = [item for song in songs if (item := _library_song_item(song))]
    items.extend(_library_result_audio_items(song_ids))
    items.extend(_library_image_items())
    items.extend(_library_video_items())
    items.sort(key=_library_sort_key, reverse=True)
    items = items[: max(1, int(limit or 500))]
    return {"success": True, "items": _jsonable(items), "counts": _library_counts(items)}


def _parse_library_item_id(kind: str, item_id: str) -> dict[str, str]:
    parts = str(item_id or "").split(":")
    if kind == "song" and len(parts) >= 2 and parts[0] == "song":
        return {"song_id": parts[1]}
    if kind == "result-audio" and len(parts) >= 3 and parts[0] == "result-audio":
        return {"result_id": parts[1], "audio_id": parts[2]}
    if kind == "image" and len(parts) >= 2 and parts[0] == "image":
        return {"result_id": parts[1]}
    if kind == "video" and len(parts) >= 2 and parts[0] == "video":
        return {"result_id": parts[1]}
    return {}


def _delete_library_song(song_id: str) -> dict[str, Any]:
    song_id = safe_id(song_id)
    song_dir = _resolve_child(SONGS_DIR, song_id)
    if not song_dir.is_dir():
        raise HTTPException(status_code=404, detail="Song not found")
    shutil.rmtree(song_dir, ignore_errors=True)
    _feed_songs[:] = [song for song in _feed_songs if str(song.get("id") or song.get("song_id") or "") != song_id]
    return {"songs": 1, "results": 0, "videos": 0, "files": 1}


def _delete_library_result_audio(result_id: str, audio_id: str = "", filename: str = "") -> dict[str, Any]:
    result_id = safe_id(result_id)
    result_dir = _resolve_child(RESULTS_DIR, result_id)
    if not result_dir.is_dir():
        raise HTTPException(status_code=404, detail="Result not found")
    meta_path = result_dir / "result.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    audios = meta.get("audios") if isinstance(meta.get("audios"), list) else []
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for audio in audios:
        if not isinstance(audio, dict):
            continue
        matches_id = bool(audio_id and str(audio.get("id") or "") == audio_id)
        matches_file = bool(filename and str(audio.get("filename") or "") == filename)
        if matches_id or matches_file:
            selected.append(audio)
        else:
            remaining.append(audio)
    if not selected and filename:
        selected = [{"filename": filename, "id": audio_id or Path(filename).stem}]
    if not selected:
        raise HTTPException(status_code=404, detail="Result audio not found")
    deleted_files = 0
    for audio in selected:
        audio_filename = _library_safe_filename(str(audio.get("filename") or filename or ""))
        if not audio_filename:
            continue
        target = _resolve_child(RESULTS_DIR, result_id, audio_filename)
        if target.is_file():
            target.unlink()
            deleted_files += 1
    remaining_audio_files = [
        path
        for path in result_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
    ]
    selected_ids = {str(audio.get("id") or "") for audio in selected}
    if not remaining and not remaining_audio_files:
        shutil.rmtree(result_dir, ignore_errors=True)
        _result_extra_cache.pop(result_id, None)
        return {"songs": 0, "results": 1, "videos": 0, "files": deleted_files}
    if meta_path.is_file():
        meta["audios"] = remaining
        recommended = meta.get("recommended_take") if isinstance(meta.get("recommended_take"), dict) else {}
        if str(recommended.get("audio_id") or "") in selected_ids:
            meta.pop("recommended_take", None)
        meta_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
    _result_extra_cache.pop(result_id, None)
    return {"songs": 0, "results": 0, "videos": 0, "files": deleted_files}


def _delete_library_video(result_id: str) -> dict[str, Any]:
    result_id = safe_id(result_id)
    result_dir = _resolve_child(MLX_VIDEO_RESULTS_DIR, result_id)
    if not result_dir.is_dir():
        raise HTTPException(status_code=404, detail="Video result not found")
    shutil.rmtree(result_dir, ignore_errors=True)
    if MLX_VIDEO_JOBS_DIR.exists():
        for job_file in MLX_VIDEO_JOBS_DIR.glob("*.json"):
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            result_summary = job.get("result_summary") if isinstance(job.get("result_summary"), dict) else {}
            if str(job.get("result_id") or result_summary.get("result_id") or "") == result_id:
                job_file.unlink(missing_ok=True)
    attachments = []
    if MLX_VIDEO_ATTACHMENTS_PATH.is_file():
        try:
            raw = json.loads(MLX_VIDEO_ATTACHMENTS_PATH.read_text(encoding="utf-8"))
            attachments = raw if isinstance(raw, list) else []
        except Exception:
            attachments = []
        filtered = [item for item in attachments if not (isinstance(item, dict) and str(item.get("result_id") or "") == result_id)]
        MLX_VIDEO_ATTACHMENTS_PATH.write_text(json.dumps(_jsonable(filtered), indent=2), encoding="utf-8")
    return {"songs": 0, "results": 0, "videos": 1, "files": 1}


def _delete_library_image(result_id: str) -> dict[str, Any]:
    result_id = safe_id(result_id)
    result_dir = _resolve_child(MFLUX_RESULTS_DIR, result_id)
    if not result_dir.is_dir():
        raise HTTPException(status_code=404, detail="MFLUX image result not found")
    shutil.rmtree(result_dir, ignore_errors=True)
    jobs_dir = MFLUX_RESULTS_DIR.parent / "jobs"
    if jobs_dir.exists():
        for job_file in jobs_dir.glob("*.json"):
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            result_summary = job.get("result_summary") if isinstance(job.get("result_summary"), dict) else {}
            if str(job.get("result_id") or result_summary.get("result_id") or "") == result_id:
                job_file.unlink(missing_ok=True)
    return {"songs": 0, "results": 0, "videos": 0, "images": 1, "files": 1}


def _delete_library_single_item(body: dict[str, Any]) -> dict[str, Any]:
    kind = str(body.get("kind") or body.get("type") or "").strip()
    item_id = str(body.get("id") or body.get("item_id") or "").strip()
    parsed = _parse_library_item_id(kind, item_id)
    if kind == "song":
        deleted = _delete_library_song(str(body.get("song_id") or parsed.get("song_id") or ""))
    elif kind == "result-audio":
        deleted = _delete_library_result_audio(
            str(body.get("result_id") or parsed.get("result_id") or ""),
            str(body.get("audio_id") or parsed.get("audio_id") or ""),
            str(body.get("filename") or ""),
        )
    elif kind == "image":
        deleted = _delete_library_image(str(body.get("result_id") or parsed.get("result_id") or ""))
    elif kind == "video":
        deleted = _delete_library_video(str(body.get("result_id") or parsed.get("result_id") or ""))
    else:
        raise HTTPException(status_code=400, detail="Unsupported library item kind")
    return deleted


def _merge_library_delete_counts(target: dict[str, Any], deleted: dict[str, Any]) -> None:
    for key in ("songs", "results", "videos", "images", "files"):
        target[key] = int(target.get(key) or 0) + int(deleted.get(key) or 0)


def _delete_library_item(body: dict[str, Any]) -> dict[str, Any]:
    if str(body.get("confirm") or "").strip() != "DELETE":
        raise HTTPException(status_code=400, detail="confirm must be DELETE")
    raw_items = body.get("items")
    if isinstance(raw_items, list):
        deleted = {"songs": 0, "results": 0, "videos": 0, "images": 0, "files": 0}
        errors: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                errors.append({"index": index, "error": "Invalid item"})
                continue
            try:
                _merge_library_delete_counts(deleted, _delete_library_single_item(item))
            except HTTPException as exc:
                errors.append({"index": index, "id": item.get("id"), "error": exc.detail})
            except Exception as exc:
                errors.append({"index": index, "id": item.get("id"), "error": str(exc)})
        return {
            "success": not errors,
            "deleted": deleted,
            "errors": errors,
            "library": _library_items(),
        }
    deleted = _delete_library_single_item(body)
    return {"success": True, "deleted": deleted, "library": _library_items()}


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
    return set(official_downloadable_model_ids())


def _official_model_runtime_status(model_name: str) -> dict[str, Any]:
    item = OFFICIAL_ACE_STEP_MODEL_REGISTRY.get(str(model_name or "")) or {}
    job = _model_download_job(model_name)
    installed = _is_model_installed(model_name)
    downloadable = bool(item.get("downloadable")) and model_name not in OFFICIAL_UNRELEASED_MODELS
    if model_name in OFFICIAL_UNRELEASED_MODELS:
        status = "unreleased"
        error = f"{model_name} is listed for parity only; the official registry does not expose downloadable weights yet."
    elif installed:
        status = "installed"
        error = ""
    elif bool(job and job.get("state") in {"queued", "running"}):
        status = "downloading"
        error = ""
    elif downloadable:
        status = "download_required"
        error = ""
    else:
        status = "not_applicable"
        error = "Helper/catalog entry is not download-enabled."
    return {
        "installed": installed,
        "downloadable": downloadable,
        "downloading": bool(job and job.get("state") in {"queued", "running"}),
        "download_job": _jsonable(job),
        "status": status,
        "error": error,
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
        commit_message="Upload private MLX Media ACE-Step LM experiment",
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
        "source": "MLX Media manual smoke gate",
    }
    (model_path / "acejam_smoke_passed.json").write_text(json.dumps(_jsonable(marker), indent=2), encoding="utf-8")
    return {"success": True, "marker": marker, "status": _ace_lm_status_payload()}


def _is_model_installed(model_name: str, ignore_active_job: bool = False) -> bool:
    if not ignore_active_job and _download_job_active(model_name):
        return False
    checkpoint_path = MODEL_CACHE_DIR / "checkpoints" / model_name
    role = str((OFFICIAL_ACE_STEP_MODEL_REGISTRY.get(model_name) or {}).get("role") or "")
    if role == "diffusers_export":
        return diffusers_pipeline_dir_ready(checkpoint_path)
    if model_name == OFFICIAL_CORE_MODEL_ID:
        checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
        return all(_checkpoint_dir_ready(checkpoint_dir / component) for component in OFFICIAL_MAIN_MODEL_COMPONENTS)
    if model_name in ACE_STEP_SHARED_RUNTIME_COMPONENTS:
        return _checkpoint_dir_ready(checkpoint_path)
    if model_name.startswith("acestep-v15-"):
        return _song_model_runtime_ready(model_name)
    if model_name.startswith("acestep-5Hz-lm-"):
        return model_name in {"auto", "none"} or _checkpoint_dir_ready(checkpoint_path)
    if model_name in OFFICIAL_ACE_STEP_MODEL_REGISTRY:
        return _checkpoint_dir_ready(checkpoint_path)
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


def _download_official_model_to_cache(model_name: str, checkpoint_dir: Path) -> None:
    repo_id = official_model_repo_id(model_name)
    if not repo_id:
        raise RuntimeError(f"{model_name} has no official download repository.")
    from huggingface_hub import snapshot_download

    if repo_id == "ACE-Step/Ace-Step1.5":
        _set_model_download_job(model_name, message=f"Downloading official main bundle {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=str(checkpoint_dir), local_dir_use_symlinks=False)
        return
    target_dir = checkpoint_dir / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _set_model_download_job(model_name, message=f"Downloading {repo_id}...")
    snapshot_download(repo_id=repo_id, local_dir=str(target_dir), local_dir_use_symlinks=False)


def _download_shared_runtime_component_to_cache(component: str, checkpoint_dir: Path, model_name: str) -> None:
    if component not in ACE_STEP_SHARED_RUNTIME_COMPONENTS:
        raise RuntimeError(f"{component} is not a known shared ACE-Step runtime component.")
    from huggingface_hub import snapshot_download

    _set_model_download_job(model_name, message=f"Downloading shared ACE-Step component {component}...")
    snapshot_download(
        repo_id=OFFICIAL_MAIN_MODEL_REPO,
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[f"{component}/**"],
    )


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
            role = str((OFFICIAL_ACE_STEP_MODEL_REGISTRY.get(model_name) or {}).get("role") or "")
            if model_name in ACE_STEP_SHARED_RUNTIME_COMPONENTS:
                _download_shared_runtime_component_to_cache(model_name, checkpoint_dir, model_name)
            elif model_name in OFFICIAL_ACE_STEP_MODEL_REGISTRY:
                _download_official_model_to_cache(model_name, checkpoint_dir)
            else:
                handler._ensure_model_downloaded(model_name, str(checkpoint_dir))
            if model_name.startswith("acestep-v15-") and role != "diffusers_export":
                for component in ACE_STEP_SHARED_RUNTIME_COMPONENTS:
                    if not _checkpoint_dir_ready(checkpoint_dir / component):
                        _download_shared_runtime_component_to_cache(component, checkpoint_dir, model_name)
            if not _is_model_installed(model_name, ignore_active_job=True):
                if role == "diffusers_export":
                    reasons = _diffusers_pipeline_status_reason(checkpoint_dir / model_name)
                else:
                    reasons = "; ".join(_song_model_runtime_missing_reasons(model_name)) if model_name.startswith("acestep-v15-") else ""
                suffix = f" Missing: {reasons}." if reasons else ""
                raise RuntimeError(f"{model_name} download finished but required checkpoint weights were not found.{suffix}")
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


def _boot_download_model_names() -> list[str]:
    if ACEJAM_BOOT_DOWNLOAD_ALL_OFFICIAL_MODELS:
        candidates = official_downloadable_model_ids()
    else:
        candidates = official_boot_model_ids(
            include_helpers=ACEJAM_BOOT_DOWNLOAD_OFFICIAL_HELPERS,
            include_best_quality=ACEJAM_BOOT_DOWNLOAD_BEST_QUALITY_MODELS,
        )
    downloadable = _downloadable_model_names()
    names: list[str] = []
    for model_name in candidates:
        if model_name not in downloadable or model_name in OFFICIAL_UNRELEASED_MODELS:
            continue
        if model_name not in names:
            names.append(model_name)
    return names


def _queue_boot_model_downloads() -> dict[str, Any]:
    names = _boot_download_model_names()
    queued: list[str] = []
    skipped: list[str] = []
    failed: dict[str, str] = {}
    if not ACEJAM_BOOT_DOWNLOAD_ENABLED:
        return {"enabled": False, "models": names, "queued": queued, "skipped": names, "failed": failed}
    if os.environ.get("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "").strip() == "1":
        return {"enabled": False, "reason": "tests", "models": names, "queued": queued, "skipped": names, "failed": failed}
    for model_name in names:
        try:
            if _is_model_installed(model_name):
                skipped.append(model_name)
                continue
            _start_model_download(model_name)
            queued.append(model_name)
        except Exception as exc:
            failed[model_name] = str(exc)
    if queued:
        print(f"[startup] Queued official ACE-Step boot downloads: {', '.join(queued)}", flush=True)
    if skipped:
        print(f"[startup] Official ACE-Step boot downloads already installed: {', '.join(skipped)}", flush=True)
    if failed:
        print(f"[startup] Official ACE-Step boot download queue failures: {failed}", flush=True)
    return {"enabled": True, "models": names, "queued": queued, "skipped": skipped, "failed": failed}


def _start_model_download_or_raise(model_name: str, context: str = "generation") -> None:
    job = _start_model_download(model_name)
    raise ModelDownloadStarted(
        model_name,
        job,
        f"{model_name} is not installed yet. MLX Media started the download for {context}. "
        "Wait until the model is installed, then press Generate again.",
    )


_boot_model_download_status: dict[str, Any] = {
    "enabled": ACEJAM_BOOT_DOWNLOAD_ENABLED,
    "models": _boot_download_model_names(),
    "queued": [],
    "skipped": [],
    "failed": {},
    "scheduled": False,
}


def _run_boot_model_download_queue() -> None:
    global _boot_model_download_status
    _boot_model_download_status = _queue_boot_model_downloads()


def _schedule_boot_model_downloads() -> None:
    global _boot_model_download_status
    if not ACEJAM_BOOT_DOWNLOAD_ENABLED:
        _boot_model_download_status = _queue_boot_model_downloads()
        return
    if os.environ.get("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS", "").strip() == "1":
        _boot_model_download_status = _queue_boot_model_downloads()
        return
    if ACEJAM_BOOT_DOWNLOAD_DELAY_SECONDS <= 0:
        _run_boot_model_download_queue()
        return
    _boot_model_download_status = {
        **_boot_model_download_status,
        "scheduled": True,
        "delay_seconds": ACEJAM_BOOT_DOWNLOAD_DELAY_SECONDS,
    }
    timer = threading.Timer(float(ACEJAM_BOOT_DOWNLOAD_DELAY_SECONDS), _run_boot_model_download_queue)
    timer.daemon = True
    timer.start()
    print(
        "[startup] Scheduled official ACE-Step boot downloads: "
        + ", ".join(_boot_model_download_status.get("models") or []),
        flush=True,
    )


_schedule_boot_model_downloads()


def _download_started_payload(model_name: str, job: dict[str, Any], logs: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    message = (
        f"{model_name} is not installed yet. MLX Media started downloading it. "
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
            "message": f"MLX Media started downloading {len(unique_models)} missing album model(s). Album generation will resume after install.",
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


def _is_image_generation_model_name(name: str) -> bool:
    return bool(re.search(r"(^x/(?:z-image|flux)|\b(?:z-image|flux|imagegen|image-gen|text-to-image|txt2img|sdxl|stable-diffusion|qwen-image)\b)", name or "", re.IGNORECASE))


def _is_vision_model_name(name: str) -> bool:
    return bool(re.search(r"(vision|vl\b|llava|bakllava|moondream|minicpm-v|gemma3)", name or "", re.IGNORECASE))


def _ollama_kind_from_model_name(name: str) -> str:
    if _is_embedding_model_name(name):
        return "embedding"
    return "chat"


def _ollama_show_details_for_catalog(client: Any, name: str) -> dict[str, Any]:
    try:
        return _jsonable(client.show(name))
    except Exception:
        return {}


def _ollama_capabilities_for_model(name: str, raw: dict[str, Any] | None = None) -> list[str]:
    raw = raw if isinstance(raw, dict) else {}
    caps: set[str] = set()
    raw_caps = raw.get("capabilities")
    if isinstance(raw_caps, list):
        caps.update(str(item).strip().lower() for item in raw_caps if str(item).strip())
    elif isinstance(raw_caps, dict):
        caps.update(str(key).strip().lower() for key, value in raw_caps.items() if value)
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    families = details.get("families") if isinstance(details.get("families"), list) else []
    haystack = " ".join([name, str(details.get("family") or ""), " ".join(str(item) for item in families)]).lower()
    if _is_embedding_model_name(name):
        caps.add("embedding")
    if _is_image_generation_model_name(name):
        caps.add("image_generation")
    if _is_vision_model_name(name) or "vision" in haystack or "vl" in haystack:
        caps.add("vision")
    if "embedding" not in caps and "image_generation" not in caps:
        caps.add("chat")
    return sorted(caps)


def _ollama_kind_from_capabilities(name: str, capabilities: list[str]) -> str:
    caps = set(capabilities or [])
    if "embedding" in caps:
        return "embedding"
    if "image_generation" in caps:
        return "image_generation"
    if _is_embedding_model_name(name):
        return "embedding"
    if _is_image_generation_model_name(name):
        return "image_generation"
    return "chat"


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


def _ollama_model_catalog(enrich: bool = False) -> dict[str, Any]:
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
            shown = _ollama_show_details_for_catalog(client, name) if enrich else {}
            raw_caps = shown if shown else _jsonable(item)
            capabilities = _ollama_capabilities_for_model(name, raw_caps if isinstance(raw_caps, dict) else {})
            kind = _ollama_kind_from_capabilities(name, capabilities)
            details.append(
                {
                    "name": name,
                    "model": name,
                    "provider": "ollama",
                    "size": size,
                    "size_gb": round(size / 1e9, 2) if size else 0,
                    "modified_at": str(modified_at or ""),
                    "digest": str(digest or ""),
                    "family": str(_ollama_attr(model_details, "family", "") or (model_details.get("family", "") if isinstance(model_details, dict) else "")),
                    "parameter_size": str(_ollama_attr(model_details, "parameter_size", "") or (model_details.get("parameter_size", "") if isinstance(model_details, dict) else "")),
                    "quantization_level": str(_ollama_attr(model_details, "quantization_level", "") or (model_details.get("quantization_level", "") if isinstance(model_details, dict) else "")),
                    "format": str(_ollama_attr(model_details, "format", "") or (model_details.get("format", "") if isinstance(model_details, dict) else "")),
                    "kind": kind,
                    "type": "embedding" if kind == "embedding" else ("image_generation" if kind == "image_generation" else "llm"),
                    "capabilities": capabilities,
                    "vision": "vision" in capabilities,
                    "image_generation": "image_generation" in capabilities,
                    "raw_show": shown if enrich else {},
                }
            )
        model_names = [item["name"] for item in details]
        embedding_models = [item["name"] for item in details if item["kind"] == "embedding"]
        image_models = [item["name"] for item in details if item["kind"] == "image_generation"]
        chat_models = [item["name"] for item in details if item["kind"] == "chat"]
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
            "image_models": image_models,
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
            "image_models": [],
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


def _first_available_model(catalog: dict[str, Any], key: str, preferred: list[str] | None = None) -> str:
    models = [str(item) for item in (catalog.get(key) or []) if str(item).strip()]
    installed = set(models)
    for item in preferred or []:
        if item in installed:
            return item
    if key == "chat_models":
        ranked = sorted(
            models,
            key=lambda name: (
                0 if re.search(r"(charaf|mlx|qwen|gpt-oss|llama|gemma)", name, re.I) else 1,
                len(name),
                name,
            ),
        )
        return ranked[0] if ranked else ""
    return models[0] if models else ""


def _local_llm_default_settings() -> dict[str, Any]:
    try:
        ollama_catalog = _ollama_model_catalog()
    except Exception:
        ollama_catalog = {"chat_models": [], "embedding_models": [], "image_models": []}
    chat_model = _first_available_model(
        ollama_catalog,
        "chat_models",
        [DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL],
    )
    embedding_model = _first_available_model(
        ollama_catalog,
        "embedding_models",
        ALBUM_EMBEDDING_FALLBACK_MODELS,
    )
    planner = planner_llm_settings_from_payload(
        {},
        default_max_tokens=8192,
        default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
    )
    return {
        "provider": "ollama",
        "chat_model": chat_model,
        "embedding_provider": "ollama",
        "embedding_model": embedding_model or DEFAULT_ALBUM_EMBEDDING_MODEL,
        "art_provider": "",
        "art_model": "",
        "art_width": 1024,
        "art_height": 1024,
        "art_steps": 0,
        "art_seed": "",
        "art_negative_prompt": "",
        "auto_single_art": False,
        "auto_album_art": False,
        "mlx_policy": "full_mlx" if _IS_APPLE_SILICON else "auto",
        **planner,
    }


def _normalize_local_llm_settings(payload: dict[str, Any] | None) -> dict[str, Any]:
    defaults = _local_llm_default_settings()
    source = payload if isinstance(payload, dict) else {}
    merged = {**defaults, **{key: value for key, value in source.items() if value is not None}}
    provider = normalize_provider(merged.get("provider") or defaults["provider"])
    if provider not in {"ollama", "lmstudio"}:
        provider = "ollama"
    embedding_provider = normalize_provider(merged.get("embedding_provider") or provider)
    if embedding_provider not in {"ollama", "lmstudio"}:
        embedding_provider = "ollama"
    planner = planner_llm_settings_from_payload(
        merged,
        default_max_tokens=8192,
        default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
    )
    normalized = {
        **defaults,
        **planner,
        "provider": provider,
        "chat_model": str(merged.get("chat_model") or merged.get("planner_model") or merged.get("ollama_model") or defaults["chat_model"] or "").strip(),
        "embedding_provider": embedding_provider,
        "embedding_model": str(merged.get("embedding_model") or defaults["embedding_model"] or "").strip(),
        "art_provider": "",
        "art_model": "",
        "art_width": clamp_int(merged.get("art_width"), defaults["art_width"], 256, 2048),
        "art_height": clamp_int(merged.get("art_height"), defaults["art_height"], 256, 2048),
        "art_steps": clamp_int(merged.get("art_steps"), defaults["art_steps"], 0, 100),
        "art_seed": str(merged.get("art_seed") or "").strip(),
        "art_negative_prompt": "",
        "auto_single_art": False,
        "auto_album_art": False,
        "mlx_policy": "full_mlx" if _IS_APPLE_SILICON else str(merged.get("mlx_policy") or "auto"),
    }
    return _jsonable(normalized)


def _load_local_llm_settings() -> dict[str, Any]:
    try:
        if LOCAL_LLM_SETTINGS_PATH.is_file():
            raw = json.loads(LOCAL_LLM_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return _normalize_local_llm_settings(raw)
    except Exception as exc:
        print(f"[local_llm_settings] load failed: {exc}", flush=True)
    return _normalize_local_llm_settings({})


def _load_local_llm_settings_fast() -> dict[str, Any]:
    """Read persisted LLM settings without probing Ollama/LM Studio.

    Album job creation must stay responsive even while the selected LLM is busy.
    The worker performs the real model preflight and reports pull/wait status.
    """
    try:
        if LOCAL_LLM_SETTINGS_PATH.is_file():
            raw = json.loads(LOCAL_LLM_SETTINGS_PATH.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
    except Exception as exc:
        print(f"[local_llm_settings] fast load failed: {exc}", flush=True)
    return {}


def _save_local_llm_settings(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_local_llm_settings(payload)
    LOCAL_LLM_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_LLM_SETTINGS_PATH.write_text(json.dumps(_jsonable(normalized), indent=2), encoding="utf-8")
    return normalized


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
        f"{model} is not installed in Ollama. MLX Media started pulling it for {context}.",
    )


def _resolve_ollama_model_selection(model_name: str, kind: str, context: str) -> str:
    if kind == "image_generation":
        raise RuntimeError("Ollama image generation is disabled in MLX Media. Use MFLUX instead.")
    model = str(model_name or "").strip()
    if model:
        _ensure_ollama_model_or_start_pull(model, context=context, kind=kind)
        return model
    catalog = _ollama_model_catalog()
    if not catalog.get("ready"):
        raise RuntimeError(catalog.get("error") or "Ollama is not running.")
    key = "embedding_models" if kind == "embedding" else "chat_models"
    models = [str(item) for item in (catalog.get(key) or []) if str(item).strip()]
    preferred_models = (
        ALBUM_EMBEDDING_FALLBACK_MODELS
        if kind == "embedding"
        else [DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL]
    )
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
        raise RuntimeError(f"No local LM Studio {kind} model available. Download/load one in LM Studio, then refresh MLX Media.")
    return models[0]


def _ollama_pull_started_payload(model_name: str, job: dict[str, Any], context: str = "Ollama", **extra: Any) -> dict[str, Any]:
    message = f"{model_name} is not installed in Ollama. MLX Media started pulling it for {context}."
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


def _recover_album_request_concept(concept: Any, payload: dict[str, Any]) -> str:
    parts: list[str] = []
    primary_prompt = next(
        (
            str(value).strip()
            for value in (payload.get("raw_user_prompt"), payload.get("user_prompt"), payload.get("prompt"))
            if isinstance(value, str) and value.strip()
        ),
        "",
    )
    candidate_values = [primary_prompt] if primary_prompt else [concept, payload.get("concept")]
    candidate_values.extend([payload.get("album_title"), payload.get("album_name")])
    for value in candidate_values:
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
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


def _album_contract_source_from_payload(payload: dict[str, Any], body: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    body = body or {}
    primary_prompt = next(
        (
            str(value).strip()
            for value in (
                body.get("raw_user_prompt"),
                body.get("user_prompt"),
                body.get("prompt"),
                payload.get("raw_user_prompt"),
                payload.get("user_prompt"),
                payload.get("prompt"),
            )
            if isinstance(value, str) and value.strip()
        ),
        "",
    )
    parts: list[str] = []
    if primary_prompt:
        parts.append(primary_prompt)
    else:
        for value in (payload.get("concept"), body.get("concept")):
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
    for value in (
        payload.get("album_title"),
        payload.get("album_name"),
        body.get("album_title"),
        body.get("album_name"),
    ):
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            break
    if not primary_prompt and not parts:
        track_rows: list[str] = []
        tracks = payload.get("tracks") or body.get("tracks") or payload.get("planned_tracks") or body.get("planned_tracks") or []
        if isinstance(tracks, list):
            for index, item in enumerate(tracks[:30]):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("locked_title") or "").strip()
                clean_fields = [
                    str(item.get(key) or "").strip()
                    for key in ("style", "vibe", "narrative", "description")
                    if str(item.get(key) or "").strip()
                ]
                if title or clean_fields:
                    row = [f"Track {index + 1}: {title}" if title else f"Track {index + 1}:"]
                    row.extend(clean_fields)
                    track_rows.append("\n".join(row))
        parts.extend(track_rows)
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
    return "\n\n".join(deduped)


def _album_prepare_contract_request_body(body: dict[str, Any], *, fallback_tracks: int = 5) -> dict[str, Any]:
    """Normalize album-plan request shape before a background job starts.

    The album wizard can have stale form values from a previous draft
    (`album_title`, `num_tracks`, `tracks`) while the textarea contains a new
    explicit album contract. The pasted contract must win, otherwise CrewAI
    plans the wrong title/count and the UI appears to have ignored the user.
    """
    payload = dict(body or {})
    source = _album_contract_source_from_payload(payload, payload)
    existing_contract = payload.get("user_album_contract")
    if isinstance(existing_contract, dict):
        contract = existing_contract
    else:
        contract = extract_user_album_contract(
            source,
            int(payload.get("num_tracks") or 0) or None,
            str(payload.get("language") or payload.get("target_language") or "en"),
            payload,
        )
    body_tracks = _json_list(payload.get("tracks"))
    contract_tracks = tracks_from_user_album_contract(contract)
    body_count = clamp_int(payload.get("num_tracks"), 0, 0, 40)
    contract_count = clamp_int(contract.get("track_count") if isinstance(contract, dict) else 0, 0, 0, 40)
    explicit_count = max(
        body_count,
        len(body_tracks),
        contract_count,
        len(contract_tracks),
        int(fallback_tracks or 0),
    )
    if isinstance(contract, dict) and contract.get("applied"):
        payload["user_album_contract"] = contract
        payload["input_contract"] = contract_prompt_context(contract)
        payload["input_contract_applied"] = True
        payload["input_contract_version"] = USER_ALBUM_CONTRACT_VERSION
        payload["blocked_unsafe_count"] = int(contract.get("blocked_unsafe_count") or 0)
        if str(contract.get("album_title") or "").strip():
            payload["album_title"] = str(contract.get("album_title") or "").strip()
        # When a fresh textarea prompt supplied explicit tracks, those locked
        # tracks are more trustworthy than stale draft rows from the form.
        primary_prompt = next(
            (
                str(payload.get(key) or "").strip()
                for key in ("raw_user_prompt", "user_prompt", "prompt")
                if str(payload.get(key) or "").strip()
            ),
            "",
        )
        if contract_tracks and primary_prompt:
            payload["tracks"] = contract_tracks
    if explicit_count:
        payload["num_tracks"] = max(1, min(40, explicit_count))
    return payload


def _effective_direct_min_lyric_lines(raw_min_lines: int, min_words: int) -> int:
    raw = int(raw_min_lines or 0)
    if raw <= 0:
        return 0
    cap = max(36, int((int(min_words or 0) / 6.25) + 0.999))
    return min(raw, cap)


def _direct_album_payload_genre_hint(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("caption", "tags", "description", "style", "vibe", "narrative", "genre_profile"):
        value = payload.get(key)
        if value:
            parts.append(" ".join(str(item) for item in value) if isinstance(value, list) else str(value))
    tag_list = payload.get("tag_list")
    if isinstance(tag_list, list):
        parts.extend(str(item) for item in tag_list if str(item).strip())
    return re.sub(r"\s+", " ", "\n".join(parts)).strip()


def _direct_album_lyric_duration_fit(payload: dict[str, Any]) -> dict[str, Any]:
    lyrics = str(payload.get("lyrics") or "")
    duration = parse_duration_seconds(payload.get("duration") or 180, 180)
    genre_hint = _direct_album_payload_genre_hint(payload)
    plan = lyric_length_plan(
        duration,
        str(payload.get("lyric_density") or "dense"),
        str(payload.get("structure_preset") or "auto"),
        genre_hint,
    )
    stats = lyric_stats(lyrics)
    min_words = int(plan.get("min_words") or 0)
    raw_min_lines = int(plan.get("min_lines") or 0)
    min_lines = _effective_direct_min_lyric_lines(raw_min_lines, min_words)
    issues: list[dict[str, Any]] = []
    instrumental = lyrics.strip().lower() == "[instrumental]" or parse_bool(payload.get("instrumental"), False)
    if lyrics.strip() and not instrumental:
        word_count = int(stats.get("word_count") or 0)
        line_count = int(stats.get("line_count") or 0)
        if word_count < min_words:
            issues.append({
                "id": "lyrics_under_length",
                "severity": "fail",
                "detail": f"{word_count}/{min_words} words",
            })
        if line_count < min_lines:
            issues.append({
                "id": "lyrics_too_few_lines",
                "severity": "fail",
                "detail": f"{line_count}/{min_lines} lines",
            })
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "duration": duration,
        "density": plan.get("density"),
        "word_count": int(stats.get("word_count") or 0),
        "line_count": int(stats.get("line_count") or 0),
        "min_words": min_words,
        "target_words": int(plan.get("target_words") or 0),
        "min_lines": min_lines,
        "target_lines": int(plan.get("target_lines") or 0),
        "raw_min_lines": raw_min_lines,
    }


def _validate_direct_album_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    caption = str(payload.get("caption") or payload.get("tags") or "")
    lyrics = str(payload.get("lyrics") or "")
    if not str(payload.get("title") or "").strip():
        issues.append({"id": "missing_title", "severity": "fail", "detail": "title is required"})
    if not caption.strip():
        issues.append({"id": "missing_caption", "severity": "fail", "detail": "caption is required"})
    if len(caption) > ACE_STEP_CAPTION_CHAR_LIMIT:
        issues.append({"id": "caption_over_limit", "severity": "fail", "detail": f"{len(caption)}/{ACE_STEP_CAPTION_CHAR_LIMIT} chars"})
    if re.search(
        r"\b(?:\d{2,3}\s*bpm|bpm\s*[:=]|\d+\/\d+\s*time|time\s*signature|"
        r"[A-G](?:#|b|♯|♭)?\s+(?:major|minor)|duration|seconds?|model|seed|producer|produced by|production)\b",
        caption,
        re.I,
    ):
        issues.append({"id": "metadata_or_credit_in_caption", "severity": "fail", "detail": "caption contains metadata or production credit"})
    for field in ("producer_credit", "artist_name", "title"):
        value = str(payload.get(field) or "").strip()
        if value and value.lower() in caption.lower():
            issues.append({"id": f"{field}_in_caption", "severity": "fail", "detail": value})
    if not lyrics.strip() or lyrics.strip().lower() == "[instrumental]":
        issues.append({"id": "missing_vocal_lyrics", "severity": "fail", "detail": "album agent payload needs complete lyrics"})
    if len(lyrics) > ACE_STEP_LYRICS_CHAR_LIMIT:
        issues.append({"id": "lyrics_over_limit", "severity": "fail", "detail": f"{len(lyrics)}/{ACE_STEP_LYRICS_CHAR_LIMIT} chars"})
    sections = re.findall(r"\[([^\]]+)\]", lyrics)
    if not any(re.search(r"chorus|hook|refrain", section, re.I) for section in sections):
        issues.append({"id": "hook_missing", "severity": "fail", "detail": "no chorus/hook/refrain section"})
    lyric_duration_fit = _direct_album_lyric_duration_fit(payload)
    issues.extend(lyric_duration_fit.get("issues") or [])
    genre_adherence = evaluate_genre_adherence(payload)
    issues.extend(dict(issue) for issue in (genre_adherence.get("issues") or []))
    return {
        "version": "direct-album-agent-payload-2026-05-01",
        "gate_passed": not issues,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "blocking_issues": issues,
        "caption_chars": len(caption),
        "lyrics_chars": len(lyrics),
        "lyrics_word_count": int(lyric_duration_fit.get("word_count") or 0),
        "lyrics_line_count": int(lyric_duration_fit.get("line_count") or 0),
        "lyric_duration_fit": lyric_duration_fit,
        "genre_intent_contract": genre_adherence.get("contract") or {},
        "genre_adherence": {key: value for key, value in genre_adherence.items() if key != "contract"},
        "sections": [f"[{section}]" for section in sections],
    }


def _normalize_album_agent_engine_value(value: Any) -> str:
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if text in {"crewai", "crew_ai", "crewai_micro", "micro_crewai", "crewai_micro_tasks", "legacy_crewai"}:
        return "crewai_micro"
    return "acejam_agents"


def _album_agent_engine_label_value(value: Any) -> str:
    engine = _normalize_album_agent_engine_value(value)
    return "CrewAI Micro Tasks" if engine == "crewai_micro" else "MLX Media Direct"


def _album_options_from_payload(payload: dict[str, Any], song_model: str = "auto") -> dict[str, Any]:
    strategy = str(payload.get("song_model_strategy") or "all_models_album")
    selection_strategy = "selected" if strategy in {"selected", "single_model_album"} else strategy
    quality_profile = _default_quality_profile_for_payload({**payload, "ui_mode": "album"}, "text2music")
    payload_song_model = str(payload.get("requested_song_model") or payload.get("song_model") or "").strip()
    selected_song_model = str(song_model or "").strip()
    requested_song_model = (
        selected_song_model
        if selected_song_model and selected_song_model != "auto"
        else payload_song_model
    )
    if selection_strategy != "selected":
        requested_song_model = "auto"
    installed_models = sorted(_installed_acestep_models())
    default_model = ALBUM_FINAL_MODEL if selection_strategy != "selected" else (requested_song_model or ALBUM_FINAL_MODEL)
    model_defaults = quality_profile_model_settings(default_model, quality_profile)
    contract_source = _album_contract_source_from_payload(payload)
    user_album_contract = payload.get("user_album_contract")
    if not isinstance(user_album_contract, dict):
        user_album_contract = extract_user_album_contract(
            contract_source,
            int(payload.get("num_tracks") or 0) or None,
            str(payload.get("language") or payload.get("target_language") or "en"),
            payload,
        )
    planner_settings = planner_llm_settings_from_payload(payload)
    return {
        "requested_song_model": requested_song_model or "auto",
        "song_model_strategy": strategy,
        "final_song_model": ALBUM_FINAL_MODEL,
        "prompt_kit_version": str(payload.get("prompt_kit_version") or PROMPT_KIT_VERSION),
        "planner_lm_provider": _album_planner_provider_from_payload(payload),
        "planner_model": str(payload.get("planner_model") or payload.get("planner_ollama_model") or payload.get("ollama_model") or ""),
        **planner_settings,
        "agent_engine": _normalize_album_agent_engine_value(payload.get("agent_engine")),
        "album_writer_mode": str(payload.get("album_writer_mode") or "per_track_writer_loop").strip() or "per_track_writer_loop",
        "max_track_repair_rounds": clamp_int(payload.get("max_track_repair_rounds"), 3, 0, 3),
        "user_prompt": str(payload.get("user_prompt") or payload.get("prompt") or payload.get("concept") or ""),
        "raw_user_prompt": str(payload.get("raw_user_prompt") or payload.get("user_prompt") or payload.get("prompt") or payload.get("concept") or ""),
        "album_agent_genre_prompt": str(payload.get("album_agent_genre_prompt") or payload.get("genre_prompt") or payload.get("custom_tags") or ""),
        "album_agent_mood_vibe": str(payload.get("album_agent_mood_vibe") or payload.get("mood_vibe") or ""),
        "album_agent_vocal_type": str(payload.get("album_agent_vocal_type") or payload.get("vocal_type") or ""),
        "album_agent_audience": str(payload.get("album_agent_audience") or payload.get("audience_platform") or ""),
        "planner_thinking": parse_bool(payload.get("planner_thinking"), False),
        "print_agent_io": parse_bool(payload.get("print_agent_io"), True),
        "planner_ollama_model": str(payload.get("planner_ollama_model") or payload.get("ollama_model") or ""),
        "embedding_lm_provider": _embedding_provider_from_payload(payload),
        "embedding_model": str(payload.get("embedding_model") or ""),
        "ace_lm_model": "none",
        "album_model_portfolio": album_model_portfolio(installed_models),
        "quality_target": str(payload.get("quality_target") or "hit"),
        "quality_profile": quality_profile,
        "duration_mode": _normalize_album_duration_mode(payload.get("duration_mode")),
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
        "vocal_clarity_recovery": _vocal_clarity_recovery_enabled(payload),
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
        "album_title": str(user_album_contract.get("album_title") or payload.get("album_title") or ""),
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


METADATA_LOCK_ALIASES: dict[str, tuple[str, ...]] = {
    "duration": ("duration", "audio_duration", "track_duration"),
    "bpm": ("bpm",),
    "key_scale": ("key_scale", "keyscale", "keyScale", "key"),
    "time_signature": ("time_signature", "timesignature", "timeSignature"),
    "vocal_language": ("vocal_language", "language"),
}


def _is_auto_metadata_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "auto", "none", "n/a", "na", "null"}


def _metadata_locks(payload: dict[str, Any]) -> dict[str, Any]:
    locks = payload.get("metadata_locks")
    return dict(locks) if isinstance(locks, dict) else {}


def _metadata_field_locked(payload: dict[str, Any], field: str) -> bool:
    aliases = METADATA_LOCK_ALIASES.get(field, (field,))
    locks = _metadata_locks(payload)
    for alias in aliases:
        if alias in locks:
            return parse_bool(locks.get(alias), False)
    for alias in aliases:
        if alias in payload and not _is_auto_metadata_value(payload.get(alias)):
            return True
    return False


def _bpm_from_payload(payload: dict[str, Any]) -> int | None:
    if not _metadata_field_locked(payload, "bpm"):
        return None
    value = payload.get("bpm")
    if _is_auto_metadata_value(value):
        return None
    return clamp_int(value, DEFAULT_BPM, BPM_MIN, BPM_MAX)


def _key_scale_from_payload(payload: dict[str, Any]) -> str:
    if not _metadata_field_locked(payload, "key_scale"):
        return ""
    return normalize_key_scale(get_param(payload, "key_scale", ""))


def _time_signature_from_payload(payload: dict[str, Any]) -> str:
    if not _metadata_field_locked(payload, "time_signature"):
        return ""
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


def _duration_from_payload(payload: dict[str, Any]) -> float:
    if not _metadata_field_locked(payload, "duration"):
        return -1.0
    value = get_param(payload, "duration")
    if _is_auto_metadata_value(value):
        return -1.0
    return clamp_float(value, 60.0, DURATION_MIN, DURATION_MAX)


def _is_standard_ace_step_model(song_model: Any) -> bool:
    model = str(song_model or "").strip().lower()
    return bool(model and "turbo" not in model)


def _official_generation_memory_plan(params: dict[str, Any]) -> dict[str, Any]:
    requested_take_count = clamp_int(params.get("batch_size"), 1, 1, MAX_BATCH_SIZE)
    force_sequential = bool(
        _IS_APPLE_SILICON
        and _is_standard_ace_step_model(params.get("song_model"))
        and str(params.get("task_type") or "text2music") == "text2music"
        and str(params.get("device") or "auto").strip().lower() in {"auto", "mps", "metal"}
    )
    actual_runner_batch_size = 1 if force_sequential else requested_take_count
    render_pass_count = requested_take_count if force_sequential else 1
    policy = "mps_standard_model_sequential_takes" if force_sequential else "normal_batch"
    reason = (
        "Apple Silicon PyTorch/MPS standard ACE-Step models render one take at a time "
        "to avoid XL-SFT/Base out-of-memory failures."
        if force_sequential
        else "Model/batch can use normal official runner batching."
    )
    return {
        "version": "acejam-official-mps-memory-policy-2026-05-07",
        "policy": policy,
        "sequential": force_sequential and requested_take_count > 1,
        "force_runner_batch_size_one": force_sequential,
        "requested_take_count": requested_take_count,
        "actual_runner_batch_size": actual_runner_batch_size,
        "render_pass_count": render_pass_count,
        "song_model": str(params.get("song_model") or ""),
        "duration": params.get("duration"),
        "inference_steps": params.get("inference_steps"),
        "shift": params.get("shift"),
        "use_lora": bool(params.get("use_lora")),
        "lora_scale": params.get("lora_scale"),
        "reason": reason,
    }


def _seed_for_official_take(params: dict[str, Any], take_index: int) -> str:
    if parse_bool(params.get("use_random_seed"), False):
        return "-1"
    seed_text = str(params.get("seed") or "").strip()
    if seed_text in {"", "-1"}:
        return "-1"
    seeds = [item.strip() for item in seed_text.split(",") if item.strip()]
    if not seeds:
        return "-1"
    if take_index < len(seeds):
        return seeds[take_index]
    try:
        return str(int(float(seeds[0])) + take_index)
    except (TypeError, ValueError):
        return seeds[0]


def _official_take_params(params: dict[str, Any], memory_plan: dict[str, Any], take_index: int) -> dict[str, Any]:
    take_params = dict(params)
    take_params["batch_size"] = int(memory_plan.get("actual_runner_batch_size") or 1)
    if memory_plan.get("force_runner_batch_size_one"):
        take_params["batch_size"] = 1
    seed = _seed_for_official_take(params, take_index)
    take_params["seed"] = seed
    take_params["use_random_seed"] = seed == "-1"
    take_params["requested_take_count"] = int(memory_plan.get("requested_take_count") or 1)
    take_params["actual_runner_batch_size"] = int(memory_plan.get("actual_runner_batch_size") or take_params["batch_size"])
    take_params["memory_policy"] = memory_plan
    return take_params


def _vocal_language_from_payload(payload: dict[str, Any]) -> str:
    if not _metadata_field_locked(payload, "vocal_language"):
        return "unknown"
    return _language_for_generation(str(get_param(payload, "vocal_language", "unknown") or "unknown"))


def _effective_metadata_locks(payload: dict[str, Any]) -> dict[str, bool]:
    return {field: _metadata_field_locked(payload, field) for field in METADATA_LOCK_ALIASES}


def _default_quality_profile_for_payload(payload: dict[str, Any], task_type: str | None = None) -> str:
    if payload.get("quality_profile"):
        return normalize_quality_profile(payload.get("quality_profile"))
    mode = str(payload.get("ui_mode") or payload.get("mode") or "").strip().lower()
    task = normalize_task_type(task_type or payload.get("task_type"))
    if mode == "simple" and task == "text2music":
        return QUALITY_PROFILE_DOCS_DAILY
    return DEFAULT_QUALITY_PROFILE


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


def _decode_art_image_bytes(raw: dict[str, Any]) -> tuple[bytes, str]:
    value = ""
    if isinstance(raw, dict):
        if raw.get("image"):
            value = str(raw.get("image") or "")
        elif isinstance(raw.get("images"), list) and raw["images"]:
            value = str(raw["images"][0] or "")
        elif raw.get("response") and str(raw.get("response")).startswith("data:image/"):
            value = str(raw.get("response") or "")
    if not value:
        raise RuntimeError("Ollama did not return image bytes.")
    ext = "png"
    if value.startswith("data:image/"):
        header, value = value.split(",", 1)
        mime = header.split(";", 1)[0].split(":", 1)[-1]
        if "jpeg" in mime or "jpg" in mime:
            ext = "jpg"
        elif "webp" in mime:
            ext = "webp"
    return base64.b64decode(value), ext


def _art_prompt_from_body(body: dict[str, Any], settings: dict[str, Any]) -> str:
    prompt = str(body.get("prompt") or "").strip()
    if prompt:
        return prompt
    title = str(body.get("title") or body.get("album_title") or body.get("scope") or "MLX Media release").strip()
    caption = str(body.get("caption") or body.get("tags") or body.get("album_concept") or "").strip()
    scope = str(body.get("scope") or "single").strip().lower()
    release_kind = "album cover" if scope == "album" else "single cover"
    parts = [
        f"Professional square {release_kind} for {title}",
        caption,
        "cinematic, music-industry cover art, high detail, no typography, no watermark",
    ]
    return ", ".join(part for part in parts if part)


def _write_art_metadata(art_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    art_dir = _resolve_child(ART_DIR, safe_id(art_id))
    art_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **_jsonable(metadata),
        "art_id": safe_id(art_id),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (art_dir / "art.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _attach_art_to_result(result_id: str, art: dict[str, Any]) -> None:
    result_id = safe_id(str(result_id or ""))
    if not result_id:
        return
    meta_path = _result_meta_path(result_id)
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    meta = _load_result_meta(result_id)
    meta["art"] = _jsonable(art)
    meta["single_art"] = _jsonable(art)
    for audio in meta.get("audios") or []:
        if isinstance(audio, dict):
            audio["art"] = _jsonable(art)
            if audio.get("song_id"):
                _merge_song_album_metadata(str(audio["song_id"]), {"art": art, "single_art": art})
    meta_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")


def _attach_art_to_album_family(family_id: str, art: dict[str, Any]) -> None:
    family_id = safe_id(str(family_id or ""))
    if not family_id:
        return
    family_manifest = _load_album_manifest(family_id)
    family_manifest["album_art"] = _jsonable(art)
    for model_album in family_manifest.get("model_albums") or []:
        if isinstance(model_album, dict):
            model_album["album_art"] = _jsonable(art)
            album_id = str(model_album.get("album_id") or "")
            if not album_id:
                continue
            try:
                album_manifest = _load_album_manifest(album_id)
                album_manifest["album_art"] = _jsonable(art)
                for track in album_manifest.get("tracks") or []:
                    if isinstance(track, dict):
                        track.setdefault("album_art", _jsonable(art))
                _write_album_manifest(album_id, album_manifest)
            except Exception as exc:
                print(f"[art] album art attach skipped for {album_id}: {exc}", flush=True)
    _write_album_manifest(family_id, family_manifest)


def _generate_art_asset(body: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("Ollama image generation is disabled in MLX Media. Use MFLUX instead.")


def _maybe_auto_generate_single_art(result: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | None:
    return None


def _maybe_auto_generate_album_art(
    family_id: str,
    concept: str,
    album_options: dict[str, Any],
    logs: list[str],
) -> dict[str, Any] | None:
    return None


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


def _art_path_from_metadata(art: dict[str, Any] | None) -> Path | None:
    if not isinstance(art, dict):
        return None
    raw_path = str(art.get("path") or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path.resolve()
    url = str(art.get("url") or art.get("image_url") or "").strip()
    try:
        if "/media/mflux/" in url:
            parts = url.split("/media/mflux/", 1)[1].split("/")
            if len(parts) >= 2 and parts[0] != "uploads":
                path = _resolve_child(MFLUX_RESULTS_DIR, safe_id(parts[0]), parts[-1])
                if path.is_file():
                    return path
        if "/media/art/" in url:
            parts = url.split("/media/art/", 1)[1].split("/")
            if len(parts) >= 2:
                path = _resolve_child(ART_DIR, safe_id(parts[0]), parts[-1])
                if path.is_file():
                    return path
    except Exception:
        return None
    return None


def _add_album_art_to_zip(zipf: zipfile.ZipFile, art: dict[str, Any] | None, arc_stem: str) -> None:
    path = _art_path_from_metadata(art)
    if not path:
        return
    arc_path = Path(arc_stem)
    folder = "/".join(safe_filename(part, "art") for part in arc_path.parts[:-1])
    safe_stem = safe_filename(arc_path.name, "art")
    arcname = f"{safe_stem}{path.suffix.lower() or '.png'}"
    if folder:
        arcname = f"{folder}/{arcname}"
    zipf.write(path, arcname)


def _video_path_from_attachment(video: dict[str, Any] | None) -> Path | None:
    if not isinstance(video, dict):
        return None
    raw_path = str(video.get("path") or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path.resolve()
    url = str(video.get("url") or video.get("video_url") or "").strip()
    try:
        if "/media/mlx-video/" in url:
            parts = url.split("/media/mlx-video/", 1)[1].split("/")
            if len(parts) >= 2 and parts[0] != "uploads":
                path = _resolve_child(MLX_VIDEO_RESULTS_DIR, safe_id(parts[0]), parts[-1])
                if path.is_file():
                    return path
    except Exception:
        return None
    return None


def _add_album_video_to_zip(zipf: zipfile.ZipFile, video: dict[str, Any], arc_stem: str) -> None:
    path = _video_path_from_attachment(video)
    if not path:
        return
    arc_path = Path(arc_stem)
    folder = "/".join(safe_filename(part, "video") for part in arc_path.parts[:-1])
    safe_stem = safe_filename(arc_path.name, "video")
    arcname = f"{safe_stem}{path.suffix.lower() or '.mp4'}"
    if folder:
        arcname = f"{folder}/{arcname}"
    zipf.write(path, arcname)


def _album_track_video_attachments(album_id: str, track: dict[str, Any], index: int) -> list[dict[str, Any]]:
    target_ids = {
        str(track.get("track_id") or ""),
        str(track.get("id") or ""),
        str(track.get("result_id") or ""),
        f"{album_id}:track:{index}",
        f"{safe_id(album_id)}:track:{index}",
    }
    for audio in track.get("audios", []) if isinstance(track.get("audios"), list) else []:
        if isinstance(audio, dict):
            target_ids.add(str(audio.get("song_id") or ""))
            target_ids.add(str(audio.get("result_id") or ""))
    target_ids.discard("")
    videos: list[dict[str, Any]] = []
    for attachment in mlx_video_list_attachments():
        if not isinstance(attachment, dict):
            continue
        if str(attachment.get("target_id") or "") in target_ids:
            videos.append(attachment)
    return videos


def _build_album_zip(album_id: str) -> Path:
    manifest = _load_album_manifest(album_id)
    zip_path = _resolve_child(ALBUMS_DIR, safe_id(album_id), f"{safe_filename(album_id, 'album')}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("album.json", json.dumps(_jsonable(manifest), indent=2))
        _add_album_art_to_zip(zipf, manifest.get("album_art"), "art/album_art")
        for video_index, video in enumerate(mlx_video_list_attachments(target_type="album", target_id=album_id), start=1):
            _add_album_video_to_zip(zipf, video, f"video/album_video_{video_index:02d}")
        for index, track in enumerate(manifest.get("tracks", []), start=1):
            if not isinstance(track, dict):
                continue
            _add_album_art_to_zip(zipf, track.get("art") or track.get("single_art") or track.get("album_art"), f"art/track_{index:02d}_art")
            for video_index, video in enumerate(_album_track_video_attachments(album_id, track, index), start=1):
                _add_album_video_to_zip(zipf, video, f"video/track_{index:02d}_video_{video_index:02d}")
            for audio in track.get("audios", []):
                _add_album_audio_to_zip(zipf, track, audio)
    return zip_path


def _build_album_family_zip(family_id: str) -> Path:
    family_manifest = _load_album_manifest(family_id)
    zip_path = _resolve_child(ALBUMS_DIR, safe_id(family_id), f"{safe_filename(family_id, 'album-family')}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("album_family.json", json.dumps(_jsonable(family_manifest), indent=2))
        _add_album_art_to_zip(zipf, family_manifest.get("album_art"), "art/album_family_art")
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
            _add_album_art_to_zip(zipf, manifest.get("album_art") or model_album.get("album_art"), f"{folder}/art/album_art")
            for video_index, video in enumerate(mlx_video_list_attachments(target_type="album", target_id=album_id), start=1):
                _add_album_video_to_zip(zipf, video, f"{folder}/video/album_video_{video_index:02d}")
            for index, track in enumerate(manifest.get("tracks", []), start=1):
                if not isinstance(track, dict):
                    continue
                _add_album_art_to_zip(zipf, track.get("art") or track.get("single_art") or track.get("album_art"), f"{folder}/art/track_{index:02d}_art")
                for video_index, video in enumerate(_album_track_video_attachments(album_id, track, index), start=1):
                    _add_album_video_to_zip(zipf, video, f"{folder}/video/track_{index:02d}_video_{video_index:02d}")
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


_GENERATION_PROMPT_2PAC_RE = re.compile(r"(?<![A-Za-z0-9])2\s*[-_ ]?\s*pac(?![A-Za-z0-9])", re.IGNORECASE)


def _apply_generation_safe_prompt_tokens(payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    preserve_raw_lora_trigger = parse_bool(
        updated.get("preserve_raw_lora_trigger_caption")
        or updated.get("allow_raw_lora_trigger_caption"),
        False,
    ) and parse_bool(updated.get("use_lora"), False)
    if preserve_raw_lora_trigger:
        warnings = list(updated.get("payload_warnings") or [])
        warning = "generation_prompt_token_2pac_preserved_for_lora_trigger_test"
        if warning not in warnings:
            warnings.append(warning)
        updated["payload_warnings"] = warnings
        return updated
    changed = False
    for key in ("caption", "tags", "global_caption", "custom_tags"):
        value = updated.get(key)
        if not isinstance(value, str) or "2" not in value:
            continue
        normalized = _GENERATION_PROMPT_2PAC_RE.sub("pac", value)
        if normalized != value:
            updated[key] = normalized
            changed = True
    if changed:
        warnings = list(updated.get("payload_warnings") or [])
        warning = "generation_prompt_token_2pac_normalized_to_pac_for_vocal_clarity"
        if warning not in warnings:
            warnings.append(warning)
        updated["payload_warnings"] = warnings
    return updated


def _parse_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _merge_nested_generation_metadata(payload)
    task_type = normalize_task_type(payload.get("task_type"))
    payload = normalize_generation_text_fields(payload, task_type=task_type)
    payload = _apply_generation_safe_prompt_tokens(payload)
    payload = apply_audio_style_conditioning(payload)
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
    quality_profile = _default_quality_profile_for_payload(payload, task_type)
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
    default_batch_size = 1
    batch_size = clamp_int(requested_batch_size, default_batch_size, 1, MAX_BATCH_SIZE)
    planner_settings = planner_llm_settings_from_payload(payload)
    duration = _duration_from_payload(payload)
    model_defaults = quality_profile_model_settings(song_model, quality_profile)
    raw_steps = payload.get("inference_steps", payload.get("infer_step"))
    inference_steps = _docs_correct_render_steps(song_model, quality_profile, raw_steps)

    bpm = _bpm_from_payload(payload)
    time_signature = _time_signature_from_payload(payload)

    vocal_clarity_recovery = _vocal_clarity_recovery_enabled(payload)
    requested_lm_model = _requested_ace_lm_model(payload)
    official_used = _active_official_fields(payload, task_type, official_fields_used(payload))
    profile_requires_official = quality_profile in {QUALITY_PROFILE_DOCS_DAILY, DEFAULT_QUALITY_PROFILE}
    audio_backend_requested = "audio_backend" in payload or "use_mlx_dit" in payload
    use_official = (
        profile_requires_official
        or bool(official_used)
        or _quality_lm_controls_enabled(payload, task_type)
        or audio_backend_requested
    )
    requested_format = str(payload.get("audio_format") or (model_defaults["audio_format"] if use_official else "wav")).strip().lower().lstrip(".")
    if use_official and requested_format == "ogg":
        raise ValueError("OGG is only available in the fast MLX Media runner. Use wav/flac/mp3/opus/aac/wav32 with official ACE-Step controls.")
    vocal_language = _vocal_language_from_payload(payload)
    track_names = normalize_track_names(payload.get("track_names") or payload.get("track_classes") or payload.get("track_name"))
    instruction = str(payload.get("instruction") or "").strip() or build_task_instruction(task_type, track_names)

    if task_type in {"cover", "cover-nofsq", "repaint", "extract", "lego", "complete"}:
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
    allow_supplied_lyrics_lm = parse_bool(payload.get("allow_supplied_lyrics_lm"), False)
    direct_lyrics_render = (
        task_type == "text2music"
        and supplied_vocal_lyrics
        and not allow_supplied_lyrics_lm
    )
    if direct_lyrics_render:
        requested_lm_model = "none"
        sample_mode = False
        sample_query = ""
        lm_controls = []
    lm_requested_explicitly = any(key in payload for key in ["ace_lm_model", "lm_model", "lm_model_path", "use_official_lm"])
    if (
        requested_lm_model == "none"
        and not lm_requested_explicitly
        and not direct_lyrics_render
        and task_type not in DOCS_BEST_SOURCE_TASK_LM_SKIPS
        and (not supplied_vocal_lyrics or allow_supplied_lyrics_lm)
    ):
        requested_lm_model = recommended_lm_model(_installed_lm_models(), quality_profile)
    lm_quality_defaults = (
        requested_lm_model != "none"
        and task_type not in DOCS_BEST_SOURCE_TASK_LM_SKIPS
        and (not supplied_vocal_lyrics or allow_supplied_lyrics_lm)
    )
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
    _validate_lora_request_for_song_model(lora_request, song_model)
    payload_warnings = list(payload.get("payload_warnings") or [])
    render_shift = _docs_correct_render_shift(song_model, quality_profile, payload.get("shift"))
    audio_code_string = str(get_param(payload, "audio_code_string", "") or "")
    src_audio = None if task_type == "text2music" else _resolve_audio_reference(payload, "src_audio_id", "src_result_id")
    if task_type == "text2music" and audio_code_string.strip() and not parse_bool(payload.get("allow_text2music_audio_codes"), False):
        audio_code_string = ""
        payload_warnings.append("audio_code_hints_cleared_for_text2music_direct_render")
    reference_audio = _resolve_audio_reference(payload, "reference_audio_id", "reference_result_id")
    dcw_mode = str(payload.get("dcw_mode") or "double").strip().lower()
    if dcw_mode not in {"low", "high", "double", "pix"}:
        dcw_mode = "double"
    dcw_wavelet = str(payload.get("dcw_wavelet") or "haar").strip() or "haar"
    retake_seed = str(payload.get("retake_seed") or "").strip()
    flow_edit_n_min = clamp_float(payload.get("flow_edit_n_min"), 0.0, 0.0, 1.0)
    flow_edit_n_max = clamp_float(payload.get("flow_edit_n_max"), 1.0, 0.0, 1.0)
    if flow_edit_n_max < flow_edit_n_min:
        flow_edit_n_min, flow_edit_n_max = flow_edit_n_max, flow_edit_n_min
    parsed = {
        "ui_mode": str(payload.get("ui_mode") or task_type),
        "quality_profile": quality_profile,
        "task_type": task_type,
        "caption": str(payload.get("caption") or ""),
        "global_caption": str(payload.get("global_caption") or ""),
        "lyrics": lyrics_text,
        "style_profile": str(payload.get("style_profile") or ""),
        "style_caption_tags": str(payload.get("style_caption_tags") or ""),
        "style_lyric_tags_applied": list(payload.get("style_lyric_tags_applied") or []),
        "style_conditioning_audit": payload.get("style_conditioning_audit") if isinstance(payload.get("style_conditioning_audit"), dict) else {},
        "caption_source": str(payload.get("caption_source") or "caption"),
        "lyrics_source": str(payload.get("lyrics_source") or "lyrics"),
        "song_intent": payload.get("song_intent") if isinstance(payload.get("song_intent"), dict) else {},
        "source_task_intent": str(payload.get("source_task_intent") or "").strip(),
        "tag_list": list(payload.get("tag_list") or []),
        "payload_warnings": payload_warnings,
        "ace_step_text_budget": dict(payload.get("ace_step_text_budget") or {}),
        "payload_quality_gate": payload.get("payload_quality_gate") if isinstance(payload.get("payload_quality_gate"), dict) else {},
        "payload_gate_status": str(payload.get("payload_gate_status") or ""),
        "payload_gate_passed": parse_bool(payload.get("payload_gate_passed"), False),
        "payload_gate_blocking_issues": list(payload.get("payload_gate_blocking_issues") or []),
        "tag_coverage": payload.get("tag_coverage") if isinstance(payload.get("tag_coverage"), dict) else {},
        "caption_integrity": payload.get("caption_integrity") if isinstance(payload.get("caption_integrity"), dict) else {},
        "lyric_duration_fit": payload.get("lyric_duration_fit") if isinstance(payload.get("lyric_duration_fit"), dict) else {},
        "repair_actions": list(payload.get("repair_actions") or []),
        "instrumental": instrumental,
        "vocal_clarity_recovery": vocal_clarity_recovery,
        "duration": duration,
        "metadata_locks": _effective_metadata_locks(payload),
        "bpm": bpm,
        "key_scale": _key_scale_from_payload(payload),
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "batch_size": batch_size,
        "seed": str(payload.get("seeds") or payload.get("seed") or "-1"),
        "use_random_seed": parse_bool(payload.get("use_random_seed"), str(payload.get("seeds") or payload.get("seed") or "-1").strip() in {"", "-1"}),
        "song_model": song_model,
        "ace_lm_model": requested_lm_model,
        "lm_model_path": requested_lm_model,
        "planner_lm_provider": normalize_provider(payload.get("planner_lm_provider") or payload.get("planner_provider") or "ollama"),
        "planner_model": str(payload.get("planner_model") or payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        "planner_ollama_model": str(payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        **planner_settings,
        "reference_audio": reference_audio,
        "src_audio": src_audio,
        "audio_code_string": audio_code_string,
        "repainting_start": clamp_float(payload.get("repainting_start"), 0.0, -DURATION_MAX, DURATION_MAX),
        "repainting_end": None if payload.get("repainting_end") in [None, "", "end"] else clamp_float(payload.get("repainting_end"), -1.0, -1.0, DURATION_MAX),
        "instruction": instruction,
        "audio_cover_strength": clamp_float(get_param(payload, "audio_cover_strength", 1.0), 1.0, 0.0, 1.0),
        "cover_noise_strength": clamp_float(payload.get("cover_noise_strength"), 0.0, 0.0, 1.0),
        "inference_steps": inference_steps,
        "guidance_scale": clamp_float(payload.get("guidance_scale"), model_defaults["guidance_scale"], 1.0, 15.0),
        "shift": render_shift,
        "infer_method": "sde" if str(payload.get("infer_method") or model_defaults["infer_method"]).lower() == "sde" else "ode",
        "sampler_mode": "euler" if str(payload.get("sampler_mode") or model_defaults["sampler_mode"]).lower() == "euler" else "heun",
        "velocity_norm_threshold": clamp_float(payload.get("velocity_norm_threshold"), 0.0, 0.0, 20.0),
        "velocity_ema_factor": clamp_float(payload.get("velocity_ema_factor"), 0.0, 0.0, 1.0),
        "dcw_enabled": parse_bool(payload.get("dcw_enabled"), True),
        "dcw_mode": dcw_mode,
        "dcw_scaler": clamp_float(payload.get("dcw_scaler"), 0.05, 0.0, 0.2),
        "dcw_high_scaler": clamp_float(payload.get("dcw_high_scaler"), 0.02, 0.0, 0.2),
        "dcw_wavelet": dcw_wavelet,
        "retake_seed": retake_seed,
        "retake_variance": clamp_float(payload.get("retake_variance"), 0.0, 0.0, 1.0),
        "flow_edit_morph": parse_bool(payload.get("flow_edit_morph"), False),
        "flow_edit_source_caption": str(payload.get("flow_edit_source_caption") or "").strip(),
        "flow_edit_source_lyrics": str(payload.get("flow_edit_source_lyrics") or "").strip(),
        "flow_edit_n_min": flow_edit_n_min,
        "flow_edit_n_max": flow_edit_n_max,
        "flow_edit_n_avg": clamp_int(payload.get("flow_edit_n_avg"), 1, 1, 16),
        "use_adg": parse_bool(payload.get("use_adg"), bool(model_defaults.get("use_adg", False))),
        "cfg_interval_start": clamp_float(payload.get("cfg_interval_start"), 0.0, 0.0, 1.0),
        "cfg_interval_end": clamp_float(payload.get("cfg_interval_end"), 1.0, 0.0, 1.0),
        "timesteps": parse_timesteps(payload.get("timesteps")),
        "audio_format": normalize_audio_format(payload.get("audio_format") or (model_defaults["audio_format"] if use_official else "wav"), allow_official=use_official),
        "mp3_bitrate": str(payload.get("mp3_bitrate") or "128k").strip() or "128k",
        "mp3_sample_rate": clamp_int(payload.get("mp3_sample_rate"), 48000, 16000, 48000),
        "auto_score": parse_bool(payload.get("auto_score"), False),
        "auto_lrc": parse_bool(payload.get("auto_lrc"), False),
        "vocal_intelligibility_gate": parse_bool(payload.get("vocal_intelligibility_gate"), ACEJAM_VOCAL_INTELLIGIBILITY_GATE),
        "vocal_intelligibility_attempts": clamp_int(
            payload.get("vocal_intelligibility_attempts") or payload.get("vocal_intelligibility_retries"),
            ACEJAM_VOCAL_INTELLIGIBILITY_ATTEMPTS,
            1,
            32,
        ),
        "vocal_intelligibility_model_rescue": parse_bool(
            payload.get("vocal_intelligibility_model_rescue"),
            ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE,
        ),
        "vocal_intelligibility_model_rescue_after": clamp_int(
            payload.get("vocal_intelligibility_model_rescue_after"),
            ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_AFTER,
            1,
            32,
        ),
        "vocal_intelligibility_model_rescue_attempts": clamp_int(
            payload.get("vocal_intelligibility_model_rescue_attempts"),
            ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_ATTEMPTS,
            1,
            32,
        ),
        "vocal_intelligibility_rescue_models": payload.get(
            "vocal_intelligibility_rescue_models",
            ACEJAM_VOCAL_INTELLIGIBILITY_RESCUE_MODELS,
        ),
        "lora_preflight_required": parse_bool(payload.get("lora_preflight_required"), False),
        "return_audio_codes": parse_bool(payload.get("return_audio_codes"), False),
        "save_to_library": parse_bool(payload.get("save_to_library"), False),
        "title": title,
        "artist_name": artist_name,
        "description": str(payload.get("description") or "").strip(),
        **lora_request,
        "album_metadata": payload.get("album_metadata") if isinstance(payload.get("album_metadata"), dict) else {},
        "track_names": track_names,
        "thinking": False if direct_lyrics_render else (True if vocal_clarity_recovery and supplied_vocal_lyrics else parse_bool(payload.get("thinking"), lm_quality_defaults)),
        "sample_mode": sample_mode,
        "sample_query": sample_query,
        "use_format": False if direct_lyrics_render else (True if vocal_clarity_recovery and supplied_vocal_lyrics else parse_bool(get_param(payload, "use_format"), True if vocal_clarity_recovery else lm_quality_defaults)),
        "analysis_only": parse_bool(payload.get("analysis_only"), False),
        "full_analysis_only": parse_bool(payload.get("full_analysis_only"), False),
        "extract_codes_only": parse_bool(payload.get("extract_codes_only"), False),
        "use_tiled_decode": parse_bool(payload.get("use_tiled_decode"), True),
        "is_format_caption": parse_bool(payload.get("is_format_caption"), False),
        "lm_temperature": clamp_float(payload.get("lm_temperature"), 0.7 if vocal_clarity_recovery else (DOCS_BEST_LM_DEFAULTS["lm_temperature"] if lm_quality_defaults else 0.85), 0.0, 2.0),
        "lm_cfg_scale": clamp_float(payload.get("lm_cfg_scale"), DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"] if lm_quality_defaults else 2.0, 0.0, 10.0),
        "lm_repetition_penalty": clamp_float(payload.get("lm_repetition_penalty") or payload.get("repetition_penalty"), 1.0, 0.1, 4.0),
        "lm_top_k": clamp_int(payload.get("lm_top_k"), DOCS_BEST_LM_DEFAULTS["lm_top_k"] if lm_quality_defaults else 0, 0, 200),
        "lm_top_p": clamp_float(payload.get("lm_top_p"), DOCS_BEST_LM_DEFAULTS["lm_top_p"] if lm_quality_defaults else 0.9, 0.0, 1.0),
        "lm_negative_prompt": str(payload.get("lm_negative_prompt") or "NO USER INPUT"),
        "lm_backend": _normalize_lm_backend(payload.get("lm_backend")),
        "use_cot_metas": False if direct_lyrics_render else parse_bool(payload.get("use_cot_metas"), False if vocal_clarity_recovery else lm_quality_defaults),
        "use_cot_caption": False if direct_lyrics_render else (False if vocal_clarity_recovery and supplied_vocal_lyrics else parse_bool(payload.get("use_cot_caption"), True if vocal_clarity_recovery else lm_quality_defaults)),
        "use_cot_lyrics": False if direct_lyrics_render else parse_bool(payload.get("use_cot_lyrics"), False),
        "use_cot_language": False if direct_lyrics_render else (True if vocal_clarity_recovery and supplied_vocal_lyrics else parse_bool(payload.get("use_cot_language"), True if vocal_clarity_recovery else lm_quality_defaults)),
        "allow_lm_batch": parse_bool(payload.get("allow_lm_batch"), False),
        "lm_batch_chunk_size": clamp_int(payload.get("lm_batch_chunk_size"), 8, 1, 64),
        "use_constrained_decoding": parse_bool(payload.get("use_constrained_decoding"), True),
        "constrained_decoding_debug": parse_bool(payload.get("constrained_decoding_debug"), False),
        "chunk_mask_mode": "explicit" if str(payload.get("chunk_mask_mode")).lower() == "explicit" else "auto",
        "track_name": track_names[0] if track_names else "",
        "track_classes": track_names,
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
        "vae_checkpoint": str(payload.get("vae_checkpoint") or "official").strip() or "official",
        "audio_backend": _normalize_audio_backend(payload.get("audio_backend"), payload.get("use_mlx_dit")),
        "use_flash_attention": payload.get("use_flash_attention", "auto"),
        "use_mlx_dit": payload.get("use_mlx_dit", "auto"),
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
    if parsed.get("style_profile") and parsed.get("style_profile") != "auto":
        parsed["tag_list"] = split_terms(parsed["caption"])
    _apply_audio_backend_defaults(parsed, source="parse")
    if vocal_clarity_recovery and task_type == "text2music" and supplied_vocal_lyrics:
        parsed["caption"] = _caption_with_vocal_clarity_traits(parsed["caption"])
        parsed["tag_list"] = split_terms(parsed["caption"])
        if "vocal_clarity_recovery_caption_traits" not in parsed["payload_warnings"]:
            parsed["payload_warnings"].append("vocal_clarity_recovery_caption_traits")
    _apply_lora_trigger_conditioning(parsed)
    _enforce_model_correct_render_settings(parsed, source="parse")
    _apply_mac_mlx_xl_repetition_guard(parsed, source="parse")
    _apply_mps_long_lora_memory_guard(parsed, source="parse")
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
    track_names = normalize_track_names(payload.get("track_names") or payload.get("track_classes") or payload.get("track_name"))
    quality_profile = _default_quality_profile_for_payload(payload, task_type)
    requested_lm_model = _requested_ace_lm_model(payload)
    official_used = _active_official_fields(payload, task_type, official_used)
    profile_requires_official = quality_profile in {QUALITY_PROFILE_DOCS_DAILY, DEFAULT_QUALITY_PROFILE}
    use_official = profile_requires_official or bool(official_used) or _quality_lm_controls_enabled(payload, task_type)
    model_defaults = quality_profile_model_settings(song_model, quality_profile)
    requested_format = str(payload.get("audio_format") or (model_defaults["audio_format"] if use_official else "wav")).strip().lower().lstrip(".")
    time_signature = _time_signature_from_payload(payload)
    bpm = _bpm_from_payload(payload)
    raw_steps = payload.get("inference_steps", payload.get("infer_step"))
    inference_steps = _docs_correct_render_steps(song_model, quality_profile, raw_steps)
    title = str(payload.get("title") or "").strip() or "Untitled"
    artist_name = _artist_name_from_payload(payload, title=title)
    preview_warnings = list(payload.get("payload_warnings") or [])
    preview_audio_code_string = str(get_param(payload, "audio_code_string", "") or "")
    if task_type == "text2music" and preview_audio_code_string.strip() and not parse_bool(payload.get("allow_text2music_audio_codes"), False):
        preview_audio_code_string = ""
        preview_warnings.append("audio_code_hints_cleared_for_text2music_direct_render")
    dcw_mode = str(payload.get("dcw_mode") or "double").strip().lower()
    if dcw_mode not in {"low", "high", "double", "pix"}:
        dcw_mode = "double"
    flow_edit_n_min = clamp_float(payload.get("flow_edit_n_min"), 0.0, 0.0, 1.0)
    flow_edit_n_max = clamp_float(payload.get("flow_edit_n_max"), 1.0, 0.0, 1.0)
    if flow_edit_n_max < flow_edit_n_min:
        flow_edit_n_min, flow_edit_n_max = flow_edit_n_max, flow_edit_n_min
    preview = {
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
        "song_intent": payload.get("song_intent") if isinstance(payload.get("song_intent"), dict) else {},
        "source_task_intent": str(payload.get("source_task_intent") or "").strip(),
        "tag_list": list(payload.get("tag_list") or []),
        "payload_warnings": preview_warnings,
        "ace_step_text_budget": dict(payload.get("ace_step_text_budget") or {}),
        "instrumental": parse_bool(payload.get("instrumental"), False),
        "duration": _duration_from_payload(payload),
        "metadata_locks": _effective_metadata_locks(payload),
        "bpm": bpm,
        "key_scale": _key_scale_from_payload(payload),
        "time_signature": time_signature,
        "vocal_language": _vocal_language_from_payload(payload),
        "batch_size": clamp_int(payload.get("batch_size"), 1, 1, MAX_BATCH_SIZE),
        "seed": str(payload.get("seeds") or payload.get("seed") or "-1"),
        "use_random_seed": parse_bool(payload.get("use_random_seed"), str(payload.get("seeds") or payload.get("seed") or "-1").strip() in {"", "-1"}),
        "song_model": song_model,
        "ace_lm_model": _requested_ace_lm_model(payload),
        "lm_model_path": _requested_ace_lm_model(payload),
        "planner_lm_provider": normalize_provider(payload.get("planner_lm_provider") or payload.get("planner_provider") or "ollama"),
        "planner_model": str(payload.get("planner_model") or payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        "planner_ollama_model": str(payload.get("planner_ollama_model") or payload.get("ollama_model") or "").strip(),
        "audio_code_string": preview_audio_code_string,
        "track_names": track_names,
        "track_name": track_names[0] if track_names else "",
        "track_classes": track_names,
        "analysis_only": parse_bool(payload.get("analysis_only"), False),
        "full_analysis_only": parse_bool(payload.get("full_analysis_only"), False),
        "extract_codes_only": parse_bool(payload.get("extract_codes_only"), False),
        "use_tiled_decode": parse_bool(payload.get("use_tiled_decode"), True),
        "is_format_caption": parse_bool(payload.get("is_format_caption"), False),
        "reference_audio_id": str(payload.get("reference_audio_id") or ""),
        "reference_result_id": str(payload.get("reference_result_id") or ""),
        "src_audio_id": "" if task_type == "text2music" else str(payload.get("src_audio_id") or ""),
        "src_result_id": "" if task_type == "text2music" else str(payload.get("src_result_id") or ""),
        "inference_steps": inference_steps,
        "guidance_scale": clamp_float(payload.get("guidance_scale"), model_defaults["guidance_scale"], 1.0, 15.0),
        "shift": _docs_correct_render_shift(song_model, quality_profile, payload.get("shift")),
        "infer_method": "sde" if str(payload.get("infer_method") or model_defaults["infer_method"]).lower() == "sde" else "ode",
        "sampler_mode": "euler" if str(payload.get("sampler_mode") or model_defaults["sampler_mode"]).lower() == "euler" else "heun",
        "dcw_enabled": parse_bool(payload.get("dcw_enabled"), True),
        "dcw_mode": dcw_mode,
        "dcw_scaler": clamp_float(payload.get("dcw_scaler"), 0.05, 0.0, 0.2),
        "dcw_high_scaler": clamp_float(payload.get("dcw_high_scaler"), 0.02, 0.0, 0.2),
        "dcw_wavelet": str(payload.get("dcw_wavelet") or "haar").strip() or "haar",
        "retake_seed": str(payload.get("retake_seed") or "").strip(),
        "retake_variance": clamp_float(payload.get("retake_variance"), 0.0, 0.0, 1.0),
        "flow_edit_morph": parse_bool(payload.get("flow_edit_morph"), False),
        "flow_edit_source_caption": str(payload.get("flow_edit_source_caption") or "").strip(),
        "flow_edit_source_lyrics": str(payload.get("flow_edit_source_lyrics") or "").strip(),
        "flow_edit_n_min": flow_edit_n_min,
        "flow_edit_n_max": flow_edit_n_max,
        "flow_edit_n_avg": clamp_int(payload.get("flow_edit_n_avg"), 1, 1, 16),
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
    _apply_mac_mlx_xl_repetition_guard(preview, source="preview")
    return _apply_mps_long_lora_memory_guard(preview, source="preview")


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
    quality_profile = _default_quality_profile_for_payload(normalized_text, task_type)
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
            "OGG is only available in the fast MLX Media runner. Use wav/flac/mp3/opus/aac/wav32 with official ACE-Step controls.",
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
    if task_type in {"cover", "cover-nofsq", "repaint", "extract", "lego", "complete"} and not source_status["present"]:
        _set_field_error(field_errors, "source", f"{task_type} requires source audio, a source result, or audio codes.")
    if source_status["present"] and not source_status["ok"]:
        _set_field_error(field_errors, "source", source_status.get("error") or "Source audio could not be resolved.")
    if reference_status["present"] and not reference_status["ok"]:
        _set_field_error(field_errors, "reference", reference_status.get("error") or "Reference audio could not be resolved.")

    track_names = normalize_track_names(normalized_text.get("track_names") or normalized_text.get("track_classes") or normalized_text.get("track_name"))
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
    params = _enforce_model_correct_render_settings(dict(params), source="official_request")
    _apply_audio_backend_defaults(params, source="official_request")
    _apply_mps_long_lora_memory_guard(params, source="official_request")
    needs_lm = _requires_lm(params)
    lm_model = _concrete_lm_model(params["ace_lm_model"]) if needs_lm else None
    seed_text = str(params.get("seed") or "-1").strip()
    official_seed = -1
    if not params.get("use_random_seed") and seed_text not in {"", "-1"}:
        first_seed = seed_text.split(",", 1)[0].strip()
        try:
            official_seed = int(float(first_seed))
        except (TypeError, ValueError):
            official_seed = -1
    print(
        f"[generation] Official request: model={params.get('song_model')}, "
        f"lm={lm_model or 'NONE'}, requires_lm={needs_lm}, "
        f"thinking={params.get('thinking')}, cot_metas={params.get('use_cot_metas')}, "
        f"cot_caption={params.get('use_cot_caption')}, cot_lyrics={params.get('use_cot_lyrics')}, "
        f"steps={params.get('inference_steps')}, guidance={params.get('guidance_scale')}, "
        f"audio_backend={params.get('audio_backend')}, use_mlx_dit={params.get('use_mlx_dit')}, "
        f"duration={params.get('duration')}, caption={str(params.get('caption',''))[:80]}",
        flush=True,
    )
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
            "Auto score and Auto LRC need MLX Media's in-process tensor cache. "
            "Disable official-only controls or turn off Auto score/LRC for this run."
        )
    official_api_fields = {
        "analysis_only": bool(params.get("analysis_only")),
        "full_analysis_only": bool(params.get("full_analysis_only")),
        "extract_codes_only": bool(params.get("extract_codes_only")),
        "use_tiled_decode": bool(params.get("use_tiled_decode")),
        "lm_model_path": lm_model or params.get("lm_model_path") or params.get("ace_lm_model") or "none",
        "is_format_caption": bool(params.get("is_format_caption")),
        "track_name": params.get("track_name") or "",
        "track_classes": list(params.get("track_classes") or []),
        "lm_repetition_penalty": params.get("lm_repetition_penalty", 1.0),
    }
    guarded_api_fields = {
        key: value
        for key, value in official_api_fields.items()
        if key in {"analysis_only", "full_analysis_only", "extract_codes_only", "use_tiled_decode", "is_format_caption"}
    }

    request = {
        "base_dir": str(BASE_DIR),
        "vendor_dir": str(OFFICIAL_ACE_STEP_DIR),
        "model_cache_dir": str(MODEL_CACHE_DIR),
        "checkpoint_dir": str(MODEL_CACHE_DIR / "checkpoints"),
        "save_dir": str(save_dir),
        "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "song_model": params["song_model"],
        "vae_checkpoint": params.get("vae_checkpoint") or "official",
        "lm_model": lm_model,
        "official_api_fields": _jsonable(official_api_fields),
        "guarded_api_fields": _jsonable(guarded_api_fields),
        "lm_sampling": {
            "temperature": params.get("lm_temperature", 0.85),
            "top_k": params.get("lm_top_k", 0),
            "top_p": params.get("lm_top_p", 0.9),
            "repetition_penalty": params.get("lm_repetition_penalty", 1.0),
            "use_constrained_decoding": params.get("use_constrained_decoding", True),
            "constrained_decoding_debug": params.get("constrained_decoding_debug", False),
        },
        "requires_lm": needs_lm,
        "use_lora": params.get("use_lora", False),
        "lora_adapter_path": params.get("lora_adapter_path", ""),
        "lora_adapter_name": params.get("lora_adapter_name", ""),
        "use_lora_trigger": params.get("use_lora_trigger", False),
        "lora_trigger_tag": params.get("lora_trigger_tag", ""),
        "lora_trigger_source": params.get("lora_trigger_source", ""),
        "lora_trigger_aliases": _jsonable(params.get("lora_trigger_aliases") or []),
        "lora_trigger_candidates": _jsonable(params.get("lora_trigger_candidates") or []),
        "lora_scale": params.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE),
        "adapter_model_variant": params.get("adapter_model_variant", ""),
        "requested_take_count": params.get("requested_take_count", params.get("batch_size", 1)),
        "actual_runner_batch_size": params.get("actual_runner_batch_size", params.get("batch_size", 1)),
        "memory_policy": _jsonable(params.get("memory_policy") or {}),
        "acejam_skip_lora_base_backup": True,
        "device": params["device"],
        "dtype": params["dtype"],
        "audio_backend": params.get("audio_backend", _normalize_audio_backend(params.get("audio_backend"), params.get("use_mlx_dit"))),
        "use_flash_attention": params["use_flash_attention"],
        "use_mlx_dit": params.get("use_mlx_dit", "auto"),
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
            "seed": official_seed,
            "guidance_scale": params["guidance_scale"],
            "use_adg": params["use_adg"],
            "cfg_interval_start": params["cfg_interval_start"],
            "cfg_interval_end": params["cfg_interval_end"],
            "shift": params["shift"],
            "infer_method": params["infer_method"],
            "sampler_mode": params["sampler_mode"],
            "velocity_norm_threshold": params["velocity_norm_threshold"],
            "velocity_ema_factor": params["velocity_ema_factor"],
            "dcw_enabled": params["dcw_enabled"],
            "dcw_mode": params["dcw_mode"],
            "dcw_scaler": params["dcw_scaler"],
            "dcw_high_scaler": params["dcw_high_scaler"],
            "dcw_wavelet": params["dcw_wavelet"],
            "timesteps": params["timesteps"],
            "repainting_start": params["repainting_start"],
            "repainting_end": params["repainting_end"],
            "chunk_mask_mode": params["chunk_mask_mode"],
            "repaint_latent_crossfade_frames": params["repaint_latent_crossfade_frames"],
            "repaint_wav_crossfade_sec": params["repaint_wav_crossfade_sec"],
            "repaint_mode": params["repaint_mode"],
            "repaint_strength": params["repaint_strength"],
            "retake_seed": params["retake_seed"] or None,
            "retake_variance": params["retake_variance"],
            "flow_edit_morph": params["flow_edit_morph"],
            "flow_edit_source_caption": params["flow_edit_source_caption"],
            "flow_edit_source_lyrics": params["flow_edit_source_lyrics"],
            "flow_edit_n_min": params["flow_edit_n_min"],
            "flow_edit_n_max": params["flow_edit_n_max"],
            "flow_edit_n_avg": params["flow_edit_n_avg"],
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
        "audio_backend": params.get("audio_backend", _normalize_audio_backend(params.get("audio_backend"), params.get("use_mlx_dit"))),
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
    request = _finalize_official_audio_backend_request(params, request)
    _print_ace_step_terminal_payload(params, request, save_dir)
    return request


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
        "metadata_locks": _jsonable(params.get("metadata_locks") or {}),
        "bpm": {"value": effective.get("bpm"), "present": effective.get("bpm") not in [None, ""]},
        "keyscale": {"value": effective.get("keyscale"), "present": effective.get("keyscale") not in [None, ""]},
        "duration": {"value": effective.get("duration"), "present": effective.get("duration") not in [None, ""]},
        "timesignature": {"value": effective.get("timesignature"), "present": effective.get("timesignature") not in [None, ""]},
        "song_model": params.get("song_model"),
        "lm_backend": params.get("lm_backend"),
        "audio_backend": params.get("audio_backend"),
        "use_mlx_dit": params.get("use_mlx_dit"),
        "inference_steps": params.get("inference_steps"),
        "guidance_scale": params.get("guidance_scale"),
        "shift": params.get("shift"),
        "audio_format": params.get("audio_format"),
        "take_count": params.get("batch_size"),
        "use_lora": params.get("use_lora"),
        "lora_adapter_path": params.get("lora_adapter_path"),
        "use_lora_trigger": params.get("use_lora_trigger"),
        "lora_trigger_tag": params.get("lora_trigger_tag"),
        "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
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
        "lm_model_path",
        "lm_backend",
        "audio_backend",
        "use_mlx_dit",
        "lm_repetition_penalty",
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
        "analysis_only",
        "full_analysis_only",
        "extract_codes_only",
        "use_tiled_decode",
        "is_format_caption",
        "track_name",
        "track_classes",
        "dcw_enabled",
        "dcw_mode",
        "dcw_scaler",
        "dcw_high_scaler",
        "dcw_wavelet",
        "retake_seed",
        "retake_variance",
        "flow_edit_morph",
        "flow_edit_n_min",
        "flow_edit_n_max",
        "flow_edit_n_avg",
        "use_adg",
        "timesteps",
        "audio_format",
        "use_lora",
        "lora_adapter_name",
        "lora_adapter_path",
        "use_lora_trigger",
        "lora_trigger_tag",
        "lora_trigger_source",
        "lora_trigger_applied",
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
    try:
        requested_raw_duration = float(params.get("duration") or 0.0)
    except (TypeError, ValueError):
        requested_raw_duration = 0.0
    requested_duration = requested_raw_duration if requested_raw_duration > 0 else 0.0
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
    locks = metadata_audit.get("metadata_locks") if isinstance(metadata_audit.get("metadata_locks"), dict) else {}
    if not metadata_audit.get("metadata_present") and any(locks.values()):
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


VOCAL_GATE_STOPWORDS = {
    "verse",
    "chorus",
    "bridge",
    "intro",
    "outro",
    "hook",
    "final",
    "instrumental",
    "fade",
    "with",
    "that",
    "this",
    "from",
    "into",
    "under",
    "over",
    "through",
    "tonight",
    "light",
    "sound",
    "song",
    "sing",
    "will",
    "your",
    "they",
    "them",
    "then",
    "when",
    "where",
    "what",
    "have",
    "still",
}

VOCAL_GATE_FILLER_WORDS = {
    "ah",
    "aha",
    "ay",
    "eh",
    "er",
    "hey",
    "hm",
    "hmm",
    "la",
    "mmm",
    "na",
    "oh",
    "ooh",
    "uh",
    "um",
    "woah",
    "yeah",
    "yea",
    "yah",
    "yo",
}


def _vocal_gate_words(text: str) -> list[str]:
    return [word.lower().strip("'") for word in re.findall(r"[A-Za-z][A-Za-z']*", str(text or ""))]


def _vocal_phrase_loop_issue(words: list[str]) -> str:
    if len(words) < 4:
        return ""
    known_loops = {
        ("i", "don't", "know"),
        ("i", "don", "t", "know"),
        ("thank", "you"),
        ("we'll", "be", "right", "back"),
        ("we", "ll", "be", "right", "back"),
        ("yeah",),
        ("oh",),
    }
    for phrase in known_loops:
        size = len(phrase)
        if len(words) < size * 2:
            continue
        count = sum(1 for index in range(0, len(words) - size + 1) if tuple(words[index : index + size]) == phrase)
        coverage = (count * size) / max(1, len(words))
        if count >= 2 and coverage >= 0.45:
            label = "_".join(phrase[:4])
            return f"asr_phrase_loop_{label}_{coverage:.2f}"
    for size in [5, 4, 3, 2]:
        counts: dict[tuple[str, ...], int] = {}
        for index in range(0, len(words) - size + 1):
            phrase = tuple(words[index : index + size])
            if len(set(phrase)) <= 1 and phrase[0] not in VOCAL_GATE_FILLER_WORDS:
                continue
            counts[phrase] = counts.get(phrase, 0) + 1
        if not counts:
            continue
        phrase, count = max(counts.items(), key=lambda item: item[1])
        coverage = (count * size) / max(1, len(words))
        if (count >= 3 and coverage >= 0.35) or (count >= 6 and coverage >= 0.18):
            label = "_".join(phrase[:4])
            return f"asr_phrase_loop_{label}_{coverage:.2f}"
    return ""


def _score_vocal_transcript(text: str, expected_keywords: list[str]) -> dict[str, Any]:
    words = _vocal_gate_words(text)
    word_count = len(words)
    word_set = set(words)
    expected = []
    for item in expected_keywords or []:
        keyword = " ".join(_vocal_gate_words(str(item or "")))
        if keyword and keyword not in expected:
            expected.append(keyword)
    lowered_text = " " + " ".join(words) + " "
    hits = [
        keyword
        for keyword in expected
        if keyword in word_set or f" {keyword} " in lowered_text
    ]
    required_keywords = min(ACEJAM_VOCAL_INTELLIGIBILITY_MIN_KEYWORDS, len(expected)) if expected else 0
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    top_word = ""
    top_count = 0
    if counts:
        top_word, top_count = max(counts.items(), key=lambda item: item[1])
    filler_count = sum(1 for word in words if word in VOCAL_GATE_FILLER_WORDS)
    filler_ratio = filler_count / word_count if word_count else 1.0
    repeat_ratio = top_count / word_count if word_count else 1.0
    unique_count = len(word_set)
    issues = []
    if word_count < ACEJAM_VOCAL_INTELLIGIBILITY_MIN_WORDS:
        issues.append(f"asr_words_{word_count}_min_{ACEJAM_VOCAL_INTELLIGIBILITY_MIN_WORDS}")
    if len(hits) < required_keywords:
        issues.append(f"asr_keywords_{len(hits)}_min_{required_keywords}")
    if unique_count < ACEJAM_VOCAL_INTELLIGIBILITY_MIN_UNIQUE_WORDS:
        issues.append(f"asr_unique_{unique_count}_min_{ACEJAM_VOCAL_INTELLIGIBILITY_MIN_UNIQUE_WORDS}")
    if word_count and filler_ratio > ACEJAM_VOCAL_INTELLIGIBILITY_MAX_FILLER_RATIO:
        issues.append(f"asr_filler_ratio_{filler_ratio:.2f}")
    if word_count and repeat_ratio > ACEJAM_VOCAL_INTELLIGIBILITY_MAX_REPEAT_RATIO:
        issues.append(f"asr_repeat_{top_word}_{repeat_ratio:.2f}")
    phrase_loop_issue = _vocal_phrase_loop_issue(words)
    if phrase_loop_issue:
        issues.append(phrase_loop_issue)
    passed = not issues
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "blocking": not passed,
        "word_count": word_count,
        "unique_word_count": unique_count,
        "keyword_hits": hits,
        "missing_keywords": [keyword for keyword in expected if keyword not in hits],
        "filler_ratio": round(filler_ratio, 4),
        "repeat_ratio": round(repeat_ratio, 4),
        "top_repeated_word": top_word,
        "issue": "" if passed else ";".join(issues),
    }


def _append_vocal_gate_debug(event: dict[str, Any]) -> None:
    try:
        path = RESULTS_DIR / "vocal_intelligibility_gate.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[vocal_gate] debug log failed: {exc}", flush=True)


def _resolve_local_whisper_model() -> str | None:
    if ACEJAM_VOCAL_ASR_MODEL:
        return ACEJAM_VOCAL_ASR_MODEL
    candidates: list[Path] = []
    for root in [
        Path.home() / ".cache" / "huggingface" / "hub",
        MODEL_CACHE_DIR / "huggingface" / "hub",
    ]:
        snapshots = root / "models--openai--whisper-large-v3-turbo" / "snapshots"
        if snapshots.is_dir():
            candidates.extend(sorted((p for p in snapshots.iterdir() if p.is_dir()), reverse=True))
    return str(candidates[0]) if candidates else None


def _ffmpeg_subprocess_bin_dirs() -> list[str]:
    dirs: list[Path] = []
    env_bin = os.environ.get("ACEJAM_FFMPEG_BIN", "").strip()
    if env_bin:
        dirs.append(Path(env_bin).expanduser())
    if shutil.which("ffmpeg"):
        try:
            dirs.append(Path(shutil.which("ffmpeg") or "").resolve().parent)
        except Exception:
            pass
    roots: list[Path] = []
    pinokio_home = os.environ.get("PINOKIO_HOME", "").strip()
    if pinokio_home:
        roots.append(Path(pinokio_home).expanduser())
    try:
        roots.append(BASE_DIR.parents[2])
    except IndexError:
        pass
    for root in roots:
        dirs.append(root / "bin" / "ffmpeg-env" / "bin")
        pkgs = root / "bin" / "ffmpeg-pkgs"
        if pkgs.is_dir():
            dirs.extend(path for path in pkgs.glob("*/bin") if path.is_dir())
    unique: list[str] = []
    seen: set[str] = set()
    for path in dirs:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if not (resolved / "ffmpeg").exists():
            continue
        value = str(resolved)
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _vocal_gate_required(params: dict[str, Any]) -> bool:
    if not parse_bool(params.get("vocal_intelligibility_gate"), ACEJAM_VOCAL_INTELLIGIBILITY_GATE):
        return False
    if params.get("instrumental"):
        return False
    lyrics = str(params.get("lyrics") or "").strip()
    if not lyrics or lyrics.lower() == "[instrumental]":
        return False
    return str(params.get("task_type") or "text2music") == "text2music"


def _manual_lora_review_allowed(params: dict[str, Any]) -> bool:
    mode = str(params.get("ui_mode") or params.get("wizard_mode") or "").strip().lower()
    if mode == "lora_benchmark":
        return True
    return parse_bool(
        params.get("manual_lora_review"),
        parse_bool(params.get("allow_manual_lora_review"), False),
    )


def _vocal_gate_expected_keywords(params: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [
            str(params.get("title") or ""),
            str(params.get("artist_name") or ""),
            str(params.get("lyrics") or ""),
        ]
    )
    text = re.sub(r"\[[^\]]+\]", " ", text)
    counts: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower()):
        word = word.strip("'")
        if len(word) < 4 or word in VOCAL_GATE_STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda item: (-counts[item], text.lower().find(item), item))
    return ranked[:12]


def _transcribe_audio_paths(paths: list[Path], *, language: str, expected_keywords: list[str]) -> list[dict[str, Any]]:
    model_path = _resolve_local_whisper_model()
    if not model_path:
        return [
            {
                "path": str(path),
                "status": "error",
                "passed": False,
                "blocking": True,
                "issue": "asr_model_unavailable",
                "text": "",
                "word_count": 0,
                "unique_word_count": 0,
                "keyword_hits": [],
                "missing_keywords": expected_keywords,
            }
            for path in paths
        ]
    request = {
        "paths": [str(path) for path in paths],
        "model_path": model_path,
        "device": ACEJAM_VOCAL_ASR_DEVICE,
        "language": language if language and language != "unknown" else "english",
        "expected_keywords": expected_keywords,
        "min_words": ACEJAM_VOCAL_INTELLIGIBILITY_MIN_WORDS,
        "min_keywords": ACEJAM_VOCAL_INTELLIGIBILITY_MIN_KEYWORDS,
    }
    code = r'''
import json
import re
import sys
import traceback

import torch
from transformers import pipeline

payload = json.loads(sys.stdin.read() or "{}")
requested_device = str(payload.get("device") or "cpu").strip().lower()
if requested_device in {"mps", "metal"} and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
elif requested_device in {"cuda", "gpu"} and torch.cuda.is_available():
    device = "cuda"
else:
    # Whisper large-v3-turbo on this Apple/MPS stack returned punctuation-only
    # transcripts for ordinary speech. CPU/float32 is slower but reliable.
    device = "cpu"
dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
pipe = pipeline(
    "automatic-speech-recognition",
    model=payload["model_path"],
    device=device,
    torch_dtype=dtype,
)
expected = [str(item).lower() for item in payload.get("expected_keywords") or []]
min_words = int(payload.get("min_words") or 3)
min_keywords = int(payload.get("min_keywords") or 1)
language = str(payload.get("language") or "english")
results = []
for raw_path in payload.get("paths") or []:
    item = {
        "path": raw_path,
        "status": "fail",
        "passed": False,
        "blocking": True,
        "text": "",
        "word_count": 0,
        "keyword_hits": [],
        "missing_keywords": expected,
    }
    try:
        output = pipe(
            raw_path,
            return_timestamps=True,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        text = re.sub(r"\s+", " ", str(output.get("text") or "")).strip()
        words = [w.lower().strip("'") for w in re.findall(r"[A-Za-z][A-Za-z']*", text)]
        word_set = set(words)
        lowered_text = " " + " ".join(words) + " "
        hits = [kw for kw in expected if kw in word_set or f" {kw} " in lowered_text]
        required_keywords = min(min_keywords, len(expected)) if expected else 0
        passed = len(words) >= min_words and len(hits) >= required_keywords
        item.update(
            {
                "status": "pass" if passed else "fail",
                "passed": passed,
                "blocking": not passed,
                "text": text,
                "word_count": len(words),
                "keyword_hits": hits,
                "missing_keywords": [kw for kw in expected if kw not in hits],
                "issue": "" if passed else f"asr_words_{len(words)}_keywords_{len(hits)}",
            }
        )
    except Exception as exc:
        item.update({"status": "error", "issue": str(exc), "traceback": traceback.format_exc()})
    results.append(item)
print(json.dumps(results))
'''
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ffmpeg_dirs = _ffmpeg_subprocess_bin_dirs()
    if ffmpeg_dirs:
        env["PATH"] = os.pathsep.join([*ffmpeg_dirs, env.get("PATH", "")])
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=json.dumps(request),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=ACEJAM_VOCAL_ASR_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        issue = f"ASR subprocess timed out after {ACEJAM_VOCAL_ASR_TIMEOUT}s"
        return [
            {
                "path": str(path),
                "status": "error",
                "passed": False,
                "blocking": False,
                "issue": issue,
                "text": "",
                "word_count": 0,
                "keyword_hits": [],
                "missing_keywords": expected_keywords,
            }
            for path in paths
        ]
    if proc.returncode != 0:
        issue = (proc.stderr or proc.stdout or "ASR subprocess failed").strip()[-1200:]
        return [
            {
                "path": str(path),
                "status": "error",
                "passed": False,
                "blocking": True,
                "issue": issue,
                "text": "",
                "word_count": 0,
                "keyword_hits": [],
                "missing_keywords": expected_keywords,
            }
            for path in paths
        ]
    try:
        raw_items = json.loads(proc.stdout.strip().splitlines()[-1])
        normalized = []
        for raw_item in raw_items if isinstance(raw_items, list) else []:
            item = raw_item if isinstance(raw_item, dict) else {"path": "", "text": str(raw_item or "")}
            status = str(item.get("status") or "").lower()
            if status in {"error", "unavailable"}:
                item.update(
                    {
                        "status": "error",
                        "passed": False,
                        "blocking": True,
                        "issue": item.get("issue") or status or "asr_error",
                    }
                )
                normalized.append(item)
                continue
            item.update(_score_vocal_transcript(str(item.get("text") or ""), expected_keywords))
            normalized.append(item)
        return normalized
    except Exception as exc:
        return [
            {
                "path": str(path),
                "status": "error",
                "passed": False,
                "blocking": True,
                "issue": f"ASR JSON parse failed: {exc}",
                "text": "",
                "word_count": 0,
                "unique_word_count": 0,
                "keyword_hits": [],
                "missing_keywords": expected_keywords,
            }
            for path in paths
        ]


def _apply_vocal_intelligibility_gate_to_result(
    result: dict[str, Any],
    params: dict[str, Any],
    *,
    attempt: int,
    max_attempts: int,
) -> dict[str, Any]:
    if result.get("error_type") in {"memory_error", "timeout_error"}:
        reason = (
            "timeout_before_audio"
            if result.get("error_type") == "timeout_error"
            else "memory_error_before_audio"
        )
        gate = {
            "version": "acejam-vocal-intelligibility-gate-2026-05-01",
            "status": "error",
            "passed": False,
            "blocking": True,
            "needs_review": True,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "reason": reason,
            "transcript_preview": [],
            "transcripts": [],
        }
        result["success"] = False
        result["needs_review"] = True
        result["vocal_intelligibility_gate"] = gate
        result.pop("recommended_take", None)
        for audio in result.get("audios") or []:
            if isinstance(audio, dict):
                audio["is_recommended_take"] = False
        _persist_result_update(result)
        return gate
    if not _vocal_gate_required(params):
        gate = {"status": "skipped", "passed": True, "blocking": False, "reason": "not_vocal_text2music"}
        result["vocal_intelligibility_gate"] = gate
        return gate
    result_id = str(result.get("result_id") or "")
    audios = result.get("audios") or []
    paths = [RESULTS_DIR / safe_id(result_id) / str(audio.get("filename") or "") for audio in audios if audio.get("filename")]
    keywords = _vocal_gate_expected_keywords(params)
    transcripts = _transcribe_audio_paths(paths, language=str(params.get("vocal_language") or "en"), expected_keywords=keywords)
    manual_lora_review = _manual_lora_review_allowed(params)
    by_path = {str(item.get("path")): item for item in transcripts}
    passed_ids: list[str] = []
    for audio in audios:
        path = RESULTS_DIR / safe_id(result_id) / str(audio.get("filename") or "")
        audit = by_path.get(str(path)) or {
            "status": "error",
            "passed": False,
            "blocking": True,
            "issue": "missing_asr_result",
            "text": "",
            "word_count": 0,
            "keyword_hits": [],
            "missing_keywords": keywords,
        }
        audio["vocal_intelligibility_audit"] = audit
        if audit.get("passed") and audit.get("status") not in {"error", "unavailable"}:
            passed_ids.append(str(audio.get("id") or ""))
    verifier_error = bool(transcripts and any(item.get("status") in {"error", "unavailable"} for item in transcripts))
    blocking = (not passed_ids) and (
        not transcripts
        or any(item.get("blocking") for item in transcripts)
    )
    if verifier_error or (manual_lora_review and not passed_ids):
        blocking = False
    status = (
        "pass"
        if passed_ids
        else ("needs_review" if verifier_error or manual_lora_review else "fail")
    )
    transcript_preview = [
        {
            "audio_id": str(((audios[index] if index < len(audios) else {}) or {}).get("id") or ""),
            "status": item.get("status"),
            "text": str(item.get("text") or "")[:320],
            "issue": item.get("issue") or item.get("error") or "",
        }
        for index, item in enumerate(transcripts)
    ]
    gate = {
        "version": "acejam-vocal-intelligibility-gate-2026-05-01",
        "status": status,
        "passed": bool(passed_ids),
        "blocking": blocking,
        "needs_review": bool((verifier_error or manual_lora_review) and not passed_ids),
        "manual_review_allowed": bool(manual_lora_review),
        "attempt": attempt,
        "max_attempts": max_attempts,
        "expected_keywords": keywords,
        "passed_audio_ids": passed_ids,
        "transcripts": transcripts,
        "transcript_preview": transcript_preview,
    }
    result["vocal_intelligibility_gate"] = gate
    if passed_ids:
        for audio in audios:
            audio["is_recommended_take"] = str(audio.get("id") or "") == passed_ids[0]
        chosen = next((audio for audio in audios if str(audio.get("id") or "") == passed_ids[0]), None)
        if chosen:
            result["recommended_take"] = {
                "audio_id": chosen.get("id"),
                "filename": chosen.get("filename"),
                "score": chosen.get("pro_quality_score", 0),
                "status": "pass",
                "reasons": ["vocal_intelligibility_pass"],
            }
    else:
        for audio in audios:
            audio["is_recommended_take"] = False
        result.pop("recommended_take", None)
    if manual_lora_review and not passed_ids:
        result["needs_review"] = True
        suggestions = list(result.get("rerender_suggestions") or [])
        suggestion = (
            "LoRA benchmark kept this render for manual listening even though the vocal intelligibility gate did not pass."
        )
        if suggestion not in suggestions:
            suggestions.append(suggestion)
        result["rerender_suggestions"] = suggestions
        warnings = list(result.get("payload_warnings") or [])
        warning = (
            "vocal_intelligibility_verifier_error_manual_review"
            if verifier_error
            else "vocal_intelligibility_gate_manual_review"
        )
        if warning not in warnings:
            warnings.append(warning)
        result["payload_warnings"] = warnings
        result.pop("error", None)
    elif verifier_error and not passed_ids:
        result["success"] = False
        result["needs_review"] = True
        result["error"] = "Vocal intelligibility verifier could not complete."
        suggestions = list(result.get("rerender_suggestions") or [])
        suggestions.extend(
            [
                "Vocal intelligibility verifier could not complete; listen manually before publishing.",
                "Retry with no LoRA, lower LoRA scale, same seed, and ACE-Step docs-correct shift/steps.",
            ]
        )
        result["rerender_suggestions"] = suggestions
        warnings = list(result.get("payload_warnings") or [])
        if "vocal_intelligibility_verifier_error" not in warnings:
            warnings.append("vocal_intelligibility_verifier_error")
        result["payload_warnings"] = warnings
    elif not passed_ids:
        result["success"] = False
        result["error"] = "Vocal intelligibility gate rejected every take."
        suggestions = list(result.get("rerender_suggestions") or [])
        suggestions.extend(
            [
                "Vocal intelligibility gate rejected every take; regenerate with a clearer vocal route.",
                "Retry with no LoRA, lower LoRA scale, same seed, and ACE-Step docs-correct shift/steps.",
            ]
        )
        result["rerender_suggestions"] = suggestions
    result_path = RESULTS_DIR / safe_id(result_id) / "result.json"
    if result_path.is_file():
        try:
            meta = json.loads(result_path.read_text(encoding="utf-8"))
            meta["success"] = result.get("success", meta.get("success"))
            if "error" in result:
                meta["error"] = result.get("error")
            meta["audios"] = audios
            meta["payload_warnings"] = result.get("payload_warnings") or meta.get("payload_warnings") or []
            meta["vocal_intelligibility_gate"] = gate
            if result.get("recommended_take"):
                meta["recommended_take"] = result.get("recommended_take")
            else:
                meta.pop("recommended_take", None)
            if result.get("needs_review"):
                meta["needs_review"] = True
            meta["rerender_suggestions"] = result.get("rerender_suggestions") or meta.get("rerender_suggestions") or []
            result_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[vocal_gate] result.json update failed for {result_id}: {exc}", flush=True)
    _append_vocal_gate_debug(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event": "vocal_intelligibility_gate",
            "result_id": result_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "song_model": params.get("song_model"),
            "vocal_intelligibility_rescue_model": params.get("vocal_intelligibility_rescue_model"),
            "status": status,
            "blocking": gate["blocking"],
            "transcript_preview": [
                {
                    "path": Path(str(item.get("path") or "")).name,
                    "status": item.get("status"),
                    "word_count": item.get("word_count"),
                    "unique_word_count": item.get("unique_word_count"),
                    "keyword_hits": item.get("keyword_hits"),
                    "filler_ratio": item.get("filler_ratio"),
                    "repeat_ratio": item.get("repeat_ratio"),
                    "text": str(item.get("text") or "")[:240],
                    "issue": item.get("issue"),
                }
                for item in transcripts
            ],
        }
    )
    return gate


def _defer_library_save_until_vocal_pass(params: dict[str, Any]) -> dict[str, Any]:
    if not _vocal_gate_required(params) or not parse_bool(params.get("save_to_library"), False):
        return params
    updated = dict(params)
    updated["save_to_library"] = False
    updated["_deferred_save_to_library"] = True
    return updated


def _save_vocal_gate_passed_result_to_library(result: dict[str, Any], params: dict[str, Any]) -> None:
    if not parse_bool(params.get("_deferred_save_to_library"), False):
        return
    gate = result.get("vocal_intelligibility_gate") if isinstance(result.get("vocal_intelligibility_gate"), dict) else {}
    if not gate.get("passed"):
        return
    if result.get("_library_saved_after_vocal_gate"):
        return
    result_id = safe_id(str(result.get("result_id") or ""))
    audios = result.get("audios") if isinstance(result.get("audios"), list) else []
    recommended = result.get("recommended_take") if isinstance(result.get("recommended_take"), dict) else {}
    recommended_id = str(recommended.get("audio_id") or "")
    chosen = next(
        (
            audio
            for audio in audios
            if isinstance(audio, dict)
            and (str(audio.get("id") or "") == recommended_id or audio.get("is_recommended_take"))
        ),
        None,
    )
    if not isinstance(chosen, dict):
        return
    filename = str(chosen.get("filename") or "").strip()
    if not filename:
        return
    audio_path = RESULTS_DIR / result_id / filename
    if not audio_path.is_file():
        return
    entry = _save_song_entry(
        {
            "artist_name": params.get("artist_name"),
            "title": chosen.get("title") or params.get("title"),
            "description": params.get("description"),
            "tags": params.get("caption"),
            "tag_list": params.get("tag_list"),
            "lyrics": "[Instrumental]" if params.get("instrumental") else params.get("lyrics"),
            "caption_source": params.get("caption_source"),
            "lyrics_source": params.get("lyrics_source"),
            "payload_warnings": params.get("payload_warnings") or result.get("payload_warnings") or [],
            "generation_metadata_audit": chosen.get("generation_metadata_audit") or result.get("generation_metadata_audit"),
            "audio_quality_audit": chosen.get("audio_quality_audit"),
            "metadata_adherence": chosen.get("metadata_adherence"),
            "hit_readiness": chosen.get("hit_readiness") or result.get("hit_readiness"),
            "vocal_intelligibility_gate": gate,
            "lora_adapter": chosen.get("lora_adapter") or result.get("lora_adapter"),
            "runner_plan": params.get("runner_plan"),
            "ui_mode": params.get("ui_mode"),
            "bpm": params.get("bpm"),
            "key_scale": params.get("key_scale"),
            "time_signature": params.get("time_signature"),
            "language": params.get("vocal_language"),
            "duration": params.get("duration"),
            "task_type": params.get("task_type"),
            "song_model": params.get("song_model"),
            "ace_lm_model": params.get("ace_lm_model"),
            "seed": chosen.get("seed") or params.get("seed"),
            "parameters": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
            "album": _jsonable(params.get("album_metadata") or {}),
            "album_concept": (params.get("album_metadata") or {}).get("album_concept")
            if isinstance(params.get("album_metadata"), dict)
            else None,
            "album_id": (params.get("album_metadata") or {}).get("album_id")
            if isinstance(params.get("album_metadata"), dict)
            else None,
            "track_number": (params.get("album_metadata") or {}).get("track_number")
            if isinstance(params.get("album_metadata"), dict)
            else None,
            "result_id": result_id,
            "runner": result.get("runner") or params.get("runner_plan"),
            "preferred_audio_file": filename,
        },
        audio_path,
    )
    chosen["song_id"] = entry["id"]
    chosen["library_url"] = entry["audio_url"]
    result["_library_saved_after_vocal_gate"] = True
    result_path = RESULTS_DIR / result_id / "result.json"
    if result_path.is_file():
        try:
            meta = json.loads(result_path.read_text(encoding="utf-8"))
            meta["audios"] = audios
            meta["_library_saved_after_vocal_gate"] = True
            result_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[vocal_gate] deferred library save metadata update failed for {result_id}: {exc}", flush=True)


def _vocal_retry_params(
    params: dict[str, Any],
    *,
    attempt: int,
    last_gate: dict[str, Any] | None,
    allow_lora_changes: bool = True,
) -> dict[str, Any]:
    retry = dict(params)
    retry["seed"] = "-1"
    retry["use_random_seed"] = True
    retry["caption"] = _caption_with_vocal_clarity_traits(retry.get("caption") or "")
    retry["tag_list"] = split_terms(retry["caption"])
    lora_retry_note = ""
    if allow_lora_changes and parse_bool(params.get("use_lora"), False):
        if attempt == 2:
            retry["use_lora"] = False
            retry["lora_adapter_path"] = ""
            retry["lora_adapter_name"] = ""
            retry["lora_scale"] = 0.0
            retry["adapter_model_variant"] = ""
            lora_retry_note = "vocal_intelligibility_retry_no_lora"
        else:
            retry["lora_scale"] = min(float(params.get("lora_scale") or DEFAULT_LORA_GENERATION_SCALE), 0.25)
            lora_retry_note = "vocal_intelligibility_retry_lower_lora_scale"
    retry["repair_actions"] = list(retry.get("repair_actions") or []) + [
        {
            "type": "vocal_intelligibility_retry",
            "attempt": attempt,
            "lora_retry": lora_retry_note,
            "issues": [
                item.get("issue")
                for item in ((last_gate or {}).get("transcripts") or [])
                if item.get("issue")
            ],
        }
    ]
    payload_warnings = list(retry.get("payload_warnings") or [])
    if "vocal_intelligibility_retry" not in payload_warnings:
        payload_warnings.append("vocal_intelligibility_retry")
    if lora_retry_note and lora_retry_note not in payload_warnings:
        payload_warnings.append(lora_retry_note)
    retry["payload_warnings"] = payload_warnings
    return retry


def _vocal_gate_transcript_preview(gate: dict[str, Any] | None) -> str:
    if not isinstance(gate, dict):
        return ""
    preview = gate.get("transcript_preview")
    if isinstance(preview, list):
        parts = [
            str((item if isinstance(item, dict) else {}).get("text") or "").strip()
            for item in preview
        ]
        joined = " ".join(part for part in parts if part)
        if joined:
            return joined[:500]
    transcripts = gate.get("transcripts")
    if isinstance(transcripts, list):
        parts = [
            str((item if isinstance(item, dict) else {}).get("text") or "").strip()
            for item in transcripts
        ]
        return " ".join(part for part in parts if part)[:500]
    return ""


def _attempt_failure_reason(result: dict[str, Any] | None, gate: dict[str, Any] | None = None) -> str:
    if isinstance(result, dict) and result.get("error"):
        return str(result.get("error") or "")
    if isinstance(gate, dict):
        issues = [
            str((item if isinstance(item, dict) else {}).get("issue") or "").strip()
            for item in (gate.get("transcripts") if isinstance(gate.get("transcripts"), list) else [])
        ]
        issues = [issue for issue in issues if issue]
        if issues:
            return "; ".join(dict.fromkeys(issues))[:800]
        status = str(gate.get("status") or "").strip()
        if status:
            return f"Vocal gate status: {status}"
    return ""


def _attempt_lora_quality_status(params: dict[str, Any]) -> str:
    if not parse_bool(params.get("use_lora"), False):
        return ""
    try:
        quality = _lora_quality_for_params(params)
        return str(quality.get("quality_status") or "")
    except Exception:
        return ""


def _generation_attempt_summary(
    result: dict[str, Any],
    params: dict[str, Any],
    *,
    role: str,
    gate: dict[str, Any] | None = None,
    requested_params: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    gate = gate if isinstance(gate, dict) else (
        result.get("vocal_intelligibility_gate") if isinstance(result.get("vocal_intelligibility_gate"), dict) else {}
    )
    result_params = result.get("params") if isinstance(result.get("params"), dict) else {}
    requested_params = requested_params if isinstance(requested_params, dict) else params
    actual_model = (
        result.get("active_song_model")
        or result.get("song_model")
        or result_params.get("song_model")
        or params.get("song_model")
    )
    return _jsonable(
        {
            "attempt_role": role,
            "result_id": result.get("result_id"),
            "requested_song_model": requested_params.get("song_model"),
            "actual_song_model": actual_model,
            "song_model": params.get("song_model"),
            "inference_steps": params.get("inference_steps"),
            "shift": params.get("shift"),
            "with_lora": parse_bool(params.get("use_lora"), False),
            "lora_scale": params.get("lora_scale"),
            "lora_adapter_name": params.get("lora_adapter_name"),
            "lora_adapter_path": params.get("lora_adapter_path"),
            "lora_quality_status": _attempt_lora_quality_status(params),
            "vocal_gate_status": gate.get("status"),
            "passed": bool(gate.get("passed")),
            "blocking": bool(gate.get("blocking")),
            "transcript_preview": _vocal_gate_transcript_preview(gate),
            "failure_reason": reason or _attempt_failure_reason(result, gate),
        }
    )


def _annotate_generation_attempt_result(
    result: dict[str, Any],
    params: dict[str, Any],
    *,
    role: str,
    gate: dict[str, Any] | None = None,
    requested_params: dict[str, Any] | None = None,
    primary_attempt_id: str | None = None,
    diagnostic_attempts: list[dict[str, Any]] | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    summary = _generation_attempt_summary(
        result,
        params,
        role=role,
        gate=gate,
        requested_params=requested_params,
        reason=failure_reason,
    )
    result["attempt_role"] = role
    result["requested_song_model"] = summary.get("requested_song_model")
    result["actual_song_model"] = summary.get("actual_song_model")
    result["with_lora"] = summary.get("with_lora")
    result["lora_scale"] = summary.get("lora_scale")
    result["lora_quality_status"] = summary.get("lora_quality_status")
    result["vocal_gate_status"] = summary.get("vocal_gate_status")
    result["transcript_preview"] = summary.get("transcript_preview")
    if failure_reason or summary.get("failure_reason"):
        result["failure_reason"] = failure_reason or summary.get("failure_reason")
    result["primary_attempt_id"] = primary_attempt_id or result.get("primary_attempt_id") or result.get("result_id")
    if diagnostic_attempts is not None:
        result["diagnostic_attempts"] = diagnostic_attempts
    return result


def _short_vocal_attempt_params(
    params: dict[str, Any],
    *,
    label: str,
    use_lora: bool | None = None,
    lora_scale: float | None = None,
    song_model: str | None = None,
    warning_prefix: str = "vocal_preflight",
) -> dict[str, Any]:
    attempt = dict(params)
    attempt["duration"] = min(
        float(params.get("duration") or ACEJAM_LORA_PREFLIGHT_DURATION_SECONDS),
        float(ACEJAM_LORA_PREFLIGHT_DURATION_SECONDS),
    )
    attempt["batch_size"] = 1
    attempt["save_to_library"] = False
    attempt["_deferred_save_to_library"] = False
    attempt["auto_song_art"] = False
    attempt["auto_album_art"] = False
    attempt["auto_video_clip"] = False
    attempt["vocal_intelligibility_attempts"] = 1
    attempt["vocal_intelligibility_model_rescue"] = False
    attempt["use_random_seed"] = False
    attempt["seed"] = str(params.get("seed") or "42")
    if attempt["seed"].strip() in {"", "-1"}:
        attempt["seed"] = "42"
    if song_model:
        quality_profile = normalize_quality_profile(attempt.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
        model_defaults = quality_profile_model_settings(song_model, quality_profile)
        attempt["song_model"] = song_model
        attempt["inference_steps"] = int(model_defaults["inference_steps"])
        attempt["guidance_scale"] = float(model_defaults["guidance_scale"])
        attempt["shift"] = float(model_defaults["shift"])
        attempt["infer_method"] = str(model_defaults["infer_method"])
        attempt["sampler_mode"] = str(model_defaults["sampler_mode"])
        attempt["use_adg"] = bool(model_defaults.get("use_adg", False))
        attempt["audio_format"] = str(model_defaults.get("audio_format") or attempt.get("audio_format") or "wav32")
    attempt["caption"] = _caption_with_vocal_clarity_traits(str(attempt.get("caption") or ""))
    attempt["tag_list"] = split_terms(attempt["caption"])
    if use_lora is not None:
        attempt["use_lora"] = bool(use_lora)
    if parse_bool(attempt.get("use_lora"), False):
        if lora_scale is not None:
            attempt["lora_scale"] = round(float(lora_scale), 4)
    else:
        _strip_lora_trigger_conditioning(
            attempt,
            warning="lora_trigger_stripped_for_no_lora_preflight",
        )
        attempt["lora_adapter_path"] = ""
        attempt["lora_adapter_name"] = ""
        attempt["lora_scale"] = 0.0
        attempt["adapter_model_variant"] = ""
        attempt["adapter_song_model"] = ""
        attempt["adapter_metadata"] = {}
        attempt["use_lora_trigger"] = False
        attempt["lora_trigger_tag"] = ""
        attempt["lora_trigger_source"] = ""
        attempt["lora_trigger_aliases"] = []
        attempt["lora_trigger_candidates"] = []
    warnings = list(attempt.get("payload_warnings") or [])
    for warning in [warning_prefix, f"{warning_prefix}_{label}"]:
        if warning not in warnings:
            warnings.append(warning)
    attempt["payload_warnings"] = warnings
    attempt["repair_actions"] = list(attempt.get("repair_actions") or []) + [
        {
            "type": warning_prefix,
            "label": label,
            "song_model": attempt.get("song_model"),
            "use_lora": bool(attempt.get("use_lora")),
            "lora_scale": attempt.get("lora_scale"),
        }
    ]
    return _enforce_model_correct_render_settings(attempt, source=warning_prefix)


def _vocal_rescue_models(params: dict[str, Any]) -> list[str]:
    if not parse_bool(
        params.get("vocal_intelligibility_model_rescue"),
        ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE,
    ):
        return []
    raw_models = params.get("vocal_intelligibility_rescue_models")
    if isinstance(raw_models, str):
        candidates = [item.strip() for item in raw_models.split(",") if item.strip()]
    elif isinstance(raw_models, (list, tuple, set)):
        candidates = [str(item).strip() for item in raw_models if str(item).strip()]
    else:
        candidates = ACEJAM_VOCAL_INTELLIGIBILITY_RESCUE_MODELS
    try:
        installed = set(_installed_acestep_models())
    except Exception as exc:
        print(f"Vocal intelligibility model rescue unavailable: installed model scan failed: {exc}", flush=True)
        return []
    original_model = str(params.get("song_model") or "").strip()
    models: list[str] = []
    for model in candidates:
        if model == original_model or model in models:
            continue
        if model.startswith("acestep-v15-") and model in installed:
            models.append(model)
    return models


def _vocal_rescue_model_for_attempt(params: dict[str, Any], attempt: int, rescue_models: list[str]) -> str:
    original_model = str(params.get("song_model") or "").strip()
    if not rescue_models:
        return original_model
    rescue_after = clamp_int(
        params.get("vocal_intelligibility_model_rescue_after"),
        ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_AFTER,
        1,
        32,
    )
    if parse_bool(params.get("use_lora"), False):
        rescue_after += 2
    if attempt <= rescue_after:
        return original_model
    attempts_per_model = clamp_int(
        params.get("vocal_intelligibility_model_rescue_attempts"),
        ACEJAM_VOCAL_INTELLIGIBILITY_MODEL_RESCUE_ATTEMPTS,
        1,
        32,
    )
    index = max(0, (attempt - rescue_after - 1) // attempts_per_model)
    return rescue_models[min(index, len(rescue_models) - 1)]


def _lora_preflight_required(params: dict[str, Any]) -> bool:
    if not parse_bool(params.get("use_lora"), False):
        return False
    if not _vocal_gate_required(params):
        return False
    quality = _lora_quality_for_params(params)
    status = str(quality.get("quality_status") or "").lower()
    if status in ACEJAM_LORA_UNSAFE_QUALITY_STATUSES and not parse_bool(
        params.get("allow_unsafe_lora_for_benchmark"), False
    ):
        raise RuntimeError(
            "Selected LoRA is quarantined and cannot be used for generation"
            + (
                ": " + "; ".join(str(item) for item in quality.get("quality_reasons") or [] if str(item))
                if quality.get("quality_reasons")
                else "."
            )
        )
    if parse_bool(params.get("lora_preflight_required"), False):
        return True
    return False


def _lora_preflight_attempt_params(
    params: dict[str, Any],
    *,
    use_lora: bool,
    scale: float,
    label: str,
) -> dict[str, Any]:
    return _short_vocal_attempt_params(
        params,
        label=label,
        use_lora=use_lora,
        lora_scale=scale,
        warning_prefix="lora_preflight_audition",
    )


def _run_lora_preflight_attempt(params: dict[str, Any], *, attempt: int, max_attempts: int) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _run_advanced_generation_once(params)
    gate = _apply_vocal_intelligibility_gate_to_result(result, params, attempt=attempt, max_attempts=max_attempts)
    label = ""
    if params.get("repair_actions"):
        label = str((params.get("repair_actions") or [{}])[-1].get("label") or "")
    result["lora_preflight_attempt"] = {
        "label": label,
        "use_lora": bool(params.get("use_lora")),
        "lora_scale": params.get("lora_scale"),
        "result_id": result.get("result_id"),
        "gate_status": gate.get("status"),
        "passed": gate.get("passed"),
        "transcript_preview": gate.get("transcript_preview") or [],
    }
    return result, gate


def _persist_result_update(result: dict[str, Any]) -> None:
    result_id = str(result.get("result_id") or "").strip()
    if not result_id:
        return
    try:
        path = RESULTS_DIR / safe_id(result_id) / "result.json"
    except ValueError:
        return
    if not path.is_file():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(_jsonable(result))
        path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[generation] result update failed for {result_id}: {exc}", flush=True)


def _run_lora_preflight_verifier(params: dict[str, Any]) -> dict[str, Any] | None:
    if not _lora_preflight_required(params):
        return None
    adapter_path = str(params.get("lora_adapter_path") or "")
    selected_scale = clamp_float(params.get("lora_scale"), DEFAULT_LORA_GENERATION_SCALE, 0.0, 1.0)
    total_attempts = 2
    baseline_params = _lora_preflight_attempt_params(params, use_lora=False, scale=0.0, label="baseline")
    baseline_result, baseline_gate = _run_lora_preflight_attempt(baseline_params, attempt=1, max_attempts=total_attempts)
    preflight = {
        "status": "running",
        "baseline": baseline_result.get("lora_preflight_attempt"),
        "attempts": [baseline_result.get("lora_preflight_attempt")],
        "adapter_path": adapter_path,
        "requested_scale": selected_scale,
    }
    if not baseline_gate.get("passed"):
        baseline_gate_status = str(baseline_gate.get("status") or "").strip().lower()
        if baseline_gate_status == "needs_review":
            preflight["status"] = "needs_review"
            baseline_result["needs_review"] = True
            baseline_result["error"] = (
                "LoRA preflight could not verify the no-LoRA baseline; listen manually before approving this adapter."
            )
        else:
            preflight["status"] = "base_failed"
            baseline_result["error"] = "LoRA preflight baseline failed; base model/prompt/runtime must be fixed before testing LoRA."
        baseline_result["success"] = False
        baseline_result["lora_preflight"] = preflight
        _persist_result_update(baseline_result)
        return baseline_result
    lora_params = _lora_preflight_attempt_params(
        params,
        use_lora=True,
        scale=selected_scale,
        label=f"lora_selected_{selected_scale:g}",
    )
    lora_result, lora_gate = _run_lora_preflight_attempt(lora_params, attempt=2, max_attempts=total_attempts)
    preflight["attempts"].append(lora_result.get("lora_preflight_attempt"))
    if lora_gate.get("passed"):
        preflight["status"] = "passed"
        preflight["selected_scale"] = selected_scale
        _update_lora_adapter_quality_metadata(
            adapter_path,
            quality_status="verified",
            reason=f"LoRA preflight passed at the user-selected scale {selected_scale:g}.",
            recommended_lora_scale=selected_scale,
            audition={
                "status": "succeeded",
                "type": "lora_preflight",
                "result_id": lora_result.get("result_id"),
                "lora_scale": selected_scale,
                "vocal_intelligibility_gate": lora_gate,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        params["lora_scale"] = selected_scale
        params["adapter_metadata"] = infer_adapter_model_metadata(Path(adapter_path))
        warnings = list(params.get("payload_warnings") or [])
        if "lora_preflight_passed" not in warnings:
            warnings.append("lora_preflight_passed")
        params["payload_warnings"] = warnings
        params["lora_preflight"] = preflight
        return None
    preflight["status"] = "failed_audition"
    _update_lora_adapter_quality_metadata(
        adapter_path,
        quality_status="failed_audition",
        reason=f"LoRA preflight failed at the user-selected scale {selected_scale:g} while no-LoRA baseline passed.",
        audition={
            "status": "failed",
            "type": "lora_preflight",
            "requested_scale": selected_scale,
            "baseline_result_id": baseline_result.get("result_id"),
            "attempts": preflight["attempts"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    baseline_result["success"] = False
    baseline_result["error"] = (
        "LoRA preflight failed at the user-selected scale; adapter was marked failed_audition "
        "and the no-LoRA baseline was kept for review."
    )
    baseline_result["lora_preflight"] = preflight
    _persist_result_update(baseline_result)
    return baseline_result


def _vocal_preflight_required(params: dict[str, Any]) -> bool:
    if not _vocal_gate_required(params):
        return False
    if parse_bool(params.get("use_lora"), False):
        return False
    return parse_bool(params.get("vocal_preflight_required"), False)


def _run_vocal_preflight_verifier(params: dict[str, Any]) -> dict[str, Any] | None:
    if not _vocal_preflight_required(params):
        return None
    preflight_params = _short_vocal_attempt_params(
        params,
        label="selected_model",
        use_lora=False,
        warning_prefix="vocal_preflight",
    )
    result = _run_advanced_generation_once(preflight_params)
    gate = _apply_vocal_intelligibility_gate_to_result(result, preflight_params, attempt=1, max_attempts=1)
    attempt = _generation_attempt_summary(
        result,
        preflight_params,
        role="primary",
        gate=gate,
        requested_params=params,
        reason=_attempt_failure_reason(result, gate),
    )
    preflight = {
        "status": "passed" if gate.get("passed") else "failed",
        "attempt": attempt,
        "required_for_long_render": True,
        "duration": preflight_params.get("duration"),
    }
    params["vocal_preflight"] = preflight
    result["vocal_preflight"] = preflight
    if gate.get("passed"):
        _persist_result_update(result)
        return None
    result["success"] = False
    result["needs_review"] = True
    result["error"] = (
        "Selected ACE-Step model failed the short vocal preflight; long render was not started."
    )
    _annotate_generation_attempt_result(
        result,
        preflight_params,
        role="primary",
        gate=gate,
        requested_params=params,
        failure_reason=result["error"],
    )
    _persist_result_update(result)
    return result


def _run_vocal_diagnostic_attempts(
    params: dict[str, Any],
    *,
    primary_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    labels: set[tuple[str, bool]] = set()
    requested_model = str(params.get("song_model") or "").strip()
    if not requested_model:
        return diagnostics

    def add_diagnostic(label: str, *, song_model: str | None = None, use_lora: bool = False) -> None:
        model_name = song_model or str(params.get("song_model") or "")
        key = (model_name, bool(use_lora))
        if key in labels:
            return
        labels.add(key)
        diag_params = _short_vocal_attempt_params(
            params,
            label=label,
            use_lora=use_lora,
            song_model=model_name,
            warning_prefix="vocal_diagnostic",
        )
        try:
            diag_result = _run_advanced_generation_once(diag_params)
            diag_gate = _apply_vocal_intelligibility_gate_to_result(
                diag_result,
                diag_params,
                attempt=1,
                max_attempts=1,
            )
            diag_summary = _generation_attempt_summary(
                diag_result,
                diag_params,
                role="diagnostic",
                gate=diag_gate,
                requested_params=params,
                reason=_attempt_failure_reason(diag_result, diag_gate),
            )
            diag_summary["label"] = label
            _annotate_generation_attempt_result(
                diag_result,
                diag_params,
                role="diagnostic",
                gate=diag_gate,
                requested_params=params,
                primary_attempt_id=str((primary_result or {}).get("result_id") or ""),
                failure_reason=diag_summary.get("failure_reason") or "",
            )
            _persist_result_update(diag_result)
            diagnostics.append(diag_summary)
        except Exception as exc:
            diagnostics.append(
                _jsonable(
                    {
                        "attempt_role": "diagnostic",
                        "label": label,
                        "requested_song_model": params.get("song_model"),
                        "actual_song_model": model_name,
                        "with_lora": bool(use_lora),
                        "passed": False,
                        "vocal_gate_status": "error",
                        "failure_reason": str(exc),
                    }
                )
            )

    if parse_bool(params.get("use_lora"), False):
        add_diagnostic("same_model_no_lora", song_model=requested_model, use_lora=False)
    for model in _vocal_rescue_models(params):
        add_diagnostic(f"{safe_id(model)}_no_lora", song_model=model, use_lora=False)
    return diagnostics


def _finalize_failed_primary_with_diagnostics(
    result: dict[str, Any],
    params: dict[str, Any],
    *,
    gate: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    failure_reason: str = "",
    run_diagnostics: bool = True,
) -> dict[str, Any]:
    diagnostics = _run_vocal_diagnostic_attempts(params, primary_result=result) if run_diagnostics else []
    diagnostic_passed = any(item.get("passed") for item in diagnostics)
    result["success"] = False
    result["needs_review"] = True
    result["error"] = failure_reason or _attempt_failure_reason(result, gate) or "Primary vocal render failed."
    if diagnostic_passed:
        result["error"] = (
            f"{result['error']} Diagnostic fallback passed, but the requested model/LoRA primary remains failed."
        )
    if history is not None:
        result["vocal_intelligibility_history"] = history
    if params.get("lora_preflight"):
        result["lora_preflight"] = params.get("lora_preflight")
    if params.get("vocal_preflight"):
        result["vocal_preflight"] = params.get("vocal_preflight")
    _annotate_generation_attempt_result(
        result,
        params,
        role="primary",
        gate=gate,
        requested_params=params,
        diagnostic_attempts=diagnostics,
        failure_reason=result["error"],
    )
    warnings = list(result.get("payload_warnings") or params.get("payload_warnings") or [])
    if diagnostic_passed and "vocal_diagnostic_passed_primary_failed" not in warnings:
        warnings.append("vocal_diagnostic_passed_primary_failed")
    if "primary_vocal_gate_failed" not in warnings:
        warnings.append("primary_vocal_gate_failed")
    result["payload_warnings"] = warnings
    _persist_result_update(result)
    return result


def _with_vocal_rescue_model(params: dict[str, Any], rescue_model: str, *, attempt: int) -> dict[str, Any]:
    original_model = str(params.get("song_model") or "").strip()
    if not rescue_model or rescue_model == original_model:
        return params
    rescue = dict(params)
    quality_profile = normalize_quality_profile(rescue.get("quality_profile") or DEFAULT_QUALITY_PROFILE)
    model_defaults = quality_profile_model_settings(rescue_model, quality_profile)
    rescue["song_model"] = rescue_model
    rescue["inference_steps"] = int(model_defaults["inference_steps"])
    rescue["guidance_scale"] = float(model_defaults["guidance_scale"])
    rescue["shift"] = float(model_defaults["shift"])
    rescue["infer_method"] = str(model_defaults["infer_method"])
    rescue["sampler_mode"] = str(model_defaults["sampler_mode"])
    rescue["use_adg"] = bool(model_defaults.get("use_adg", False))
    rescue["audio_format"] = str(model_defaults.get("audio_format") or rescue.get("audio_format") or "wav32")
    rescue["vocal_intelligibility_original_model"] = original_model
    rescue["vocal_intelligibility_rescue_model"] = rescue_model
    rescue["vocal_intelligibility_rescue_attempt"] = attempt
    disabled_rescue_lora = parse_bool(rescue.get("use_lora"), False)
    if disabled_rescue_lora:
        rescue["use_lora"] = False
        rescue["lora_adapter_path"] = ""
        rescue["lora_adapter_name"] = ""
        rescue["lora_scale"] = 0.0
        rescue["adapter_model_variant"] = ""
    rescue["final_model_policy"] = {
        **(rescue.get("final_model_policy") if isinstance(rescue.get("final_model_policy"), dict) else {}),
        "vocal_intelligibility_rescue": True,
        "original_model": original_model,
        "rescue_model": rescue_model,
        "reason": "Original model failed the vocal intelligibility gate; rendering with the clearest installed fallback.",
    }
    album_metadata = dict(rescue.get("album_metadata") or {})
    album_metadata["vocal_intelligibility_original_model"] = original_model
    album_metadata["vocal_intelligibility_rescue_model"] = rescue_model
    rescue["album_metadata"] = album_metadata
    payload_warnings = list(rescue.get("payload_warnings") or [])
    warning = f"vocal_intelligibility_model_rescue:{original_model}->{rescue_model}"
    if warning not in payload_warnings:
        payload_warnings.append(warning)
    if disabled_rescue_lora and "vocal_intelligibility_rescue_lora_disabled" not in payload_warnings:
        payload_warnings.append("vocal_intelligibility_rescue_lora_disabled")
    rescue["payload_warnings"] = payload_warnings
    return rescue


def _redact_official_runner_log_line(line: str) -> str:
    """Keep official ACE-Step subprocess logs compact in Pinokio terminals."""
    if "formatted_prompt_with_cot=" in line:
        return re.sub(
            r"formatted_prompt_with_cot=.*",
            "formatted_prompt_with_cot=[redacted by MLX Media: prompt/audio-code payload]",
            line,
        )
    if "Debug output text:" in line and "<|audio_code_" in line:
        return re.sub(
            r"Debug output text:.*",
            "Debug output text: [redacted by MLX Media: audio-code payload]",
            line,
        )
    if "<|audio_code_" in line:
        return re.sub(r"(?:<\|audio_code_\d+\|>){3,}", "<|audio_code_REDACTED|>", line)
    if len(line) > 1600:
        ending = "\n" if line.endswith("\n") else ""
        return f"{line[:1600].rstrip()} ... [truncated by MLX Media]{ending}"
    return line


def _redact_official_runner_stream_line(line: str, state: dict[str, Any]) -> str:
    if "conditioning_text:_prepare_text_conditioning_inputs" in line:
        if not ACEJAM_REDACT_OFFICIAL_LOG_TEXT:
            state["conditioning_block"] = False
            return _redact_official_runner_log_line(line)
        state["conditioning_block"] = True
        if "text_prompt:" in line:
            return re.sub(r"text_prompt:.*", "text_prompt: [redacted by MLX Media: conditioning prompt]", line)
        if "lyrics_text:" in line:
            return re.sub(r"lyrics_text:.*", "lyrics_text: [redacted by MLX Media: conditioning lyrics]", line)
        return ""
    if state.get("conditioning_block"):
        if re.match(r"^\d{4}-\d{2}-\d{2}\s", line):
            state["conditioning_block"] = False
            return _redact_official_runner_log_line(line)
        return ""
    return _redact_official_runner_log_line(line)


def _official_runner_timeout_seconds(request_payload: dict[str, Any], requested_timeout: int | None = None) -> int:
    floor = max(60, int(requested_timeout or ACEJAM_OFFICIAL_RUNNER_TIMEOUT_SECONDS))
    params = request_payload.get("params") if isinstance(request_payload.get("params"), dict) else {}
    config = request_payload.get("config") if isinstance(request_payload.get("config"), dict) else {}
    steps = clamp_int(params.get("inference_steps") or params.get("infer_steps"), 32, 1, 200)
    duration = clamp_float(params.get("duration") or params.get("audio_duration"), 180.0, DURATION_MIN, DURATION_MAX)
    batch_size = clamp_int(config.get("batch_size"), 1, 1, 16)
    backend = _normalize_audio_backend(request_payload.get("audio_backend"), request_payload.get("use_mlx_dit"))
    is_mlx = backend == "mlx"

    seconds_per_step_per_minute = 18.0 if is_mlx else 8.0
    batch_factor = 1.0 + max(0, batch_size - 1) * (0.35 if is_mlx else 0.2)
    estimated = int(900 + steps * max(duration, 30.0) / 60.0 * seconds_per_step_per_minute * batch_factor)
    return min(max(floor, estimated), ACEJAM_OFFICIAL_RUNNER_MAX_TIMEOUT_SECONDS)


def _official_service_generation_timeout_seconds(request_payload: dict[str, Any], runner_timeout: int) -> int:
    """Timeout passed into ACE-Step's internal service_generate watchdog.

    ACE-Step defaults this to 600s via ACESTEP_GENERATION_TIMEOUT. That is too
    short for docs-correct XL-SFT/Base full-song renders on PyTorch/MPS, where a
    single 180s, 50-step take can legitimately take more than ten minutes.
    Keep the service timeout slightly below the outer subprocess timeout so a
    true stall still returns a clean error payload instead of being OS-killed.
    """
    margin = 60 if runner_timeout > 900 else 10
    maximum = max(1, runner_timeout - margin)
    override = str(os.environ.get("ACEJAM_OFFICIAL_SERVICE_TIMEOUT_SECONDS") or "").strip()
    if override:
        try:
            return max(1, min(int(float(override)), maximum))
        except (TypeError, ValueError):
            pass
    return max(1, min(maximum, max(600, maximum)))


def _run_official_runner_request(request_payload: dict[str, Any], work_dir: Path, timeout: int | None = None) -> dict[str, Any]:
    if not OFFICIAL_ACE_STEP_DIR.exists():
        raise RuntimeError("Official ACE-Step runner requires app/vendor/ACE-Step-1.5. Run Install/Update first.")
    if not OFFICIAL_RUNNER_SCRIPT.exists():
        raise RuntimeError("Official ACE-Step runner script is missing.")

    if "audio_backend" in request_payload or "use_mlx_dit" in request_payload:
        backend = _normalize_audio_backend(request_payload.get("audio_backend"), request_payload.get("use_mlx_dit"))
        request_payload = dict(request_payload)
        request_payload["audio_backend"] = backend
        request_payload["use_mlx_dit"] = backend == "mlx"
        request_payload["requested_audio_backend"] = backend
        request_payload["requested_use_mlx_dit"] = backend == "mlx"
        request_payload.setdefault(
            "audio_backend_contract",
            {
                "requested_audio_backend": backend,
                "requested_use_mlx_dit": backend == "mlx",
                "enforced_at": "runner_launch",
            },
        )

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
    runner_timeout = _official_runner_timeout_seconds(request_payload, timeout)
    service_timeout = _official_service_generation_timeout_seconds(request_payload, runner_timeout)
    env["ACESTEP_GENERATION_TIMEOUT"] = str(service_timeout)
    print(
        f"[official_runner] starting action={request_payload.get('action') or 'generate'} "
        f"timeout={runner_timeout}s service_timeout={service_timeout}s work_dir={work_dir}",
        flush=True,
    )
    memory_before = _mps_memory_snapshot("official_runner:start")

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
        returncode = process.wait(timeout=runner_timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        returncode = process.wait(timeout=30)
        raise RuntimeError(f"Official ACE-Step runner timed out after {runner_timeout}s") from exc
    finally:
        for pipe in [process.stdout, process.stderr]:
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass
        for reader in readers:
            reader.join(timeout=2)
    memory_after = _mps_memory_snapshot("official_runner:end")
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
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if isinstance(response, dict):
        response["mps_memory"] = {"before": memory_before, "after": memory_after}
        backend = _normalize_audio_backend(request_payload.get("audio_backend"), request_payload.get("use_mlx_dit"))
        if backend == "mlx":
            status = response.get("audio_backend_status") if isinstance(response.get("audio_backend_status"), dict) else {}
            if status.get("effective_mlx_dit_active") is not True:
                fallback = status.get("fallback_reason") or "official runner did not confirm active native MLX DiT"
                raise RuntimeError(f"MLX audio backend was requested, but the runner did not activate MLX DiT: {fallback}")
    return response


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


def _should_retry_mps_oom_without_dcw(params: dict[str, Any], error: Any) -> bool:
    if not _is_mps_oom_error(error):
        return False
    if not _IS_APPLE_SILICON:
        return False
    if _audio_backend_uses_mlx(params):
        return False
    if str(params.get("device") or "auto").strip().lower() not in {"auto", "mps", "metal"}:
        return False
    if not parse_bool(params.get("dcw_enabled"), False):
        return False
    if "xl" not in str(params.get("song_model") or "").strip().lower():
        return False
    return normalize_task_type(params.get("task_type")) == "text2music"


def _disable_dcw_for_mps_oom_retry(params: dict[str, Any], *, source: str) -> dict[str, Any]:
    retry = dict(params)
    retry["dcw_enabled"] = False
    warnings = list(retry.get("payload_warnings") or [])
    warning = f"mps_oom_retry_without_dcw:{source}:dcw_enabled=True->False"
    if warning not in warnings:
        warnings.append(warning)
    retry["payload_warnings"] = warnings
    repairs = list(retry.get("repair_actions") or [])
    repairs.append(
        {
            "type": "mps_oom_retry_without_dcw",
            "source": source,
            "song_model": retry.get("song_model"),
            "duration": retry.get("duration"),
            "lora_scale": retry.get("lora_scale"),
        }
    )
    retry["repair_actions"] = repairs
    return retry


def _run_official_generation(params: dict[str, Any]) -> dict[str, Any]:
    params = _enforce_model_correct_render_settings(_album_ace_lm_disabled_payload(dict(params)), source="official_generation")
    _apply_audio_backend_defaults(params, source="official_generation")
    _apply_mps_long_lora_memory_guard(params, source="official_generation")
    result_id = uuid.uuid4().hex[:12]
    result_dir = RESULTS_DIR / result_id
    official_dir = result_dir / "official"
    result_dir.mkdir(parents=True, exist_ok=True)
    official_dir.mkdir(parents=True, exist_ok=True)

    memory_plan = _official_generation_memory_plan(params)
    params["requested_take_count"] = memory_plan["requested_take_count"]
    params["actual_runner_batch_size"] = memory_plan["actual_runner_batch_size"]
    params["memory_policy"] = memory_plan
    if memory_plan.get("sequential"):
        warning = (
            f"mps_sequential_takes:{memory_plan['requested_take_count']}"
            f"x1_for_{params.get('song_model')}"
        )
        if warning not in params["payload_warnings"]:
            params["payload_warnings"].append(warning)

    metadata_audit: dict[str, Any] = {}
    hit_readiness = params.get("hit_readiness") or hit_readiness_report(
        params,
        task_type=params.get("task_type"),
        song_model=params.get("song_model"),
        runner_plan=params.get("runner_plan"),
    )
    official_lora_status: dict[str, Any] = {
        "active": bool(params.get("use_lora")),
        "path": params.get("lora_adapter_path", ""),
        "scale": params.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE),
    }
    audios: list[dict[str, Any]] = []
    official_runs: list[dict[str, Any]] = []
    memory_events: list[dict[str, Any]] = []
    time_costs_by_take: list[dict[str, Any]] = []
    lm_metadata: Any = None
    generation_task_id = str(params.get("generation_task_id") or "").strip()

    def publish_completed_takes() -> None:
        if not generation_task_id or not audios:
            return
        requested = max(1, int(memory_plan.get("requested_take_count") or len(audios) or 1))
        completed = min(len(audios), requested)
        progress = max(25, min(95, 25 + int(65 * completed / requested)))
        partial_result = {
            "success": True,
            "partial": completed < requested,
            "in_progress": completed < requested,
            "full_takes_ready": completed,
            "total_take_count": requested,
            "completed_full_takes": _jsonable(audios),
            "result_id": result_id,
            "active_song_model": params["song_model"],
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
            "requested_take_count": requested,
            "completed_take_count": completed,
            "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
            "memory_policy": _jsonable(memory_plan),
            "official_runs": _jsonable(official_runs),
            "inference_steps": params["inference_steps"],
            "guidance_scale": params["guidance_scale"],
            "shift": params["shift"],
            "audio_format": params["audio_format"],
            "lm_backend": params["lm_backend"],
            "audio_backend": params["audio_backend"],
            "use_mlx_dit": params.get("use_mlx_dit"),
            "audio_backend_status": _jsonable(
                (official_runs[-1].get("audio_backend_status") if official_runs else {}) or {}
            ),
            "use_lora": params["use_lora"],
            "lora_adapter_path": params["lora_adapter_path"],
            "lora_adapter_name": params["lora_adapter_name"],
            "use_lora_trigger": params.get("use_lora_trigger", False),
            "lora_trigger_tag": params.get("lora_trigger_tag", ""),
            "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
            "lora_scale": params["lora_scale"],
            "adapter_model_variant": params["adapter_model_variant"],
            "lora_adapter": _jsonable(official_lora_status),
            "tags": params["caption"],
            "tag_list": params["tag_list"],
            "lyrics": "[Instrumental]" if params["instrumental"] else params["lyrics"],
            "payload_warnings": params["payload_warnings"],
            "style_profile": params.get("style_profile", ""),
            "style_caption_tags": params.get("style_caption_tags", ""),
            "style_lyric_tags_applied": _jsonable(params.get("style_lyric_tags_applied") or []),
            "style_conditioning_audit": _jsonable(params.get("style_conditioning_audit") or {}),
            "generation_metadata_audit": _jsonable(metadata_audit),
            "hit_readiness": _jsonable(hit_readiness),
            "payload_quality_gate": _jsonable(params.get("payload_quality_gate") or {}),
            "payload_gate_status": params.get("payload_gate_status") or "",
            "payload_gate_passed": bool(params.get("payload_gate_passed")),
            "payload_gate_blocking_issues": _jsonable(params.get("payload_gate_blocking_issues") or []),
            "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
            "audios": _jsonable(audios),
        }
        stage = (
            f"Volledige take {completed}/{requested} klaar; volgende take rendert"
            if completed < requested
            else f"Volledige take {completed}/{requested} klaar"
        )
        _set_api_generation_task(
            generation_task_id,
            state="running",
            status=0,
            stage=stage,
            progress=progress,
            result=partial_result,
            logs=[f"Volledige take {completed}/{requested} klaar: {audios[-1].get('filename') or audios[-1].get('audio_url')}"],
        )

    def memory_failure(error: Any) -> dict[str, Any]:
        error_text = str(error or "MPS backend out of memory")
        failure = {
            "id": result_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active_song_model": params["song_model"],
            "success": False,
            "needs_review": True,
            "error_type": "memory_error",
            "error": (
                "ACE-Step ran out of Apple unified memory for this exact high-quality render. "
                "The model/steps/shift were preserved; retry after freeing memory or reduce duration/takes."
            ),
            "failure_reason": error_text,
            "runner": "official",
            "runner_plan": params["runner_plan"],
            "ui_mode": params["ui_mode"],
            "title": params["title"],
            "artist_name": params["artist_name"],
            "quality_profile": params["quality_profile"],
            "task_type": params["task_type"],
            "song_model": params["song_model"],
            "duration": params["duration"],
            "batch_size": params["batch_size"],
            "requested_take_count": memory_plan["requested_take_count"],
            "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
            "memory_policy": _jsonable(memory_plan),
            "mps_memory": {
                "events": _jsonable(memory_events),
                "failure": _mps_memory_snapshot("official_generation:memory_error"),
            },
            "inference_steps": params["inference_steps"],
            "shift": params["shift"],
            "audio_format": params["audio_format"],
            "lm_backend": params["lm_backend"],
            "audio_backend": params["audio_backend"],
            "use_mlx_dit": params.get("use_mlx_dit"),
            "use_lora": params["use_lora"],
            "lora_adapter_path": params["lora_adapter_path"],
            "lora_adapter_name": params["lora_adapter_name"],
            "use_lora_trigger": params.get("use_lora_trigger", False),
            "lora_trigger_tag": params.get("lora_trigger_tag", ""),
            "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
            "lora_scale": params["lora_scale"],
            "adapter_model_variant": params["adapter_model_variant"],
            "lora_adapter": _jsonable(official_lora_status),
            "payload_warnings": [*params["payload_warnings"], "official_runner_mps_memory_error"],
            "tags": params["caption"],
            "tag_list": params["tag_list"],
            "lyrics": "[Instrumental]" if params["instrumental"] else params["lyrics"],
            "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
            "generation_metadata_audit": _jsonable(metadata_audit or _generation_metadata_audit(params)),
            "hit_readiness": _jsonable(hit_readiness),
            "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
            "audios": audios,
            "recommended_take": None,
            "rerender_suggestions": [
                "Free memory by closing local LLM/image/video apps, then retry.",
                "Keep XL-SFT 50 steps and shift 1.0, but render one take at a time.",
                "If memory still fails at one take, shorten duration for the LoRA test.",
            ],
        }
        (result_dir / "result.json").write_text(json.dumps(_jsonable(failure), indent=2), encoding="utf-8")
        return {
            "success": False,
            "needs_review": True,
            "error_type": "memory_error",
            "error": failure["error"],
            "failure_reason": error_text,
            "result_id": result_id,
            "active_song_model": params["song_model"],
            "runner": "official",
            "audios": audios,
            "params": failure["params"],
            "duration": params["duration"],
            "song_model": params["song_model"],
            "quality_profile": params["quality_profile"],
            "batch_size": params["batch_size"],
            "requested_take_count": memory_plan["requested_take_count"],
            "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
            "memory_policy": _jsonable(memory_plan),
            "mps_memory": failure["mps_memory"],
            "inference_steps": params["inference_steps"],
            "guidance_scale": params["guidance_scale"],
            "shift": params["shift"],
            "audio_format": params["audio_format"],
            "payload_warnings": failure["payload_warnings"],
            "audio_backend": params["audio_backend"],
            "use_mlx_dit": params.get("use_mlx_dit"),
            "use_lora": params["use_lora"],
            "lora_adapter_path": params["lora_adapter_path"],
            "lora_adapter_name": params["lora_adapter_name"],
            "use_lora_trigger": params.get("use_lora_trigger", False),
            "lora_trigger_tag": params.get("lora_trigger_tag", ""),
            "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
            "lora_scale": params["lora_scale"],
            "adapter_model_variant": params["adapter_model_variant"],
            "lora_adapter": _jsonable(official_lora_status),
        }

    def timeout_failure(error: Any) -> dict[str, Any]:
        error_text = str(error or "ACE-Step generation timed out")
        failure = {
            "id": result_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active_song_model": params["song_model"],
            "success": False,
            "needs_review": True,
            "error_type": "timeout_error",
            "error": (
                "ACE-Step took longer than the allowed high-quality render timeout. "
                "The model/steps/shift were preserved; this is a runtime timeout, not a quality downgrade."
            ),
            "failure_reason": error_text,
            "runner": "official",
            "runner_plan": params["runner_plan"],
            "ui_mode": params["ui_mode"],
            "title": params["title"],
            "artist_name": params["artist_name"],
            "quality_profile": params["quality_profile"],
            "task_type": params["task_type"],
            "song_model": params["song_model"],
            "duration": params["duration"],
            "batch_size": params["batch_size"],
            "requested_take_count": memory_plan["requested_take_count"],
            "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
            "memory_policy": _jsonable(memory_plan),
            "mps_memory": {
                "events": _jsonable(memory_events),
                "failure": _mps_memory_snapshot("official_generation:timeout_error"),
            },
            "inference_steps": params["inference_steps"],
            "guidance_scale": params["guidance_scale"],
            "shift": params["shift"],
            "audio_format": params["audio_format"],
            "lm_backend": params["lm_backend"],
            "audio_backend": params["audio_backend"],
            "use_mlx_dit": params.get("use_mlx_dit"),
            "use_lora": params["use_lora"],
            "lora_adapter_path": params["lora_adapter_path"],
            "lora_adapter_name": params["lora_adapter_name"],
            "use_lora_trigger": params.get("use_lora_trigger", False),
            "lora_trigger_tag": params.get("lora_trigger_tag", ""),
            "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
            "lora_scale": params["lora_scale"],
            "adapter_model_variant": params["adapter_model_variant"],
            "lora_adapter": _jsonable(official_lora_status),
            "payload_warnings": [*params["payload_warnings"], "official_runner_generation_timeout"],
            "tags": params["caption"],
            "tag_list": params["tag_list"],
            "lyrics": "[Instrumental]" if params["instrumental"] else params["lyrics"],
            "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
            "generation_metadata_audit": _jsonable(metadata_audit or _generation_metadata_audit(params)),
            "hit_readiness": _jsonable(hit_readiness),
            "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
            "audios": audios,
            "recommended_take": None,
            "rerender_suggestions": [
                "Retry after freeing memory; high-quality XL-SFT full-song renders can take a long time on MPS.",
                "Keep XL-SFT 50 steps and shift 1.0; do not use Turbo as a silent fallback for this primary render.",
                "If it still exceeds the timeout, shorten duration for the LoRA test before a full-song render.",
            ],
        }
        (result_dir / "result.json").write_text(json.dumps(_jsonable(failure), indent=2), encoding="utf-8")
        return {
            "success": False,
            "needs_review": True,
            "error_type": "timeout_error",
            "error": failure["error"],
            "failure_reason": error_text,
            "result_id": result_id,
            "active_song_model": params["song_model"],
            "runner": "official",
            "audios": audios,
            "params": failure["params"],
            "duration": params["duration"],
            "song_model": params["song_model"],
            "quality_profile": params["quality_profile"],
            "batch_size": params["batch_size"],
            "requested_take_count": memory_plan["requested_take_count"],
            "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
            "memory_policy": _jsonable(memory_plan),
            "mps_memory": failure["mps_memory"],
            "inference_steps": params["inference_steps"],
            "guidance_scale": params["guidance_scale"],
            "shift": params["shift"],
            "audio_format": params["audio_format"],
            "payload_warnings": failure["payload_warnings"],
            "audio_backend": params["audio_backend"],
            "use_mlx_dit": params.get("use_mlx_dit"),
            "use_lora": params["use_lora"],
            "lora_adapter_path": params["lora_adapter_path"],
            "lora_adapter_name": params["lora_adapter_name"],
            "use_lora_trigger": params.get("use_lora_trigger", False),
            "lora_trigger_tag": params.get("lora_trigger_tag", ""),
            "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
            "lora_scale": params["lora_scale"],
            "adapter_model_variant": params["adapter_model_variant"],
            "lora_adapter": _jsonable(official_lora_status),
        }

    total_passes = int(memory_plan.get("render_pass_count") or 1)
    for pass_index in range(total_passes):
        take_params = _official_take_params(params, memory_plan, pass_index)
        take_save_dir = official_dir if pass_index == 0 else result_dir / f"official_take_{pass_index + 1}"
        take_work_dir = result_dir if pass_index == 0 else result_dir / f"official_runner_take_{pass_index + 1}"
        take_save_dir.mkdir(parents=True, exist_ok=True)
        take_work_dir.mkdir(parents=True, exist_ok=True)
        official_request = _official_request_payload(take_params, take_save_dir)
        official_request["memory_policy"] = _jsonable(memory_plan)
        official_request["requested_take_count"] = memory_plan["requested_take_count"]
        official_request["actual_runner_batch_size"] = memory_plan["actual_runner_batch_size"]
        if not metadata_audit:
            metadata_audit = _generation_metadata_audit(take_params, official_request)
        retry_reason = ""
        try:
            with _official_generation_runner_lock:
                memory_events.append(
                    _prepare_audio_generation_memory(
                        f"official_generation_take_{pass_index + 1}",
                        release_handler=True,
                    )
                )
                official = _run_official_runner_request(official_request, take_work_dir)
        except Exception as exc:
            if _should_retry_mps_oom_without_dcw(take_params, exc):
                retry_reason = "mps_oom_retry_without_dcw"
                take_params = _disable_dcw_for_mps_oom_retry(
                    take_params,
                    source=f"take_{pass_index + 1}",
                )
                params["dcw_enabled"] = False
                params["payload_warnings"] = list(
                    dict.fromkeys([*params.get("payload_warnings", []), *take_params.get("payload_warnings", [])])
                )
                params["repair_actions"] = list(params.get("repair_actions") or []) + [
                    item
                    for item in list(take_params.get("repair_actions") or [])
                    if isinstance(item, dict) and item.get("type") == "mps_oom_retry_without_dcw"
                ]
                official_request = _official_request_payload(take_params, take_save_dir)
                official_request["memory_policy"] = _jsonable(memory_plan)
                official_request["requested_take_count"] = memory_plan["requested_take_count"]
                official_request["actual_runner_batch_size"] = memory_plan["actual_runner_batch_size"]
                try:
                    with _official_generation_runner_lock:
                        memory_events.append(
                            _prepare_audio_generation_memory(
                                f"official_generation_take_{pass_index + 1}:retry_without_dcw",
                                release_handler=True,
                            )
                        )
                        official = _run_official_runner_request(official_request, take_work_dir)
                except Exception as retry_exc:
                    if _is_mps_oom_error(retry_exc):
                        return memory_failure(retry_exc)
                    if _is_acestep_generation_timeout_error(retry_exc):
                        return timeout_failure(retry_exc)
                    raise
            elif _is_mps_oom_error(exc):
                return memory_failure(exc)
            elif _is_acestep_generation_timeout_error(exc):
                return timeout_failure(exc)
            else:
                raise
        if not official.get("success"):
            error = official.get("error") or official.get("status_message") or "Official ACE-Step generation failed"
            if _should_retry_mps_oom_without_dcw(take_params, error):
                retry_reason = "mps_oom_retry_without_dcw"
                take_params = _disable_dcw_for_mps_oom_retry(
                    take_params,
                    source=f"take_{pass_index + 1}",
                )
                params["dcw_enabled"] = False
                params["payload_warnings"] = list(
                    dict.fromkeys([*params.get("payload_warnings", []), *take_params.get("payload_warnings", [])])
                )
                params["repair_actions"] = list(params.get("repair_actions") or []) + [
                    item
                    for item in list(take_params.get("repair_actions") or [])
                    if isinstance(item, dict) and item.get("type") == "mps_oom_retry_without_dcw"
                ]
                official_request = _official_request_payload(take_params, take_save_dir)
                official_request["memory_policy"] = _jsonable(memory_plan)
                official_request["requested_take_count"] = memory_plan["requested_take_count"]
                official_request["actual_runner_batch_size"] = memory_plan["actual_runner_batch_size"]
                with _official_generation_runner_lock:
                    memory_events.append(
                        _prepare_audio_generation_memory(
                            f"official_generation_take_{pass_index + 1}:retry_without_dcw",
                            release_handler=True,
                        )
                    )
                    official = _run_official_runner_request(official_request, take_work_dir)
                if not official.get("success"):
                    error = official.get("error") or official.get("status_message") or "Official ACE-Step generation failed"
                    if _is_mps_oom_error(error):
                        return memory_failure(error)
                    if _is_acestep_generation_timeout_error(error):
                        return timeout_failure(error)
                    raise RuntimeError(error)
            elif _is_mps_oom_error(error):
                return memory_failure(error)
            if _is_acestep_generation_timeout_error(error):
                return timeout_failure(error)
            raise RuntimeError(error)
        if official.get("lora_status"):
            official_lora_status = official.get("lora_status") or official_lora_status
        if official.get("time_costs"):
            time_costs_by_take.append(_jsonable(official.get("time_costs") or {}))
        if lm_metadata is None and official.get("lm_metadata") is not None:
            lm_metadata = official.get("lm_metadata")
        official_runs.append(
            {
                "take": pass_index + 1,
                "work_dir": str(take_work_dir),
                "save_dir": str(take_save_dir),
                "seed": take_params.get("seed"),
                "batch_size": take_params.get("batch_size"),
                "success": True,
                "retry_reason": retry_reason,
                "dcw_enabled": take_params.get("dcw_enabled"),
                "mps_memory": _jsonable(official.get("mps_memory") or {}),
                "audio_backend_status": _jsonable(official.get("audio_backend_status") or {}),
            }
        )
        for audio in official.get("audios", []):
            index = len(audios)
            preferred_filename = _preferred_audio_filename(params, params["song_model"], index)
            path, filename = _copy_official_audio(result_dir, audio, index, params["audio_format"], preferred_filename)
            audio_id = f"take-{index + 1}"
            audio_params = audio.get("params") or {}
            seed_text = str(audio_params.get("seed") or take_params["seed"] or params["seed"] or "-1")
            audio_audit = _audio_quality_audit(path, params, seed=seed_text)
            adherence = _metadata_adherence(params, metadata_audit, audio_audit)
            title_total = int(memory_plan.get("requested_take_count") or len(official.get("audios", [])) or 1)
            item = {
                "id": audio_id,
                "result_id": result_id,
                "filename": filename,
                "audio_url": _result_public_url(result_id, filename),
                "download_url": _result_public_url(result_id, filename),
                "artist_name": params["artist_name"],
                "title": params["title"] if title_total == 1 else f"{params['title']} {index + 1}",
                "seed": seed_text,
                "sample_rate": int(audio.get("sample_rate") or 48000),
                "runner": "official",
                "audio_backend": params["audio_backend"],
                "use_mlx_dit": params.get("use_mlx_dit"),
                "audio_backend_status": _jsonable(official.get("audio_backend_status") or {}),
                "payload_warnings": params["payload_warnings"],
                "requested_take_count": memory_plan["requested_take_count"],
                "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
                "memory_policy": _jsonable(memory_plan),
                "style_profile": params.get("style_profile", ""),
                "style_caption_tags": params.get("style_caption_tags", ""),
                "style_lyric_tags_applied": _jsonable(params.get("style_lyric_tags_applied") or []),
                "style_conditioning_audit": _jsonable(params.get("style_conditioning_audit") or {}),
                "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
                "generation_metadata_audit": metadata_audit,
                "audio_quality_audit": _jsonable(audio_audit),
                "metadata_adherence": _jsonable(adherence),
                "hit_readiness": _jsonable(hit_readiness),
                "lora_adapter": {
                    "use_lora": params.get("use_lora", False),
                    "path": params.get("lora_adapter_path", ""),
                    "name": params.get("lora_adapter_name", ""),
                    "scale": params.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE),
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
                            "scale": params.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE),
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
            publish_completed_takes()

    if not audios:
        raise RuntimeError("Official ACE-Step runner did not return any audio files")

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
        "requested_take_count": memory_plan["requested_take_count"],
        "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
        "memory_policy": _jsonable(memory_plan),
        "official_runs": _jsonable(official_runs),
        "mps_memory": {
            "events": _jsonable(memory_events),
            "after": _mps_memory_snapshot("official_generation:complete"),
        },
        "inference_steps": params["inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "shift": params["shift"],
        "infer_method": params["infer_method"],
        "sampler_mode": params["sampler_mode"],
        "use_adg": params["use_adg"],
        "use_lora": params["use_lora"],
        "lora_adapter_path": params["lora_adapter_path"],
        "lora_adapter_name": params["lora_adapter_name"],
        "use_lora_trigger": params.get("use_lora_trigger", False),
        "lora_trigger_tag": params.get("lora_trigger_tag", ""),
        "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
        "lora_scale": params["lora_scale"],
        "adapter_model_variant": params["adapter_model_variant"],
        "lora_adapter": _jsonable(official_lora_status),
        "audio_format": params["audio_format"],
        "lm_backend": params["lm_backend"],
        "audio_backend": params["audio_backend"],
        "use_mlx_dit": params.get("use_mlx_dit"),
        "audio_backend_status": _jsonable(
            (official_runs[-1].get("audio_backend_status") if official_runs else {}) or {}
        ),
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
        "style_profile": params.get("style_profile", ""),
        "style_caption_tags": params.get("style_caption_tags", ""),
        "style_lyric_tags_applied": _jsonable(params.get("style_lyric_tags_applied") or []),
        "style_conditioning_audit": _jsonable(params.get("style_conditioning_audit") or {}),
        "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "official_features": params["official_fields"],
        "generation_metadata_audit": _jsonable(metadata_audit),
        "hit_readiness": _jsonable(hit_readiness),
        "payload_quality_gate": _jsonable(params.get("payload_quality_gate") or {}),
        "payload_gate_status": params.get("payload_gate_status") or "",
        "payload_gate_passed": bool(params.get("payload_gate_passed")),
        "payload_gate_blocking_issues": _jsonable(params.get("payload_gate_blocking_issues") or []),
        "tag_coverage": _jsonable(params.get("tag_coverage") or {}),
        "caption_integrity": _jsonable(params.get("caption_integrity") or {}),
        "lyric_duration_fit": _jsonable(params.get("lyric_duration_fit") or {}),
        "effective_settings": _jsonable(params.get("effective_settings") or _effective_settings_summary(params)),
        "settings_coverage": _jsonable(params.get("settings_coverage") or ace_step_settings_registry().get("coverage") or {}),
        "runtime_planner": _jsonable(params.get("runtime_planner") or runtime_planner_report(params)),
        "pro_quality_audit": _jsonable(pro_quality_audit),
        "recommended_take": _jsonable(recommended_take),
        "rerender_suggestions": _jsonable(pro_quality_audit.get("rerender_suggestions") or []),
        "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
        "time_costs": _jsonable({"takes": time_costs_by_take} if len(time_costs_by_take) != 1 else time_costs_by_take[0]),
        "lm_metadata": _jsonable(lm_metadata),
        "audios": audios,
    }
    (result_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    auto_art_context = {
        "result_id": result_id,
        "title": params["title"],
        "tags": params["caption"],
        "payload_warnings": params["payload_warnings"],
    }
    single_art = _maybe_auto_generate_single_art(auto_art_context, params)
    if single_art:
        meta["single_art"] = _jsonable(single_art)
        meta["art"] = _jsonable(single_art)
        for audio in audios:
            audio["art"] = _jsonable(single_art)
        (result_dir / "result.json").write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
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
        "requested_take_count": memory_plan["requested_take_count"],
        "actual_runner_batch_size": memory_plan["actual_runner_batch_size"],
        "memory_policy": _jsonable(memory_plan),
        "official_runs": _jsonable(official_runs),
        "mps_memory": meta["mps_memory"],
        "inference_steps": params["inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "shift": params["shift"],
        "audio_format": params["audio_format"],
        "lm_backend": params["lm_backend"],
        "audio_backend": params["audio_backend"],
        "use_mlx_dit": params.get("use_mlx_dit"),
        "generation_metadata_audit": metadata_audit,
        "hit_readiness": hit_readiness,
        "payload_quality_gate": _jsonable(params.get("payload_quality_gate") or {}),
        "payload_gate_status": params.get("payload_gate_status") or "",
        "payload_gate_passed": bool(params.get("payload_gate_passed")),
        "payload_gate_blocking_issues": _jsonable(params.get("payload_gate_blocking_issues") or []),
        "tag_coverage": _jsonable(params.get("tag_coverage") or {}),
        "caption_integrity": _jsonable(params.get("caption_integrity") or {}),
        "lyric_duration_fit": _jsonable(params.get("lyric_duration_fit") or {}),
        "effective_settings": meta["effective_settings"],
        "settings_coverage": meta["settings_coverage"],
        "runtime_planner": meta["runtime_planner"],
        "pro_quality_audit": pro_quality_audit,
        "recommended_take": recommended_take,
        "rerender_suggestions": pro_quality_audit.get("rerender_suggestions") or [],
        "payload_warnings": params["payload_warnings"],
        "style_profile": params.get("style_profile", ""),
        "style_caption_tags": params.get("style_caption_tags", ""),
        "style_lyric_tags_applied": _jsonable(params.get("style_lyric_tags_applied") or []),
        "style_conditioning_audit": _jsonable(params.get("style_conditioning_audit") or {}),
        "single_art": _jsonable(single_art),
        "ace_step_text_budget": meta["ace_step_text_budget"],
        "use_lora": params["use_lora"],
        "lora_adapter_path": params["lora_adapter_path"],
        "lora_adapter_name": params["lora_adapter_name"],
        "use_lora_trigger": params.get("use_lora_trigger", False),
        "lora_trigger_tag": params.get("lora_trigger_tag", ""),
        "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
        "lora_scale": params["lora_scale"],
        "adapter_model_variant": params["adapter_model_variant"],
        "lora_adapter": _jsonable(official_lora_status),
    }


def _run_advanced_generation_once(params: dict[str, Any]) -> dict[str, Any]:
    params = _enforce_model_correct_render_settings(_album_ace_lm_disabled_payload(dict(params)), source="runner")
    _apply_audio_backend_defaults(params, source="runner")
    if params["requires_official_runner"]:
        return _run_official_generation(params)
    use_random_seed = bool(params["use_random_seed"])
    result_id = uuid.uuid4().hex[:12]
    result_dir = RESULTS_DIR / result_id
    _prepare_audio_generation_memory("fast_generation", release_handler=True)
    with handler_lock:
        active_song_model = _ensure_song_model(params["song_model"])
        lora_status = _apply_lora_request(params)
        debug_save_dir = result_dir / "direct_handler"
        _print_ace_step_terminal_payload(
            params,
            _direct_ace_step_debug_request(params, debug_save_dir, active_song_model),
            debug_save_dir,
        )
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
            "audio_backend": params["audio_backend"],
            "use_mlx_dit": params.get("use_mlx_dit"),
            "payload_warnings": params["payload_warnings"],
            "style_profile": params.get("style_profile", ""),
            "style_caption_tags": params.get("style_caption_tags", ""),
            "style_lyric_tags_applied": _jsonable(params.get("style_lyric_tags_applied") or []),
            "style_conditioning_audit": _jsonable(params.get("style_conditioning_audit") or {}),
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
        "use_lora_trigger": params.get("use_lora_trigger", False),
        "lora_trigger_tag": params.get("lora_trigger_tag", ""),
        "lora_trigger_conditioning_audit": _jsonable(params.get("lora_trigger_conditioning_audit") or {}),
        "lora_scale": params["lora_scale"],
        "adapter_model_variant": params["adapter_model_variant"],
        "lora_adapter": _jsonable(lora_status),
        "audio_format": params["audio_format"],
        "audio_backend": params["audio_backend"],
        "use_mlx_dit": params.get("use_mlx_dit"),
        "tags": params["caption"],
        "tag_list": params["tag_list"],
        "lyrics": params["lyrics"],
        "payload_warnings": params["payload_warnings"],
        "style_profile": params.get("style_profile", ""),
        "style_caption_tags": params.get("style_caption_tags", ""),
        "style_lyric_tags_applied": _jsonable(params.get("style_lyric_tags_applied") or []),
        "style_conditioning_audit": _jsonable(params.get("style_conditioning_audit") or {}),
        "ace_step_text_budget": _jsonable(params.get("ace_step_text_budget") or {}),
        "generation_metadata_audit": _jsonable(metadata_audit),
        "hit_readiness": _jsonable(hit_readiness),
        "payload_quality_gate": _jsonable(params.get("payload_quality_gate") or {}),
        "payload_gate_status": params.get("payload_gate_status") or "",
        "payload_gate_passed": bool(params.get("payload_gate_passed")),
        "payload_gate_blocking_issues": _jsonable(params.get("payload_gate_blocking_issues") or []),
        "tag_coverage": _jsonable(params.get("tag_coverage") or {}),
        "caption_integrity": _jsonable(params.get("caption_integrity") or {}),
        "lyric_duration_fit": _jsonable(params.get("lyric_duration_fit") or {}),
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
    auto_art_context = {
        "result_id": result_id,
        "title": params["title"],
        "tags": params["caption"],
        "payload_warnings": params["payload_warnings"],
    }
    single_art = _maybe_auto_generate_single_art(auto_art_context, params)
    if single_art:
        meta["single_art"] = _jsonable(single_art)
        meta["art"] = _jsonable(single_art)
        for audio in audios:
            audio["art"] = _jsonable(single_art)
        (result_dir / "result.json").write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
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
        "audio_backend": params["audio_backend"],
        "use_mlx_dit": params.get("use_mlx_dit"),
        "generation_metadata_audit": metadata_audit,
        "hit_readiness": hit_readiness,
        "payload_quality_gate": _jsonable(params.get("payload_quality_gate") or {}),
        "payload_gate_status": params.get("payload_gate_status") or "",
        "payload_gate_passed": bool(params.get("payload_gate_passed")),
        "payload_gate_blocking_issues": _jsonable(params.get("payload_gate_blocking_issues") or []),
        "tag_coverage": _jsonable(params.get("tag_coverage") or {}),
        "caption_integrity": _jsonable(params.get("caption_integrity") or {}),
        "lyric_duration_fit": _jsonable(params.get("lyric_duration_fit") or {}),
        "effective_settings": meta["effective_settings"],
        "settings_coverage": meta["settings_coverage"],
        "runtime_planner": meta["runtime_planner"],
        "pro_quality_audit": pro_quality_audit,
        "recommended_take": recommended_take,
        "rerender_suggestions": pro_quality_audit.get("rerender_suggestions") or [],
        "payload_warnings": params["payload_warnings"],
        "style_profile": params.get("style_profile", ""),
        "style_caption_tags": params.get("style_caption_tags", ""),
        "style_lyric_tags_applied": _jsonable(params.get("style_lyric_tags_applied") or []),
        "style_conditioning_audit": _jsonable(params.get("style_conditioning_audit") or {}),
        "single_art": _jsonable(single_art),
        "ace_step_text_budget": meta["ace_step_text_budget"],
        "time_costs": meta["time_costs"],
    }


def _run_advanced_generation(raw_payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_training_idle()
    raw_payload = _album_ace_lm_disabled_payload(raw_payload)
    params = _enforce_model_correct_render_settings(_parse_generation_payload(raw_payload), source="generation")
    if raw_payload.get("generation_task_id"):
        params["generation_task_id"] = str(raw_payload.get("generation_task_id") or "").strip()
    if params["instrumental"] and not params["lyrics"].strip():
        params["lyrics"] = "[Instrumental]"
    params = _defer_library_save_until_vocal_pass(params)
    max_attempts = int(params.get("vocal_intelligibility_attempts") or ACEJAM_VOCAL_INTELLIGIBILITY_ATTEMPTS)
    max_attempts = max(1, min(32, max_attempts))
    if not _vocal_gate_required(params):
        result = _run_advanced_generation_once(params)
        if result.get("error_type") in {"memory_error", "timeout_error"}:
            return result
        _apply_vocal_intelligibility_gate_to_result(result, params, attempt=1, max_attempts=1)
        return result

    preflight_result = _run_lora_preflight_verifier(params)
    if preflight_result is not None:
        preflight = preflight_result.get("lora_preflight") if isinstance(preflight_result.get("lora_preflight"), dict) else {}
        synthetic_gate = {
            "status": preflight.get("status") or "lora_preflight_failed",
            "passed": False,
            "blocking": True,
            "transcript_preview": preflight.get("attempts") or [],
        }
        return _finalize_failed_primary_with_diagnostics(
            preflight_result,
            params,
            gate=synthetic_gate,
            history=[],
            failure_reason=str(preflight_result.get("error") or "LoRA preflight failed; long render was not started."),
            run_diagnostics=False,
        )

    preflight_result = _run_vocal_preflight_verifier(params)
    if preflight_result is not None:
        gate = preflight_result.get("vocal_intelligibility_gate") if isinstance(preflight_result.get("vocal_intelligibility_gate"), dict) else {}
        return _finalize_failed_primary_with_diagnostics(
            preflight_result,
            params,
            gate=gate,
            history=[],
            failure_reason=str(preflight_result.get("error") or "Selected model vocal preflight failed; long render was not started."),
        )

    history: list[dict[str, Any]] = []
    last_gate: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_params = (
            params
            if attempt == 1
            else _vocal_retry_params(params, attempt=attempt, last_gate=last_gate, allow_lora_changes=False)
        )
        result = _run_advanced_generation_once(attempt_params)
        if result.get("error_type") in {"memory_error", "timeout_error"}:
            _apply_vocal_intelligibility_gate_to_result(result, attempt_params, attempt=attempt, max_attempts=max_attempts)
            return result
        gate = _apply_vocal_intelligibility_gate_to_result(
            result,
            attempt_params,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        _annotate_generation_attempt_result(
            result,
            attempt_params,
            role="primary",
            gate=gate,
            requested_params=params,
            failure_reason=_attempt_failure_reason(result, gate),
        )
        last_result = result
        last_gate = gate
        history_item = {
            "attempt": attempt,
            "attempt_role": "primary",
            "result_id": result.get("result_id"),
            "song_model": attempt_params.get("song_model"),
            "requested_song_model": params.get("song_model"),
            "actual_song_model": result.get("actual_song_model") or result.get("active_song_model") or attempt_params.get("song_model"),
            "with_lora": parse_bool(attempt_params.get("use_lora"), False),
            "lora_scale": attempt_params.get("lora_scale"),
            "lora_quality_status": result.get("lora_quality_status"),
            "inference_steps": attempt_params.get("inference_steps"),
            "shift": attempt_params.get("shift"),
            "status": gate.get("status"),
            "blocking": gate.get("blocking"),
            "passed_audio_ids": gate.get("passed_audio_ids") or [],
            "transcripts": [
                {
                    "file": Path(str(item.get("path") or "")).name,
                    "status": item.get("status"),
                    "word_count": item.get("word_count"),
                    "unique_word_count": item.get("unique_word_count"),
                    "keyword_hits": item.get("keyword_hits"),
                    "filler_ratio": item.get("filler_ratio"),
                    "repeat_ratio": item.get("repeat_ratio"),
                    "text": str(item.get("text") or "")[:180],
                    "issue": item.get("issue"),
                }
                for item in gate.get("transcripts", [])
            ],
        }
        history.append(history_item)
        if gate.get("passed") and not gate.get("blocking"):
            result["vocal_intelligibility_history"] = history
            if params.get("lora_preflight"):
                result["lora_preflight"] = params.get("lora_preflight")
            if params.get("vocal_preflight"):
                result["vocal_preflight"] = params.get("vocal_preflight")
            _save_vocal_gate_passed_result_to_library(result, attempt_params)
            _persist_result_update(result)
            return result
        if gate.get("needs_review") or gate.get("status") == "needs_review":
            result["vocal_intelligibility_history"] = history
            if params.get("lora_preflight"):
                result["lora_preflight"] = params.get("lora_preflight")
            if params.get("vocal_preflight"):
                result["vocal_preflight"] = params.get("vocal_preflight")
            _persist_result_update(result)
            return result
        if gate.get("status") == "error":
            result["vocal_intelligibility_history"] = history
            if params.get("lora_preflight"):
                result["lora_preflight"] = params.get("lora_preflight")
            if params.get("vocal_preflight"):
                result["vocal_preflight"] = params.get("vocal_preflight")
            _persist_result_update(result)
            return result
        issue_summary = ", ".join(
            sorted(
                {
                    str(item.get("issue") or item.get("status") or "failed")
                    for item in gate.get("transcripts", [])
                    if not item.get("passed")
                }
            )
        ) or str(gate.get("status") or "failed")
        print(
            f"Vocal intelligibility retry: attempt {attempt}/{max_attempts}: {issue_summary} "
            f"(model {attempt_params.get('song_model')}, result {result.get('result_id')})",
            flush=True,
        )

    if last_result is not None:
        compact_history = "; ".join(
            f"attempt {item['attempt']} model={item.get('song_model')} result {item['result_id']} status={item['status']} "
            f"transcripts={[t.get('text') or t.get('issue') for t in item['transcripts']]}"
            for item in history
        )
        debug_path = RESULTS_DIR / "vocal_intelligibility_gate.jsonl"
        return _finalize_failed_primary_with_diagnostics(
            last_result,
            params,
            gate=last_gate,
            history=history,
            failure_reason=(
                "Vocal intelligibility gate failed after "
                f"{max_attempts} primary attempt(s). Debug log: {debug_path}. History: {compact_history}"
            ),
        )

    debug_path = RESULTS_DIR / "vocal_intelligibility_gate.jsonl"
    compact_history = "; ".join(
        f"attempt {item['attempt']} model={item.get('song_model')} result {item['result_id']} status={item['status']} "
        f"transcripts={[t.get('text') or t.get('issue') for t in item['transcripts']]}"
        for item in history
    )
    raise RuntimeError(
        "Vocal intelligibility gate failed after "
        f"{max_attempts} attempt(s). ACE-Step kept producing unintelligible vocals. "
        f"Debug log: {debug_path}. History: {compact_history}"
    )


def _portfolio_generation_payload(raw_payload: dict[str, Any], model_item: dict[str, Any], family_id: str) -> dict[str, Any]:
    model_name = str(model_item.get("model") or "")
    quality_profile = _default_quality_profile_for_payload(raw_payload, "text2music")
    model_defaults = quality_profile_model_settings(model_name, quality_profile)
    payload = dict(raw_payload or {})
    has_vocal_lyrics = bool(str(payload.get("lyrics") or "").strip() and str(payload.get("lyrics") or "").strip().lower() != "[instrumental]")
    lm_defaults_enabled = not has_vocal_lyrics and _requested_ace_lm_model(raw_payload) != "none"
    payload.update(
        {
            "task_type": "text2music",
            "quality_profile": quality_profile,
            "song_model": model_name,
            "batch_size": 1,
            "inference_steps": int(model_item.get("default_steps") or _quality_default_steps(model_name, quality_profile)),
            "guidance_scale": float(model_item.get("default_guidance_scale") or model_defaults["guidance_scale"]),
            "shift": float(model_item.get("default_shift") or model_defaults["shift"]),
            "ace_lm_model": "none" if has_vocal_lyrics else _requested_ace_lm_model(raw_payload),
            "allow_supplied_lyrics_lm": False if has_vocal_lyrics else parse_bool(raw_payload.get("allow_supplied_lyrics_lm"), False),
            "planner_lm_provider": normalize_provider(raw_payload.get("planner_lm_provider") or raw_payload.get("planner_provider") or "ollama"),
            "planner_model": str(raw_payload.get("planner_model") or raw_payload.get("planner_ollama_model") or raw_payload.get("ollama_model") or ""),
            "render_strategy": SONG_PORTFOLIO_STRATEGY,
            "thinking": False if has_vocal_lyrics else parse_bool(raw_payload.get("thinking"), lm_defaults_enabled and DOCS_BEST_LM_DEFAULTS["thinking"]),
            "sample_mode": False if has_vocal_lyrics else parse_bool(raw_payload.get("sample_mode"), False),
            "sample_query": "" if has_vocal_lyrics else str(raw_payload.get("sample_query") or ""),
            "use_format": False if has_vocal_lyrics else parse_bool(raw_payload.get("use_format"), lm_defaults_enabled and DOCS_BEST_LM_DEFAULTS["use_format"]),
            "use_cot_metas": False if has_vocal_lyrics else parse_bool(raw_payload.get("use_cot_metas"), lm_defaults_enabled and DOCS_BEST_LM_DEFAULTS["use_cot_metas"]),
            "use_cot_caption": False if has_vocal_lyrics else parse_bool(raw_payload.get("use_cot_caption"), lm_defaults_enabled and DOCS_BEST_LM_DEFAULTS["use_cot_caption"]),
            "use_cot_lyrics": False if has_vocal_lyrics else parse_bool(raw_payload.get("use_cot_lyrics"), lm_defaults_enabled and DOCS_BEST_LM_DEFAULTS["use_cot_lyrics"]),
            "use_cot_language": False if has_vocal_lyrics else parse_bool(raw_payload.get("use_cot_language"), lm_defaults_enabled and DOCS_BEST_LM_DEFAULTS["use_cot_language"]),
            "use_constrained_decoding": parse_bool(raw_payload.get("use_constrained_decoding"), DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"]),
            "lm_temperature": clamp_float(raw_payload.get("lm_temperature"), DOCS_BEST_LM_DEFAULTS["lm_temperature"], 0.0, 2.0),
            "lm_cfg_scale": clamp_float(raw_payload.get("lm_cfg_scale"), DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"], 0.0, 10.0),
            "lm_top_p": clamp_float(raw_payload.get("lm_top_p"), DOCS_BEST_LM_DEFAULTS["lm_top_p"], 0.0, 1.0),
            "lm_top_k": clamp_int(raw_payload.get("lm_top_k"), DOCS_BEST_LM_DEFAULTS["lm_top_k"], 0, 200),
            "audio_code_string": "",
            "src_audio_id": "",
            "src_result_id": "",
            "reference_audio_id": "",
            "reference_result_id": "",
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
    raw_payload = _album_ace_lm_disabled_payload(raw_payload)
    task_type = normalize_task_type(raw_payload.get("task_type") or "text2music")
    if task_type != "text2music":
        raise ValueError("Render all official models is only available for Simple/Custom text2music.")
    installed = _installed_acestep_models()
    portfolio = album_model_portfolio(installed)
    missing = [str(item["model"]) for item in portfolio if not item.get("installed")]
    family_id = raw_payload.get("portfolio_family_id") or f"songfam-{uuid.uuid4().hex[:10]}"
    logs = [
        "Render all official ACE-Step render models requested.",
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
            f"MLX Media started downloading {len(missing)} missing model(s). "
            "The official model portfolio song render will resume after install."
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
        "active_song_model": "all official render models",
        "runner": "portfolio",
        "params": validation.get("normalized_payload") or validation_payload,
        "payload_warnings": validation.get("payload_warnings") or [],
        "logs": logs,
    }


app = Server(title="MLX Media")


@app.middleware("http")
async def _prevent_stale_react_shell_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    content_type = response.headers.get("content-type", "")
    is_react_shell = path == "/" or (
        (path == "/v2" or path.startswith("/v2/")) and content_type.startswith("text/html")
    )
    if is_react_shell:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---- React (shadcn) UI mount at /v2 ------------------------------------------------
# When ACEJAM_DEV=1 the Vite dev server runs on :5173; allow CORS so it can
# reach this FastAPI server without rebuilding on every change. In production
# the built React app is served as static assets from app/web/dist mounted on
# /v2, so the page and the API share an origin and CORS is not needed.
if os.getenv("ACEJAM_DEV", "").strip() in {"1", "true", "yes"}:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

_REACT_WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
if _REACT_WEB_DIST.is_dir():
    # html=True makes StaticFiles serve index.html for unknown paths under /v2,
    # which is what we need for the SPA (HashRouter keeps client routes after #).
    app.mount(
        "/v2",
        StaticFiles(directory=str(_REACT_WEB_DIST), html=True),
        name="acejam_web_v2",
    )


@app.api(name="compose", concurrency_limit=1, time_limit=120)
def compose(
    description: str,
    audio_duration: float = 60.0,
    composer_profile: str = "auto",
    instrumental: bool = False,
    ollama_model: str = "",
    planner_lm_provider: str = "ollama",
    planner_model: str = "",
    planner_llm_settings: dict[str, Any] | None = None,
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
            planner_llm_settings=planner_llm_settings,
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
    return json.dumps(_community_feed(50))


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
    local_llm_settings = _load_local_llm_settings()
    lmstudio_catalog = lmstudio_model_catalog()
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
            "default_planner_lm_provider": local_llm_settings["provider"],
            "default_album_planner_ollama_model": local_llm_settings["chat_model"] or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            "default_album_embedding_model": local_llm_settings["embedding_model"] or DEFAULT_ALBUM_EMBEDDING_MODEL,
            "default_album_planner_model": local_llm_settings["chat_model"] or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
            "default_album_embedding_provider": local_llm_settings["embedding_provider"],
            "local_llm": {
                "default_provider": local_llm_settings["provider"],
                "ollama_host": _ollama_host(),
                "lmstudio_host": lmstudio_catalog.get("host", ""),
                "settings": local_llm_settings,
                "mlx_policy": local_llm_settings.get("mlx_policy", "auto"),
                "full_mlx": bool(_IS_APPLE_SILICON and str(local_llm_settings.get("mlx_policy")) == "full_mlx"),
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
            "official_model_registry": {
                name: {**meta, **_official_model_runtime_status(name)}
                for name, meta in official_model_registry().items()
            },
            "official_boot_downloads": _jsonable(_boot_model_download_status),
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


@app.api(name="generate_album", concurrency_limit=1, time_limit=ACEJAM_GENERATE_ALBUM_TIME_LIMIT_SECONDS)
def generate_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL,
    language: str = "en",
    song_model: str = "auto",
    embedding_model: str = DEFAULT_ALBUM_EMBEDDING_MODEL,
    ace_lm_model: str = "none",
    request_json: str = "",
    planner_lm_provider: str = "ollama",
    embedding_lm_provider: str = "ollama",
) -> str:
    """Plan album with the selected local planning engine, then generate through the audio engine."""
    logs: list[str] = []
    try:
        request_payload = json.loads(request_json or "{}")
        request_payload.setdefault("concept", concept)
        request_payload.setdefault("num_tracks", num_tracks)
        request_payload.setdefault("track_duration", track_duration)
        request_payload.setdefault("language", language)
        request_payload.setdefault("album_writer_mode", "per_track_writer_loop")
        recovered_concept = _recover_album_request_concept(concept, request_payload)
        if recovered_concept:
            concept = recovered_concept
            request_payload["concept"] = recovered_concept
            request_payload.setdefault("user_prompt", recovered_concept)
        global_llm_settings = _load_local_llm_settings()
        request_payload = _album_ace_lm_disabled_payload(request_payload)
        ace_lm_model = "none"
        request_payload["ace_lm_model"] = "none"
        album_job_id = str(request_payload.get("album_job_id") or "")
        album_debug_id = album_job_id or f"manual-{uuid.uuid4().hex[:12]}"
        album_debug = AlbumRunDebugLogger(DATA_DIR, album_debug_id)
        request_payload["album_debug_dir"] = str(album_debug.root)
        request_payload["llm_debug_log_file"] = str(album_debug.path("04_agent_responses.jsonl"))
        album_debug.write_json(
            "01_request.json",
            {
                "concept": concept,
                "num_tracks": num_tracks,
                "track_duration": track_duration,
                "language": language,
                "song_model": song_model,
                "request_payload": request_payload,
                "album_writer_mode": request_payload.get("album_writer_mode"),
            },
        )
        logs.append(f"Album debug log dir: {album_debug.root}")
        _album_job_log(
            album_job_id,
            f"Album debug log dir: {album_debug.root}",
            album_debug_dir=str(album_debug.root),
            album_payload_gate_version=ALBUM_PAYLOAD_GATE_VERSION,
        )
        planner_lm_provider = _album_planner_provider_from_payload(request_payload, planner_lm_provider or "ollama")
        request_payload["planner_lm_provider"] = planner_lm_provider
        embedding_lm_provider = _embedding_provider_from_payload(request_payload, embedding_lm_provider or "ollama")
        request_payload["embedding_lm_provider"] = embedding_lm_provider
        if not ollama_model:
            ollama_model = str(global_llm_settings.get("chat_model") or (DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL if planner_lm_provider == "ollama" else ""))
        if not embedding_model:
            embedding_model = str(global_llm_settings.get("embedding_model") or DEFAULT_ALBUM_EMBEDDING_MODEL)
        track_duration = parse_duration_seconds(request_payload.get("track_duration") or request_payload.get("duration") or track_duration, track_duration)
        album_options = _album_options_from_payload(request_payload, song_model=song_model)
        planning_engine = str(album_options.get("agent_engine") or "acejam_agents")
        planning_engine_label = _album_agent_engine_label_value(planning_engine)
        album_options["album_debug_dir"] = str(album_debug.root)
        album_options["llm_debug_log_file"] = str(album_debug.path("04_agent_responses.jsonl"))
        album_debug.write_json(
            "02_contract.json",
            {
                "user_album_contract": album_options.get("user_album_contract"),
                "input_contract_applied": album_options.get("input_contract_applied"),
                "blocked_unsafe_count": album_options.get("blocked_unsafe_count"),
                "album_options_preview": {
                    key: value
                    for key, value in album_options.items()
                    if key not in {"album_model_portfolio", "user_album_contract"}
                },
            },
        )
        planned_tracks = _json_list(request_payload.get("tracks") or request_payload.get("planned_tracks"))
        render_from_existing_tracks = bool(planned_tracks) and (
            parse_bool(request_payload.get("render_from_existing_tracks"), False)
            or parse_bool(request_payload.get("skip_album_planning"), False)
            or str(request_payload.get("album_generation_mode") or "").strip().lower()
            in {"render_existing_tracks", "direct_render", "ui_tracks"}
        )
        album_lora_request = _lora_adapter_request(request_payload)
        if album_lora_request.get("use_lora") and album_lora_request.get("adapter_song_model"):
            adapter_song_model = str(album_lora_request.get("adapter_song_model") or "").strip()
            if adapter_song_model:
                request_payload["song_model"] = adapter_song_model
                request_payload["requested_song_model"] = adapter_song_model
                request_payload["song_model_strategy"] = "single_model_album"
                album_options["requested_song_model"] = adapter_song_model
                album_options["song_model_strategy"] = "single_model_album"
                song_model = adapter_song_model
                logs.append(
                    "Album LoRA model lock: using "
                    f"{adapter_song_model} because the selected adapter was trained for that model."
                )
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
        if album_lora_request.get("use_lora"):
            for item in album_models:
                _validate_lora_request_for_song_model(album_lora_request, str(item["model"]))
            request_payload.update(
                {
                    "use_lora": album_lora_request["use_lora"],
                    "lora_adapter_path": album_lora_request["lora_adapter_path"],
                    "lora_adapter_name": album_lora_request["lora_adapter_name"],
                    "use_lora_trigger": album_lora_request["use_lora_trigger"],
                    "lora_trigger_tag": album_lora_request["lora_trigger_tag"],
                    "lora_trigger_source": album_lora_request.get("lora_trigger_source", ""),
                    "lora_trigger_aliases": album_lora_request.get("lora_trigger_aliases", []),
                    "lora_trigger_candidates": album_lora_request.get("lora_trigger_candidates", []),
                    "lora_scale": album_lora_request["lora_scale"],
                    "adapter_model_variant": album_lora_request["adapter_model_variant"],
                    "adapter_song_model": album_lora_request["adapter_song_model"],
                }
            )
            logs.append(
                "Album LoRA: "
                f"{album_lora_request.get('lora_adapter_name') or Path(str(album_lora_request.get('lora_adapter_path') or '')).name}; "
                f"scale={album_lora_request.get('lora_scale')}; "
                f"trigger={album_lora_request.get('lora_trigger_tag') or 'off'}."
            )

        if render_from_existing_tracks:
            album_options["album_writer_mode"] = "render_existing_tracks"
            logs.append(
                f"Phase 1 skipped: rendering {len(planned_tracks)} UI-approved track(s) directly; "
                "no album agents will run on Generate."
            )
            _album_job_log(
                album_job_id,
                f"Phase 1 skipped: rendering {len(planned_tracks)} UI-approved track(s) directly.",
                status="Rendering approved UI tracks",
                stage="render_existing_tracks",
                current_task="Render existing Album Wizard tracks",
                progress=8,
                planning_engine="existing_ui_tracks",
                album_writer_mode="render_existing_tracks",
                custom_agents_used=False,
                crewai_used=False,
            )
            tracks = []
            for index, raw_track in enumerate(planned_tracks, start=1):
                track = dict(raw_track) if isinstance(raw_track, dict) else {}
                track["track_number"] = clamp_int(track.get("track_number"), index, 1, 999)
                track["title"] = str(track.get("title") or f"Track {index}").strip()
                track["duration"] = parse_duration_seconds(track.get("duration") or track_duration, track_duration)
                track["language"] = str(track.get("language") or track.get("vocal_language") or language or "en")
                track["vocal_language"] = str(track.get("vocal_language") or track.get("language") or language or "en")
                track["bpm"] = clamp_int(track.get("bpm") or request_payload.get("bpm"), DEFAULT_BPM, 40, 220)
                track["key_scale"] = normalize_key_scale(track.get("key_scale") or track.get("key") or request_payload.get("key_scale") or DEFAULT_KEY_SCALE)
                track["time_signature"] = str(track.get("time_signature") or request_payload.get("time_signature") or "4")
                if track.get("caption") and not track.get("tags"):
                    track["tags"] = str(track.get("caption") or "")
                if track.get("tags") and not track.get("caption"):
                    track["caption"] = str(track.get("tags") or "")
                track["planning_status"] = str(track.get("planning_status") or "ui_approved")
                track["agent_complete_payload"] = False
                tracks.append(track)
            result = {
                "success": True,
                "tracks": tracks,
                "logs": [
                    "Album Generate used existing UI tracks and skipped album_crew.plan_album.",
                ],
                "planning_engine": "existing_ui_tracks",
                "album_writer_mode": "render_existing_tracks",
                "custom_agents_used": False,
                "crewai_used": False,
                "toolbelt_fallback": False,
                "agent_debug_dir": str(album_debug.root),
                "agent_rounds": [],
                "agent_repair_count": 0,
                "memory_enabled": False,
                "context_chunks": 0,
                "retrieval_rounds": 0,
                "input_contract_applied": False,
                "contract_repair_count": 0,
                "blocked_unsafe_count": 0,
                "toolkit_report": {},
            }
        else:
            _ensure_album_agent_modules_current()
            from album_crew import plan_album as _plan_album

            logs.append(f"Phase 1: Planning album with {planning_engine_label} and deterministic gates...")
            logs.append(f"Album writer mode: {album_options.get('album_writer_mode')}; per-track AI writing, audit, repair and debug before render.")
            _album_job_log(
                album_job_id,
                f"Phase 1: Planning album with {planning_engine_label} and deterministic gates.",
                status="Planning album",
                progress=3,
                planning_engine=planning_engine,
                album_writer_mode=album_options.get("album_writer_mode"),
                crewai_used=planning_engine == "crewai_micro",
            )
            result = _plan_album(
                concept=concept,
                num_tracks=num_tracks,
                track_duration=track_duration,
                ollama_model=ollama_model,
                language=language,
                embedding_model=embedding_model,
                options=album_options,
                use_crewai=True,
                input_tracks=planned_tracks if planned_tracks else None,
                planner_provider=planner_lm_provider,
                embedding_provider=embedding_lm_provider,
            )
        tracks = result.get("tracks", [])
        album_debug.write_json("04_plan_outputs.json", result)
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
                album_writer_mode=str(result.get("album_writer_mode") or album_options.get("album_writer_mode") or "per_track_writer_loop"),
                custom_agents_used=bool(result.get("custom_agents_used")),
                crewai_used=bool(result.get("crewai_used")),
                toolbelt_fallback=bool(result.get("toolbelt_fallback")),
                memory_enabled=bool(result.get("memory_enabled") or actual_memory.get("enabled")),
                context_chunks=int(result.get("context_chunks") or actual_memory.get("context_chunks") or 0),
                retrieval_rounds=int(result.get("retrieval_rounds") or actual_memory.get("retrieval_rounds") or 0),
                agent_context_store=str(result.get("agent_context_store") or actual_memory.get("context_store") or ""),
                context_store_index=str(result.get("context_store_index") or ""),
                sequence_repair_count=int(result.get("sequence_repair_count") or 0),
                sequence_report=result.get("sequence_report") or {},
                agent_debug_dir=str(result.get("agent_debug_dir") or album_debug.root),
                agent_repair_count=int(result.get("agent_repair_count") or 0),
                agent_rounds=result.get("agent_rounds") or [],
                input_contract=result.get("input_contract") or contract_prompt_context(album_options.get("user_album_contract")),
                input_contract_applied=bool(result.get("input_contract_applied") or album_options.get("input_contract_applied")),
                input_contract_version=str(result.get("input_contract_version") or USER_ALBUM_CONTRACT_VERSION),
                blocked_unsafe_count=int(result.get("blocked_unsafe_count") or album_options.get("blocked_unsafe_count") or 0),
                contract_repair_count=int(result.get("contract_repair_count") or 0),
                album_debug_dir=str(album_debug.root),
                album_payload_gate_version=ALBUM_PAYLOAD_GATE_VERSION,
            )

        planning_failed_tracks = [
            track for track in tracks
            if track.get("skip_render") or str(track.get("planning_status") or "").lower() == "failed"
        ]
        renderable_tracks = [
            track for track in tracks
            if not (track.get("skip_render") or str(track.get("planning_status") or "").lower() == "failed")
        ]
        if not result.get("success", True) or not tracks or not renderable_tracks:
            logs.append("ERROR: Album planning failed")
            return json.dumps({
                "tracks": tracks,
                "logs": logs,
                "success": False,
                "error": result.get("error") or "Planning failed",
                "album_debug_dir": str(album_debug.root),
                "album_payload_gate_version": ALBUM_PAYLOAD_GATE_VERSION,
            })
        if planning_failed_tracks:
            logs.append(
                "Phase 1 partial: "
                f"{len(renderable_tracks)}/{len(tracks)} tracks planned; "
                f"failed tracks: {', '.join(str(item.get('track_number') or '?') for item in planning_failed_tracks)}."
            )

        logs.append(
            f"Phase 1 complete: {len(tracks)} UI track(s) ready"
            if render_from_existing_tracks
            else f"Phase 1 complete: {len(tracks)} tracks planned"
        )
        if render_from_existing_tracks:
            logs.append("Local AI Writer/Planner skipped on Generate; using the Album Wizard tracks already filled in the UI.")
        else:
            logs.append(
                f"Local AI Writer/Planner: {provider_label(planner_lm_provider)} ({ollama_model}) for lyrics, tags, BPM, key and captions; "
                "ACE-Step LM disabled for album agents."
            )
            logs.append("ACE-Step LM disabled for album agents.")
        logs.append(f"Album model policy: {len(album_models)} full model album(s), {len(tracks) * len(album_models)} total render(s)")
        agent_direct_payloads = str(result.get("planning_engine") or "") in {"acejam_agents", "crewai_micro"}
        album_global_caption = "" if agent_direct_payloads else build_album_global_sonic_caption(
            concept,
            tracks,
            existing=request_payload.get("global_caption") or "",
        )
        logs.append(f"AlbumPayloadQualityGate: enabled ({ALBUM_PAYLOAD_GATE_VERSION}); debug={album_debug.root}")
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

        def publish_album_progress(
            *,
            current_album_id: str = "",
            current_album_model: str = "",
            current_album_model_label: str = "",
            current_album_tracks: list[dict[str, Any]] | None = None,
            current_album_audios: list[dict[str, Any]] | None = None,
        ) -> None:
            if not album_job_id:
                return
            expected = max(1, len(tracks) * len(album_models))
            completed_renders = sum(
                1
                for base_track in tracks
                for model_result in (base_track.get("model_results") or [])
                if isinstance(model_result, dict) and model_result.get("generated")
            )
            progress = max(10, min(95, 10 + int(85 * completed_renders / expected)))
            progress_model_albums = list(model_albums)
            if current_album_id:
                partial_tracks = list(current_album_tracks or [])
                partial_audios = list(current_album_audios or [])
                progress_model_albums.append(
                    {
                        "album_id": current_album_id,
                        "album_family_id": album_family_id,
                        "album_model": current_album_model,
                        "album_model_label": current_album_model_label or model_label(current_album_model),
                        "album_status": "running",
                        "track_count": len(tracks),
                        "generated_count": sum(1 for item in partial_tracks if item.get("generated")),
                        "failed_count": sum(1 for item in partial_tracks if not item.get("generated")),
                        "tracks": _jsonable(partial_tracks),
                        "audios": _jsonable(partial_audios),
                    }
                )
            _set_album_job(
                album_job_id,
                status=f"{completed_renders}/{expected} volledige track-render(s) klaar",
                progress=progress,
                generated_count=completed_renders,
                completed_track_count=completed_renders,
                completed_audio_count=len(generated_audios),
                expected_count=expected,
                album_family_id=album_family_id,
                result={
                    "success": False,
                    "in_progress": completed_renders < expected,
                    "partial": completed_renders < expected,
                    "album_family_id": album_family_id,
                    "album_status": "running" if completed_renders < expected else "completed",
                    "track_count": len(tracks),
                    "expected_renders": expected,
                    "generated_count": completed_renders,
                    "completed_track_count": completed_renders,
                    "completed_audio_count": len(generated_audios),
                    "full_tracks_ready": completed_renders,
                    "tracks": _jsonable(tracks),
                    "audios": _jsonable(generated_audios),
                    "model_albums": _jsonable(progress_model_albums),
                    "album_model_portfolio": _jsonable(album_models),
                    "album_debug_dir": str(album_debug.root),
                    "album_payload_gate_version": ALBUM_PAYLOAD_GATE_VERSION,
                    "logs": logs[-200:],
                },
            )

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
                raw_track_artist = track.get("artist_name") or track.get("artist")
                producer_credit = str(track.get("producer_credit") or "").strip()
                if producer_credit:
                    normalized_raw_artist = normalize_artist_name(raw_track_artist or "", "")
                    normalized_title_artist = normalize_artist_name(track_title, "")
                    if not normalized_raw_artist or normalized_raw_artist == normalized_title_artist or normalized_raw_artist.lower() == "unknown":
                        raw_track_artist = producer_credit
                track_artist = normalize_artist_name(
                    raw_track_artist,
                    derive_artist_name(track_title, concept, track.get("tags") or track.get("caption") or "", i),
                )
                track["artist_name"] = track_artist
                if producer_credit:
                    track["producer_credit"] = producer_credit
                base_track["artist_name"] = track_artist
                if producer_credit:
                    base_track["producer_credit"] = producer_credit
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
                if track.get("skip_render") or str(track.get("planning_status") or "").lower() == "failed":
                    track["generated"] = False
                    track["album_id"] = album_id
                    track["album_family_id"] = album_family_id
                    track["error"] = str(track.get("planning_error") or track.get("error") or "Track planning failed")
                    track["payload_gate_status"] = track.get("payload_gate_status") or "planning_failed"
                    track["payload_gate_passed"] = False
                    album_debug.append_jsonl(
                        "07_generation_results.jsonl",
                        {
                            "status": "skipped_planning_failed",
                            "album_id": album_id,
                            "album_model": track_model,
                            "track_number": track.get("track_number", i + 1),
                            "title": track_title,
                            "error": track["error"],
                            "debug_paths": track.get("debug_paths") or {},
                        },
                    )
                    base_track["model_results"].append(_jsonable(track))
                    album_tracks.append(track)
                    logs.append(f"  SKIPPED planning-failed track {i + 1}/{len(tracks)}: {track_title} -> {track['error']}")
                    _album_job_log(
                        album_job_id,
                        f"SKIPPED planning-failed track {i + 1}/{len(tracks)}: {track_title}",
                        current_model_album=track_model,
                        current_track=f"{i + 1}/{len(tracks)} {track_title}",
                        errors=[track["error"]],
                    )
                    continue
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
                    track_has_vocal_lyrics = bool(str(track.get("lyrics") or "").strip() and str(track.get("lyrics") or "").strip().lower() != "[instrumental]")
                    direct_agent_payload = bool(track.get("agent_complete_payload") or agent_direct_payloads)
                    vocal_clarity_probe = {
                        **request_payload,
                        **track,
                        "task_type": "text2music",
                        "lyrics": track.get("lyrics", ""),
                    }
                    vocal_clarity_recovery = _vocal_clarity_recovery_enabled(vocal_clarity_probe)
                    if vocal_clarity_recovery and track_has_vocal_lyrics:
                        logs.append("  Vocal clarity recovery: direct lyrics render kept; ACE LM rewrite disabled for supplied lyrics.")
                    album_allow_ace_lm_rewrite = False
                    album_use_supplied_lyrics_lm = False
                    track_lm_model = "none"
                    track_lm_enabled = False
                    def _album_lm_switch(field: str, default: bool) -> bool:
                        if direct_agent_payload:
                            return False
                        if track_has_vocal_lyrics and not album_use_supplied_lyrics_lm:
                            return False
                        if track_has_vocal_lyrics and track_lm_enabled:
                            return default
                        return parse_bool(track.get(field, request_payload.get(field)), default)

                    lora_source = dict(request_payload)
                    for lora_key in (
                        "use_lora",
                        "lora_adapter_path",
                        "lora_adapter_name",
                        "use_lora_trigger",
                        "lora_trigger_tag",
                        "lora_trigger_source",
                        "lora_trigger_aliases",
                        "lora_trigger_candidates",
                        "lora_scale",
                        "adapter_model_variant",
                        "adapter_song_model",
                    ):
                        if lora_key in track and track.get(lora_key) not in (None, ""):
                            lora_source[lora_key] = track.get(lora_key)
                    track_lora_request = _lora_adapter_request(lora_source)
                    _validate_lora_request_for_song_model(track_lora_request, track_model)
                    track_lora_trigger_applied = bool(
                        track_lora_request.get("use_lora")
                        and track_lora_request.get("use_lora_trigger")
                        and track_lora_request.get("lora_trigger_tag")
                    )
                    track.update(
                        {
                            "use_lora": track_lora_request["use_lora"],
                            "lora_adapter_path": track_lora_request["lora_adapter_path"],
                            "lora_adapter_name": track_lora_request["lora_adapter_name"],
                            "use_lora_trigger": track_lora_request["use_lora_trigger"],
                            "lora_trigger_tag": track_lora_request["lora_trigger_tag"],
                            "lora_trigger_source": track_lora_request.get("lora_trigger_source", ""),
                            "lora_trigger_aliases": track_lora_request.get("lora_trigger_aliases", []),
                            "lora_trigger_candidates": track_lora_request.get("lora_trigger_candidates", []),
                            "lora_scale": track_lora_request["lora_scale"],
                            "adapter_model_variant": track_lora_request["adapter_model_variant"],
                            "adapter_song_model": track_lora_request["adapter_song_model"],
                            "lora_trigger_applied": track_lora_trigger_applied,
                        }
                    )
                    quality_profile = _default_quality_profile_for_payload({**request_payload, **track}, "text2music")
                    model_defaults = quality_profile_model_settings(track_model, quality_profile)
                    request_key_scale = request_payload.get("key_scale") or request_payload.get("keyscale") or request_payload.get("key")
                    effective_key_scale = (
                        track.get("key_scale")
                        if parse_bool(track.get("input_contract_key_scale_locked"), False) and track.get("key_scale")
                        else request_key_scale or track.get("key_scale") or DEFAULT_KEY_SCALE
                    )
                    effective_audio_format = request_payload.get("audio_format") or track.get("audio_format") or model_defaults["audio_format"]
                    generation_payload = {
                        "task_type": "text2music",
                        "ui_mode": "album",
                        "quality_profile": quality_profile,
                        "artist_name": track_artist,
                        "title": track_title,
                        "description": track.get("description", ""),
                        "producer_credit": producer_credit,
                        "caption": track.get("tags") or track.get("caption") or "",
                        "tag_list": track.get("tag_list", []),
                        "lyrics": track.get("lyrics", ""),
                        "style_profile": track.get("style_profile") or request_payload.get("style_profile") or "",
                        "required_phrases": track.get("required_phrases", []),
                        "style": track.get("style", ""),
                        "vibe": track.get("vibe", ""),
                        "narrative": track.get("narrative", ""),
                        "lyric_density": track.get("lyric_density") or album_options.get("lyric_density") or "dense",
                        "structure_preset": track.get("structure_preset") or album_options.get("structure_preset") or "auto",
                        "duration": track.get("duration") or track_duration,
                        "bpm": track.get("bpm") or request_payload.get("bpm") or DEFAULT_BPM,
                        "key_scale": effective_key_scale,
                        "time_signature": track.get("time_signature") or request_payload.get("time_signature") or "4",
                        "vocal_language": track.get("language") or request_payload.get("vocal_language") or language,
                        "batch_size": variants,
                        "seed": str(track.get("seed") or request_payload.get("seed") or request_payload.get("seeds") or "-1"),
                        "song_model": track_model,
                        "ace_lm_model": track_lm_model,
                        "vocal_clarity_recovery": vocal_clarity_recovery,
                        "global_caption": album_global_caption,
                        "inference_steps": clamp_int(
                            model_render_settings.get("inference_steps", track.get("inference_steps", request_payload.get("inference_steps", model_item.get("default_steps")))),
                            int(model_item.get("default_steps") or _quality_default_steps(track_model, quality_profile)),
                            1,
                            200,
                        ),
                        "guidance_scale": clamp_float(
                            model_render_settings.get("guidance_scale", track.get("guidance_scale", request_payload.get("guidance_scale", model_item.get("default_guidance_scale")))),
                            float(model_item.get("default_guidance_scale") or model_defaults["guidance_scale"]),
                            1.0,
                            15.0,
                        ),
                        "shift": clamp_float(
                            model_render_settings.get("shift", track.get("shift", request_payload.get("shift", model_item.get("default_shift")))),
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
                        "audio_format": str(effective_audio_format),
                        "mp3_bitrate": str(track.get("mp3_bitrate") or request_payload.get("mp3_bitrate") or "128k"),
                        "mp3_sample_rate": track.get("mp3_sample_rate") or request_payload.get("mp3_sample_rate") or 48000,
                        "auto_score": parse_bool(request_payload.get("auto_score"), False),
                        "auto_lrc": parse_bool(request_payload.get("auto_lrc"), False),
                        "return_audio_codes": parse_bool(request_payload.get("return_audio_codes"), False),
                        "save_to_library": parse_bool(track.get("save_to_library", request_payload.get("save_to_library")), True),
                        "allow_supplied_lyrics_lm": bool(track_has_vocal_lyrics and track_lm_enabled),
                        "lm_backend": _normalize_lm_backend(track.get("lm_backend") or request_payload.get("lm_backend") or ACE_LM_BACKEND_DEFAULT),
                        "audio_backend": _normalize_audio_backend(track.get("audio_backend") or request_payload.get("audio_backend"), track.get("use_mlx_dit", request_payload.get("use_mlx_dit"))),
                        "use_mlx_dit": _normalize_audio_backend(track.get("audio_backend") or request_payload.get("audio_backend"), track.get("use_mlx_dit", request_payload.get("use_mlx_dit"))) == "mlx",
                        "thinking": _album_lm_switch("thinking", DOCS_BEST_LM_DEFAULTS["thinking"] if track_lm_enabled else False),
                        "sample_mode": False,
                        "sample_query": "",
                        "use_format": _album_lm_switch("use_format", DOCS_BEST_LM_DEFAULTS["use_format"] if track_lm_enabled else False),
                        "lm_temperature": clamp_float(track.get("lm_temperature", request_payload.get("lm_temperature")), 0.7 if vocal_clarity_recovery and track_lm_enabled else (DOCS_BEST_LM_DEFAULTS["lm_temperature"] if track_lm_enabled else 0.85), 0.0, 2.0),
                        "lm_cfg_scale": clamp_float(track.get("lm_cfg_scale", request_payload.get("lm_cfg_scale")), DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"] if track_lm_enabled else 2.0, 0.0, 10.0),
                        "lm_top_k": clamp_int(track.get("lm_top_k", request_payload.get("lm_top_k")), DOCS_BEST_LM_DEFAULTS["lm_top_k"] if track_lm_enabled else 0, 0, 200),
                        "lm_top_p": clamp_float(track.get("lm_top_p", request_payload.get("lm_top_p")), DOCS_BEST_LM_DEFAULTS["lm_top_p"] if track_lm_enabled else 0.9, 0.0, 1.0),
                        "lm_repetition_penalty": clamp_float(track.get("lm_repetition_penalty", request_payload.get("lm_repetition_penalty")), 1.0, 0.1, 4.0),
                        "use_cot_metas": False,
                        "use_cot_caption": False,
                        "use_cot_lyrics": False,
                        "use_cot_language": False,
                        "use_constrained_decoding": parse_bool(track.get("use_constrained_decoding", request_payload.get("use_constrained_decoding")), DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"]),
                        "audio_code_string": "",
                        "src_audio_id": "",
                        "src_result_id": "",
                        "reference_audio_id": "",
                        "reference_result_id": "",
                        "use_lora": track_lora_request["use_lora"],
                        "lora_adapter_path": track_lora_request["lora_adapter_path"],
                        "lora_adapter_name": track_lora_request["lora_adapter_name"],
                        "use_lora_trigger": track_lora_request["use_lora_trigger"],
                        "lora_trigger_tag": track_lora_request["lora_trigger_tag"],
                        "lora_trigger_source": track_lora_request.get("lora_trigger_source", ""),
                        "lora_trigger_aliases": track_lora_request.get("lora_trigger_aliases", []),
                        "lora_trigger_candidates": track_lora_request.get("lora_trigger_candidates", []),
                        "lora_scale": track_lora_request["lora_scale"],
                        "adapter_model_variant": track_lora_request["adapter_model_variant"],
                        "adapter_song_model": track_lora_request["adapter_song_model"],
                        "lora_trigger_applied": track_lora_trigger_applied,
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
                            "lora_adapter_name": track_lora_request["lora_adapter_name"],
                            "lora_scale": track_lora_request["lora_scale"],
                            "lora_trigger_tag": track_lora_request["lora_trigger_tag"],
                            "lora_trigger_applied": track_lora_trigger_applied,
                            "audio_backend": _normalize_audio_backend(track.get("audio_backend") or request_payload.get("audio_backend"), track.get("use_mlx_dit", request_payload.get("use_mlx_dit"))),
                            "album_debug_dir": str(album_debug.root),
                            "payload_gate_version": ALBUM_PAYLOAD_GATE_VERSION,
                        },
                    }
                    album_debug.append_jsonl(
                        "05_generation_payloads.jsonl",
                        {
                            "phase": "pre_gate",
                            "album_id": album_id,
                            "album_model": track_model,
                            "track_number": track.get("track_number", i + 1),
                            "title": track_title,
                            "payload": generation_payload,
                        },
                    )
                    if direct_agent_payload:
                        gate = _validate_direct_album_agent_payload(generation_payload)
                    else:
                        generation_payload["lyrics"] = strip_ace_step_lyrics_leakage(generation_payload.get("lyrics"))
                        gate = evaluate_album_payload_quality(
                            generation_payload,
                            options={
                                **album_options,
                                "track_duration": generation_payload.get("duration") or track_duration,
                            },
                            repair=True,
                        )
                        generation_payload = gate["repaired_payload"]
                    for public_field in [
                        "caption",
                        "tags",
                        "tag_list",
                        "lyrics",
                        "required_phrases",
                        "description",
                        "style",
                        "vibe",
                        "narrative",
                        "duration",
                        "bpm",
                        "key_scale",
                        "time_signature",
                        "vocal_language",
                        "lyrics_quality",
                        "use_lora",
                        "lora_adapter_name",
                        "lora_scale",
                        "lora_trigger_tag",
                        "lora_trigger_applied",
                        "adapter_song_model",
                    ]:
                        if public_field in generation_payload:
                            track[public_field] = generation_payload[public_field]
                    track["payload_quality_gate"] = {key: value for key, value in gate.items() if key != "repaired_payload"}
                    track["payload_gate_status"] = gate.get("status")
                    track["payload_gate_passed"] = bool(gate.get("gate_passed"))
                    track["payload_gate_blocking_issues"] = gate.get("blocking_issues") or []
                    track["tag_coverage"] = gate.get("tag_coverage")
                    track["caption_integrity"] = gate.get("caption_integrity")
                    track["lyric_duration_fit"] = gate.get("lyric_duration_fit")
                    track["lyrics_quality"] = generation_payload.get("lyrics_quality") or gate.get("lyrics_quality") or track.get("lyrics_quality") or {}
                    track["repair_actions"] = [] if direct_agent_payload else (gate.get("repair_actions") or [])
                    if model_index == 1:
                        for public_field in [
                            "caption",
                            "tags",
                            "tag_list",
                            "lyrics",
                            "required_phrases",
                            "payload_quality_gate",
                            "payload_gate_status",
                            "payload_gate_passed",
                            "payload_gate_blocking_issues",
                            "tag_coverage",
                            "caption_integrity",
                            "lyric_duration_fit",
                            "lyrics_quality",
                            "repair_actions",
                            "use_lora",
                            "lora_adapter_name",
                            "lora_scale",
                            "lora_trigger_tag",
                            "lora_trigger_applied",
                            "adapter_song_model",
                        ]:
                            base_track[public_field] = _jsonable(track.get(public_field))
                    album_debug.append_jsonl(
                        "06_quality_audit.jsonl",
                        {
                            "album_id": album_id,
                            "album_model": track_model,
                            "track_number": track.get("track_number", i + 1),
                            "title": track_title,
                            "gate": {key: value for key, value in gate.items() if key != "repaired_payload"},
                        },
                    )
                    if not gate.get("gate_passed"):
                        blocking_issues = gate.get("blocking_issues") or [
                            item for item in (gate.get("issues") or []) if item.get("severity") == "fail"
                        ]
                        issue_preview = "; ".join(
                            f"{item.get('id')}: {item.get('detail')}"
                            for item in blocking_issues[:6]
                        )
                        rejected_payload_record = {
                            "album_id": album_id,
                            "album_model": track_model,
                            "track_number": track.get("track_number", i + 1),
                            "title": track_title,
                            "issues": blocking_issues,
                            "payload": generation_payload,
                        }
                        album_debug.append_jsonl("06_rejected_payloads.jsonl", rejected_payload_record)
                        print(
                            "[album_payload_gate][REJECTED]\n"
                            + _ace_payload_debug_block("rejected_gate_issues_json", blocking_issues)
                            + "\n"
                            + _ace_payload_debug_block("rejected_payload_json", generation_payload),
                            flush=True,
                        )
                        if render_from_existing_tracks:
                            track["payload_gate_non_blocking"] = True
                            track["payload_gate_status"] = gate.get("status") or "needs_review"
                            track["payload_gate_passed"] = False
                            track["payload_gate_blocking_issues"] = blocking_issues
                            if model_index == 1:
                                base_track["payload_gate_non_blocking"] = True
                                base_track["payload_gate_status"] = track["payload_gate_status"]
                                base_track["payload_gate_passed"] = False
                                base_track["payload_gate_blocking_issues"] = _jsonable(blocking_issues)
                            logs.append(
                                "    Payload gate warning on UI-approved track "
                                f"{track_title}: {issue_preview or gate.get('status')}. Continuing render without another agent loop."
                            )
                            _album_job_log(
                                album_job_id,
                                f"Payload gate warning on UI-approved track {i + 1}: {track_title}; continuing render.",
                                current_track=f"{i + 1}/{len(tracks)} {track_title}",
                                stage="render_existing_tracks",
                                current_task="Render existing Album Wizard tracks",
                                warnings=[issue_preview or str(gate.get("status") or "payload_gate_warning")],
                            )
                        else:
                            raise ValueError(f"AlbumPayloadQualityGate failed: {issue_preview or gate.get('status')}")
                    if gate.get("status") == "auto_repair" and not direct_agent_payload:
                        logs.append(
                            f"    Payload gate auto-repaired {track_title}: "
                            f"{', '.join(str(item) for item in gate.get('repair_actions') or [])}"
                        )
                        _album_job_log(
                            album_job_id,
                            f"Payload gate auto-repaired {album_id} track {i + 1}: {track_title}",
                            album_debug_dir=str(album_debug.root),
                        )
                    album_debug.append_jsonl(
                        "05_generation_payloads.jsonl",
                        {
                            "phase": "post_gate",
                            "album_id": album_id,
                            "album_model": track_model,
                            "track_number": track.get("track_number", i + 1),
                            "title": track_title,
                            "payload": generation_payload,
                        },
                    )
                    payload_validation = _validate_generation_payload(generation_payload)
                    track["payload_validation"] = payload_validation
                    track["payload_warnings"] = payload_validation.get("payload_warnings", [])
                    if not payload_validation.get("valid"):
                        raise ValueError(f"Invalid track payload: {payload_validation.get('field_errors')}")
                    generation_result = _run_advanced_generation(generation_payload)
                    if not generation_result.get("success"):
                        raise RuntimeError(generation_result.get("error") or "Track generation failed")
                    album_debug.append_jsonl(
                        "07_generation_results.jsonl",
                        {
                            "status": "success",
                            "album_id": album_id,
                            "album_model": track_model,
                            "track_number": track.get("track_number", i + 1),
                            "title": track_title,
                            "result_id": generation_result.get("result_id"),
                            "active_song_model": generation_result.get("active_song_model"),
                            "audios": generation_result.get("audios", []),
                            "payload_gate_status": track.get("payload_gate_status"),
                        },
                    )

                    track["result_id"] = generation_result.get("result_id")
                    track["active_song_model"] = generation_result.get("active_song_model")
                    track["audios"] = generation_result.get("audios", [])
                    actual_render_model = str(
                        generation_result.get("active_song_model")
                        or generation_result.get("song_model")
                        or track_model
                    )
                    rescue_model = str(
                        (generation_result.get("params") or {}).get("vocal_intelligibility_rescue_model") or ""
                    ).strip()
                    if actual_render_model and actual_render_model != track_model:
                        track["requested_album_model"] = track_model
                        track["render_model"] = actual_render_model
                        track["album_model_rescued"] = True
                    if rescue_model:
                        track["vocal_intelligibility_rescue_model"] = rescue_model
                    track["payload_warnings"] = generation_result.get("payload_warnings", [])
                    track["runner"] = generation_result.get("runner")
                    track["generation_params"] = generation_result.get("params", {})
                    rendered_params = track["generation_params"] if isinstance(track.get("generation_params"), dict) else {}
                    track["lora_trigger_applied"] = bool(rendered_params.get("lora_trigger_applied", track.get("lora_trigger_applied")))
                    track["lora_trigger_conditioning_audit"] = rendered_params.get("lora_trigger_conditioning_audit") or track.get("lora_trigger_conditioning_audit") or {}
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
                        audio["album_model_requested"] = track_model
                        audio["album_model"] = actual_render_model
                        audio["album_model_label"] = model_label(actual_render_model)
                        if rescue_model:
                            audio["vocal_intelligibility_rescue_model"] = rescue_model
                        audio["payload_gate_status"] = track.get("payload_gate_status")
                        audio["payload_gate_passed"] = track.get("payload_gate_passed")
                        audio["lora_adapter_name"] = track.get("lora_adapter_name")
                        audio["lora_scale"] = track.get("lora_scale")
                        audio["lora_trigger_tag"] = track.get("lora_trigger_tag")
                        audio["lora_trigger_applied"] = track.get("lora_trigger_applied")
                        if audio.get("song_id"):
                            _merge_song_album_metadata(
                                audio["song_id"],
                                {
                                    "artist_name": track_artist,
                                    "album_concept": concept,
                                    "album_family_id": album_family_id,
                                    "album_id": album_id,
                                    "album_model_requested": track_model,
                                    "album_model": actual_render_model,
                                    "album_model_label": model_label(actual_render_model),
                                    "vocal_intelligibility_rescue_model": rescue_model,
                                    "track_number": track.get("track_number", i + 1),
                                    "track_variant": audio_index + 1,
                                    "album_toolkit_report": result.get("toolkit_report", {}),
                                    "tool_report": track.get("tool_report", {}),
                                    "production_team": track.get("production_team", {}),
                                    "final_model_policy": track.get("final_model_policy", {}),
                                    "tag_list": track.get("tag_list", []),
                                    "lyrics_quality": track.get("lyrics_quality", {}),
                                    "lora_adapter_name": track.get("lora_adapter_name"),
                                    "lora_scale": track.get("lora_scale"),
                                    "lora_trigger_tag": track.get("lora_trigger_tag"),
                                    "lora_trigger_applied": track.get("lora_trigger_applied"),
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
                    album_debug.append_jsonl(
                        "07_generation_results.jsonl",
                        {
                            "status": "failed",
                            "album_id": album_id,
                            "album_model": track_model,
                            "track_number": track.get("track_number", i + 1),
                            "title": track_title,
                            "error": str(track_exc),
                            "payload_gate_status": track.get("payload_gate_status"),
                            "payload_quality_gate": track.get("payload_quality_gate"),
                        },
                    )
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
                publish_album_progress(
                    current_album_id=album_id,
                    current_album_model=track_model,
                    current_album_model_label=str(model_item.get("label") or track_model),
                    current_album_tracks=album_tracks,
                    current_album_audios=album_audios,
                )

            album_failed_tracks = [
                {
                    "track_number": item.get("track_number"),
                    "title": item.get("title"),
                    "planning_status": item.get("planning_status"),
                    "error": item.get("error") or item.get("planning_error") or "",
                }
                for item in album_tracks
                if not item.get("generated")
            ]
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
                    "failed_count": len(album_failed_tracks),
                    "failed_tracks": album_failed_tracks,
                    "tracks": album_tracks,
                    "audios": album_audios,
                    "toolkit_report": result.get("toolkit_report", {}),
                    "album_options": album_options,
                    "album_writer_mode": album_options.get("album_writer_mode"),
                    "album_debug_dir": str(album_debug.root),
                    "album_payload_gate_version": ALBUM_PAYLOAD_GATE_VERSION,
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
                    "failed_count": len(album_failed_tracks),
                    "failed_tracks": album_failed_tracks,
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
        failed_tracks_by_number: dict[str, dict[str, Any]] = {}
        for album in model_albums:
            for item in album.get("failed_tracks") or []:
                key = str(item.get("track_number") or item.get("title") or len(failed_tracks_by_number) + 1)
                failed_tracks_by_number.setdefault(key, dict(item))
        failed_tracks = list(failed_tracks_by_number.values())
        failed_summary = ", ".join(
            f"{item.get('track_number')} {item.get('title')}: {item.get('error') or item.get('planning_status') or 'failed'}"
            for item in failed_tracks[:8]
        )
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
                "failed_count": len(failed_tracks),
                "failed_tracks": failed_tracks,
                "planning_failed_count": int(result.get("planning_failed_count") or 0),
                "model_albums": model_albums,
                "album_model_portfolio": album_models,
                "toolkit_report": result.get("toolkit_report", {}),
                "album_options": album_options,
                "album_writer_mode": album_options.get("album_writer_mode"),
                "album_debug_dir": str(album_debug.root),
                "album_payload_gate_version": ALBUM_PAYLOAD_GATE_VERSION,
                "download_url": f"/api/album-families/{album_family_id}/download",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        album_art = _maybe_auto_generate_album_art(album_family_id, concept, album_options, logs)
        if album_art:
            family_manifest["album_art"] = _jsonable(album_art)
            for model_album in model_albums:
                model_album["album_art"] = _jsonable(album_art)
        logs.append("---")
        logs.append(f"Album family {album_status}: {generated_count}/{expected_count} track/model renders generated.")
        if failed_summary:
            logs.append(f"Failed tracks: {failed_summary}.")
        _album_job_log(
            album_job_id,
            (
                f"Album family {album_status}: {generated_count}/{expected_count} track/model renders generated."
                + (f" Failed tracks: {failed_summary}." if failed_summary else "")
            ),
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
            "failed_count": len(failed_tracks),
            "failed_tracks": failed_tracks,
            "planning_failed_count": int(result.get("planning_failed_count") or 0),
            "planning_failures": result.get("planning_failures") or [],
            "final_song_model": "all_models_album" if strategy == "all_models_album" else (album_models[0]["model"] if album_models else ALBUM_FINAL_MODEL),
            "family_download_url": f"/api/album-families/{album_family_id}/download",
            "manifest": family_manifest,
            "album_art": _jsonable(album_art),
            "toolkit": result.get("toolkit", _songwriting_toolkit_payload()),
            "toolkit_report": result.get("toolkit_report", {}),
            "planner_model": ollama_model,
            "planner_provider": planner_lm_provider,
            "embedding_model": embedding_model,
            "embedding_provider": embedding_lm_provider,
            "planning_engine": str(result.get("planning_engine") or ""),
            "album_writer_mode": str(result.get("album_writer_mode") or album_options.get("album_writer_mode") or "per_track_writer_loop"),
            "custom_agents_used": bool(result.get("custom_agents_used")),
            "crewai_used": bool(result.get("crewai_used")),
            "toolbelt_fallback": bool(result.get("toolbelt_fallback")),
            "crewai_error": str(result.get("crewai_error") or ""),
            "agent_error": str(result.get("agent_error") or ""),
            "agent_debug_dir": str(result.get("agent_debug_dir") or album_debug.root),
            "agent_repair_count": int(result.get("agent_repair_count") or 0),
            "agent_rounds": result.get("agent_rounds") or [],
            "album_debug_dir": str(album_debug.root),
            "album_payload_gate_version": ALBUM_PAYLOAD_GATE_VERSION,
            "logs": logs,
            "success": album_success,
            "error": "" if album_success else (
                f"Album incomplete: {generated_count}/{expected_count} track/model renders generated."
                + (f" Failed tracks: {failed_summary}." if failed_summary else "")
            ),
        })
    except ModelDownloadStarted as exc:
        print(f"[generate_album DOWNLOAD] {exc.message}")
        logs.append(exc.message)
        return json.dumps(_download_started_payload(exc.model_name, exc.job, logs))
    except Exception as exc:
        print(f"[generate_album ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        logs.append(f"ERROR: {exc}")
        debug_dir = ""
        if "album_debug" in locals():
            debug_dir = str(album_debug.root)
            try:
                album_debug.write_json("99_exception.json", {"error": str(exc), "traceback": traceback.format_exc(), "logs": logs})
            except Exception:
                pass
        return json.dumps({"tracks": [], "logs": logs, "success": False, "error": str(exc), "album_debug_dir": debug_dir})


@app.api(name="generate_advanced", concurrency_limit=1, time_limit=ACEJAM_GENERATE_ADVANCED_TIME_LIMIT_SECONDS)
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
    vocal_health = vocal_dataset_health(items)
    lyrics_ready = int(vocal_health.get("real_lyrics_count") or 0)
    durations = [clamp_float(item.get("duration"), 0.0, 0.0, 600.0) for item in items]
    durations = [value for value in durations if value > 0]
    total_duration = round(sum(durations), 2)
    avg_duration = round(total_duration / len(durations), 2) if durations else 0.0
    languages = sorted({str(item.get("language") or "unknown") for item in items})
    genre_values = sorted(
        {
            term
            for item in items
            for term in _training_split_genre_terms(item.get("genre"), item.get("style_profile"), item.get("genre_profile"))
        }
    )
    genre_sources: dict[str, int] = {}
    for item in items:
        source = str(item.get("genre_label_source") or "").strip() or "unknown"
        genre_sources[source] = genre_sources.get(source, 0) + 1
    score = 0
    checks = []

    def add(check_id: str, ok: bool, detail: str, points: int) -> None:
        nonlocal score
        if ok:
            score += points
        checks.append({"id": check_id, "status": "pass" if ok else "warn", "detail": detail, "points": points})

    add("sample_count", count >= 8, f"{count} audio sample(s)", 25)
    add("caption_labels", count > 0 and labeled == count, f"{labeled}/{count} captioned", 25)
    add("lyrics_labels", count > 0 and lyrics_ready >= max(1, int(count * 0.95)), f"{lyrics_ready}/{count} real vocal lyrics", 15)
    add("duration_total", total_duration >= 300, f"{total_duration}s total", 20)
    add("duration_shape", not durations or 5 <= avg_duration <= 300, f"{avg_duration}s average", 10)
    add("language_metadata", not vocal_health.get("blocking") and bool(languages), ", ".join(languages[:6]) or "unknown", 5)
    checks.append(
        {
            "id": "genre_labels",
            "status": "pass" if genre_values else "warn",
            "detail": ", ".join(genre_values[:8]) if genre_values else "no genre/style labels yet",
            "points": 0,
        }
    )
    status = "needs_work" if vocal_health.get("blocking") else ("ready" if score >= 85 else "usable" if score >= 65 else "needs_work")
    health = {
        "version": PRO_QUALITY_AUDIT_VERSION,
        "status": status,
        "score": min(100, score),
        "sample_count": count,
        "labeled_count": labeled,
        "lyrics_ready_count": lyrics_ready,
        "total_duration_seconds": total_duration,
        "average_duration_seconds": avg_duration,
        "languages": languages,
        "genres": genre_values,
        "genre_labeled_count": sum(1 for item in items if str(item.get("genre") or item.get("style_profile") or "").strip()),
        "ai_genre_labeled_count": genre_sources.get("ai_local_llm", 0),
        "metadata_genre_labeled_count": genre_sources.get("metadata", 0) + genre_sources.get("id3_metadata", 0),
        "musicbrainz_genre_labeled_count": genre_sources.get("musicbrainz", 0),
        "genre_label_sources": genre_sources,
        "checks": checks,
        "audition_plan": {
            "adapter_scales": [0.3, 0.6, 0.8, 1.0],
            "compare_to_baseline": True,
            "recommended_seed_count": 2,
        },
    }
    health.update(vocal_health)
    return health


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


def _git_stdout(args: list[str], *, cwd: Path, timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _git_run(args: list[str], *, cwd: Path, timeout: float = 30.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"returncode": 1, "stdout": "", "stderr": str(exc), "args": args}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "args": args,
    }


def _official_source_status() -> dict[str, Any]:
    vendor_dir = OFFICIAL_ACE_STEP_DIR
    local_commit = _git_stdout(["rev-parse", "HEAD"], cwd=vendor_dir)
    local_branch = _git_stdout(["branch", "--show-current"], cwd=vendor_dir)
    porcelain = _git_stdout(["status", "--porcelain"], cwd=vendor_dir)
    dirty_files = [line.strip() for line in porcelain.splitlines() if line.strip()]
    dirty = bool(dirty_files)
    remote_url = _git_stdout(["config", "--get", "remote.origin.url"], cwd=vendor_dir)
    remote_main_head = ""
    if remote_url:
        remote_line = _git_stdout(["ls-remote", remote_url, "refs/heads/main"], cwd=vendor_dir, timeout=8.0)
        remote_main_head = remote_line.split()[0] if remote_line else ""
    behind_main = bool(local_commit and remote_main_head and local_commit != remote_main_head)
    return {
        "vendor_dir": str(vendor_dir),
        "local_commit": local_commit,
        "local_branch": local_branch,
        "dirty": dirty,
        "dirty_files": dirty_files,
        "remote_url": remote_url,
        "remote_main_head": remote_main_head,
        "behind_main": behind_main,
        "status": "behind" if behind_main else ("current" if remote_main_head else "unknown"),
        "sync_confirm": ACE_STEP_VENDOR_SYNC_CONFIRM,
        "sync_mode": "patch-preserving manual endpoint",
        "update_note": "Status only by default. Vendor sync requires explicit confirm and preserves dirty diffs before touching source.",
    }


def _official_vendor_sync(body: dict[str, Any]) -> dict[str, Any]:
    confirm = str((body or {}).get("confirm") or "")
    apply_update = parse_bool((body or {}).get("apply"), False)
    if confirm != ACE_STEP_VENDOR_SYNC_CONFIRM:
        raise HTTPException(status_code=400, detail=f"confirm must be {ACE_STEP_VENDOR_SYNC_CONFIRM}")
    vendor_dir = OFFICIAL_ACE_STEP_DIR
    if not (vendor_dir / ".git").exists():
        raise HTTPException(status_code=400, detail="Vendored ACE-Step folder is not a git checkout")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sync_dir = DATA_DIR / "debug" / "vendor_sync" / stamp
    sync_dir.mkdir(parents=True, exist_ok=True)
    status_before = _official_source_status()
    patch_text = _git_stdout(["diff", "--binary"], cwd=vendor_dir, timeout=30.0)
    staged_patch_text = _git_stdout(["diff", "--cached", "--binary"], cwd=vendor_dir, timeout=30.0)
    (sync_dir / "status-before.json").write_text(json.dumps(_jsonable(status_before), indent=2), encoding="utf-8")
    (sync_dir / "vendor-working-tree.patch").write_text(patch_text, encoding="utf-8")
    (sync_dir / "vendor-staged.patch").write_text(staged_patch_text, encoding="utf-8")

    steps: list[dict[str, Any]] = []
    fetch = _git_run(["fetch", "origin", "main"], cwd=vendor_dir, timeout=120.0)
    steps.append(fetch)
    if fetch["returncode"] != 0:
        return {
            "success": False,
            "applied": False,
            "patch_dir": str(sync_dir),
            "status_before": status_before,
            "steps": _jsonable(steps),
            "error": "git fetch origin main failed; local vendor source was not changed.",
        }
    if not apply_update:
        return {
            "success": True,
            "applied": False,
            "patch_dir": str(sync_dir),
            "status_before": status_before,
            "status_after": _official_source_status(),
            "steps": _jsonable(steps),
            "next_step": "Re-run with apply=true and the same confirm token to fast-forward vendor source and reapply local dirty changes.",
        }

    stash_created = False
    if status_before.get("dirty"):
        stash = _git_run(["stash", "push", "--include-untracked", "-m", f"AceJAM vendor sync preserve {stamp}"], cwd=vendor_dir, timeout=120.0)
        steps.append(stash)
        if stash["returncode"] != 0:
            return {
                "success": False,
                "applied": False,
                "patch_dir": str(sync_dir),
                "status_before": status_before,
                "steps": _jsonable(steps),
                "error": "Could not stash dirty vendor changes; vendor source was not updated.",
            }
        stash_created = "No local changes" not in f"{stash.get('stdout','')} {stash.get('stderr','')}"

    merge = _git_run(["merge", "--ff-only", "origin/main"], cwd=vendor_dir, timeout=180.0)
    steps.append(merge)
    if merge["returncode"] != 0:
        if stash_created:
            steps.append(_git_run(["stash", "pop"], cwd=vendor_dir, timeout=120.0))
        return {
            "success": False,
            "applied": False,
            "patch_dir": str(sync_dir),
            "status_before": status_before,
            "status_after": _official_source_status(),
            "steps": _jsonable(steps),
            "error": "Fast-forward to official main failed; any stashed changes were restored if possible.",
        }

    if stash_created:
        pop = _git_run(["stash", "pop"], cwd=vendor_dir, timeout=120.0)
        steps.append(pop)
        if pop["returncode"] != 0:
            (sync_dir / "conflict-status.json").write_text(
                json.dumps(_jsonable(_official_source_status()), indent=2),
                encoding="utf-8",
            )
            return {
                "success": False,
                "applied": True,
                "patch_dir": str(sync_dir),
                "status_before": status_before,
                "status_after": _official_source_status(),
                "steps": _jsonable(steps),
                "error": "Official main was applied, but local dirty vendor changes conflicted during reapply. Resolve conflicts manually; saved patches are in patch_dir.",
            }

    status_after = _official_source_status()
    (sync_dir / "status-after.json").write_text(json.dumps(_jsonable(status_after), indent=2), encoding="utf-8")
    return {
        "success": True,
        "applied": True,
        "patch_dir": str(sync_dir),
        "status_before": status_before,
        "status_after": status_after,
        "steps": _jsonable(steps),
    }


def _official_parity_payload(request: Request | None = None) -> dict[str, Any]:
    manifest = official_manifest()
    installed_models = _installed_acestep_models()
    installed_lms = _installed_lm_models()
    recommended_actions: list[str] = []
    model_registry = manifest.get("model_registry") if isinstance(manifest.get("model_registry"), dict) else {}
    for name, meta in model_registry.items():
        meta.update(_official_model_runtime_status(name))
        if meta.get("render_usable"):
            meta["tasks"] = supported_tasks_for_model(name)
    for name, meta in manifest.get("dit_models", {}).items():
        meta.update(_model_runtime_status(name))
        meta["tasks"] = supported_tasks_for_model(name)
    if isinstance(manifest.get("core_bundle"), dict):
        manifest["core_bundle"].update(_official_model_runtime_status(OFFICIAL_CORE_MODEL_ID))
        manifest["core_bundle"]["component_status"] = {
            component: _checkpoint_status_reason(MODEL_CACHE_DIR / "checkpoints" / component)
            for component in OFFICIAL_MAIN_MODEL_COMPONENTS
        }
    for group in ["helper_models", "lora_models", "legacy_models"]:
        for name, meta in (manifest.get(group) or {}).items():
            if isinstance(meta, dict):
                meta.update(_official_model_runtime_status(name))
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
    source_status = _official_source_status()
    if schema_parity["generation_params"]["unsupported_by_vendor"]:
        recommended_actions.append("MLX Media will drop unsupported GenerationParams fields before calling the official runner.")
    if source_status.get("behind_main"):
        recommended_actions.append("Vendored ACE-Step source is behind official main; review/update manually before replacing local changes.")
    if source_status.get("dirty"):
        recommended_actions.append("Vendored ACE-Step has dirty local changes; vendor sync will save patches before any update.")
    if DOCS_BEST_DEFAULT_LM_MODEL not in installed_lms:
        recommended_actions.append(f"Install {DOCS_BEST_DEFAULT_LM_MODEL}; it is the Docs-best default for official LM controls.")
    missing_boot_models = [
        model_name
        for model_name in _boot_download_model_names()
        if not _is_model_installed(model_name)
    ]
    if missing_boot_models:
        recommended_actions.append(
            "Boot quality bundle is still downloading or missing: " + ", ".join(missing_boot_models) + "."
        )
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
        "boot_downloads": _jsonable(_boot_model_download_status),
        "source_status": source_status,
        "trainer": training_manager.status(),
        "stats": _job_stats(),
        "server_url": str(request.base_url).rstrip("/") if request is not None else "",
        "api_key_enabled": bool(os.environ.get("ACESTEP_API_KEY", "").strip()),
    }
    manifest["quality_policy"] = docs_best_quality_policy()
    manifest["source_status"] = source_status
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
                "id": task_id,
                "task_id": task_id,
                "kind": "generation",
                "status": 0,
                "state": "queued",
                "stage": "Queued",
                "progress": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "payload": {},
                "payload_summary": {},
                "result": None,
                "error": "",
                "logs": [],
                "errors": [],
            },
        )
        if "logs" in updates:
            old_logs = list(task.get("logs") or [])
            new_logs = updates.pop("logs")
            if isinstance(new_logs, list):
                task["logs"] = (old_logs + [str(item) for item in new_logs])[-500:]
            elif new_logs:
                task["logs"] = (old_logs + [str(new_logs)])[-500:]
        if "errors" in updates:
            old_errors = list(task.get("errors") or [])
            new_errors = updates.pop("errors")
            if isinstance(new_errors, list):
                task["errors"] = (old_errors + [str(item) for item in new_errors])[-100:]
            elif new_errors:
                task["errors"] = (old_errors + [str(new_errors)])[-100:]
        task.update(_jsonable(updates))
        task["updated_at"] = datetime.now(timezone.utc).isoformat()
        if len(_api_generation_tasks) > GENERATION_JOB_KEEP_LIMIT:
            removable = sorted(
                _api_generation_tasks.values(),
                key=lambda item: str(item.get("finished_at") or item.get("created_at") or ""),
            )
            for old in removable[: max(0, len(_api_generation_tasks) - GENERATION_JOB_KEEP_LIMIT)]:
                if old.get("state") not in {"queued", "running"}:
                    _api_generation_tasks.pop(str(old.get("task_id") or old.get("id")), None)
        return dict(task)


def _generation_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    try:
        display_payload = apply_audio_style_conditioning(payload)
    except Exception:
        display_payload = dict(payload)
    lyrics = str(display_payload.get("lyrics") or payload.get("lyrics") or "")
    caption = str(
        display_payload.get("caption")
        or payload.get("caption")
        or payload.get("song_description")
        or payload.get("simple_description")
        or ""
    )
    task_type = str(display_payload.get("task_type") or payload.get("task_type") or "text2music").strip() or "text2music"
    memory_policy = payload.get("memory_policy") if isinstance(payload.get("memory_policy"), dict) else {}
    actual_runner_batch_size = payload.get("actual_runner_batch_size")
    requested_take_count = payload.get("requested_take_count") or payload.get("batch_size")
    if not memory_policy or actual_runner_batch_size in (None, ""):
        try:
            plan = _official_generation_memory_plan(
                {
                    "batch_size": payload.get("batch_size") or requested_take_count,
                    "song_model": display_payload.get("song_model") or payload.get("song_model"),
                    "task_type": task_type,
                    "device": payload.get("device") or "auto",
                    "duration": display_payload.get("duration") or payload.get("duration") or payload.get("audio_duration"),
                    "inference_steps": display_payload.get("inference_steps") or payload.get("inference_steps") or payload.get("infer_step"),
                    "shift": display_payload.get("shift") or payload.get("shift"),
                    "use_lora": parse_bool(payload.get("use_lora"), False),
                    "lora_scale": payload.get("lora_scale"),
                }
            )
            if not memory_policy:
                memory_policy = plan
            actual_runner_batch_size = actual_runner_batch_size or plan.get("actual_runner_batch_size")
            requested_take_count = requested_take_count or plan.get("requested_take_count")
        except Exception:
            memory_policy = memory_policy or {}
    lora_trigger_tag = str(payload.get("lora_trigger_tag") or payload.get("lora_trigger") or "").strip()
    lora_trigger_source = str(payload.get("lora_trigger_source") or "").strip()
    lora_trigger_aliases = list(payload.get("lora_trigger_aliases") or [])
    lora_trigger_candidates = list(payload.get("lora_trigger_candidates") or [])
    lora_adapter_path = str(payload.get("lora_adapter_path") or payload.get("lora_path") or payload.get("lora_name_or_path") or "").strip()
    if lora_adapter_path and (not lora_trigger_tag or not lora_trigger_source):
        try:
            adapter_metadata = infer_adapter_model_metadata(Path(lora_adapter_path).expanduser())
        except Exception:
            adapter_metadata = {}
        if adapter_metadata:
            lora_trigger_tag = lora_trigger_tag or str(
                adapter_metadata.get("generation_trigger_tag")
                or adapter_metadata.get("trigger_tag")
                or ""
            ).strip()
            lora_trigger_source = lora_trigger_source or str(adapter_metadata.get("trigger_source") or "metadata").strip()
            lora_trigger_aliases = lora_trigger_aliases or list(adapter_metadata.get("trigger_aliases") or [])
            lora_trigger_candidates = lora_trigger_candidates or list(adapter_metadata.get("trigger_candidates") or [])
    lora_trigger_tag = safe_generation_trigger_tag(lora_trigger_tag)
    trigger_explicitly_disabled = (
        ("use_lora_trigger" in payload or "lora_use_trigger" in payload)
        and not parse_bool(payload.get("use_lora_trigger", payload.get("lora_use_trigger")), True)
    )
    use_lora = parse_bool(payload.get("use_lora"), bool(lora_adapter_path))
    use_lora_trigger = bool(use_lora and lora_trigger_tag and not trigger_explicitly_disabled)
    if not use_lora_trigger:
        lora_trigger_source = "disabled" if use_lora and lora_trigger_tag and trigger_explicitly_disabled else ""
    lora_trigger_audit = payload.get("lora_trigger_conditioning_audit") if isinstance(payload.get("lora_trigger_conditioning_audit"), dict) else {}
    if use_lora and lora_trigger_tag and not lora_trigger_audit:
        lora_trigger_audit = {
            "status": "planned" if use_lora_trigger else "disabled",
            "caption_only": True,
            "trigger_tag": lora_trigger_tag,
            "trigger_source": lora_trigger_source,
            "trigger_aliases": lora_trigger_aliases,
            "trigger_candidates": lora_trigger_candidates,
            "applied": False,
            "already_present": False,
            "in_lyrics": bool(_caption_contains_lora_trigger(lyrics, lora_trigger_tag)),
        }
    return _jsonable(
        {
            "task_type": task_type,
            "title": str(display_payload.get("title") or payload.get("title") or "").strip(),
            "artist_name": str(display_payload.get("artist_name") or payload.get("artist_name") or "").strip(),
            "caption": caption.strip(),
            "tags": str(display_payload.get("tags") or payload.get("tags") or "").strip(),
            "duration": display_payload.get("duration") or payload.get("duration") or payload.get("audio_duration"),
            "song_model": str(display_payload.get("song_model") or payload.get("song_model") or "").strip(),
            "quality_profile": str(display_payload.get("quality_profile") or payload.get("quality_profile") or "").strip(),
            "batch_size": payload.get("batch_size"),
            "requested_take_count": requested_take_count,
            "actual_runner_batch_size": actual_runner_batch_size,
            "memory_policy": memory_policy,
            "style_profile": str(display_payload.get("style_profile") or "").strip(),
            "style_caption_tags": str(display_payload.get("style_caption_tags") or "").strip(),
            "style_lyric_tags_applied": list(display_payload.get("style_lyric_tags_applied") or []),
            "style_conditioning_audit": display_payload.get("style_conditioning_audit") if isinstance(display_payload.get("style_conditioning_audit"), dict) else {},
            "seed": payload.get("seed"),
            "bpm": display_payload.get("bpm") or payload.get("bpm"),
            "key_scale": str(display_payload.get("key_scale") or payload.get("key_scale") or "").strip(),
            "time_signature": str(display_payload.get("time_signature") or payload.get("time_signature") or "").strip(),
            "vocal_language": str(display_payload.get("vocal_language") or payload.get("vocal_language") or "").strip(),
            "instrumental": parse_bool(payload.get("instrumental"), lyrics.strip() == "[Instrumental]"),
            "lyrics_word_count": len(re.findall(r"\b\w+\b", lyrics)) if lyrics and lyrics != "[Instrumental]" else 0,
            "has_lyrics": bool(lyrics.strip() and lyrics.strip() != "[Instrumental]"),
            "has_source_audio": bool(payload.get("src_audio_id") or payload.get("src_result_id") or payload.get("audio_code_string")),
            "has_reference_audio": bool(payload.get("reference_audio_id") or payload.get("reference_result_id")),
            "with_lora": use_lora,
            "lora_adapter_name": str(payload.get("lora_adapter_name") or "").strip(),
            "lora_adapter_path": lora_adapter_path,
            "use_lora_trigger": use_lora_trigger,
            "lora_trigger_tag": lora_trigger_tag if use_lora_trigger else "",
            "lora_trigger_source": lora_trigger_source,
            "lora_trigger_aliases": lora_trigger_aliases,
            "lora_trigger_candidates": lora_trigger_candidates,
            "lora_trigger_conditioning_audit": lora_trigger_audit,
            "lora_scale": payload.get("lora_scale"),
            "adapter_model_variant": str(payload.get("adapter_model_variant") or "").strip(),
            "auto_song_art": parse_bool(payload.get("auto_song_art"), False),
            "auto_album_art": parse_bool(payload.get("auto_album_art"), False),
            "auto_video_clip": parse_bool(payload.get("auto_video_clip"), False),
        }
    )


def _generation_result_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    audios = result.get("audios") or []
    first_audio = audios[0] if isinstance(audios, list) and audios and isinstance(audios[0], dict) else {}
    first_lora = first_audio.get("lora_adapter") if isinstance(first_audio.get("lora_adapter"), dict) else {}
    return _jsonable(
        {
            "success": bool(result.get("success", True)),
            "result_id": result.get("result_id") or first_audio.get("result_id"),
            "audio_url": result.get("audio_url") or first_audio.get("audio_url") or first_audio.get("download_url"),
            "title": result.get("title"),
            "artist_name": result.get("artist_name"),
            "caption": result.get("caption"),
            "lyrics": result.get("lyrics"),
            "duration": result.get("duration"),
            "bpm": result.get("bpm"),
            "key_scale": result.get("key_scale"),
            "payload_warnings": list(result.get("payload_warnings") or []),
            "vocal_intelligibility_gate": result.get("vocal_intelligibility_gate"),
            "vocal_intelligibility_history": result.get("vocal_intelligibility_history"),
            "vocal_preflight": result.get("vocal_preflight"),
            "lora_preflight": result.get("lora_preflight"),
            "requested_song_model": result.get("requested_song_model"),
            "actual_song_model": result.get("actual_song_model"),
            "primary_attempt_id": result.get("primary_attempt_id"),
            "attempt_role": result.get("attempt_role"),
            "with_lora": result.get("with_lora") if result.get("with_lora") is not None else first_lora.get("use_lora"),
            "lora_scale": result.get("lora_scale") if result.get("lora_scale") is not None else first_lora.get("scale"),
            "lora_adapter_name": result.get("lora_adapter_name") or first_lora.get("name"),
            "lora_adapter_path": result.get("lora_adapter_path") or first_lora.get("path"),
            "lora_quality_status": result.get("lora_quality_status"),
            "use_lora_trigger": result.get("use_lora_trigger"),
            "lora_trigger_tag": result.get("lora_trigger_tag"),
            "lora_trigger_source": result.get("lora_trigger_source"),
            "lora_trigger_applied": result.get("lora_trigger_applied"),
            "lora_trigger_conditioning_audit": result.get("lora_trigger_conditioning_audit"),
            "style_profile": result.get("style_profile"),
            "style_caption_tags": result.get("style_caption_tags"),
            "style_lyric_tags_applied": result.get("style_lyric_tags_applied"),
            "style_conditioning_audit": result.get("style_conditioning_audit"),
            "vocal_gate_status": result.get("vocal_gate_status"),
            "transcript_preview": result.get("transcript_preview"),
            "failure_reason": result.get("failure_reason"),
            "diagnostic_attempts": result.get("diagnostic_attempts"),
            "requested_take_count": result.get("requested_take_count"),
            "completed_take_count": result.get("completed_take_count"),
            "actual_runner_batch_size": result.get("actual_runner_batch_size"),
            "memory_policy": result.get("memory_policy"),
            "mps_memory": result.get("mps_memory"),
            "recommended_take": result.get("recommended_take"),
            "needs_review": result.get("needs_review"),
            "error": result.get("error"),
            "audio_count": len(audios) if isinstance(audios, list) else 0,
        }
    )


def _generation_status_label(task: dict[str, Any]) -> str:
    state = str(task.get("state") or "queued").lower()
    if task.get("stage"):
        return str(task["stage"])
    if state == "succeeded":
        return "Complete"
    if state == "failed":
        return "Failed"
    if state == "running":
        return "Rendering"
    return "Queued"


def _generation_job_view(task: dict[str, Any]) -> dict[str, Any]:
    result = task.get("result") if isinstance(task.get("result"), dict) else None
    summary = dict(task.get("payload_summary") or _generation_payload_summary(dict(task.get("payload") or {})))
    warnings = []
    if isinstance(result, dict):
        warnings = [str(item) for item in result.get("payload_warnings") or [] if str(item)]
    error = str(task.get("error") or "")
    errors = [str(item) for item in task.get("errors") or [] if str(item)]
    if error and error not in errors:
        errors.append(error)
    return _jsonable(
        {
            "id": str(task.get("task_id") or task.get("id") or ""),
            "task_id": str(task.get("task_id") or task.get("id") or ""),
            "kind": "generation",
            "state": str(task.get("state") or "queued"),
            "status": _generation_status_label(task),
            "status_code": int(task.get("status") or 0),
            "stage": str(task.get("stage") or ""),
            "progress": clamp_int(task.get("progress"), 0, 0, 100),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "finished_at": task.get("finished_at"),
            "payload": task.get("payload") or {},
            "payload_summary": summary,
            "result": result,
            "result_summary": _generation_result_summary(result),
            "warnings": warnings,
            "automation": task.get("automation") or {},
            "logs": list(task.get("logs") or [])[-500:],
            "errors": errors[-100:],
            "error": error,
        }
    )


def _generation_job_snapshot(job_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    with _api_generation_tasks_lock:
        if job_id:
            task = dict(_api_generation_tasks.get(job_id) or {})
            return _generation_job_view(task) if task else {}
        tasks = [dict(task) for task in _api_generation_tasks.values()]
    return [
        _generation_job_view(task)
        for task in sorted(tasks, key=lambda item: str(item.get("created_at") or ""), reverse=True)
    ]


def _generation_job_log(job_id: str, line: str, **updates: Any) -> None:
    if not job_id:
        return
    _set_api_generation_task(job_id, logs=[line], **updates)


def _wait_for_background_job(getter: Callable[[str], dict[str, Any] | None], job_id: str, *, timeout_seconds: float = 7200.0) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last: dict[str, Any] = {}
    while time.time() < deadline:
        job = getter(job_id) or {}
        if job:
            last = job
        state = str(job.get("state") or "").lower()
        if state in {"succeeded", "complete", "completed", "success", "failed", "error"}:
            return job
        time.sleep(2.0)
    raise TimeoutError(f"Background job timed out: {job_id}")


def _automation_art_prompt(payload: dict[str, Any], result: dict[str, Any], *, album: bool = False) -> str:
    override = str(payload.get("art_prompt") or "").strip()
    if override:
        return override
    title = str(result.get("title") or payload.get("title") or ("album" if album else "song")).strip()
    artist = str(result.get("artist_name") or payload.get("artist_name") or "").strip()
    context = str(result.get("caption") or payload.get("caption") or payload.get("tags") or payload.get("simple_description") or "").strip()
    kind = "album cover" if album else "song cover"
    return f"Premium square {kind} artwork for '{title}' by {artist}. {context}. Cinematic, high detail, no text, no logo, no watermark."


def _automation_video_prompt(payload: dict[str, Any], result: dict[str, Any]) -> str:
    override = str(payload.get("video_prompt") or "").strip()
    if override:
        return override
    title = str(result.get("title") or payload.get("title") or "song").strip()
    artist = str(result.get("artist_name") or payload.get("artist_name") or "").strip()
    context = str(result.get("caption") or payload.get("caption") or payload.get("tags") or payload.get("simple_description") or "").strip()
    return f"Short real-life music video clip for '{title}' by {artist}. {context}. Natural camera motion, cinematic lighting, no text, no watermark."


def _audio_url_from_generation_result(result: dict[str, Any]) -> str:
    if str(result.get("audio_url") or "").strip():
        return str(result["audio_url"])
    audios = result.get("audios")
    if isinstance(audios, list):
        for item in audios:
            if isinstance(item, dict):
                url = str(item.get("audio_url") or item.get("download_url") or "").strip()
                if url:
                    return url
    return ""


def _attach_mflux_job_result(job: dict[str, Any], target_type: str, target_id: str) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    result_id = str(result.get("result_id") or (job.get("result_summary") or {}).get("result_id") or "").strip()
    if not result_id:
        raise RuntimeError("MFLUX automation job did not produce a result_id.")
    art = _mflux_art_metadata(result_id, scope=target_type or "mflux")
    if target_type in {"result", "generation", "generation_result"} and target_id:
        _attach_art_to_result(target_id, art)
    elif target_type == "song" and target_id:
        _merge_song_album_metadata(target_id, {"art": art, "single_art": art})
    elif target_type in {"album", "album_family"} and target_id:
        _attach_art_to_album_family(target_id, art)
    return art


def _run_generation_automation(task_id: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    result_id = str(result.get("result_id") or "").strip()
    song_id = str(result.get("song_id") or "").strip()
    target_type = "generation_result" if result_id else "song"
    target_id = result_id or song_id
    automation: dict[str, Any] = {
        "auto_song_art": parse_bool(payload.get("auto_song_art"), False),
        "auto_album_art": parse_bool(payload.get("auto_album_art"), False),
        "auto_video_clip": parse_bool(payload.get("auto_video_clip"), False),
        "jobs": [],
        "errors": [],
    }
    if not any([automation["auto_song_art"], automation["auto_album_art"], automation["auto_video_clip"]]):
        return
    _generation_job_log(task_id, "Automation started: art/video package jobs queued.", automation=automation)
    try:
        if automation["auto_song_art"] and target_id:
            art_job = mflux_create_job(
                {
                    "action": "generate",
                    "prompt": _automation_art_prompt(payload, result, album=False),
                    "model_id": str(payload.get("art_model_id") or "qwen-image"),
                    "width": 1024,
                    "height": 1024,
                    "steps": 30,
                    "seed": -1,
                    "target_type": target_type,
                    "target_id": target_id,
                }
            )
            automation["jobs"].append({"kind": "mflux_song_art", "job_id": art_job.get("id")})
            _generation_job_log(task_id, f"Song art job queued: {art_job.get('id')}", automation=automation)
            done = _wait_for_background_job(mflux_get_job, str(art_job.get("id")))
            if str(done.get("state") or "").lower() == "succeeded":
                automation["song_art"] = _attach_mflux_job_result(done, target_type, target_id)
                _generation_job_log(task_id, "Song art attached.", automation=automation)
            else:
                raise RuntimeError(str(done.get("error") or "Song art job failed"))
        if automation["auto_album_art"]:
            album_target = str(payload.get("album_family_id") or payload.get("album_id") or result.get("album_family_id") or result.get("album_id") or "").strip()
            if album_target:
                album_job = mflux_create_job(
                    {
                        "action": "generate",
                        "prompt": _automation_art_prompt(payload, result, album=True),
                        "model_id": str(payload.get("art_model_id") or "qwen-image"),
                        "width": 1024,
                        "height": 1024,
                        "steps": 30,
                        "seed": -1,
                        "target_type": "album_family",
                        "target_id": album_target,
                    }
                )
                automation["jobs"].append({"kind": "mflux_album_art", "job_id": album_job.get("id")})
                _generation_job_log(task_id, f"Album art job queued: {album_job.get('id')}", automation=automation)
                done = _wait_for_background_job(mflux_get_job, str(album_job.get("id")))
                if str(done.get("state") or "").lower() == "succeeded":
                    automation["album_art"] = _attach_mflux_job_result(done, "album_family", album_target)
                    _generation_job_log(task_id, "Album art attached.", automation=automation)
                else:
                    raise RuntimeError(str(done.get("error") or "Album art job failed"))
            else:
                automation["errors"].append("Album art skipped: no album target exists for this render.")
        if automation["auto_video_clip"] and target_id:
            audio_url = _audio_url_from_generation_result(result)
            if not audio_url:
                raise RuntimeError("Video automation requires a source audio URL from the completed song render.")
            video_job = mlx_video_create_job(
                {
                    "action": "song_video",
                    "prompt": _automation_video_prompt(payload, result),
                    "model_id": str(payload.get("video_model_id") or "ltx2-fast-draft"),
                    "width": 512,
                    "height": 320,
                    "num_frames": 33,
                    "fps": 24,
                    "steps": 8,
                    "seed": -1,
                    "audio_url": audio_url,
                    "audio_policy": "replace_with_source",
                    "mux_audio": True,
                    "target_type": target_type,
                    "target_id": target_id,
                }
            )
            automation["jobs"].append({"kind": "mlx_video", "job_id": video_job.get("id")})
            _generation_job_log(task_id, f"Music-video job queued: {video_job.get('id')}", automation=automation)
            done = _wait_for_background_job(mlx_video_get_job, str(video_job.get("id")), timeout_seconds=14400.0)
            if str(done.get("state") or "").lower() == "succeeded":
                video_result_id = str((done.get("result_summary") or {}).get("result_id") or (done.get("result") or {}).get("result_id") or "")
                automation["video"] = mlx_video_attach(
                    {"source_result_id": video_result_id, "target_type": target_type, "target_id": target_id}
                )
                _generation_job_log(task_id, "Muxed music video attached.", automation=automation)
            else:
                raise RuntimeError(str(done.get("error") or "Video automation job failed"))
    except Exception as exc:
        automation["errors"].append(str(exc))
        _generation_job_log(task_id, f"Automation warning: {exc}", automation=automation)
    finally:
        _set_api_generation_task(task_id, automation=automation)


def _generation_task_worker(task_id: str, payload: dict[str, Any]) -> None:
    summary = _generation_payload_summary(payload)
    _set_api_generation_task(
        task_id,
        state="running",
        status=0,
        stage="Preparing render",
        progress=5,
        payload_summary=summary,
        started_at=datetime.now(timezone.utc).isoformat(),
        logs=[
            f"Song job {task_id} started.",
            f"Mode: {summary.get('task_type')}; model: {summary.get('song_model') or 'auto'}; duration: {summary.get('duration') or 'auto'}s.",
        ],
    )
    try:
        _generation_job_log(task_id, "ACE-Step render running.", stage="Rendering", progress=25)
        worker_payload = dict(payload)
        worker_payload["generation_task_id"] = task_id
        result = _run_advanced_generation_with_download_retry(worker_payload)
        success = bool(result.get("success", True)) if isinstance(result, dict) else True
        warnings = [str(item) for item in (result.get("payload_warnings") or [])] if isinstance(result, dict) else []
        if success:
            _set_api_generation_task(
                task_id,
                state="succeeded",
                status=1,
                stage="Complete",
                progress=100,
                result=result,
                error="",
                finished_at=datetime.now(timezone.utc).isoformat(),
                logs=[
                    f"Render complete: {result.get('title') or summary.get('title') or 'track'}",
                    *([f"Warning: {item}" for item in warnings] if warnings else []),
                ],
            )
            threading.Thread(target=_run_generation_automation, args=(task_id, payload, result), daemon=True).start()
        else:
            error = str(result.get("error") or "Generation failed")
            _set_api_generation_task(
                task_id,
                state="failed",
                status=2,
                stage="Failed",
                progress=100,
                result=result,
                error=error,
                errors=[error],
                finished_at=datetime.now(timezone.utc).isoformat(),
                logs=[f"ERROR: {error}"],
            )
    except Exception as exc:
        _set_api_generation_task(
            task_id,
            state="failed",
            status=2,
            stage="Failed",
            progress=100,
            result=None,
            error=str(exc),
            errors=[str(exc)],
            finished_at=datetime.now(timezone.utc).isoformat(),
            logs=[f"ERROR: {exc}"],
        )
    finally:
        _cleanup_accelerator_memory()


def _submit_api_generation_task(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    summary = _generation_payload_summary(payload)
    _set_api_generation_task(
        task_id,
        payload=payload,
        payload_summary=summary,
        stage="Queued",
        progress=0,
        logs=[f"Song job {task_id} queued."],
    )
    thread = threading.Thread(target=_generation_task_worker, args=(task_id, payload), daemon=True)
    thread.start()
    return {"task_id": task_id, "job_id": task_id, "status": 0, "job": _generation_job_snapshot(task_id)}


def _song_batch_job_path(job_id: str) -> Path:
    return SONG_BATCHES_DIR / safe_id(job_id) / "job.json"


def _persist_song_batch_job(job: dict[str, Any]) -> None:
    job_id = safe_id(str(job.get("id") or ""))
    if not job_id:
        return
    path = _song_batch_job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(job), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _song_batch_audio_urls(result: dict[str, Any] | None) -> list[str]:
    if not isinstance(result, dict):
        return []
    urls: list[str] = []
    for key in ("audio_url", "download_url"):
        value = str(result.get(key) or "").strip()
        if value and value not in urls:
            urls.append(value)
    for audio in result.get("audios") or []:
        if not isinstance(audio, dict):
            continue
        for key in ("audio_url", "download_url"):
            value = str(audio.get(key) or "").strip()
            if value and value not in urls:
                urls.append(value)
    return urls


def _song_batch_payload_summary(body: dict[str, Any]) -> dict[str, Any]:
    songs = body.get("songs") if isinstance(body.get("songs"), list) else []
    return {
        "batch_title": str(body.get("batch_title") or body.get("title") or "Batch Songs").strip(),
        "song_count": len(songs),
        "stop_on_error": parse_bool(body.get("stop_on_error"), False),
    }


def _song_batch_song_entry(index: int, payload: dict[str, Any]) -> dict[str, Any]:
    summary = _generation_payload_summary(payload)
    title = str(summary.get("title") or payload.get("title") or f"Song {index + 1}").strip()
    return {
        "index": index,
        "track_number": index + 1,
        "title": title,
        "state": "queued",
        "status": "Queued",
        "progress": 0,
        "generation_job_id": "",
        "payload": _jsonable(payload),
        "payload_summary": summary,
        "result": None,
        "result_summary": {},
        "audio_urls": [],
        "error": "",
        "started_at": None,
        "finished_at": None,
    }


def _normalise_song_batch_body(body: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(body, dict):
        raise ValueError("Batch payload must be an object.")
    raw_songs = body.get("songs")
    if not isinstance(raw_songs, list) or not raw_songs:
        raise ValueError("Batch payload requires at least one song in songs[].")
    if len(raw_songs) > 40:
        raise ValueError("Batch Songs supports maximaal 40 nummers per queue.")

    songs: list[dict[str, Any]] = []
    field_errors: list[str] = []
    for index, item in enumerate(raw_songs):
        if not isinstance(item, dict):
            field_errors.append(f"Song {index + 1}: payload must be an object.")
            continue
        payload = dict(item)
        payload.setdefault("task_type", "text2music")
        payload.setdefault("wizard_mode", "batch")
        if "audio_backend" in payload or "use_mlx_dit" in payload:
            backend = _normalize_audio_backend(payload.get("audio_backend"), payload.get("use_mlx_dit"))
            payload["audio_backend"] = backend
            payload["use_mlx_dit"] = backend == "mlx"
        validation = _validate_generation_payload(payload)
        if not validation.get("valid"):
            errors = validation.get("field_errors") if isinstance(validation.get("field_errors"), dict) else {}
            reason = "; ".join(f"{key}: {value}" for key, value in errors.items()) or "invalid payload"
            field_errors.append(f"Song {index + 1}: {reason}")
        songs.append(payload)
    if field_errors:
        raise ValueError("Batch validation failed: " + " | ".join(field_errors))

    normalised = {
        "batch_title": str(body.get("batch_title") or body.get("title") or "Batch Songs").strip() or "Batch Songs",
        "stop_on_error": parse_bool(body.get("stop_on_error"), False),
        "songs": songs,
    }
    return normalised, [_song_batch_song_entry(index, payload) for index, payload in enumerate(songs)]


def _set_song_batch_job(job_id: str, **updates: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _song_batch_jobs_lock:
        job = _song_batch_jobs.setdefault(
            job_id,
            {
                "id": job_id,
                "kind": "song_batch",
                "state": "queued",
                "status": "Queued",
                "stage": "queued",
                "progress": 0,
                "batch_title": "Batch Songs",
                "payload": {},
                "payload_summary": {},
                "songs": [],
                "logs": [],
                "errors": [],
                "current_song": 0,
                "total_songs": 0,
                "completed_songs": 0,
                "failed_songs": 0,
                "remaining_songs": 0,
                "child_generation_job_id": "",
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "updated_at": now,
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
        updates.setdefault("updated_at", now)
        job.update(_jsonable(updates))
        if len(_song_batch_jobs) > SONG_BATCH_JOB_KEEP_LIMIT:
            removable = sorted(
                _song_batch_jobs.values(),
                key=lambda item: str(item.get("finished_at") or item.get("created_at") or ""),
            )
            for old in removable[: max(0, len(_song_batch_jobs) - SONG_BATCH_JOB_KEEP_LIMIT)]:
                if old.get("state") not in {"queued", "running"}:
                    _song_batch_jobs.pop(str(old.get("id")), None)
        snapshot = dict(job)
    try:
        _persist_song_batch_job(snapshot)
    except Exception as exc:
        print(f"[song_batch] failed to persist {job_id}: {exc}")
    return snapshot


def _song_batch_job_view(job: dict[str, Any]) -> dict[str, Any]:
    return _jsonable(dict(job))


def _song_batch_snapshot(job_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    with _song_batch_jobs_lock:
        if job_id:
            job = dict(_song_batch_jobs.get(job_id) or {})
            if not job:
                path = _song_batch_job_path(job_id)
                if path.is_file():
                    try:
                        job = json.loads(path.read_text(encoding="utf-8"))
                        _song_batch_jobs[job_id] = dict(job)
                    except Exception:
                        job = {}
            return _song_batch_job_view(job) if job else {}
        jobs = [dict(job) for job in _song_batch_jobs.values()]
        known_ids = {str(job.get("id") or "") for job in jobs}
    for path in SONG_BATCHES_DIR.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_id_from_disk = str(job.get("id") or path.parent.name)
        if job_id_from_disk in known_ids:
            continue
        with _song_batch_jobs_lock:
            _song_batch_jobs[job_id_from_disk] = dict(job)
        jobs.append(dict(job))
    return [
        _song_batch_job_view(job)
        for job in sorted(jobs, key=lambda item: str(item.get("created_at") or item.get("started_at") or ""), reverse=True)
    ]


def _song_batch_worker(job_id: str, payload: dict[str, Any]) -> None:
    songs = list(payload.get("songs") or [])
    existing = _song_batch_snapshot(job_id)
    entries = list(existing.get("songs") or []) if isinstance(existing, dict) else []
    total = len(songs)
    completed = 0
    failed = 0
    stop_on_error = parse_bool(payload.get("stop_on_error"), False)
    _set_song_batch_job(
        job_id,
        state="running",
        status="Batch running",
        stage="running",
        progress=1,
        started_at=datetime.now(timezone.utc).isoformat(),
        total_songs=total,
        remaining_songs=total,
        logs=[f"Batch {job_id} started with {total} song(s)."],
    )
    try:
        for index, song_payload in enumerate(songs):
            track_number = index + 1
            if not isinstance(song_payload, dict):
                continue
            if entries and index < len(entries):
                entries[index] = {
                    **entries[index],
                    "state": "running",
                    "status": "Rendering",
                    "progress": 0,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "error": "",
                }
            title = str((entries[index] if index < len(entries) else {}).get("title") or song_payload.get("title") or f"Song {track_number}")
            base_progress = int((index / max(1, total)) * 100)
            _set_song_batch_job(
                job_id,
                songs=entries,
                current_song=track_number,
                stage="rendering",
                status=f"Rendering song {track_number}/{total}",
                progress=max(1, base_progress),
                remaining_songs=max(0, total - index),
                logs=[f"Song {track_number}/{total} queued: {title}"],
            )
            child_payload = {
                **song_payload,
                "wizard_mode": "batch",
                "song_batch_id": job_id,
                "song_batch_index": track_number,
            }
            child = _submit_api_generation_task(child_payload)
            child_id = str(child.get("job_id") or child.get("task_id") or "")
            if entries and index < len(entries):
                entries[index]["generation_job_id"] = child_id
            _set_song_batch_job(
                job_id,
                child_generation_job_id=child_id,
                songs=entries,
                logs=[f"Song {track_number}/{total} render started: generation job {child_id}."],
            )
            last_progress = -1
            child_snapshot: dict[str, Any] = {}
            while True:
                child_snapshot = _generation_job_snapshot(child_id)
                if not isinstance(child_snapshot, dict) or not child_snapshot:
                    raise RuntimeError(f"Child generation job missing: {child_id}")
                child_progress = clamp_int(child_snapshot.get("progress"), 0, 0, 100)
                combined_progress = int(((index + child_progress / 100.0) / max(1, total)) * 100)
                if child_progress != last_progress:
                    last_progress = child_progress
                    if entries and index < len(entries):
                        entries[index]["progress"] = child_progress
                        entries[index]["status"] = str(child_snapshot.get("status") or child_snapshot.get("stage") or "Rendering")
                    _set_song_batch_job(
                        job_id,
                        songs=entries,
                        progress=max(base_progress, min(99, combined_progress)),
                        status=f"Rendering song {track_number}/{total}",
                        stage=str(child_snapshot.get("stage") or "rendering").lower() or "rendering",
                        logs=[f"Song {track_number}/{total}: {child_progress}%"],
                    )
                state = str(child_snapshot.get("state") or "").lower()
                if state in {"succeeded", "complete", "completed", "success", "failed", "error", "stopped"}:
                    break
                time.sleep(2.0)
            result = child_snapshot.get("result") if isinstance(child_snapshot.get("result"), dict) else None
            result_summary = child_snapshot.get("result_summary") if isinstance(child_snapshot.get("result_summary"), dict) else _generation_result_summary(result)
            state = str(child_snapshot.get("state") or "").lower()
            if state == "succeeded" and (not isinstance(result, dict) or result.get("success") is not False):
                completed += 1
                if entries and index < len(entries):
                    entries[index].update(
                        state="succeeded",
                        status="Complete",
                        progress=100,
                        result=result,
                        result_summary=result_summary,
                        audio_urls=_song_batch_audio_urls(result),
                        error="",
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    )
                _set_song_batch_job(
                    job_id,
                    songs=entries,
                    completed_songs=completed,
                    failed_songs=failed,
                    remaining_songs=max(0, total - completed - failed),
                    progress=int(((index + 1) / max(1, total)) * 100),
                    logs=[f"Song {track_number}/{total} complete: {title}"],
                )
            else:
                failed += 1
                error = str(child_snapshot.get("error") or (result or {}).get("error") or "Generation failed")
                if entries and index < len(entries):
                    entries[index].update(
                        state="failed",
                        status="Failed",
                        progress=100,
                        result=result,
                        result_summary=result_summary,
                        audio_urls=_song_batch_audio_urls(result),
                        error=error,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    )
                _set_song_batch_job(
                    job_id,
                    songs=entries,
                    completed_songs=completed,
                    failed_songs=failed,
                    remaining_songs=max(0, total - completed - failed),
                    errors=[f"Song {track_number}: {error}"],
                    logs=[f"Song {track_number}/{total} failed: {error}"],
                )
                if stop_on_error:
                    break
        finished = datetime.now(timezone.utc).isoformat()
        if failed and stop_on_error:
            state = "failed"
            status = "Batch stopped after failed song"
        elif failed:
            state = "succeeded"
            status = "Batch completed with errors"
        else:
            state = "succeeded"
            status = "Batch completed"
        _set_song_batch_job(
            job_id,
            state=state,
            status=status,
            stage="complete" if state == "succeeded" else "failed",
            progress=100,
            current_song=completed + failed,
            completed_songs=completed,
            failed_songs=failed,
            remaining_songs=max(0, total - completed - failed),
            child_generation_job_id="",
            finished_at=finished,
            logs=[status],
        )
    except Exception as exc:
        _set_song_batch_job(
            job_id,
            state="failed",
            status="Batch failed",
            stage="failed",
            progress=100,
            errors=[str(exc)],
            logs=[traceback.format_exc()],
            child_generation_job_id="",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        _cleanup_accelerator_memory()


def _submit_song_batch_job(body: dict[str, Any]) -> dict[str, Any]:
    payload, song_entries = _normalise_song_batch_body(body)
    job_id = uuid.uuid4().hex[:12]
    summary = _song_batch_payload_summary(payload)
    _set_song_batch_job(
        job_id,
        payload=payload,
        payload_summary=summary,
        batch_title=summary["batch_title"],
        total_songs=len(song_entries),
        remaining_songs=len(song_entries),
        songs=song_entries,
        logs=[f"Batch {job_id} queued with {len(song_entries)} song(s)."],
    )
    thread = threading.Thread(target=_song_batch_worker, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id, "job": _song_batch_snapshot(job_id)}


def _lora_benchmark_job_path(job_id: str) -> Path:
    return LORA_BENCHMARKS_DIR / safe_id(job_id) / "job.json"


def _persist_lora_benchmark_job(job: dict[str, Any]) -> None:
    job_id = safe_id(str(job.get("id") or ""))
    if not job_id:
        return
    path = _lora_benchmark_job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(job), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _lora_benchmark_adapter_label(adapter: dict[str, Any]) -> str:
    metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
    return str(
        adapter.get("display_name")
        or adapter.get("label")
        or adapter.get("name")
        or metadata.get("display_name")
        or metadata.get("generation_trigger_tag")
        or metadata.get("trigger_tag_raw")
        or metadata.get("trigger_tag")
        or "LoRA"
    ).strip() or "LoRA"


def _lora_benchmark_adapter_trigger(adapter: dict[str, Any], *, mode: str, custom: str) -> tuple[str, bool, str]:
    mode = str(mode or "auto").strip().lower()
    if mode in {"off", "none", "disabled"}:
        return "", False, "disabled"
    if mode == "custom":
        trigger = safe_generation_trigger_tag(custom)
        return trigger, bool(trigger), "custom" if trigger else "missing"
    metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
    candidates: list[Any] = [
        adapter.get("generation_trigger_tag"),
        metadata.get("generation_trigger_tag"),
        adapter.get("trigger_tag"),
        metadata.get("trigger_tag"),
        adapter.get("trigger_tag_raw"),
        metadata.get("trigger_tag_raw"),
    ]
    for list_key in ("trigger_aliases", "trigger_candidates"):
        value = adapter.get(list_key) or metadata.get(list_key)
        if isinstance(value, list):
            candidates.extend(value)
    for value in candidates:
        trigger = safe_generation_trigger_tag(str(value or "").strip())
        if trigger:
            source = str(adapter.get("trigger_source") or metadata.get("trigger_source") or "metadata").strip() or "metadata"
            return trigger, True, source
    return "", False, "missing"


def _lora_benchmark_adapter_epoch(adapter: dict[str, Any]) -> int | None:
    metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
    for key in ("epoch", "best_loss_epoch", "completed_epochs"):
        raw = adapter.get(key, metadata.get(key))
        try:
            value = int(raw)
            if value >= 0:
                return value
        except (TypeError, ValueError):
            pass
    text_blob = " ".join(
        str(adapter.get(key) or metadata.get(key) or "")
        for key in ("name", "display_name", "label", "path")
    )
    match = re.search(r"epoch[_\-\s]*(\d+)", text_blob, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _lora_benchmark_adapter_loss(adapter: dict[str, Any]) -> float | None:
    metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
    for key in ("loss", "last_loss", "best_loss"):
        raw = adapter.get(key, metadata.get(key))
        try:
            value = float(raw)
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    text_blob = " ".join(
        str(adapter.get(key) or metadata.get(key) or "")
        for key in ("name", "display_name", "label", "path")
    )
    match = re.search(r"loss[_\-\s]*(\d+(?:\.\d+)?)", text_blob, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _lora_benchmark_quality_status(adapter: dict[str, Any]) -> str:
    metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
    direct = str(adapter.get("quality_status") or metadata.get("quality_status") or "").strip()
    if direct:
        return direct
    quality = adapter_quality_metadata(metadata, adapter_type=str(adapter.get("adapter_type") or metadata.get("adapter_type") or "lora"))
    return str(quality.get("quality_status") or "unknown")


def _lora_benchmark_scales(value: Any) -> list[float]:
    raw_items = value if isinstance(value, list) else str(value or "").split(",")
    scales: list[float] = []
    for item in raw_items:
        try:
            scale = clamp_float(item, DEFAULT_LORA_GENERATION_SCALE, 0.0, 1.0)
        except Exception:
            continue
        if not any(abs(scale - existing) < 0.0001 for existing in scales):
            scales.append(round(scale, 4))
    return scales[:8] or [1.0]


def _lora_benchmark_base_payload(body: dict[str, Any]) -> dict[str, Any]:
    source = body.get("render_payload") if isinstance(body.get("render_payload"), dict) else body
    payload = dict(source or {})
    payload["task_type"] = str(payload.get("task_type") or "text2music").strip() or "text2music"
    payload["title"] = str(payload.get("title") or "LoRA Benchmark Test").strip() or "LoRA Benchmark Test"
    payload["artist_name"] = str(payload.get("artist_name") or "").strip()
    payload["caption"] = str(
        payload.get("caption")
        or "rap, hip hop, rhythmic spoken-word vocal, hard drums, deep bass, polished full mix"
    ).strip()
    payload["tags"] = str(payload.get("tags") or "rap, hip hop, hard drums, deep bass").strip()
    payload["negative_tags"] = str(
        payload.get("negative_tags") or "generic lyrics, muddy mix, mumbled vocals"
    ).strip()
    payload["lyrics"] = str(
        payload.get("lyrics")
        or "[Verse - rap, rhythmic spoken flow]\nEvery bar lands heavy while the drums keep pressure\nPocket full of thunder, every syllable measured\n\n[Chorus - rap hook]\nRun the benchmark loud, let the best take show\nSame words, same beat, hear the LoRA grow"
    ).strip()
    payload["instrumental"] = parse_bool(payload.get("instrumental"), payload["lyrics"].strip() == "[Instrumental]")
    duration = clamp_int(payload.get("audio_duration") or payload.get("duration"), 30, 10, 600)
    payload["duration"] = duration
    payload["audio_duration"] = duration
    payload["bpm"] = clamp_int(payload.get("bpm"), 92, 40, 220)
    payload["key_scale"] = str(payload.get("key_scale") or payload.get("keyscale") or "D minor").strip() or "D minor"
    payload["time_signature"] = str(payload.get("time_signature") or payload.get("timesignature") or "4/4").strip() or "4/4"
    payload["vocal_language"] = str(payload.get("vocal_language") or payload.get("language") or "en").strip() or "en"
    payload["song_model"] = str(payload.get("song_model") or "acestep-v15-xl-sft").strip() or "acestep-v15-xl-sft"
    backend = _normalize_audio_backend(payload.get("audio_backend"), payload.get("use_mlx_dit"))
    payload["audio_backend"] = backend
    payload["use_mlx_dit"] = backend == "mlx"
    payload["quality_profile"] = str(payload.get("quality_profile") or "chart_master").strip() or "chart_master"
    defaults = quality_profile_model_settings(payload["song_model"], payload["quality_profile"])
    payload["inference_steps"] = clamp_int(payload.get("inference_steps"), int(defaults.get("inference_steps") or 50), 1, 200)
    payload["guidance_scale"] = clamp_float(payload.get("guidance_scale"), float(defaults.get("guidance_scale") or 7), 1.0, 15.0)
    payload["shift"] = clamp_float(payload.get("shift"), float(defaults.get("shift") or 1.0), 0.0, 10.0)
    payload["audio_format"] = str(payload.get("audio_format") or "wav32").strip() or "wav32"
    payload["batch_size"] = 1
    payload["seed"] = clamp_int(payload.get("seed"), -1, -1, 2_147_483_647)
    payload["vocal_intelligibility_gate"] = parse_bool(payload.get("vocal_intelligibility_gate"), True)
    payload["vocal_intelligibility_model_rescue"] = False
    payload["manual_lora_review"] = True
    payload["lora_preflight_required"] = False
    payload["save_to_library"] = parse_bool(payload.get("save_to_library"), False)
    payload["wizard_mode"] = "lora_benchmark"
    payload["ui_mode"] = "lora_benchmark"
    return payload


def _lora_benchmark_adapter_from_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        adapter = dict(item)
    else:
        adapter = {"path": str(item or "").strip()}
    path = str(adapter.get("path") or adapter.get("lora_adapter_path") or "").strip()
    if not path:
        raise ValueError("Adapter path is required.")
    adapter["path"] = path
    adapter.setdefault("name", Path(path).name)
    if "metadata" not in adapter and Path(path).expanduser().exists():
        try:
            adapter["metadata"] = infer_adapter_model_metadata(Path(path).expanduser())
        except Exception:
            adapter["metadata"] = {}
    metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
    adapter.setdefault("adapter_type", metadata.get("adapter_type") or "lora")
    adapter.setdefault("model_variant", metadata.get("model_variant") or "")
    adapter.setdefault("song_model", metadata.get("song_model") or "")
    adapter["display_name"] = _lora_benchmark_adapter_label(adapter)
    adapter["quality_status"] = _lora_benchmark_quality_status(adapter)
    adapter["epoch"] = _lora_benchmark_adapter_epoch(adapter)
    adapter["loss"] = _lora_benchmark_adapter_loss(adapter)
    return _jsonable(adapter)


def _lora_benchmark_attempt_entry(
    index: int,
    *,
    role: str,
    adapter: dict[str, Any] | None,
    scale: float,
    trigger_mode: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    adapter = adapter or {}
    return {
        "attempt_id": f"attempt_{index + 1:03d}",
        "index": index,
        "attempt_number": index + 1,
        "attempt_role": role,
        "state": "queued",
        "status": "Queued",
        "progress": 0,
        "generation_job_id": "",
        "adapter": _jsonable(adapter),
        "adapter_name": _lora_benchmark_adapter_label(adapter) if adapter else "No LoRA",
        "adapter_path": str(adapter.get("path") or ""),
        "adapter_epoch": adapter.get("epoch"),
        "adapter_loss": adapter.get("loss"),
        "quality_status": str(adapter.get("quality_status") or ("baseline" if role == "baseline" else "unknown")),
        "lora_scale": scale,
        "trigger_mode": trigger_mode,
        "trigger_tag": str(payload.get("lora_trigger_tag") or ""),
        "payload": _jsonable(payload),
        "payload_summary": _generation_payload_summary(payload),
        "result": None,
        "result_summary": {},
        "score": 0,
        "score_breakdown": {},
        "audio_urls": [],
        "gate_status": "",
        "transcript_preview": "",
        "user_rating": 0,
        "user_scores": {},
        "user_verdict": "",
        "user_notes": "",
        "reviewed_at": "",
        "played_at": "",
        "error": "",
        "started_at": None,
        "finished_at": None,
    }


def _normalise_lora_benchmark_body(body: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(body, dict):
        raise ValueError("Benchmark payload must be an object.")
    raw_adapters = body.get("adapters") if isinstance(body.get("adapters"), list) else []
    if not raw_adapters and isinstance(body.get("adapter_paths"), list):
        raw_adapters = [{"path": item} for item in body.get("adapter_paths") or []]
    adapters: list[dict[str, Any]] = []
    adapter_errors: list[str] = []
    for index, item in enumerate(raw_adapters):
        try:
            adapters.append(_lora_benchmark_adapter_from_item(item))
        except Exception as exc:
            adapter_errors.append(f"Adapter {index + 1}: {exc}")
    include_baseline = parse_bool(body.get("include_baseline"), True)
    if not adapters and not include_baseline:
        raise ValueError("Select at least one adapter or enable the no-LoRA baseline.")
    if len(adapters) > 80:
        raise ValueError("LoRA Benchmark supports maximaal 80 adapters per run.")
    if adapter_errors:
        raise ValueError("Benchmark adapter validation failed: " + " | ".join(adapter_errors))

    scales = _lora_benchmark_scales(body.get("lora_scales", body.get("scales", [1.0])))
    trigger_mode = str(body.get("trigger_mode") or "auto").strip().lower() or "auto"
    if trigger_mode not in {"auto", "custom", "off"}:
        trigger_mode = "auto"
    custom_trigger = str(body.get("lora_trigger_tag") or body.get("custom_trigger_tag") or "").strip()
    base_payload = _lora_benchmark_base_payload(body)
    stop_on_error = parse_bool(body.get("stop_on_error"), False)

    attempts: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    if include_baseline:
        baseline_payload = {
            **base_payload,
            "title": f"{base_payload['title']} — no LoRA",
            "use_lora": False,
            "lora_adapter_path": "",
            "lora_adapter_name": "",
            "use_lora_trigger": False,
            "lora_trigger_tag": "",
            "lora_scale": 0,
        }
        validation = _validate_generation_payload(baseline_payload)
        if not validation.get("valid"):
            errors = validation.get("field_errors") if isinstance(validation.get("field_errors"), dict) else {}
            reason = "; ".join(f"{key}: {value}" for key, value in errors.items()) or "invalid baseline payload"
            validation_errors.append(f"Baseline: {reason}")
        attempts.append(
            _lora_benchmark_attempt_entry(
                len(attempts),
                role="baseline",
                adapter=None,
                scale=0,
                trigger_mode="off",
                payload=baseline_payload,
            )
        )

    for adapter in adapters:
        for scale in scales:
            trigger_tag, use_trigger, trigger_source = _lora_benchmark_adapter_trigger(adapter, mode=trigger_mode, custom=custom_trigger)
            label = _lora_benchmark_adapter_label(adapter)
            payload = {
                **base_payload,
                "title": f"{base_payload['title']} — {label} {int(round(scale * 100))}%",
                "use_lora": True,
                "lora_adapter_path": adapter["path"],
                "lora_adapter_name": label,
                "lora_scale": scale,
                "use_lora_trigger": use_trigger,
                "lora_trigger_tag": trigger_tag,
                "lora_trigger_source": trigger_source if use_trigger else "",
                "lora_trigger_aliases": adapter.get("trigger_aliases") or (adapter.get("metadata") or {}).get("trigger_aliases") or [],
                "lora_trigger_candidates": adapter.get("trigger_candidates") or (adapter.get("metadata") or {}).get("trigger_candidates") or [],
                "adapter_model_variant": adapter.get("model_variant") or "",
                "adapter_song_model": adapter.get("song_model") or "",
                "allow_unsafe_lora_for_benchmark": True,
            }
            validation = _validate_generation_payload(payload)
            if not validation.get("valid"):
                errors = validation.get("field_errors") if isinstance(validation.get("field_errors"), dict) else {}
                reason = "; ".join(f"{key}: {value}" for key, value in errors.items()) or "invalid payload"
                validation_errors.append(f"{label} {scale:g}: {reason}")
            attempts.append(
                _lora_benchmark_attempt_entry(
                    len(attempts),
                    role="lora",
                    adapter=adapter,
                    scale=scale,
                    trigger_mode=trigger_mode,
                    payload=payload,
                )
            )
    if len(attempts) > 240:
        raise ValueError("LoRA Benchmark supports maximaal 240 attempts per run.")
    if validation_errors:
        raise ValueError("Benchmark validation failed: " + " | ".join(validation_errors))

    payload = {
        "benchmark_title": str(body.get("benchmark_title") or body.get("title") or "LoRA Benchmark").strip() or "LoRA Benchmark",
        "stop_on_error": stop_on_error,
        "include_baseline": include_baseline,
        "trigger_mode": trigger_mode,
        "custom_trigger_tag": custom_trigger,
        "lora_scales": scales,
        "base_payload": _jsonable(base_payload),
        "adapters": adapters,
        "attempts": attempts,
    }
    return payload, attempts


def _lora_benchmark_payload_summary(body: dict[str, Any]) -> dict[str, Any]:
    attempts = body.get("attempts") if isinstance(body.get("attempts"), list) else []
    adapters = body.get("adapters") if isinstance(body.get("adapters"), list) else []
    return {
        "benchmark_title": str(body.get("benchmark_title") or "LoRA Benchmark").strip(),
        "adapter_count": len(adapters),
        "attempt_count": len(attempts),
        "lora_scales": list(body.get("lora_scales") or []),
        "include_baseline": parse_bool(body.get("include_baseline"), True),
        "trigger_mode": str(body.get("trigger_mode") or "auto"),
        "stop_on_error": parse_bool(body.get("stop_on_error"), False),
        "duration": (body.get("base_payload") or {}).get("duration") if isinstance(body.get("base_payload"), dict) else None,
        "song_model": (body.get("base_payload") or {}).get("song_model") if isinstance(body.get("base_payload"), dict) else None,
        "audio_backend": (body.get("base_payload") or {}).get("audio_backend") if isinstance(body.get("base_payload"), dict) else None,
    }


def _lora_benchmark_score(child_snapshot: dict[str, Any], result: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    result = result if isinstance(result, dict) else {}
    audios = result.get("audios") if isinstance(result.get("audios"), list) else []
    gate = result.get("vocal_intelligibility_gate") if isinstance(result.get("vocal_intelligibility_gate"), dict) else {}
    style_audit = result.get("style_conditioning_audit") if isinstance(result.get("style_conditioning_audit"), dict) else {}
    trigger_audit = result.get("lora_trigger_conditioning_audit") if isinstance(result.get("lora_trigger_conditioning_audit"), dict) else {}
    pro_audit = result.get("pro_quality_audit") if isinstance(result.get("pro_quality_audit"), dict) else {}
    recommended = pro_audit.get("recommended_take") if isinstance(pro_audit.get("recommended_take"), dict) else {}
    pro_score = recommended.get("score")
    if pro_score in (None, "") and audios and isinstance(audios[0], dict):
        pro_score = audios[0].get("pro_quality_score")
    try:
        pro_value = max(0.0, min(100.0, float(pro_score)))
    except (TypeError, ValueError):
        pro_value = 0.0

    score = 0.0
    reasons: list[str] = []
    child_state = str(child_snapshot.get("state") or "").lower()
    success = child_state in {"succeeded", "complete", "completed", "success"} and result.get("success") is not False
    audio_urls = _song_batch_audio_urls(result)
    if success and audio_urls:
        score += 20
        reasons.append("audio_written")
    else:
        score -= 35
        reasons.append("generation_failed")

    if gate:
        if parse_bool(gate.get("passed"), False) or str(gate.get("status") or "").lower() == "pass":
            score += 35
            reasons.append("vocal_gate_pass")
        elif gate.get("blocking") is False and str(gate.get("status") or "").lower() == "needs_review":
            score += 8
            reasons.append("vocal_gate_needs_review")
        else:
            score -= 20
            reasons.append(str(gate.get("issue") or gate.get("status") or "vocal_gate_fail"))
        hits = gate.get("keyword_hits") if isinstance(gate.get("keyword_hits"), list) else []
        score += min(15, len(hits) * 4)
        repeat_ratio = gate.get("repeat_ratio")
        filler_ratio = gate.get("filler_ratio")
        try:
            score -= max(0.0, float(repeat_ratio) - 0.18) * 40
        except (TypeError, ValueError):
            pass
        try:
            score -= max(0.0, float(filler_ratio) - 0.18) * 30
        except (TypeError, ValueError):
            pass
    else:
        reasons.append("vocal_gate_missing")

    score += pro_value * 0.18
    if str(style_audit.get("status") or "").lower() in {"pass", "ok"}:
        score += 8
        reasons.append("style_audit_pass")
    elif style_audit:
        score -= 4
        reasons.append("style_audit_warn")

    if parse_bool(payload.get("use_lora"), False):
        if parse_bool(result.get("lora_trigger_applied") or trigger_audit.get("applied"), False):
            score += 7
            reasons.append("lora_trigger_applied")
        elif parse_bool(payload.get("use_lora_trigger"), False):
            score -= 5
            reasons.append("lora_trigger_missing")
        if result.get("with_lora") is False:
            score -= 12
            reasons.append("lora_not_active")

    requested = str(result.get("requested_song_model") or payload.get("song_model") or "").strip()
    actual = str(result.get("actual_song_model") or requested).strip()
    if requested and actual and requested != actual:
        score -= 10
        reasons.append("model_mismatch")
    else:
        score += 5
        reasons.append("model_match")

    score = round(max(0.0, min(100.0, score)), 2)
    breakdown = {
        "score": score,
        "success": success,
        "audio_url_count": len(audio_urls),
        "vocal_gate_status": gate.get("status") or result.get("vocal_gate_status") or "",
        "vocal_gate_passed": gate.get("passed"),
        "keyword_hits": gate.get("keyword_hits") or [],
        "repeat_ratio": gate.get("repeat_ratio"),
        "filler_ratio": gate.get("filler_ratio"),
        "pro_quality_score": pro_value,
        "style_audit_status": style_audit.get("status") or "",
        "lora_trigger_applied": result.get("lora_trigger_applied") or trigger_audit.get("applied"),
        "reasons": reasons,
    }
    return score, _jsonable(breakdown)


def _lora_benchmark_rankings(results: list[dict[str, Any]]) -> dict[str, Any]:
    auto_candidates = [item for item in results if str(item.get("state") or "").lower() in {"succeeded", "success", "complete", "completed"}]
    best_auto = max(auto_candidates or results, key=lambda item: float(item.get("score") or 0), default={})
    def manual_score(item: dict[str, Any]) -> float:
        rating = float(item.get("user_rating") or 0)
        raw_scores = item.get("user_scores") if isinstance(item.get("user_scores"), dict) else {}
        score_values = [float(raw_scores.get(key) or 0) for key in ("vocal", "style", "mix", "fit")]
        score_values = [value for value in score_values if value > 0]
        category_average = sum(score_values) / len(score_values) if score_values else 0.0
        verdict_bonus = {"keep": 0.75, "maybe": 0.25, "reject": -1.0}.get(str(item.get("user_verdict") or "").lower(), 0.0)
        return max(rating, category_average) + verdict_bonus

    rated = [item for item in results if manual_score(item) > 0]
    best_manual = max(
        rated,
        key=lambda item: (manual_score(item), float(item.get("score") or 0)),
        default={},
    )
    review_summary = {
        "reviewed": sum(1 for item in results if manual_score(item) > 0 or item.get("user_notes")),
        "played": sum(1 for item in results if item.get("played_at")),
        "keep": sum(1 for item in results if str(item.get("user_verdict") or "").lower() == "keep"),
        "maybe": sum(1 for item in results if str(item.get("user_verdict") or "").lower() == "maybe"),
        "reject": sum(1 for item in results if str(item.get("user_verdict") or "").lower() == "reject"),
    }
    return {
        "best_auto_result_id": best_auto.get("attempt_id") or "",
        "best_manual_result_id": best_manual.get("attempt_id") or "",
        "best_result_id": (best_manual or best_auto).get("attempt_id") or "",
        "review_summary": review_summary,
    }


def _set_lora_benchmark_job(job_id: str, **updates: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _lora_benchmark_jobs_lock:
        job = _lora_benchmark_jobs.setdefault(
            job_id,
            {
                "id": job_id,
                "kind": "lora_benchmark",
                "state": "queued",
                "status": "Queued",
                "stage": "queued",
                "progress": 0,
                "benchmark_title": "LoRA Benchmark",
                "payload": {},
                "payload_summary": {},
                "attempts": [],
                "results": [],
                "logs": [],
                "errors": [],
                "current_attempt": 0,
                "total_attempts": 0,
                "completed_attempts": 0,
                "failed_attempts": 0,
                "remaining_attempts": 0,
                "child_generation_job_id": "",
                "best_auto_result_id": "",
                "best_manual_result_id": "",
                "best_result_id": "",
                "review_summary": {"reviewed": 0, "played": 0, "keep": 0, "maybe": 0, "reject": 0},
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "updated_at": now,
            },
        )
        if "logs" in updates:
            old_logs = list(job.get("logs") or [])
            new_logs = updates.pop("logs")
            if isinstance(new_logs, list):
                job["logs"] = (old_logs + [str(item) for item in new_logs])[-700:]
            elif new_logs:
                job["logs"] = (old_logs + [str(new_logs)])[-700:]
        if "errors" in updates:
            old_errors = list(job.get("errors") or [])
            new_errors = updates.pop("errors")
            if isinstance(new_errors, list):
                job["errors"] = (old_errors + [str(item) for item in new_errors])[-150:]
            elif new_errors:
                job["errors"] = (old_errors + [str(new_errors)])[-150:]
        updates.setdefault("updated_at", now)
        job.update(_jsonable(updates))
        if "results" in updates:
            rankings = _lora_benchmark_rankings(list(job.get("results") or []))
            job.update(rankings)
        if len(_lora_benchmark_jobs) > LORA_BENCHMARK_JOB_KEEP_LIMIT:
            removable = sorted(
                _lora_benchmark_jobs.values(),
                key=lambda item: str(item.get("finished_at") or item.get("created_at") or ""),
            )
            for old in removable[: max(0, len(_lora_benchmark_jobs) - LORA_BENCHMARK_JOB_KEEP_LIMIT)]:
                if old.get("state") not in {"queued", "running"}:
                    _lora_benchmark_jobs.pop(str(old.get("id")), None)
        snapshot = dict(job)
    try:
        _persist_lora_benchmark_job(snapshot)
    except Exception as exc:
        print(f"[lora_benchmark] failed to persist {job_id}: {exc}")
    return snapshot


def _lora_benchmark_job_view(job: dict[str, Any]) -> dict[str, Any]:
    return _jsonable(dict(job))


def _lora_benchmark_snapshot(job_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    with _lora_benchmark_jobs_lock:
        if job_id:
            job = dict(_lora_benchmark_jobs.get(job_id) or {})
            if not job:
                path = _lora_benchmark_job_path(job_id)
                if path.is_file():
                    try:
                        job = json.loads(path.read_text(encoding="utf-8"))
                        _lora_benchmark_jobs[job_id] = dict(job)
                    except Exception:
                        job = {}
            return _lora_benchmark_job_view(job) if job else {}
        jobs = [dict(job) for job in _lora_benchmark_jobs.values()]
        known_ids = {str(job.get("id") or "") for job in jobs}
    for path in LORA_BENCHMARKS_DIR.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_id_from_disk = str(job.get("id") or path.parent.name)
        if job_id_from_disk in known_ids:
            continue
        with _lora_benchmark_jobs_lock:
            _lora_benchmark_jobs[job_id_from_disk] = dict(job)
        jobs.append(dict(job))
    return [
        _lora_benchmark_job_view(job)
        for job in sorted(jobs, key=lambda item: str(item.get("created_at") or item.get("started_at") or ""), reverse=True)
    ]


def _lora_benchmark_worker(job_id: str, payload: dict[str, Any]) -> None:
    attempts = list(payload.get("attempts") or [])
    existing = _lora_benchmark_snapshot(job_id)
    entries = list(existing.get("attempts") or []) if isinstance(existing, dict) else []
    results: list[dict[str, Any]] = []
    total = len(attempts)
    completed = 0
    failed = 0
    stop_on_error = parse_bool(payload.get("stop_on_error"), False)
    _set_lora_benchmark_job(
        job_id,
        state="running",
        status="Benchmark running",
        stage="running",
        progress=1,
        started_at=datetime.now(timezone.utc).isoformat(),
        total_attempts=total,
        remaining_attempts=total,
        logs=[f"LoRA Benchmark {job_id} started with {total} attempt(s)."],
    )
    try:
        for index, attempt in enumerate(attempts):
            snapshot = _lora_benchmark_snapshot(job_id)
            if isinstance(snapshot, dict) and parse_bool(snapshot.get("stop_requested"), False):
                _set_lora_benchmark_job(
                    job_id,
                    state="stopped",
                    status="Benchmark stopped",
                    stage="stopped",
                    progress=int((completed + failed) / max(1, total) * 100),
                    child_generation_job_id="",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    logs=["LoRA Benchmark stopped before the next attempt."],
                )
                return
            if not isinstance(attempt, dict):
                continue
            attempt_number = index + 1
            attempt_id = str(attempt.get("attempt_id") or f"attempt_{attempt_number:03d}")
            payload_for_generation = dict(attempt.get("payload") or {})
            label = str(attempt.get("adapter_name") or payload_for_generation.get("title") or attempt_id)
            if entries and index < len(entries):
                entries[index] = {
                    **entries[index],
                    "state": "running",
                    "status": "Rendering",
                    "progress": 0,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "error": "",
                }
            base_progress = int((index / max(1, total)) * 100)
            _set_lora_benchmark_job(
                job_id,
                attempts=entries,
                current_attempt=attempt_number,
                stage="rendering",
                status=f"Rendering attempt {attempt_number}/{total}",
                progress=max(1, base_progress),
                remaining_attempts=max(0, total - index),
                logs=[
                    f"Attempt {attempt_number}/{total} queued: {label} · scale {attempt.get('lora_scale')} · trigger {attempt.get('trigger_tag') or 'off'}"
                ],
            )
            child_payload = {
                **payload_for_generation,
                "wizard_mode": "lora_benchmark",
                "lora_benchmark_id": job_id,
                "lora_benchmark_attempt_id": attempt_id,
            }
            child = _submit_api_generation_task(child_payload)
            child_id = str(child.get("job_id") or child.get("task_id") or "")
            if entries and index < len(entries):
                entries[index]["generation_job_id"] = child_id
            _set_lora_benchmark_job(
                job_id,
                child_generation_job_id=child_id,
                attempts=entries,
                logs=[f"Attempt {attempt_number}/{total} render started: generation job {child_id}."],
            )
            last_progress = -1
            child_snapshot: dict[str, Any] = {}
            while True:
                child_snapshot = _generation_job_snapshot(child_id)
                if not isinstance(child_snapshot, dict) or not child_snapshot:
                    raise RuntimeError(f"Child generation job missing: {child_id}")
                child_progress = clamp_int(child_snapshot.get("progress"), 0, 0, 100)
                combined_progress = int(((index + child_progress / 100.0) / max(1, total)) * 100)
                if child_progress != last_progress:
                    last_progress = child_progress
                    if entries and index < len(entries):
                        entries[index]["progress"] = child_progress
                        entries[index]["status"] = str(child_snapshot.get("status") or child_snapshot.get("stage") or "Rendering")
                    _set_lora_benchmark_job(
                        job_id,
                        attempts=entries,
                        progress=max(base_progress, min(99, combined_progress)),
                        status=f"Rendering attempt {attempt_number}/{total}",
                        stage=str(child_snapshot.get("stage") or "rendering").lower() or "rendering",
                        logs=[f"Attempt {attempt_number}/{total}: {child_progress}%"],
                    )
                state = str(child_snapshot.get("state") or "").lower()
                if state in {"succeeded", "complete", "completed", "success", "failed", "error", "stopped"}:
                    break
                snapshot = _lora_benchmark_snapshot(job_id)
                if isinstance(snapshot, dict) and parse_bool(snapshot.get("stop_requested"), False):
                    _set_lora_benchmark_job(
                        job_id,
                        state="stopping",
                        status="Stop requested; waiting for current render result",
                        stage="stopping",
                        attempts=entries,
                        logs=[f"Stop requested during attempt {attempt_number}/{total}."],
                    )
                time.sleep(2.0)

            result = child_snapshot.get("result") if isinstance(child_snapshot.get("result"), dict) else None
            result_summary = child_snapshot.get("result_summary") if isinstance(child_snapshot.get("result_summary"), dict) else _generation_result_summary(result)
            state = str(child_snapshot.get("state") or "").lower()
            score, breakdown = _lora_benchmark_score(child_snapshot, result, payload_for_generation)
            audio_urls = _song_batch_audio_urls(result)
            gate = (result or {}).get("vocal_intelligibility_gate") if isinstance((result or {}).get("vocal_intelligibility_gate"), dict) else {}
            manual_review_success = bool(
                audio_urls
                and _manual_lora_review_allowed(payload_for_generation)
                and isinstance(result, dict)
                and str(gate.get("status") or "").lower() in {"needs_review", "fail"}
            )
            success = (
                state == "succeeded"
                and (not isinstance(result, dict) or result.get("success") is not False)
            ) or manual_review_success
            entry_update = {
                "state": "succeeded" if success else "failed",
                "status": "Complete" if success and not manual_review_success else ("Manual review" if manual_review_success else "Failed"),
                "progress": 100,
                "generation_job_id": child_id,
                "result": result,
                "result_summary": result_summary,
                "score": score,
                "score_breakdown": breakdown,
                "audio_urls": audio_urls,
                "gate_status": gate.get("status") or (result or {}).get("vocal_gate_status") or "",
                "transcript_preview": gate.get("transcript_preview") or (result or {}).get("transcript_preview") or "",
                "error": "" if success else str(child_snapshot.get("error") or (result or {}).get("error") or "Generation failed"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            if entries and index < len(entries):
                entries[index].update(entry_update)
                result_entry = dict(entries[index])
            else:
                result_entry = {**attempt, **entry_update}
            results.append(_jsonable(result_entry))
            if success:
                completed += 1
                log_line = f"Attempt {attempt_number}/{total} complete: {label} · score {score:.1f}"
            else:
                failed += 1
                log_line = f"Attempt {attempt_number}/{total} failed: {entry_update['error']}"
            _set_lora_benchmark_job(
                job_id,
                attempts=entries,
                results=results,
                completed_attempts=completed,
                failed_attempts=failed,
                remaining_attempts=max(0, total - completed - failed),
                progress=int(((index + 1) / max(1, total)) * 100),
                logs=[log_line],
                errors=[] if success else [str(entry_update["error"])],
            )
            if (not success) and stop_on_error:
                break

        finished = datetime.now(timezone.utc).isoformat()
        if failed and stop_on_error:
            state = "failed"
            status = "Benchmark stopped after failed attempt"
        elif failed:
            state = "succeeded"
            status = "Benchmark completed with errors"
        else:
            state = "succeeded"
            status = "Benchmark completed"
        _set_lora_benchmark_job(
            job_id,
            state=state,
            status=status,
            stage="complete" if state == "succeeded" else "failed",
            progress=100,
            current_attempt=completed + failed,
            completed_attempts=completed,
            failed_attempts=failed,
            remaining_attempts=max(0, total - completed - failed),
            child_generation_job_id="",
            finished_at=finished,
            logs=[status],
        )
    except Exception as exc:
        _set_lora_benchmark_job(
            job_id,
            state="failed",
            status="Benchmark failed",
            stage="failed",
            progress=100,
            errors=[str(exc)],
            logs=[traceback.format_exc()],
            child_generation_job_id="",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        _cleanup_accelerator_memory()


def _submit_lora_benchmark_job(body: dict[str, Any]) -> dict[str, Any]:
    payload, attempts = _normalise_lora_benchmark_body(body)
    job_id = uuid.uuid4().hex[:12]
    summary = _lora_benchmark_payload_summary(payload)
    _set_lora_benchmark_job(
        job_id,
        payload=payload,
        payload_summary=summary,
        benchmark_title=summary["benchmark_title"],
        total_attempts=len(attempts),
        remaining_attempts=len(attempts),
        attempts=attempts,
        logs=[f"LoRA Benchmark {job_id} queued with {len(attempts)} attempt(s)."],
    )
    thread = threading.Thread(target=_lora_benchmark_worker, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id, "job": _lora_benchmark_snapshot(job_id)}


def _stop_lora_benchmark_job(job_id: str) -> dict[str, Any]:
    job = _lora_benchmark_snapshot(job_id)
    if not isinstance(job, dict) or not job:
        raise KeyError("LoRA benchmark job not found")
    state = str(job.get("state") or "").lower()
    if state in {"succeeded", "success", "failed", "error", "stopped"}:
        return job
    return _set_lora_benchmark_job(
        job_id,
        state="stopped",
        status="Benchmark stopped",
        stage="stopped",
        stop_requested=True,
        child_generation_job_id="",
        finished_at=datetime.now(timezone.utc).isoformat(),
        logs=["LoRA Benchmark stopped."],
    )


def _rate_lora_benchmark_result(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    job = _lora_benchmark_snapshot(job_id)
    if not isinstance(job, dict) or not job:
        raise KeyError("LoRA benchmark job not found")
    attempt_id = str(body.get("attempt_id") or body.get("result_id") or "").strip()
    if not attempt_id:
        raise ValueError("attempt_id is required")
    has_rating = "user_rating" in body
    rating = clamp_int(body.get("user_rating"), 0, 0, 5) if has_rating else None
    has_notes = "user_notes" in body or "notes" in body
    notes = str(body.get("user_notes") or body.get("notes") or "").strip() if has_notes else None
    raw_scores = body.get("user_scores") if isinstance(body.get("user_scores"), dict) else {}
    scores: dict[str, int] = {}
    for key in ("vocal", "style", "mix", "fit"):
        if key in raw_scores:
            scores[key] = clamp_int(raw_scores.get(key), 0, 0, 5)
    verdict = str(body.get("user_verdict") or "").strip().lower()
    if verdict not in {"", "keep", "maybe", "reject"}:
        raise ValueError("user_verdict must be keep, maybe, reject, or empty")
    now = datetime.now(timezone.utc).isoformat()
    results = list(job.get("results") or [])
    attempts = list(job.get("attempts") or [])
    updated = False
    for collection in (results, attempts):
        for item in collection:
            if isinstance(item, dict) and str(item.get("attempt_id") or "") == attempt_id:
                if has_rating:
                    item["user_rating"] = rating
                if has_notes:
                    item["user_notes"] = notes
                if raw_scores:
                    item["user_scores"] = scores
                if "user_verdict" in body:
                    item["user_verdict"] = verdict
                if item.get("user_rating") or item.get("user_scores") or item.get("user_verdict") or item.get("user_notes"):
                    item["reviewed_at"] = now
                if parse_bool(body.get("played"), False):
                    item["played_at"] = now
                updated = True
    if not updated:
        raise KeyError("Benchmark attempt not found")
    suffix = f"{rating}/5" if has_rating else "updated"
    if "user_verdict" in body and verdict:
        suffix += f" · {verdict}"
    return _set_lora_benchmark_job(job_id, results=results, attempts=attempts, logs=[f"Review saved for {attempt_id}: {suffix}"])


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
                    "generation_info": f"MLX Media {result.get('runner', 'fast')} generation",
                    "seed_value": audio.get("seed") or "",
                    "lm_model": params.get("ace_lm_model") or "",
                    "dit_model": result.get("active_song_model") or params.get("song_model") or "",
                    "result_id": audio.get("result_id") or result.get("result_id") or "",
                    "audio_id": audio.get("id") or "",
                }
            )
        return {
            "task_id": task_id,
            "status": 1,
            "result": json.dumps(result_payload),
            "acejam_result": _jsonable(result),
            "error": None,
        }
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
        "message": "Open MLX Media through the Pinokio Web UI HTTP URL, not a file:// path.",
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
        "active_generation_jobs": [
            job
            for job in _generation_job_snapshot()
            if isinstance(job, dict) and str(job.get("state") or "").lower() in {"queued", "running"}
        ],
        "active_song_batch_jobs": [
            job
            for job in _song_batch_snapshot()
            if isinstance(job, dict) and str(job.get("state") or "").lower() in {"queued", "running"}
        ],
        "active_lora_benchmark_jobs": [
            job
            for job in _lora_benchmark_snapshot()
            if isinstance(job, dict) and str(job.get("state") or "").lower() in {"queued", "running"}
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
    now = datetime.now(timezone.utc).isoformat()
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
                "stage": "queued",
                "current_task": "Queued",
                "current_agent": "",
                "current_model_album": "",
                "current_track": "",
                "total_tracks": 0,
                "completed_tracks": 0,
                "remaining_tracks": 0,
                "waiting_on_llm": False,
                "llm_provider": "",
                "llm_model": "",
                "llm_wait_started_at": "",
                "llm_wait_elapsed_s": 0,
                "last_log_at": None,
                "last_update_at": now,
                "updated_at": now,
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
        updates.setdefault("updated_at", now)
        updates.setdefault("last_update_at", now)
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
    updates.setdefault("last_log_at", datetime.now(timezone.utc).isoformat())
    _set_album_job(job_id, logs=[line], **updates)


def _album_plan_agent_weight(agent_name: str) -> float:
    lower = str(agent_name or "").lower()
    if "album intake" in lower or "bible" in lower:
        return 0.05
    if "track concept" in lower:
        return 0.08
    if "tag" in lower:
        return 0.16
    if "section" in lower:
        return 0.28
    if "hook" in lower:
        return 0.38
    if "lyrics" in lower or "lyric" in lower:
        return 0.58
    if "caption" in lower or "sonic" in lower:
        return 0.76
    if "performance" in lower or "settings" in lower:
        return 0.86
    if "final payload" in lower or "assembler" in lower:
        return 0.95
    return 0.5


def _album_plan_track_progress(track_number: int, total_tracks: int, agent_name: str = "") -> int:
    total_tracks = max(1, int(total_tracks or 1))
    track_number = max(1, min(total_tracks, int(track_number or 1)))
    weight = max(0.0, min(0.99, _album_plan_agent_weight(agent_name)))
    return int(max(5, min(98, round(10 + 85 * (((track_number - 1) + weight) / total_tracks)))))


def _album_plan_progress_updates_from_log(line: str, job: dict[str, Any]) -> dict[str, Any]:
    compact = str(line or "").replace("\n", " ")[:700]
    lower = compact.lower()
    total_tracks = clamp_int(job.get("total_tracks") or job.get("expected_count") or 0, 0, 0, 1000)
    if total_tracks <= 0:
        total_tracks = 0
    current_track = clamp_int(job.get("current_track") or 0, 0, 0, max(1, total_tracks or 1))
    updates: dict[str, Any] = {}

    writing_match = re.search(r"writing track\s+(\d+)\s*/\s*(\d+)", lower)
    if writing_match:
        current_track = int(writing_match.group(1))
        total_tracks = int(writing_match.group(2))
        updates.update(
            stage="track_writer",
            status=f"Writing track {current_track}/{total_tracks}",
            current_task=f"Track {current_track}/{total_tracks}: writer loop",
            current_track=current_track,
            total_tracks=total_tracks,
            completed_tracks=max(0, current_track - 1),
            remaining_tracks=max(0, total_tracks - current_track + 1),
            progress=_album_plan_track_progress(current_track, total_tracks, "Track Concept Agent"),
        )

    completed_match = re.search(r"writing track\s+(\d+)\s*/\s*(\d+).*completed", lower)
    if completed_match:
        completed = int(completed_match.group(1))
        total_tracks = int(completed_match.group(2))
        updates.update(
            stage="track_complete",
            status=f"Track {completed}/{total_tracks} ready",
            current_task=f"Track {completed}/{total_tracks}: render payload ready",
            completed_tracks=completed,
            remaining_tracks=max(0, total_tracks - completed),
            progress=_album_plan_track_progress(completed, total_tracks, "Final Payload Assembler"),
        )

    call_match = re.search(
        r"crewai micro agent call:\s*(?P<agent>.+?)\s+attempt\s+(?P<attempt>\d+)\s+via\s+(?P<provider>.+?)\s+\(prompt_chars=(?P<prompt>\d+),\s*system=(?P<system>\d+),\s*user=(?P<user>\d+)\)",
        compact,
        flags=re.IGNORECASE,
    )
    if call_match:
        agent = call_match.group("agent").strip()
        provider = call_match.group("provider").strip()
        if current_track <= 0 and total_tracks > 0 and "album intake" not in agent.lower():
            current_track = 1
        progress = (
            _album_plan_track_progress(current_track, total_tracks, agent)
            if current_track and total_tracks
            else int(8 + 22 * _album_plan_agent_weight(agent))
        )
        updates.update(
            stage="waiting_on_llm",
            status=f"Waiting on {provider}: {agent}",
            current_task=f"Waiting for {agent} response",
            current_agent=agent,
            agent_attempt=int(call_match.group("attempt")),
            llm_provider=provider,
            llm_prompt_chars=int(call_match.group("prompt")),
            llm_system_chars=int(call_match.group("system")),
            llm_user_chars=int(call_match.group("user")),
            waiting_on_llm=True,
            llm_wait_started_at=datetime.now(timezone.utc).isoformat(),
            llm_wait_elapsed_s=0,
            progress=max(int(job.get("progress") or 0), min(97, progress)),
        )

    response_match = re.search(r"crewai micro agent response:\s*(?P<agent>.+?)\s+\d+\s+chars\s+in\s+(?P<elapsed>[\d.]+)s", compact, flags=re.IGNORECASE)
    if response_match:
        agent = response_match.group("agent").strip()
        updates.update(
            stage="agent_response",
            status=f"{agent} response received",
            current_task=f"{agent} response received",
            current_agent=agent,
            waiting_on_llm=False,
            llm_wait_elapsed_s=float(response_match.group("elapsed")),
            llm_last_elapsed_s=float(response_match.group("elapsed")),
        )

    parsed_match = re.search(r"crewai micro agent parsed .*?:\s*(?P<agent>.+?)\s+attempt\s+\d+\s+ok", compact, flags=re.IGNORECASE)
    if parsed_match:
        agent = parsed_match.group("agent").strip()
        updates.update(
            stage="agent_parse",
            status=f"{agent} parsed",
            current_task=f"{agent} parsed OK",
            current_agent=agent,
            waiting_on_llm=False,
        )

    micro_match = re.search(r"micro setting call:\s*(?P<agent>.+?)\s+for\s+track\s+(?P<track>\d+)", compact, flags=re.IGNORECASE)
    if micro_match:
        current_track = int(micro_match.group("track"))
        agent = micro_match.group("agent").strip()
        updates.update(
            stage="track_settings",
            status=f"Track {current_track}/{total_tracks or '?'} settings",
            current_task=f"Track {current_track}: {agent}",
            current_agent=agent,
            current_track=current_track,
            progress=_album_plan_track_progress(current_track, total_tracks, agent) if total_tracks else max(int(job.get("progress") or 0), 55),
        )

    produced_match = re.search(r"acejam director produced\s+(\d+)\s+direct", lower)
    if produced_match:
        completed = int(produced_match.group(1))
        updates.update(
            stage="assembly",
            status="MLX Media direct payloads ready",
            current_task="Final album payload assembly",
            completed_tracks=completed,
            remaining_tracks=max(0, (total_tracks or completed) - completed),
            progress=95,
            waiting_on_llm=False,
        )

    if "planning album bible with acejam agents" in lower:
        updates.update(stage="album_bible", status="MLX Media album bible running", current_task="Building album bible", progress=max(int(job.get("progress") or 0), 15))
    elif "acejam agents planned" in lower:
        updates.update(stage="track_blueprints", status="MLX Media track blueprints ready", current_task="Track blueprints ready", progress=max(int(job.get("progress") or 0), 45))
    elif "acejam agents produced" in lower:
        updates.update(stage="plan_ready", status="MLX Media plan ready", current_task="Album plan ready", progress=95, waiting_on_llm=False)

    return updates


def _album_job_worker(job_id: str, body: dict[str, Any]) -> None:
    body = _album_ace_lm_disabled_payload(body)
    planner_provider = _album_planner_provider_from_payload(body)
    embedding_provider = _embedding_provider_from_payload(body)
    planner_model = str(body.get("planner_model") or body.get("ollama_model") or body.get("planner_ollama_model") or (DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL if planner_provider == "ollama" else ""))
    embedding_model = str(body.get("embedding_model") or DEFAULT_ALBUM_EMBEDDING_MODEL)
    planning_engine = _normalize_album_agent_engine_value(body.get("agent_engine"))
    planning_engine_label = _album_agent_engine_label_value(planning_engine)
    album_writer_mode = str(body.get("album_writer_mode") or "per_track_writer_loop").strip() or "per_track_writer_loop"
    direct_render_tracks = _json_list(body.get("tracks") or body.get("planned_tracks"))
    direct_existing_render = bool(direct_render_tracks) and (
        parse_bool(body.get("render_from_existing_tracks"), False)
        or parse_bool(body.get("skip_album_planning"), False)
        or str(body.get("album_generation_mode") or "").strip().lower()
        in {"render_existing_tracks", "direct_render", "ui_tracks"}
    )
    started = datetime.now(timezone.utc).isoformat()
    _set_album_job(
        job_id,
        state="running",
        status="Rendering approved UI tracks" if direct_existing_render else f"Running {planning_engine_label} album planner",
        stage="render_existing_tracks" if direct_existing_render else "planning",
        current_task="Render existing Album Wizard tracks" if direct_existing_render else "Planning album",
        progress=1,
        started_at=started,
        finished_at=None,
        payload=body,
        planner_model=planner_model,
        planner_provider=planner_provider,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        planning_engine="existing_ui_tracks" if direct_existing_render else planning_engine,
        album_writer_mode="render_existing_tracks" if direct_existing_render else album_writer_mode,
        custom_agents_used=False if direct_existing_render else True,
        crewai_used=False if direct_existing_render else planning_engine == "crewai_micro",
        memory_enabled=False,
        logs=[
            f"Album job {job_id} started.",
            (
                f"Direct render: {len(direct_render_tracks)} UI-approved track(s); no album agents will run."
                if direct_existing_render
                else f"Planning Engine: {planning_engine_label} ({planning_engine})"
            ),
            f"Album writer mode: {'render_existing_tracks' if direct_existing_render else album_writer_mode}",
            (
                "Local AI Writer/Planner skipped for Generate; existing UI tracks are the source of truth."
                if direct_existing_render
                else f"Local AI Writer/Planner: {provider_label(planner_provider)} ({planner_model})"
            ),
            (
                "Album memory embedding skipped for Generate."
                if direct_existing_render
                else f"Album memory embedding: {provider_label(embedding_provider)} ({embedding_model}); hidden unless memory/debug is enabled."
            ),
            "ACE-Step Audio Models render final music after local text/settings planning.",
            "ACE-Step LM disabled for album agents.",
            (
                "Agent memory skipped; deterministic render debug logs are job-scoped."
                if direct_existing_render
                else "Agent memory: pending embedding preflight; deterministic debug logs are job-scoped."
            ),
        ],
    )
    try:
        track_duration = parse_duration_seconds(body.get("track_duration") or body.get("duration") or 180.0, 180.0)
        num_tracks = clamp_int(body.get("num_tracks"), 7, 1, 40)
        lora_request = _lora_adapter_request(body)
        if lora_request.get("use_lora") and lora_request.get("adapter_song_model"):
            body = dict(body)
            adapter_song_model = str(lora_request.get("adapter_song_model") or "").strip()
            if adapter_song_model:
                body["song_model"] = adapter_song_model
                body["requested_song_model"] = adapter_song_model
                body["song_model_strategy"] = "single_model_album"
                _set_album_job(
                    job_id,
                    status=(
                        f"Rendering album with LoRA model lock: {adapter_song_model}"
                        if direct_existing_render
                        else f"Planning album with LoRA model lock: {adapter_song_model}"
                    ),
                    logs=[
                        *(_album_jobs.get(job_id, {}).get("logs") or []),
                        f"LoRA adapter requires {adapter_song_model}; album model strategy locked to single_model_album.",
                    ],
                )
        strategy = str(body.get("song_model_strategy") or "all_models_album")
        expected_models = album_models_for_strategy(
            strategy,
            _installed_acestep_models(),
            str(body.get("requested_song_model") or body.get("song_model") or ""),
        )
        expected_count = len(expected_models) * num_tracks
        _set_album_job(
            job_id,
            expected_count=expected_count,
            status="Rendering album tracks" if direct_existing_render else "Planning album",
            progress=2,
        )
        request_body = dict(body)
        request_body["album_job_id"] = job_id
        request_body["planner_lm_provider"] = planner_provider
        request_body["embedding_lm_provider"] = embedding_provider
        request_body["planner_model"] = planner_model
        request_body["album_writer_mode"] = album_writer_mode
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
            ace_lm_model="none",
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
            album_writer_mode=str(result.get("album_writer_mode") or album_writer_mode),
            memory_enabled=bool(result.get("memory_enabled") or ((result.get("toolkit_report") or {}).get("memory") or {}).get("enabled")) if isinstance(result.get("toolkit_report"), dict) else bool(result.get("memory_enabled")),
            context_chunks=int(result.get("context_chunks") or ((result.get("toolkit_report") or {}).get("memory") or {}).get("context_chunks") or 0) if isinstance(result.get("toolkit_report"), dict) else int(result.get("context_chunks") or 0),
            retrieval_rounds=int(result.get("retrieval_rounds") or ((result.get("toolkit_report") or {}).get("memory") or {}).get("retrieval_rounds") or 0) if isinstance(result.get("toolkit_report"), dict) else int(result.get("retrieval_rounds") or 0),
            agent_context_store=str(result.get("agent_context_store") or (((result.get("toolkit_report") or {}).get("memory") or {}).get("context_store") if isinstance(result.get("toolkit_report"), dict) else "") or ""),
            context_store_index=str(result.get("context_store_index") or ""),
            sequence_repair_count=int(result.get("sequence_repair_count") or 0),
            sequence_report=result.get("sequence_report") or {},
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
    body = _album_prepare_contract_request_body(_album_ace_lm_disabled_payload(body), fallback_tracks=5)
    concept = _recover_album_request_concept(body.get("concept") or "", body)
    num_tracks = int(body.get("num_tracks") or 5)
    track_duration = parse_duration_seconds(body.get("track_duration") or body.get("duration") or 180.0, 180.0)
    language = str(body.get("language") or "en")
    song_model = str(body.get("song_model") or "auto")
    planner_provider = _album_planner_provider_from_payload(body)
    embedding_provider = _embedding_provider_from_payload(body)
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
    _ensure_album_agent_modules_current()
    from album_crew import plan_album as _plan_album

    result = _plan_album(
        concept=concept,
        num_tracks=num_tracks,
        track_duration=track_duration,
        ollama_model=planner_model,
        language=language,
        embedding_model=embedding_model,
        options=options,
        use_crewai=True,
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
    body = _album_prepare_contract_request_body(_album_ace_lm_disabled_payload(body), fallback_tracks=5)
    planner_provider = _album_planner_provider_from_payload(body)
    embedding_provider = _embedding_provider_from_payload(body)
    planner_model = str(body.get("planner_model") or body.get("ollama_model") or body.get("planner_ollama_model") or DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL)
    embedding_model = str(body.get("embedding_model") or DEFAULT_ALBUM_EMBEDDING_MODEL)
    toolbelt_only = False
    requested_engine = _normalize_album_agent_engine_value(body.get("agent_engine"))
    requested_engine_label = _album_agent_engine_label_value(requested_engine)
    crewai_output_log_file = str(body.get("crewai_output_log_file") or "")
    user_album_contract = body.get("user_album_contract")
    if not isinstance(user_album_contract, dict):
        user_album_contract = extract_user_album_contract(
            _album_contract_source_from_payload(body, body),
            int(body.get("num_tracks") or 0) or None,
            str(body.get("language") or "en"),
            body,
        )
    num_tracks = clamp_int(body.get("num_tracks") or len(_json_list(body.get("tracks"))) or 5, 5, 1, 40)
    start_logs = [
        f"Album plan job {job_id} started.",
        f"Planning engine requested: {requested_engine_label} ({requested_engine}).",
        f"Planning Engine: {requested_engine_label} ({requested_engine})",
        f"Prompt Kit: {PROMPT_KIT_VERSION}.",
        f"Local AI Writer/Planner: {provider_label(planner_provider)} ({planner_model})",
        f"Album memory embedding: {provider_label(embedding_provider)} ({embedding_model}); hidden unless memory/debug is enabled.",
        "ACE-Step LM disabled for album agents.",
        "Agent memory: pending embedding preflight; prompts/responses go to the job debug folder.",
    ]
    if user_album_contract.get("applied"):
        start_logs.append(
            "Input Contract: applied; "
            f"locked_tracks={len(user_album_contract.get('tracks') or [])}; "
            f"blocked_unsafe={int(user_album_contract.get('blocked_unsafe_count') or 0)}"
        )
    if crewai_output_log_file:
        start_logs.append(f"Legacy CrewAI output log file requested; selected planner writes standard MLX Media debug JSONL: {crewai_output_log_file}")
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
        planning_engine=requested_engine,
        custom_agents_used=True,
        crewai_used=requested_engine == "crewai_micro",
        prompt_kit_version=PROMPT_KIT_VERSION,
        input_contract=contract_prompt_context(user_album_contract),
        input_contract_applied=bool(user_album_contract.get("applied")),
        input_contract_version=USER_ALBUM_CONTRACT_VERSION,
        blocked_unsafe_count=int(user_album_contract.get("blocked_unsafe_count") or 0),
        memory_enabled=False,
        stage="queued",
        current_task="Queued album planning job",
        total_tracks=num_tracks,
        current_track=0,
        completed_tracks=0,
        remaining_tracks=num_tracks,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        logs=start_logs,
    )
    try:
        wait_warn_seconds = max(10.0, _env_float("ACEJAM_ALBUM_PLAN_LLM_WAIT_WARN_SECONDS", 45.0))
        planner_timeout_seconds = float(
            planner_llm_settings_from_payload(
                body,
                default_max_tokens=8192,
                default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
            ).get("planner_timeout")
            or PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS
        )
        default_hard_timeout = max(PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS, planner_timeout_seconds + 300.0)
        hard_timeout_seconds = max(
            0.0,
            _env_float("ACEJAM_ALBUM_PLAN_LLM_HARD_TIMEOUT_SECONDS", default_hard_timeout),
        )
        _album_job_log(
            job_id,
            f"Album LLM wait policy: warn_after={int(wait_warn_seconds)}s; hard_timeout={int(hard_timeout_seconds)}s.",
            planner_timeout=planner_timeout_seconds,
            llm_hard_timeout_seconds=hard_timeout_seconds,
        )
        wait_state: dict[str, Any] = {"running": True, "waiting": False, "started": 0.0, "agent": "", "provider": "", "timed_out": False}

        def _wait_watchdog() -> None:
            last_reported_bucket = -1
            while wait_state.get("running"):
                time.sleep(5.0)
                if not wait_state.get("waiting"):
                    continue
                started_at = float(wait_state.get("started") or 0.0)
                if started_at <= 0:
                    continue
                elapsed = max(0.0, time.time() - started_at)
                bucket = int(elapsed // 15)
                if elapsed >= wait_warn_seconds and bucket != last_reported_bucket:
                    last_reported_bucket = bucket
                    snapshot = _album_job_snapshot(job_id)
                    if isinstance(snapshot, dict) and snapshot.get("state") in {"failed", "succeeded"}:
                        continue
                    _set_album_job(
                        job_id,
                        state="running",
                        stage="waiting_on_llm",
                        status=f"Waiting on {wait_state.get('provider') or 'LLM'}: {wait_state.get('agent') or 'album agent'}",
                        current_task=f"Still waiting for {wait_state.get('agent') or 'album agent'} ({int(elapsed)}s)",
                        current_agent=str(wait_state.get("agent") or ""),
                        waiting_on_llm=True,
                        llm_provider=str(wait_state.get("provider") or ""),
                        llm_wait_elapsed_s=int(elapsed),
                        logs=[f"Waiting on {wait_state.get('provider') or 'LLM'} {wait_state.get('agent') or 'album agent'} for {int(elapsed)}s."],
                    )
                if hard_timeout_seconds and elapsed >= hard_timeout_seconds and not wait_state.get("timed_out"):
                    wait_state["timed_out"] = True
                    _set_album_job(
                        job_id,
                        state="failed",
                        stage="llm_timeout",
                        status="Album AI timed out waiting on LLM",
                        current_task=f"{wait_state.get('agent') or 'Album agent'} exceeded {int(hard_timeout_seconds)}s",
                        waiting_on_llm=False,
                        llm_wait_elapsed_s=int(elapsed),
                        timed_out=True,
                        errors=[f"Album AI call timed out after {int(elapsed)}s waiting on {wait_state.get('provider') or 'LLM'} {wait_state.get('agent') or 'album agent'}."],
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    )
                    return

        threading.Thread(target=_wait_watchdog, daemon=True).start()

        def _stream_plan_log(line: str) -> None:
            compact = str(line).replace("\n", " ")[:700]
            job = _album_job_snapshot(job_id)
            if isinstance(job, dict) and job.get("timed_out"):
                print(f"[album_plan_job][{job_id}] {compact}", file=sys.__stdout__, flush=True)
                _set_album_job(job_id, logs=[compact])
                return
            updates = _album_plan_progress_updates_from_log(compact, job if isinstance(job, dict) else {})
            if updates.get("waiting_on_llm"):
                wait_state.update(
                    waiting=True,
                    started=time.time(),
                    agent=str(updates.get("current_agent") or ""),
                    provider=str(updates.get("llm_provider") or ""),
                )
            elif "crewai micro agent response:" in compact.lower() or "crewai micro agent parsed" in compact.lower():
                wait_state.update(waiting=False, started=0.0)
            print(f"[album_plan_job][{job_id}] {compact}", file=sys.__stdout__, flush=True)
            _album_job_log(job_id, compact, **updates)

        result = _run_album_plan_from_payload(body, log_callback=_stream_plan_log)
        wait_state["running"] = False
        wait_state["waiting"] = False
        snapshot = _album_job_snapshot(job_id)
        if isinstance(snapshot, dict) and snapshot.get("timed_out"):
            return
        if isinstance(snapshot, dict) and snapshot.get("stop_requested"):
            _set_album_job(
                job_id,
                state="stopped",
                stage="stopped",
                status="Album plan stopped",
                current_task="Album plan stopped after current LLM call returned",
                waiting_on_llm=False,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            return
        tracks = result.get("tracks") or []
        success = bool(result.get("success", True)) and bool(tracks)
        _set_album_job(
            job_id,
            state="succeeded" if success else "failed",
            status="Album plan ready" if success else "Album plan failed",
            progress=100,
            stage="complete" if success else "failed",
            current_task="Album plan ready" if success else "Album plan failed",
            waiting_on_llm=False,
            result=result,
            logs=result.get("logs") or [],
            errors=[] if success else [result.get("error") or "Album planning failed"],
            planner_model=str(result.get("planner_model") or planner_model),
            embedding_model=str(result.get("embedding_model") or embedding_model),
            planning_engine=str(result.get("planning_engine") or requested_engine.lower()),
            custom_agents_used=bool(result.get("custom_agents_used")),
            crewai_used=bool(result.get("crewai_used")),
            toolbelt_fallback=bool(result.get("toolbelt_fallback")),
            crewai_output_log_file=str(result.get("crewai_output_log_file") or crewai_output_log_file),
            agent_debug_dir=str(result.get("agent_debug_dir") or ""),
            agent_repair_count=int(result.get("agent_repair_count") or 0),
            agent_rounds=result.get("agent_rounds") or [],
            memory_enabled=bool(result.get("memory_enabled") or ((result.get("toolkit_report") or {}).get("memory") or {}).get("enabled")) if isinstance(result.get("toolkit_report"), dict) else bool(result.get("memory_enabled")),
            context_chunks=int(result.get("context_chunks") or ((result.get("toolkit_report") or {}).get("memory") or {}).get("context_chunks") or 0) if isinstance(result.get("toolkit_report"), dict) else int(result.get("context_chunks") or 0),
            retrieval_rounds=int(result.get("retrieval_rounds") or ((result.get("toolkit_report") or {}).get("memory") or {}).get("retrieval_rounds") or 0) if isinstance(result.get("toolkit_report"), dict) else int(result.get("retrieval_rounds") or 0),
            agent_context_store=str(result.get("agent_context_store") or (((result.get("toolkit_report") or {}).get("memory") or {}).get("context_store") if isinstance(result.get("toolkit_report"), dict) else "") or ""),
            context_store_index=str(result.get("context_store_index") or ""),
            sequence_repair_count=int(result.get("sequence_repair_count") or 0),
            sequence_report=result.get("sequence_report") or {},
            prompt_kit_version=str(result.get("prompt_kit_version") or PROMPT_KIT_VERSION),
            input_contract=result.get("input_contract") or contract_prompt_context(user_album_contract),
            input_contract_applied=bool(result.get("input_contract_applied") or user_album_contract.get("applied")),
            input_contract_version=str(result.get("input_contract_version") or USER_ALBUM_CONTRACT_VERSION),
            blocked_unsafe_count=int(result.get("blocked_unsafe_count") or user_album_contract.get("blocked_unsafe_count") or 0),
            contract_repair_count=int(result.get("contract_repair_count") or 0),
            expected_count=len(tracks),
            planned_count=len(tracks),
            total_tracks=len(tracks) or num_tracks,
            completed_tracks=len(tracks) if success else int(snapshot.get("completed_tracks") or 0) if isinstance(snapshot, dict) else 0,
            remaining_tracks=0 if success else max(0, num_tracks - (int(snapshot.get("completed_tracks") or 0) if isinstance(snapshot, dict) else 0)),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except OllamaPullStarted as exc:
        wait_state["running"] = False
        wait_state["waiting"] = False
        _set_album_job(
            job_id,
            state="waiting_on_model",
            job_type="album_plan",
            status=f"Downloading planner model: {exc.model_name}",
            progress=max(1, int((_jsonable(exc.job) or {}).get("progress") or 1)),
            stage="model_download",
            current_task=f"Planner-model downloaden: {exc.model_name}",
            current_agent="Local LLM install",
            waiting_on_llm=False,
            ollama_pull_job=_jsonable(exc.job),
            errors=[],
            logs=[exc.message],
        )
    except Exception as exc:
        wait_state["running"] = False
        wait_state["waiting"] = False
        snapshot = _album_job_snapshot(job_id)
        if isinstance(snapshot, dict) and snapshot.get("timed_out"):
            return
        _set_album_job(
            job_id,
            state="failed",
            job_type="album_plan",
            status="Album plan failed",
            progress=100,
            stage="failed",
            current_task="Album plan failed",
            waiting_on_llm=False,
            errors=[str(exc)],
            logs=[traceback.format_exc()],
            crewai_output_log_file=crewai_output_log_file,
            prompt_kit_version=PROMPT_KIT_VERSION,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        try:
            wait_state["running"] = False
        except Exception:
            pass
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


@app.post("/api/ace-step/vendor-sync")
async def api_ace_step_vendor_sync(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(_official_vendor_sync(body if isinstance(body, dict) else {}))


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
        body = _album_ace_lm_disabled_payload(await request.json())
        result = _run_album_plan_from_payload(body)
        return JSONResponse(result, status_code=200 if result.get("success", True) else 400)
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "album planning"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "logs": [str(exc)]}, status_code=400)


@app.post("/api/album/plan/jobs")
async def api_create_album_plan_job(request: Request):
    try:
        body = _album_prepare_contract_request_body(_album_ace_lm_disabled_payload(await request.json()), fallback_tracks=5)
        global_llm_settings = _load_local_llm_settings_fast()
        planner_provider = _album_planner_provider_from_payload(body)
        embedding_provider = _embedding_provider_from_payload(body)
        planner_model = str(
            body.get("planner_model")
            or body.get("planner_ollama_model")
            or body.get("ollama_model")
            or global_llm_settings.get("chat_model")
            or (DEFAULT_ALBUM_PLANNER_OLLAMA_MODEL if planner_provider == "ollama" else "")
            or ""
        ).strip()
        embedding_model = str(
            body.get("embedding_model")
            or global_llm_settings.get("embedding_model")
            or (DEFAULT_ALBUM_EMBEDDING_MODEL if embedding_provider == "ollama" else "")
            or ""
        )
        job_id = uuid.uuid4().hex[:12]
        planning_engine = _normalize_album_agent_engine_value(body.get("agent_engine"))
        planning_engine_label = _album_agent_engine_label_value(planning_engine)
        request_body = {
            **body,
            "agent_engine": planning_engine,
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
            planning_engine=planning_engine,
            custom_agents_used=True,
            crewai_used=planning_engine == "crewai_micro",
            memory_enabled=False,
            stage="queued",
            current_task="Queued album planning job",
            total_tracks=int(body.get("num_tracks") or 5),
            current_track=0,
            completed_tracks=0,
            remaining_tracks=int(body.get("num_tracks") or 5),
            expected_count=int(body.get("num_tracks") or 5),
            logs=[
                f"Queued album plan job {job_id}.",
                f"Planning Engine: {planning_engine_label} ({planning_engine})",
                "ACE-Step LM disabled for album agents.",
            ],
        )
        threading.Thread(target=_album_plan_job_worker, args=(job_id, request_body), daemon=True).start()
        return JSONResponse({"success": True, "job_id": job_id, "job": _album_job_snapshot(job_id)})
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "album planning"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "logs": [str(exc)]}, status_code=400)


@app.get("/api/community")
async def api_community():
    return JSONResponse(_community_feed(100))


@app.get("/api/library")
async def api_library(limit: int = 500):
    return JSONResponse(_library_items(limit=limit))


@app.post("/api/library/delete")
async def api_library_delete(request: Request):
    try:
        body = await request.json()
        return JSONResponse(_delete_library_item(body if isinstance(body, dict) else {}))
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


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


def _ace_step_lm_catalog() -> dict[str, Any]:
    lm_models = sorted(_installed_lm_models())
    writer_models = [name for name in lm_models if name not in {"auto", "none"}]
    details = []
    for name in writer_models:
        profile = lm_model_profiles_for_models([name], set(lm_models)).get(name, {})
        details.append(
            {
                "name": name,
                "model": name,
                "provider": "ace_step_lm",
                "kind": "chat",
                "type": "official_ace_step_5hz_lm",
                "format": "mlx" if _IS_APPLE_SILICON else "pt",
                "capabilities": ["audio_understanding", "composition", "metadata", "caption", "language"],
                "mlx_preferred": _IS_APPLE_SILICON,
                "loaded": False,
                "status": profile.get("status") or ("installed" if name in lm_models else "download_required"),
                "profile": _jsonable(profile),
            }
        )
    return {
        "success": True,
        "ready": bool(writer_models),
        "provider": "ace_step_lm",
        "provider_label": "ACE-Step 5Hz LM",
        "host": "local official runner",
        "models": writer_models,
        "chat_models": writer_models,
        "embedding_models": [],
        "image_models": [],
        "details": details,
        "loaded_models": [],
        "running_models": [],
        "error": "" if writer_models else "No ACE-Step 5Hz LM model is installed.",
    }


def _combined_local_llm_catalog(enrich: bool = False) -> dict[str, Any]:
    settings = _load_local_llm_settings()
    ollama = _ollama_model_catalog(enrich=enrich)
    lmstudio = lmstudio_model_catalog()
    ace_writer = _ace_step_lm_catalog()
    catalogs = {
        "ollama": ollama,
        "lmstudio": lmstudio,
        "ace_step_lm": ace_writer,
    }
    all_details: list[dict[str, Any]] = []
    for provider, catalog in catalogs.items():
        for item in catalog.get("details") or []:
            row = dict(item)
            row.setdefault("provider", provider)
            row["key"] = f"{provider}:{row.get('name') or row.get('model')}"
            row.setdefault("display_name", row.get("name") or row.get("model") or row["key"])
            if provider == "lmstudio" and str(row.get("format") or "").lower() == "mlx":
                row["mlx_preferred"] = True
            if provider == "ollama" and _IS_APPLE_SILICON:
                row.setdefault("mlx_hint", "Ollama uses Apple GPU acceleration when the installed model/backend supports it.")
            all_details.append(row)
    return {
        "success": True,
        "settings": settings,
        "mlx": {
            "apple_silicon": _IS_APPLE_SILICON,
            "ace_step_lm_backend": ACE_LM_BACKEND_DEFAULT,
            "policy": settings.get("mlx_policy", "auto"),
            "label": "Full MLX" if _IS_APPLE_SILICON else "Auto",
        },
        "providers": [
            {"id": "ollama", "label": "Ollama", "host": ollama.get("ollama_host") or _ollama_host(), "ready": bool(ollama.get("ready"))},
            {"id": "lmstudio", "label": "LM Studio", "host": lmstudio.get("host", ""), "ready": bool(lmstudio.get("ready"))},
            {"id": "ace_step_lm", "label": "ACE-Step 5Hz LM", "host": "local official runner", "ready": bool(ace_writer.get("ready"))},
        ],
        "catalogs": catalogs,
        "details": all_details,
        "models": [item["key"] for item in all_details],
        "chat_models": [item["key"] for item in all_details if item.get("kind") == "chat"],
        "embedding_models": [item["key"] for item in all_details if item.get("kind") == "embedding"],
        "image_models": [],
    }


@app.get("/api/local-llm/settings")
async def api_local_llm_settings():
    return JSONResponse({"success": True, "settings": _load_local_llm_settings()})


@app.post("/api/local-llm/settings")
async def api_save_local_llm_settings(request: Request):
    try:
        body = await request.json()
        settings = _save_local_llm_settings(body if isinstance(body, dict) else {})
        return JSONResponse({"success": True, "settings": settings})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/local-llm/catalog")
async def api_local_llm_catalog(enrich: bool = False):
    return JSONResponse(_combined_local_llm_catalog(enrich=enrich))


@app.get("/api/local-llm/providers")
async def api_local_llm_providers():
    catalog = _combined_local_llm_catalog(enrich=False)
    return JSONResponse(
        {
            "success": True,
            "default_provider": catalog.get("settings", {}).get("provider") or "ollama",
            "settings": catalog.get("settings", {}),
            "providers": catalog.get("providers", []),
        }
    )


@app.get("/api/local-llm/models")
async def api_local_llm_models(provider: str = "ollama", enrich: bool = False):
    provider_name = normalize_provider(provider)
    if provider_name == "ace_step_lm":
        return JSONResponse(_ace_step_lm_catalog())
    if provider_name == "ollama":
        data = _ollama_model_catalog(enrich=enrich)
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
        if provider == "ace_step_lm":
            requested = _concrete_lm_model_or_download(model or "auto", "ACE-Step 5Hz LM writer test")
            return JSONResponse(
                {
                    "success": True,
                    "provider": "ace_step_lm",
                    "provider_label": "ACE-Step 5Hz LM",
                    "model": requested,
                    "kind": "chat",
                    "response": "OK",
                }
            )
        if provider == "ollama":
            _ensure_ollama_model_or_start_pull(model, context=f"{kind} test", kind="embedding" if kind == "embedding" else "chat")
        planner_settings = planner_llm_settings_from_payload(body, default_max_tokens=16, default_timeout=60.0)
        return JSONResponse(local_llm_test_model(provider, model, kind, planner_settings))
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
        kind = str(body.get("kind") or _ollama_kind_from_model_name(model_name)).strip().lower()
        if provider == "ace_step_lm":
            job = _start_model_download(model_name or ACE_LM_PREFERRED_MODEL)
            return JSONResponse({"success": True, "provider": "ace_step_lm", "job": job})
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


@app.post("/api/art/generate")
async def api_generate_art(request: Request):
    return JSONResponse(
        {"success": False, "error": "Ollama image generation is disabled in MLX Media. Use the MFLUX Image Studio instead."},
        status_code=410,
    )


@app.get("/api/mflux/status")
async def api_mflux_status(check_help: bool = False):
    return JSONResponse(mflux_status(check_help=check_help))


@app.get("/api/mflux/models")
async def api_mflux_models():
    payload = mflux_models()
    payload["status"] = mflux_status(check_help=False)
    return JSONResponse(payload)


@app.get("/api/mflux/jobs")
async def api_mflux_jobs():
    return JSONResponse({"success": True, "jobs": mflux_list_jobs()})


@app.post("/api/mflux/jobs")
async def api_mflux_jobs_create(request: Request):
    try:
        body = await request.json()
        job = mflux_create_job(body if isinstance(body, dict) else {})
        return JSONResponse({"success": True, "job_id": job.get("id"), "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/mflux/jobs/{job_id}")
async def api_mflux_job(job_id: str):
    job = mflux_get_job(job_id)
    if not job:
        return JSONResponse({"success": False, "error": "MFLUX job not found."}, status_code=404)
    return JSONResponse({"success": True, "job": job})


@app.post("/api/mflux/lora/train")
async def api_mflux_lora_train(request: Request):
    try:
        body = await request.json()
        job = mflux_start_lora_training(body if isinstance(body, dict) else {})
        return JSONResponse({"success": True, "job_id": job.get("id"), "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/mflux/lora/adapters")
async def api_mflux_lora_adapters():
    return JSONResponse({"success": True, "adapters": mflux_list_lora_adapters()})


@app.post("/api/mflux/uploads")
async def api_mflux_upload_image(file: UploadFile = File(...)):
    original_name = file.filename or "image.png"
    suffix = Path(original_name).suffix.lower()
    if suffix not in MFLUX_ALLOWED_IMAGE_EXTENSIONS:
        return JSONResponse({"success": False, "error": f"Unsupported image file: {suffix or '(none)'}"}, status_code=400)
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = MFLUX_UPLOADS_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_filename(Path(original_name).stem, 'image')}{suffix}"
    target = upload_dir / filename
    target.write_bytes(await file.read())
    return JSONResponse(
        {
            "success": True,
            "upload_id": upload_id,
            "id": upload_id,
            "filename": filename,
            "path": str(target),
            "url": mflux_public_upload_url(upload_id, filename),
        }
    )


def _mflux_art_metadata(result_id: str, scope: str = "mflux") -> dict[str, Any]:
    rid = safe_id(result_id)
    meta_path = _resolve_child(MFLUX_RESULTS_DIR, rid, "mflux_result.json")
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="MFLUX result not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    filename = str(meta.get("filename") or "")
    if not filename:
        raise HTTPException(status_code=404, detail="MFLUX result image missing")
    return {
        "art_id": f"mflux-{rid}",
        "mflux_result_id": rid,
        "filename": filename,
        "path": str(_resolve_child(MFLUX_RESULTS_DIR, rid, filename)),
        "url": str(meta.get("image_url") or meta.get("url") or f"/media/mflux/{rid}/{filename}"),
        "scope": scope,
        "prompt": str(meta.get("prompt") or ""),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "model": str(meta.get("model_label") or meta.get("model_id") or "MFLUX"),
        "created_at": meta.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "source": "mflux",
    }


@app.post("/api/mflux/art/attach")
async def api_mflux_art_attach(request: Request):
    try:
        body = await request.json()
        result_id = str(body.get("source_result_id") or body.get("mflux_result_id") or body.get("result_id") or "").strip()
        target_type = str(body.get("target_type") or body.get("scope") or "").strip().lower()
        target_id = str(body.get("target_id") or body.get("song_id") or body.get("album_id") or body.get("album_family_id") or "").strip()
        art = _mflux_art_metadata(result_id, scope=target_type or "mflux")
        if target_type in {"result", "generation", "generation_result"} and target_id:
            _attach_art_to_result(target_id, art)
        elif target_type == "song" and target_id:
            _merge_song_album_metadata(target_id, {"art": art, "single_art": art})
        elif target_type in {"album", "album_family"} and target_id:
            _attach_art_to_album_family(target_id, art)
        return JSONResponse({"success": True, "art": art, "target_type": target_type, "target_id": target_id})
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/mlx-video/status")
async def api_mlx_video_status():
    return JSONResponse(mlx_video_status())


@app.get("/api/mlx-video/models")
async def api_mlx_video_models():
    payload = mlx_video_models()
    payload["status"] = mlx_video_status()
    return JSONResponse(payload)


@app.get("/api/mlx-video/jobs")
async def api_mlx_video_jobs():
    return JSONResponse({"success": True, "jobs": mlx_video_list_jobs()})


@app.post("/api/mlx-video/jobs")
async def api_mlx_video_jobs_create(request: Request):
    try:
        body = await request.json()
        job = mlx_video_create_job(body if isinstance(body, dict) else {})
        return JSONResponse({"success": True, "job_id": job.get("id"), "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/mlx-video/jobs/{job_id}")
async def api_mlx_video_job(job_id: str):
    job = mlx_video_get_job(job_id)
    if not job:
        return JSONResponse({"success": False, "error": "MLX video job not found."}, status_code=404)
    return JSONResponse({"success": True, "job": job})


@app.post("/api/mlx-video/uploads")
async def api_mlx_video_upload(file: UploadFile = File(...)):
    original_name = file.filename or "media.bin"
    suffix = Path(original_name).suffix.lower()
    if suffix not in MLX_VIDEO_ALLOWED_UPLOAD_EXTENSIONS:
        return JSONResponse({"success": False, "error": f"Unsupported video studio file: {suffix or '(none)'}"}, status_code=400)
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = MLX_VIDEO_UPLOADS_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_filename(Path(original_name).stem, 'media')}{suffix}"
    target = upload_dir / filename
    target.write_bytes(await file.read())
    return JSONResponse(
        {
            "success": True,
            "upload_id": upload_id,
            "id": upload_id,
            "filename": filename,
            "path": str(target),
            "url": mlx_video_public_upload_url(upload_id, filename),
            "media_kind": "image" if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} else "audio" if suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"} else "video",
        }
    )


@app.get("/api/mlx-video/loras")
async def api_mlx_video_loras():
    return JSONResponse({"success": True, "adapters": mlx_video_list_loras()})


@app.post("/api/mlx-video/lora/train")
async def api_mlx_video_lora_train_unavailable():
    return JSONResponse(
        {
            "success": False,
            "available": False,
            "error": "Video-LoRA training is not available because upstream mlx-video does not expose a stable train command yet.",
        },
        status_code=501,
    )


@app.get("/api/mlx-video/attachments")
async def api_mlx_video_attachments(target_type: str | None = None, target_id: str | None = None):
    return JSONResponse(
        {
            "success": True,
            "attachments": mlx_video_list_attachments(target_type=target_type, target_id=target_id),
        }
    )


@app.post("/api/mlx-video/model-dirs")
async def api_mlx_video_register_model_dir(request: Request):
    try:
        body = await request.json()
        entry = mlx_video_register_model_dir(body if isinstance(body, dict) else {})
        return JSONResponse({"success": True, "entry": entry, "model_dirs": mlx_video_registered_model_dirs()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/mlx-video/attach")
async def api_mlx_video_attach(request: Request):
    try:
        body = await request.json()
        attachment = mlx_video_attach(body if isinstance(body, dict) else {})
        return JSONResponse({"success": True, "attachment": attachment})
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
        kind = str(body.get("kind") or _ollama_kind_from_model_name(model_name))
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
        _ensure_ollama_model_or_start_pull(model_name, context="model inspect", kind=_ollama_kind_from_model_name(model_name))
        data = _ollama_client().show(model_name)
        return JSONResponse({"success": True, "model": model_name, "details": _jsonable(data)})
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "model inspect"))
    except Exception as exc:
        if _ollama_error_is_missing_model(exc):
            model_name = str(body.get("model") or body.get("model_name") or "").strip()
            job = _start_ollama_pull(model_name, reason="model inspect", kind=_ollama_kind_from_model_name(model_name))
            return JSONResponse(_ollama_pull_started_payload(model_name, job, "model inspect"))
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/ollama/test")
async def api_ollama_test(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
        model_name = str(body.get("model") or body.get("model_name") or "").strip()
        kind = str(body.get("kind") or _ollama_kind_from_model_name(model_name))
        _ensure_ollama_model_or_start_pull(model_name, context=f"{kind} test", kind=kind)
        client = _ollama_client()
        if kind == "embedding":
            response = client.embed(model=model_name, input="MLX Media embedding test")
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
            kind = str(body.get("kind") or _ollama_kind_from_model_name(model_name))
            job = _start_ollama_pull(model_name, reason=f"{kind} test", kind=kind)
            return JSONResponse(_ollama_pull_started_payload(model_name, job, "Ollama test"))
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/prompt-assistant/prompts")
async def api_prompt_assistant_prompts():
    prompts = []
    for mode, info in PROMPT_ASSISTANT_MODES.items():
        if mode in PROMPT_ASSISTANT_DISABLED_MODES:
            continue
        path = BASE_DIR / "prompts" / info["file"]
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
        if mode == "trainer":
            return JSONResponse(
                {
                    "success": False,
                    "error": "Trainer uses the official ACE-Step dataset/LoRA workflow; Ollama/LM Studio AI Fill is disabled for Trainer.",
                    "raw_text": "",
                    "warnings": [],
                },
                status_code=400,
            )
        user_prompt = str(body.get("user_prompt") or body.get("prompt") or "").strip()
        if not user_prompt:
            return JSONResponse({"success": False, "error": "Prompt is empty.", "raw_text": ""}, status_code=400)
        current_payload = body.get("current_payload") if isinstance(body.get("current_payload"), dict) else {}
        planner_provider = _writer_provider_from_payload(body)
        global_llm_settings = _load_local_llm_settings()
        planner_model = str(
            body.get("planner_model")
            or body.get("planner_ollama_model")
            or body.get("ollama_model")
            or global_llm_settings.get("chat_model")
            or ""
        ).strip()
        planner_settings = planner_llm_settings_from_payload(
            {**global_llm_settings, **body},
            default_max_tokens=8192,
            default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
        )
        if planner_provider == "ace_step_lm":
            fallback_provider = normalize_provider(global_llm_settings.get("provider") or "ollama")
            if fallback_provider == "ace_step_lm":
                fallback_provider = "ollama"
            planner_provider = fallback_provider
            planner_model = str(global_llm_settings.get("chat_model") or planner_model or "").strip()
        # Album wizard fill goes through the CrewAI Micro Tasks director
        # instead of a single Ollama call. Each track field is filled by a
        # specialised agent (Topline Hook Writer, Tier-1 Lyric Writer, Sonic
        # Tags Engineer, etc.) via plan_album. The user can edit every field
        # afterwards; ACE-Step audio render runs separately track-by-track
        # when the user clicks Generate.
        if mode == "album":
            crew_result = _run_prompt_assistant_album_crew(
                body,
                user_prompt,
                current_payload,
                planner_provider,
                planner_model,
            )
            crew_payload = crew_result.get("payload") or {}
            crew_warnings = list(crew_result.get("warnings") or [])
            crew_paste_blocks = _server_paste_blocks_from_payload(crew_payload, mode)
            return JSONResponse(
                {
                    "success": bool(crew_result.get("success", False)),
                    "mode": mode,
                    "prompt_kit_version": PROMPT_KIT_VERSION,
                    "prompt_file": PROMPT_ASSISTANT_MODES[mode]["file"],
                    "payload": _jsonable(crew_payload),
                    "paste_blocks": crew_paste_blocks,
                    "warnings": crew_warnings,
                    "validation": None,
                    "raw_text": str(crew_result.get("raw_text") or ""),
                    "error": str(crew_result.get("error") or "") or None,
                }
            )
        system_prompt = _prompt_assistant_system_prompt(mode)
        raw_text = _run_prompt_assistant_local_staged(
            system_prompt,
            user_prompt,
            planner_provider,
            planner_model,
            current_payload,
            planner_settings,
            mode=mode,
        )
        parsed_payload, paste_blocks = _extract_prompt_assistant_json(raw_text, mode)
        parsed_payload, structured_warnings, structured_paste_blocks = _unwrap_prompt_assistant_structured_payload(parsed_payload)
        if structured_paste_blocks:
            paste_blocks = structured_paste_blocks
        payload, warnings = _normalize_prompt_assistant_payload(mode, parsed_payload, body)
        if structured_warnings:
            warnings.extend(structured_warnings)
        if not paste_blocks:
            paste_blocks = _server_paste_blocks_from_payload(payload, mode)
        validation = None
        if mode not in {"album", "trainer", "image", "video"}:
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
        if isinstance(exc, PromptAssistantStageError) and not raw_text:
            raw_text = exc.raw_text
        error_text = str(exc)
        if "JSON object was not closed" in error_text or "Expecting" in error_text and raw_text:
            error_text = "AI Fill response was truncated; increase Settings > Max output or shorten lyrics."
        return JSONResponse(
            {
                "success": False,
                "error": error_text,
                "raw_text": raw_text,
                "raw_preview": raw_text[:1200],
                "warnings": [],
            },
            status_code=400,
        )


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
        body = _album_ace_lm_disabled_payload(await request.json())
        provider = _writer_provider_from_payload(body)
        global_llm_settings = _load_local_llm_settings()
        planner_model = str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or global_llm_settings.get("chat_model") or "").strip()
        planner_settings = planner_llm_settings_from_payload(
            {**global_llm_settings, **body},
            default_max_tokens=8192,
            default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
        )
        raw = compose(
            description=str(body.get("description") or ""),
            audio_duration=float(body.get("audio_duration") or body.get("duration") or 60.0),
            composer_profile=str(body.get("composer_profile") or "auto"),
            instrumental=parse_bool(body.get("instrumental"), False),
            ollama_model=str(body.get("ollama_model") or ""),
            planner_lm_provider=provider,
            planner_model=planner_model,
            planner_llm_settings=planner_settings,
        )
        return JSONResponse(json.loads(raw))
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "compose"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/create_sample")
async def api_create_sample(request: Request):
    try:
        body = _album_ace_lm_disabled_payload(_apply_studio_lm_policy(await request.json()))
        global_llm_settings = _load_local_llm_settings()
        provider = _writer_provider_from_payload(body)
        planner_model = str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or global_llm_settings.get("chat_model") or "").strip()
        raw = compose(
            description=str(body.get("query") or body.get("description") or body.get("caption") or ""),
            audio_duration=float(body.get("duration") or 60.0),
            composer_profile="auto",
            instrumental=parse_bool(body.get("instrumental"), False),
            ollama_model=str(body.get("ollama_model") or ""),
            planner_lm_provider=provider,
            planner_model=planner_model,
            planner_llm_settings=planner_llm_settings_from_payload(
                {**global_llm_settings, **body},
                default_max_tokens=8192,
                default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
            ),
        )
        data = json.loads(raw)
        data["artist_name"] = normalize_artist_name(
            body.get("artist_name") or data.get("artist_name"),
            derive_artist_name(data.get("title") or "", body.get("description") or body.get("caption") or "", data.get("tags") or ""),
        )
        return JSONResponse({"success": True, "engine": provider, **data})
    except ModelDownloadStarted as exc:
        return JSONResponse(_download_started_payload(exc.model_name, exc.job))
    except OllamaPullStarted as exc:
        return JSONResponse(_ollama_pull_started_payload(exc.model_name, exc.job, "create sample"))
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/format_sample")
async def api_format_sample(request: Request):
    try:
        body = _album_ace_lm_disabled_payload(_apply_studio_lm_policy(await request.json()))
        raw = compose(
            description=str(body.get("caption") or body.get("description") or "custom song"),
            audio_duration=float(body.get("duration") or 60.0),
            composer_profile="auto",
            instrumental=parse_bool(body.get("instrumental"), False),
            ollama_model=str(body.get("ollama_model") or ""),
            planner_lm_provider=str(body.get("planner_lm_provider") or body.get("planner_provider") or "ollama"),
            planner_model=str(body.get("planner_model") or body.get("planner_ollama_model") or body.get("ollama_model") or ""),
            planner_llm_settings=planner_llm_settings_from_payload(
                body,
                default_max_tokens=2048,
                default_timeout=PLANNER_LLM_DEFAULT_TIMEOUT_SECONDS,
            ),
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


@app.get("/api/generation/jobs")
async def api_generation_jobs_list():
    return JSONResponse({"success": True, "jobs": _generation_job_snapshot()})


@app.post("/api/generation/jobs")
async def api_generation_jobs_start(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Generation payload must be an object")
        task = _submit_api_generation_task(payload)
        return JSONResponse({"success": True, "job_id": task["job_id"], "task_id": task["task_id"], "job": task["job"]})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/generation/jobs/{job_id}")
async def api_generation_job_detail(job_id: str):
    job = _generation_job_snapshot(safe_id(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return JSONResponse({"success": True, "job": job})


@app.get("/api/generation/jobs/{job_id}/log")
async def api_generation_job_log(job_id: str):
    job = _generation_job_snapshot(safe_id(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")
    logs = [str(item) for item in (job.get("logs") or []) if str(item)]
    return JSONResponse({"success": True, "log": "\n".join(logs), "logs": logs})


@app.get("/api/song-batches/jobs")
async def api_song_batch_jobs_list():
    jobs = _song_batch_snapshot()
    return JSONResponse({"success": True, "jobs": jobs})


@app.post("/api/song-batches/jobs")
async def api_song_batch_jobs_start(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Batch payload must be an object")
        task = _submit_song_batch_job(payload)
        return JSONResponse({"success": True, "job_id": task["job_id"], "job": task["job"]})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/song-batches/jobs/{job_id}")
async def api_song_batch_job_detail(job_id: str):
    job = _song_batch_snapshot(safe_id(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Song batch job not found")
    return JSONResponse({"success": True, "job": job})


@app.get("/api/song-batches/jobs/{job_id}/log")
async def api_song_batch_job_log(job_id: str):
    job = _song_batch_snapshot(safe_id(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Song batch job not found")
    logs = [str(item) for item in (job.get("logs") or []) if str(item)]
    return JSONResponse({"success": True, "log": "\n".join(logs), "logs": logs})


@app.get("/api/lora/benchmarks/jobs")
async def api_lora_benchmark_jobs_list():
    jobs = _lora_benchmark_snapshot()
    return JSONResponse({"success": True, "jobs": jobs})


@app.post("/api/lora/benchmarks/jobs")
async def api_lora_benchmark_jobs_start(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Benchmark payload must be an object")
        task = _submit_lora_benchmark_job(payload)
        return JSONResponse({"success": True, "job_id": task["job_id"], "job": task["job"]})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/lora/benchmarks/jobs/{job_id}")
async def api_lora_benchmark_job_detail(job_id: str):
    job = _lora_benchmark_snapshot(safe_id(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="LoRA benchmark job not found")
    return JSONResponse({"success": True, "job": job})


@app.get("/api/lora/benchmarks/jobs/{job_id}/log")
async def api_lora_benchmark_job_log(job_id: str):
    job = _lora_benchmark_snapshot(safe_id(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="LoRA benchmark job not found")
    logs = [str(item) for item in (job.get("logs") or []) if str(item)]
    return JSONResponse({"success": True, "log": "\n".join(logs), "logs": logs})


@app.post("/api/lora/benchmarks/jobs/{job_id}/stop")
async def api_lora_benchmark_job_stop(job_id: str):
    try:
        job = _stop_lora_benchmark_job(safe_id(job_id))
        return JSONResponse({"success": True, "job": job})
    except KeyError as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/benchmarks/jobs/{job_id}/rating")
async def api_lora_benchmark_job_rating(job_id: str, request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Rating payload must be an object")
        job = _rate_lora_benchmark_result(safe_id(job_id), payload)
        return JSONResponse({"success": True, "job": job})
    except KeyError as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/generate_advanced")
async def api_generate_advanced(request: Request):
    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
        result = await asyncio.to_thread(_run_advanced_generation, payload)
        return JSONResponse(result)
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
        _schedule_accelerator_cleanup("api_generate_advanced")


@app.post("/api/generate_portfolio")
async def api_generate_portfolio(request: Request):
    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
        result = await asyncio.to_thread(_run_model_portfolio_generation, payload)
        return JSONResponse(result)
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
        _schedule_accelerator_cleanup("api_generate_portfolio")


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
    try:
        body = await request.json() if request.headers.get("content-length") else {}
    except Exception:
        body = {}
    result_id_raw = str(body.get("result_id") or "").strip()
    if not result_id_raw:
        return JSONResponse(
            {"success": False, "error": "result_id is required"},
            status_code=400,
        )
    try:
        result_id = safe_id(result_id_raw)
        audio_id = str(body.get("audio_id") or "take-1")
        meta = _load_result_meta(result_id)
        extra = _result_extra_cache.get(result_id)
        if not extra:
            return JSONResponse(
                {"success": False, "error": "LRC cache expired; regenerate with Auto LRC enabled"},
                status_code=400,
            )
        index = max(0, int(audio_id.split("-")[-1]) - 1) if "-" in audio_id else 0
        params = meta.get("params", {})
        seed = int((meta.get("audios", [{}])[index].get("seed") or 42))
        lrc = _calculate_lrc(
            _extra_for_index(extra, index),
            float(params.get("duration") or 60),
            str(params.get("vocal_language") or "unknown"),
            int(params.get("inference_steps") or 8),
            seed,
        )
        _update_result_item(result_id, audio_id, "lrc", lrc)
        return JSONResponse(lrc)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/score")
async def api_score(request: Request):
    try:
        body = await request.json() if request.headers.get("content-length") else {}
    except Exception:
        body = {}
    result_id_raw = str(body.get("result_id") or "").strip()
    if not result_id_raw:
        return JSONResponse(
            {"success": False, "error": "result_id is required"},
            status_code=400,
        )
    try:
        result_id = safe_id(result_id_raw)
        audio_id = str(body.get("audio_id") or "take-1")
        meta = _load_result_meta(result_id)
        extra = _result_extra_cache.get(result_id)
        if not extra:
            return JSONResponse(
                {"success": False, "error": "Score cache expired; regenerate with Auto Score enabled"},
                status_code=400,
            )
        index = max(0, int(audio_id.split("-")[-1]) - 1) if "-" in audio_id else 0
        params = meta.get("params", {})
        seed = int((meta.get("audios", [{}])[index].get("seed") or 42))
        score = _calculate_score(
            _extra_for_index(extra, index),
            str(params.get("vocal_language") or "unknown"),
            int(params.get("inference_steps") or 8),
            seed,
        )
        _update_result_item(result_id, audio_id, "score", score)
        return JSONResponse(score)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


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
    genre: str = Form(""),
    genre_label_mode: str = Form("ai_auto"),
    genre_label_provider: str = Form(""),
    genre_label_model: str = Form(""),
    overwrite_existing_labels: str = Form("false"),
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
            genre=genre,
            genre_label_mode=genre_label_mode,
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
                "genre_label_mode": _normalize_training_genre_label_mode(genre_label_mode),
                "genre_label_provider": normalize_provider(genre_label_provider or _load_local_llm_settings().get("provider") or "ollama"),
                "genre_label_model": str(genre_label_model or _load_local_llm_settings().get("chat_model") or "").strip(),
                "overwrite_existing_labels": parse_bool(overwrite_existing_labels, False),
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
        cached = _MB_ARTIST_CACHE[key].get("tags", [])
        print(f"[autolabel] MusicBrainz cached: {artist_name} → {cached}", flush=True)
        return cached
    print(f"[autolabel] MusicBrainz lookup: {artist_name}...", flush=True)
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
    print(f"[autolabel] MusicBrainz result: {artist_name} → {genre_tags}", flush=True)
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


def _genius_slug(artist_name: str, song_title: str) -> str:
    slug = f"{artist_name} {song_title}".lower()
    slug = slug.replace("&", " and ")
    slug = slug.replace("’", "'").replace("`", "'")
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
    text = html_lib.unescape(text)
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


def _search_lyrics_online(artist: str, title: str) -> str:
    """Fetch lyrics from Genius. Returns lyrics text or empty string."""
    if not artist or not title:
        return ""
    print(f"[autolabel] Genius lyrics lookup: {artist} - {title}...", flush=True)

    for index, candidate_title in enumerate(_lyrics_title_candidates(title)):
        try:
            if index:
                print(f"[autolabel] Genius retry with title variant: {candidate_title}", flush=True)
            slug = _genius_slug(artist, candidate_title)
            url = f"https://genius.com/{slug}-lyrics"
            html = _fetch_genius_page(url)
            lyrics = _extract_genius_lyrics(html)
            if lyrics and len(lyrics) > 50:
                lines = len(lyrics.split("\n"))
                suffix = " (variant)" if index else ""
                print(f"[autolabel] Genius lyrics FOUND{suffix}: {len(lyrics)} chars, {lines} lines", flush=True)
                return lyrics
        except Exception as exc:
            if index == 0:
                print(f"[autolabel] Genius primary lookup failed: {exc}", flush=True)
            continue

    print(f"[autolabel] Genius lyrics NOT FOUND for: {artist} - {title}", flush=True)
    return ""


def _detect_bpm_key(audio_path: str) -> tuple[int | None, str]:
    """Detect BPM and musical key from audio file using scipy. No librosa needed."""
    import numpy as np

    print(f"[autolabel] Audio analysis: {Path(audio_path).name}...", flush=True)
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

        print(f"[autolabel] Audio result: BPM={bpm}, key={key}", flush=True)
        return bpm, key
    except Exception as exc:
        print(f"[autolabel] Audio analysis failed: {exc}", flush=True)
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


def _missing_vocal_lora_label(
    audio_path: Path,
    *,
    filename: str | None = None,
    caption: str = "",
    duration: float | None = None,
    bpm: Any = None,
    keyscale: str = "",
    label_source: str = "lyrics_missing",
    error: str = "",
    trigger_tag: str = "",
    tag_position: str = "prepend",
) -> dict[str, Any]:
    return {
        "path": str(audio_path),
        "filename": filename or audio_path.name,
        "caption": caption or audio_path.stem.replace("-", " ").replace("_", " "),
        "lyrics": "",
        "lyrics_status": "missing",
        "requires_review": True,
        "genre": "",
        "bpm": bpm,
        "keyscale": keyscale,
        "timesignature": "4",
        "language": "unknown",
        "duration": duration or 0,
        "is_instrumental": False,
        "label_source": label_source,
        "error": error,
        "trigger_tag": trigger_tag,
        "tag_position": tag_position,
    }


def _smart_autolabel_file(audio_path: Path, filename: str, trigger_tag: str = "", tag_position: str = "prepend") -> dict[str, Any]:
    """Auto-label a single audio file using MusicBrainz + online lyrics + audio analysis. No LLM needed."""
    artist, title = _parse_artist_title(filename)
    print(f"\n[autolabel] === {filename} ===", flush=True)
    print(f"[autolabel] Parsed: artist={artist!r}, title={title!r}", flush=True)

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
    if not has_vocals:
        label = _missing_vocal_lora_label(
            audio_path,
            filename=filename,
            caption=caption,
            duration=duration,
            bpm=bpm,
            keyscale=key,
            label_source="online_lyrics_missing",
            trigger_tag=trigger_tag,
            tag_position=tag_position,
        )
        label.update(
            {
                "genre": ", ".join(genre_tags[:3]) if genre_tags else "",
                "musicbrainz_tags": genre_tags,
                "musicbrainz_album": mb_info.get("album", ""),
                "musicbrainz_year": mb_info.get("year", ""),
            }
        )
        return label

    return {
        "path": str(audio_path),
        "filename": filename,
        "caption": caption,
        "lyrics": lyrics,
        "lyrics_status": "verified",
        "requires_review": False,
        "genre": ", ".join(genre_tags[:3]) if genre_tags else "",
        "bpm": bpm,
        "keyscale": key,
        "timesignature": "4",
        "language": language,
        "duration": duration,
        "is_instrumental": False,
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
        print(f"\n{'='*60}", flush=True)
        print(f"[autolabel] START: mode={mode}, files={len(files)}, trigger={trigger_tag!r}", flush=True)
        print(f"{'='*60}", flush=True)

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
            allow_instrumental = parse_bool(body.get("instrumental_training"), False) or str(body.get("training_target") or "").lower() in {"instrumental", "style", "style_only"}
            for item in files[: int(body.get("limit") or 24)]:
                path = Path(str(item.get("path") if isinstance(item, dict) else item)).expanduser()
                duration = None
                try:
                    info = sf.info(str(path))
                    duration = round(info.frames / info.samplerate, 2)
                except Exception:
                    pass
                caption = path.stem.replace("-", " ").replace("_", " ")
                if allow_instrumental:
                    labels.append({
                        "path": str(path), "filename": path.name,
                        "caption": caption,
                        "lyrics": "[Instrumental]", "lyrics_status": "present", "requires_review": False,
                        "genre": "", "bpm": None, "keyscale": "",
                        "timesignature": "4", "language": "instrumental", "duration": duration,
                        "is_instrumental": True, "label_source": "filename_fallback",
                        "trigger_tag": trigger_tag, "tag_position": tag_position,
                    })
                else:
                    labels.append(
                        _missing_vocal_lora_label(
                            path,
                            caption=caption,
                            duration=duration,
                            label_source="filename_fallback",
                            trigger_tag=trigger_tag,
                            tag_position=tag_position,
                        )
                    )
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
            existing_caption = (item.get("caption") if isinstance(item, dict) else "") or path.stem.replace("-", " ").replace("_", " ")
            existing_lyrics = str((item.get("lyrics") if isinstance(item, dict) else "") or "").strip()
            fallback = _missing_vocal_lora_label(
                path,
                caption=existing_caption,
                duration=duration or (item.get("duration") if isinstance(item, dict) else 0),
                bpm=(item.get("bpm") if isinstance(item, dict) else None) or None,
                keyscale=(item.get("keyscale") if isinstance(item, dict) else "") or "",
                label_source="filename_duration_fallback",
                trigger_tag=body.get("custom_tag") or body.get("trigger_tag") or "",
                tag_position=body.get("tag_position") or "prepend",
            )
            if existing_lyrics and not is_missing_vocal_lyrics({"lyrics": existing_lyrics}):
                fallback.update(
                    {
                        "lyrics": existing_lyrics,
                        "lyrics_status": "present",
                        "requires_review": False,
                        "language": (item.get("language") if isinstance(item, dict) else "") or "unknown",
                        "is_instrumental": False,
                    }
                )
            if isinstance(item, dict):
                fallback["genre"] = item.get("genre", "")
                fallback["timesignature"] = item.get("timesignature") or "4"
            if use_official and path.is_file():
                try:
                    with handler_lock:
                        _ensure_song_model(body.get("song_model"))
                        codes = handler.convert_src_audio_to_codes(str(path))
                    understood = _run_official_lm_aux("understand_music", body, audio_codes=codes)
                    lyrics_text = str(understood.get("lyrics") or "").strip()
                    missing = is_missing_vocal_lyrics({**understood, "lyrics": lyrics_text})
                    fallback.update(
                        {
                            "caption": understood.get("caption") or fallback["caption"],
                            "lyrics": "" if missing else lyrics_text,
                            "lyrics_status": "missing" if missing else "present",
                            "requires_review": missing,
                            "bpm": understood.get("bpm") or fallback["bpm"],
                            "keyscale": understood.get("key_scale") or fallback["keyscale"],
                            "timesignature": understood.get("time_signature") or fallback["timesignature"],
                            "language": understood.get("language") or ("unknown" if missing else fallback["language"]),
                            "is_instrumental": False if missing else str(lyrics_text).strip().lower() == "[instrumental]",
                            "label_source": "official_ace_step_understand_music" if not missing else "online_lyrics_missing",
                            "official_understanding": True,
                            "ace_lm_model": understood.get("ace_lm_model"),
                        }
                    )
                except ModelDownloadStarted:
                    raise
                except Exception as official_exc:
                    fallback["official_error"] = str(official_exc)
                    fallback["error"] = str(official_exc)
                    fallback["lyrics_status"] = "missing"
                    fallback["requires_review"] = True
            labels.append(fallback)
        return JSONResponse({"success": True, "labels": labels, "dataset_health": _lora_dataset_health(labels)})
    except ModelDownloadStarted as exc:
        return JSONResponse(_download_started_payload(exc.model_name, exc.job))


# ---------------------------------------------------------------------------
# Background dataset auto-labeling job
#
# Runs the official ACE-Step LM `understand_music` aux on every audio file in
# a LoRA training dataset. Writes `<stem>.lyrics.txt` and `<stem>.json`
# sidecar files so `_run_one_click_job` -> `label_entries` picks them up
# instead of falling back to "[Instrumental]".
#
# Per ACE-Step LoRA training tutorial (https://github.com/ace-step/ACE-Step-1.5
# /blob/main/docs/en/LoRA_Training_Tutorial.md): training samples need
# .lyrics.txt + .json sidecars; transcribed lyrics may contain errors and
# users are encouraged to review them after this auto-pass.
# ---------------------------------------------------------------------------


def _lora_autolabel_worker(job_id: str, body: dict[str, Any]) -> None:
    try:
        dataset_id = str(body.get("dataset_id") or "").strip()
        if not dataset_id:
            _set_lora_autolabel_job(
                job_id,
                state="error",
                status="dataset_id required",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            return
        import_root = training_manager.import_root_for(safe_id(dataset_id))
        if not import_root.is_dir():
            _set_lora_autolabel_job(
                job_id,
                state="error",
                status=f"dataset directory not found: {import_root}",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            return
        audio_paths = sorted(
            p for p in import_root.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
        )
        total = len(audio_paths)
        _set_lora_autolabel_job(
            job_id,
            state="running",
            status=f"Processing {total} audio file(s)",
            total=total,
            dataset_id=dataset_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        labels: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0
        skip_existing = parse_bool(body.get("skip_existing"), True)
        overwrite_existing_labels = parse_bool(body.get("overwrite_existing_labels"), False)
        request_body = dict(body)
        request_body.setdefault("vocal_language", body.get("language") or "unknown")

        for idx, audio_path in enumerate(audio_paths):
            stem = audio_path.stem
            existing_lyrics = audio_path.with_name(f"{stem}.lyrics.txt")
            existing_meta = audio_path.with_name(f"{stem}.json")
            relative = audio_path.relative_to(import_root)
            _set_lora_autolabel_job(
                job_id,
                processed=idx,
                progress=int(round(100 * idx / max(total, 1))),
                current_file=str(relative),
                status=f"[{idx + 1}/{total}] {relative}",
            )
            if skip_existing and not overwrite_existing_labels and existing_lyrics.is_file() and existing_meta.is_file():
                try:
                    existing_lyric_text = existing_lyrics.read_text(encoding="utf-8", errors="replace").strip()
                except Exception:
                    existing_lyric_text = ""
                try:
                    existing_metadata = json.loads(existing_meta.read_text(encoding="utf-8"))
                    if not isinstance(existing_metadata, dict):
                        existing_metadata = {}
                except Exception:
                    existing_metadata = {}
                missing_existing = is_missing_vocal_lyrics({"lyrics": existing_lyric_text})
                labels.append(
                    {
                        "path": str(audio_path),
                        "filename": audio_path.name,
                        "lyrics": "" if missing_existing else existing_lyric_text,
                        "lyrics_status": "missing" if missing_existing else "present",
                        "requires_review": missing_existing,
                        "label_source": "existing_sidecar",
                        "caption": str(existing_metadata.get("caption") or ""),
                        "language": str(existing_metadata.get("language") or "unknown"),
                        "bpm": existing_metadata.get("bpm"),
                        "keyscale": str(existing_metadata.get("keyscale") or existing_metadata.get("key_scale") or ""),
                        "genre": str(existing_metadata.get("genre") or ""),
                        "style_profile": str(existing_metadata.get("style_profile") or existing_metadata.get("genre_profile") or ""),
                        "caption_tags": str(existing_metadata.get("caption_tags") or ""),
                        "genre_label_source": str(existing_metadata.get("genre_label_source") or "metadata"),
                        "genre_confidence": existing_metadata.get("genre_confidence"),
                        "genre_reason": str(existing_metadata.get("genre_reason") or ""),
                    }
                )
                if missing_existing:
                    failed += 1
                else:
                    succeeded += 1
                continue
            try:
                understood = _training_lookup_online_lyrics(audio_path, request_body)
                paths = _training_write_label_sidecars(audio_path, understood)
                lyrics_text = str(understood.get("lyrics") or "").strip()
                missing = is_missing_vocal_lyrics({**understood, "lyrics": lyrics_text})
                lyrics_status = str(understood.get("lyrics_status") or ("missing" if missing else "present"))
                requires_review = parse_bool(understood.get("requires_review"), missing)
                labels.append(
                    {
                        "path": str(audio_path),
                        "filename": audio_path.name,
                        "lyrics": "" if missing else lyrics_text,
                        "lyrics_status": lyrics_status,
                        "requires_review": requires_review,
                        "caption": str(understood.get("caption") or ""),
                        "language": str(understood.get("language") or "unknown"),
                        "bpm": understood.get("bpm"),
                        "keyscale": str(understood.get("key_scale") or understood.get("keyscale") or ""),
                        "label_source": str(understood.get("label_source") or ("online_lyrics_missing" if missing else "official_ace_step_understand_music")),
                        "genre": str(understood.get("genre") or ""),
                        "style_profile": str(understood.get("style_profile") or understood.get("genre_profile") or ""),
                        "caption_tags": str(understood.get("caption_tags") or ""),
                        "genre_label_source": str(understood.get("genre_label_source") or ""),
                        "genre_confidence": understood.get("genre_confidence"),
                        "genre_reason": str(understood.get("genre_reason") or ""),
                        "genre_label_provider": str(understood.get("genre_label_provider") or ""),
                        "genre_label_model": str(understood.get("genre_label_model") or ""),
                        "genre_label_error": str(understood.get("genre_label_error") or ""),
                        **paths,
                    }
                )
                if missing:
                    failed += 1
                else:
                    succeeded += 1
            except ModelDownloadStarted as dl:
                _set_lora_autolabel_job(
                    job_id,
                    state="error",
                    status=f"Model download required: {dl.model_name}",
                    errors=[str(dl)],
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                return
            except Exception as exc:
                failed += 1
                err = f"{audio_path.name}: {exc}"
                _set_lora_autolabel_job(job_id, logs=[err])
                labels.append(
                    {
                        "path": str(audio_path),
                        "filename": audio_path.name,
                        "lyrics": "",
                        "lyrics_status": "missing",
                        "requires_review": True,
                        "label_source": "understand_music_failed",
                        "error": str(exc),
                    }
                )
            finally:
                # Push labels live so the React UI can render per-file rows
                # as they arrive instead of waiting for completion.
                _set_lora_autolabel_job(
                    job_id,
                    succeeded=succeeded,
                    failed=failed,
                    labels=list(labels),
                )

        _set_lora_autolabel_job(
            job_id,
            state="complete",
            status=f"Auto-label complete: {succeeded} succeeded, {failed} failed",
            progress=100,
            processed=total,
            current_file="",
            labels=list(labels),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _set_lora_autolabel_job(
            job_id,
            state="error",
            status=f"Worker crashed: {exc}",
            errors=[traceback.format_exc()],
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


@app.post("/api/lora/dataset/autolabel/jobs")
async def api_lora_autolabel_create_job(request: Request):
    try:
        body = await request.json() if request.headers.get("content-length") else {}
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    dataset_id = str(body.get("dataset_id") or "").strip()
    if not dataset_id:
        return JSONResponse({"success": False, "error": "dataset_id is required"}, status_code=400)
    # Dedup: if an active job already exists for this dataset, hand back
    # that job's id instead of spawning a parallel one. Without this guard
    # a stray re-render or double click in the wizard creates duplicate
    # workers that fight over `handler_lock` and the official-runner
    # subprocess, doubling the work for zero benefit.
    with _lora_autolabel_jobs_lock:
        for existing_id, existing in _lora_autolabel_jobs.items():
            if str(existing.get("dataset_id") or "") != dataset_id:
                continue
            existing_state = str(existing.get("state") or "").lower()
            if existing_state in {"queued", "running"}:
                snapshot = _jsonable(dict(existing))
                return JSONResponse(
                    {
                        "success": True,
                        "job_id": existing_id,
                        "job": snapshot,
                        "reused": True,
                    }
                )
    job_id = uuid.uuid4().hex[:12]
    snapshot = _set_lora_autolabel_job(job_id, dataset_id=dataset_id, state="queued", status="Queued auto-label job")
    threading.Thread(target=_lora_autolabel_worker, args=(job_id, dict(body)), daemon=True).start()
    return JSONResponse({"success": True, "job_id": job_id, "job": snapshot})


@app.get("/api/lora/dataset/autolabel/jobs")
async def api_lora_autolabel_list_jobs():
    return JSONResponse({"success": True, "jobs": _lora_autolabel_job_snapshot(None)})


@app.get("/api/lora/dataset/autolabel/jobs/{job_id}")
async def api_lora_autolabel_get_job(job_id: str):
    snapshot = _lora_autolabel_job_snapshot(job_id)
    if not snapshot:
        return JSONResponse({"success": False, "error": "Job not found"}, status_code=404)
    return JSONResponse({"success": True, "job": snapshot})


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


@app.get("/api/lora/epoch-audition/genres")
async def api_lora_epoch_audition_genres():
    return JSONResponse({"success": True, "genres": epoch_audition_genre_options()})


@app.get("/api/audio/style-profiles")
async def api_audio_style_profiles():
    return JSONResponse({"success": True, "profiles": audio_style_profiles()})


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


@app.post("/api/lora/jobs/{job_id}/resume")
async def api_lora_job_resume(job_id: str):
    try:
        return JSONResponse({"success": True, "job": training_manager.resume_job(job_id)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


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
        body = _album_ace_lm_disabled_payload(await request.json())
        planner_provider = _album_planner_provider_from_payload(body)
        embedding_provider = _embedding_provider_from_payload(body)
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
        planning_engine = _normalize_album_agent_engine_value(body.get("agent_engine"))
        request_body = {
            **body,
            "agent_engine": planning_engine,
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
            ace_lm_model="none",
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
        body = _album_ace_lm_disabled_payload(await request.json())
        direct_render_tracks = _json_list(body.get("tracks") or body.get("planned_tracks"))
        direct_existing_render = bool(direct_render_tracks) and (
            parse_bool(body.get("render_from_existing_tracks"), False)
            or parse_bool(body.get("skip_album_planning"), False)
            or str(body.get("album_generation_mode") or "").strip().lower()
            in {"render_existing_tracks", "direct_render", "ui_tracks"}
        )
        planner_provider = _album_planner_provider_from_payload(body)
        embedding_provider = _embedding_provider_from_payload(body)
        if direct_existing_render:
            planner_model = str(
                body.get("planner_model")
                or body.get("planner_ollama_model")
                or body.get("ollama_model")
                or ""
            ).strip()
            embedding_model = str(body.get("embedding_model") or "").strip()
        else:
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
        planning_engine = _normalize_album_agent_engine_value(body.get("agent_engine"))
        planning_engine_label = _album_agent_engine_label_value(planning_engine)
        request_body = {
            **body,
            "agent_engine": planning_engine,
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
            planning_engine="existing_ui_tracks" if direct_existing_render else planning_engine,
            custom_agents_used=False if direct_existing_render else True,
            crewai_used=False if direct_existing_render else planning_engine == "crewai_micro",
            memory_enabled=False,
            logs=[
                f"Queued album job {job_id}.",
                (
                    f"Direct render: {len(direct_render_tracks)} UI-approved track(s); no album agents will run."
                    if direct_existing_render
                    else f"Planning Engine: {planning_engine_label} ({planning_engine})"
                ),
                "ACE-Step LM disabled for album agents.",
            ],
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


@app.post("/api/album/jobs/{job_id}/stop")
async def api_album_job_stop(job_id: str):
    job = _album_job_snapshot(job_id)
    if not job:
        return JSONResponse({"success": False, "error": "Album job not found"}, status_code=404)
    state = str((job or {}).get("state") or "").lower() if isinstance(job, dict) else ""
    if state in {"succeeded", "success", "failed", "error", "stopped"}:
        return JSONResponse({"success": True, "job": job, "message": "Album job already finished"})
    updated = _set_album_job(
        job_id,
        state="stopping",
        stage="cancel_requested",
        status="Stop requested",
        current_task="Stop requested; waiting for current LLM call to return",
        stop_requested=True,
        logs=["Stop requested from JobTracker."],
    )
    return JSONResponse({"success": True, "job": updated})


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


@app.get("/media/art/{art_id}/{filename}")
async def art_media(art_id: str, filename: str):
    target = _resolve_child(ART_DIR, safe_id(art_id), filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/media/mflux/{result_id}/{filename}")
async def mflux_media(result_id: str, filename: str):
    target = _resolve_child(MFLUX_RESULTS_DIR, safe_id(result_id), filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/media/mflux/uploads/{upload_id}/{filename}")
async def mflux_upload_media(upload_id: str, filename: str):
    target = _resolve_child(MFLUX_UPLOADS_DIR, safe_id(upload_id), filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/media/mlx-video/{result_id}/{filename}")
async def mlx_video_media(result_id: str, filename: str):
    target = _resolve_child(MLX_VIDEO_RESULTS_DIR, safe_id(result_id), filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/media/mlx-video/uploads/{upload_id}/{filename}")
async def mlx_video_upload_media(upload_id: str, filename: str):
    target = _resolve_child(MLX_VIDEO_UPLOADS_DIR, safe_id(upload_id), filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    web_index = BASE_DIR / "web" / "dist" / "index.html"
    if web_index.is_file():
        return web_index.read_text(encoding="utf-8")
    return (
        "<h1>MLX Media web UI is not built yet</h1>"
        "<p>Run <code>npm install &amp;&amp; npm run build</code> in <code>app/web</code>, "
        "or use the Pinokio installer / updater.</p>"
    )


demo = app


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False)
