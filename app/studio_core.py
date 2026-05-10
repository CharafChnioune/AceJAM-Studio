from __future__ import annotations

import copy
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from acestep.constants import (
    BPM_MAX,
    BPM_MIN,
    KEYSCALE_ACCIDENTALS,
    KEYSCALE_MODES,
    KEYSCALE_NOTES,
    TASK_INSTRUCTIONS,
    TASK_TYPES,
    TASK_TYPES_BASE,
    TASK_TYPES_TURBO,
    TRACK_NAMES,
    VALID_KEYSCALES,
)

KNOWN_ACE_STEP_MODELS = [
    "acestep-v15-turbo",
    "acestep-v15-turbo-shift1",
    "acestep-v15-turbo-shift3",
    "acestep-v15-turbo-continuous",
    "acestep-v15-sft",
    "acestep-v15-base",
    "acestep-v15-xl-turbo",
    "acestep-v15-xl-sft",
    "acestep-v15-xl-base",
    "acestep-v15-turbo-rl",
]

ACE_STEP_LM_MODELS = [
    "auto",
    "none",
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-4B",
]

ACE_STEP_MODEL_SOURCES = [
    "https://github.com/ace-step/ACE-Step-1.5",
    "https://huggingface.co/ACE-Step/Ace-Step1.5",
    "https://huggingface.co/ACE-Step/acestep-v15-sft",
    "https://huggingface.co/ACE-Step/acestep-v15-turbo-shift1",
    "https://huggingface.co/ACE-Step/acestep-v15-turbo-continuous",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-sft",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-base",
    "https://huggingface.co/ACE-Step/acestep-captioner",
    "https://huggingface.co/ACE-Step/acestep-transcriber",
    "https://huggingface.co/ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md",
    "https://arxiv.org/abs/2506.00045",
    "https://arxiv.org/abs/2602.00744",
]

STANDARD_TASKS = ["text2music", "cover", "cover-nofsq", "repaint"]
ALL_TASKS = ["text2music", "cover", "cover-nofsq", "repaint", "extract", "lego", "complete"]
OFFICIAL_UNRELEASED_MODELS = {"acestep-v15-turbo-rl"}
OFFICIAL_CORE_MODEL_ID = "main"
OFFICIAL_MAIN_MODEL_REPO = "ACE-Step/Ace-Step1.5"
OFFICIAL_MAIN_MODEL_COMPONENTS = [
    "acestep-v15-turbo",
    "vae",
    "Qwen3-Embedding-0.6B",
    "acestep-5Hz-lm-1.7B",
]
OFFICIAL_HELPER_MODEL_IDS = [
    "acestep-captioner",
    "acestep-transcriber",
    "ace-step-v1.5-1d-vae-stable-audio-format",
    "acestep-v15-xl-turbo-diffusers",
]
OFFICIAL_BOOT_QUALITY_MODEL_IDS = [
    "acestep-v15-xl-sft",
    "acestep-v15-sft",
]
OFFICIAL_LORA_MODEL_IDS = [
    "ACE-Step-v1-chinese-rap-LoRA",
    "ACE-Step-v1.5-chinese-new-year-LoRA",
]
OFFICIAL_LEGACY_MODEL_IDS = ["ACE-Step-v1-3.5B"]
DIFFUSERS_PIPELINE_COMPONENTS = (
    "condition_encoder",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
DIFFUSERS_PIPELINE_WEIGHT_COMPONENTS = ("condition_encoder", "text_encoder", "transformer", "vae")
DIFFUSERS_PIPELINE_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".ckpt")
OFFICIAL_ACE_STEP_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    OFFICIAL_CORE_MODEL_ID: {
        "id": OFFICIAL_CORE_MODEL_ID,
        "repo_id": OFFICIAL_MAIN_MODEL_REPO,
        "role": "core_bundle",
        "tasks": [],
        "quality": "required",
        "steps": None,
        "cfg": None,
        "downloadable": True,
        "render_usable": False,
        "components": OFFICIAL_MAIN_MODEL_COMPONENTS,
        "source": "official_main_bundle",
    },
    "acestep-v15-turbo": {
        "id": "acestep-v15-turbo",
        "repo_id": OFFICIAL_MAIN_MODEL_REPO,
        "role": "render_dit",
        "tasks": STANDARD_TASKS,
        "quality": "Very High",
        "steps": 8,
        "cfg": False,
        "downloadable": True,
        "render_usable": True,
        "source": "main_bundle",
    },
    "acestep-v15-turbo-shift1": {
        "id": "acestep-v15-turbo-shift1",
        "repo_id": "ACE-Step/acestep-v15-turbo-shift1",
        "role": "render_dit",
        "tasks": STANDARD_TASKS,
        "quality": "Very High",
        "steps": 8,
        "cfg": False,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-turbo-shift3": {
        "id": "acestep-v15-turbo-shift3",
        "repo_id": "ACE-Step/acestep-v15-turbo-shift3",
        "role": "render_dit",
        "tasks": STANDARD_TASKS,
        "quality": "Very High",
        "steps": 8,
        "cfg": False,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-turbo-continuous": {
        "id": "acestep-v15-turbo-continuous",
        "repo_id": "ACE-Step/acestep-v15-turbo-continuous",
        "role": "render_dit",
        "tasks": STANDARD_TASKS,
        "quality": "Very High",
        "steps": 8,
        "cfg": False,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-sft": {
        "id": "acestep-v15-sft",
        "repo_id": "ACE-Step/acestep-v15-sft",
        "role": "render_dit",
        "tasks": STANDARD_TASKS,
        "quality": "High",
        "steps": 50,
        "cfg": True,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-base": {
        "id": "acestep-v15-base",
        "repo_id": "ACE-Step/acestep-v15-base",
        "role": "render_dit",
        "tasks": ALL_TASKS,
        "quality": "Medium",
        "steps": 50,
        "cfg": True,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-xl-turbo": {
        "id": "acestep-v15-xl-turbo",
        "repo_id": "ACE-Step/acestep-v15-xl-turbo",
        "role": "render_dit",
        "tasks": STANDARD_TASKS,
        "quality": "Very High",
        "steps": 8,
        "cfg": False,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-xl-sft": {
        "id": "acestep-v15-xl-sft",
        "repo_id": "ACE-Step/acestep-v15-xl-sft",
        "role": "render_dit",
        "tasks": STANDARD_TASKS,
        "quality": "Very High",
        "steps": 50,
        "cfg": True,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-xl-base": {
        "id": "acestep-v15-xl-base",
        "repo_id": "ACE-Step/acestep-v15-xl-base",
        "role": "render_dit",
        "tasks": ALL_TASKS,
        "quality": "High",
        "steps": 50,
        "cfg": True,
        "downloadable": True,
        "render_usable": True,
        "source": "official_submodel",
    },
    "acestep-v15-turbo-rl": {
        "id": "acestep-v15-turbo-rl",
        "repo_id": "",
        "role": "unreleased",
        "tasks": STANDARD_TASKS,
        "quality": "Very High",
        "steps": 8,
        "cfg": False,
        "downloadable": False,
        "render_usable": False,
        "source": "official_unreleased",
        "status": "unreleased",
    },
    "acestep-5Hz-lm-0.6B": {
        "id": "acestep-5Hz-lm-0.6B",
        "repo_id": "ACE-Step/acestep-5Hz-lm-0.6B",
        "role": "ace_lm",
        "tasks": ["metadata", "caption_rewrite", "audio_understanding", "semantic_codes"],
        "quality": "Medium",
        "steps": None,
        "cfg": "LM CFG",
        "downloadable": True,
        "render_usable": False,
        "source": "official_submodel",
    },
    "acestep-5Hz-lm-1.7B": {
        "id": "acestep-5Hz-lm-1.7B",
        "repo_id": OFFICIAL_MAIN_MODEL_REPO,
        "role": "ace_lm",
        "tasks": ["metadata", "caption_rewrite", "audio_understanding", "semantic_codes"],
        "quality": "Medium",
        "steps": None,
        "cfg": "LM CFG",
        "downloadable": True,
        "render_usable": False,
        "source": "main_bundle",
    },
    "acestep-5Hz-lm-4B": {
        "id": "acestep-5Hz-lm-4B",
        "repo_id": "ACE-Step/acestep-5Hz-lm-4B",
        "role": "ace_lm",
        "tasks": ["metadata", "caption_rewrite", "audio_understanding", "semantic_codes"],
        "quality": "Strong",
        "steps": None,
        "cfg": "LM CFG",
        "downloadable": True,
        "render_usable": False,
        "source": "official_submodel",
    },
    "acestep-captioner": {
        "id": "acestep-captioner",
        "repo_id": "ACE-Step/acestep-captioner",
        "role": "helper",
        "tasks": ["audio_captioning"],
        "quality": "helper",
        "steps": None,
        "cfg": None,
        "downloadable": True,
        "render_usable": False,
        "source": "official_helper",
    },
    "acestep-transcriber": {
        "id": "acestep-transcriber",
        "repo_id": "ACE-Step/acestep-transcriber",
        "role": "helper",
        "tasks": ["transcription"],
        "quality": "helper",
        "steps": None,
        "cfg": None,
        "downloadable": True,
        "render_usable": False,
        "source": "official_helper",
    },
    "ace-step-v1.5-1d-vae-stable-audio-format": {
        "id": "ace-step-v1.5-1d-vae-stable-audio-format",
        "repo_id": "ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format",
        "role": "helper",
        "tasks": ["vae_experiment"],
        "quality": "helper",
        "steps": None,
        "cfg": None,
        "downloadable": True,
        "render_usable": False,
        "source": "official_helper",
    },
    "acestep-v15-xl-turbo-diffusers": {
        "id": "acestep-v15-xl-turbo-diffusers",
        "repo_id": "ACE-Step/acestep-v15-xl-turbo-diffusers",
        "role": "diffusers_export",
        "tasks": ["diffusers_pipeline"],
        "quality": "Very High",
        "steps": 8,
        "cfg": False,
        "downloadable": True,
        "render_usable": False,
        "source": "official_diffusers_export",
    },
    "ACE-Step-v1-3.5B": {
        "id": "ACE-Step-v1-3.5B",
        "repo_id": "ACE-Step/ACE-Step-v1-3.5B",
        "role": "legacy",
        "tasks": ["legacy_v1"],
        "quality": "legacy",
        "steps": None,
        "cfg": None,
        "downloadable": True,
        "render_usable": False,
        "source": "official_legacy",
    },
    "ACE-Step-v1-chinese-rap-LoRA": {
        "id": "ACE-Step-v1-chinese-rap-LoRA",
        "repo_id": "ACE-Step/ACE-Step-v1-chinese-rap-LoRA",
        "role": "lora",
        "tasks": ["adapter"],
        "quality": "adapter",
        "steps": None,
        "cfg": None,
        "downloadable": True,
        "render_usable": False,
        "source": "official_lora",
    },
    "ACE-Step-v1.5-chinese-new-year-LoRA": {
        "id": "ACE-Step-v1.5-chinese-new-year-LoRA",
        "repo_id": "ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA",
        "role": "lora",
        "tasks": ["adapter"],
        "quality": "adapter",
        "steps": None,
        "cfg": None,
        "downloadable": True,
        "render_usable": False,
        "source": "official_lora",
    },
}


def _has_diffusers_weight_file(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    return any(
        child.is_file() and child.stat().st_size > 0 and child.suffix in DIFFUSERS_PIPELINE_WEIGHT_SUFFIXES
        for child in children
    )


def diffusers_pipeline_missing_reasons(path: Path | str) -> list[str]:
    """Return missing pieces for a Diffusers pipeline export directory."""
    pipeline_path = Path(path)
    if not pipeline_path.exists():
        return ["missing directory"]
    if not pipeline_path.is_dir():
        return ["not a directory"]

    reasons: list[str] = []
    model_index_path = pipeline_path / "model_index.json"
    model_index: dict[str, Any] = {}
    if not model_index_path.is_file():
        reasons.append("missing model_index.json")
    else:
        try:
            parsed = json.loads(model_index_path.read_text(encoding="utf-8"))
            model_index = parsed if isinstance(parsed, dict) else {}
        except Exception:
            reasons.append("invalid model_index.json")

    for component in DIFFUSERS_PIPELINE_COMPONENTS:
        if model_index and component not in model_index:
            reasons.append(f"model_index.json missing {component}")
        component_path = pipeline_path / component
        if not component_path.is_dir():
            reasons.append(f"missing {component}/")
            continue
        if component in DIFFUSERS_PIPELINE_WEIGHT_COMPONENTS:
            if not (component_path / "config.json").is_file():
                reasons.append(f"missing {component}/config.json")
            if not _has_diffusers_weight_file(component_path):
                reasons.append(f"missing {component} weights")
        elif component == "scheduler":
            if not (component_path / "scheduler_config.json").is_file():
                reasons.append("missing scheduler/scheduler_config.json")
        elif component == "tokenizer":
            if not (component_path / "tokenizer_config.json").is_file():
                reasons.append("missing tokenizer/tokenizer_config.json")
            if not any((component_path / name).is_file() for name in ("tokenizer.json", "vocab.json")):
                reasons.append("missing tokenizer vocabulary")
    return reasons


def diffusers_pipeline_dir_ready(path: Path | str) -> bool:
    return not diffusers_pipeline_missing_reasons(path)

DEFAULT_BPM = 95
DEFAULT_KEY_SCALE = "A minor"
KEYSCALE_AUTO_VALUE = "auto"
METADATA_AUTO_VALUE = "auto"
VALID_KEY_SCALES = [
    f"{note}{accidental} {mode}"
    for note in KEYSCALE_NOTES
    for accidental in KEYSCALE_ACCIDENTALS
    for mode in KEYSCALE_MODES
    if f"{note}{accidental} {mode}" in VALID_KEYSCALES
]
VALID_KEY_SCALE_SET = set(VALID_KEY_SCALES)

OFFICIAL_GENERATION_PARAMS = [
    "task_type",
    "instruction",
    "reference_audio",
    "src_audio",
    "audio_codes",
    "caption",
    "global_caption",
    "lyrics",
    "instrumental",
    "vocal_language",
    "bpm",
    "keyscale",
    "timesignature",
    "duration",
    "enable_normalization",
    "normalization_db",
    "fade_in_duration",
    "fade_out_duration",
    "latent_shift",
    "latent_rescale",
    "inference_steps",
    "seed",
    "guidance_scale",
    "use_adg",
    "cfg_interval_start",
    "cfg_interval_end",
    "shift",
    "infer_method",
    "sampler_mode",
    "velocity_norm_threshold",
    "velocity_ema_factor",
    "dcw_enabled",
    "dcw_mode",
    "dcw_scaler",
    "dcw_high_scaler",
    "dcw_wavelet",
    "timesteps",
    "repainting_start",
    "repainting_end",
    "chunk_mask_mode",
    "repaint_latent_crossfade_frames",
    "repaint_wav_crossfade_sec",
    "repaint_mode",
    "repaint_strength",
    "retake_seed",
    "retake_variance",
    "flow_edit_morph",
    "flow_edit_source_caption",
    "flow_edit_source_lyrics",
    "flow_edit_n_min",
    "flow_edit_n_max",
    "flow_edit_n_avg",
    "audio_cover_strength",
    "cover_noise_strength",
    "thinking",
    "lm_temperature",
    "lm_cfg_scale",
    "lm_top_k",
    "lm_top_p",
    "lm_negative_prompt",
    "use_cot_metas",
    "use_cot_caption",
    "use_cot_lyrics",
    "use_cot_language",
    "use_constrained_decoding",
    "cot_bpm",
    "cot_keyscale",
    "cot_timesignature",
    "cot_duration",
    "cot_vocal_language",
    "cot_caption",
    "cot_lyrics",
]

OFFICIAL_GENERATION_CONFIG_FIELDS = [
    "batch_size",
    "allow_lm_batch",
    "use_random_seed",
    "seeds",
    "lm_batch_chunk_size",
    "constrained_decoding_debug",
    "audio_format",
    "mp3_bitrate",
    "mp3_sample_rate",
]

OFFICIAL_API_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/release_task": {"method": "POST", "status": "supported", "feature": "async generation"},
    "/query_result": {"method": "POST", "status": "supported", "feature": "async result query"},
    "/format_input": {"method": "POST", "status": "supported", "feature": "official LM format helper"},
    "/create_random_sample": {"method": "POST", "status": "supported", "feature": "sample preset/form fill"},
    "/v1/models": {"method": "GET", "status": "supported", "feature": "model listing"},
    "/v1/init": {"method": "POST", "status": "guarded", "feature": "lazy model init/switch"},
    "/v1/stats": {"method": "GET", "status": "supported", "feature": "local job statistics"},
    "/v1/audio": {"method": "GET", "status": "supported", "feature": "audio download"},
    "/health": {"method": "GET", "status": "supported", "feature": "health check"},
    "/v1/training/start": {"method": "POST", "status": "supported", "feature": "LoRA training wrapper"},
    "/v1/training/start_lokr": {"method": "POST", "status": "supported", "feature": "LoKr training wrapper"},
}

OFFICIAL_RUNTIME_CONTROLS: dict[str, dict[str, Any]] = {
    "ACESTEP_CONFIG_PATH": {"status": "supported", "ui_field": "song_model"},
    "ACESTEP_CONFIG_PATH2": {"status": "guarded", "ui_field": "v1_init_slot_2"},
    "ACESTEP_CONFIG_PATH3": {"status": "guarded", "ui_field": "v1_init_slot_3"},
    "ACESTEP_DEVICE": {"status": "guarded", "ui_field": "device"},
    "ACESTEP_USE_FLASH_ATTENTION": {"status": "guarded", "ui_field": "use_flash_attention"},
    "ACESTEP_OFFLOAD_TO_CPU": {"status": "guarded", "ui_field": "offload_to_cpu"},
    "ACESTEP_OFFLOAD_DIT_TO_CPU": {"status": "guarded", "ui_field": "offload_dit_to_cpu"},
    "ACESTEP_INIT_LLM": {"status": "guarded", "ui_field": "ace_lm_model"},
    "ACESTEP_LM_MODEL_PATH": {"status": "supported", "ui_field": "ace_lm_model"},
    "ACESTEP_LM_BACKEND": {"status": "supported", "ui_field": "lm_backend"},
    "ACESTEP_LM_DEVICE": {"status": "guarded", "ui_field": "lm_device"},
    "ACESTEP_LM_OFFLOAD_TO_CPU": {"status": "guarded", "ui_field": "lm_offload_to_cpu"},
    "ACESTEP_API_KEY": {"status": "supported", "ui_field": "official API compatible routes"},
}

OFFICIAL_TRAINING_FEATURES: dict[str, dict[str, Any]] = {
    "dataset_scan": {"status": "supported", "endpoint": "/api/lora/dataset/scan"},
    "dataset_save": {"status": "supported", "endpoint": "/api/lora/dataset/save"},
    "auto_label": {"status": "guarded", "endpoint": "/api/lora/dataset/autolabel"},
    "preprocess": {"status": "supported", "endpoint": "/api/lora/preprocess"},
    "lora_train": {"status": "supported", "endpoint": "/api/lora/train"},
    "lokr_train": {"status": "supported", "endpoint": "/api/lora/train"},
    "estimate": {"status": "supported", "endpoint": "/api/lora/estimate"},
    "stop": {"status": "supported", "endpoint": "/api/lora/jobs/{id}/stop"},
    "export": {"status": "supported", "endpoint": "/api/lora/export"},
    "tensorboard_runs": {"status": "guarded", "endpoint": "/api/lora/status"},
}

DOCS_BEST_QUALITY_POLICY_VERSION = "ace-step-docs-correct-defaults-2026-05-05"
DOCS_BEST_AUDIO_FORMAT = "wav32"
DOCS_BEST_TURBO_STEPS = 8
DOCS_BEST_TURBO_HIGH_CAP_STEPS = 20
PREVIEW_FAST_STANDARD_STEPS = 50
BALANCED_PRO_STANDARD_STEPS = 50
CHART_MASTER_STANDARD_STEPS = 50
DOCS_BEST_STANDARD_STEPS = CHART_MASTER_STANDARD_STEPS
DOCS_BEST_MODEL_STEPS = {
    "acestep-v15-turbo": 8,
    "acestep-v15-turbo-shift1": 8,
    "acestep-v15-turbo-shift3": 8,
    "acestep-v15-turbo-continuous": 8,
    "acestep-v15-turbo-rl": 8,
    "acestep-v15-sft": CHART_MASTER_STANDARD_STEPS,
    "acestep-v15-base": CHART_MASTER_STANDARD_STEPS,
    "acestep-v15-xl-turbo": 8,
    "acestep-v15-xl-sft": CHART_MASTER_STANDARD_STEPS,
    "acestep-v15-xl-base": CHART_MASTER_STANDARD_STEPS,
}
BALANCED_PRO_MODEL_STEPS = {
    model: (DOCS_BEST_TURBO_STEPS if is_turbo else BALANCED_PRO_STANDARD_STEPS)
    for model, is_turbo in {
        "acestep-v15-turbo": True,
        "acestep-v15-turbo-shift3": True,
        "acestep-v15-turbo-rl": True,
        "acestep-v15-sft": False,
        "acestep-v15-base": False,
        "acestep-v15-xl-turbo": True,
        "acestep-v15-xl-sft": False,
        "acestep-v15-xl-base": False,
    }.items()
}
DOCS_BEST_TURBO_GUIDANCE = 7.0
DOCS_BEST_STANDARD_GUIDANCE = 7.0
DOCS_BEST_TURBO_SHIFT = 3.0
PREVIEW_FAST_STANDARD_SHIFT = 1.0
BALANCED_PRO_STANDARD_SHIFT = 1.0
CHART_MASTER_STANDARD_SHIFT = 1.0
DOCS_BEST_STANDARD_SHIFT = CHART_MASTER_STANDARD_SHIFT
DOCS_DAILY_DEFAULT_LM_MODEL = "acestep-5Hz-lm-1.7B"
MAX_QUALITY_DEFAULT_LM_MODEL = "acestep-5Hz-lm-4B"
DOCS_BEST_DEFAULT_LM_MODEL = MAX_QUALITY_DEFAULT_LM_MODEL
DOCS_BEST_DEFAULT_LM_BACKEND = "mlx" if sys.platform == "darwin" and platform.machine() == "arm64" else "pt"
ACE_STEP_SETTINGS_POLICY_VERSION = "ace-step-settings-parity-2026-04-26"
PRO_QUALITY_AUDIT_VERSION = "ace-step-pro-quality-audit-2026-04-27"
ACE_STEP_CAPTION_CHAR_LIMIT = 512
ACE_STEP_LYRICS_CHAR_LIMIT = 4096
ACE_STEP_LYRICS_SOFT_TARGET_MIN = 3200
ACE_STEP_LYRICS_SOFT_TARGET_MAX = 3600
ACE_STEP_LYRICS_WARNING_CHAR_LIMIT = 3800
ACE_STEP_LYRICS_SAFE_HEADROOM = 200
ACE_STEP_DIT_LYRICS_TOKEN_LIMIT = 2048
DOCS_BEST_SOURCE_TASK_LM_SKIPS = {"cover", "cover-nofsq", "repaint", "extract"}
DOCS_BEST_LM_TASKS = {"text2music", "lego", "complete"}
DOCS_BEST_LM_DEFAULTS: dict[str, Any] = {
    "ace_lm_model": DOCS_BEST_DEFAULT_LM_MODEL,
    "lm_backend": DOCS_BEST_DEFAULT_LM_BACKEND,
    "thinking": True,
    "use_format": True,
    "use_cot_metas": True,
    "use_cot_caption": True,
    "use_cot_language": True,
    "use_cot_lyrics": False,
    "use_constrained_decoding": True,
    "lm_cfg_scale": 2.0,
    "lm_api_cfg_scale": 2.5,
    "lm_temperature": 0.85,
    "lm_top_p": 0.9,
    "lm_top_k": 0,
}
PRO_AUDIO_TARGETS: dict[str, Any] = {
    "peak_dbfs_target": -1.0,
    "peak_linear_target": 0.8913,
    "peak_linear_min": 0.25,
    "peak_linear_max": 0.98,
    "clip_linear_threshold": 0.999,
    "clip_percent_max": 0.01,
    "duration_tolerance_seconds": 2.0,
    "duration_tolerance_ratio": 0.05,
    "silence_edge_seconds_warn": 3.0,
    "bpm_tolerance": 6.0,
}

QUALITY_PROFILE_OFFICIAL_RAW = "official_raw"
QUALITY_PROFILE_DOCS_DAILY = "docs_daily"
QUALITY_PROFILE_PREVIEW_FAST = "preview_fast"
QUALITY_PROFILE_BALANCED_PRO = "balanced_pro"
QUALITY_PROFILE_CHART_MASTER = "chart_master"
DEFAULT_QUALITY_PROFILE = QUALITY_PROFILE_CHART_MASTER
QUALITY_PROFILES = [
    QUALITY_PROFILE_OFFICIAL_RAW,
    QUALITY_PROFILE_DOCS_DAILY,
    QUALITY_PROFILE_PREVIEW_FAST,
    QUALITY_PROFILE_BALANCED_PRO,
    QUALITY_PROFILE_CHART_MASTER,
]
CHART_MASTER_SINGLE_TAKES = 1
CHART_MASTER_ALBUM_TAKES = 1

OFFICIAL_HELPER_FUNCTIONS = {
    "generate_music": {"status": "supported", "feature": "main generation"},
    "understand_music": {"status": "supported", "feature": "audio semantic-code understanding"},
    "create_sample": {"status": "supported", "feature": "LM sample/caption/lyrics helper"},
    "format_sample": {"status": "supported", "feature": "LM formatting helper"},
}

OFFICIAL_RESULT_FIELDS = {
    "GenerationResult.success": "active",
    "GenerationResult.error": "active",
    "GenerationResult.status_message": "active",
    "GenerationResult.extra_outputs": "active",
    "GenerationResult.audios": "active",
    "Audio.path": "active",
    "Audio.tensor": "read_only",
    "Audio.key": "active",
    "Audio.sample_rate": "active",
    "Audio.params": "active",
}


def is_turbo_song_model(song_model: Any) -> bool:
    normalized = str(song_model or "").strip().lower()
    return "turbo" in normalized


def is_base_song_model(song_model: Any) -> bool:
    normalized = str(song_model or "").strip().lower()
    return normalized.endswith("-base") or "-base" in normalized


def normalize_quality_profile(value: Any) -> str:
    normalized = str(value or DEFAULT_QUALITY_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "best": QUALITY_PROFILE_CHART_MASTER,
        "max": QUALITY_PROFILE_CHART_MASTER,
        "max_quality": QUALITY_PROFILE_CHART_MASTER,
        "chart": QUALITY_PROFILE_CHART_MASTER,
        "chartmaster": QUALITY_PROFILE_CHART_MASTER,
        "chart_master": QUALITY_PROFILE_CHART_MASTER,
        "high": QUALITY_PROFILE_CHART_MASTER,
        "hoog": QUALITY_PROFILE_CHART_MASTER,
        "docs": QUALITY_PROFILE_DOCS_DAILY,
        "docs_daily": QUALITY_PROFILE_DOCS_DAILY,
        "daily": QUALITY_PROFILE_DOCS_DAILY,
        "standard": QUALITY_PROFILE_BALANCED_PRO,
        "middle": QUALITY_PROFILE_BALANCED_PRO,
        "medium": QUALITY_PROFILE_BALANCED_PRO,
        "middel": QUALITY_PROFILE_BALANCED_PRO,
        "balanced": QUALITY_PROFILE_BALANCED_PRO,
        "balanced_pro": QUALITY_PROFILE_BALANCED_PRO,
        "draft": QUALITY_PROFILE_PREVIEW_FAST,
        "low": QUALITY_PROFILE_PREVIEW_FAST,
        "laag": QUALITY_PROFILE_PREVIEW_FAST,
        "fast": QUALITY_PROFILE_PREVIEW_FAST,
        "preview": QUALITY_PROFILE_PREVIEW_FAST,
        "preview_fast": QUALITY_PROFILE_PREVIEW_FAST,
        "official": QUALITY_PROFILE_OFFICIAL_RAW,
        "official_raw": QUALITY_PROFILE_OFFICIAL_RAW,
        "raw": QUALITY_PROFILE_OFFICIAL_RAW,
    }
    return aliases.get(normalized, DEFAULT_QUALITY_PROFILE)


def _quality_profile_base_settings(profile: str) -> dict[str, Any]:
    normalized = normalize_quality_profile(profile)
    if normalized == QUALITY_PROFILE_OFFICIAL_RAW:
        return {
            "quality_profile": QUALITY_PROFILE_OFFICIAL_RAW,
            "quality_preset": "official-raw-ace-step",
            "inference_steps": 8,
            "guidance_scale": 7.0,
            "shift": 3.0,
            "infer_method": "ode",
            "sampler_mode": "euler",
            "audio_format": "flac",
            "use_adg": False,
            "single_song_takes": 1,
            "album_takes": 1,
        }
    if normalized == QUALITY_PROFILE_DOCS_DAILY:
        return {
            "quality_profile": QUALITY_PROFILE_DOCS_DAILY,
            "quality_preset": "docs-daily-ace-step",
            "inference_steps": DOCS_BEST_STANDARD_STEPS,
            "guidance_scale": DOCS_BEST_STANDARD_GUIDANCE,
            "shift": DOCS_BEST_STANDARD_SHIFT,
            "infer_method": "ode",
            "sampler_mode": "heun",
            "audio_format": "flac",
            "use_adg": False,
            "single_song_takes": 1,
            "album_takes": 1,
        }
    if normalized == QUALITY_PROFILE_PREVIEW_FAST:
        return {
            "quality_profile": QUALITY_PROFILE_PREVIEW_FAST,
            "quality_preset": "low-docs-correct-ace-step",
            "inference_steps": PREVIEW_FAST_STANDARD_STEPS,
            "guidance_scale": DOCS_BEST_STANDARD_GUIDANCE,
            "shift": PREVIEW_FAST_STANDARD_SHIFT,
            "infer_method": "ode",
            "sampler_mode": "heun",
            "audio_format": "wav",
            "use_adg": False,
            "single_song_takes": 1,
            "album_takes": 1,
        }
    if normalized == QUALITY_PROFILE_BALANCED_PRO:
        return {
            "quality_profile": QUALITY_PROFILE_BALANCED_PRO,
            "quality_preset": "balanced-pro-2026-04-26",
            "inference_steps": BALANCED_PRO_STANDARD_STEPS,
            "guidance_scale": DOCS_BEST_STANDARD_GUIDANCE,
            "shift": BALANCED_PRO_STANDARD_SHIFT,
            "infer_method": "ode",
            "sampler_mode": "heun",
            "audio_format": DOCS_BEST_AUDIO_FORMAT,
            "use_adg": False,
            "single_song_takes": 1,
            "album_takes": 1,
        }
    return {
        "quality_profile": QUALITY_PROFILE_CHART_MASTER,
        "quality_preset": DOCS_BEST_QUALITY_POLICY_VERSION,
        "inference_steps": CHART_MASTER_STANDARD_STEPS,
        "guidance_scale": DOCS_BEST_STANDARD_GUIDANCE,
        "shift": CHART_MASTER_STANDARD_SHIFT,
        "infer_method": "ode",
        "sampler_mode": "heun",
        "audio_format": DOCS_BEST_AUDIO_FORMAT,
        "use_adg": True,
        "single_song_takes": CHART_MASTER_SINGLE_TAKES,
        "album_takes": CHART_MASTER_ALBUM_TAKES,
    }


def quality_profile_model_settings(song_model: Any, quality_profile: Any = DEFAULT_QUALITY_PROFILE, *, high_turbo_cap: bool = False) -> dict[str, Any]:
    """Return ACE-Step-calibrated defaults for one model/profile pair."""
    profile = normalize_quality_profile(quality_profile)
    settings = _quality_profile_base_settings(profile)
    turbo = is_turbo_song_model(song_model)
    if turbo:
        settings["inference_steps"] = DOCS_BEST_TURBO_HIGH_CAP_STEPS if high_turbo_cap else DOCS_BEST_TURBO_STEPS
        settings["guidance_scale"] = DOCS_BEST_TURBO_GUIDANCE
        settings["shift"] = DOCS_BEST_TURBO_SHIFT
        settings["use_adg"] = False
        if profile == QUALITY_PROFILE_PREVIEW_FAST:
            settings["audio_format"] = "wav"
        elif profile == QUALITY_PROFILE_DOCS_DAILY:
            settings["audio_format"] = "flac"
    elif profile == QUALITY_PROFILE_BALANCED_PRO:
        settings["inference_steps"] = int(BALANCED_PRO_MODEL_STEPS.get(str(song_model or "").strip(), BALANCED_PRO_STANDARD_STEPS))
        settings["shift"] = BALANCED_PRO_STANDARD_SHIFT
    elif profile == QUALITY_PROFILE_DOCS_DAILY:
        settings["inference_steps"] = DOCS_BEST_STANDARD_STEPS
        settings["guidance_scale"] = DOCS_BEST_STANDARD_GUIDANCE
        settings["shift"] = DOCS_BEST_STANDARD_SHIFT
        settings["use_adg"] = is_base_song_model(song_model)
    elif profile == QUALITY_PROFILE_CHART_MASTER:
        settings["inference_steps"] = int(DOCS_BEST_MODEL_STEPS.get(str(song_model or "").strip(), CHART_MASTER_STANDARD_STEPS))
        settings["shift"] = CHART_MASTER_STANDARD_SHIFT
        settings["use_adg"] = is_base_song_model(song_model)
    return settings


def docs_best_model_settings(song_model: Any, *, high_turbo_cap: bool = False) -> dict[str, Any]:
    """Return AceJAM's default Chart Master settings for the selected DiT model."""
    return quality_profile_model_settings(song_model, QUALITY_PROFILE_CHART_MASTER, high_turbo_cap=high_turbo_cap)


def quality_profiles_payload() -> dict[str, Any]:
    return {
        QUALITY_PROFILE_OFFICIAL_RAW: {
            "label": "Official raw",
            "summary": "Literal ACE-Step defaults for parity/debugging.",
            "models": {model: quality_profile_model_settings(model, QUALITY_PROFILE_OFFICIAL_RAW) for model in KNOWN_ACE_STEP_MODELS},
            "single_song_takes": 1,
            "album_takes": 1,
        },
        QUALITY_PROFILE_DOCS_DAILY: {
            "label": "Docs Daily",
            "summary": "Simple-mode docs default: auto kiest XL Turbo/Turbo 8/shift 3; gekozen Base/SFT gebruikt 50/shift 1.",
            "models": {model: quality_profile_model_settings(model, QUALITY_PROFILE_DOCS_DAILY) for model in KNOWN_ACE_STEP_MODELS},
            "preferred_model_order": ["acestep-v15-xl-turbo", "acestep-v15-turbo"],
            "single_song_takes": 1,
            "album_takes": 1,
        },
        QUALITY_PROFILE_PREVIEW_FAST: {
            "label": "Laag",
            "summary": "Model-correcte preview: Turbo blijft 8/shift 3; Base/SFT gebruikt de docs-standaard 50/shift 1.",
            "models": {model: quality_profile_model_settings(model, QUALITY_PROFILE_PREVIEW_FAST) for model in KNOWN_ACE_STEP_MODELS},
            "preferred_model_order": ["acestep-v15-xl-sft", "acestep-v15-sft", "acestep-v15-xl-turbo", "acestep-v15-turbo"],
            "single_song_takes": 1,
            "album_takes": 1,
        },
        QUALITY_PROFILE_BALANCED_PRO: {
            "label": "Middel",
            "summary": "Docs-correcte standaard: Turbo 8/shift 3; Base/SFT 50/shift 1.",
            "models": {model: quality_profile_model_settings(model, QUALITY_PROFILE_BALANCED_PRO) for model in KNOWN_ACE_STEP_MODELS},
            "preferred_model_order": ["acestep-v15-xl-sft", "acestep-v15-sft", "acestep-v15-xl-turbo", "acestep-v15-turbo"],
            "single_song_takes": 1,
            "album_takes": 1,
        },
        QUALITY_PROFILE_CHART_MASTER: {
            "label": "Hoog",
            "summary": "Beste standaardkwaliteit: XL/SFT voorkeur met ACE-Step docs-correcte 50 steps en shift 1.",
            "models": {model: quality_profile_model_settings(model, QUALITY_PROFILE_CHART_MASTER) for model in KNOWN_ACE_STEP_MODELS},
            "preferred_model_order": ["acestep-v15-xl-sft", "acestep-v15-sft", "acestep-v15-xl-base", "acestep-v15-base", "acestep-v15-xl-turbo", "acestep-v15-turbo"],
            "single_song_takes": CHART_MASTER_SINGLE_TAKES,
            "album_takes": CHART_MASTER_ALBUM_TAKES,
        },
    }


def docs_best_quality_policy() -> dict[str, Any]:
    profiles = quality_profiles_payload()
    return {
        "version": DOCS_BEST_QUALITY_POLICY_VERSION,
        "default_profile": DEFAULT_QUALITY_PROFILE,
        "default_by_mode": {
            "simple": QUALITY_PROFILE_DOCS_DAILY,
            "custom": QUALITY_PROFILE_CHART_MASTER,
            "album": QUALITY_PROFILE_CHART_MASTER,
            "cover": QUALITY_PROFILE_CHART_MASTER,
            "repaint": QUALITY_PROFILE_CHART_MASTER,
            "extract": QUALITY_PROFILE_CHART_MASTER,
            "lego": QUALITY_PROFILE_CHART_MASTER,
            "complete": QUALITY_PROFILE_CHART_MASTER,
        },
        "max_quality_profile": QUALITY_PROFILE_CHART_MASTER,
        "chart_master_alias": QUALITY_PROFILE_CHART_MASTER,
        "profiles": profiles,
        "standard": "Kwaliteit is model-correct: Laag/Middel/Hoog gebruiken Turbo 8/shift 3 of Base/SFT 50/shift 1.",
        "settings_policy_version": ACE_STEP_SETTINGS_POLICY_VERSION,
        "audio_format": DOCS_BEST_AUDIO_FORMAT,
        "metadata_defaults": {
            "bpm": None,
            "key_scale": "",
            "time_signature": "",
            "duration": -1,
            "vocal_language": "unknown",
            "lock_rule": "Concrete user or AI values are locked; blank/auto values are passed through to ACE-Step auto metadata.",
        },
        "model_step_defaults": dict(DOCS_BEST_MODEL_STEPS),
        "turbo_models": {
            "inference_steps": DOCS_BEST_TURBO_STEPS,
            "optional_high_cap_steps": DOCS_BEST_TURBO_HIGH_CAP_STEPS,
            "guidance_scale": DOCS_BEST_TURBO_GUIDANCE,
            "shift": DOCS_BEST_TURBO_SHIFT,
            "infer_method": "ode",
            "sampler_mode": "heun",
        },
        "sft_base_models": {
            "inference_steps": DOCS_BEST_STANDARD_STEPS,
            "guidance_scale": DOCS_BEST_STANDARD_GUIDANCE,
            "shift": DOCS_BEST_STANDARD_SHIFT,
            "infer_method": "ode",
            "sampler_mode": "heun",
            "single_song_takes": CHART_MASTER_SINGLE_TAKES,
            "album_takes": CHART_MASTER_ALBUM_TAKES,
        },
        "balanced_pro_models": {
            "inference_steps": BALANCED_PRO_STANDARD_STEPS,
            "guidance_scale": DOCS_BEST_STANDARD_GUIDANCE,
            "shift": BALANCED_PRO_STANDARD_SHIFT,
            "infer_method": "ode",
            "sampler_mode": "heun",
        },
        "lm_defaults": dict(DOCS_BEST_LM_DEFAULTS),
        "lm_defaults_by_profile": {
            QUALITY_PROFILE_DOCS_DAILY: {
                **dict(DOCS_BEST_LM_DEFAULTS),
                "ace_lm_model": DOCS_DAILY_DEFAULT_LM_MODEL,
            },
            QUALITY_PROFILE_CHART_MASTER: dict(DOCS_BEST_LM_DEFAULTS),
        },
        "lm_task_policy": {
            "uses_lm_when_controls_active": sorted(DOCS_BEST_LM_TASKS),
            "skips_lm_for_source_tasks": sorted(DOCS_BEST_SOURCE_TASK_LM_SKIPS),
            "planner_writer": "ollama",
            "note": "Ollama writes prompts and lyrics; the ACE-Step 5Hz LM is reserved for official generation controls.",
        },
    }


OFFICIAL_ACE_STEP_MANIFEST: dict[str, Any] = {
    "manifest_version": "2026-04-26",
    "settings_policy_version": ACE_STEP_SETTINGS_POLICY_VERSION,
    "sources": ACE_STEP_MODEL_SOURCES,
    "status_values": ["supported", "guarded", "missing", "unreleased", "not_applicable"],
    "papers": [
        "https://arxiv.org/abs/2506.00045",
        "https://arxiv.org/abs/2602.00744",
    ],
    "tasks": {
        "standard": STANDARD_TASKS,
        "all": ALL_TASKS,
        "turbo_sft_models": STANDARD_TASKS,
        "base_models": ALL_TASKS,
    },
    "dit_models": {
        "acestep-v15-base": {
            "status": "supported",
            "cfg": True,
            "steps": 50,
            "reference_audio": True,
            "tasks": ALL_TASKS,
            "quality": "Medium",
            "diversity": "High",
            "fine_tunability": "Easy",
        },
        "acestep-v15-sft": {
            "status": "supported",
            "cfg": True,
            "steps": 50,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "High",
            "diversity": "Medium",
            "fine_tunability": "Easy",
        },
        "acestep-v15-turbo": {
            "status": "supported",
            "cfg": False,
            "steps": 8,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "Very High",
            "diversity": "Medium",
            "fine_tunability": "Medium",
        },
        "acestep-v15-turbo-shift1": {
            "status": "supported",
            "cfg": False,
            "steps": 8,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "Very High",
            "diversity": "Medium",
            "fine_tunability": "Medium",
        },
        "acestep-v15-turbo-shift3": {
            "status": "supported",
            "cfg": False,
            "steps": 8,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "Very High",
            "diversity": "Medium",
            "fine_tunability": "Medium",
        },
        "acestep-v15-turbo-continuous": {
            "status": "supported",
            "cfg": False,
            "steps": 8,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "Very High",
            "diversity": "Medium",
            "fine_tunability": "Medium",
        },
        "acestep-v15-turbo-rl": {
            "status": "unreleased",
            "cfg": False,
            "steps": 8,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "Very High",
            "diversity": "Medium",
            "fine_tunability": "Medium",
        },
        "acestep-v15-xl-base": {
            "status": "supported",
            "cfg": True,
            "steps": 50,
            "reference_audio": True,
            "tasks": ALL_TASKS,
            "quality": "High",
            "diversity": "High",
            "fine_tunability": "Easy",
        },
        "acestep-v15-xl-sft": {
            "status": "supported",
            "cfg": True,
            "steps": 50,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "Very High",
            "diversity": "Medium",
            "fine_tunability": "Easy",
        },
        "acestep-v15-xl-turbo": {
            "status": "supported",
            "cfg": False,
            "steps": 8,
            "reference_audio": True,
            "tasks": STANDARD_TASKS,
            "quality": "Very High",
            "diversity": "Medium",
            "fine_tunability": "Medium",
        },
    },
    "lm_models": {
        "acestep-5Hz-lm-0.6B": {"status": "supported", "audio_understanding": "Medium", "composition": "Medium", "copy_melody": "Weak"},
        "acestep-5Hz-lm-1.7B": {"status": "supported", "audio_understanding": "Medium", "composition": "Medium", "copy_melody": "Medium"},
        "acestep-5Hz-lm-4B": {"status": "supported", "audio_understanding": "Strong", "composition": "Strong", "copy_melody": "Strong"},
    },
    "model_registry": copy.deepcopy(OFFICIAL_ACE_STEP_MODEL_REGISTRY),
    "core_bundle": {
        "id": OFFICIAL_CORE_MODEL_ID,
        "repo_id": OFFICIAL_MAIN_MODEL_REPO,
        "components": OFFICIAL_MAIN_MODEL_COMPONENTS,
        "status": "supported",
    },
    "helper_models": {
        name: copy.deepcopy(OFFICIAL_ACE_STEP_MODEL_REGISTRY[name])
        for name in OFFICIAL_HELPER_MODEL_IDS
    },
    "lora_models": {
        name: copy.deepcopy(OFFICIAL_ACE_STEP_MODEL_REGISTRY[name])
        for name in OFFICIAL_LORA_MODEL_IDS
    },
    "legacy_models": {
        name: copy.deepcopy(OFFICIAL_ACE_STEP_MODEL_REGISTRY[name])
        for name in OFFICIAL_LEGACY_MODEL_IDS
    },
    "generation_params": {name: {"status": "supported"} for name in OFFICIAL_GENERATION_PARAMS},
    "generation_config": {name: {"status": "supported"} for name in OFFICIAL_GENERATION_CONFIG_FIELDS},
    "settings_registry": {"status": "supported", "version": ACE_STEP_SETTINGS_POLICY_VERSION},
    "api_endpoints": OFFICIAL_API_ENDPOINTS,
    "runtime_controls": OFFICIAL_RUNTIME_CONTROLS,
    "training_features": OFFICIAL_TRAINING_FEATURES,
    "quality_policy": docs_best_quality_policy(),
}

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "acestep-v15-turbo": {
        "label": "Turbo",
        "dropdown_label": "Turbo - best default, fast",
        "summary": "Best default daily driver: balanced, proven, very fast 8-step generation.",
        "best_for": ["Simple", "Custom", "Cover", "Repaint", "Daily driver"],
        "quality": "Very high",
        "speed": "Fastest",
        "vram": "<4GB class",
        "steps": "8",
        "cfg": "No",
        "tasks": STANDARD_TASKS,
        "recommended_for": ["simple", "custom", "cover", "repaint", "album"],
        "warnings": ["Use Base or XL Base for extract, lego, and complete."],
        "notes": "Official docs recommend starting with default turbo as the balanced, thoroughly tested choice.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-turbo-shift1": {
        "label": "Turbo Shift1",
        "dropdown_label": "Turbo Shift1 - richer details, looser semantics",
        "summary": "Official turbo variant distilled on shift=1: richer detail texture with weaker semantic lock.",
        "best_for": ["Detail variants", "Creative texture", "Fast experiments"],
        "quality": "Very high",
        "speed": "Fastest",
        "vram": "<4GB class",
        "steps": "8",
        "cfg": "No",
        "tasks": STANDARD_TASKS,
        "recommended_for": ["custom", "cover", "repaint"],
        "warnings": ["Less semantically locked than default Turbo.", "Use Base or XL Base for extract, lego, and complete."],
        "notes": "Official tutorial describes shift1 as detail-rich but weaker in semantic adherence.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-turbo-shift3": {
        "label": "Turbo Shift3",
        "dropdown_label": "Turbo Shift3 - clear timbre, dry",
        "summary": "Niche turbo variant with clearer, richer timbre and a drier, more minimal arrangement feel.",
        "best_for": ["Clear timbre", "Minimal arrangements", "Fast variants"],
        "quality": "Very high",
        "speed": "Fastest",
        "vram": "<4GB class",
        "steps": "8",
        "cfg": "No",
        "tasks": STANDARD_TASKS,
        "recommended_for": ["custom", "cover", "repaint"],
        "warnings": ["Less balanced than default Turbo.", "Use Base or XL Base for extract, lego, and complete."],
        "notes": "Official tutorial describes shift3 as clearer/richer in timbre, but potentially dry with minimal orchestration.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-turbo-continuous": {
        "label": "Turbo Continuous",
        "dropdown_label": "Turbo Continuous - experimental shift 1-5",
        "summary": "Official experimental turbo variant with continuous shift control from 1 to 5.",
        "best_for": ["Advanced shift tuning", "Fast experiments", "A/B testing"],
        "quality": "Very high",
        "speed": "Fastest",
        "vram": "<4GB class",
        "steps": "8",
        "cfg": "No",
        "tasks": STANDARD_TASKS,
        "recommended_for": ["custom", "cover", "repaint"],
        "warnings": ["Official tutorial calls it experimental and less thoroughly tested.", "Use Base or XL Base for extract, lego, and complete."],
        "notes": "Continuous turbo exposes the widest shift-tuning range but should not replace default Turbo as the safe default.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-sft": {
        "label": "SFT",
        "dropdown_label": "SFT - CFG detail tuning",
        "summary": "Best non-XL choice when you want CFG control, richer detail, and 50-step tuning.",
        "best_for": ["CFG tuning", "Detail", "Prompt adherence"],
        "quality": "High",
        "speed": "Slower",
        "vram": "<4GB class",
        "steps": "50",
        "cfg": "Yes",
        "tasks": STANDARD_TASKS,
        "recommended_for": ["custom", "cover", "repaint"],
        "warnings": ["Slower than Turbo.", "Use Base or XL Base for extract, lego, and complete."],
        "notes": "Official tutorial positions SFT as useful when inference time matters less than CFG and detail control.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-base": {
        "label": "Base",
        "dropdown_label": "Base - all tasks, fine-tuning",
        "summary": "Most flexible 2B model: required for extract, lego, complete, and larger fine-tuning experiments.",
        "best_for": ["Extract", "Lego", "Complete", "Fine-tuning", "All tasks"],
        "quality": "Medium",
        "speed": "Slower",
        "vram": "<4GB class",
        "steps": "50 quality default",
        "cfg": "Yes",
        "tasks": ALL_TASKS,
        "recommended_for": ["extract", "lego", "complete", "lora"],
        "warnings": ["Not the fastest default for simple song generation."],
        "notes": "Official docs call Base the master model for the advanced context tasks and strongest plasticity.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-xl-turbo": {
        "label": "XL Turbo",
        "dropdown_label": "XL Turbo - best 20GB+ daily driver",
        "summary": "Best quality daily driver on large GPUs: XL audio quality with fast 8-step turbo inference.",
        "best_for": ["High quality", "Fast XL", "Large GPU daily driver"],
        "quality": "Very high",
        "speed": "Fastest XL",
        "vram": "12GB+ offload, 20GB+ recommended",
        "steps": "8",
        "cfg": "No",
        "tasks": STANDARD_TASKS,
        "recommended_for": ["simple", "custom", "cover", "repaint", "album"],
        "warnings": ["Needs substantially more VRAM than the 2B Turbo model.", "Use XL Base for extract, lego, and complete."],
        "notes": "Official XL card describes this as 4B quality with turbo speed and recommends 20GB+ without offload.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-xl-sft": {
        "label": "XL SFT",
        "dropdown_label": "XL SFT - highest detail, CFG",
        "summary": "Highest-detail standard XL model for CFG tuning when speed is less important.",
        "best_for": ["Highest detail", "CFG tuning", "XL quality"],
        "quality": "Very high",
        "speed": "Slow",
        "vram": "12GB+ offload, 20GB+ recommended",
        "steps": "50",
        "cfg": "Yes",
        "tasks": STANDARD_TASKS,
        "recommended_for": ["custom", "cover", "repaint"],
        "warnings": ["Slower than XL Turbo.", "Use XL Base for extract, lego, and complete."],
        "notes": "Official XL model zoo lists XL SFT as very high quality with standard tasks and CFG.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-xl-base": {
        "label": "XL Base",
        "dropdown_label": "XL Base - all tasks, XL quality",
        "summary": "Best XL all-rounder: supports extract, lego, complete, CFG, and full advanced task coverage.",
        "best_for": ["All tasks", "Extract", "Lego", "Complete", "XL flexibility"],
        "quality": "High",
        "speed": "Slow",
        "vram": "12GB+ offload, 20GB+ recommended",
        "steps": "50 quality default",
        "cfg": "Yes",
        "tasks": ALL_TASKS,
        "recommended_for": ["extract", "lego", "complete", "lora"],
        "warnings": ["Most demanding DiT option."],
        "notes": "Official XL model zoo lists XL Base for all tasks, including extract, lego, and complete.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-v15-turbo-rl": {
        "label": "Turbo RL",
        "dropdown_label": "Turbo RL - official, unreleased",
        "summary": "Official model-table entry for a future RL turbo checkpoint. It is not released or downloadable yet.",
        "best_for": ["Future RL checkpoint", "Not available yet"],
        "quality": "Very high",
        "speed": "Fastest",
        "vram": "<4GB class",
        "steps": "8",
        "cfg": "No",
        "tasks": STANDARD_TASKS,
        "recommended_for": [],
        "warnings": ["Officially listed as to-be-released; AceJAM will not auto-download it until weights exist."],
        "notes": "Hugging Face ACE-Step 1.5 model zoo lists turbo-rl as to be released.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
        "official_status": "unreleased",
    },
}

LM_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "auto": {
        "label": "Auto",
        "dropdown_label": "Auto - ACE LM when needed",
        "summary": "Ollama stays the planner/writer; ACE-Step LM is selected only for official LM features.",
        "best_for": ["Official LM actions", "Hybrid workflow"],
        "quality": "Recommended",
        "speed": "Balanced",
        "vram": "Auto",
        "steps": "N/A",
        "cfg": "LM CFG 2.0 default",
        "tasks": ["sample mode", "format", "audio understanding", "metadata"],
        "warnings": [],
        "notes": "The Studio routes creative planning and album agents through Ollama, then uses ACE-Step LM only for native LM controls.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "none": {
        "label": "No ACE LM",
        "dropdown_label": "None - manual, fastest",
        "summary": "ACE-Step LM off. Ollama handles all creative planning, writing, formatting, and album-agent work.",
        "best_for": ["Ollama control", "Fastest ACE-Step route", "Precise metadata"],
        "quality": "Manual",
        "speed": "Fastest",
        "vram": "Lowest",
        "steps": "N/A",
        "cfg": "No LM CFG",
        "tasks": ["manual metadata"],
        "warnings": [],
        "notes": "AceJAM uses Ollama instead of ACE-Step 5Hz LM so local uncensored/abliterated models can plan the music.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-5Hz-lm-0.6B": {
        "label": "0.6B LM",
        "dropdown_label": "0.6B LM - low VRAM prototype",
        "summary": "Smallest ACE-Step planner: good for low-VRAM prototyping and faster experiments.",
        "best_for": ["Low VRAM", "Prototype", "Fast planning"],
        "quality": "Medium",
        "speed": "Fast",
        "vram": "Low",
        "steps": "N/A",
        "cfg": "LM CFG supported",
        "tasks": ["query rewrite", "metadata", "audio understanding"],
        "warnings": ["Weak copy-melody capability in official model table."],
        "notes": "Official model table lists medium audio understanding and composition capability.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-5Hz-lm-1.7B": {
        "label": "1.7B LM",
        "dropdown_label": "1.7B LM - best default planner",
        "summary": "Recommended default ACE-Step planner: balanced quality, speed, and memory.",
        "best_for": ["Default planning", "Metadata", "Caption rewrite"],
        "quality": "Medium",
        "speed": "Balanced",
        "vram": "Medium",
        "steps": "N/A",
        "cfg": "LM CFG supported",
        "tasks": ["query rewrite", "metadata", "audio understanding"],
        "warnings": [],
        "notes": "Included in the main ACE-Step 1.5 model set and balanced for everyday planning.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "acestep-5Hz-lm-4B": {
        "label": "4B LM",
        "dropdown_label": "4B LM - strongest planning",
        "summary": "Strongest official ACE-Step planner for complex composition and audio understanding on large GPUs.",
        "best_for": ["Complex planning", "Audio understanding", "Copy melody"],
        "quality": "Strong",
        "speed": "Slow",
        "vram": "High",
        "steps": "N/A",
        "cfg": "LM CFG supported",
        "tasks": ["query rewrite", "metadata", "audio understanding", "composition"],
        "warnings": ["Highest LM memory cost."],
        "notes": "Official model table lists strong audio understanding, composition, and copy-melody capability.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
}

SUPPORTED_AUDIO_FORMATS = {"wav", "flac", "ogg"}
OFFICIAL_AUDIO_FORMATS = {"flac", "mp3", "opus", "aac", "wav", "wav32"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac", ".m4a"}
MAX_BATCH_SIZE = 8

FAST_HANDLER_FIELDS = {
    "audio_code_string",
    "audio_cover_strength",
    "auto_lrc",
    "auto_score",
    "batch_size",
    "bpm",
    "caption",
    "cfg_interval_end",
    "cfg_interval_start",
    "duration",
    "guidance_scale",
    "infer_method",
    "inference_steps",
    "instruction",
    "key_scale",
    "lyrics",
    "repainting_end",
    "repainting_start",
    "return_audio_codes",
    "seed",
    "seeds",
    "shift",
    "task_type",
    "time_signature",
    "timesteps",
    "track_name",
    "track_names",
    "use_adg",
    "use_random_seed",
    "vocal_language",
}

OFFICIAL_ONLY_FIELDS = {
    "allow_lm_batch",
    "analysis_only",
    "chunk_mask_mode",
    "compile_model",
    "constrained_decoding_debug",
    "cover_noise_strength",
    "dcw_enabled",
    "dcw_high_scaler",
    "dcw_mode",
    "dcw_scaler",
    "dcw_wavelet",
    "device",
    "dtype",
    "enable_normalization",
    "fade_in_duration",
    "fade_out_duration",
    "global_caption",
    "latent_rescale",
    "latent_shift",
    "lm_batch_chunk_size",
    "lm_backend",
    "lm_cfg_scale",
    "lm_device",
    "lm_dtype",
    "lm_negative_prompt",
    "lm_offload_to_cpu",
    "lm_repetition_penalty",
    "lm_temperature",
    "lm_top_k",
    "lm_top_p",
    "offload_dit_to_cpu",
    "offload_to_cpu",
    "repaint_latent_crossfade_frames",
    "repaint_mode",
    "repaint_strength",
    "repaint_wav_crossfade_sec",
    "retake_seed",
    "retake_variance",
    "sample_mode",
    "sample_query",
    "sampler_mode",
    "flow_edit_morph",
    "flow_edit_source_caption",
    "flow_edit_source_lyrics",
    "flow_edit_n_min",
    "flow_edit_n_max",
    "flow_edit_n_avg",
    "full_analysis_only",
    "extract_codes_only",
    "is_format_caption",
    "thinking",
    "track_name",
    "track_classes",
    "use_flash_attention",
    "use_constrained_decoding",
    "use_cot_caption",
    "use_cot_language",
    "use_cot_lyrics",
    "use_cot_metas",
    "use_format",
    "use_random_seed",
    "use_tiled_decode",
    "velocity_ema_factor",
    "velocity_norm_threshold",
}

OFFICIAL_FIELD_DEFAULTS: dict[str, Any] = {
    "allow_lm_batch": False,
    "analysis_only": False,
    "chunk_mask_mode": "auto",
    "compile_model": False,
    "constrained_decoding_debug": False,
    "cover_noise_strength": 0.0,
    "dcw_enabled": True,
    "dcw_mode": "double",
    "dcw_scaler": 0.05,
    "dcw_high_scaler": 0.02,
    "dcw_wavelet": "haar",
    "device": "auto",
    "dtype": "auto",
    "enable_normalization": True,
    "fade_in_duration": 0.0,
    "fade_out_duration": 0.0,
    "global_caption": "",
    "instrumental": False,
    "latent_rescale": 1.0,
    "latent_shift": 0.0,
    "lm_batch_chunk_size": 8,
    "lm_backend": DOCS_BEST_LM_DEFAULTS["lm_backend"],
    "lm_cfg_scale": DOCS_BEST_LM_DEFAULTS["lm_cfg_scale"],
    "lm_device": "auto",
    "lm_dtype": "auto",
    "lm_negative_prompt": "NO USER INPUT",
    "lm_offload_to_cpu": False,
    "lm_repetition_penalty": 1.0,
    "lm_temperature": DOCS_BEST_LM_DEFAULTS["lm_temperature"],
    "lm_top_k": DOCS_BEST_LM_DEFAULTS["lm_top_k"],
    "lm_top_p": DOCS_BEST_LM_DEFAULTS["lm_top_p"],
    "mp3_bitrate": "128k",
    "mp3_sample_rate": 48000,
    "repaint_latent_crossfade_frames": 10,
    "repaint_mode": "balanced",
    "repaint_strength": 0.5,
    "repaint_wav_crossfade_sec": 0.0,
    "retake_seed": "",
    "retake_variance": 0.0,
    "sample_mode": False,
    "sample_query": "",
    "sampler_mode": "heun",
    "flow_edit_morph": False,
    "flow_edit_source_caption": "",
    "flow_edit_source_lyrics": "",
    "flow_edit_n_min": 0.0,
    "flow_edit_n_max": 1.0,
    "flow_edit_n_avg": 1,
    "full_analysis_only": False,
    "extract_codes_only": False,
    "is_format_caption": False,
    "thinking": False,
    "track_name": "",
    "track_classes": [],
    "offload_dit_to_cpu": False,
    "offload_to_cpu": False,
    "use_flash_attention": "auto",
    "use_constrained_decoding": DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"],
    "use_cot_caption": DOCS_BEST_LM_DEFAULTS["use_cot_caption"],
    "use_cot_language": DOCS_BEST_LM_DEFAULTS["use_cot_language"],
    "use_cot_lyrics": False,
    "use_cot_metas": DOCS_BEST_LM_DEFAULTS["use_cot_metas"],
    "use_format": False,
    "use_random_seed": True,
    "use_tiled_decode": True,
    "velocity_ema_factor": 0.0,
    "velocity_norm_threshold": 0.0,
}

ACE_STEP_OFFICIAL_DEFAULTS: dict[str, Any] = {
    "task_type": "text2music",
    "instruction": "Fill the audio semantic mask based on the given conditions:",
    "reference_audio": None,
    "src_audio": None,
    "audio_codes": "",
    "caption": "",
    "global_caption": "",
    "lyrics": "",
    "instrumental": False,
    "vocal_language": "unknown",
    "bpm": None,
    "keyscale": "",
    "timesignature": "",
    "duration": -1.0,
    "enable_normalization": True,
    "normalization_db": -1.0,
    "fade_in_duration": 0.0,
    "fade_out_duration": 0.0,
    "latent_shift": 0.0,
    "latent_rescale": 1.0,
    "inference_steps": 8,
    "seed": -1,
    "guidance_scale": 7.0,
    "use_adg": False,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "shift": 3.0,
    "infer_method": "ode",
    "sampler_mode": "euler",
    "velocity_norm_threshold": 0.0,
    "velocity_ema_factor": 0.0,
    "dcw_enabled": True,
    "dcw_mode": "double",
    "dcw_scaler": 0.05,
    "dcw_high_scaler": 0.02,
    "dcw_wavelet": "haar",
    "timesteps": None,
    "repainting_start": 0.0,
    "repainting_end": -1.0,
    "chunk_mask_mode": "auto",
    "repaint_latent_crossfade_frames": 10,
    "repaint_wav_crossfade_sec": 0.0,
    "repaint_mode": "balanced",
    "repaint_strength": 0.5,
    "retake_seed": None,
    "retake_variance": 0.0,
    "flow_edit_morph": False,
    "flow_edit_source_caption": "",
    "flow_edit_source_lyrics": "",
    "flow_edit_n_min": 0.0,
    "flow_edit_n_max": 1.0,
    "flow_edit_n_avg": 1,
    "audio_cover_strength": 1.0,
    "cover_noise_strength": 0.0,
    "thinking": True,
    "lm_temperature": 0.85,
    "lm_cfg_scale": 2.0,
    "lm_top_k": 0,
    "lm_top_p": 0.9,
    "lm_negative_prompt": "NO USER INPUT",
    "use_cot_metas": True,
    "use_cot_caption": True,
    "use_cot_lyrics": False,
    "use_cot_language": True,
    "use_constrained_decoding": True,
    "cot_bpm": None,
    "cot_keyscale": "",
    "cot_timesignature": "",
    "cot_duration": None,
    "cot_vocal_language": "unknown",
    "cot_caption": "",
    "cot_lyrics": "",
    "batch_size": 2,
    "allow_lm_batch": False,
    "use_random_seed": True,
    "seeds": None,
    "lm_batch_chunk_size": 8,
    "constrained_decoding_debug": False,
    "audio_format": "flac",
    "mp3_bitrate": "128k",
    "mp3_sample_rate": 48000,
}

ACEJAM_EXTENSION_DEFAULTS: dict[str, Any] = {
    "quality_profile": DEFAULT_QUALITY_PROFILE,
    "ace_lm_model": DOCS_BEST_DEFAULT_LM_MODEL,
    "lm_model_path": DOCS_BEST_DEFAULT_LM_MODEL,
    "lm_backend": DOCS_BEST_DEFAULT_LM_BACKEND,
    "lm_repetition_penalty": 1.0,
    "analysis_only": False,
    "full_analysis_only": False,
    "extract_codes_only": False,
    "use_tiled_decode": True,
    "is_format_caption": False,
    "track_name": "",
    "track_classes": [],
    "sample_mode": False,
    "sample_query": "",
    "use_format": False,
    "device": "auto",
    "dtype": "auto",
    "compile_model": False,
    "use_flash_attention": "auto",
    "offload_to_cpu": False,
    "offload_dit_to_cpu": False,
    "lm_device": "auto",
    "lm_dtype": "auto",
    "lm_offload_to_cpu": False,
}

ACE_STEP_SETTING_SECTIONS: dict[str, list[str]] = {
    "core": ["task_type", "instruction", "caption", "global_caption", "lyrics", "instrumental"],
    "music_metadata": ["vocal_language", "bpm", "keyscale", "timesignature", "duration"],
    "model_quality": ["quality_profile", "inference_steps", "seed", "guidance_scale", "shift", "infer_method", "sampler_mode", "timesteps"],
    "diffusion": [
        "use_adg",
        "cfg_interval_start",
        "cfg_interval_end",
        "velocity_norm_threshold",
        "velocity_ema_factor",
        "dcw_enabled",
        "dcw_mode",
        "dcw_scaler",
        "dcw_high_scaler",
        "dcw_wavelet",
        "retake_seed",
        "retake_variance",
        "latent_shift",
        "latent_rescale",
    ],
    "source_edit_audio": [
        "reference_audio",
        "src_audio",
        "audio_codes",
        "audio_cover_strength",
        "cover_noise_strength",
        "repainting_start",
        "repainting_end",
        "chunk_mask_mode",
        "repaint_mode",
        "repaint_strength",
        "repaint_latent_crossfade_frames",
        "repaint_wav_crossfade_sec",
        "flow_edit_morph",
        "flow_edit_source_caption",
        "flow_edit_source_lyrics",
        "flow_edit_n_min",
        "flow_edit_n_max",
        "flow_edit_n_avg",
    ],
    "lm_cot": [
        "ace_lm_model",
        "lm_model_path",
        "lm_backend",
        "thinking",
        "use_format",
        "sample_mode",
        "sample_query",
        "lm_temperature",
        "lm_cfg_scale",
        "lm_top_k",
        "lm_top_p",
        "lm_repetition_penalty",
        "lm_negative_prompt",
        "use_cot_metas",
        "use_cot_caption",
        "use_cot_lyrics",
        "use_cot_language",
        "use_constrained_decoding",
        "cot_bpm",
        "cot_keyscale",
        "cot_timesignature",
        "cot_duration",
        "cot_vocal_language",
        "cot_caption",
        "cot_lyrics",
    ],
    "api_service": [
        "analysis_only",
        "full_analysis_only",
        "extract_codes_only",
        "use_tiled_decode",
        "is_format_caption",
        "track_name",
        "track_classes",
    ],
    "output": [
        "batch_size",
        "allow_lm_batch",
        "use_random_seed",
        "seeds",
        "lm_batch_chunk_size",
        "constrained_decoding_debug",
        "audio_format",
        "mp3_bitrate",
        "mp3_sample_rate",
        "enable_normalization",
        "normalization_db",
        "fade_in_duration",
        "fade_out_duration",
    ],
    "runtime": [
        "device",
        "dtype",
        "compile_model",
        "use_flash_attention",
        "offload_to_cpu",
        "offload_dit_to_cpu",
        "lm_device",
        "lm_dtype",
        "lm_offload_to_cpu",
    ],
}

ACE_STEP_READ_ONLY_LM_OUTPUT_FIELDS = {
    "cot_bpm",
    "cot_keyscale",
    "cot_timesignature",
    "cot_duration",
    "cot_vocal_language",
    "cot_caption",
    "cot_lyrics",
}

ACE_STEP_TASK_REQUIRED_FIELDS: dict[str, list[str]] = {
    "text2music": ["caption_or_lyrics"],
    "cover": ["src_audio", "caption"],
    "cover-nofsq": ["src_audio", "caption"],
    "repaint": ["src_audio", "caption", "repainting_start", "repainting_end"],
    "extract": ["src_audio", "instruction"],
    "lego": ["src_audio", "instruction", "track_name"],
    "complete": ["src_audio", "instruction", "caption", "track_names"],
}

ACE_STEP_SOURCE_LOCKED_DURATION_TASKS = {"cover", "cover-nofsq", "repaint", "lego", "extract", "complete"}


def _field_section(field: str) -> str:
    for section, fields in ACE_STEP_SETTING_SECTIONS.items():
        if field in fields:
            return section
    return "other"


def _field_status(field: str) -> str:
    if field in ACE_STEP_READ_ONLY_LM_OUTPUT_FIELDS:
        return "read_only_lm_output"
    if field == "use_cot_lyrics":
        return "reserved"
    if field in OFFICIAL_ONLY_FIELDS:
        return "official_only"
    if field in ACEJAM_EXTENSION_DEFAULTS:
        return "advanced"
    return "active"


def _field_options(field: str) -> list[Any]:
    options: dict[str, list[Any]] = {
        "task_type": ALL_TASKS,
        "audio_format": sorted(OFFICIAL_AUDIO_FORMATS),
        "infer_method": ["ode", "sde"],
        "sampler_mode": ["euler", "heun"],
        "dcw_mode": ["low", "high", "double", "pix"],
        "dcw_wavelet": ["haar", "db4", "sym8", "coif1", "bior2.2"],
        "chunk_mask_mode": ["auto", "explicit"],
        "repaint_mode": ["balanced", "conservative", "aggressive"],
        "vocal_language": ["unknown", "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "nl", "ar", "ru"],
        "keyscale": [KEYSCALE_AUTO_VALUE, *VALID_KEY_SCALES],
        "key_scale": [KEYSCALE_AUTO_VALUE, *VALID_KEY_SCALES],
        "lm_backend": ["mlx", "pt", "vllm"],
        "ace_lm_model": ACE_STEP_LM_MODELS,
        "lm_model_path": ACE_STEP_LM_MODELS,
        "quality_profile": QUALITY_PROFILES,
        "track_name": TRACK_NAMES,
        "track_classes": TRACK_NAMES,
    }
    return list(options.get(field, []))


def _field_range(field: str) -> list[float] | None:
    ranges: dict[str, list[float]] = {
        "bpm": [BPM_MIN, BPM_MAX],
        "duration": [10, 600],
        "inference_steps": [1, 200],
        "guidance_scale": [1.0, 15.0],
        "shift": [1.0, 5.0],
        "cfg_interval_start": [0.0, 1.0],
        "cfg_interval_end": [0.0, 1.0],
        "audio_cover_strength": [0.0, 1.0],
        "cover_noise_strength": [0.0, 1.0],
        "repaint_strength": [0.0, 1.0],
        "dcw_scaler": [0.0, 0.2],
        "dcw_high_scaler": [0.0, 0.2],
        "retake_variance": [0.0, 1.0],
        "flow_edit_n_min": [0.0, 1.0],
        "flow_edit_n_max": [0.0, 1.0],
        "flow_edit_n_avg": [1, 16],
        "normalization_db": [-24.0, 0.0],
        "fade_in_duration": [0.0, 20.0],
        "fade_out_duration": [0.0, 20.0],
        "latent_shift": [-2.0, 2.0],
        "latent_rescale": [0.1, 3.0],
        "lm_temperature": [0.0, 2.0],
        "lm_cfg_scale": [0.0, 10.0],
        "lm_top_k": [0, 200],
        "lm_top_p": [0.0, 1.0],
        "lm_repetition_penalty": [0.1, 4.0],
        "batch_size": [1, MAX_BATCH_SIZE],
        "lm_batch_chunk_size": [1, 64],
    }
    return ranges.get(field)


def _field_note(field: str) -> str:
    notes = {
        "caption": f"Official request budget: less than {ACE_STEP_CAPTION_CHAR_LIMIT} characters.",
        "lyrics": f"Official request budget: less than {ACE_STEP_LYRICS_CHAR_LIMIT} characters.",
        "duration": "Official range is 10-600 seconds; source-audio tasks lock duration to the source where ACE-Step does that internally.",
        "quality_profile": "MLX Media profile selector: laag/preview_fast, middel/balanced_pro, hoog/chart_master, plus official_raw for parity debugging.",
        "inference_steps": "ACE-Step docs-correct defaults: Turbo uses 8 steps; Base/SFT/XL SFT use 50 steps for normal/high quality.",
        "guidance_scale": "Only effective for non-turbo models.",
        "shift": "ACE-Step docs-correct defaults: shift 3.0 for Turbo, shift 1.0 for Base/SFT/XL SFT. Custom timesteps override shift.",
        "timesteps": "Overrides inference_steps and shift when present.",
        "audio_format": "Official runner supports flac/mp3/opus/aac/wav/wav32; fast in-process runner supports a smaller subset.",
        "audio_cover_strength": "Higher values keep more source structure; lower values transform more freely.",
        "cover_noise_strength": "Vendor code: 0 is no cover/pure noise, 1 is closest to source. Keep 0 unless intentionally experimenting.",
        "repaint_strength": "Vendor code: in balanced mode, 0.0 is aggressive and 1.0 is conservative.",
        "dcw_enabled": "Vendor 1.5 DCW wavelet-domain correction defaults on; changing it requires the official runner.",
        "dcw_mode": "Vendor-supported DCW modes: low, high, double, pix. Double is the tuned default.",
        "retake_variance": "Retake variation control. 0 is a no-op; values above 0 use retake_seed when supplied.",
        "flow_edit_morph": "Flow-edit overlay for cover/cover-nofsq: source caption/lyrics describe the original, target caption/lyrics describe the destination.",
        "analysis_only": "Official /release_task analysis branch: return LM metadata without rendering audio.",
        "full_analysis_only": "Official /release_task deep analysis branch: encode source audio and return metadata/codes without rendering.",
        "extract_codes_only": "Official /release_task helper: convert source audio to 5Hz semantic codes without rendering.",
        "use_tiled_decode": "Official API/server decode control. Guarded unless the active runner exposes tiled decode.",
        "is_format_caption": "Official Gradio/API state: caption has already been formatted by the 5Hz LM.",
        "track_name": "Official extract/lego stem selector. Values come from ACE-Step TRACK_NAMES.",
        "track_classes": "Official complete-task stem list. Values come from ACE-Step TRACK_NAMES.",
        "thinking": "Official LM thinking is ignored for cover, repaint and extract.",
        "use_cot_lyrics": "Reserved/future in ACE-Step docs; AceJAM keeps it off by default.",
        "cot_caption": "Read-only value generated by the ACE-Step LM when COT is active.",
        "cot_lyrics": "Read-only value generated by the ACE-Step LM when COT is active.",
        "lm_backend": "macOS Apple Silicon uses MLX by default per ACE-Step macOS scripts.",
    }
    return notes.get(field, "")


class AceStepSettingsRegistry:
    """Central docs-parity registry for ACE-Step request, config, and runtime controls."""

    version = ACE_STEP_SETTINGS_POLICY_VERSION

    @classmethod
    def settings(cls) -> dict[str, dict[str, Any]]:
        fields = list(dict.fromkeys(OFFICIAL_GENERATION_PARAMS + OFFICIAL_GENERATION_CONFIG_FIELDS + list(ACEJAM_EXTENSION_DEFAULTS)))
        settings: dict[str, dict[str, Any]] = {}
        for field in fields:
            settings[field] = {
                "field": field,
                "section": _field_section(field),
                "status": _field_status(field),
                "official_default": ACE_STEP_OFFICIAL_DEFAULTS.get(field),
                "acejam_default": OFFICIAL_FIELD_DEFAULTS.get(field, ACEJAM_EXTENSION_DEFAULTS.get(field, ACE_STEP_OFFICIAL_DEFAULTS.get(field))),
                "options": _field_options(field),
                "range": _field_range(field),
                "note": _field_note(field),
                "official_generation_param": field in OFFICIAL_GENERATION_PARAMS,
                "official_generation_config": field in OFFICIAL_GENERATION_CONFIG_FIELDS,
                "acejam_extension": field in ACEJAM_EXTENSION_DEFAULTS,
            }
        return settings

    @classmethod
    def docs_recommended(cls) -> dict[str, Any]:
        return {
            "turbo": {
                "inference_steps": DOCS_BEST_TURBO_STEPS,
                "optional_high_cap_steps": DOCS_BEST_TURBO_HIGH_CAP_STEPS,
                "shift": DOCS_BEST_TURBO_SHIFT,
                "guidance_scale": DOCS_BEST_TURBO_GUIDANCE,
                "effective_guidance": False,
            },
            "base_sft": {
                "inference_steps_range": [50, 64],
                "chart_master_default_steps": CHART_MASTER_STANDARD_STEPS,
                "balanced_pro_default_steps": BALANCED_PRO_STANDARD_STEPS,
                "guidance_scale_range": [5.0, 9.0],
                "acejam_default_guidance": DOCS_BEST_STANDARD_GUIDANCE,
                "chart_master_shift": CHART_MASTER_STANDARD_SHIFT,
                "balanced_pro_shift": BALANCED_PRO_STANDARD_SHIFT,
                "effective_guidance": True,
            },
            "lm": {
                "lm_temperature": 0.85,
                "lm_top_p": 0.9,
                "lm_top_k": 0,
                "lm_cfg_scale_local": 2.0,
                "lm_cfg_scale_api": 2.5,
                "use_cot_lyrics": "reserved",
            },
        }

    @classmethod
    def acejam_quality(cls) -> dict[str, Any]:
        return {
            "version": DOCS_BEST_QUALITY_POLICY_VERSION,
            "default_profile": DEFAULT_QUALITY_PROFILE,
            "audio_format": DOCS_BEST_AUDIO_FORMAT,
            "models": {model: docs_best_model_settings(model) for model in KNOWN_ACE_STEP_MODELS},
            "quality_profiles": quality_profiles_payload(),
            "lm_defaults": dict(DOCS_BEST_LM_DEFAULTS),
        }

    @classmethod
    def coverage(cls) -> dict[str, Any]:
        settings = cls.settings()
        manifest_fields = set(OFFICIAL_GENERATION_PARAMS + OFFICIAL_GENERATION_CONFIG_FIELDS)
        missing_fields = sorted(field for field in manifest_fields if field not in settings)
        known_controls = set(settings)
        known_controls.update(OFFICIAL_RUNTIME_CONTROLS)
        known_controls.update(OFFICIAL_API_ENDPOINTS)
        known_controls.update(OFFICIAL_HELPER_FUNCTIONS)
        known_controls.update(OFFICIAL_RESULT_FIELDS)
        return {
            "status": "complete" if not missing_fields else "incomplete",
            "missing_fields": missing_fields,
            "missing_count": len(missing_fields),
            "generation_params_count": len(OFFICIAL_GENERATION_PARAMS),
            "generation_config_count": len(OFFICIAL_GENERATION_CONFIG_FIELDS),
            "api_endpoints": copy.deepcopy(OFFICIAL_API_ENDPOINTS),
            "runtime_controls": copy.deepcopy(OFFICIAL_RUNTIME_CONTROLS),
            "helper_functions": copy.deepcopy(OFFICIAL_HELPER_FUNCTIONS),
            "result_fields": copy.deepcopy(OFFICIAL_RESULT_FIELDS),
            "known_control_count": len(known_controls),
        }

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return {
            "version": cls.version,
            "sources": ACE_STEP_MODEL_SOURCES,
            "settings": cls.settings(),
            "sections": copy.deepcopy(ACE_STEP_SETTING_SECTIONS),
            "profiles": {
                "official_defaults": dict(ACE_STEP_OFFICIAL_DEFAULTS),
                "official_raw": quality_profiles_payload()[QUALITY_PROFILE_OFFICIAL_RAW],
                "docs_daily": quality_profiles_payload()[QUALITY_PROFILE_DOCS_DAILY],
                "docs_recommended": cls.docs_recommended(),
                "preview_fast": quality_profiles_payload()[QUALITY_PROFILE_PREVIEW_FAST],
                "balanced_pro": quality_profiles_payload()[QUALITY_PROFILE_BALANCED_PRO],
                "chart_master": quality_profiles_payload()[QUALITY_PROFILE_CHART_MASTER],
                "acejam_quality": cls.acejam_quality(),
            },
            "default_quality_profile": DEFAULT_QUALITY_PROFILE,
            "coverage": cls.coverage(),
            "task_policy": {
                "all_tasks": list(ALL_TASKS),
                "base_model_tasks": list(ALL_TASKS),
                "turbo_sft_tasks": list(STANDARD_TASKS),
                "lm_uses": sorted(DOCS_BEST_LM_TASKS),
                "lm_skips": sorted(DOCS_BEST_SOURCE_TASK_LM_SKIPS),
                "source_locked_duration": sorted(ACE_STEP_SOURCE_LOCKED_DURATION_TASKS),
                "required_fields": copy.deepcopy(ACE_STEP_TASK_REQUIRED_FIELDS),
            },
            "runner_support": {
                "official_audio_formats": sorted(OFFICIAL_AUDIO_FORMATS),
                "fast_audio_formats": sorted(SUPPORTED_AUDIO_FORMATS),
                "caption_char_limit": ACE_STEP_CAPTION_CHAR_LIMIT,
                "lyrics_char_limit": ACE_STEP_LYRICS_CHAR_LIMIT,
            },
            "pro_quality": pro_quality_policy(),
        }

    @classmethod
    def compliance(cls, payload: dict[str, Any], *, task_type: str, song_model: str, runner_plan: str = "") -> dict[str, Any]:
        task = normalize_task_type(task_type)
        model = str(song_model or "").strip()
        runner = str(runner_plan or payload.get("runner_plan") or "").strip() or "fast"
        settings = cls.settings()
        statuses: dict[str, str] = {}
        notes: list[str] = []
        active: list[str] = []
        ignored: list[str] = []
        official_only: list[str] = []
        unsupported: list[str] = []
        read_only: list[str] = []
        for field in settings:
            if field not in payload:
                continue
            status = "active"
            if field in ACE_STEP_READ_ONLY_LM_OUTPUT_FIELDS:
                status = "read_only_lm_output"
            elif field == "use_cot_lyrics":
                status = "reserved"
            elif field == "duration" and task in ACE_STEP_SOURCE_LOCKED_DURATION_TASKS:
                status = "source_locked"
            elif field in {"thinking", "use_cot_metas", "use_cot_caption", "use_cot_language"} and task in DOCS_BEST_SOURCE_TASK_LM_SKIPS:
                status = "ignored_for_task"
            elif field == "guidance_scale" and is_turbo_song_model(model):
                status = "ignored_for_task"
            elif field == "use_adg" and not is_base_song_model(model):
                status = "ignored_for_task"
            elif field == "timesteps" and payload.get(field):
                notes.append("timesteps_override_steps_shift")
            elif field == "audio_format":
                fmt = str(payload.get(field) or "").lower()
                if runner == "fast" and fmt and fmt not in SUPPORTED_AUDIO_FORMATS:
                    status = "unsupported"
            elif field in OFFICIAL_ONLY_FIELDS and runner == "fast":
                status = "official_only"
            if status == "active":
                active.append(field)
            elif status == "ignored_for_task":
                ignored.append(field)
            elif status == "unsupported":
                unsupported.append(field)
            elif status == "read_only_lm_output":
                read_only.append(field)
            elif status == "official_only":
                official_only.append(field)
            statuses[field] = status
        if task in ACE_STEP_SOURCE_LOCKED_DURATION_TASKS:
            notes.append("duration_source_locked")
        if task in DOCS_BEST_SOURCE_TASK_LM_SKIPS:
            notes.append("ace_step_lm_ignored_for_task")
        return {
            "version": cls.version,
            "task_type": task,
            "song_model": model,
            "runner_plan": runner,
            "field_status": statuses,
            "active": sorted(active),
            "ignored": sorted(ignored),
            "official_only": sorted(official_only),
            "unsupported": sorted(unsupported),
            "read_only": sorted(read_only),
            "notes": sorted(set(notes)),
            "valid": not unsupported,
        }


def ace_step_settings_registry() -> dict[str, Any]:
    return AceStepSettingsRegistry.as_dict()


def ace_step_settings_compliance(payload: dict[str, Any], *, task_type: str, song_model: str, runner_plan: str = "") -> dict[str, Any]:
    return AceStepSettingsRegistry.compliance(payload, task_type=task_type, song_model=song_model, runner_plan=runner_plan)


def pro_quality_policy() -> dict[str, Any]:
    return {
        "version": PRO_QUALITY_AUDIT_VERSION,
        "default_profile": DEFAULT_QUALITY_PROFILE,
        "single_song_takes": CHART_MASTER_SINGLE_TAKES,
        "album_takes": CHART_MASTER_ALBUM_TAKES,
        "audio_targets": dict(PRO_AUDIO_TARGETS),
        "checks": [
            "caption_budget",
            "lyrics_budget",
            "lyrics_presence",
            "hook_or_chorus",
            "section_map",
            "metadata_presence",
            "settings_compliance",
            "runtime_cost",
            "audio_quality",
            "metadata_adherence",
        ],
    }


def _quality_status_from_score(score: int, *, failure: bool = False) -> str:
    if failure:
        return "fail"
    if score >= 85:
        return "pass"
    if score >= 70:
        return "warn"
    return "review"


def hit_readiness_report(payload: dict[str, Any], *, task_type: str | None = None, song_model: str = "", runner_plan: str = "") -> dict[str, Any]:
    task = normalize_task_type(task_type or payload.get("task_type"))
    caption = str(payload.get("caption") or "")
    lyrics = str(payload.get("lyrics") or "")
    instrumental = parse_bool(payload.get("instrumental"), lyrics.strip().lower() == "[instrumental]")
    sample_mode = parse_bool(payload.get("sample_mode"), False) or bool(str(payload.get("sample_query") or "").strip())
    checks: list[dict[str, Any]] = []

    def add(check_id: str, status: str, detail: str, points: int = 0) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail, "points": points})

    caption_len = len(caption)
    add(
        "caption_budget",
        "pass" if caption_len <= ACE_STEP_CAPTION_CHAR_LIMIT else "fail",
        f"{caption_len}/{ACE_STEP_CAPTION_CHAR_LIMIT} chars",
        12,
    )

    lyrics_len = len(lyrics)
    if instrumental or task in DOCS_BEST_SOURCE_TASK_LM_SKIPS:
        add("lyrics_budget", "pass", "lyrics not required for this mode", 10)
    else:
        add(
            "lyrics_budget",
            "pass" if lyrics_len <= ACE_STEP_LYRICS_CHAR_LIMIT else "fail",
            f"{lyrics_len}/{ACE_STEP_LYRICS_CHAR_LIMIT} chars",
            12,
        )

    vocal_lyrics = has_vocal_lyrics(lyrics)
    if task == "text2music" and not instrumental and not sample_mode:
        add("lyrics_presence", "pass" if vocal_lyrics else "fail", "vocal lyrics present" if vocal_lyrics else "vocal lyrics missing", 16)
    else:
        add("lyrics_presence", "pass", "not required for this mode", 8)

    hook_hits = sum(
        1
        for section in INLINE_LYRIC_SECTION_RE.findall(lyrics)
        if any(token in lyric_section_key(section) for token in ("chorus", "hook", "refrain"))
    )
    add("hook_or_chorus", "pass" if instrumental or hook_hits else "warn", f"{hook_hits} hook/chorus section(s)", 10)

    section_hits = len(INLINE_LYRIC_SECTION_RE.findall(lyrics))
    expected_sections = 2 if task == "text2music" and not instrumental else 0
    add("section_map", "pass" if section_hits >= expected_sections else "warn", f"{section_hits} section tag(s)", 10)

    caption_leak_count = len(
        re.findall(
            r"(?i)(\[[^\]]*(?:verse|chorus|hook|bridge|intro|outro)[^\]]*\]|\b(?:verse|lyrics|naming drop|track\s+\d+|album|bpm|keyscale|duration|metadata|produced by|artist|description|tags)\s*:)",
            caption,
        )
    )
    add("caption_integrity", "pass" if caption_leak_count == 0 else "fail", f"{caption_leak_count} caption leak marker(s)", 10)

    meta_leak_count = sum(1 for line in lyrics.splitlines() if _META_LEAK_LINE_RE.search(line.strip()))
    add("no_meta_leakage", "pass" if meta_leak_count == 0 else "fail", f"{meta_leak_count} meta/planning line(s)", 14)

    fallback_artifact_count = len(
        re.findall(
            r"(?i)\b(?:morning finds the|light is leaning through the door|kept the receipt from the life before|now i want the sound and nothing more|the you|the was|the are|the is)\b",
            lyrics,
        )
    )
    add("no_fallback_artifacts", "pass" if fallback_artifact_count == 0 else "fail", f"{fallback_artifact_count} fallback artifact(s)", 10)

    key_value = payload.get("key_scale", payload.get("keyscale"))
    signature_value = payload.get("time_signature", payload.get("timesignature"))
    metadata_present = bool(payload.get("bpm") not in [None, ""] and key_value not in [None, ""] and signature_value not in [None, ""])
    add("metadata_presence", "pass" if metadata_present else "warn", "BPM/key/time present" if metadata_present else "metadata is auto or missing", 8)

    gate = payload.get("payload_quality_gate") if isinstance(payload.get("payload_quality_gate"), dict) else {}
    gate_status = str(payload.get("payload_gate_status") or gate.get("status") or "").strip()
    if gate_status:
        add(
            "album_payload_gate",
            "pass" if gate_status == "pass" else "warn" if gate_status == "auto_repair" else "fail",
            gate_status,
            12,
        )

    settings = payload.get("settings_compliance") or {}
    settings_valid = settings.get("valid", True) is not False
    add("settings_compliance", "pass" if settings_valid else "warn", settings.get("version") or ACE_STEP_SETTINGS_POLICY_VERSION, 10)

    score = 0
    failure = False
    for check in checks:
        status = check["status"]
        points = int(check.get("points") or 0)
        if status == "pass":
            score += points
        elif status == "warn":
            score += max(0, int(points * 0.55))
        else:
            failure = True
    max_score = sum(int(check.get("points") or 0) for check in checks) or 1
    normalized_score = int(round(score / max_score * 100))
    issues = [f"{check['id']}: {check['detail']}" for check in checks if check["status"] != "pass"]
    return {
        "version": PRO_QUALITY_AUDIT_VERSION,
        "status": _quality_status_from_score(normalized_score, failure=failure),
        "score": normalized_score,
        "task_type": task,
        "song_model": str(song_model or payload.get("song_model") or ""),
        "runner_plan": str(runner_plan or payload.get("runner_plan") or ""),
        "checks": checks,
        "issues": issues,
        "caption_char_count": caption_len,
        "lyrics_char_count": lyrics_len,
        "section_count": section_hits,
        "hook_count": hook_hits,
    }


def runtime_planner_report(payload: dict[str, Any], *, task_type: str | None = None, song_model: str = "", quality_profile: str | None = None) -> dict[str, Any]:
    task = normalize_task_type(task_type or payload.get("task_type"))
    model = str(song_model or payload.get("song_model") or "").strip()
    profile = normalize_quality_profile(quality_profile or payload.get("quality_profile"))
    defaults = quality_profile_model_settings(model, profile)
    duration = clamp_float(payload.get("duration"), 180.0, 10.0, 600.0)
    steps = clamp_int(payload.get("inference_steps"), int(defaults.get("inference_steps") or CHART_MASTER_STANDARD_STEPS), 1, 200)
    takes = clamp_int(payload.get("batch_size"), CHART_MASTER_SINGLE_TAKES if profile == QUALITY_PROFILE_CHART_MASTER and task == "text2music" else 1, 1, MAX_BATCH_SIZE)
    lowered = model.lower()
    model_factor = 1.0
    if "xl" in lowered:
        model_factor = 1.65
    elif "base" in lowered or "sft" in lowered:
        model_factor = 1.2
    if "turbo" in lowered:
        model_factor *= 0.45
    backend = str(payload.get("lm_backend") or DOCS_BEST_DEFAULT_LM_BACKEND)
    backend_factor = 0.9 if backend == "mlx" else 1.15 if backend == "pt" and sys.platform == "darwin" else 1.0
    estimated_seconds = int(max(20, round(duration * steps * takes * model_factor * backend_factor / 27.0)))
    if estimated_seconds > 1800:
        risk = "high"
    elif estimated_seconds > 600:
        risk = "medium"
    else:
        risk = "low"
    notes = []
    if profile == QUALITY_PROFILE_CHART_MASTER and takes > 1:
        notes.append("chart_master_multi_take")
    if risk == "high":
        notes.append("long_final_render")
    if task in ACE_STEP_SOURCE_LOCKED_DURATION_TASKS:
        notes.append("source_audio_task")
    return {
        "version": PRO_QUALITY_AUDIT_VERSION,
        "task_type": task,
        "song_model": model,
        "quality_profile": profile,
        "duration": duration,
        "steps": steps,
        "takes": takes,
        "backend": backend,
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": round(estimated_seconds / 60.0, 1),
        "risk": risk,
        "notes": notes,
    }

PARAM_ALIASES: dict[str, list[str]] = {
    "ace_lm_model": ["ace_lm_model", "lm_model", "lm_model_path", "lmModel"],
    "analysis_only": ["analysis_only", "analysisOnly"],
    "audio_code_string": ["audio_code_string", "audioCodeString", "audio_codes"],
    "audio_cover_strength": ["audio_cover_strength", "audioCoverStrength", "cover_strength", "coverStrength"],
    "audio_format": ["audio_format", "audioFormat", "format"],
    "caption": ["caption", "prompt", "tags"],
    "duration": ["duration", "audio_duration", "audioDuration", "target_duration", "targetDuration"],
    "extract_codes_only": ["extract_codes_only", "extractCodesOnly"],
    "full_analysis_only": ["full_analysis_only", "fullAnalysisOnly"],
    "is_format_caption": ["is_format_caption", "isFormatCaption"],
    "key_scale": ["key_scale", "keyscale", "keyScale", "key"],
    "sample_query": ["sample_query", "sampleQuery"],
    "song_model": ["song_model", "model", "model_name", "modelName", "dit_model", "ditModel"],
    "src_audio_path": ["src_audio_path", "src_audio", "source_audio_path", "source_audio"],
    "time_signature": ["time_signature", "timesignature", "timeSignature"],
    "track_classes": ["track_classes", "trackClasses", "instruments"],
    "track_name": ["track_name", "trackName"],
    "reference_audio_path": ["reference_audio_path", "reference_audio", "reference"],
    "use_tiled_decode": ["use_tiled_decode", "useTiledDecode"],
    "use_format": ["use_format", "useFormat"],
    "vocal_language": ["vocal_language", "vocalLanguage", "language"],
}

LYRIC_SECTION_RE = re.compile(
    r"(?im)^\s*\*{0,2}\[(?:intro|verse|pre[-\s]?chorus|chorus|final\s+chorus|hook|bridge|drop|break|interlude|outro|refrain|rap|spoken)"
    r"(?:\s+\d+)?(?:\s*[-:–—][^\]]+)?\]\s*$"
)
INLINE_LYRIC_SECTION_RE = re.compile(
    r"(?i)\[(?:intro|verse|pre[-\s]?chorus|chorus|final\s+chorus|hook|bridge|drop|break|interlude|outro|refrain|rap|spoken)"
    r"(?:\s+\d+)?(?:\s*[-:–—][^\]]+)?\]"
)


def normalize_lyric_section_marker(line: str | None) -> str:
    """Normalize markdown-emphasized section-only lines such as **[Chorus]**."""
    text = str(line or "")
    match = re.fullmatch(r"\s*[*_`~]*\s*(\[[^\]]+\])\s*[*_`~]*\s*", text)
    if not match:
        return text
    return match.group(1)


def lyric_section_key(section: str | None) -> str:
    text = re.sub(r"[*_`~]", "", str(section or ""))
    text = re.sub(r"[\[\]]", "", text).lower()
    text = re.sub(r"\s*-\s*.*$", "", text)
    text = re.sub(r"\s+\d+$", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{6,80}", value or ""):
        raise ValueError("invalid id")
    return value


def safe_filename(value: str, fallback: str = "audio") -> str:
    stem = Path(value or fallback).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-._")
    return stem[:80] or fallback


def normalize_task_type(value: str | None) -> str:
    task = (value or "text2music").strip().lower()
    aliases = {"custom": "text2music", "simple": "text2music", "remix": "cover"}
    task = aliases.get(task, task)
    if task not in TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task}")
    return task


def has_vocal_lyrics(lyrics: str | None) -> bool:
    text = (lyrics or "").strip()
    return bool(text and text.lower() != "[instrumental]")


def looks_like_lyrics(value: str | None) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if INLINE_LYRIC_SECTION_RE.search(text):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    comma_count = text.count(",")
    if comma_count >= len(lines) * 2 and len(lines) <= 8:
        return False
    lyricish_lines = [
        line for line in lines
        if len(line) <= 120 and not re.fullmatch(r"[\w\s-]+(?:,\s*[\w\s-]+){2,}", line)
    ]
    rhyme_markers = sum(1 for line in lines if re.search(r"\b(chorus|verse|bridge|hook|yeah|oh|woah|baby)\b", line, re.I))
    return len(lyricish_lines) >= max(4, int(len(lines) * 0.6)) or rhyme_markers >= 2


def split_caption_tags(value: str | None) -> list[str]:
    text = (value or "").strip()
    if not text or looks_like_lyrics(text):
        return []
    tags: list[str] = []
    for item in re.split(r"[,;\n]+", text):
        tag = re.sub(r"\s+", " ", item).strip(" .")
        if tag and tag.lower() not in {existing.lower() for existing in tags}:
            tags.append(tag)
    return tags


def _first_text(payload: dict[str, Any], keys: list[str]) -> tuple[str, str]:
    for key in keys:
        value = payload.get(key)
        if value not in [None, ""]:
            return str(value), key
    return "", ""


def _fallback_caption(payload: dict[str, Any], moved_text: str = "") -> str:
    candidates = [
        str(payload.get("tags") or ""),
        str(payload.get("custom_tags") or ""),
        str(payload.get("global_caption") or ""),
        str(payload.get("description") or ""),
        str(payload.get("title") or ""),
    ]
    tags: list[str] = []
    for candidate in candidates:
        if not candidate or candidate.strip() == moved_text.strip() or looks_like_lyrics(candidate):
            continue
        for tag in split_caption_tags(candidate) or [candidate.strip()]:
            if tag and tag.lower() not in {existing.lower() for existing in tags} and tag.lower() != "untitled":
                tags.append(tag)
    if tags:
        return ", ".join(tags[:14])
    return "polished vocal song, clear lead vocal, full arrangement, radio-ready mix"


def normalize_generation_text_fields(payload: dict[str, Any], task_type: str | None = None) -> dict[str, Any]:
    normalized = dict(payload or {})
    warnings = list(normalized.get("payload_warnings") or [])
    caption, caption_source = _first_text(normalized, ["caption", "prompt", "tags"])
    description = str(normalized.get("description") or "").strip()
    if not caption and description and not looks_like_lyrics(description):
        caption = description
        caption_source = "description"
    lyrics, lyrics_source = _first_text(normalized, ["lyrics", "lyric", "song_lyrics"])
    if caption and normalized.get("caption_source"):
        caption_source = str(normalized.get("caption_source") or caption_source)
    if lyrics and normalized.get("lyrics_source"):
        lyrics_source = str(normalized.get("lyrics_source") or lyrics_source)
    instrumental = parse_bool(normalized.get("instrumental"), False)
    description_output = description

    if not has_vocal_lyrics(lyrics) and looks_like_lyrics(caption):
        lyrics = caption
        lyrics_source = f"{caption_source or 'caption'}_repaired"
        caption = _fallback_caption(normalized, moved_text=lyrics)
        caption_source = "repaired_from_tags"
        warnings.append("Moved likely lyrics from caption/tags into lyrics.")
        if description_output.strip() == lyrics.strip():
            description_output = ""

    if not has_vocal_lyrics(lyrics) and looks_like_lyrics(description):
        lyrics = description
        lyrics_source = "description_repaired"
        if not caption or caption.strip() == description:
            caption = _fallback_caption(normalized, moved_text=lyrics)
            caption_source = "repaired_from_description"
        warnings.append("Moved likely lyrics from description into lyrics.")
        description_output = ""
    elif description_output and looks_like_lyrics(description_output) and description_output.strip() == lyrics.strip():
        description_output = ""

    if instrumental and not lyrics.strip():
        lyrics = "[Instrumental]"
        lyrics_source = "instrumental_default"

    tag_list = split_caption_tags(caption)
    normalized.update(
        {
            "caption": caption.strip(),
            "description": description_output,
            "lyrics": lyrics,
            "instrumental": instrumental,
            "caption_source": caption_source or "empty",
            "lyrics_source": lyrics_source or "empty",
            "tag_list": tag_list,
            "payload_warnings": warnings,
            "ui_mode": str(normalized.get("ui_mode") or normalized.get("mode") or task_type or "").strip(),
        }
    )
    return normalized


_META_LEAK_LINE_RE = re.compile(
    r"(?i)^\s*(?:[-*>_`#\s]+)?(?:"
    r"\[(?:producer credit|locked title|duration|bpm|key|artist|title|metadata|tags|song model|quality profile)[^\]]*\]|"
    r"thought:|reasoning:|analysis:|self[-\s]?correction:|draft:|note:|"
    r"i will now\b|i'?ll now\b|let'?s write\b|let me\b|here(?:'s| is)\b|"
    r"the complete production spec\b|track metadata:|artist:|description:|tags:|"
    r"duration:|bpm:|key(?:[_\s-]?scale)?:|time(?:[_\s-]?signature)?:|title:|metadata:|"
    r"\[?ace[-\s]?step(?:\s+metadata)?\]?:?|ace[-\s]?step metadata:|"
    r"tag(?:[_\s-]?list)?:|start:|end:|vocal(?:[_\s-]?role)?:|"
    r"song(?:[_\s-]?model)?:|quality(?:[_\s-]?profile)?:|"
    r"model(?:[_\s-]?advice)?:|seed:|inference(?:[_\s-]?steps)?:|guidance(?:[_\s-]?scale)?:|"
    r"shift:|audio(?:[_\s-]?format)?:|json\b)"
)
_SECTION_TIMING_LINE_RE = re.compile(r"(?i)^\s*-?\s*\[[^\]]+\]\s*\([^)]*\bsec(?:ond)?s?\b[^)]*\)\s*$")
_BRACKET_CONTRACT_META_LINE_RE = re.compile(
    r"(?i)^\s*\[(?:producer credit|locked title|duration|bpm|key|artist|title|metadata|tags|song model|quality profile)[^\]]*\]\s*$"
)
_LYRICS_LABEL_RE = re.compile(r"(?i)^\s*(?:[-*>_`#\s]+)?lyrics\s*:\s*$")
_REQUIRED_PHRASES_SECTION_RE = re.compile(r"(?i)^\s*\[required phrases?\]\s*$")
_ACE_STEP_TIMING_BLOCK_RE = re.compile(
    r"(?ims)^\s*\[ACE-Step\]\s*\n"
    r"(?:(?:\s*(?:Tag|Start|End|Vocal\s+Role)\s*:[^\n]*\n?)|\s*)+"
)


def strip_ace_step_lyrics_leakage(lyrics: str | None) -> str:
    """Remove assistant/planning prose that can leak into the ACE-Step lyrics field."""
    text = str(lyrics or "")
    if not text.strip():
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)```(?:json|markdown|text)?\s*", "", text)
    text = text.replace("```", "")
    text = _ACE_STEP_TIMING_BLOCK_RE.sub("", text)
    kept: list[str] = []
    saw_section = False
    for raw_line in text.splitlines():
        line = normalize_lyric_section_marker(raw_line.rstrip())
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _BRACKET_CONTRACT_META_LINE_RE.search(stripped):
            continue
        if _SECTION_TIMING_LINE_RE.search(stripped):
            continue
        if _LYRICS_LABEL_RE.search(stripped):
            kept = []
            saw_section = False
            continue
        if _REQUIRED_PHRASES_SECTION_RE.search(stripped):
            continue
        if INLINE_LYRIC_SECTION_RE.search(stripped):
            saw_section = True
            kept.append(line)
            continue
        if not saw_section and _META_LEAK_LINE_RE.search(stripped):
            continue
        if saw_section and _META_LEAK_LINE_RE.search(stripped):
            break
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _lyric_section_blocks(text: str) -> list[str]:
    lines = str(text or "").splitlines()
    blocks: list[str] = []
    current: list[str] = []
    section_re = re.compile(r"^\s*\[[^\]]+\]\s*$")
    for line in lines:
        if section_re.match(line) and current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current = [line]
        else:
            current.append(line)
    tail = "\n".join(current).strip()
    if tail:
        blocks.append(tail)
    return blocks


def _join_lyric_blocks(blocks: list[str]) -> str:
    return "\n".join(block.strip() for block in blocks if str(block or "").strip()).strip()


def fit_ace_step_lyrics_to_limit(lyrics: str | None, limit: int = ACE_STEP_LYRICS_CHAR_LIMIT) -> str:
    """Fit lyrics using complete sections/lines only; never slice through a word or invent an outro."""
    text = str(lyrics or "").strip()
    if len(text) <= limit:
        return text
    hard_limit = max(256, int(limit or ACE_STEP_LYRICS_CHAR_LIMIT))
    target_limit = min(hard_limit - ACE_STEP_LYRICS_SAFE_HEADROOM, ACE_STEP_LYRICS_SOFT_TARGET_MAX)
    target_limit = max(512, target_limit)
    blocks = _lyric_section_blocks(text)
    kept: list[str] = []
    outro_blocks = [block for block in blocks if re.match(r"^\s*\[.*outro.*\]", block, re.I)]
    for block in blocks:
        candidate = _join_lyric_blocks([*kept, block])
        if len(candidate) <= target_limit:
            kept.append(block)
            continue
        break
    if outro_blocks and not any(re.match(r"^\s*\[.*outro.*\]", block, re.I) for block in kept):
        outro = outro_blocks[-1]
        while kept and len(_join_lyric_blocks([*kept, outro])) > target_limit:
            kept.pop()
        if len(_join_lyric_blocks([*kept, outro])) <= target_limit:
            kept.append(outro)
    fitted = _join_lyric_blocks(kept)
    if fitted:
        return fitted

    kept_lines: list[str] = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        candidate = "\n".join([*kept_lines, line]).strip()
        if len(candidate) > target_limit:
            break
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def apply_ace_step_text_budget(payload: dict[str, Any], *, task_type: str | None = None) -> dict[str, Any]:
    """Normalize text fields to the official ACE-Step request limits and record what changed."""
    normalized = dict(payload or {})
    warnings = list(normalized.get("payload_warnings") or [])
    budget = dict(normalized.get("ace_step_text_budget") or {})
    caption = str(normalized.get("caption") or "")
    lyrics = str(normalized.get("lyrics") or "")
    source_caption_count = len(caption)
    source_lyrics_count = len(lyrics)
    action = "none"

    if len(caption) > ACE_STEP_CAPTION_CHAR_LIMIT:
        caption = caption[:ACE_STEP_CAPTION_CHAR_LIMIT].rstrip()
        warnings.append(f"Caption fit to ACE-Step {ACE_STEP_CAPTION_CHAR_LIMIT}-character limit.")
        action = "caption_trimmed"

    cleaned_lyrics = strip_ace_step_lyrics_leakage(lyrics)
    if cleaned_lyrics != lyrics.strip():
        warnings.append("Removed assistant planning prose from lyrics before ACE-Step generation.")
        action = "lyrics_cleaned" if action == "none" else f"{action}+lyrics_cleaned"
    lyrics = cleaned_lyrics

    policy = str(normalized.get("lyrics_overflow_policy") or "auto_fit").strip().lower() or "auto_fit"
    if len(lyrics) > ACE_STEP_LYRICS_CHAR_LIMIT:
        lyrics = fit_ace_step_lyrics_to_limit(lyrics, ACE_STEP_LYRICS_CHAR_LIMIT)
        warnings.append(f"Lyrics fit to ACE-Step {ACE_STEP_LYRICS_CHAR_LIMIT}-character runtime limit.")
        action = "lyrics_auto_fit" if action == "none" else f"{action}+lyrics_auto_fit"

    budget.update(
        {
            "caption_char_limit": ACE_STEP_CAPTION_CHAR_LIMIT,
            "lyrics_char_limit": ACE_STEP_LYRICS_CHAR_LIMIT,
            "dit_lyrics_token_limit": ACE_STEP_DIT_LYRICS_TOKEN_LIMIT,
            "source_caption_char_count": source_caption_count,
            "runtime_caption_char_count": len(caption),
            "source_lyrics_char_count": source_lyrics_count,
            "runtime_lyrics_char_count": len(lyrics),
            "lyrics_overflow_policy": policy,
            "lyrics_overflow_action": action,
            "task_type": task_type or normalized.get("task_type") or "",
        }
    )
    normalized["caption"] = caption
    normalized["lyrics"] = lyrics
    normalized["ace_step_text_budget"] = budget
    normalized["payload_warnings"] = warnings
    return normalized


def needs_vocal_lyrics(
    *,
    task_type: str,
    instrumental: bool,
    lyrics: str | None,
    sample_mode: bool = False,
    sample_query: str | None = "",
) -> bool:
    if task_type != "text2music" or instrumental:
        return False
    if has_vocal_lyrics(lyrics):
        return False
    return not (sample_mode or bool((sample_query or "").strip()))


def model_label(name: str) -> str:
    profile = MODEL_PROFILES.get(name)
    return str(profile["label"]) if profile else name


def _profile_with_runtime_fields(profile: dict[str, Any], name: str, installed: bool) -> dict[str, Any]:
    item = {key: value[:] if isinstance(value, list) else value for key, value in profile.items()}
    item["name"] = name
    item["installed"] = installed
    item["official_status"] = str(item.get("official_status") or OFFICIAL_ACE_STEP_MANIFEST["dit_models"].get(name, {}).get("status") or "local")
    item["downloadable"] = name not in OFFICIAL_UNRELEASED_MODELS
    if item["official_status"] == "unreleased":
        item["installed_label"] = "Unreleased"
    else:
        item["installed_label"] = "Installed" if installed else "Download required"
    return item


def model_profile(name: str, installed: bool = False) -> dict[str, Any]:
    profile = MODEL_PROFILES.get(name)
    if profile is None:
        tasks = supported_tasks_for_model(name)
        is_base = "base" in (name or "").lower()
        profile = {
            "label": name,
            "dropdown_label": f"{name} - locally discovered",
            "summary": "Locally discovered ACE-Step model. Compatibility is inferred from its name.",
            "best_for": ["Local model", "Custom experiments"] + (["All tasks"] if is_base else ["Standard tasks"]),
            "quality": "Unknown",
            "speed": "Unknown",
            "vram": "Unknown",
            "steps": "50 quality default" if is_base else "8-50 inferred",
            "cfg": "Yes" if is_base or "sft" in (name or "").lower() else "No for turbo, unknown otherwise",
            "tasks": tasks,
            "recommended_for": ["custom"],
            "warnings": ["This model is not in AceJAM's official profile table; verify its model card."],
            "notes": "Fallback profile generated from the model folder name.",
            "source_urls": ACE_STEP_MODEL_SOURCES,
        }
    item = _profile_with_runtime_fields(profile, name, installed)
    item["tasks"] = supported_tasks_for_model(name)
    return item


def lm_model_profile(name: str, installed: bool = False) -> dict[str, Any]:
    profile = LM_MODEL_PROFILES.get(name)
    if profile is None:
        profile = {
            "label": name,
            "dropdown_label": f"{name} - locally discovered",
            "summary": "Locally discovered ACE-Step LM. Capability is unknown until its model card is checked.",
            "best_for": ["Local planner"],
            "quality": "Unknown",
            "speed": "Unknown",
            "vram": "Unknown",
            "steps": "N/A",
            "cfg": "LM CFG supported if model-compatible",
            "tasks": ["metadata", "caption rewrite"],
            "warnings": ["This LM is not in AceJAM's official profile table."],
            "notes": "Fallback LM profile generated from the model folder name.",
            "source_urls": ACE_STEP_MODEL_SOURCES,
        }
    return _profile_with_runtime_fields(profile, name, installed or name in {"auto", "none"})


def model_profiles_for_models(models: list[str], installed_models: set[str] | list[str]) -> dict[str, dict[str, Any]]:
    installed = set(installed_models)
    return {name: model_profile(name, name in installed) for name in models}


def lm_model_profiles_for_models(models: list[str], installed_models: set[str] | list[str]) -> dict[str, dict[str, Any]]:
    installed = set(installed_models)
    return {name: lm_model_profile(name, name in installed) for name in models}


def official_model_registry() -> dict[str, dict[str, Any]]:
    """Return the official ACE-Step model catalog grouped by role in metadata."""
    registry = copy.deepcopy(OFFICIAL_ACE_STEP_MODEL_REGISTRY)
    for name, item in registry.items():
        item.setdefault("id", name)
        item.setdefault("downloadable", False)
        item.setdefault("render_usable", False)
        item.setdefault("source_urls", ACE_STEP_MODEL_SOURCES)
    return registry


def official_model_repo_id(model_name: str) -> str:
    item = OFFICIAL_ACE_STEP_MODEL_REGISTRY.get(str(model_name or ""))
    return str((item or {}).get("repo_id") or "")


def official_downloadable_model_ids(*, include_helpers: bool = True) -> list[str]:
    ids = []
    for name, item in OFFICIAL_ACE_STEP_MODEL_REGISTRY.items():
        if not item.get("downloadable"):
            continue
        role = str(item.get("role") or "")
        if not include_helpers and role not in {"core_bundle", "render_dit", "ace_lm"}:
            continue
        ids.append(name)
    return ids


def official_helper_model_ids() -> list[str]:
    """Return official non-render helpers that can improve analysis/metadata workflows."""
    return [
        name
        for name in OFFICIAL_HELPER_MODEL_IDS
        if OFFICIAL_ACE_STEP_MODEL_REGISTRY.get(name, {}).get("downloadable")
    ]


def official_boot_model_ids(*, include_helpers: bool = False, include_best_quality: bool = True) -> list[str]:
    """Return the default boot download bundle for quality-first local runs."""
    ids: list[str] = []

    def add(model_name: str) -> None:
        if model_name and model_name not in ids:
            ids.append(model_name)

    if include_best_quality:
        for model_name in OFFICIAL_BOOT_QUALITY_MODEL_IDS:
            add(model_name)
    if include_helpers:
        for model_name in official_helper_model_ids():
            add(model_name)
    downloadable = set(official_downloadable_model_ids())
    return [model_name for model_name in ids if model_name in downloadable]


def official_render_model_ids() -> list[str]:
    return [name for name in KNOWN_ACE_STEP_MODELS if OFFICIAL_ACE_STEP_MODEL_REGISTRY.get(name, {}).get("render_usable")]


def official_manifest() -> dict[str, Any]:
    """Return a copy of the official ACE-Step parity manifest."""
    manifest = copy.deepcopy(OFFICIAL_ACE_STEP_MANIFEST)
    manifest["quality_profiles"] = copy.deepcopy(quality_profiles_payload())
    manifest["default_quality_profile"] = DEFAULT_QUALITY_PROFILE
    manifest["model_registry"] = official_model_registry()
    manifest["boot_quality_models"] = official_boot_model_ids()
    manifest["helper_functions"] = copy.deepcopy(OFFICIAL_HELPER_FUNCTIONS)
    manifest["result_fields"] = copy.deepcopy(OFFICIAL_RESULT_FIELDS)
    manifest["settings_coverage"] = AceStepSettingsRegistry.coverage()
    manifest["payload_aliases"] = copy.deepcopy(PARAM_ALIASES)
    manifest["acejam_extension_params"] = {
        "status": "supported",
        "fields": [
            "prompt",
            "lm_model_path",
            "reference_audio_path",
            "src_audio_path",
            "metas",
            "metadata",
            "user_metadata",
            "param_obj",
            "lm_repetition_penalty",
            "analysis_only",
            "full_analysis_only",
            "extract_codes_only",
            "use_tiled_decode",
            "is_format_caption",
            "track_name",
            "track_classes",
            "device",
            "dtype",
            "use_flash_attention",
            "compile_model",
            "offload_to_cpu",
            "offload_dit_to_cpu",
            "lm_device",
            "lm_dtype",
            "lm_offload_to_cpu",
        ],
        "notes": "AceJAM accepts official aliases and guarded runtime knobs, then normalizes them before dispatch.",
    }
    return manifest


def recommended_song_model(installed_models: set[str] | list[str] | None = None) -> str:
    installed = set(installed_models or [])
    if not installed:
        return "acestep-v15-xl-sft"
    for candidate in ["acestep-v15-xl-sft", "acestep-v15-sft", "acestep-v15-xl-base", "acestep-v15-base", "acestep-v15-xl-turbo", "acestep-v15-turbo"]:
        if candidate in installed:
            return candidate
    return "acestep-v15-xl-sft"


def recommended_lm_model(installed_models: set[str] | list[str] | None = None, quality_profile: str | None = None) -> str:
    installed = set(installed_models or [])
    profile = normalize_quality_profile(quality_profile) if quality_profile else ""
    if profile == QUALITY_PROFILE_DOCS_DAILY:
        for candidate in ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"]:
            if candidate in installed:
                return candidate
        return "none"
    for candidate in sorted(installed):
        lowered = candidate.lower()
        if "acestep-5hz-lm-4b" in lowered and "abliter" in lowered:
            return candidate
    for candidate in ["acestep-5Hz-lm-4B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B"]:
        if candidate in installed:
            return candidate
    return "none"


def supported_tasks_for_model(model_name: str) -> list[str]:
    lowered = (model_name or "").lower()
    if lowered.endswith("-base") or "-base" in lowered:
        tasks = list(TASK_TYPES_BASE)
    elif "turbo" in lowered or lowered.endswith("-sft") or "-sft" in lowered:
        tasks = list(TASK_TYPES_TURBO)
    else:
        tasks = list(TASK_TYPES_TURBO)
    if "cover" in tasks and "cover-nofsq" not in tasks:
        cover_index = tasks.index("cover")
        tasks.insert(cover_index + 1, "cover-nofsq")
    return tasks


def ensure_task_supported(model_name: str, task_type: str) -> None:
    supported = supported_tasks_for_model(model_name)
    if task_type not in supported:
        raise ValueError(
            f"{model_label(model_name)} does not support {task_type}. "
            "Use a base or xl-base model for extract, lego, and complete."
        )


def clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def parse_timesteps(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return None
        raw_items = [item.strip() for item in text.split(",")]
    parsed = [clamp_float(item, 0.0, 0.0, 1.0) for item in raw_items if str(item).strip()]
    return parsed or None


def normalize_key_scale(value: Any, *, default: str | None = None) -> str:
    """Normalize user-facing key aliases to ACE-Step's constrained keyscale values."""
    if value is None:
        return default or ""
    text = str(value).strip()
    if not text or text.lower() in {"auto", "none", "n/a", "na"}:
        return default or ""
    match = re.fullmatch(r"([A-Ga-g])\s*([#b♯♭]?)\s*(major|minor|maj|min|m)?", text)
    if match:
        note = match.group(1).upper()
        accidental = match.group(2) or ""
        mode_text = (match.group(3) or "major").lower()
        mode = "minor" if mode_text in {"minor", "min", "m"} else "major"
        candidate = f"{note}{accidental} {mode}"
        if candidate in VALID_KEY_SCALE_SET:
            return candidate
    parts = text.split()
    if len(parts) == 2:
        note_part = parts[0][:1].upper() + parts[0][1:]
        mode_part = parts[1].lower()
        if mode_part in {"major", "minor"}:
            candidate = f"{note_part} {mode_part}"
            if candidate in VALID_KEY_SCALE_SET:
                return candidate
    raise ValueError(f"Unsupported ACE-Step key scale: {text}. Choose Auto or one of the official keyscales.")


def get_param(payload: dict[str, Any], canonical: str, default: Any = None) -> Any:
    names = PARAM_ALIASES.get(canonical, [canonical])
    for name in names:
        if name in payload:
            return payload.get(name)
    return default


def _is_default_value(value: Any, default: Any) -> bool:
    if isinstance(default, bool):
        return parse_bool(value, default) == default
    if isinstance(default, int) and not isinstance(default, bool):
        return clamp_int(value, default, -1000000, 1000000) == default
    if isinstance(default, float):
        return abs(clamp_float(value, default, -1000000.0, 1000000.0) - default) < 1e-9
    return str(value or "").strip() == str(default or "").strip()


def official_fields_used(payload: dict[str, Any]) -> list[str]:
    used: list[str] = []
    for field in sorted(OFFICIAL_ONLY_FIELDS):
        if field not in payload:
            continue
        value = payload.get(field)
        if field in {"thinking", "use_format", "use_cot_caption", "use_cot_language", "use_cot_lyrics", "use_cot_metas"} and not parse_bool(value, False):
            continue
        if not _is_default_value(value, OFFICIAL_FIELD_DEFAULTS.get(field, "")):
            used.append(field)
    fmt = str(get_param(payload, "audio_format", payload.get("audio_format") or "wav")).strip().lower().lstrip(".")
    if fmt and fmt not in SUPPORTED_AUDIO_FORMATS:
        used.append("audio_format")
    return used


def normalize_audio_format(value: str | None, allow_official: bool = False) -> str:
    fmt = (value or "wav").strip().lower().lstrip(".")
    valid = OFFICIAL_AUDIO_FORMATS if allow_official else SUPPORTED_AUDIO_FORMATS
    if fmt not in valid:
        return "flac" if allow_official else "wav"
    return fmt


def build_task_instruction(task_type: str, track_names: Any = None) -> str:
    task = normalize_task_type(task_type)
    tracks = normalize_track_names(track_names)
    primary = tracks[0] if tracks else ""
    if task == "extract":
        template = TASK_INSTRUCTIONS["extract"] if primary else TASK_INSTRUCTIONS["extract_default"]
        return template.format(TRACK_NAME=primary)
    if task == "lego":
        template = TASK_INSTRUCTIONS["lego"] if primary else TASK_INSTRUCTIONS["lego_default"]
        return template.format(TRACK_NAME=primary)
    if task == "complete":
        template = TASK_INSTRUCTIONS["complete"] if tracks else TASK_INSTRUCTIONS["complete_default"]
        return template.format(TRACK_CLASSES=", ".join(tracks))
    return TASK_INSTRUCTIONS[task]


def normalize_track_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    allowed = set(TRACK_NAMES)
    tracks = []
    for item in items:
        normalized = item.lower().replace(" ", "_").replace("-", "_")
        if normalized in allowed and normalized not in tracks:
            tracks.append(normalized)
    return tracks


def ordered_models(discovered: list[str]) -> list[str]:
    known_order = {name: index for index, name in enumerate(KNOWN_ACE_STEP_MODELS)}
    merged = {name for name in discovered if name.startswith("acestep-v15-")}
    merged.update(KNOWN_ACE_STEP_MODELS)
    return sorted(merged, key=lambda name: (known_order.get(name, len(known_order)), name))


def studio_ui_schema() -> dict[str, Any]:
    settings_registry = ace_step_settings_registry()
    return {
        "sources": ACE_STEP_MODEL_SOURCES,
        "payload_contract_version": "2026-04-26",
        "official_manifest_version": OFFICIAL_ACE_STEP_MANIFEST["manifest_version"],
        "official_model_registry": official_model_registry(),
        "official_render_models": official_render_model_ids(),
        "official_downloadable_models": official_downloadable_model_ids(),
        "settings_policy_version": settings_registry["version"],
        "default_quality_profile": settings_registry["default_quality_profile"],
        "quality_profiles": settings_registry["profiles"],
        "ace_step_coverage": settings_registry["coverage"],
        "payload_validation_endpoint": "/api/payload/validate",
        "official_parity_endpoint": "/api/ace-step/parity",
        "ace_step_settings_registry": settings_registry,
        "default_bpm": None,
        "default_key_scale": KEYSCALE_AUTO_VALUE,
        "metadata_auto": {
            "bpm": None,
            "key_scale": KEYSCALE_AUTO_VALUE,
            "time_signature": "",
            "duration": METADATA_AUTO_VALUE,
            "vocal_language": "unknown",
        },
        "valid_keyscales": VALID_KEY_SCALES,
        "task_required_fields": copy.deepcopy(ACE_STEP_TASK_REQUIRED_FIELDS),
        "task_policy": settings_registry["task_policy"],
        "audio_formats": sorted(OFFICIAL_AUDIO_FORMATS),
        "fast_audio_formats": sorted(SUPPORTED_AUDIO_FORMATS),
        "ace_step_text_budget": {
            "caption_char_limit": ACE_STEP_CAPTION_CHAR_LIMIT,
            "lyrics_char_limit": ACE_STEP_LYRICS_CHAR_LIMIT,
            "dit_lyrics_token_limit": ACE_STEP_DIT_LYRICS_TOKEN_LIMIT,
        },
        "quality_policy": docs_best_quality_policy(),
        "task_lm_usage": {
            "uses_lm": sorted(DOCS_BEST_LM_TASKS),
            "skips_lm": sorted(DOCS_BEST_SOURCE_TASK_LM_SKIPS),
            "planner": "ollama",
            "note": "ACE-Step LM is used only for official generation controls; Ollama remains prompt and lyric writer.",
        },
        "custom_sections": {
            "song": ["title", "duration", "instrumental", "vocal_language", "caption", "lyrics", "reference_audio"],
            "music_metadata": ["bpm", "key_scale", "time_signature", "global_caption"],
            "generation": [
                "quality_profile",
                "batch_size",
                "seed",
                "use_random_seed",
                "instruction",
                "inference_steps",
                "guidance_scale",
                "shift",
                "infer_method",
                "sampler_mode",
                "timesteps",
                "use_adg",
                "cfg_interval_start",
                "cfg_interval_end",
                "dcw_enabled",
                "dcw_mode",
                "dcw_scaler",
                "dcw_high_scaler",
                "dcw_wavelet",
                "retake_seed",
                "retake_variance",
            ],
            "ace_step_lm": [],
            "api_service": ["analysis_only", "full_analysis_only", "extract_codes_only", "use_tiled_decode", "is_format_caption", "track_name", "track_classes"],
            "ollama_planner": ["ollama_model", "planner_ollama_model", "planner_lm_provider"],
            "output": ["audio_format", "mp3_bitrate", "mp3_sample_rate", "auto_score", "auto_lrc", "return_audio_codes", "save_to_library"],
            "post_processing": ["enable_normalization", "normalization_db", "fade_in_duration", "fade_out_duration", "latent_shift", "latent_rescale"],
            "repaint_cover": [
                "audio_cover_strength",
                "cover_noise_strength",
                "chunk_mask_mode",
                "repaint_latent_crossfade_frames",
                "repaint_wav_crossfade_sec",
                "repaint_mode",
                "repaint_strength",
                "flow_edit_morph",
                "flow_edit_source_caption",
                "flow_edit_source_lyrics",
                "flow_edit_n_min",
                "flow_edit_n_max",
                "flow_edit_n_avg",
            ],
            "runtime": [
                "device",
                "dtype",
                "use_flash_attention",
                "compile_model",
                "offload_to_cpu",
                "offload_dit_to_cpu",
            ],
        },
        "ranges": {
            "bpm": [BPM_MIN, BPM_MAX],
            "duration": [10, 600],
            "inference_steps": [1, 200],
            "guidance_scale": [1.0, 15.0],
            "shift": [1.0, 5.0],
            "lm_temperature": [0.0, 2.0],
            "lm_cfg_scale": [0.0, 10.0],
            "lm_top_k": [0, 200],
            "lm_top_p": [0.0, 1.0],
            "audio_cover_strength": [0.0, 1.0],
            "cover_noise_strength": [0.0, 1.0],
            "repaint_strength": [0.0, 1.0],
            "dcw_scaler": [0.0, 0.2],
            "dcw_high_scaler": [0.0, 0.2],
            "retake_variance": [0.0, 1.0],
            "flow_edit_n_min": [0.0, 1.0],
            "flow_edit_n_max": [0.0, 1.0],
            "flow_edit_n_avg": [1, 16],
            "normalization_db": [-24.0, 0.0],
            "fade_in_duration": [0.0, 20.0],
            "fade_out_duration": [0.0, 20.0],
            "latent_shift": [-2.0, 2.0],
            "latent_rescale": [0.1, 3.0],
            "lm_repetition_penalty": [0.1, 4.0],
        },
        "defaults": dict(OFFICIAL_FIELD_DEFAULTS),
        "options": {
            "key_scale": [KEYSCALE_AUTO_VALUE, *VALID_KEY_SCALES],
            "time_signature": [2, 3, 4, 6],
            "track_name": TRACK_NAMES,
            "track_classes": TRACK_NAMES,
        },
        "official_defaults": dict(ACE_STEP_OFFICIAL_DEFAULTS),
        "docs_recommended": settings_registry["profiles"]["docs_recommended"],
        "chart_master": settings_registry["profiles"]["chart_master"],
        "balanced_pro": settings_registry["profiles"]["balanced_pro"],
        "official_only_fields": sorted(OFFICIAL_ONLY_FIELDS),
        "fast_handler_fields": sorted(FAST_HANDLER_FIELDS),
        "official_runtime_controls": OFFICIAL_RUNTIME_CONTROLS,
    }
