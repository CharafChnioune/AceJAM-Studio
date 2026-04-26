from __future__ import annotations

import copy
import platform
import re
import sys
from pathlib import Path
from typing import Any

from acestep.constants import TASK_INSTRUCTIONS, TASK_TYPES, TASK_TYPES_BASE, TASK_TYPES_TURBO, TRACK_NAMES

KNOWN_ACE_STEP_MODELS = [
    "acestep-v15-turbo",
    "acestep-v15-turbo-shift3",
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
    "https://huggingface.co/ACE-Step/Ace-Step1.5",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md",
    "https://arxiv.org/abs/2506.00045",
    "https://arxiv.org/abs/2602.00744",
]

STANDARD_TASKS = ["text2music", "cover", "repaint"]
ALL_TASKS = ["text2music", "cover", "repaint", "extract", "lego", "complete"]
OFFICIAL_UNRELEASED_MODELS = {"acestep-v15-turbo-rl"}

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
    "timesteps",
    "repainting_start",
    "repainting_end",
    "chunk_mask_mode",
    "repaint_latent_crossfade_frames",
    "repaint_wav_crossfade_sec",
    "repaint_mode",
    "repaint_strength",
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

DOCS_BEST_QUALITY_POLICY_VERSION = "docs-best-2026-04-25"
DOCS_BEST_AUDIO_FORMAT = "wav32"
DOCS_BEST_TURBO_STEPS = 8
DOCS_BEST_TURBO_HIGH_CAP_STEPS = 20
DOCS_BEST_STANDARD_STEPS = 64
DOCS_BEST_TURBO_GUIDANCE = 7.0
DOCS_BEST_STANDARD_GUIDANCE = 8.0
DOCS_BEST_TURBO_SHIFT = 3.0
DOCS_BEST_STANDARD_SHIFT = 1.0
DOCS_BEST_DEFAULT_LM_MODEL = "acestep-5Hz-lm-4B"
DOCS_BEST_DEFAULT_LM_BACKEND = "mlx" if sys.platform == "darwin" and platform.machine() == "arm64" else "pt"
DOCS_BEST_SOURCE_TASK_LM_SKIPS = {"cover", "repaint", "extract"}
DOCS_BEST_LM_TASKS = {"text2music", "lego", "complete"}
DOCS_BEST_LM_DEFAULTS: dict[str, Any] = {
    "ace_lm_model": DOCS_BEST_DEFAULT_LM_MODEL,
    "lm_backend": DOCS_BEST_DEFAULT_LM_BACKEND,
    "thinking": True,
    "use_format": True,
    "use_cot_metas": True,
    "use_cot_caption": True,
    "use_cot_language": True,
    "use_cot_lyrics": True,
    "use_constrained_decoding": True,
    "lm_cfg_scale": 10.0,
    "lm_temperature": 1.0,
    "lm_top_p": 1.0,
    "lm_top_k": 40,
}


def is_turbo_song_model(song_model: Any) -> bool:
    normalized = str(song_model or "").strip().lower()
    return "turbo" in normalized


def docs_best_model_settings(song_model: Any, *, high_turbo_cap: bool = False) -> dict[str, Any]:
    """Return ACE-Step-calibrated quality defaults for the selected DiT model."""
    turbo = is_turbo_song_model(song_model)
    if turbo:
        steps = DOCS_BEST_TURBO_HIGH_CAP_STEPS if high_turbo_cap else DOCS_BEST_TURBO_STEPS
        guidance = DOCS_BEST_TURBO_GUIDANCE
        shift = DOCS_BEST_TURBO_SHIFT
    else:
        steps = DOCS_BEST_STANDARD_STEPS
        guidance = DOCS_BEST_STANDARD_GUIDANCE
        shift = DOCS_BEST_STANDARD_SHIFT
    return {
        "quality_preset": DOCS_BEST_QUALITY_POLICY_VERSION,
        "inference_steps": steps,
        "guidance_scale": guidance,
        "shift": shift,
        "infer_method": "ode",
        "sampler_mode": "heun",
        "audio_format": DOCS_BEST_AUDIO_FORMAT,
    }


def docs_best_quality_policy() -> dict[str, Any]:
    return {
        "version": DOCS_BEST_QUALITY_POLICY_VERSION,
        "standard": "ACE-Step docs calibrated quality settings, not blind maximum sliders.",
        "audio_format": DOCS_BEST_AUDIO_FORMAT,
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
        },
        "lm_defaults": dict(DOCS_BEST_LM_DEFAULTS),
        "lm_task_policy": {
            "uses_lm_when_controls_active": sorted(DOCS_BEST_LM_TASKS),
            "skips_lm_for_source_tasks": sorted(DOCS_BEST_SOURCE_TASK_LM_SKIPS),
            "planner_writer": "ollama",
            "note": "Ollama writes prompts and lyrics; the ACE-Step 5Hz LM is reserved for official generation controls.",
        },
    }


OFFICIAL_ACE_STEP_MANIFEST: dict[str, Any] = {
    "manifest_version": "2026-04-25",
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
    "generation_params": {name: {"status": "supported"} for name in OFFICIAL_GENERATION_PARAMS},
    "generation_config": {name: {"status": "supported"} for name in OFFICIAL_GENERATION_CONFIG_FIELDS},
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
    "chunk_mask_mode",
    "compile_model",
    "constrained_decoding_debug",
    "cover_noise_strength",
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
    "sample_mode",
    "sample_query",
    "sampler_mode",
    "thinking",
    "use_flash_attention",
    "use_constrained_decoding",
    "use_cot_caption",
    "use_cot_language",
    "use_cot_lyrics",
    "use_cot_metas",
    "use_format",
    "use_random_seed",
    "velocity_ema_factor",
    "velocity_norm_threshold",
}

OFFICIAL_FIELD_DEFAULTS: dict[str, Any] = {
    "allow_lm_batch": False,
    "chunk_mask_mode": "auto",
    "compile_model": False,
    "constrained_decoding_debug": False,
    "cover_noise_strength": 0.0,
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
    "sample_mode": False,
    "sample_query": "",
    "sampler_mode": "heun",
    "thinking": DOCS_BEST_LM_DEFAULTS["thinking"],
    "offload_dit_to_cpu": False,
    "offload_to_cpu": False,
    "use_flash_attention": "auto",
    "use_constrained_decoding": DOCS_BEST_LM_DEFAULTS["use_constrained_decoding"],
    "use_cot_caption": DOCS_BEST_LM_DEFAULTS["use_cot_caption"],
    "use_cot_language": DOCS_BEST_LM_DEFAULTS["use_cot_language"],
    "use_cot_lyrics": DOCS_BEST_LM_DEFAULTS["use_cot_lyrics"],
    "use_cot_metas": DOCS_BEST_LM_DEFAULTS["use_cot_metas"],
    "use_format": DOCS_BEST_LM_DEFAULTS["use_format"],
    "use_random_seed": True,
    "velocity_ema_factor": 0.0,
    "velocity_norm_threshold": 0.0,
}

PARAM_ALIASES: dict[str, list[str]] = {
    "ace_lm_model": ["ace_lm_model", "lm_model", "lm_model_path", "lmModel"],
    "audio_code_string": ["audio_code_string", "audioCodeString", "audio_codes"],
    "audio_cover_strength": ["audio_cover_strength", "audioCoverStrength", "cover_strength", "coverStrength"],
    "audio_format": ["audio_format", "audioFormat", "format"],
    "caption": ["caption", "prompt", "tags"],
    "duration": ["duration", "audio_duration", "audioDuration", "target_duration", "targetDuration"],
    "key_scale": ["key_scale", "keyscale", "keyScale", "key"],
    "sample_query": ["sample_query", "sampleQuery"],
    "song_model": ["song_model", "model", "model_name", "modelName", "dit_model", "ditModel"],
    "src_audio_path": ["src_audio_path", "src_audio", "source_audio_path", "source_audio"],
    "time_signature": ["time_signature", "timesignature", "timeSignature"],
    "reference_audio_path": ["reference_audio_path", "reference_audio", "reference"],
    "use_format": ["use_format", "useFormat"],
    "vocal_language": ["vocal_language", "vocalLanguage", "language"],
}

LYRIC_SECTION_RE = re.compile(
    r"(?im)^\s*\[(?:intro|verse|pre[-\s]?chorus|chorus|hook|bridge|drop|break|interlude|outro|refrain|rap|spoken)"
    r"(?:\s+\d+)?\]\s*$"
)
INLINE_LYRIC_SECTION_RE = re.compile(
    r"(?i)\[(?:intro|verse|pre[-\s]?chorus|chorus|hook|bridge|drop|break|interlude|outro|refrain|rap|spoken)"
    r"(?:\s+\d+)?\]"
)


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


def official_manifest() -> dict[str, Any]:
    """Return a copy of the official ACE-Step parity manifest."""
    manifest = copy.deepcopy(OFFICIAL_ACE_STEP_MANIFEST)
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
    if "acestep-v15-turbo" in installed or not installed:
        return "acestep-v15-turbo"
    for candidate in ["acestep-v15-xl-turbo", "acestep-v15-sft", "acestep-v15-base", "acestep-v15-xl-base"]:
        if candidate in installed:
            return candidate
    return "acestep-v15-turbo"


def recommended_lm_model(installed_models: set[str] | list[str] | None = None) -> str:
    installed = set(installed_models or [])
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
        return list(TASK_TYPES_BASE)
    if "turbo" in lowered or lowered.endswith("-sft") or "-sft" in lowered:
        return list(TASK_TYPES_TURBO)
    return list(TASK_TYPES_TURBO)


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
    return {
        "sources": ACE_STEP_MODEL_SOURCES,
        "payload_contract_version": "2026-04-25",
        "official_manifest_version": OFFICIAL_ACE_STEP_MANIFEST["manifest_version"],
        "payload_validation_endpoint": "/api/payload/validate",
        "official_parity_endpoint": "/api/ace-step/parity",
        "audio_formats": sorted(OFFICIAL_AUDIO_FORMATS),
        "fast_audio_formats": sorted(SUPPORTED_AUDIO_FORMATS),
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
            ],
            "ace_step_lm": [],
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
            "bpm": [30, 300],
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
            "normalization_db": [-24.0, 0.0],
            "fade_in_duration": [0.0, 20.0],
            "fade_out_duration": [0.0, 20.0],
            "latent_shift": [-2.0, 2.0],
            "latent_rescale": [0.1, 3.0],
            "lm_repetition_penalty": [0.1, 4.0],
        },
        "defaults": dict(OFFICIAL_FIELD_DEFAULTS),
        "official_only_fields": sorted(OFFICIAL_ONLY_FIELDS),
        "fast_handler_fields": sorted(FAST_HANDLER_FIELDS),
        "official_runtime_controls": OFFICIAL_RUNTIME_CONTROLS,
    }
