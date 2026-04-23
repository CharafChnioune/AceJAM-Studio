from __future__ import annotations

import re
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
]

STANDARD_TASKS = ["text2music", "cover", "repaint"]
ALL_TASKS = ["text2music", "cover", "repaint", "extract", "lego", "complete"]

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
        "steps": "32-64 recommended",
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
        "steps": "32-64 recommended",
        "cfg": "Yes",
        "tasks": ALL_TASKS,
        "recommended_for": ["extract", "lego", "complete", "lora"],
        "warnings": ["Most demanding DiT option."],
        "notes": "Official XL model zoo lists XL Base for all tasks, including extract, lego, and complete.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
}

LM_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "auto": {
        "label": "Auto",
        "dropdown_label": "Auto - use best available LM",
        "summary": "Use AceJAM's recommended ACE-Step LM when local routing supports it.",
        "best_for": ["Default planning"],
        "quality": "Recommended",
        "speed": "Balanced",
        "vram": "Auto",
        "steps": "N/A",
        "cfg": "LM CFG 2.0 default",
        "tasks": ["metadata", "caption rewrite", "audio understanding"],
        "warnings": [],
        "notes": "AceJAM keeps Ollama as the songwriter layer and exposes ACE-Step LM guidance for official 5Hz planner choices.",
        "source_urls": ACE_STEP_MODEL_SOURCES,
    },
    "none": {
        "label": "No ACE LM",
        "dropdown_label": "None - manual, fastest",
        "summary": "Fastest/manual path when metadata, lyrics, and prompt structure are already controlled.",
        "best_for": ["Manual control", "Fastest flow", "Precise metadata"],
        "quality": "Manual",
        "speed": "Fastest",
        "vram": "Lowest",
        "steps": "N/A",
        "cfg": "No LM CFG",
        "tasks": ["manual metadata"],
        "warnings": ["Less automatic caption and metadata help."],
        "notes": "Inference docs recommend disabling LM when precise metadata is already provided or speed matters most.",
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
    "vocal_language",
}

OFFICIAL_ONLY_FIELDS = {
    "allow_lm_batch",
    "chunk_mask_mode",
    "constrained_decoding_debug",
    "cover_noise_strength",
    "enable_normalization",
    "fade_in_duration",
    "fade_out_duration",
    "global_caption",
    "latent_rescale",
    "latent_shift",
    "lm_batch_chunk_size",
    "lm_backend",
    "lm_cfg_scale",
    "lm_negative_prompt",
    "lm_temperature",
    "lm_top_k",
    "lm_top_p",
    "repaint_latent_crossfade_frames",
    "repaint_mode",
    "repaint_strength",
    "repaint_wav_crossfade_sec",
    "sample_mode",
    "sample_query",
    "sampler_mode",
    "thinking",
    "use_constrained_decoding",
    "use_cot_caption",
    "use_cot_language",
    "use_cot_lyrics",
    "use_cot_metas",
    "use_format",
    "velocity_ema_factor",
    "velocity_norm_threshold",
}

OFFICIAL_FIELD_DEFAULTS: dict[str, Any] = {
    "allow_lm_batch": False,
    "chunk_mask_mode": "auto",
    "constrained_decoding_debug": False,
    "cover_noise_strength": 0.0,
    "enable_normalization": True,
    "fade_in_duration": 0.0,
    "fade_out_duration": 0.0,
    "global_caption": "",
    "instrumental": False,
    "latent_rescale": 1.0,
    "latent_shift": 0.0,
    "lm_batch_chunk_size": 8,
    "lm_backend": "auto",
    "lm_cfg_scale": 2.0,
    "lm_negative_prompt": "NO USER INPUT",
    "lm_temperature": 0.85,
    "lm_top_k": 0,
    "lm_top_p": 0.9,
    "mp3_bitrate": "128k",
    "mp3_sample_rate": 48000,
    "repaint_latent_crossfade_frames": 10,
    "repaint_mode": "balanced",
    "repaint_strength": 0.5,
    "repaint_wav_crossfade_sec": 0.0,
    "sample_mode": False,
    "sample_query": "",
    "sampler_mode": "euler",
    "thinking": False,
    "use_constrained_decoding": True,
    "use_cot_caption": True,
    "use_cot_language": True,
    "use_cot_lyrics": False,
    "use_cot_metas": True,
    "use_format": False,
    "velocity_ema_factor": 0.0,
    "velocity_norm_threshold": 0.0,
}

PARAM_ALIASES: dict[str, list[str]] = {
    "audio_code_string": ["audio_code_string", "audioCodeString", "audio_codes"],
    "audio_cover_strength": ["audio_cover_strength", "audioCoverStrength", "cover_strength", "coverStrength"],
    "duration": ["duration", "audio_duration", "audioDuration", "target_duration", "targetDuration"],
    "key_scale": ["key_scale", "keyscale", "keyScale", "key"],
    "sample_query": ["sample_query", "sampleQuery", "description", "desc"],
    "song_model": ["song_model", "model", "model_name", "modelName", "dit_model", "ditModel"],
    "time_signature": ["time_signature", "timesignature", "timeSignature"],
    "use_format": ["use_format", "useFormat", "format"],
    "vocal_language": ["vocal_language", "vocalLanguage", "language"],
}


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


def model_label(name: str) -> str:
    profile = MODEL_PROFILES.get(name)
    return str(profile["label"]) if profile else name


def _profile_with_runtime_fields(profile: dict[str, Any], name: str, installed: bool) -> dict[str, Any]:
    item = {key: value[:] if isinstance(value, list) else value for key, value in profile.items()}
    item["name"] = name
    item["installed"] = installed
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
            "steps": "32-64 recommended" if is_base else "8-50 inferred",
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
    if "acestep-5Hz-lm-1.7B" in installed or not installed:
        return "acestep-5Hz-lm-1.7B"
    for candidate in ["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B", "auto", "none"]:
        if candidate in installed:
            return candidate
    return "auto"


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
        "audio_formats": sorted(OFFICIAL_AUDIO_FORMATS),
        "fast_audio_formats": sorted(SUPPORTED_AUDIO_FORMATS),
        "task_lm_usage": {
            "uses_lm": ["text2music", "lego", "complete"],
            "skips_lm": ["cover", "repaint", "extract"],
        },
        "custom_sections": {
            "song": ["title", "duration", "instrumental", "vocal_language", "caption", "lyrics", "reference_audio"],
            "music_metadata": ["bpm", "key_scale", "time_signature", "global_caption"],
            "generation": [
                "batch_size",
                "seed",
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
            "ace_step_lm": [
                "ace_lm_model",
                "thinking",
                "sample_mode",
                "sample_query",
                "use_format",
                "lm_temperature",
                "lm_cfg_scale",
                "lm_top_k",
                "lm_top_p",
                "lm_negative_prompt",
                "lm_backend",
                "use_cot_metas",
                "use_cot_caption",
                "use_cot_lyrics",
                "use_cot_language",
                "allow_lm_batch",
                "use_constrained_decoding",
                "constrained_decoding_debug",
            ],
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
        },
        "defaults": dict(OFFICIAL_FIELD_DEFAULTS),
        "official_only_fields": sorted(OFFICIAL_ONLY_FIELDS),
        "fast_handler_fields": sorted(FAST_HANDLER_FIELDS),
    }
