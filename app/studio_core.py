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

SUPPORTED_AUDIO_FORMATS = {"wav", "flac", "ogg"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac", ".m4a"}
MAX_BATCH_SIZE = 8


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
    labels = {
        "acestep-v15-turbo": "Turbo",
        "acestep-v15-turbo-shift3": "Turbo Shift3",
        "acestep-v15-sft": "SFT",
        "acestep-v15-base": "Base",
        "acestep-v15-xl-turbo": "XL Turbo",
        "acestep-v15-xl-sft": "XL SFT",
        "acestep-v15-xl-base": "XL Base",
    }
    return labels.get(name, name)


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


def normalize_audio_format(value: str | None) -> str:
    fmt = (value or "wav").strip().lower().lstrip(".")
    if fmt not in SUPPORTED_AUDIO_FORMATS:
        return "wav"
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
