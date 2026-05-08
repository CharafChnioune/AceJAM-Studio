from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lora_utils import safe_peft_adapter_name

try:
    import soundfile as sf
except Exception:  # pragma: no cover - dependency is installed in the app env.
    sf = None


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac", ".m4a"}
JOB_ACTIVE_STATES = {"queued", "running", "stopping"}
EPOCH_AUDITION_DURATION_SECONDS = 30
EPOCH_AUDITION_CHARS_PER_SECOND = 8
EPOCH_AUDITION_MAX_SUNG_LINES_PER_SECTION = 2
EPOCH_AUDITION_SECONDS_PER_SUNG_LINE = 5.0
EPOCH_AUDITION_SECTION_STYLE = "clear dry lead vocal, intelligible delivery"
EPOCH_AUDITION_CLARITY_CAPTION = (
    "30-second LoRA audition, clear intelligible vocal, dry upfront lead vocal, "
    "sparse arrangement, steady rhythm, no noisy vocal artifacts"
)
DEFAULT_LORA_GENERATION_SCALE = 0.45
DEFAULT_SFT_BASE_INFERENCE_STEPS = 50
DEFAULT_SFT_BASE_SHIFT = 1.0
DEFAULT_TURBO_INFERENCE_STEPS = 8
DEFAULT_TURBO_SHIFT = 3.0
VOCAL_MIN_REAL_LYRICS_RATIO = 0.95
VOCAL_MIN_METADATA_RATIO = 0.90
VOCAL_MIN_LANGUAGE_RATIO = 0.90
INSTRUMENTAL_SENTINEL = "[Instrumental]"
MISSING_LYRICS_SOURCES = {
    "default_instrumental",
    "deterministic_filename",
    "filename_fallback",
    "filename_duration_fallback",
    "online_lyrics_missing",
    "understand_music_failed",
    "error",
}
TRUSTED_ONLINE_LYRICS_SOURCES = {
    "genius",
    "online_lyrics",
    "online_lyrics_genius",
    "online_lyrics_ovh",
    "smart_musicbrainz",
}
NONFINITE_TRAINING_LOSS_RE = re.compile(r"\bLoss:\s*(?:nan|[-+]?inf(?:inity)?)\b", re.IGNORECASE)
DEFAULT_LORA_TRAINING_SONG_MODEL = "acestep-v15-xl-sft"
VARIANT_TO_SONG_MODEL = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
    "xl_turbo": "acestep-v15-xl-turbo",
    "xl_base": "acestep-v15-xl-base",
    "xl_sft": "acestep-v15-xl-sft",
}
SONG_MODEL_TO_VARIANT = {model: variant for variant, model in VARIANT_TO_SONG_MODEL.items()}

EPOCH_AUDITION_GENRE_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "key": "rap",
        "label": "Rap / Hip-hop",
        "terms": ("rap", "hip hop", "hip-hop", "trap", "drill", "boom bap", "west coast", "gangster", "2pac", "tupac"),
        "caption_tags": "rap, hip hop, rhythmic spoken-word vocal, clear rap flow, deep bass, hard drums",
        "lyrics_section_tags": {"verse": "rap, rhythmic spoken flow", "chorus": "rap hook"},
        "lyrics": "[Verse - rap, rhythmic spoken flow]\nI step to the light with the pressure on ten\nEvery bar lands clean when the drums come in\n\n[Chorus - rap hook]\nHands in the air when the bassline rolls\nSay it one time and the whole room knows",
        "bpm": 95,
        "keyscale": "A minor",
        "timesignature": "4",
        "vocal_language": "en",
    },
    {
        "key": "pop",
        "label": "Pop",
        "terms": ("pop", "radio", "dance pop", "synth pop", "electropop"),
        "caption_tags": "modern pop groove, bright hook, clean lead vocal, radio-ready drums",
        "lyrics_section_tags": {"verse": "clean pop vocal", "chorus": "bright pop hook"},
        "lyrics": "[Verse - clean pop vocal]\nCity lights are turning gold tonight\nWe chase the spark until the morning light\n\n[Chorus - bright pop hook]\nHold on hold on we are alive\nHearts beat louder when the chorus arrives",
        "bpm": 118,
        "keyscale": "C major",
        "timesignature": "4",
        "vocal_language": "en",
    },
    {
        "key": "rnb",
        "label": "Soul / R&B",
        "terms": ("r&b", "rnb", "soul", "neo soul", "smooth", "slow jam"),
        "caption_tags": "smooth rnb groove, warm keys, clean intimate lead vocal, soft harmonies",
        "lyrics_section_tags": {"verse": "smooth rnb vocal", "chorus": "soulful rnb hook"},
        "lyrics": "[Verse - smooth rnb vocal]\nLate night glow on the window frame\nYour voice comes close and it says my name\n\n[Chorus - soulful rnb hook]\nStay right here where the rhythm is slow\nLet the whole room breathe when the candles glow",
        "bpm": 82,
        "keyscale": "D minor",
        "timesignature": "4",
        "vocal_language": "en",
    },
    {
        "key": "rock",
        "label": "Rock",
        "terms": ("rock", "guitar", "punk", "metal", "alt rock", "indie rock"),
        "caption_tags": "driving rock drums, electric guitars, clear lead vocal, strong chorus",
        "lyrics_section_tags": {"verse": "rock lead vocal", "chorus": "strong rock chorus"},
        "lyrics": "[Verse - rock lead vocal]\nRoad lights flash on the edge of town\nWe hit the floor when the walls come down\n\n[Chorus - strong rock chorus]\nRaise it up with the thunder and fire\nOne loud heart in a live wire choir",
        "bpm": 128,
        "keyscale": "E minor",
        "timesignature": "4",
        "vocal_language": "en",
    },
    {
        "key": "edm",
        "label": "EDM / Dance",
        "terms": ("edm", "house", "techno", "trance", "club", "dance", "electronic"),
        "caption_tags": "electronic dance beat, pulsing synth bass, clean vocal hook, club energy",
        "lyrics_section_tags": {"verse": "dance vocal", "chorus": "club vocal hook"},
        "lyrics": "[Verse - dance vocal]\nBlue lights move when the kick comes through\nEvery heartbeat locks into the groove\n\n[Chorus - club vocal hook]\nLift me higher when the drop arrives\nWe come alive under flashing lights",
        "bpm": 124,
        "keyscale": "F# minor",
        "timesignature": "4",
        "vocal_language": "en",
    },
    {
        "key": "cinematic",
        "label": "Cinematic",
        "terms": ("cinematic", "orchestral", "score", "trailer", "choir", "epic"),
        "caption_tags": "cinematic drums, wide strings, clear dramatic vocal, spacious arrangement",
        "lyrics_section_tags": {"verse": "dramatic vocal", "chorus": "cinematic anthem"},
        "lyrics": "[Verse - dramatic vocal]\nStars lean close as the shadows rise\nWe hold the line under open skies\n\n[Chorus - cinematic anthem]\nStand as one when the thunder calls\nLight breaks through every ancient wall",
        "bpm": 88,
        "keyscale": "D minor",
        "timesignature": "4",
        "vocal_language": "en",
    },
    {
        "key": "country",
        "label": "Country / Folk",
        "terms": ("country", "folk", "americana", "acoustic", "banjo"),
        "caption_tags": "warm acoustic guitars, steady country drums, clear heartfelt vocal",
        "lyrics_section_tags": {"verse": "country lead vocal", "chorus": "heartfelt country hook"},
        "lyrics": "[Verse - country lead vocal]\nDust on my boots and the sun sinking low\nOne more mile down a familiar road\n\n[Chorus - heartfelt country hook]\nTake me home where the porch light shines\nGood hearts gather at closing time",
        "bpm": 96,
        "keyscale": "G major",
        "timesignature": "4",
        "vocal_language": "en",
    },
)
EPOCH_AUDITION_DEFAULT_PROFILE_KEY = "pop"
STYLE_PROFILE_AUTO = "auto"


def _style_term_key(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.replace("_", " ")


def _style_profile_by_key(key: str | None) -> dict[str, Any] | None:
    requested = _style_term_key(key)
    if not requested or requested == STYLE_PROFILE_AUTO:
        return None
    for profile in EPOCH_AUDITION_GENRE_PROFILES:
        keys = [
            str(profile.get("key") or ""),
            str(profile.get("label") or ""),
            *[str(term) for term in profile.get("terms", ())],
        ]
        if any(_style_term_key(item) == requested for item in keys):
            return dict(profile)
    return None


def _style_profile_public(profile: dict[str, Any]) -> dict[str, Any]:
    lyrics = str(profile.get("lyrics") or "").strip()
    section_tags = dict(profile.get("lyrics_section_tags") or {})
    return {
        "key": str(profile.get("key") or ""),
        "style_profile": str(profile.get("key") or ""),
        "label": str(profile.get("label") or profile.get("key") or "").strip(),
        "caption_tags": str(profile.get("caption_tags") or "").strip(),
        "lyrics_section_tags": section_tags,
        "lyrics": lyrics,
        "test_lyrics": lyrics,
        "bpm": profile.get("bpm"),
        "keyscale": str(profile.get("keyscale") or "").strip(),
        "timesignature": str(profile.get("timesignature") or "").strip(),
        "vocal_language": str(profile.get("vocal_language") or "en").strip() or "en",
        "terms": list(profile.get("terms") or []),
        "default": profile.get("key") == EPOCH_AUDITION_DEFAULT_PROFILE_KEY,
    }


def audio_style_profiles(*, include_auto: bool = True) -> list[dict[str, Any]]:
    """Public catalog shared by audio generation and LoRA epoch auditions."""
    profiles: list[dict[str, Any]] = []
    if include_auto:
        profiles.append(
            {
                "key": STYLE_PROFILE_AUTO,
                "style_profile": STYLE_PROFILE_AUTO,
                "label": "Auto",
                "caption_tags": "Infer style from the prompt/caption without forcing a fallback genre.",
                "lyrics_section_tags": {},
                "lyrics": "",
                "test_lyrics": "",
                "bpm": None,
                "keyscale": "",
                "timesignature": "",
                "vocal_language": "",
                "default": False,
            }
        )
    profiles.extend(_style_profile_public(dict(profile)) for profile in EPOCH_AUDITION_GENRE_PROFILES)
    return profiles


def _dedupe_csv_terms(*values: Any) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            chunks = value
        else:
            chunks = str(value or "").split(",")
        for chunk in chunks:
            cleaned = re.sub(r"\s+", " ", str(chunk or "").strip())
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(cleaned)
    return ", ".join(parts)


def _section_kind_and_display(raw_label: str) -> tuple[str, str, str]:
    raw = re.sub(r"\s+", " ", str(raw_label or "").strip())
    raw = re.sub(r"\s*[-:|]\s*.*$", "", raw).strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    normalized_no_num = re.sub(r"\s+\d+$", "", normalized).strip()
    if normalized_no_num in {"pre chorus", "prechorus"}:
        return "pre_chorus", "Pre-Chorus", raw
    if normalized_no_num in {"chorus", "final chorus", "hook", "refrain"}:
        return "chorus", "Chorus", raw
    if normalized_no_num in {"verse", "rap", "spoken", "spoken word"}:
        return "verse", "Verse", raw
    if normalized_no_num in {"intro"}:
        return "intro", "Intro", raw
    if normalized_no_num in {"bridge"}:
        return "bridge", "Bridge", raw
    if normalized_no_num in {"drop", "break", "interlude", "breakdown"}:
        return "break", "Break", raw
    if normalized_no_num in {"outro"}:
        return "outro", "Outro", raw
    return "", raw, raw


def _concise_performance_tags(tags: str) -> str:
    allowed = ("rap", "spoken", "vocal", "hook", "rnb", "soul", "pop", "rock", "country", "dance", "dramatic", "anthem", "lead", "flow")
    kept: list[str] = []
    for part in str(tags or "").split(","):
        cleaned = re.sub(r"\s+", " ", part).strip()
        if not cleaned:
            continue
        text = cleaned.lower()
        if any(token in text for token in allowed):
            kept.append(cleaned)
        if len(kept) >= 3:
            break
    return _dedupe_csv_terms(kept)


def _existing_section_tags(raw_label: str, display_raw: str) -> str:
    raw = str(raw_label or "").strip()
    match = re.search(r"\s[-:|]\s*(.+)$", raw)
    if match:
        return match.group(1).strip()
    # Keep unstructured descriptors after "Verse 1 ..." only when they look
    # like performance tags, not when they are merely section numbering.
    remainder = raw[len(display_raw):].strip() if display_raw and raw.lower().startswith(display_raw.lower()) else ""
    remainder = re.sub(r"^\d+\s*", "", remainder).strip()
    return remainder


def _style_tags_for_section(profile: dict[str, Any], section_kind: str) -> str:
    section_tags = dict(profile.get("lyrics_section_tags") or {})
    if section_kind == "pre_chorus":
        return str(section_tags.get("chorus") or "").strip()
    if section_kind in {"hook", "refrain"}:
        return str(section_tags.get("chorus") or "").strip()
    return str(section_tags.get(section_kind) or "").strip()


def _format_section_header(display: str, tags: str = "") -> str:
    clean_display = re.sub(r"\s+", " ", str(display or "Verse").strip()) or "Verse"
    clean_tags = _dedupe_csv_terms(tags)
    return f"[{clean_display} - {clean_tags}]" if clean_tags else f"[{clean_display}]"


def _enrich_audio_style_lyrics(lyrics: str, profile: dict[str, Any]) -> tuple[str, list[str]]:
    raw = str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip() or raw.strip().lower() == INSTRUMENTAL_SENTINEL.lower():
        return raw, []
    applied: list[str] = []
    output: list[str] = []
    saw_section = False
    for line in raw.splitlines():
        match = re.fullmatch(r"\s*[*_`~]*\s*\[([^\]]+)\]\s*[*_`~]*\s*", line)
        if not match:
            output.append(line)
            continue
        saw_section = True
        raw_label = match.group(1).strip()
        section_kind, display, display_raw = _section_kind_and_display(raw_label)
        style_tags = _style_tags_for_section(profile, section_kind)
        if not section_kind or not style_tags:
            output.append(line)
            continue
        existing_tags = _existing_section_tags(raw_label, display_raw)
        merged_tags = _dedupe_csv_terms(style_tags, existing_tags)
        header = _format_section_header(display, merged_tags)
        output.append(header)
        applied.append(f"{display}: {merged_tags}")
    if not saw_section and raw.strip():
        section_tags = _style_tags_for_section(profile, "verse")
        header = _format_section_header("Verse", section_tags)
        return f"{header}\n{raw.strip()}", [f"Verse: {section_tags}"] if section_tags else []
    return "\n".join(output).strip(), applied


def _style_caption_has_rap(caption: str) -> bool:
    text = _style_term_key(caption)
    return any(term in text for term in ("rap", "hip hop", "hip-hop", "spoken-word", "spoken word"))


def _style_lyrics_have_rap(lyrics: str) -> bool:
    text = _style_term_key(lyrics)
    return bool(re.search(r"\[(?:verse|chorus|hook)[^\]]*(?:rap|spoken)", text, flags=re.IGNORECASE))


def audio_style_conditioning_audit(caption: str, lyrics: str, style_profile: str | None) -> dict[str, Any]:
    profile = _style_profile_by_key(style_profile)
    if not profile:
        return {"status": "skipped", "style_profile": str(style_profile or STYLE_PROFILE_AUTO)}
    key = str(profile.get("key") or "")
    missing: list[str] = []
    if key == "rap":
        if not _style_caption_has_rap(caption):
            missing.append("caption_rap_hiphop_spoken_word")
        if str(lyrics or "").strip().lower() != INSTRUMENTAL_SENTINEL.lower() and not _style_lyrics_have_rap(lyrics):
            missing.append("lyrics_rap_performance_section")
    else:
        caption_tags = str(profile.get("caption_tags") or "")
        first_tag = next((term.strip() for term in caption_tags.split(",") if term.strip()), key)
        if first_tag and first_tag.lower() not in str(caption or "").lower():
            missing.append("caption_style_tags")
        style_tags = ", ".join(str(v) for v in dict(profile.get("lyrics_section_tags") or {}).values())
        if style_tags and str(lyrics or "").strip().lower() != INSTRUMENTAL_SENTINEL.lower():
            first_lyric_tag = next((term.strip() for term in style_tags.split(",") if term.strip()), "")
            if first_lyric_tag and first_lyric_tag.lower() not in str(lyrics or "").lower():
                missing.append("lyrics_performance_section_tags")
    return {
        "status": "fail" if missing else "pass",
        "style_profile": key,
        "missing": missing,
        "caption_tags": str(profile.get("caption_tags") or ""),
        "lyrics_section_tags": dict(profile.get("lyrics_section_tags") or {}),
    }


def apply_audio_style_conditioning(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply docs-correct style conditioning to a generation payload.

    ACE-Step's docs recommend complex style/timbre/instrument guidance in
    caption, while lyric tags should stay concise. This normalizer keeps that
    split and only enriches section headers in lyrics.
    """
    updated = dict(payload or {})
    requested = str(
        updated.get("style_profile")
        or updated.get("genre_profile")
        or updated.get("audio_style_profile")
        or STYLE_PROFILE_AUTO
    ).strip().lower() or STYLE_PROFILE_AUTO
    profile = _style_profile_by_key(requested)
    if profile is None and requested == STYLE_PROFILE_AUTO:
        inferred = epoch_audition_genre_profile(
            str(updated.get("caption") or updated.get("tags") or updated.get("custom_tags") or ""),
            str(updated.get("lyrics") or ""),
            "",
        )
        search_text = _epoch_audition_search_text(
            str(updated.get("caption") or ""),
            str(updated.get("tags") or ""),
            str(updated.get("custom_tags") or ""),
            str(updated.get("lyrics") or ""),
        )
        inferred_terms = [str(term) for term in inferred.get("terms", ())]
        if any(_epoch_audition_search_text(term) in search_text for term in inferred_terms):
            profile = inferred
    if profile is None:
        updated["style_profile"] = STYLE_PROFILE_AUTO
        updated["style_caption_tags"] = ""
        updated["style_lyric_tags_applied"] = []
        updated["style_conditioning_audit"] = audio_style_conditioning_audit(
            str(updated.get("caption") or ""),
            str(updated.get("lyrics") or ""),
            STYLE_PROFILE_AUTO,
        )
        return updated

    caption = _dedupe_csv_terms(
        str(profile.get("caption_tags") or ""),
        updated.get("caption"),
        updated.get("tags"),
        updated.get("custom_tags"),
    )
    lyrics, lyric_tags = _enrich_audio_style_lyrics(str(updated.get("lyrics") or ""), profile)
    key = str(profile.get("key") or "")
    updated["caption"] = caption
    if str(updated.get("lyrics") or "").strip():
        updated["lyrics"] = lyrics
    updated["style_profile"] = key
    updated["style_caption_tags"] = str(profile.get("caption_tags") or "")
    updated["style_lyric_tags_applied"] = lyric_tags
    updated["style_conditioning_audit"] = audio_style_conditioning_audit(caption, str(updated.get("lyrics") or ""), key)
    warnings = list(updated.get("payload_warnings") or [])
    warning = f"audio_style_profile_applied:{key}"
    if warning not in warnings:
        warnings.append(warning)
    updated["payload_warnings"] = warnings
    return updated


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str, fallback: str = "item") -> str:
    import re

    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or fallback)).strip("-._")
    return text[:90] or fallback


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def parse_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _epoch_audition_section_marker(line: str) -> str | None:
    match = re.fullmatch(r"\s*[*_`~]*\s*\[([^\]]+)\]\s*[*_`~]*\s*", str(line or ""))
    if not match:
        return None
    raw_label = match.group(1).strip()
    section_kind, display, display_raw = _section_kind_and_display(raw_label)
    if not section_kind:
        return ""
    existing_tags = _concise_performance_tags(_existing_section_tags(raw_label, display_raw))
    return _format_section_header(display, existing_tags)


def _epoch_audition_blocks(lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current_section = "[Verse]"
    current_lines: list[str] = []
    for line in lines:
        section = _epoch_audition_section_marker(line)
        if section is not None:
            if current_lines:
                blocks.append({"section": current_section, "lines": current_lines})
            elif current_section.lower().startswith("[verse") and section.lower().startswith("[outro"):
                current_lines = []
                continue
            current_section = section or current_section
            current_lines = []
            continue
        text = str(line or "").strip()
        if text:
            current_lines.append(text)
    if current_lines:
        blocks.append({"section": current_section, "lines": current_lines})
    return blocks


def _format_epoch_audition_time(seconds: float | int) -> str:
    total = max(0, int(round(float(seconds))))
    return f"{total // 60}:{total % 60:02d}"


def _epoch_audition_timed_section(section: str, start: float, end: float) -> str:
    section_name = str(section or "[Verse]").strip().strip("[]") or "Verse"
    # ACE-Step docs warn that stacked lyric tags can be sung literally or
    # confuse the model. Keep timing in metadata, and keep lyric tags concise.
    if section_name.lower().startswith("verse") and " - " not in section_name:
        return f"[{section_name} - spoken word]"
    return f"[{section_name}]"


def _epoch_audition_search_text(*values: str | None) -> str:
    text = " ".join(str(value or "") for value in values)
    text = text.replace("-", " ")
    return re.sub(r"[^a-z0-9&]+", " ", text.lower()).strip()


def epoch_audition_genre_profile(
    caption: str | None = "",
    lyrics_hint: str | None = "",
    genre_key: str | None = "",
) -> dict[str, Any]:
    requested = _epoch_audition_search_text(genre_key)
    if requested and requested != "auto":
        for profile in EPOCH_AUDITION_GENRE_PROFILES:
            keys = [
                str(profile.get("key") or ""),
                str(profile.get("label") or ""),
                *[str(term) for term in profile.get("terms", ())],
            ]
            if any(_epoch_audition_search_text(item) == requested for item in keys):
                return dict(profile)
    search_text = _epoch_audition_search_text(caption, lyrics_hint)
    for profile in EPOCH_AUDITION_GENRE_PROFILES:
        for term in profile.get("terms", ()):
            if _epoch_audition_search_text(str(term)) in search_text:
                return dict(profile)
    for profile in EPOCH_AUDITION_GENRE_PROFILES:
        if profile.get("key") == EPOCH_AUDITION_DEFAULT_PROFILE_KEY:
            return dict(profile)
    return dict(EPOCH_AUDITION_GENRE_PROFILES[0])


def epoch_audition_genre_options() -> list[dict[str, Any]]:
    """Public UI catalog for LoRA test-WAV lyrics, tags, and metadata."""
    options = audio_style_profiles(include_auto=True)
    if options:
        options[0]["caption_tags"] = "Uses the dataset captions to infer rap, pop, soul, rock, EDM, cinematic, or country."
        options[0]["lyrics"] = ""
        options[0]["test_lyrics"] = ""
    return options


def _append_caption_parts(parts: list[str], value: str | None, seen: set[str]) -> None:
    for part in str(value or "").split(","):
        cleaned = re.sub(r"\s+", " ", part).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(cleaned)


def _caption_contains_trigger_tag(caption: str, trigger_tag: str) -> bool:
    trigger = str(trigger_tag or "").strip()
    if not trigger:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(trigger)}(?![A-Za-z0-9])"
    return re.search(pattern, str(caption or ""), flags=re.IGNORECASE) is not None


def safe_generation_trigger_tag(trigger_tag: str | None) -> str:
    """Return a trigger text that is less likely to poison ACE-Step test prompts."""
    trigger = re.sub(r"\s+", " ", str(trigger_tag or "").strip())
    if not trigger:
        return ""
    compact = re.sub(r"[^A-Za-z0-9]+", "", trigger).lower()
    if compact == "2pac":
        return "pac"
    if re.fullmatch(r"\d+[A-Za-z][A-Za-z0-9_-]*", trigger):
        candidate = re.sub(r"^\d+", "", trigger)
        candidate = re.sub(r"[_-]+", " ", candidate).strip()
        return candidate or trigger
    return trigger


def build_epoch_audition_caption(
    caption: str | None = "",
    *,
    trigger_tag: str | None = "",
    genre_key: str | None = "",
    genre_profile: dict[str, Any] | None = None,
) -> str:
    profile = dict(genre_profile or epoch_audition_genre_profile(caption, genre_key=genre_key))
    parts: list[str] = []
    seen: set[str] = set()
    _append_caption_parts(parts, safe_generation_trigger_tag(trigger_tag), seen)
    _append_caption_parts(parts, caption, seen)
    _append_caption_parts(parts, str(profile.get("caption_tags") or ""), seen)
    _append_caption_parts(parts, EPOCH_AUDITION_CLARITY_CAPTION, seen)
    return ", ".join(parts)


def default_epoch_audition_lyrics(
    caption: str | None = "",
    *,
    trigger_tag: str | None = "",
    lyrics_hint: str | None = "",
    genre_key: str | None = "",
) -> tuple[str, dict[str, Any]]:
    profile = epoch_audition_genre_profile(caption, lyrics_hint, genre_key)
    lyrics = str(profile.get("lyrics") or "").strip()
    meta = {
        "lyrics_source": "genre_default",
        "genre_profile": str(profile.get("key") or EPOCH_AUDITION_DEFAULT_PROFILE_KEY),
        "trigger_in_lyrics": bool(trigger_tag and re.search(re.escape(str(trigger_tag).strip()), lyrics, flags=re.IGNORECASE)),
        "user_lyrics_chars": len(str(lyrics_hint or "").strip()),
    }
    return lyrics, meta


def _profile_audition_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    bpm = parse_int(profile.get("bpm"), 0, 0, 300)
    return {
        "bpm": bpm if bpm > 0 else None,
        "keyscale": str(profile.get("keyscale") or "").strip(),
        "timesignature": str(profile.get("timesignature") or "").strip(),
    }


def _most_common_nonempty(values: list[Any]) -> str:
    counts: dict[str, int] = {}
    order: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text not in counts:
            counts[text] = 0
            order.append(text)
        counts[text] += 1
    if not counts:
        return ""
    return sorted(order, key=lambda item: (-counts[item], order.index(item)))[0]


def _median_positive_bpm(values: list[Any]) -> int | None:
    bpms: list[float] = []
    for value in values:
        try:
            bpm = float(value)
        except (TypeError, ValueError):
            continue
        if bpm > 0:
            bpms.append(bpm)
    if not bpms:
        return None
    bpms.sort()
    mid = len(bpms) // 2
    if len(bpms) % 2:
        return int(round(bpms[mid]))
    return int(round((bpms[mid - 1] + bpms[mid]) / 2.0))


def dataset_epoch_audition_metadata(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Return stable music metadata for no-LM epoch auditions.

    ACE-Step can infer metadata when its LM formatting path is enabled, but
    epoch auditions intentionally run without LM for speed and reproducibility.
    Passing BPM/key/time from the reviewed dataset keeps those calls complete.
    """
    bpm = _median_positive_bpm([entry.get("bpm") for entry in labels])
    keyscale = _most_common_nonempty([entry.get("keyscale") or entry.get("key_scale") for entry in labels])
    timesignature = _most_common_nonempty([entry.get("timesignature") or entry.get("time_signature") for entry in labels])
    return {
        "bpm": bpm,
        "keyscale": keyscale,
        "timesignature": timesignature,
    }


def _add_epoch_audition_timing(blocks: list[dict[str, Any]], *, duration: int) -> tuple[str, list[dict[str, Any]]]:
    active_blocks = [block for block in blocks if block.get("lines")]
    if not active_blocks:
        return "", []
    section_count = len(active_blocks)
    segment = max(1.0, float(duration) / float(section_count))
    output: list[str] = []
    slices: list[dict[str, Any]] = []
    for index, block in enumerate(active_blocks):
        start = min(float(duration), index * segment)
        end = float(duration) if index == section_count - 1 else min(float(duration), (index + 1) * segment)
        section = str(block.get("section") or "[Verse]")
        lines = [str(line).strip() for line in list(block.get("lines") or []) if str(line).strip()]
        if not lines:
            continue
        if output:
            output.append("")
        output.append(_epoch_audition_timed_section(section, start, end))
        output.extend(lines)
        slices.append(
            {
                "section": section.strip("[]"),
                "start": round(start, 3),
                "end": round(end, 3),
                "start_label": _format_epoch_audition_time(start),
                "end_label": _format_epoch_audition_time(end),
                "line_count": len(lines),
            }
        )
    return "\n".join(output).strip(), slices


def fit_epoch_audition_lyrics(lyrics: str | None, *, duration: int = EPOCH_AUDITION_DURATION_SECONDS) -> tuple[str, dict[str, Any]]:
    raw = str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    duration_seconds = max(10, int(duration or EPOCH_AUDITION_DURATION_SECONDS))
    lyric_char_budget = int(duration_seconds * EPOCH_AUDITION_CHARS_PER_SECOND)
    # Timing is tracked in metadata, not embedded into the lyric tags. This keeps
    # runtime lyrics aligned with ACE-Step's concise tag guidance.
    max_chars = max(280, min(420, lyric_char_budget + 140))
    max_sung_lines = max(2, min(8, int(round(duration_seconds / EPOCH_AUDITION_SECONDS_PER_SUNG_LINE))))
    source_lines = [line for line in raw.splitlines() if line.strip()]
    normalized_lines: list[str] = []
    for raw_line in source_lines:
        stripped = raw_line.strip()
        section = _epoch_audition_section_marker(stripped)
        if section is not None:
            if section:
                normalized_lines.append(section)
            continue
        if re.match(r"(?i)^(lyrics?|caption|metadata|bpm|keyscale|duration|language)\s*:", stripped):
            continue
        normalized_lines.append(stripped)

    blocks = _epoch_audition_blocks(normalized_lines)
    if not blocks and normalized_lines:
        blocks = [{"section": "[Verse]", "lines": [line for line in normalized_lines if _epoch_audition_section_marker(line) is None]}]

    output_blocks: list[dict[str, Any]] = []
    sung_lines = 0
    for block in blocks:
        if sung_lines >= max_sung_lines:
            break
        lines = list(block.get("lines") or [])[:EPOCH_AUDITION_MAX_SUNG_LINES_PER_SECTION]
        if not lines:
            continue
        section = str(block.get("section") or "[Verse]")
        selected: list[str] = []
        added_for_section = 0
        for line in lines:
            if sung_lines >= max_sung_lines:
                break
            candidate_blocks = [*output_blocks, {"section": section, "lines": [*selected, line]}]
            candidate, _ = _add_epoch_audition_timing(candidate_blocks, duration=duration_seconds)
            if len(candidate) > max_chars:
                break
            selected.append(line)
            sung_lines += 1
            added_for_section += 1
        if added_for_section == 0:
            break
        output_blocks.append({"section": section, "lines": selected})

    fitted, time_slices = _add_epoch_audition_timing(output_blocks, duration=duration_seconds)
    if not fitted and raw:
        kept: list[str] = ["[Verse]"]
        for line in source_lines:
            if _epoch_audition_section_marker(line) is not None:
                continue
            candidate, _ = _add_epoch_audition_timing([{"section": "[Verse]", "lines": [*kept[1:], line.strip()]}], duration=duration_seconds)
            if len(candidate) > max_chars or len(kept) - 1 >= max_sung_lines:
                break
            kept.append(line.strip())
        fitted, time_slices = _add_epoch_audition_timing([{"section": "[Verse]", "lines": kept[1:]}], duration=duration_seconds)

    source_char_count = len(raw)
    runtime_char_count = len(fitted)
    runtime_lines = [line for line in fitted.splitlines() if line.strip()]
    runtime_sung_lines = [line for line in runtime_lines if _epoch_audition_section_marker(line) is None]
    meta = {
        "duration": duration_seconds,
        "max_chars": max_chars,
        "max_sung_lines": max_sung_lines,
        "source_lyrics_chars": source_char_count,
        "runtime_lyrics_chars": runtime_char_count,
        "source_lyrics_lines": len(source_lines),
        "runtime_lyrics_lines": len(runtime_lines),
        "runtime_sung_lines": len(runtime_sung_lines),
        "timed_structure": bool(time_slices),
        "time_slices": time_slices,
        "action": "none" if fitted == raw else f"fit_for_{duration_seconds}s",
    }
    return fitted, meta


def model_to_variant(model_name: str | None) -> str:
    name = (model_name or DEFAULT_LORA_TRAINING_SONG_MODEL).strip()
    aliases = {"auto": "xl_sft", **SONG_MODEL_TO_VARIANT}
    return aliases.get(name, name)


def model_from_variant(variant: str | None, fallback: str | None = None) -> str:
    normalized = model_to_variant(variant)
    if normalized in VARIANT_TO_SONG_MODEL:
        return VARIANT_TO_SONG_MODEL[normalized]
    return normalize_training_song_model(fallback)


def normalize_training_song_model(song_model: str | None) -> str:
    value = (song_model or "").strip()
    if not value or value == "auto":
        return DEFAULT_LORA_TRAINING_SONG_MODEL
    if value in SONG_MODEL_TO_VARIANT:
        return value
    return DEFAULT_LORA_TRAINING_SONG_MODEL


def infer_training_model_from_text(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().lower().replace("_", "-")
    checks = (
        ("xl-sft", "xl_sft"),
        ("xl-base", "xl_base"),
        ("xl-turbo", "xl_turbo"),
        ("sft", "sft"),
        ("base", "base"),
        ("turbo", "turbo"),
    )
    for needle, variant in checks:
        if needle in text:
            return variant, VARIANT_TO_SONG_MODEL[variant]
    return "", ""


def training_inference_defaults(model_or_variant: Any) -> dict[str, Any]:
    variant = model_to_variant(str(model_or_variant or DEFAULT_LORA_TRAINING_SONG_MODEL))
    song_model = model_from_variant(variant, normalize_training_song_model(str(model_or_variant or "")))
    if "turbo" in song_model:
        return {
            "training_shift": DEFAULT_TURBO_SHIFT,
            "num_inference_steps": DEFAULT_TURBO_INFERENCE_STEPS,
            "song_model": song_model,
            "model_variant": variant,
        }
    return {
        "training_shift": DEFAULT_SFT_BASE_SHIFT,
        "num_inference_steps": DEFAULT_SFT_BASE_INFERENCE_STEPS,
        "song_model": song_model,
        "model_variant": variant,
    }


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _instrumental_like_lyrics(value: Any) -> bool:
    text = _normalized_text(value).strip()
    if not text:
        return False
    lowered = text.lower().strip()
    if lowered in {"instrumental", "[instrumental]", "(instrumental)", "instrumental."}:
        return True
    return bool(re.fullmatch(r"\[?\s*instrumental\s*\]?", lowered))


def _trusted_online_vocal_label(entry: dict[str, Any], lyrics: str, status: str, label_source: str) -> bool:
    if label_source not in TRUSTED_ONLINE_LYRICS_SOURCES:
        return False
    if not lyrics or _instrumental_like_lyrics(lyrics):
        return False
    return status not in {"missing", "failed", "error"}


def is_missing_vocal_lyrics(entry: dict[str, Any] | Any) -> bool:
    if isinstance(entry, dict):
        lyrics = str(entry.get("lyrics") or "").strip()
        status = str(entry.get("lyrics_status") or "").strip().lower()
        label_source = str(entry.get("label_source") or entry.get("lyrics_source") or "").strip().lower()
        if _trusted_online_vocal_label(entry, lyrics, status, label_source):
            return False
        if status in {"missing", "failed", "error", "needs_review", "unreviewed"}:
            return True
        if parse_bool(entry.get("requires_review"), False) and status != "verified":
            return True
        if not lyrics or _instrumental_like_lyrics(lyrics):
            return True
        if label_source in MISSING_LYRICS_SOURCES and _instrumental_like_lyrics(lyrics):
            return True
        return False
    lyrics = str(entry or "").strip()
    return not lyrics or _instrumental_like_lyrics(lyrics)


def has_real_vocal_lyrics(entry: dict[str, Any] | Any) -> bool:
    return not is_missing_vocal_lyrics(entry)


def split_missing_vocal_lyrics_labels(labels: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in labels or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        if is_missing_vocal_lyrics(entry):
            removed.append(entry)
        else:
            kept.append(entry)
    return kept, removed


def _metadata_ready(entry: dict[str, Any]) -> bool:
    keyscale = str(entry.get("keyscale") or entry.get("key_scale") or "").strip()
    bpm = entry.get("bpm")
    try:
        bpm_value = float(bpm)
    except (TypeError, ValueError):
        bpm_value = 0.0
    return bpm_value > 0 and bool(keyscale)


def _language_ready(entry: dict[str, Any]) -> bool:
    language = str(entry.get("language") or entry.get("vocal_language") or "").strip().lower()
    return language not in {"", "unknown", "instrumental", "none", "auto"}


def _entry_name(entry: dict[str, Any]) -> str:
    return str(entry.get("filename") or Path(str(entry.get("path") or entry.get("audio_path") or "")).name or "sample")


def vocal_dataset_health(labels: list[dict[str, Any]] | None) -> dict[str, Any]:
    entries = [dict(item) for item in (labels or []) if isinstance(item, dict)]
    total = len(entries)
    missing_lyrics = [entry for entry in entries if is_missing_vocal_lyrics(entry)]
    missing_metadata = [entry for entry in entries if not _metadata_ready(entry)]
    missing_caption = [entry for entry in entries if not str(entry.get("caption") or "").strip()]
    weak_language = [entry for entry in entries if not _language_ready(entry)]
    real_lyrics_count = total - len(missing_lyrics)
    metadata_ready_count = total - len(missing_metadata)
    language_ready_count = total - len(weak_language)
    real_lyrics_ratio = real_lyrics_count / max(total, 1)
    metadata_ratio = metadata_ready_count / max(total, 1)
    language_ratio = language_ready_count / max(total, 1)

    blocking_reasons: list[str] = []
    warnings: list[str] = []
    if total <= 0:
        blocking_reasons.append("Dataset has no samples.")
    if total > 0 and real_lyrics_ratio < VOCAL_MIN_REAL_LYRICS_RATIO:
        blocking_reasons.append(
            f"Only {real_lyrics_count}/{total} samples have real vocal lyrics; vocal LoRA training requires at least 95%."
        )
    if missing_caption:
        blocking_reasons.append(f"{len(missing_caption)}/{total} samples are missing captions.")
    if total > 0 and metadata_ratio < VOCAL_MIN_METADATA_RATIO:
        blocking_reasons.append(f"Only {metadata_ready_count}/{total} samples have BPM and key metadata; at least 90% is required.")
    if total > 0 and language_ratio < VOCAL_MIN_LANGUAGE_RATIO:
        blocking_reasons.append(
            f"Only {language_ready_count}/{total} samples have known vocal language metadata; at least 90% is required."
        )
    if total < 8:
        warnings.append(f"{total} sample(s) is a tiny training set; quality may be unstable.")
    if missing_lyrics:
        warnings.append(
            f"{len(missing_lyrics)}/{total} samples have missing, failed, filename-only, or [Instrumental] lyrics."
        )
    if missing_metadata:
        warnings.append(f"{len(missing_metadata)}/{total} samples are missing BPM or key metadata.")

    def compact(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "filename": _entry_name(entry),
            "path": str(entry.get("path") or entry.get("audio_path") or ""),
            "label_source": str(entry.get("label_source") or entry.get("lyrics_source") or ""),
            "lyrics_status": str(entry.get("lyrics_status") or ""),
            "error": str(entry.get("error") or entry.get("official_error") or ""),
        }

    blocking = bool(blocking_reasons)
    return {
        "sample_count": total,
        "real_lyrics_count": real_lyrics_count,
        "missing_lyrics_count": len(missing_lyrics),
        "caption_ready_count": total - len(missing_caption),
        "missing_caption_count": len(missing_caption),
        "metadata_ready_count": metadata_ready_count,
        "missing_metadata_count": len(missing_metadata),
        "language_ready_count": language_ready_count,
        "weak_language_count": len(weak_language),
        "real_lyrics_ratio": round(real_lyrics_ratio, 4),
        "metadata_ready_ratio": round(metadata_ratio, 4),
        "language_ready_ratio": round(language_ratio, 4),
        "min_real_lyrics_ratio": VOCAL_MIN_REAL_LYRICS_RATIO,
        "min_metadata_ratio": VOCAL_MIN_METADATA_RATIO,
        "min_language_ratio": VOCAL_MIN_LANGUAGE_RATIO,
        "vocal_audition_unreliable": bool(missing_lyrics or blocking),
        "blocking": blocking,
        "can_train_vocal": not blocking,
        "warnings": warnings,
        "blocking_reasons": blocking_reasons,
        "missing_lyrics_files": [compact(entry) for entry in missing_lyrics[:50]],
        "missing_metadata_files": [compact(entry) for entry in missing_metadata[:50]],
        "missing_caption_files": [compact(entry) for entry in missing_caption[:50]],
        "weak_language_files": [compact(entry) for entry in weak_language[:50]],
    }


def is_vocal_training_request(params: dict[str, Any] | None) -> bool:
    params = dict(params or {})
    target = str(params.get("training_target") or params.get("dataset_type") or "").strip().lower()
    if target in {"instrumental", "instrumental_style", "style", "style_only"}:
        return False
    if parse_bool(params.get("instrumental_training"), False) or parse_bool(params.get("all_instrumental"), False):
        return False
    if parse_bool(params.get("style_only"), False):
        return False
    return True


def dataset_block_message(health: dict[str, Any]) -> str:
    reasons = [str(item) for item in list(health.get("blocking_reasons") or []) if str(item).strip()]
    files = [str(item.get("filename") or item.get("path") or "") for item in list(health.get("missing_lyrics_files") or [])[:8] if isinstance(item, dict)]
    message = "Vocal LoRA dataset is not healthy enough to train."
    if reasons:
        message += " " + " ".join(reasons)
    if files:
        message += " Fix lyrics/metadata for: " + ", ".join(files)
    return message


def schedule_mismatch_reasons(params: dict[str, Any] | None) -> list[str]:
    params = dict(params or {})
    variant = model_to_variant(str(params.get("model_variant") or params.get("song_model") or DEFAULT_LORA_TRAINING_SONG_MODEL))
    defaults = training_inference_defaults(variant)
    reasons: list[str] = []
    if params.get("training_shift") not in (None, "") or params.get("shift") not in (None, ""):
        shift = parse_float(params.get("training_shift", params.get("shift")), defaults["training_shift"], 0.1, 10.0)
        if abs(shift - float(defaults["training_shift"])) > 1e-6:
            reasons.append(
                f"{variant} adapter metadata has shift={shift}; expected shift={defaults['training_shift']}."
            )
    if params.get("num_inference_steps") not in (None, ""):
        steps = parse_int(params.get("num_inference_steps"), defaults["num_inference_steps"], 1, 200)
        if int(steps) != int(defaults["num_inference_steps"]):
            reasons.append(
                f"{variant} adapter metadata has {steps} inference steps; expected {defaults['num_inference_steps']}."
            )
    return reasons


def adapter_quality_metadata(metadata: dict[str, Any] | None, *, adapter_type: str = "lora") -> dict[str, Any]:
    meta = dict(metadata or {})
    quality_status = str(meta.get("quality_status") or "").strip().lower() or "unknown"
    reasons = [str(item) for item in list(meta.get("quality_reasons") or []) if str(item).strip()]
    schedule_reasons = schedule_mismatch_reasons(meta)
    if schedule_reasons:
        quality_status = "quarantined"
        reasons.extend(schedule_reasons)
    dataset_warnings = meta.get("dataset_warnings") if isinstance(meta.get("dataset_warnings"), dict) else {}
    if parse_bool(dataset_warnings.get("blocking"), False):
        quality_status = "quarantined"
        reasons.append("Training dataset failed vocal preflight.")
    auditions = list(meta.get("epoch_auditions") or [])
    passed_audition = any(
        isinstance(item, dict)
        and str(item.get("status") or "").lower() == "succeeded"
        and isinstance(item.get("vocal_intelligibility_gate"), dict)
        and str((item.get("vocal_intelligibility_gate") or {}).get("status") or "").lower() in {"pass", "passed"}
        and parse_bool((item.get("vocal_intelligibility_gate") or {}).get("passed"), False)
        for item in auditions
    )
    if not passed_audition and quality_status == "unknown" and adapter_type == "lora":
        quality_status = "needs_review"
        reasons.append("No adapter audition passed the vocal quality gate.")
    if adapter_type != "lora":
        quality_status = "not_generation_loadable"
    return {
        "quality_status": quality_status,
        "quality_reasons": list(dict.fromkeys(reasons)),
        "audition_passed": passed_audition,
    }


def infer_adapter_model_metadata(adapter_path: Path | str) -> dict[str, Any]:
    path = Path(adapter_path).expanduser()
    metadata: dict[str, Any] = {}
    meta_path = path / "acejam_adapter.json"
    if meta_path.is_file():
        try:
            metadata.update(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    if metadata.get("song_model") or metadata.get("model_variant"):
        variant = model_to_variant(str(metadata.get("model_variant") or metadata.get("song_model") or ""))
        song_model = model_from_variant(variant, normalize_training_song_model(str(metadata.get("song_model") or "")))
        return {**metadata, "model_variant": variant, "song_model": song_model}

    config_path = path / "adapter_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        base_model = str(config.get("base_model_name_or_path") or config.get("revision") or "")
        variant, song_model = infer_training_model_from_text(base_model)
        if variant or song_model:
            metadata.update(
                {
                    "model_variant": variant,
                    "song_model": song_model,
                    "base_model_name_or_path": base_model,
                }
            )
    return metadata


def _torch_mps_available() -> bool:
    try:
        import torch

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        return bool(cuda is not None and cuda.is_available())
    except Exception:
        return False


def _apple_silicon_mps_available() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64" and _torch_mps_available()


def _allow_cpu_training_on_apple_silicon() -> bool:
    return parse_bool(os.environ.get("ACEJAM_ALLOW_CPU_TRAINING"), False)


def _allow_mps_fp16_training() -> bool:
    return parse_bool(os.environ.get("ACEJAM_ALLOW_MPS_FP16_TRAINING"), False)


def default_training_device(requested: Any = None) -> str:
    device = str(requested or "auto").strip().lower()
    if device == "metal":
        device = "mps"
    if device and device != "auto":
        if device == "cpu" and _apple_silicon_mps_available() and not _allow_cpu_training_on_apple_silicon():
            raise RuntimeError(
                "CPU LoRA training is blocked on Apple Silicon while MPS is available. "
                "Choose Trainer device auto/mps, or set ACEJAM_ALLOW_CPU_TRAINING=1 for debugging."
            )
        return device
    if _apple_silicon_mps_available():
        return "mps"
    if _torch_cuda_available():
        return "cuda"
    return "cpu"


def training_precision_for_device(device: Any, requested: Any = None) -> str:
    precision = str(requested or "auto").strip().lower()
    aliases = {
        "float32": "fp32",
        "32": "fp32",
        "32-true": "fp32",
        "float16": "fp16",
        "16": "fp16",
        "16-mixed": "fp16",
        "bfloat16": "bf16",
        "bf16-mixed": "bf16",
    }
    precision = aliases.get(precision, precision or "auto")
    device_type = str(device or "").split(":", 1)[0].lower()
    if device_type == "mps" and not _allow_mps_fp16_training():
        return "fp32"
    return precision


def training_device_policy() -> dict[str, Any]:
    apple_mps = _apple_silicon_mps_available()
    cpu_allowed = (not apple_mps) or _allow_cpu_training_on_apple_silicon()
    return {
        "default": default_training_device("auto"),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "mps_available": apple_mps,
        "cuda_available": _torch_cuda_available(),
        "cpu_allowed": cpu_allowed,
        "cpu_blocked": not cpu_allowed,
        "cpu_override_env": "ACEJAM_ALLOW_CPU_TRAINING",
        "default_precision": training_precision_for_device(default_training_device("auto"), "auto"),
        "mps_fp16_allowed": _allow_mps_fp16_training(),
        "mps_fp16_override_env": "ACEJAM_ALLOW_MPS_FP16_TRAINING",
    }


@dataclass
class TrainingJob:
    id: str
    kind: str
    state: str
    created_at: str
    updated_at: str
    command: list[str]
    params: dict[str, Any]
    paths: dict[str, str]
    log_path: str
    pid: int | None = None
    return_code: int | None = None
    error: str = ""
    stage: str = ""
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "command": self.command,
            "params": self.params,
            "paths": self.paths,
            "log_path": self.log_path,
            "pid": self.pid,
            "return_code": self.return_code,
            "error": self.error,
            "stage": self.stage,
            "progress": self.progress,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJob":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            state=str(data["state"]),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            command=[str(part) for part in data.get("command", [])],
            params=dict(data.get("params") or {}),
            paths={str(k): str(v) for k, v in dict(data.get("paths") or {}).items()},
            log_path=str(data.get("log_path") or ""),
            pid=data.get("pid"),
            return_code=data.get("return_code"),
            error=str(data.get("error") or ""),
            stage=str(data.get("stage") or ""),
            progress=parse_float(data.get("progress"), 0.0, 0.0, 100.0),
            result=dict(data.get("result") or {}),
        )


class AceTrainingManager:
    def __init__(
        self,
        *,
        base_dir: Path,
        data_dir: Path,
        model_cache_dir: Path,
        release_models: Callable[[], None] | None = None,
        adapter_ready: Callable[[Path, float], dict[str, Any]] | None = None,
        audition_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        understand_music: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
        write_label_sidecars: Callable[[Path, dict[str, Any]], dict[str, str]] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.model_cache_dir = model_cache_dir
        self.vendor_dir = base_dir / "vendor" / "ACE-Step-1.5"
        self.checkpoint_dir = model_cache_dir / "checkpoints"
        self.datasets_dir = data_dir / "lora_datasets"
        self.tensor_dir = data_dir / "lora_tensors"
        self.training_dir = data_dir / "lora_training"
        self.exports_dir = data_dir / "loras"
        self.imports_dir = data_dir / "lora_imports"
        self.jobs_dir = data_dir / "lora_jobs"
        self.release_models = release_models
        self.adapter_ready = adapter_ready
        self.audition_runner = audition_runner
        self.understand_music = understand_music
        self.write_label_sidecars = write_label_sidecars
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}

        for directory in [
            self.datasets_dir,
            self.tensor_dir,
            self.training_dir,
            self.exports_dir,
            self.imports_dir,
            self.jobs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self._mark_stale_jobs()

    def status(self) -> dict[str, Any]:
        missing = self.missing_dependencies()
        vendor_ready = self.vendor_ready()
        active = self.active_job()
        return {
            "vendor_ready": vendor_ready,
            "vendor_path": str(self.vendor_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "missing_dependencies": missing,
            "ready": vendor_ready and not missing,
            "active_job": active,
            "adapter_count": len(self.list_adapters()),
            "tensorboard_runs": self.tensorboard_runs(),
            "trainer_device_policy": training_device_policy(),
        }

    def vendor_ready(self) -> bool:
        return (self.vendor_dir / "train.py").is_file() and (self.vendor_dir / "acestep" / "training_v2").is_dir()

    def missing_dependencies(self) -> list[str]:
        modules = {
            "lightning": "lightning",
            "lycoris-lora": "lycoris",
            "tensorboard": "tensorboard",
            "toml": "toml",
            "modelscope": "modelscope",
            "typer-slim": "typer",
            "peft": "peft",
            "torchao": "torchao",
        }
        missing = []
        for package, module in modules.items():
            if importlib.util.find_spec(module) is None:
                missing.append(package)
        return missing

    def require_ready(self) -> None:
        if not self.vendor_ready():
            raise RuntimeError(
                "Official ACE-Step trainer is not installed. Run Pinokio Install first so app/vendor/ACE-Step-1.5 is cloned."
            )
        missing = self.missing_dependencies()
        if missing:
            raise RuntimeError(
                "Trainer dependencies are missing: "
                + ", ".join(missing)
                + ". Run Pinokio Install/Update to install training extras."
            )
        # Patch vendor VARIANT_DIR_MAP to include XL models (upstream only has turbo/base/sft)
        try:
            from acestep.training_v2.cli.args import VARIANT_DIR_MAP
            VARIANT_DIR_MAP.setdefault("turbo_shift3", "acestep-v15-turbo-shift3")
            VARIANT_DIR_MAP.setdefault("xl_turbo", "acestep-v15-xl-turbo")
            VARIANT_DIR_MAP.setdefault("xl_base", "acestep-v15-xl-base")
            VARIANT_DIR_MAP.setdefault("xl_sft", "acestep-v15-xl-sft")
        except ImportError:
            pass
        # Patch torchaudio.load to use soundfile backend (avoids torchcodec/FFmpeg dependency)
        try:
            import torchaudio
            _orig_torchaudio_load = torchaudio.load

            def _patched_torchaudio_load(filepath, *args, **kwargs):
                if "backend" not in kwargs:
                    kwargs["backend"] = "soundfile"
                try:
                    return _orig_torchaudio_load(filepath, *args, **kwargs)
                except Exception:
                    kwargs.pop("backend", None)
                    return _orig_torchaudio_load(filepath, *args, **kwargs)

            torchaudio.load = _patched_torchaudio_load
        except ImportError:
            pass

    def active_job(self) -> dict[str, Any] | None:
        with self._lock:
            for job in self._load_jobs_unlocked():
                if job.state in JOB_ACTIVE_STATES:
                    return self._public_job(job)
        return None

    def is_busy(self) -> bool:
        return self.active_job() is not None

    def scan_dataset(self, root: Path) -> dict[str, Any]:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {root}")
        csv_meta = self._load_csv_metadata(root)
        files = []
        for index, audio_path in enumerate(
            path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ):
            files.append(self._sample_from_audio(audio_path, root, csv_meta, index))
        dataset_id = uuid.uuid4().hex[:12]
        scan_path = self.datasets_dir / f"{dataset_id}.scan.json"
        scan_path.write_text(
            json.dumps({"id": dataset_id, "root": str(root), "files": files}, indent=2),
            encoding="utf-8",
        )
        return {"dataset_id": dataset_id, "root": str(root), "files": files, "dataset_path": str(scan_path)}

    def import_root_for(self, dataset_id: str) -> Path:
        return self.imports_dir / slug(dataset_id, "dataset")

    @staticmethod
    def _needs_understand(item: dict[str, Any]) -> bool:
        """True when an audio sample lacks usable lyric/caption/metadata sidecars."""
        if not isinstance(item, dict):
            return False
        path = Path(str(item.get("path") or ""))
        if not path.is_file():
            return False
        stem = path.stem
        lyrics_path = path.with_name(f"{stem}.lyrics.txt")
        legacy_lyrics_path = path.with_suffix(".txt")
        json_path = path.with_name(f"{stem}.json")
        metadata: dict[str, Any] = {}
        if json_path.is_file():
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        existing_lyrics = str(item.get("lyrics") or metadata.get("lyrics") or "").strip()
        if not existing_lyrics and lyrics_path.is_file():
            existing_lyrics = lyrics_path.read_text(encoding="utf-8", errors="replace").strip()
        if not existing_lyrics and legacy_lyrics_path.is_file():
            existing_lyrics = legacy_lyrics_path.read_text(encoding="utf-8", errors="replace").strip()

        if is_missing_vocal_lyrics({**metadata, **item, "lyrics": existing_lyrics}):
            return True
        if not str(item.get("caption") or metadata.get("caption") or "").strip():
            return True
        if not _metadata_ready({**metadata, **item}):
            return True
        return False

    def label_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        trigger_tag: str,
        language: str,
        tag_position: str = "prepend",
        genre_ratio: int | float | str = 0,
    ) -> list[dict[str, Any]]:
        """Create ACE-Step training labels without language detection or LM guessing."""
        trigger = str(trigger_tag or "").strip()
        fixed_language = str(language or "unknown").strip() or "unknown"
        position = str(tag_position or "prepend").strip().lower()
        if position not in {"prepend", "append", "replace"}:
            position = "prepend"
        ratio = parse_int(genre_ratio, 0, 0, 100)
        labeled: list[dict[str, Any]] = []
        for item in entries:
            entry = dict(item or {})
            fallback_caption = self._caption_fallback(entry)
            caption = str(entry.get("caption") or fallback_caption).strip() or fallback_caption
            caption = self._apply_trigger_tag(caption, trigger, position)
            lyrics = str(entry.get("lyrics") or "").strip() or INSTRUMENTAL_SENTINEL
            missing_vocal_lyrics = is_missing_vocal_lyrics({**entry, "lyrics": lyrics})
            entry.update(
                {
                    "caption": caption,
                    "lyrics": lyrics,
                    "lyrics_status": entry.get("lyrics_status") or ("missing" if missing_vocal_lyrics else "present"),
                    "requires_review": parse_bool(entry.get("requires_review"), False) or missing_vocal_lyrics,
                    "language": fixed_language,
                    "custom_tag": trigger,
                    "trigger_tag": trigger,
                    "tag_position": position,
                    "genre_ratio": ratio,
                    "label_source": entry.get("label_source")
                    or ("sidecar_metadata" if entry.get("caption_path") or entry.get("metadata_path") else "deterministic_filename"),
                    "is_instrumental": parse_bool(entry.get("is_instrumental"), _instrumental_like_lyrics(lyrics)),
                    "labeled": True,
                }
            )
            labeled.append(entry)
        return labeled

    def auto_epochs(self, sample_count: int) -> int:
        count = max(0, int(sample_count or 0))
        if count <= 20:
            return 800
        if count <= 100:
            return 500
        return 300

    def _epoch_audition_config(
        self,
        payload: dict[str, Any],
        *,
        trigger_tag: str = "",
        training_seed: int = 42,
    ) -> dict[str, Any]:
        user_caption = str(payload.get("epoch_audition_caption") or "").strip()
        user_lyrics = str(payload.get("epoch_audition_lyrics") or "").strip()
        user_genre = str(payload.get("epoch_audition_genre") or "auto").strip().lower() or "auto"
        enabled = parse_bool(payload.get("epoch_audition_enabled"), True) or bool(user_caption or user_lyrics)
        profile = epoch_audition_genre_profile(user_caption, user_lyrics, user_genre)
        safe_trigger = safe_generation_trigger_tag(trigger_tag)
        profile_metadata = _profile_audition_metadata(profile)
        lyrics, lyrics_meta = default_epoch_audition_lyrics(
            user_caption,
            trigger_tag=safe_trigger,
            lyrics_hint=user_lyrics,
            genre_key=user_genre,
        )
        if not enabled:
            lyrics = ""
        caption = build_epoch_audition_caption(user_caption, trigger_tag=safe_trigger, genre_key=user_genre, genre_profile=profile) if enabled else ""
        vocal_language = str(payload.get("vocal_language") or payload.get("language") or "unknown").strip() or "unknown"
        return {
            "enabled": enabled,
            "caption": caption,
            "lyrics": lyrics,
            "generation_trigger_tag": safe_trigger,
            "user_caption": user_caption,
            "user_lyrics": user_lyrics,
            "lyrics_source": lyrics_meta["lyrics_source"],
            "genre": user_genre,
            "genre_profile": lyrics_meta["genre_profile"],
            "style_profile": lyrics_meta["genre_profile"],
            "style_caption_tags": str(profile.get("caption_tags") or ""),
            "lyrics_section_tags": dict(profile.get("lyrics_section_tags") or {}),
            "duration": EPOCH_AUDITION_DURATION_SECONDS,
            "seed": parse_int(payload.get("epoch_audition_seed"), training_seed, 0, 2**31 - 1),
            "bpm": parse_int(payload.get("epoch_audition_bpm"), profile_metadata["bpm"], 0, 300) or None,
            "keyscale": str(payload.get("epoch_audition_keyscale") or profile_metadata["keyscale"] or "").strip(),
            "timesignature": str(payload.get("epoch_audition_timesignature") or profile_metadata["timesignature"] or "").strip(),
            "scale": parse_float(
                payload.get("epoch_audition_scale"),
                parse_float(payload.get("lora_scale"), DEFAULT_LORA_GENERATION_SCALE, 0.0, 1.0),
                0.0,
                1.0,
            ),
            "vocal_language": vocal_language,
        }

    def _dataset_audition_context(self, labels: list[dict[str, Any]], *, trigger_tag: str) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        _append_caption_parts(parts, safe_generation_trigger_tag(trigger_tag), seen)
        for key in ("genre", "caption"):
            for entry in labels:
                value = str(entry.get(key) or "").strip()
                if not value:
                    continue
                if key == "caption":
                    value = re.sub(r"\b\d+\b", " ", value)
                    value = re.sub(r"\s+", " ", value).strip()
                _append_caption_parts(parts, value, seen)
                if len(parts) >= 8:
                    break
            if len(parts) >= 8:
                break
        return ", ".join(parts[:8])

    def _dataset_training_warnings(self, labels: list[dict[str, Any]]) -> dict[str, Any]:
        return vocal_dataset_health(labels)

    def _apply_one_click_dataset_context(self, params: dict[str, Any], labels: list[dict[str, Any]]) -> dict[str, Any]:
        updated = dict(params)
        warnings = self._dataset_training_warnings(labels)
        updated["dataset_warnings"] = warnings
        audition = dict(updated.get("epoch_audition") or {})
        if audition.get("enabled") and not str(audition.get("user_caption") or "").strip():
            trigger = str(updated.get("trigger_tag") or "").strip()
            safe_trigger = safe_generation_trigger_tag(trigger)
            context_caption = self._dataset_audition_context(labels, trigger_tag=trigger)
            genre = str(audition.get("genre") or "auto")
            profile = epoch_audition_genre_profile(context_caption, str(audition.get("user_lyrics") or ""), genre)
            dataset_metadata = dataset_epoch_audition_metadata(labels)
            lyrics, lyrics_meta = default_epoch_audition_lyrics(
                context_caption,
                trigger_tag=safe_trigger,
                lyrics_hint=str(audition.get("user_lyrics") or ""),
                genre_key=genre,
            )
            audition.update(
                {
                    "caption": build_epoch_audition_caption(context_caption, trigger_tag=safe_trigger, genre_key=genre, genre_profile=profile),
                    "lyrics": lyrics,
                    "generation_trigger_tag": safe_trigger,
                    "lyrics_source": lyrics_meta["lyrics_source"],
                    "genre_profile": lyrics_meta["genre_profile"],
                    "style_profile": lyrics_meta["genre_profile"],
                    "style_caption_tags": str(profile.get("caption_tags") or ""),
                    "lyrics_section_tags": dict(profile.get("lyrics_section_tags") or {}),
                    "dataset_caption_source": "labeled_dataset",
                    "bpm": dataset_metadata["bpm"] or audition.get("bpm"),
                    "keyscale": dataset_metadata["keyscale"] or audition.get("keyscale"),
                    "timesignature": dataset_metadata["timesignature"] or audition.get("timesignature"),
                    "metadata_source": "labeled_dataset",
                }
            )
        updated["epoch_audition"] = audition
        return updated

    def start_one_click_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        trigger_tag = str(payload.get("trigger_tag") or payload.get("custom_tag") or "").strip()
        language = str(payload.get("language") or payload.get("vocal_language") or "").strip()
        if not trigger_tag:
            raise ValueError("trigger_tag is required")
        if not language:
            raise ValueError("language is required")

        dataset_id = slug(str(payload.get("dataset_id") or payload.get("import_id") or uuid.uuid4().hex[:12]), "dataset")
        import_root = Path(str(payload.get("import_root") or "")).expanduser() if payload.get("import_root") else self.import_root_for(dataset_id)
        if not import_root.is_dir():
            raise FileNotFoundError(f"Imported dataset folder not found: {import_root}")

        with self._lock:
            active = next((job for job in self._load_jobs_unlocked() if job.state in JOB_ACTIVE_STATES), None)
            if active:
                raise RuntimeError(f"Training job already active: {active.id}")
            job_id = uuid.uuid4().hex[:12]
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            params = self._one_click_params(payload, dataset_id=dataset_id, import_root=import_root)
            job = TrainingJob(
                id=job_id,
                kind="one_click_train",
                state="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=["acejam-one-click-lora"],
                params=params,
                paths={"import_root": str(import_root)},
                log_path=str(job_dir / "job.log"),
                stage="queued",
                progress=0.0,
            )
            self._write_job_unlocked(job)

        thread = threading.Thread(target=self._run_one_click_job, args=(job,), daemon=True)
        thread.start()
        return self.get_job(job.id)

    def save_dataset(
        self,
        entries: list[dict[str, Any]],
        *,
        dataset_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not entries:
            raise ValueError("No dataset entries supplied")

        dataset_id = slug(dataset_id or f"dataset-{uuid.uuid4().hex[:8]}", "dataset")
        samples = [self._official_sample(entry) for entry in entries]
        samples = [sample for sample in samples if sample.get("audio_path")]
        if not samples:
            raise ValueError("Dataset contains no valid audio paths")

        dataset = {
            "metadata": {
                "tag_position": str((metadata or {}).get("tag_position") or "prepend"),
                "genre_ratio": parse_int((metadata or {}).get("genre_ratio"), 0, 0, 100),
                "custom_tag": str((metadata or {}).get("custom_tag") or ""),
                "language": str((metadata or {}).get("language") or "unknown"),
                "one_click_train": parse_bool((metadata or {}).get("one_click_train"), False),
            },
            "samples": samples,
        }
        dataset_path = self.datasets_dir / f"{dataset_id}.json"
        dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
        return {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "sample_count": len(samples),
            "metadata": dataset["metadata"],
            "samples": samples,
        }

    def start_preprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        dataset_json = self._resolve_dataset_json(payload)
        audio_dir = str(Path(str(payload.get("audio_dir") or payload.get("path") or "")).expanduser()) if payload.get("audio_dir") or payload.get("path") else ""
        if not dataset_json and not audio_dir:
            raise ValueError("Preprocess requires dataset_json, dataset_id, or audio_dir")

        job_id = uuid.uuid4().hex[:12]
        tensor_output = Path(str(payload.get("tensor_output") or "")).expanduser() if payload.get("tensor_output") else self.tensor_dir / job_id
        requested_song_model = normalize_training_song_model(str(payload.get("song_model") or ""))
        variant = model_to_variant(str(payload.get("model_variant") or requested_song_model))
        song_model = model_from_variant(variant, requested_song_model)
        device = default_training_device(payload.get("device"))
        precision = training_precision_for_device(device, payload.get("precision"))
        command = [
            sys.executable,
            "-m",
            "acestep.training_v2.cli.train_fixed",
            "--plain",
            "--yes",
            "--preprocess",
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--model-variant",
            variant,
            "--tensor-output",
            str(tensor_output),
            "--max-duration",
            str(parse_float(payload.get("max_duration"), 240.0, 10.0, 600.0)),
            "--device",
            device,
            "--precision",
            precision,
        ]
        if dataset_json:
            command.extend(["--dataset-json", dataset_json])
        if audio_dir:
            command.extend(["--audio-dir", audio_dir])
        return self._start_job(
            kind="preprocess",
            command=command,
            params={
                "model_variant": variant,
                "song_model": song_model,
                "dataset_json": dataset_json,
                "audio_dir": audio_dir,
                "device": device,
                "precision": precision,
            },
            paths={"tensor_output": str(tensor_output), "dataset_json": dataset_json, "audio_dir": audio_dir},
        )

    def start_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        dataset_dir = Path(str(payload.get("dataset_dir") or payload.get("tensor_dir") or "")).expanduser()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Tensor dataset directory not found: {dataset_dir}")
        job_id = uuid.uuid4().hex[:12]
        adapter_type = str(payload.get("adapter_type") or "lora").lower()
        if adapter_type not in {"lora", "lokr"}:
            raise ValueError("adapter_type must be lora or lokr")
        requested_song_model = normalize_training_song_model(str(payload.get("song_model") or ""))
        variant = model_to_variant(str(payload.get("model_variant") or requested_song_model))
        song_model = model_from_variant(variant, requested_song_model)
        inference_defaults = training_inference_defaults(variant)
        trigger_tag = str(payload.get("trigger_tag") or payload.get("custom_tag") or "").strip()
        epochs = parse_int(payload.get("train_epochs", payload.get("epochs")), 10, 1, 10000)
        training_seed = parse_int(payload.get("training_seed", payload.get("seed")), 42, 0, 2**31 - 1)
        epoch_audition = self._epoch_audition_config(payload, trigger_tag=trigger_tag, training_seed=training_seed)
        device = default_training_device(payload.get("device"))
        precision = training_precision_for_device(device, payload.get("precision"))
        output_name = slug(trigger_tag or adapter_type, "adapter")
        output_dir = Path(str(payload.get("output_dir") or "")).expanduser() if payload.get("output_dir") else self.training_dir / f"{output_name}-{job_id}"
        log_dir = output_dir / "runs"
        command = [
            sys.executable,
            "train.py",
            "--plain",
            "--yes",
            "fixed",
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--model-variant",
            variant,
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--adapter-type",
            adapter_type,
            "--batch-size",
            str(parse_int(payload.get("train_batch_size", payload.get("batch_size")), 1, 1, 64)),
            "--gradient-accumulation",
            str(parse_int(payload.get("gradient_accumulation"), 4, 1, 128)),
            "--epochs",
            str(epochs),
            "--save-every",
            "1",
            "--lr",
            str(parse_float(payload.get("learning_rate"), 1e-4, 1e-7, 1.0)),
            "--shift",
            str(inference_defaults["training_shift"]),
            "--seed",
            str(training_seed),
            "--num-inference-steps",
            str(inference_defaults["num_inference_steps"]),
            "--warmup-steps",
            str(parse_int(payload.get("warmup_steps"), 100, 0, 100000)),
            "--weight-decay",
            str(parse_float(payload.get("weight_decay"), 0.01, 0.0, 1.0)),
            "--max-grad-norm",
            str(parse_float(payload.get("max_grad_norm"), 1.0, 0.0, 100.0)),
            "--optimizer-type",
            str(payload.get("optimizer_type") or "adamw"),
            "--scheduler-type",
            str(payload.get("scheduler_type") or "cosine"),
            "--log-dir",
            str(log_dir),
            "--log-every",
            str(parse_int(payload.get("log_every"), 10, 1, 100000)),
            "--log-heavy-every",
            str(parse_int(payload.get("log_heavy_every"), 50, 1, 100000)),
            "--sample-every-n-epochs",
            str(parse_int(payload.get("sample_every_n_epochs"), 0, 0, 10000)),
            "--device",
            device,
            "--precision",
            precision,
        ]
        if parse_bool(payload.get("offload_encoder"), False):
            command.append("--offload-encoder")
        else:
            command.append("--no-offload-encoder")
        if parse_bool(payload.get("gradient_checkpointing"), True):
            command.append("--gradient-checkpointing")
        else:
            command.append("--no-gradient-checkpointing")

        if adapter_type == "lokr":
            command.extend(
                [
                    "--lokr-linear-dim",
                    str(parse_int(payload.get("lokr_linear_dim"), 64, 1, 256)),
                    "--lokr-linear-alpha",
                    str(parse_int(payload.get("lokr_linear_alpha"), 128, 1, 512)),
                    "--lokr-factor",
                    str(parse_int(payload.get("lokr_factor"), -1, -1, 64)),
                ]
            )
            if parse_bool(payload.get("lokr_decompose_both"), False):
                command.append("--lokr-decompose-both")
            if parse_bool(payload.get("lokr_use_tucker"), False):
                command.append("--lokr-use-tucker")
            if parse_bool(payload.get("lokr_use_scalar"), False):
                command.append("--lokr-use-scalar")
            if parse_bool(payload.get("lokr_weight_decompose"), True):
                command.append("--lokr-weight-decompose")
        else:
            command.extend(
                [
                    "--rank",
                    str(parse_int(payload.get("rank"), 64, 1, 512)),
                    "--alpha",
                    str(parse_int(payload.get("alpha"), 128, 1, 1024)),
                    "--dropout",
                    str(parse_float(payload.get("dropout"), 0.1, 0.0, 1.0)),
                    "--attention-type",
                    str(payload.get("attention_type") or "both"),
                ]
            )

        return self._start_job(
            kind="train",
            command=command,
            params={
                "adapter_type": adapter_type,
                "model_variant": variant,
                "song_model": song_model,
                "trigger_tag": trigger_tag,
                "display_name": trigger_tag,
                "epochs": epochs,
                "save_every_n_epochs": 1,
                "epoch_audition": epoch_audition,
                "training_shift": inference_defaults["training_shift"],
                "num_inference_steps": inference_defaults["num_inference_steps"],
                "lora_scale": parse_float(payload.get("lora_scale"), DEFAULT_LORA_GENERATION_SCALE, 0.0, 1.0),
                "device": device,
                "precision": precision,
            },
            paths={
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "final_adapter": str(output_dir / "final"),
                "log_dir": str(log_dir),
                "audition_dir": str(output_dir / "epoch_auditions"),
            },
        )

    def start_estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        dataset_dir = Path(str(payload.get("dataset_dir") or payload.get("tensor_dir") or "")).expanduser()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Tensor dataset directory not found: {dataset_dir}")
        job_id = uuid.uuid4().hex[:12]
        requested_song_model = normalize_training_song_model(str(payload.get("song_model") or ""))
        variant = model_to_variant(str(payload.get("model_variant") or requested_song_model))
        output = self.training_dir / f"estimate-{job_id}.json"
        device = default_training_device(payload.get("device"))
        precision = training_precision_for_device(device, payload.get("precision"))
        command = [
            sys.executable,
            "train.py",
            "--plain",
            "--yes",
            "estimate",
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--model-variant",
            variant,
            "--dataset-dir",
            str(dataset_dir),
            "--batch-size",
            str(parse_int(payload.get("batch_size"), 1, 1, 64)),
            "--estimate-batches",
            str(parse_int(payload.get("estimate_batches"), 5, 1, 1000)),
            "--top-k",
            str(parse_int(payload.get("top_k"), 16, 1, 256)),
            "--granularity",
            str(payload.get("granularity") or "module"),
            "--output",
            str(output),
            "--device",
            device,
            "--precision",
            precision,
        ]
        return self._start_job(
            kind="estimate",
            command=command,
            params={"model_variant": variant, "device": device, "precision": precision},
            paths={"dataset_dir": str(dataset_dir), "estimate_output": str(output)},
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_job(job) for job in self._load_jobs_unlocked()]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public_job(self._read_job_unlocked(job_id))

    def read_log(self, job_id: str, tail: int = 400) -> dict[str, Any]:
        with self._lock:
            job = self._read_job_unlocked(job_id)
        log_path = Path(job.log_path)
        if not log_path.is_file():
            return {"job_id": job_id, "log": ""}
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"job_id": job_id, "log": "\n".join(lines[-max(1, tail) :])}

    def stop_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._read_job_unlocked(job_id)
            process = self._processes.get(job_id)
            if job.state not in JOB_ACTIVE_STATES:
                return self._public_job(job)
            job.state = "stopping"
            job.updated_at = utc_now()
            self._write_job_unlocked(job)
        if process and process.poll() is None:
            process.terminate()
        return self.get_job(job_id)

    def resume_job(self, job_id: str) -> dict[str, Any]:
        self.require_ready()
        job_id = slug(job_id, "job")
        with self._lock:
            active = next((job for job in self._load_jobs_unlocked() if job.state in JOB_ACTIVE_STATES), None)
            if active:
                raise RuntimeError(f"Stop active training job {active.id} before resuming.")
            job = self._read_job_unlocked(job_id)
            if job.kind not in {"train", "one_click_train"}:
                raise ValueError("Only train and one-click LoRA jobs can be resumed")
            params = dict(job.params or {})
            paths = dict(job.paths or {})
            quarantine_reasons = self._resume_quarantine_reasons(job, params)
            if quarantine_reasons:
                job.state = "failed"
                job.stage = "quarantined"
                job.error = "Unsafe LoRA resume blocked: " + " ".join(quarantine_reasons)
                job.result = {
                    **dict(job.result or {}),
                    "quality_status": "quarantined",
                    "quality_reasons": quarantine_reasons,
                }
                job.updated_at = utc_now()
                self._write_job_unlocked(job)
                return self._public_job(job)

        dataset_dir = Path(str(paths.get("dataset_dir") or paths.get("tensor_output") or "")).expanduser()
        output_dir = Path(str(paths.get("output_dir") or "")).expanduser()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Tensor dataset directory not found for resume: {dataset_dir}")
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Training output directory not found for resume: {output_dir}")
        latest_checkpoint = self._latest_checkpoint(output_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found to resume in {output_dir / 'checkpoints'}")
        latest_epoch = self._checkpoint_epoch(latest_checkpoint)
        if latest_epoch <= 0:
            raise RuntimeError(f"Could not determine checkpoint epoch from {latest_checkpoint}")

        total_epochs = self._resume_total_epochs(job, params, latest_epoch)
        resume_variant = model_to_variant(str(params.get("model_variant") or params.get("song_model") or DEFAULT_LORA_TRAINING_SONG_MODEL))
        resume_defaults = training_inference_defaults(resume_variant)
        params["model_variant"] = resume_defaults["model_variant"]
        params["song_model"] = resume_defaults["song_model"]
        params["training_shift"] = resume_defaults["training_shift"]
        params["num_inference_steps"] = resume_defaults["num_inference_steps"]
        params["device"] = default_training_device("auto")
        params["precision"] = training_precision_for_device(params["device"], params.get("precision"))
        if job.kind == "one_click_train":
            params["train_epochs"] = total_epochs
        else:
            params["epochs"] = total_epochs
        log_dir = Path(str(paths.get("log_dir") or output_dir / "runs")).expanduser()
        command = self._resume_train_command(job, params, dataset_dir, output_dir, log_dir, total_epochs)
        command = self._command_with_arg(command, "--device", params["device"])
        command = self._command_with_arg(command, "--precision", params["precision"])
        command = self._command_with_arg(command, "--save-every", "1")
        command = self._command_with_arg(command, "--scheduler-epochs", str(total_epochs))
        command = self._command_without_arg(command, "--resume-from")

        with self._lock:
            current = self._read_job_unlocked(job_id)
            current.state = "queued"
            current.stage = f"resume from epoch {latest_epoch}"
            current.error = ""
            current.return_code = None
            current.command = command
            current.params = params
            current.paths.update(
                {
                    "dataset_dir": str(dataset_dir),
                    "tensor_output": str(dataset_dir),
                    "output_dir": str(output_dir),
                    "final_adapter": str(output_dir / "final"),
                    "log_dir": str(log_dir),
                    "resume_from": str(latest_checkpoint),
                }
            )
            current.updated_at = utc_now()
            self._write_job_unlocked(current)

        thread = threading.Thread(
            target=self._run_resume_job,
            args=(job_id, int(total_epochs), latest_checkpoint, int(latest_epoch)),
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id)

    def list_adapters(self) -> list[dict[str, Any]]:
        roots = [self.exports_dir, self.training_dir]
        adapters: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for child in sorted(root.rglob("*")):
                if not child.is_dir():
                    continue
                has_lora = (child / "adapter_config.json").is_file() and (
                    (child / "adapter_model.safetensors").is_file() or (child / "adapter_model.bin").is_file()
                )
                has_lokr = (child / "lokr_weights.safetensors").is_file()
                if not (has_lora or has_lokr):
                    continue
                child_key = str(child.resolve())
                if child_key in seen_paths:
                    continue
                seen_paths.add(child_key)
                meta: dict[str, Any] = {}
                meta_path = child / "acejam_adapter.json"
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                inferred_meta = infer_adapter_model_metadata(child)
                for key, value in inferred_meta.items():
                    meta.setdefault(key, value)
                adapter_type = str(meta.get("adapter_type") or ("lokr" if has_lokr and not has_lora else "lora")).strip().lower()
                display_name = str(meta.get("display_name") or meta.get("trigger_tag") or "").strip()
                if not display_name:
                    display_name = child.parent.name if child.name == "final" and root == self.training_dir else child.name
                quality = adapter_quality_metadata(meta, adapter_type=adapter_type)
                meta.setdefault("quality_status", quality["quality_status"])
                meta.setdefault("quality_reasons", quality["quality_reasons"])
                blocked_statuses = {"quarantined", "needs_review", "failed_audition", "not_generation_loadable"}
                quality_status = str(quality.get("quality_status") or "").lower()
                loadable = bool(has_lora and adapter_type == "lora" and quality_status not in blocked_statuses)
                if has_lora and adapter_type == "lora" and not meta_path.is_file():
                    generated_meta = {
                        "display_name": display_name,
                        "trigger_tag": str(meta.get("trigger_tag") or ""),
                        "adapter_type": adapter_type,
                        "model_variant": str(meta.get("model_variant") or ""),
                        "song_model": str(meta.get("song_model") or ""),
                        "trained_at": datetime.fromtimestamp(child.stat().st_mtime, timezone.utc).isoformat(),
                        "source_path": str(child),
                        "registered_path": str(child),
                        "source_paths": {"source": str(child), "registered": str(child)},
                        "quality_status": quality["quality_status"],
                        "quality_reasons": quality["quality_reasons"],
                        "audition_passed": quality["audition_passed"],
                    }
                    if meta.get("base_model_name_or_path"):
                        generated_meta["base_model_name_or_path"] = meta["base_model_name_or_path"]
                    try:
                        meta_path.write_text(json.dumps(generated_meta, indent=2), encoding="utf-8")
                        meta.update(generated_meta)
                    except Exception:
                        pass
                adapters.append(
                    {
                        "name": child.name,
                        "display_name": display_name,
                        "label": display_name,
                        "path": str(child),
                        "adapter_type": adapter_type,
                        "source": "exports" if root == self.exports_dir else "training",
                        "updated_at": datetime.fromtimestamp(child.stat().st_mtime, timezone.utc).isoformat(),
                        "trigger_tag": meta.get("trigger_tag", ""),
                        "language": meta.get("language", ""),
                        "model_variant": meta.get("model_variant", ""),
                        "song_model": meta.get("song_model", ""),
	                        "sample_count": meta.get("sample_count"),
	                        "is_loadable": loadable,
	                        "generation_loadable": loadable,
	                        "quality_status": quality["quality_status"],
	                        "quality_reasons": quality["quality_reasons"],
	                        "audition_passed": quality["audition_passed"],
	                        "metadata": meta,
	                    }
	                )
        return adapters

    def tensorboard_runs(self) -> list[dict[str, str]]:
        runs: list[dict[str, str]] = []
        for run_dir in sorted(self.training_dir.rglob("runs")):
            if not run_dir.is_dir():
                continue
            if not any(path.name.startswith("events.out.tfevents") for path in run_dir.rglob("*") if path.is_file()):
                continue
            runs.append(
                {
                    "path": str(run_dir),
                    "name": run_dir.parent.name,
                    "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat(),
                }
            )
        return runs[:20]

    def _unique_export_target(self, base_name: str) -> Path:
        export_id = slug(base_name, "adapter")
        target = self.exports_dir / export_id
        if not target.exists():
            return target
        suffix = 2
        while True:
            candidate = self.exports_dir / f"{export_id}-{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def _detect_adapter_type(self, path: Path, fallback: str = "lora") -> str:
        if (path / "adapter_config.json").is_file() and (
            (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
        ):
            return "lora"
        if (path / "lokr_weights.safetensors").is_file() or path.name == "lokr_weights.safetensors":
            return "lokr"
        fallback = str(fallback or "lora").strip().lower()
        return fallback if fallback in {"lora", "lokr"} else "lora"

    def register_adapter(
        self,
        source: Path,
        *,
        trigger_tag: str = "",
        display_name: str = "",
        adapter_type: str = "",
        model_variant: str = "",
        song_model: str = "",
        job_id: str = "",
        dataset_id: str = "",
        sample_count: int | None = None,
        epochs: int | None = None,
        language: str = "",
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        source = source.expanduser()
        if not source.exists():
            raise FileNotFoundError(f"LoRA source path not found: {source}")
        source_meta: dict[str, Any] = {}
        source_meta_path = source / "acejam_adapter.json" if source.is_dir() else None
        if source_meta_path and source_meta_path.is_file():
            try:
                source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
            except Exception:
                source_meta = {}
        trigger_tag = str(trigger_tag or source_meta.get("trigger_tag") or "").strip()
        display_name = str(display_name or source_meta.get("display_name") or trigger_tag or name or source.name).strip()
        adapter_type = self._detect_adapter_type(source, adapter_type or str(source_meta.get("adapter_type") or "lora"))
        target = self._unique_export_target(trigger_tag or display_name or name or source.name)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target / source.name)
        merged_metadata = dict(source_meta)
        merged_metadata.update(metadata or {})
        source_paths = dict(merged_metadata.get("source_paths") or {})
        source_paths.update(
            {
                "source": str(source),
                "registered": str(target),
            }
        )
        adapter_metadata = {
            **merged_metadata,
            "display_name": display_name or trigger_tag or target.name,
            "trigger_tag": trigger_tag,
            "adapter_type": adapter_type,
            "model_variant": str(model_variant or merged_metadata.get("model_variant") or ""),
            "song_model": str(song_model or merged_metadata.get("song_model") or ""),
            "job_id": str(job_id or merged_metadata.get("job_id") or ""),
            "trained_at": str(merged_metadata.get("trained_at") or utc_now()),
            "source_paths": source_paths,
            "source_path": str(source),
            "registered_path": str(target),
        }
        if dataset_id:
            adapter_metadata["dataset_id"] = dataset_id
        if sample_count is not None:
            adapter_metadata["sample_count"] = sample_count
        if epochs is not None:
            adapter_metadata["epochs"] = epochs
        if language:
            adapter_metadata["language"] = language
        quality = adapter_quality_metadata(adapter_metadata, adapter_type=adapter_type)
        adapter_metadata["quality_status"] = quality["quality_status"]
        adapter_metadata["quality_reasons"] = quality["quality_reasons"]
        adapter_metadata["audition_passed"] = quality["audition_passed"]
        (target / "acejam_adapter.json").write_text(json.dumps(adapter_metadata, indent=2), encoding="utf-8")
        return {
            "success": True,
            "path": str(target),
            "adapter": target.name,
            "display_name": adapter_metadata["display_name"],
            "trigger_tag": trigger_tag,
            "metadata": adapter_metadata,
        }

    def export_adapter(self, source: Path, name: str | None = None) -> dict[str, Any]:
        return self.register_adapter(source, display_name=str(name or ""), name=name)

    def _one_click_params(self, payload: dict[str, Any], *, dataset_id: str, import_root: Path) -> dict[str, Any]:
        sample_count = parse_int(payload.get("sample_count"), 0, 0, None)
        epochs = payload.get("train_epochs", payload.get("epochs"))
        trigger_tag = str(payload.get("trigger_tag") or payload.get("custom_tag") or "").strip()
        training_seed = parse_int(payload.get("training_seed", payload.get("seed")), 42, 0, 2**31 - 1)
        device = default_training_device(payload.get("device"))
        requested_song_model = normalize_training_song_model(str(payload.get("song_model") or ""))
        variant = model_to_variant(str(payload.get("model_variant") or requested_song_model))
        song_model = model_from_variant(variant, requested_song_model)
        inference_defaults = training_inference_defaults(variant)
        return {
            "dataset_id": dataset_id,
            "import_root": str(import_root),
            "trigger_tag": trigger_tag,
            "language": str(payload.get("language") or payload.get("vocal_language") or "").strip(),
            "adapter_type": str(payload.get("adapter_type") or "lora").strip().lower(),
            "tag_position": str(payload.get("tag_position") or "prepend").strip().lower(),
            "genre_ratio": parse_int(payload.get("genre_ratio"), 0, 0, 100),
            "song_model": song_model,
            "model_variant": variant,
            "train_batch_size": parse_int(payload.get("train_batch_size", payload.get("batch_size")), 1, 1, 64),
            "gradient_accumulation": parse_int(payload.get("gradient_accumulation"), 4, 1, 128),
            "rank": parse_int(payload.get("rank"), 64, 1, 512),
            "alpha": parse_int(payload.get("alpha"), 128, 1, 1024),
            "dropout": parse_float(payload.get("dropout"), 0.1, 0.0, 1.0),
            "training_shift": inference_defaults["training_shift"],
            "num_inference_steps": inference_defaults["num_inference_steps"],
            "training_target": str(payload.get("training_target") or payload.get("dataset_type") or "vocal").strip().lower() or "vocal",
            "instrumental_training": parse_bool(payload.get("instrumental_training"), False),
            "training_seed": training_seed,
            "train_epochs": parse_int(epochs, self.auto_epochs(sample_count), 1, 10000) if epochs not in [None, "", "auto"] else None,
            "save_every_n_epochs": 1,
            "learning_rate": parse_float(payload.get("learning_rate"), 1e-4, 1e-7, 1.0),
            "max_duration": parse_float(payload.get("max_duration"), 240.0, 10.0, 600.0),
            "device": device,
            "precision": training_precision_for_device(device, payload.get("precision")),
            "auto_load": parse_bool(payload.get("auto_load"), True),
            "lora_scale": parse_float(payload.get("lora_scale"), DEFAULT_LORA_GENERATION_SCALE, 0.0, 1.0),
            "use_official_lm_labels": parse_bool(payload.get("use_official_lm_labels"), False),
            "epoch_audition": self._epoch_audition_config(payload, trigger_tag=trigger_tag, training_seed=training_seed),
        }

    def _run_one_click_job(self, job: TrainingJob) -> None:
        log_path = Path(job.log_path)
        try:
            if self.release_models is not None:
                self.release_models()
            params = dict(job.params)
            import_root = Path(params["import_root"]).expanduser()
            dataset_id = str(params["dataset_id"])
            trigger = str(params["trigger_tag"])
            language = str(params["language"])
            adapter_type = str(params.get("adapter_type") or "lora").lower()
            if adapter_type not in {"lora", "lokr"}:
                raise ValueError("adapter_type must be lora or lokr")

            self._set_job_state(job.id, state="running", stage="import", progress=5)
            self._append_log(log_path, f"[import] scanning {import_root}\n")
            scanned = self.scan_dataset(import_root)
            files = list(scanned.get("files") or [])
            if not files:
                raise ValueError("No supported audio files found in the imported dataset")

            # ---- Auto-transcribe via ACE-Step understand_music --------------
            #
            # Per ACE-Step LoRA Training Tutorial, training samples need
            # `<stem>.lyrics.txt` + `<stem>.json` sidecars. Without them
            # `label_entries()` falls back to "[Instrumental]" and the dataset
            # health check warns "vocal/lyric epoch auditions do not validate
            # lyric conditioning". Run understand_music on every audio file
            # that lacks sidecars before labeling, so the official ACE-Step
            # pipeline gets real lyric conditioning.
            auto_understand = parse_bool(params.get("auto_understand_music"), True)
            if auto_understand and self.understand_music and self.write_label_sidecars:
                missing = [item for item in files if self._needs_understand(item)]
                if missing:
                    self._set_job_state(
                        job.id,
                        stage="transcribe",
                        progress=8,
                    )
                    self._append_log(
                        log_path,
                        f"[transcribe] looking up online lyrics for {len(missing)} file(s) without sidecars\n",
                    )
                    request_body = {
                        "vocal_language": language,
                        "language": language,
                        "ace_lm_model": params.get("ace_lm_model") or "auto",
                        "song_model": params.get("song_model"),
                        "lm_backend": params.get("lm_backend"),
                    }
                    transcribed_ok = 0
                    transcribed_failed = 0
                    transcribe_labels: list[dict[str, Any]] = []
                    for idx, item in enumerate(missing):
                        audio_path = Path(str(item.get("path") or ""))
                        if not audio_path.is_file():
                            continue
                        progress = 8 + int(round(8 * idx / max(len(missing), 1)))
                        self._set_job_state(
                            job.id,
                            stage="transcribe",
                            progress=progress,
                            current_file=audio_path.name,
                            transcribe_processed=idx,
                            transcribe_total=len(missing),
                            transcribe_succeeded=transcribed_ok,
                            transcribe_failed=transcribed_failed,
                            transcribe_labels=list(transcribe_labels),
                        )
                        try:
                            understood = self.understand_music(audio_path, request_body)
                            self.write_label_sidecars(audio_path, understood)
                            lyrics_text = str(understood.get("lyrics") or "").strip()
                            missing_lyrics = is_missing_vocal_lyrics({**understood, "lyrics": lyrics_text})
                            lyrics_status = str(understood.get("lyrics_status") or ("missing" if missing_lyrics else "present"))
                            requires_review = parse_bool(understood.get("requires_review"), missing_lyrics)
                            if missing_lyrics:
                                understood.setdefault("lyrics_status", lyrics_status or "missing")
                                understood.setdefault("requires_review", True)
                                transcribed_failed += 1
                            else:
                                transcribed_ok += 1
                            transcribe_labels.append(
                                {
                                    "path": str(audio_path),
                                    "filename": audio_path.name,
                                    "lyrics": lyrics_text,
                                    "lyrics_status": lyrics_status,
                                    "requires_review": requires_review,
                                    "caption": str(understood.get("caption") or ""),
                                    "language": str(understood.get("language") or "unknown"),
                                    "bpm": understood.get("bpm"),
                                    "keyscale": str(understood.get("key_scale") or understood.get("keyscale") or ""),
                                    "label_source": str(understood.get("label_source") or "online_lyrics"),
                                }
                            )
                            self._append_log(
                                log_path,
                                f"[transcribe] {audio_path.name}: {(understood.get('lyrics') or '')[:80]!r}\n",
                            )
                        except Exception as transcribe_exc:
                            transcribed_failed += 1
                            transcribe_labels.append(
                                {
                                    "path": str(audio_path),
                                    "filename": audio_path.name,
                                    "lyrics": "",
                                    "lyrics_status": "missing",
                                    "requires_review": True,
                                    "label_source": "understand_music_failed",
                                    "error": str(transcribe_exc),
                                }
                            )
                            self._append_log(
                                log_path,
                                f"[transcribe] {audio_path.name}: FAILED — {transcribe_exc}\n",
                            )
                    self._set_job_state(
                        job.id,
                        stage="transcribe",
                        progress=16,
                        transcribe_processed=len(missing),
                        transcribe_succeeded=transcribed_ok,
                        transcribe_failed=transcribed_failed,
                        transcribe_labels=list(transcribe_labels),
                        current_file="",
                    )
                    # Re-scan so the freshly written sidecars are picked up
                    scanned = self.scan_dataset(import_root)
                    files = list(scanned.get("files") or [])

            self._set_job_state(job.id, stage="label", progress=16)
            labels = self.label_entries(
                files,
                trigger_tag=trigger,
                language=language,
                tag_position=str(params.get("tag_position") or "prepend"),
                genre_ratio=params.get("genre_ratio", 0),
            )
            excluded_missing_lyrics: list[dict[str, Any]] = []
            if is_vocal_training_request(params):
                labels, excluded_missing_lyrics = split_missing_vocal_lyrics_labels(labels)
                if excluded_missing_lyrics:
                    names = ", ".join(_entry_name(entry) for entry in excluded_missing_lyrics[:12])
                    suffix = "" if len(excluded_missing_lyrics) <= 12 else f", +{len(excluded_missing_lyrics) - 12} more"
                    self._append_log(
                        log_path,
                        f"[dataset] removed {len(excluded_missing_lyrics)} sample(s) without real vocal lyrics after labeling: {names}{suffix}\n",
                    )
                    self._set_job_state(
                        job.id,
                        result={
                            "excluded_missing_lyrics_count": len(excluded_missing_lyrics),
                            "excluded_missing_lyrics_files": [
                                {
                                    "filename": _entry_name(entry),
                                    "path": str(entry.get("path") or entry.get("audio_path") or ""),
                                    "lyrics_status": str(entry.get("lyrics_status") or ""),
                                    "label_source": str(entry.get("label_source") or entry.get("lyrics_source") or ""),
                                }
                                for entry in excluded_missing_lyrics[:50]
                            ],
                        },
                    )
            params = self._apply_one_click_dataset_context(params, labels)
            dataset_warnings = dict(params.get("dataset_warnings") or {})
            if is_vocal_training_request(params) and parse_bool(dataset_warnings.get("blocking"), False):
                self._set_job_state(
                    job.id,
                    state="failed",
                    stage="dataset_blocked",
                    progress=24,
                    result={
                        "sample_count": len(labels),
                        "excluded_missing_lyrics_count": len(excluded_missing_lyrics),
                        "excluded_missing_lyrics_files": [
                            {
                                "filename": _entry_name(entry),
                                "path": str(entry.get("path") or entry.get("audio_path") or ""),
                                "lyrics_status": str(entry.get("lyrics_status") or ""),
                                "label_source": str(entry.get("label_source") or entry.get("lyrics_source") or ""),
                            }
                            for entry in excluded_missing_lyrics[:50]
                        ],
                        "dataset_warnings": dataset_warnings,
                        "labels": labels[:10],
                    },
                )
                raise ValueError(dataset_block_message(dataset_warnings))
            epochs = params.get("train_epochs") or self.auto_epochs(len(labels))
            params["train_epochs"] = epochs

            self._set_job_state(job.id, stage="save_dataset", progress=24)
            saved = self.save_dataset(
                labels,
                dataset_id=dataset_id,
                metadata={
                    "custom_tag": trigger,
                    "tag_position": params.get("tag_position") or "prepend",
                    "genre_ratio": params.get("genre_ratio") or 0,
                    "language": language,
                    "one_click_train": True,
                },
            )
            dataset_json = saved["dataset_path"]
            tensor_output = self.tensor_dir / job.id
            preprocess_command = [
                sys.executable,
                str(self.base_dir / "_acejam_train_bootstrap.py"),
                "--plain",
                "--yes",
                "--preprocess",
                "--checkpoint-dir",
                str(self.checkpoint_dir),
                "--model-variant",
                str(params["model_variant"]),
                "--tensor-output",
                str(tensor_output),
                "--max-duration",
                str(params["max_duration"]),
                "--device",
                str(params["device"]),
                "--precision",
                str(params["precision"]),
                "--dataset-json",
                dataset_json,
                "--audio-dir",
                str(import_root),
            ]
            self._set_job_state(
                job.id,
                stage="preprocess",
                progress=34,
                paths={"import_root": str(import_root), "dataset_json": dataset_json, "tensor_output": str(tensor_output)},
                result={"sample_count": len(labels), "epochs": epochs, "dataset_warnings": params.get("dataset_warnings") or {}},
            )
            self._run_command_step(job.id, preprocess_command, log_path, stage="preprocess")
            tensor_count = len([path for path in tensor_output.glob("*.pt") if not path.name.endswith(".tmp.pt")])
            if tensor_count < len(labels):
                raise RuntimeError(
                    f"Preprocess produced only {tensor_count}/{len(labels)} tensor samples. "
                    "Fix failed samples before training so the LoRA dataset is complete."
                )

            output_dir = self.training_dir / f"{slug(trigger or adapter_type)}-{job.id}"
            log_dir = output_dir / "runs"
            train_command = [
                sys.executable,
                str(self.base_dir / "_acejam_train_bootstrap.py"),
                "--plain",
                "--yes",
                "--checkpoint-dir",
                str(self.checkpoint_dir),
                "--model-variant",
                str(params["model_variant"]),
                "--dataset-dir",
                str(tensor_output),
                "--output-dir",
                str(output_dir),
                "--adapter-type",
                adapter_type,
                "--batch-size",
                str(params["train_batch_size"]),
                "--gradient-accumulation",
                str(params["gradient_accumulation"]),
                "--epochs",
                str(epochs),
                "--save-every",
                "1",
                "--lr",
                str(params["learning_rate"]),
                "--shift",
                str(params["training_shift"]),
                "--seed",
                str(params["training_seed"]),
                "--num-inference-steps",
                str(params["num_inference_steps"]),
                "--warmup-steps",
                "100",
                "--weight-decay",
                "0.01",
                "--max-grad-norm",
                "1.0",
                "--optimizer-type",
                "adamw",
                "--scheduler-type",
                "cosine",
                "--log-dir",
                str(log_dir),
                "--log-every",
                "10",
                "--log-heavy-every",
                "50",
                "--sample-every-n-epochs",
                "0",
                "--device",
                str(params["device"]),
                "--precision",
                str(params["precision"]),
                "--gradient-checkpointing",
                "--no-offload-encoder",
                "--num-workers",
                "0",
            ]
            if adapter_type == "lokr":
                train_command.extend(["--lokr-linear-dim", "64", "--lokr-linear-alpha", "128", "--lokr-factor", "-1", "--lokr-weight-decompose"])
            else:
                train_command.extend(
                    [
                        "--rank",
                        str(params["rank"]),
                        "--alpha",
                        str(params["alpha"]),
                        "--dropout",
                        str(params["dropout"]),
                        "--attention-type",
                        "both",
                    ]
                )
            with self._lock:
                current = self._read_job_unlocked(job.id)
                current.command = list(train_command)
                current.params = dict(params)
                current.updated_at = utc_now()
                self._write_job_unlocked(current)
            self._set_job_state(
                job.id,
                stage="train",
                progress=58,
                paths={
                    "import_root": str(import_root),
                    "dataset_json": dataset_json,
                    "tensor_output": str(tensor_output),
                    "output_dir": str(output_dir),
                    "final_adapter": str(output_dir / "final"),
                    "log_dir": str(log_dir),
                },
                result={"sample_count": len(labels), "epochs": epochs, "dataset_warnings": params.get("dataset_warnings") or {}},
            )
            self._run_train_command_with_epoch_auditions(
                job.id,
                train_command,
                output_dir,
                log_path,
                epochs=int(epochs),
                params=params,
                progress_start=58.0,
                progress_end=88.0,
            )

            final_adapter = output_dir / "final"
            if not final_adapter.exists():
                raise FileNotFoundError(f"Training finished but no final adapter was found at {final_adapter}")

            self._set_job_state(job.id, stage="register", progress=90)
            with self._lock:
                pre_register_result = dict(self._read_job_unlocked(job.id).result or {})
            registered_info = self.register_adapter(
                final_adapter,
                trigger_tag=trigger,
                display_name=trigger,
                adapter_type=adapter_type,
                model_variant=str(params["model_variant"]),
                song_model=str(params["song_model"]),
                job_id=job.id,
                dataset_id=dataset_id,
                sample_count=len(labels),
                epochs=int(epochs),
                language=language,
                metadata={
                    "epoch_audition": params.get("epoch_audition") or {},
                    "epoch_auditions": list(pre_register_result.get("epoch_auditions") or []),
                    "dataset_warnings": params.get("dataset_warnings") or {},
                    "training_shift": params.get("training_shift"),
                    "num_inference_steps": params.get("num_inference_steps"),
                    "lora_scale": params.get("lora_scale"),
                    "source_paths": {
                        "import_root": str(import_root),
                        "dataset_json": dataset_json,
                        "tensor_output": str(tensor_output),
                        "output_dir": str(output_dir),
                        "final_adapter": str(final_adapter),
                    }
                },
            )
            registered = Path(str(registered_info["path"]))

            load_status: dict[str, Any] = {"requested": False}
            if params.get("auto_load") and self.adapter_ready is not None:
                self._set_job_state(job.id, stage="load", progress=96)
                raw_scale = params.get("lora_scale", DEFAULT_LORA_GENERATION_SCALE)
                load_status = self.adapter_ready(
                    registered,
                    float(DEFAULT_LORA_GENERATION_SCALE if raw_scale in (None, "") else raw_scale),
                )

            with self._lock:
                current = self._read_job_unlocked(job.id)
                existing_result = dict(current.result or {})
                current.state = "succeeded"
                current.stage = "complete"
                current.progress = 100.0
                current.return_code = 0
                current.updated_at = utc_now()
                current.result = {
                    "dataset_id": dataset_id,
                    "dataset_json": dataset_json,
                    "tensor_output": str(tensor_output),
                    "output_dir": str(output_dir),
                    "final_adapter": str(final_adapter),
                    "registered_adapter_path": str(registered),
                    "adapter_name": registered_info.get("adapter", registered.name),
                    "display_name": registered_info.get("display_name", trigger),
                    "trigger_tag": registered_info.get("trigger_tag", trigger),
                    "sample_count": len(labels),
                    "epochs": epochs,
                    "auto_load": bool(params.get("auto_load")),
                    "use_lora": bool(load_status.get("success", False)),
                    "load_status": load_status,
                    "labels": labels[:5],
                    "epoch_audition": params.get("epoch_audition") or {},
                    "epoch_auditions": existing_result.get("epoch_auditions", []),
                    "epoch_auditions_skipped_reason": existing_result.get("epoch_auditions_skipped_reason", ""),
                    "dataset_warnings": params.get("dataset_warnings") or {},
                }
                self._write_job_unlocked(current)
            self._append_log(log_path, f"\n[complete] adapter registered at {registered}\n")
        except Exception as exc:
            with self._lock:
                try:
                    current = self._read_job_unlocked(job.id)
                except Exception:
                    current = job
                current.state = "failed"
                current.error = str(exc)
                current.updated_at = utc_now()
                self._write_job_unlocked(current)
            self._append_log(log_path, f"\n[failed] {exc}\n")

    def _run_command_step(self, job_id: str, command: list[str], log_path: Path, *, stage: str) -> None:
        env = self._training_env()
        self._append_log(log_path, f"\n[{stage}] $ {' '.join(command)}\n\n")
        nonfinite_loss_line = ""
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(self.vendor_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._processes[job_id] = process
                current = self._read_job_unlocked(job_id)
                current.pid = process.pid
                current.updated_at = utc_now()
                self._write_job_unlocked(current)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if NONFINITE_TRAINING_LOSS_RE.search(line):
                    nonfinite_loss_line = line.strip()
                    process.terminate()
                    break
            return_code = process.wait()
        with self._lock:
            self._processes.pop(job_id, None)
            current = self._read_job_unlocked(job_id)
            current.pid = None
            current.return_code = return_code
            current.updated_at = utc_now()
            self._write_job_unlocked(current)
        if nonfinite_loss_line:
            raise RuntimeError(f"{stage} produced non-finite loss: {nonfinite_loss_line}")
        if return_code != 0:
            raise RuntimeError(f"{stage} exited with code {return_code}")

    def _epoch_audition_enabled(self, params: dict[str, Any]) -> bool:
        config = params.get("epoch_audition")
        return isinstance(config, dict) and parse_bool(config.get("enabled"), False)

    def _command_with_arg(self, command: list[str], flag: str, value: Any) -> list[str]:
        updated = list(command)
        if flag in updated:
            index = updated.index(flag)
            if index + 1 < len(updated):
                updated[index + 1] = str(value)
            else:
                updated.append(str(value))
        else:
            updated.extend([flag, str(value)])
        return updated

    def _command_without_arg(self, command: list[str], flag: str) -> list[str]:
        updated = list(command)
        while flag in updated:
            index = updated.index(flag)
            del updated[index : min(index + 2, len(updated))]
        return updated

    def _latest_checkpoint_for_epoch(self, output_dir: Path, epoch: int) -> Path | None:
        checkpoints_dir = output_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            return None
        matches = [path for path in checkpoints_dir.glob(f"epoch_{int(epoch)}_*") if path.exists()]
        if not matches:
            matches = [path for path in checkpoints_dir.glob(f"*epoch*{int(epoch)}*") if path.exists()]
        if not matches:
            return None
        return max(matches, key=lambda path: path.stat().st_mtime)

    def _checkpoint_epoch(self, checkpoint_path: Path) -> int:
        import re

        match = re.search(r"(?:^|[_-])epoch[_-](\d+)(?:[_-]|$)", checkpoint_path.name)
        if not match:
            match = re.search(r"epoch[_-]?(\d+)", checkpoint_path.name)
        if not match:
            return 0
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return 0

    def _latest_checkpoint(self, output_dir: Path) -> Path | None:
        checkpoints_dir = output_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            return None
        matches = [path for path in checkpoints_dir.iterdir() if path.exists() and self._checkpoint_epoch(path) > 0]
        if not matches:
            return None
        return max(matches, key=lambda path: (self._checkpoint_epoch(path), path.stat().st_mtime))

    def _resume_total_epochs(self, job: TrainingJob, params: dict[str, Any], latest_epoch: int) -> int:
        existing_result = dict(job.result or {})
        raw = params.get("train_epochs") or params.get("epochs") or existing_result.get("epochs")
        if raw in (None, "", "auto"):
            raise ValueError("Cannot resume because the original total epoch count is unknown")
        total = parse_int(raw, latest_epoch, 1, 10000)
        if total < latest_epoch:
            raise ValueError(f"Cannot resume epoch {latest_epoch}; total epochs is only {total}")
        return total

    def _resume_quarantine_reasons(self, job: TrainingJob, params: dict[str, Any]) -> list[str]:
        reasons = schedule_mismatch_reasons(params)
        dataset_warnings = params.get("dataset_warnings")
        if not isinstance(dataset_warnings, dict):
            dataset_warnings = (job.result or {}).get("dataset_warnings") if isinstance((job.result or {}).get("dataset_warnings"), dict) else {}
        if is_vocal_training_request(params) and parse_bool(dataset_warnings.get("blocking"), False):
            reasons.append("Original dataset failed the vocal LoRA preflight.")
        return list(dict.fromkeys(str(reason) for reason in reasons if str(reason).strip()))

    def _audition_needs_resume_for_epoch(self, job: TrainingJob, epoch: int) -> bool:
        found = False
        for item in list((job.result or {}).get("epoch_auditions") or []):
            if not isinstance(item, dict):
                continue
            try:
                item_epoch = int(item.get("epoch") or -1)
            except (TypeError, ValueError):
                item_epoch = -1
            if item_epoch != int(epoch):
                continue
            found = True
            status = str(item.get("status") or "").lower()
            if status not in {"succeeded", "needs_review"}:
                return True
            if not (str(item.get("audio_url") or "").strip() or str(item.get("result_id") or "").strip()):
                return True
        return not found

    def _mark_interrupted_epoch_auditions(
        self,
        result: dict[str, Any],
        *,
        error: str = "Interrupted by app restart",
    ) -> tuple[dict[str, Any], bool]:
        auditions = list(result.get("epoch_auditions") or [])
        changed = False
        recovered: list[Any] = []
        for item in auditions:
            if not isinstance(item, dict):
                recovered.append(item)
                continue
            record = dict(item)
            if str(record.get("status") or "").lower() == "running":
                record["status"] = "failed"
                record["error"] = str(record.get("error") or error)
                record["updated_at"] = utc_now()
                changed = True
            recovered.append(record)
        if changed:
            updated = dict(result)
            updated["epoch_auditions"] = recovered
            return updated, True
        return result, False

    def _resume_train_command(
        self,
        job: TrainingJob,
        params: dict[str, Any],
        dataset_dir: Path,
        output_dir: Path,
        log_dir: Path,
        total_epochs: int,
    ) -> list[str]:
        existing = list(job.command or [])
        variant = model_to_variant(str(params.get("model_variant") or params.get("song_model") or DEFAULT_LORA_TRAINING_SONG_MODEL))
        inference_defaults = training_inference_defaults(variant)
        if "--dataset-dir" in existing and "--output-dir" in existing:
            command = existing
            command = self._command_with_arg(command, "--dataset-dir", str(dataset_dir))
            command = self._command_with_arg(command, "--output-dir", str(output_dir))
            command = self._command_with_arg(command, "--log-dir", str(log_dir))
            command = self._command_with_arg(command, "--epochs", str(total_epochs))
            command = self._command_with_arg(command, "--model-variant", variant)
            command = self._command_with_arg(
                command,
                "--shift",
                str(inference_defaults["training_shift"]),
            )
            command = self._command_with_arg(
                command,
                "--num-inference-steps",
                str(inference_defaults["num_inference_steps"]),
            )
            command = self._command_with_arg(command, "--device", str(params.get("device") or "auto"))
            command = self._command_with_arg(
                command,
                "--precision",
                training_precision_for_device(params.get("device") or "auto", params.get("precision")),
            )
            return command

        adapter_type = str(params.get("adapter_type") or "lora").lower()
        command = [
            sys.executable,
            str(self.base_dir / "_acejam_train_bootstrap.py"),
            "--plain",
            "--yes",
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--model-variant",
            variant,
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--adapter-type",
            adapter_type,
            "--batch-size",
            str(parse_int(params.get("train_batch_size", params.get("batch_size")), 1, 1, 64)),
            "--gradient-accumulation",
            str(parse_int(params.get("gradient_accumulation"), 4, 1, 128)),
            "--epochs",
            str(total_epochs),
            "--save-every",
            "1",
            "--lr",
            str(parse_float(params.get("learning_rate"), 1e-4, 1e-7, 1.0)),
	            "--shift",
	            str(inference_defaults["training_shift"]),
            "--seed",
            str(parse_int(params.get("training_seed", params.get("seed")), 42, 0, 2**31 - 1)),
	            "--num-inference-steps",
	            str(inference_defaults["num_inference_steps"]),
            "--warmup-steps",
            "100",
            "--weight-decay",
            "0.01",
            "--max-grad-norm",
            "1.0",
            "--optimizer-type",
            "adamw",
            "--scheduler-type",
            "cosine",
            "--log-dir",
            str(log_dir),
            "--log-every",
            "10",
            "--log-heavy-every",
            "50",
            "--sample-every-n-epochs",
            "0",
            "--device",
            str(params.get("device") or "auto"),
            "--precision",
            training_precision_for_device(params.get("device") or "auto", params.get("precision")),
            "--gradient-checkpointing",
            "--no-offload-encoder",
            "--num-workers",
            "0",
        ]
        if adapter_type == "lokr":
            command.extend(["--lokr-linear-dim", "64", "--lokr-linear-alpha", "128", "--lokr-factor", "-1", "--lokr-weight-decompose"])
        else:
            command.extend(
                [
                    "--rank",
                    str(parse_int(params.get("rank"), 64, 1, 512)),
                    "--alpha",
                    str(parse_int(params.get("alpha"), 128, 1, 1024)),
                    "--dropout",
                    str(parse_float(params.get("dropout"), 0.1, 0.0, 1.0)),
                    "--attention-type",
                    str(params.get("attention_type") or "both"),
                ]
            )
        return command

    def _record_epoch_audition(self, job_id: str, audition: dict[str, Any]) -> None:
        def _audition_epoch(item: dict[str, Any]) -> int:
            try:
                return int(item.get("epoch") or -1)
            except (TypeError, ValueError):
                return -1

        with self._lock:
            current = self._read_job_unlocked(job_id)
            result = dict(current.result or {})
            audition_epoch = _audition_epoch(audition)
            auditions = [
                dict(item)
                for item in list(result.get("epoch_auditions") or [])
                if isinstance(item, dict) and _audition_epoch(item) != audition_epoch
            ]
            auditions.append(dict(audition))
            auditions.sort(key=lambda item: _audition_epoch(item))
            result["epoch_auditions"] = auditions
            current.result = result
            current.updated_at = utc_now()
            self._write_job_unlocked(current)

    def _run_epoch_audition(self, job_id: str, params: dict[str, Any], checkpoint_path: Path, epoch: int, log_path: Path) -> None:
        config = dict(params.get("epoch_audition") or {})
        duration = parse_int(config.get("duration"), EPOCH_AUDITION_DURATION_SECONDS, 10, 60)
        audition_lyrics_source = str(config.get("lyrics") or "")
        if str(config.get("style_profile") or config.get("genre_profile") or "").strip():
            styled = apply_audio_style_conditioning(
                {
                    "style_profile": str(config.get("style_profile") or config.get("genre_profile") or ""),
                    "caption": str(config.get("caption") or ""),
                    "lyrics": audition_lyrics_source,
                }
            )
            audition_lyrics_source = str(styled.get("lyrics") or audition_lyrics_source)
        runtime_lyrics, lyrics_fit = fit_epoch_audition_lyrics(audition_lyrics_source, duration=duration)
        vocal_language = (
            str(config.get("vocal_language") or config.get("language") or params.get("vocal_language") or params.get("language") or "unknown").strip()
            or "unknown"
        )
        base_record = {
            "epoch": int(epoch),
            "checkpoint_path": str(checkpoint_path),
            "status": "running",
            "error": "",
            "result_id": "",
            "audio_url": "",
            "created_at": utc_now(),
            "duration": duration,
            "source_lyrics_chars": lyrics_fit["source_lyrics_chars"],
            "runtime_lyrics_chars": lyrics_fit["runtime_lyrics_chars"],
            "lyrics_fit_action": lyrics_fit["action"],
            "lyrics_source": str(config.get("lyrics_source") or "custom"),
            "genre_profile": str(config.get("genre_profile") or ""),
            "style_profile": str(config.get("style_profile") or config.get("genre_profile") or ""),
            "style_caption_tags": str(config.get("style_caption_tags") or ""),
            "lyrics_section_tags": dict(config.get("lyrics_section_tags") or {}),
            "vocal_language": vocal_language,
            "bpm": config.get("bpm"),
            "keyscale": str(config.get("keyscale") or ""),
            "timesignature": str(config.get("timesignature") or ""),
        }
        self._record_epoch_audition(job_id, base_record)
        if self.audition_runner is None:
            skipped = {**base_record, "status": "skipped", "error": "No epoch audition runner is configured"}
            self._record_epoch_audition(job_id, skipped)
            self._append_log(log_path, f"[audition epoch {epoch}] skipped: no audition runner configured\n")
            return

        variant = model_to_variant(str(params.get("model_variant") or params.get("song_model") or DEFAULT_LORA_TRAINING_SONG_MODEL))
        song_model = model_from_variant(variant, normalize_training_song_model(str(params.get("song_model") or "")))
        request = {
            "job_id": job_id,
            "epoch": int(epoch),
            "checkpoint_path": str(checkpoint_path),
            "lora_adapter_name": safe_peft_adapter_name(f"epoch_{int(epoch)}_{checkpoint_path.name}"),
            "caption": str(config.get("caption") or ""),
            "lyrics": runtime_lyrics,
            "duration": duration,
            "seed": parse_int(config.get("seed"), parse_int(params.get("training_seed"), 42, 0, 2**31 - 1), 0, 2**31 - 1),
            "lora_scale": parse_float(
                config.get("scale"),
                parse_float(params.get("lora_scale"), DEFAULT_LORA_GENERATION_SCALE, 0.0, 1.0),
                0.0,
                1.0,
            ),
            "vocal_language": vocal_language,
            "language": vocal_language,
            "lyrics_fit": lyrics_fit,
            "lyrics_source": str(config.get("lyrics_source") or "custom"),
            "genre_profile": str(config.get("genre_profile") or ""),
            "style_profile": str(config.get("style_profile") or config.get("genre_profile") or ""),
            "style_caption_tags": str(config.get("style_caption_tags") or ""),
            "lyrics_section_tags": dict(config.get("lyrics_section_tags") or {}),
            "user_lyrics": str(config.get("user_lyrics") or ""),
            "trigger_tag": str(params.get("trigger_tag") or ""),
            "song_model": song_model,
            "model_variant": variant,
            "adapter_type": str(params.get("adapter_type") or "lora"),
            "bpm": config.get("bpm"),
            "keyscale": str(config.get("keyscale") or ""),
            "timesignature": str(config.get("timesignature") or ""),
        }
        if lyrics_fit["action"] != "none":
            self._append_log(
                log_path,
                (
                    f"[audition epoch {epoch}] fitted lyrics for {duration}s: "
                    f"{lyrics_fit['source_lyrics_chars']}->{lyrics_fit['runtime_lyrics_chars']} chars, "
                    f"{lyrics_fit['source_lyrics_lines']}->{lyrics_fit['runtime_lyrics_lines']} lines\n"
                ),
            )
        try:
            result = self.audition_runner(request)
        except Exception as exc:
            record = {**base_record, "status": "failed", "error": str(exc), "created_at": utc_now()}
            self._record_epoch_audition(job_id, record)
            self._append_log(log_path, f"[audition epoch {epoch}] failed; stopping vocal training: {exc}\n")
            raise

        audios = list(result.get("audios") or []) if isinstance(result, dict) else []
        first_audio = audios[0] if audios else {}
        gate = {}
        if isinstance(result, dict):
            if isinstance(result.get("vocal_intelligibility_gate"), dict):
                gate = dict(result.get("vocal_intelligibility_gate") or {})
            elif isinstance(first_audio.get("vocal_intelligibility_gate"), dict):
                gate = dict(first_audio.get("vocal_intelligibility_gate") or {})
        audio_url = str(first_audio.get("audio_url") or (result or {}).get("audio_url") or "")
        result_id = str((result or {}).get("result_id") or first_audio.get("result_id") or "")
        failure_reason = ""
        review_reason = ""
        gate_status = str(gate.get("status") or "").strip().lower()
        preflight = result.get("lora_preflight") if isinstance(result, dict) and isinstance(result.get("lora_preflight"), dict) else {}
        preflight_status = str(preflight.get("status") or "").strip().lower()
        has_audio = bool(audio_url or result_id)
        if isinstance(result, dict) and result.get("success") is False:
            if gate_status == "needs_review" and has_audio:
                review_reason = str(result.get("error") or gate.get("reason") or gate.get("error") or "vocal gate needs manual review")
            else:
                failure_reason = str(result.get("error") or "audition result was not successful")
        if not gate:
            failure_reason = failure_reason or "audition produced no vocal intelligibility gate"
        elif gate_status == "needs_review" and has_audio:
            review_reason = review_reason or str(
                gate.get("reason")
                or gate.get("error")
                or f"vocal gate status={gate_status}"
            )
        elif gate_status not in {"pass", "passed"} or not parse_bool(gate.get("passed"), False):
            failure_reason = str(gate.get("reason") or gate.get("error") or f"vocal gate status={gate_status or 'unknown'}")
        if preflight_status and preflight_status not in {"passed", "needs_review"}:
            failure_reason = failure_reason or f"LoRA preflight status={preflight_status}"
        elif preflight_status == "needs_review" and has_audio:
            review_reason = review_reason or "LoRA preflight needs manual review"
        if not (audio_url or result_id):
            failure_reason = failure_reason or "audition produced no audio result"
        record_status = "failed" if failure_reason else ("needs_review" if review_reason else "succeeded")
        record = {
            **base_record,
            "status": record_status,
            "error": failure_reason or review_reason,
            "result_id": result_id,
            "audio_url": audio_url,
            "created_at": utc_now(),
            "vocal_intelligibility_gate": gate,
            "transcript_preview": str(gate.get("transcript_preview") or gate.get("transcript") or "")[:500],
            "lora_preflight": preflight,
            "lora_scale": request.get("lora_scale"),
            "song_model": request.get("song_model"),
            "model_variant": request.get("model_variant"),
            "inference_steps": result.get("inference_steps") if isinstance(result, dict) else None,
            "shift": result.get("shift") if isinstance(result, dict) else None,
            "style_profile": request.get("style_profile"),
            "style_caption_tags": request.get("style_caption_tags"),
            "lyrics_section_tags": request.get("lyrics_section_tags"),
            "style_conditioning_audit": result.get("style_conditioning_audit") if isinstance(result, dict) else {},
            "style_lyric_tags_applied": result.get("style_lyric_tags_applied") if isinstance(result, dict) else [],
        }
        self._record_epoch_audition(job_id, record)
        if failure_reason:
            self._append_log(log_path, f"[audition epoch {epoch}] failed vocal quality gate; stopping training: {failure_reason}\n")
            raise RuntimeError(f"Epoch {epoch} audition failed vocal quality gate: {failure_reason}")
        if review_reason:
            self._append_log(log_path, f"[audition epoch {epoch}] needs manual review; training will continue: {review_reason}\n")
            return
        self._append_log(log_path, f"[audition epoch {epoch}] generated {record['audio_url'] or record['result_id']}\n")

    def _run_train_command_with_epoch_auditions(
        self,
        job_id: str,
        train_command: list[str],
        output_dir: Path,
        log_path: Path,
        *,
        epochs: int,
        params: dict[str, Any],
        progress_start: float = 0.0,
        progress_end: float = 95.0,
        start_epoch: int = 1,
        initial_checkpoint: Path | None = None,
    ) -> None:
        command = self._command_with_arg(train_command, "--save-every", "1")
        command = self._command_with_arg(command, "--device", str(params.get("device") or "auto"))
        command = self._command_with_arg(
            command,
            "--precision",
            training_precision_for_device(params.get("device") or "auto", params.get("precision")),
        )
        audition_enabled = self._epoch_audition_enabled(params)
        adapter_type = str(params.get("adapter_type") or "lora").lower()
        if not audition_enabled:
            self._run_command_step(job_id, command, log_path, stage="train")
            return
        if adapter_type != "lora":
            reason = "Epoch auditions are skipped for LoKr because standard ACE-Step generation can only load PEFT LoRA adapters."
            self._append_log(log_path, f"[audition] {reason}\n")
            self._set_job_state(job_id, result={"epoch_auditions_skipped_reason": reason})
            self._run_command_step(job_id, command, log_path, stage="train")
            return

        total_epochs = max(1, int(epochs or 1))
        first_epoch = max(1, int(start_epoch or 1))
        last_checkpoint: Path | None = initial_checkpoint
        for epoch in range(first_epoch, total_epochs + 1):
            before_progress = progress_start + ((epoch - 1) / total_epochs) * (progress_end - progress_start)
            self._set_job_state(job_id, stage=f"train epoch {epoch}/{total_epochs}", progress=before_progress)
            chunk_command = self._command_with_arg(command, "--epochs", str(epoch))
            chunk_command = self._command_with_arg(chunk_command, "--scheduler-epochs", str(total_epochs))
            if last_checkpoint is not None:
                chunk_command = self._command_with_arg(chunk_command, "--resume-from", str(last_checkpoint))
            else:
                chunk_command = self._command_without_arg(chunk_command, "--resume-from")
            self._run_command_step(job_id, chunk_command, log_path, stage=f"train epoch {epoch}/{total_epochs}")
            checkpoint = self._latest_checkpoint_for_epoch(output_dir, epoch)
            if checkpoint is None:
                raise FileNotFoundError(f"Epoch {epoch} finished but no checkpoint was found in {output_dir / 'checkpoints'}")
            last_checkpoint = checkpoint
            audition_progress = progress_start + (epoch / total_epochs) * (progress_end - progress_start)
            self._set_job_state(job_id, stage=f"audition epoch {epoch}/{total_epochs}", progress=audition_progress)
            self._run_epoch_audition(job_id, params, checkpoint, epoch, log_path)
        self._set_job_state(job_id, stage="train", progress=progress_end)

    def _run_resume_job(self, job_id: str, total_epochs: int, latest_checkpoint: Path, latest_epoch: int) -> None:
        try:
            if self.release_models is not None:
                self.release_models()
            with self._lock:
                job = self._read_job_unlocked(job_id)
                job.state = "running"
                job.stage = f"resume from epoch {latest_epoch}"
                job.updated_at = utc_now()
                self._write_job_unlocked(job)
            params = dict(job.params or {})
            output_dir = Path(job.paths.get("output_dir") or "")
            log_path = Path(job.log_path)
            self._append_log(
                log_path,
                f"\n[resume] continuing from {latest_checkpoint} on device {params.get('device') or 'auto'}\n",
            )
            if self._epoch_audition_enabled(params) and self._audition_needs_resume_for_epoch(job, latest_epoch):
                self._append_log(log_path, f"[resume] retrying incomplete audition for epoch {latest_epoch}\n")
                self._set_job_state(job_id, stage=f"audition epoch {latest_epoch}/{total_epochs}", progress=0.0)
                self._run_epoch_audition(job_id, params, latest_checkpoint, latest_epoch, log_path)

            if latest_epoch < total_epochs:
                self._run_train_command_with_epoch_auditions(
                    job_id,
                    job.command,
                    output_dir,
                    log_path,
                    epochs=total_epochs,
                    params=params,
                    progress_start=(latest_epoch / total_epochs) * 95.0,
                    progress_end=95.0,
                    start_epoch=latest_epoch + 1,
                    initial_checkpoint=latest_checkpoint,
                )
            with self._lock:
                current = self._read_job_unlocked(job_id)
                current.return_code = 0
                current.updated_at = utc_now()
                if current.state == "stopping":
                    current.state = "stopped"
                else:
                    current.state = "succeeded"
                    current.stage = "complete"
                    current.progress = 100.0
                    current.result = self._job_result(current)
                self._write_job_unlocked(current)
        except Exception as exc:
            with self._lock:
                try:
                    current = self._read_job_unlocked(job_id)
                except Exception:
                    return
                current.state = "failed"
                current.error = str(exc)
                current.updated_at = utc_now()
                self._write_job_unlocked(current)
            try:
                self._append_log(Path(current.log_path), f"\n[resume failed] {exc}\n")
            except Exception:
                pass

    def _run_train_job_with_epoch_auditions(self, job: TrainingJob) -> None:
        with self._lock:
            job.state = "running"
            job.stage = "train"
            job.updated_at = utc_now()
            self._write_job_unlocked(job)
        params = dict(job.params or {})
        output_dir = Path(job.paths.get("output_dir") or "")
        log_path = Path(job.log_path)
        epochs = parse_int(params.get("epochs"), 1, 1, 10000)
        self._run_train_command_with_epoch_auditions(
            job.id,
            job.command,
            output_dir,
            log_path,
            epochs=epochs,
            params=params,
            progress_start=0.0,
            progress_end=95.0,
        )
        with self._lock:
            current = self._read_job_unlocked(job.id)
            current.return_code = 0
            current.updated_at = utc_now()
            if current.state == "stopping":
                current.state = "stopped"
            else:
                current.state = "succeeded"
                current.stage = "complete"
                current.progress = 100.0
                current.result = self._job_result(current)
            self._write_job_unlocked(current)

    def _training_env(self) -> dict[str, str]:
        env = os.environ.copy()
        py_paths = [
            str(self.vendor_dir),
            str(self.vendor_dir / "acestep" / "third_parts" / "nano-vllm"),
        ]
        if env.get("PYTHONPATH"):
            py_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_paths)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
        env.setdefault("HF_MODULES_CACHE", str(self.model_cache_dir / "hf_modules"))
        env.setdefault("MPLCONFIGDIR", str(self.model_cache_dir / "matplotlib"))
        # Write a training bootstrap script that patches torchaudio + VARIANT_DIR_MAP
        # before running the actual ACE-Step training CLI. This runs as a subprocess.
        # Write bootstrap as a TRACKED file (not generated at runtime) so it works on all machines.
        # The bootstrap resolves paths relative to its own location at runtime.
        return env

    def _set_job_state(
        self,
        job_id: str,
        *,
        state: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        paths: dict[str, str] | None = None,
        result: dict[str, Any] | None = None,
        **extras: Any,
    ) -> None:
        """Update job state. Anything that doesn't match the typed fields lands
        in `current.result` so callers can attach progress detail (current_file,
        transcribe_processed/total/succeeded/failed, transcribe_labels, etc.)
        without growing the dataclass schema for every new metric.
        """
        with self._lock:
            current = self._read_job_unlocked(job_id)
            if state is not None:
                current.state = state
            if stage is not None:
                current.stage = stage
            if progress is not None:
                current.progress = parse_float(progress, 0.0, 0.0, 100.0)
            if paths:
                current.paths.update({str(k): str(v) for k, v in paths.items()})
            if result:
                current.result.update(result)
            for key, value in extras.items():
                if value is None:
                    current.result.pop(key, None)
                else:
                    current.result[key] = value
            current.updated_at = utc_now()
            self._write_job_unlocked(current)

    def _append_log(self, log_path: Path, text: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def _caption_fallback(self, entry: dict[str, Any]) -> str:
        relative = str(entry.get("relative_path") or entry.get("filename") or entry.get("path") or "sample")
        path = Path(relative)
        parts = [part for part in path.with_suffix("").parts if part not in {".", ""}]
        text = " ".join(parts[-3:]) if parts else path.stem
        return text.replace("_", " ").replace("-", " ").strip() or "training sample"

    def _apply_trigger_tag(self, caption: str, trigger_tag: str, tag_position: str) -> str:
        caption = str(caption or "").strip()
        trigger = str(trigger_tag or "").strip()
        if not trigger:
            return caption
        if _caption_contains_trigger_tag(caption, trigger):
            return caption
        if tag_position == "replace":
            return trigger
        if tag_position == "append":
            return f"{caption}, {trigger}" if caption else trigger
        return f"{trigger}, {caption}" if caption else trigger

    def _start_job(
        self,
        *,
        kind: str,
        command: list[str],
        params: dict[str, Any],
        paths: dict[str, str],
    ) -> dict[str, Any]:
        with self._lock:
            active = next((job for job in self._load_jobs_unlocked() if job.state in JOB_ACTIVE_STATES), None)
            if active:
                raise RuntimeError(f"Training job already active: {active.id}")
            job_id = uuid.uuid4().hex[:12]
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            job = TrainingJob(
                id=job_id,
                kind=kind,
                state="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=command,
                params=params,
                paths=paths,
                log_path=str(job_dir / "job.log"),
            )
            self._write_job_unlocked(job)

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return self.get_job(job.id)

    def _run_job(self, job: TrainingJob) -> None:
        try:
            if self.release_models is not None:
                self.release_models()
            if job.kind == "train" and self._epoch_audition_enabled(job.params):
                self._run_train_job_with_epoch_auditions(job)
                return
            env = os.environ.copy()
            py_paths = [
                str(self.vendor_dir),
                str(self.vendor_dir / "acestep" / "third_parts" / "nano-vllm"),
            ]
            if env.get("PYTHONPATH"):
                py_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(py_paths)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
            env.setdefault("HF_MODULES_CACHE", str(self.model_cache_dir / "hf_modules"))
            env.setdefault("MPLCONFIGDIR", str(self.model_cache_dir / "matplotlib"))

            with self._lock:
                job.state = "running"
                job.updated_at = utc_now()
                self._write_job_unlocked(job)

            log_path = Path(job.log_path)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"$ {' '.join(job.command)}\n\n")
                log.flush()
                process = subprocess.Popen(
                    job.command,
                    cwd=str(self.vendor_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self._lock:
                    self._processes[job.id] = process
                    job.pid = process.pid
                    job.updated_at = utc_now()
                    self._write_job_unlocked(job)

                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()

                return_code = process.wait()

            with self._lock:
                self._processes.pop(job.id, None)
                current = self._read_job_unlocked(job.id)
                current.return_code = return_code
                current.updated_at = utc_now()
                if current.state == "stopping":
                    current.state = "stopped"
                elif return_code == 0:
                    current.state = "succeeded"
                    current.result = self._job_result(current)
                else:
                    current.state = "failed"
                    current.error = f"Trainer exited with code {return_code}"
                self._write_job_unlocked(current)
        except Exception as exc:
            with self._lock:
                try:
                    current = self._read_job_unlocked(job.id)
                except Exception:
                    current = job
                current.state = "failed"
                current.error = str(exc)
                current.updated_at = utc_now()
                self._write_job_unlocked(current)

    def _job_result(self, job: TrainingJob) -> dict[str, Any]:
        if job.kind in {"train", "one_click_train"}:
            final_adapter = job.paths.get("final_adapter", "")
            existing_result = dict(job.result or {})
            result = {
                "final_adapter": final_adapter,
                "adapter_exists": Path(final_adapter).exists(),
                "epoch_audition": dict((job.params or {}).get("epoch_audition") or {}),
                "epoch_auditions": list(existing_result.get("epoch_auditions") or []),
            }
            if existing_result.get("epoch_auditions_skipped_reason"):
                result["epoch_auditions_skipped_reason"] = existing_result["epoch_auditions_skipped_reason"]
            if result["adapter_exists"]:
                params = dict(job.params or {})
                registered = self.register_adapter(
                    Path(final_adapter),
                    trigger_tag=str(params.get("trigger_tag") or ""),
                    display_name=str(params.get("display_name") or params.get("trigger_tag") or ""),
                    adapter_type=str(params.get("adapter_type") or "lora"),
                    model_variant=str(params.get("model_variant") or ""),
                    song_model=str(params.get("song_model") or ""),
                    job_id=job.id,
                    epochs=parse_int(params.get("epochs") or params.get("train_epochs") or existing_result.get("epochs"), 0, 0, None) or None,
	                    metadata={
	                        "epoch_audition": params.get("epoch_audition") or {},
	                        "epoch_auditions": list(existing_result.get("epoch_auditions") or []),
	                        "dataset_warnings": params.get("dataset_warnings") or existing_result.get("dataset_warnings") or {},
	                        "training_shift": params.get("training_shift"),
	                        "num_inference_steps": params.get("num_inference_steps"),
                        "lora_scale": params.get("lora_scale"),
                        "source_paths": {
                            "dataset_dir": job.paths.get("dataset_dir", ""),
                            "output_dir": job.paths.get("output_dir", ""),
                            "final_adapter": final_adapter,
                            "log_dir": job.paths.get("log_dir", ""),
                        }
                    },
                )
                result.update(
                    {
                        "registered_adapter_path": registered["path"],
                        "adapter_name": registered["adapter"],
                        "display_name": registered.get("display_name", ""),
                        "trigger_tag": registered.get("trigger_tag", ""),
                    }
                )
            return result
        if job.kind == "preprocess":
            output = Path(job.paths.get("tensor_output", ""))
            tensors = len(list(output.glob("*.pt"))) if output.is_dir() else 0
            return {"tensor_output": str(output), "tensor_count": tensors}
        if job.kind == "estimate":
            output = Path(job.paths.get("estimate_output", ""))
            return {"estimate_output": str(output), "exists": output.is_file()}
        return {}

    def _public_job(self, job: TrainingJob) -> dict[str, Any]:
        data = job.to_dict()
        data["log_url"] = f"/api/lora/jobs/{job.id}/log"
        return data

    def _mark_stale_jobs(self) -> None:
        with self._lock:
            for job in self._load_jobs_unlocked():
                result, auditions_changed = self._mark_interrupted_epoch_auditions(dict(job.result or {}))
                if auditions_changed:
                    job.result = result
                if job.state in JOB_ACTIVE_STATES:
                    job.state = "failed"
                    job.error = "Job was interrupted by an app restart"
                    job.updated_at = utc_now()
                    self._write_job_unlocked(job)
                elif auditions_changed:
                    job.updated_at = utc_now()
                    self._write_job_unlocked(job)

    def _load_jobs_unlocked(self) -> list[TrainingJob]:
        jobs = []
        for job_path in sorted(self.jobs_dir.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                jobs.append(TrainingJob.from_dict(json.loads(job_path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return jobs

    def _read_job_unlocked(self, job_id: str) -> TrainingJob:
        job_id = slug(job_id, "job")
        path = self.jobs_dir / job_id / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"Job not found: {job_id}")
        return TrainingJob.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_job_unlocked(self, job: TrainingJob) -> None:
        job_dir = self.jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def _resolve_dataset_json(self, payload: dict[str, Any]) -> str:
        if payload.get("dataset_json"):
            path = Path(str(payload["dataset_json"])).expanduser()
            if path.is_file():
                return str(path)
        if payload.get("dataset_id"):
            dataset_id = slug(str(payload["dataset_id"]), "dataset")
            path = self.datasets_dir / f"{dataset_id}.json"
            if path.is_file():
                return str(path)
        return ""

    def _load_csv_metadata(self, root: Path) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for csv_path in sorted(root.glob("*.csv")):
            try:
                with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        keys = {key.lower().strip(): key for key in row.keys() if key}
                        filename = row.get(keys.get("file", ""), "") or row.get(keys.get("filename", ""), "")
                        if not filename:
                            continue
                        rows[Path(filename).name] = row
            except Exception:
                continue
        return rows

    def _sample_from_audio(self, audio_path: Path, root: Path, csv_meta: dict[str, dict[str, Any]], index: int) -> dict[str, Any]:
        stem = audio_path.stem
        lyrics_path = audio_path.with_name(f"{stem}.lyrics.txt")
        legacy_lyrics_path = audio_path.with_suffix(".txt")
        caption_path = audio_path.with_name(f"{stem}.caption.txt")
        json_path = audio_path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if json_path.is_file():
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        csv_row = csv_meta.get(audio_path.name, {})
        csv_lyrics = (
            csv_row.get("Lyrics")
            or csv_row.get("lyrics")
            or csv_row.get("Lyric")
            or csv_row.get("lyric")
            or csv_row.get("Formatted Lyrics")
            or csv_row.get("formatted_lyrics")
            or ""
        )
        lyrics = ""
        lyrics_source = "default_instrumental"
        if lyrics_path.is_file():
            lyrics = lyrics_path.read_text(encoding="utf-8", errors="replace").strip()
            lyrics_source = "sidecar_lyrics"
        elif legacy_lyrics_path.is_file():
            lyrics = legacy_lyrics_path.read_text(encoding="utf-8", errors="replace").strip()
            lyrics_source = "sidecar_lyrics"
        elif str(metadata.get("lyrics") or "").strip():
            lyrics = str(metadata.get("lyrics") or "").strip()
            lyrics_source = "metadata_lyrics"
        elif str(csv_lyrics or "").strip():
            lyrics = str(csv_lyrics or "").strip()
            lyrics_source = "csv_lyrics"
        else:
            lyrics = "[Instrumental]"
        caption = ""
        if caption_path.is_file():
            caption = caption_path.read_text(encoding="utf-8", errors="replace").strip()
        caption = caption or str(metadata.get("caption") or csv_row.get("Caption") or stem.replace("_", " ").replace("-", " "))
        label_source = str(metadata.get("label_source") or metadata.get("lyrics_source") or lyrics_source or "").strip().lower()
        lyrics_status = str(metadata.get("lyrics_status") or ("missing" if _instrumental_like_lyrics(lyrics) else "present")).strip().lower()
        requires_review = parse_bool(metadata.get("requires_review"), _instrumental_like_lyrics(lyrics))
        if _trusted_online_vocal_label(metadata, lyrics, lyrics_status, label_source):
            lyrics_status = "verified"
            requires_review = False
        duration = None
        if sf is not None:
            try:
                info = sf.info(str(audio_path))
                duration = round(info.frames / info.samplerate, 3)
            except Exception:
                duration = None
        return {
            "id": f"{index + 1}-{slug(stem)}",
            "filename": audio_path.name,
            "path": str(audio_path),
            "relative_path": str(audio_path.relative_to(root)) if root in audio_path.parents else audio_path.name,
            "lyrics_path": str(lyrics_path if lyrics_path.is_file() else legacy_lyrics_path if legacy_lyrics_path.is_file() else ""),
            "caption_path": str(caption_path if caption_path.is_file() else ""),
            "metadata_path": str(json_path if json_path.is_file() else ""),
            "caption": caption,
            "lyrics": lyrics,
            "lyrics_source": lyrics_source,
            "label_source": label_source,
            "lyrics_status": lyrics_status,
            "requires_review": requires_review,
            "genre": str(metadata.get("genre") or csv_row.get("Genre") or ""),
            "bpm": metadata.get("bpm") or csv_row.get("BPM") or None,
            "keyscale": metadata.get("keyscale") or metadata.get("key_scale") or csv_row.get("Key") or "",
            "timesignature": metadata.get("timesignature") or metadata.get("time_signature") or "4",
            "language": metadata.get("language") or metadata.get("vocal_language") or "unknown",
            "duration": duration or metadata.get("duration") or 0,
            "is_instrumental": parse_bool(metadata.get("is_instrumental"), lyrics.strip().lower() == "[instrumental]"),
            "labeled": bool(caption),
        }

    def _official_sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        audio_path = Path(str(entry.get("audio_path") or entry.get("path") or entry.get("filename") or "")).expanduser()
        if not audio_path.is_file():
            return {}
        lyrics = str(entry.get("lyrics") or "[Instrumental]").strip() or "[Instrumental]"
        missing_lyrics = is_missing_vocal_lyrics({**entry, "lyrics": lyrics})
        return {
            "filename": audio_path.name,
            "audio_path": str(audio_path),
            "caption": str(entry.get("caption") or audio_path.stem.replace("_", " ").replace("-", " ")),
            "lyrics": lyrics,
            "lyrics_status": str(entry.get("lyrics_status") or ("missing" if missing_lyrics else "present")),
            "requires_review": parse_bool(entry.get("requires_review"), missing_lyrics),
            "genre": str(entry.get("genre") or ""),
            "bpm": None if entry.get("bpm") in [None, "", "auto"] else parse_int(entry.get("bpm"), 120, 30, 300),
            "keyscale": str(entry.get("keyscale") or entry.get("key_scale") or ""),
            "timesignature": str(entry.get("timesignature") or entry.get("time_signature") or "4"),
            "language": str(entry.get("language") or entry.get("vocal_language") or "unknown"),
            "duration": parse_float(entry.get("duration"), 0.0, 0.0, None),
            "is_instrumental": parse_bool(entry.get("is_instrumental"), lyrics.lower() == "[instrumental]"),
            "custom_tag": str(entry.get("custom_tag") or ""),
            "prompt_override": entry.get("prompt_override") or None,
        }
