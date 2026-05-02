from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


USER_ALBUM_CONTRACT_VERSION = "user-album-contract-2026-04-26"

TRACK_HEADER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:#{1,6}\s*)?(?:track\s*)?(\d{1,2})\s*[:.\)-]\s*"
    r"(?:[\"“]([^\"”\n]+)[\"”]|'([^'\n]+)'|([^\n(]+?))"
    r"(?:\s*\(([^)\n]+)\))?(?=\n|$)"
)

LABEL_PATTERNS = [
    "album", "album title", "album name", "concept", "language", "track", "bpm", "key", "keyscale", "key scale", "duration",
    "track duration", "style",
    "vibe", "the vibe", "narrative", "the narrative", "verse", "the verse", "lyrics", "explicit lyrics", "required lyrics",
    "naming drop", "naming drop style", "required hook phrase", "required phrase", "required phrases", "hook phrase",
    "produced by", "producer", "prod", "prod.", "engineered by", "engineer", "mixed by", "artist", "performer",
]

UNSAFE_CONTENT_RE = None  # content filtering removed


def _clip(value: Any, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dict(dumped) if isinstance(dumped, dict) else {}
    if isinstance(value, (list, tuple)):
        try:
            converted = dict(value)
            return converted if isinstance(converted, dict) else {}
        except Exception:
            return {}
    return {}


def _list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    return []


def _clean_title(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\((?:produced|prod\.?|engineered|mixed)\s+by\b[^)]*\)\s*$", "", text, flags=re.I)
    return text.strip(" \t-:;\"'“”")


def _track_title_is_contaminated(value: Any) -> bool:
    title = str(value or "").strip()
    if not title:
        return True
    if len(title) > 120:
        return True
    if len(re.findall(r"[A-Za-z0-9À-ÿ']+", title)) > 18:
        return True
    if re.search(r"\[[^\]]+\]", title):
        return True
    if re.search(r"\b(?:style|vibe|lyrics?|verse|chorus|hook|bridge|intro|outro|caption|tags?|bpm|duration)\s*:", title, re.I):
        return True
    return False


def _contract_concept_source(prompt: Any, payload: dict[str, Any]) -> str:
    for value in (payload.get("raw_user_prompt"), payload.get("user_prompt"), payload.get("prompt")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in (prompt, payload.get("concept")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _capture_field(block: str, labels: list[str], *, multiline: bool = True) -> str:
    label_alt = "|".join(re.escape(label) for label in labels)
    if multiline:
        stop_alt = "|".join(re.escape(label) for label in LABEL_PATTERNS)
        pattern = (
            rf"(?ims)^\s*(?:the\s+)?(?:{label_alt})\s*[:=-]\s*(.*?)"
            rf"(?=^\s*(?:{stop_alt})\s*[:=-]|\n\s*(?:track\s*)?\d{{1,2}}\s*[:.\)-]|\Z)"
        )
    else:
        pattern = rf"(?im)^\s*(?:the\s+)?(?:{label_alt})\s*[:=-]\s*([^\n]+)"
    match = re.search(pattern, block)
    return str(match.group(1)).strip() if match else ""


def _capture_inline(block: str, label: str) -> str:
    # Inline metadata often appears inside "(BPM: 95 | Style: boom-bap)".
    # Require a field boundary so "Naming Drop Style:" is not mistaken for
    # the musical Style field.
    match = re.search(rf"(?im)(?:^|[\n(|;])\s*{re.escape(label)}\s*[:=-]\s*([^|\n)]+)", block)
    return str(match.group(1)).strip() if match else ""


def _quoted_phrases(value: str) -> list[str]:
    phrases: list[str] = []
    for match in re.finditer(r"[\"“]([^\"”\n]{2,120})[\"”]|'([^'\n]{2,120})'", value or ""):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _lyric_lines(value: str) -> list[str]:
    lines: list[str] = []
    for line in str(value or "").splitlines():
        cleaned = line.strip().strip("\"“”")
        if (
            not cleaned
            or cleaned in {"[]", "[ ]"}
            or cleaned.lower().startswith(("the explicit", "lyrics", "naming drop"))
        ):
            continue
        if cleaned not in lines:
            lines.append(cleaned)
    return lines[:12]


def _content_policy_status(value: str) -> str:
    return "safe"


def _phrase_norm(value: Any) -> str:
    text = str(value or "").lower()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("…", " ")
        .replace("—", " ")
        .replace("–", " ")
    )
    words = re.findall(r"[A-Za-z0-9À-ÿ\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff']+", text)
    return " ".join(words)


def _phrase_present(phrase: Any, text: Any) -> bool:
    raw_phrase = str(phrase or "").strip()
    raw_text = str(text or "")
    if not raw_phrase:
        return True
    if raw_phrase.lower() in raw_text.lower():
        return True
    normalized = _phrase_norm(raw_phrase)
    return bool(normalized and normalized in _phrase_norm(raw_text))


def _parse_duration_value(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    minutes_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b", text)
    seconds_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", text)
    total = 0.0
    if minutes_match:
        total += float(minutes_match.group(1)) * 60.0
    if seconds_match:
        total += float(seconds_match.group(1))
    if total:
        return int(round(total))
    colon_match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if colon_match:
        return int(colon_match.group(1)) * 60 + int(colon_match.group(2))
    plain = re.search(r"\b(\d{2,3})\b", text)
    if plain:
        return int(plain.group(1))
    return None


def _track_blocks(text: str) -> list[tuple[re.Match[str], str]]:
    matches = list(TRACK_HEADER_RE.finditer(text or ""))
    blocks: list[tuple[re.Match[str], str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match, text[start:end]))
    return blocks


def extract_user_album_contract(
    prompt: Any,
    num_tracks: int | None = None,
    language: str = "en",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    text = _contract_concept_source(prompt, payload) or str(prompt or "")
    album_title = (
        str(payload.get("album_title") or payload.get("album_name") or "").strip()
        or _capture_field(text, ["album title", "album name", "album"], multiline=False)
    )
    concept_source = text
    concept = _capture_field(concept_source, ["concept"], multiline=True) or _clip(concept_source, 900)
    requested_tracks = int(num_tracks or payload.get("num_tracks") or 0)
    tracks: list[dict[str, Any]] = []
    for match, block in _track_blocks(text):
        track_number = int(match.group(1))
        raw_title = match.group(2) or match.group(3) or match.group(4) or ""
        title = _clean_title(raw_title)
        if _track_title_is_contaminated(title):
            continue
        paren = str(match.group(5) or "")
        producer_credit = ""
        if re.search(r"(?i)\b(?:produced\s+by|prod\.?)\b", paren):
            producer_credit = re.sub(r"(?i)^\s*(?:produced\s+by|prod\.?)\s*", "", paren).strip()
        producer_credit = producer_credit or _capture_field(block, ["produced by", "producer", "prod.", "prod"], multiline=False)
        producer_credit = re.sub(r"(?i)^\s*(?:produced\s+by|producer|prod\.?)\s*", "", producer_credit).strip()
        engineer_credit = _capture_field(block, ["engineered by", "engineer", "mixed by"], multiline=False)
        engineer_credit = re.sub(r"(?i)^\s*(engineered|mixed)\s+by\s*", "", engineer_credit).strip()
        bpm_raw = _capture_inline(block, "BPM") or _capture_field(block, ["bpm"], multiline=False)
        bpm_match = re.search(r"\d{2,3}", bpm_raw)
        duration_raw = (
            _capture_inline(block, "Duration")
            or _capture_inline(block, "Track Duration")
            or _capture_field(block, ["track duration", "duration"], multiline=False)
        )
        duration = _parse_duration_value(duration_raw)
        key_scale = (
            _capture_inline(block, "Key Scale")
            or _capture_inline(block, "Keyscale")
            or _capture_inline(block, "Key")
            or _capture_field(block, ["key scale", "keyscale", "key"], multiline=False)
        )
        style = _capture_inline(block, "Style") or _capture_field(block, ["style"], multiline=False)
        vibe = _capture_field(block, ["the vibe", "vibe"], multiline=True)
        narrative = _capture_field(block, ["the narrative", "narrative"], multiline=True)
        lyrics = _capture_field(block, ["explicit lyrics", "required lyrics", "lyrics", "the verse", "verse"], multiline=True)
        naming_drop = _capture_field(block, ["naming drop style", "naming drop"], multiline=True)
        required_phrase_text = _capture_field(
            block,
            ["required hook phrase", "required phrase", "required phrases", "hook phrase"],
            multiline=True,
        )
        required_phrases = _quoted_phrases(naming_drop) + _lyric_lines(lyrics) + _lyric_lines(required_phrase_text)
        tracks.append(
            {
                "track_number": track_number,
                "locked_title": title,
                "source_title": title,
                "producer_credit": _clip(producer_credit, 160),
                "engineer_credit": _clip(engineer_credit, 160),
                "artist_role": _clip(_capture_field(block, ["artist", "performer"], multiline=False), 160),
                "bpm": int(bpm_match.group(0)) if bpm_match else None,
                "duration": duration,
                "key_scale": _clip(key_scale, 80),
                "style": _clip(style, 240),
                "vibe": _clip(vibe, 500),
                "narrative": _clip(narrative, 650),
                "required_lyrics": lyrics,
                "required_phrases": required_phrases,
                "blocked_required_excerpt": "",
                "forbidden_changes": ["title", "track_number", "producer_credit", "duration", "bpm", "key_scale", "style", "vibe", "narrative"],
                "content_policy_status": "safe",
                "source_excerpt": _clip(block, 650),
            }
        )
    if requested_tracks <= 0:
        requested_tracks = len(tracks)
    if requested_tracks:
        tracks = [track for track in tracks if int(track.get("track_number") or 0) <= requested_tracks]
    return {
        "version": USER_ALBUM_CONTRACT_VERSION,
        "album_title": _clean_title(album_title),
        "concept": concept,
        "language": str(language or payload.get("language") or "en"),
        "track_count": requested_tracks or len(tracks),
        "global_rules": [
            "Exact user-provided album title, track titles, order, producer credits, BPM/style/vibe/narrative are locked.",
            "Agents may expand missing details, but must not rename, reorder, translate, or reinterpret locked fields.",
        ],
        "tracks": tracks,
        "applied": bool(album_title or tracks),
        "blocked_unsafe_count": 0,
        "repair_policy": "auto_repair",
    }


def contract_track(contract: dict[str, Any] | None, track_number: int | None = None, index: int | None = None) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None
    target = int(track_number or 0)
    for track in contract.get("tracks") or []:
        if target and int(track.get("track_number") or 0) == target:
            return track
    if index is not None:
        tracks = contract.get("tracks") or []
        if 0 <= index < len(tracks):
            return tracks[index]
    return None


def contract_prompt_context(contract: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    return {
        "version": contract.get("version"),
        "album_title": contract.get("album_title"),
        "track_count": contract.get("track_count"),
        "repair_policy": contract.get("repair_policy"),
        "blocked_unsafe_count": contract.get("blocked_unsafe_count", 0),
        "tracks": [
            {
                "track_number": item.get("track_number"),
                "locked_title": item.get("locked_title"),
                "producer_credit": item.get("producer_credit"),
                "engineer_credit": item.get("engineer_credit"),
                "duration": item.get("duration"),
                "bpm": item.get("bpm"),
                "key_scale": item.get("key_scale"),
                "style": item.get("style"),
                "vibe": item.get("vibe"),
                "narrative": item.get("narrative"),
                "required_phrases": item.get("required_phrases"),
                "content_policy_status": item.get("content_policy_status"),
            }
            for item in contract.get("tracks") or []
        ],
    }


def tracks_from_user_album_contract(contract: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(contract, dict):
        return []
    tracks: list[dict[str, Any]] = []
    for item in contract.get("tracks") or []:
        title = str(item.get("locked_title") or "").strip()
        if not title:
            continue
        tracks.append(
            {
                "track_number": item.get("track_number"),
                "title": title,
                "locked_title": title,
                "source_title": item.get("source_title") or title,
                "producer_credit": item.get("producer_credit") or "",
                "engineer_credit": item.get("engineer_credit") or "",
                "duration": item.get("duration"),
                "bpm": item.get("bpm"),
                "key_scale": item.get("key_scale") or "",
                "style": item.get("style") or "",
                "vibe": item.get("vibe") or "",
                "narrative": item.get("narrative") or "",
                "description": item.get("narrative") or item.get("vibe") or item.get("style") or "",
                "lyrics": item.get("required_lyrics") or "",
                "required_phrases": item.get("required_phrases") or [],
                "content_policy_status": item.get("content_policy_status") or "safe",
                "input_contract_applied": True,
            }
        )
    return tracks


def _replace_phrase_fields(track: dict[str, Any], old_title: str, new_title: str) -> None:
    if not old_title or not new_title or old_title == new_title:
        return
    for field in ["lyrics", "hook_promise", "description", "tool_notes"]:
        value = track.get(field)
        if isinstance(value, str) and old_title in value:
            track[field] = value.replace(old_title, new_title)


def _ensure_required_phrases(track: dict[str, Any], required_phrases: list[str]) -> list[str]:
    lyrics = str(track.get("lyrics") or "")
    missing = [phrase for phrase in required_phrases if phrase and not _phrase_present(phrase, lyrics)]
    if missing:
        addition = "\n".join(missing[:8])
        if lyrics.strip():
            track["lyrics"] = lyrics.rstrip() + "\n\n[Required phrases]\n" + addition
        else:
            track["lyrics"] = "[Required phrases]\n" + addition
    return missing


def _required_lyrics_present(required: str, lyrics: str) -> bool:
    required_text = str(required or "").strip()
    lyric_text = str(lyrics or "")
    if not required_text:
        return True
    if required_text in lyric_text:
        return True
    required_lines = [
        re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", line.strip()).strip()
        for line in required_text.splitlines()
        if line.strip()
    ]
    required_lines = [
        line
        for line in required_lines
        if line and not re.match(r"^(?:lyrics|verse|naming drop|track\s+\d+)\s*:", line, re.I)
    ]
    if not required_lines:
        return False
    return all(_phrase_present(line, lyric_text) for line in required_lines)


def _required_lyrics_is_full_script(required: str) -> bool:
    text = str(required or "").strip()
    if not text:
        return False
    lines = _lyric_lines(text)
    word_count = len(re.findall(r"[A-Za-z0-9À-ÖØ-öø-ÿ'’]+", text))
    section_count = len(re.findall(r"\[[^\]]+\]", text))
    return section_count >= 3 or len(lines) >= 18 or word_count >= 140


def apply_user_album_contract_to_track(
    track: dict[str, Any],
    contract: dict[str, Any] | None,
    index: int = 0,
    logs: list[str] | None = None,
) -> dict[str, Any]:
    result = deepcopy(track or {})
    item = contract_track(contract, result.get("track_number"), index)
    if not item:
        return result
    repaired: list[str] = []
    compliance = _dict_or_empty(result.get("contract_compliance"))
    locked_title = str(item.get("locked_title") or "").strip()
    if locked_title:
        old_title = str(result.get("title") or "").strip()
        if old_title and old_title != locked_title:
            repaired.append("title")
            _replace_phrase_fields(result, old_title, locked_title)
        elif not old_title:
            repaired.append("title")
        result["title"] = locked_title
        result["locked_title"] = locked_title
        result["source_title"] = item.get("source_title") or locked_title
        compliance["title"] = "repaired" if "title" in repaired else "kept"
    for field in ["producer_credit", "engineer_credit", "artist_role", "key_scale", "style", "vibe", "narrative"]:
        value = item.get(field)
        if value not in (None, ""):
            if result.get(field) != value:
                repaired.append(field)
            result[field] = value
            compliance[field] = "repaired" if field in repaired else "kept"
    result["input_contract_key_scale_locked"] = bool(item.get("key_scale"))
    if item.get("bpm"):
        if int(result.get("bpm") or 0) != int(item["bpm"]):
            repaired.append("bpm")
        result["bpm"] = int(item["bpm"])
        compliance["bpm"] = "repaired" if "bpm" in repaired else "kept"
    if item.get("duration"):
        if int(result.get("duration") or 0) != int(item["duration"]):
            repaired.append("duration")
        result["duration"] = int(item["duration"])
        compliance["duration"] = "repaired" if "duration" in repaired else "kept"
    if item.get("narrative"):
        result["description"] = item["narrative"]
    elif item.get("vibe") and not result.get("description"):
        result["description"] = item["vibe"]
    if item.get("required_lyrics"):
        required = str(item.get("required_lyrics") or "").strip()
        if required and _required_lyrics_is_full_script(required) and not _required_lyrics_present(required, str(result.get("lyrics") or "")):
            result["lyrics"] = required + ("\n\n" + str(result.get("lyrics") or "").strip() if str(result.get("lyrics") or "").strip() else "")
            repaired.append("required_lyrics")
        compliance["required_lyrics"] = "repaired" if "required_lyrics" in repaired else "required_phrases_only"
    required_phrases = [
        str(phrase).strip()
        for phrase in _list_or_empty(item.get("required_phrases"))
        if str(phrase).strip() and str(phrase).strip() not in {"[]", "[ ]"}
    ]
    result["required_phrases"] = required_phrases
    missing = _ensure_required_phrases(result, required_phrases)
    if missing:
        repaired.append("required_phrase")
        compliance["required_phrase"] = "repaired"
    result["content_policy_status"] = "safe"
    result["input_contract_applied"] = True
    result["input_contract_version"] = USER_ALBUM_CONTRACT_VERSION
    result["contract_repaired_fields"] = sorted(set(repaired))
    result["contract_compliance"] = compliance
    if logs is not None and repaired:
        logs.append(f"Contract repaired: track {result.get('track_number') or index + 1} {', '.join(sorted(set(repaired)))}")
    return result


def apply_user_album_contract_to_tracks(
    tracks: list[dict[str, Any]],
    contract: dict[str, Any] | None,
    logs: list[str] | None = None,
) -> list[dict[str, Any]]:
    return [apply_user_album_contract_to_track(track, contract, index, logs) for index, track in enumerate(tracks or [])]
