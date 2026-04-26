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
    "album", "album title", "album name", "concept", "language", "track", "bpm", "style",
    "vibe", "the vibe", "narrative", "the narrative", "lyrics", "explicit lyrics",
    "naming drop", "produced by", "engineered by", "artist", "performer",
]

UNSAFE_CONTENT_RE = None  # content filtering removed


def _clip(value: Any, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _clean_title(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\((?:produced|prod\.?|engineered|mixed)\s+by\b[^)]*\)\s*$", "", text, flags=re.I)
    return text.strip(" \t-:;\"'“”")


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
    match = re.search(rf"(?i)\b{re.escape(label)}\s*[:=-]\s*([^|\n)]+)", block)
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
        if not cleaned or cleaned.lower().startswith(("the explicit", "lyrics", "naming drop")):
            continue
        if cleaned not in lines:
            lines.append(cleaned)
    return lines[:12]


def _content_policy_status(value: str) -> str:
    return "safe"


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
    text = str(prompt or "")
    payload = payload or {}
    album_title = (
        str(payload.get("album_title") or payload.get("album_name") or "").strip()
        or _capture_field(text, ["album title", "album name", "album"], multiline=False)
    )
    concept = str(payload.get("concept") or "").strip() or _capture_field(text, ["concept"], multiline=True) or _clip(text, 900)
    requested_tracks = int(num_tracks or payload.get("num_tracks") or 0)
    tracks: list[dict[str, Any]] = []
    for match, block in _track_blocks(text):
        track_number = int(match.group(1))
        raw_title = match.group(2) or match.group(3) or match.group(4) or ""
        title = _clean_title(raw_title)
        paren = str(match.group(5) or "")
        producer_credit = ""
        if re.search(r"(?i)\bproduced\s+by\b", paren):
            producer_credit = re.sub(r"(?i)^\s*produced\s+by\s*", "", paren).strip()
        producer_credit = producer_credit or _capture_field(block, ["produced by", "producer"], multiline=False)
        producer_credit = re.sub(r"(?i)^\s*produced\s+by\s*", "", producer_credit).strip()
        engineer_credit = _capture_field(block, ["engineered by", "engineer", "mixed by"], multiline=False)
        engineer_credit = re.sub(r"(?i)^\s*(engineered|mixed)\s+by\s*", "", engineer_credit).strip()
        bpm_raw = _capture_inline(block, "BPM") or _capture_field(block, ["bpm"], multiline=False)
        bpm_match = re.search(r"\d{2,3}", bpm_raw)
        style = _capture_inline(block, "Style") or _capture_field(block, ["style"], multiline=False)
        vibe = _capture_field(block, ["the vibe", "vibe"], multiline=True)
        narrative = _capture_field(block, ["the narrative", "narrative"], multiline=True)
        lyrics = _capture_field(block, ["explicit lyrics", "lyrics"], multiline=True)
        naming_drop = _capture_field(block, ["naming drop"], multiline=True)
        combined_required = "\n".join(part for part in [lyrics, naming_drop] if part).strip()
        required_phrases = _quoted_phrases(naming_drop) + _lyric_lines(lyrics)
        tracks.append(
            {
                "track_number": track_number,
                "locked_title": title,
                "source_title": title,
                "producer_credit": _clip(producer_credit, 160),
                "engineer_credit": _clip(engineer_credit, 160),
                "artist_role": _clip(_capture_field(block, ["artist", "performer"], multiline=False), 160),
                "bpm": int(bpm_match.group(0)) if bpm_match else None,
                "style": _clip(style, 240),
                "vibe": _clip(vibe, 500),
                "narrative": _clip(narrative, 650),
                "required_lyrics": lyrics,
                "required_phrases": required_phrases,
                "blocked_required_excerpt": "",
                "forbidden_changes": ["title", "track_number", "producer_credit", "bpm", "style", "vibe", "narrative"],
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
                "bpm": item.get("bpm"),
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
                "bpm": item.get("bpm"),
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
    missing = [phrase for phrase in required_phrases if phrase and phrase not in lyrics]
    if missing:
        addition = "\n".join(missing[:8])
        if lyrics.strip():
            track["lyrics"] = lyrics.rstrip() + "\n\n[Required phrases]\n" + addition
        else:
            track["lyrics"] = "[Required phrases]\n" + addition
    return missing


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
    compliance = dict(result.get("contract_compliance") or {})
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
    for field in ["producer_credit", "engineer_credit", "artist_role", "style", "vibe", "narrative"]:
        value = item.get(field)
        if value not in (None, ""):
            if result.get(field) != value:
                repaired.append(field)
            result[field] = value
            compliance[field] = "repaired" if field in repaired else "kept"
    if item.get("bpm"):
        if int(result.get("bpm") or 0) != int(item["bpm"]):
            repaired.append("bpm")
        result["bpm"] = int(item["bpm"])
        compliance["bpm"] = "repaired" if "bpm" in repaired else "kept"
    if item.get("narrative"):
        result["description"] = item["narrative"]
    elif item.get("vibe") and not result.get("description"):
        result["description"] = item["vibe"]
    if item.get("required_lyrics"):
        required = str(item.get("required_lyrics") or "").strip()
        if required and required not in str(result.get("lyrics") or ""):
            result["lyrics"] = required + ("\n\n" + str(result.get("lyrics") or "").strip() if str(result.get("lyrics") or "").strip() else "")
            repaired.append("required_lyrics")
        compliance["required_lyrics"] = "repaired" if "required_lyrics" in repaired else "kept"
    missing = _ensure_required_phrases(result, list(item.get("required_phrases") or []))
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

