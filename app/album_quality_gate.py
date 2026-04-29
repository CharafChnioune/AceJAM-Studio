from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from studio_core import ACE_STEP_CAPTION_CHAR_LIMIT, ACE_STEP_LYRICS_CHAR_LIMIT, has_vocal_lyrics


ALBUM_PAYLOAD_GATE_VERSION = "album-payload-quality-gate-2026-04-29"

TAG_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "genre_style": (
        "pop", "rap", "hip-hop", "trap", "drill", "r&b", "soul", "rock", "metal",
        "punk", "house", "techno", "trance", "dancehall", "reggaeton", "afro",
        "amapiano", "cinematic", "ambient", "schlager", "latin", "j-pop", "k-pop",
    ),
    "rhythm_groove": (
        "groove", "bounce", "swing", "boom-bap", "four-on-the-floor", "breakbeat",
        "drums", "hi-hat", "hi hats", "snare", "kick", "percussion", "shuffle",
        "rhythm", "syncopation", "tempo", "groovy", "rubber bass", "handclap", "handclaps",
    ),
    "instrumentation": (
        "piano", "guitar", "bass", "808", "sub-bass", "synth", "strings", "brass",
        "drums", "accordion", "organ", "rhodes", "pad", "lead", "horn", "violin",
        "cello", "sax", "trumpet",
    ),
    "vocal_style": (
        "vocal", "voice", "rap", "sung", "singer", "male", "female", "duet",
        "choir", "harmony", "harmonies", "ad-lib", "hook", "chorus", "chant",
    ),
    "mood_atmosphere": (
        "dark", "bright", "uplifting", "melancholic", "euphoric", "dreamy",
        "nostalgic", "aggressive", "romantic", "hopeful", "tense", "warm", "cold",
        "cinematic", "gritty", "emotional", "intimate", "confident",
    ),
    "arrangement_energy": (
        "chorus", "hook", "build", "drop", "bridge", "anthemic", "breakdown",
        "dynamic", "riser", "climax", "intro", "outro", "final", "stadium",
    ),
    "mix_production": (
        "mix", "master", "polished", "crisp", "clean", "wide stereo", "high-fidelity",
        "radio", "studio", "glossy", "punchy", "warm analog", "lo-fi",
    ),
}

DIMENSION_REPAIR_TERMS: dict[str, str] = {
    "genre_style": "modern pop",
    "rhythm_groove": "steady groove",
    "instrumentation": "layered drums and melodic instruments",
    "vocal_style": "clear lead vocal",
    "mood_atmosphere": "emotional atmosphere",
    "arrangement_energy": "dynamic hook arrangement",
    "mix_production": "polished studio mix",
}

CAPTION_LEAK_PATTERNS = [
    re.compile(r"\[[^\]]*(?:verse|chorus|hook|bridge|intro|outro)[^\]]*\]", re.I),
    re.compile(r"\b(?:verse|lyrics|naming drop|track\s+\d+|album|bpm|keyscale|key|duration|metadata|produced by|prod\.|artist|description|tags)\s*:", re.I),
    re.compile(r"\b(?:return strict json|json object|tool context|deterministic acejam|lyric target|crewai)\b", re.I),
    re.compile(r"[{}]|\b(?:true|false|null)\b|['\"][A-Za-z0-9_ -]{1,40}['\"]\s*:", re.I),
]

META_LEAK_LINE_RE = re.compile(
    r"^\s*(?:[-*>_`#\s]+)?(?:"
    r"\[(?:producer credit|locked title|duration|bpm|key|artist|title|metadata|tags|song model|quality profile)[^\]]*\]|"
    r"thought:|reasoning:|analysis:|final answer:|track metadata:|artist:|"
    r"description:|tags:|duration:|bpm:|key(?:[_\s-]?scale)?:|time(?:[_\s-]?signature)?:|metadata:|"
    r"\[?ace[-\s]?step(?:\s+metadata)?\]?:?|ace[-\s]?step metadata:|"
    r"tag(?:[_\s-]?list)?:|start:|end:|vocal(?:[_\s-]?role)?:|"
    r"song(?:[_\s-]?model)?:|quality(?:[_\s-]?profile)?:|"
    r"model(?:[_\s-]?advice)?:|seed:|inference(?:[_\s-]?steps)?:|guidance(?:[_\s-]?scale)?:|"
    r"shift:|audio(?:[_\s-]?format)?:|i will now|the complete production spec|return strict json|```)",
    re.I,
)
FALLBACK_ARTIFACT_RE = re.compile(
    r"\b(?:morning finds the|light is leaning through the door|kept the receipt from the life before|"
    r"now i want the sound and nothing more|the you|the was|the are|the is)\b",
    re.I,
)
PLACEHOLDER_RE = re.compile(r"\b(?:placeholder|repeat chorus|same as before|continue|tbd|todo|\.\.\.)\b", re.I)
SECTION_RE = re.compile(r"\[([^\]]+)\]")
WORD_RE = re.compile(r"[A-Za-z0-9À-ÿ\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff']+")


def _safe_job_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "manual").strip())
    return (text or "manual")[:80]


class AlbumRunDebugLogger:
    """Write local, job-scoped album debug artifacts without terminal prompt dumps."""

    def __init__(self, data_dir: Path | str, job_id: Any):
        self.root = Path(data_dir) / "debug" / "album_runs" / _safe_job_id(job_id)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def write_json(self, name: str, payload: Any) -> str:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
        return str(path)

    def append_jsonl(self, name: str, payload: Any) -> str:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(_jsonable(payload) if isinstance(payload, dict) else {"payload": _jsonable(payload)}),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _words(text: str) -> list[str]:
    return WORD_RE.findall(re.sub(r"\[[^\]]+\]", " ", str(text or "")))


def _section_key(section: str) -> str:
    text = re.sub(r"[*_`]", "", str(section or ""))
    text = re.sub(r"[\[\]]", "", text).lower()
    text = re.sub(r"\s*-\s*.*$", "", text)
    text = re.sub(r"\s+\d+$", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _normalize_lyric_section_line(line: str) -> str:
    """Strip markdown emphasis around bare section tags before lyric analysis."""
    stripped = str(line or "").strip()
    match = re.fullmatch(r"[*_`~\s]*\[\s*([^\]]+?)\s*\][*_`~\s]*", stripped)
    if not match:
        return str(line or "")
    return f"[{match.group(1).strip()}]"


def _metadata_probe(line: str) -> str:
    text = str(line or "").strip()
    text = re.sub(r"^[\s>*_`#-]+", "", text)
    text = re.sub(r"[\s*_`#-]+$", "", text)
    text = text.replace("**", "")
    return text.strip()


def _split_terms(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[,;\n|]+", str(value or ""))
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        term = re.sub(r"\s+", " ", str(item or "")).strip(" .")
        if not term:
            continue
        key = term.lower()
        if key not in seen:
            terms.append(term)
            seen.add(key)
    return terms


def _lyric_stats(lyrics: str) -> dict[str, Any]:
    raw_lines = [_normalize_lyric_section_line(line).strip() for line in str(lyrics or "").splitlines() if line.strip()]
    lyric_lines = [re.sub(r"\[[^\]]+\]", "", line).strip() for line in raw_lines]
    lyric_lines = [line for line in lyric_lines if line]
    words = _words("\n".join(lyric_lines))
    sections = SECTION_RE.findall(str(lyrics or ""))
    section_keys = [_section_key(section) for section in sections]
    repeated = [
        line
        for line, count in Counter(line.lower() for line in lyric_lines).items()
        if count > 1 and len(line) > 8
    ]
    unique_ratio = round(len(set(line.lower() for line in lyric_lines)) / max(1, len(lyric_lines)), 2)
    return {
        "word_count": len(words),
        "line_count": len(lyric_lines),
        "section_count": len(sections),
        "sections": [f"[{item}]" for item in sections],
        "char_count": len(str(lyrics or "")),
        "hook_count": sum(1 for key in section_keys if any(token in key for token in ("chorus", "hook", "refrain"))),
        "repeated_lines": repeated,
        "unique_line_ratio": unique_ratio,
        "meta_leak_lines": [line for line in raw_lines if META_LEAK_LINE_RE.search(_metadata_probe(line))],
        "fallback_artifact_count": len(FALLBACK_ARTIFACT_RE.findall(str(lyrics or ""))),
        "placeholder_count": len(PLACEHOLDER_RE.findall(str(lyrics or ""))),
    }


def _section_only_line(line: str) -> bool:
    return bool(re.fullmatch(r"[*_`~\s]*\[\s*[^\]]+?\s*\][*_`~\s]*", str(line or "").strip()))


def _line_word_chunks(line: str, target_words: int = 6) -> list[str]:
    """Split an overlong sung/rap line into short ACE-Step-friendly lines."""
    text = str(line or "").strip()
    if not text or _section_only_line(text):
        return [text] if text else []
    tag_prefix = ""
    tag_match = re.match(r"^(\[[^\]]+\])\s*(.+)$", text)
    if tag_match:
        tag_prefix = tag_match.group(1)
        text = tag_match.group(2).strip()
    pieces = [
        piece.strip(" ,;:.!?-")
        for piece in re.split(
            r"\s*(?:,|;|:|—|–| - |\band\b|\bbut\b|\bwhile\b|\bwhen\b|\bwhere\b|\bso\b|\bcause\b)\s+",
            text,
            flags=re.I,
        )
        if piece.strip(" ,;:.!?-")
    ]
    words = _words(text)
    if len(pieces) <= 1 and len(words) > target_words + 2:
        pieces = [" ".join(words[index : index + target_words]) for index in range(0, len(words), target_words)]
    if len(pieces) > 1 and all(len(_words(piece)) >= 2 for piece in pieces):
        return ([tag_prefix] if tag_prefix else []) + pieces
    segments: list[str] = []
    current: list[str] = []
    for piece in pieces:
        piece_words = _words(piece)
        if not piece_words:
            continue
        if current and len(current) + len(piece_words) > target_words + 2:
            segments.append(" ".join(current))
            current = piece_words
        else:
            current.extend(piece_words)
    if current:
        segments.append(" ".join(current))
    if len(segments) <= 1:
        return [line]
    if tag_prefix:
        return [tag_prefix] + segments
    return segments


def _reflow_lyrics_to_min_lines(lyrics: str, min_lines: int, max_chars: int = ACE_STEP_LYRICS_CHAR_LIMIT) -> tuple[str, bool]:
    """Increase line count by splitting long lines, without adding filler words."""
    text = str(lyrics or "").strip()
    stats = _lyric_stats(text)
    target = int(min_lines or 0)
    if not text or stats["line_count"] >= target:
        return text, False
    deficit = target - stats["line_count"]
    output: list[str] = []
    changed = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if deficit <= 0 or _section_only_line(line) or not line.strip():
            output.append(line)
            continue
        words = _words(line)
        if len(words) < 8 and len(line) < 48:
            output.append(line)
            continue
        chunks = _line_word_chunks(line, target_words=6)
        if len(chunks) > 1:
            keep = min(len(chunks), deficit + 1)
            output.extend(chunks[:keep])
            if keep < len(chunks):
                output.append(" ".join(chunks[keep:]))
            deficit -= keep - 1
            changed = True
        else:
            output.append(line)
    repaired = "\n".join(output).strip()
    if changed and len(repaired) <= max_chars and _lyric_stats(repaired)["line_count"] > stats["line_count"]:
        return repaired, True
    return text, False


def _lyric_plan(duration: float, density: str, structure_preset: str, genre_hint: str) -> dict[str, Any]:
    try:
        from songwriting_toolkit import lyric_length_plan

        return lyric_length_plan(duration, density, structure_preset, genre_hint)
    except Exception:
        dur = int(float(duration or 180))
        min_words = 120 if dur <= 120 else 240 if dur <= 180 else 430
        return {
            "duration": dur,
            "sections": ["Intro", "Verse 1", "Chorus", "Verse 2", "Bridge", "Final Chorus", "Outro"],
            "target_words": min_words + 80,
            "min_words": min_words,
            "target_lines": max(24, min_words // 5),
            "min_lines": max(16, min_words // 7),
            "max_lyrics_chars": ACE_STEP_LYRICS_CHAR_LIMIT,
        }


def _caption_has_leakage(text: str) -> list[str]:
    issues: list[str] = []
    for pattern in CAPTION_LEAK_PATTERNS:
        if pattern.search(text or ""):
            issues.append(pattern.pattern)
    if len(str(text or "")) > ACE_STEP_CAPTION_CHAR_LIMIT:
        issues.append("caption_over_budget")
    return issues


def tag_dimension_coverage(caption: str, tag_list: Any = None) -> dict[str, Any]:
    combined = " ".join([str(caption or ""), " ".join(_split_terms(tag_list))]).lower()
    dimensions = []
    missing = []
    for dimension, keywords in TAG_DIMENSION_KEYWORDS.items():
        matched = sorted({keyword for keyword in keywords if keyword in combined})
        status = "pass" if matched else "missing"
        dimensions.append({"dimension": dimension, "status": status, "matched": matched[:8]})
        if status != "pass":
            missing.append(dimension)
    return {
        "version": ALBUM_PAYLOAD_GATE_VERSION,
        "status": "pass" if not missing else "repair_needed",
        "dimensions": dimensions,
        "missing": missing,
    }


def _clean_caption(payload: dict[str, Any], coverage: dict[str, Any]) -> str:
    terms: list[str] = []
    for value in [payload.get("tag_list"), payload.get("caption"), payload.get("style"), payload.get("vibe"), payload.get("description")]:
        for term in _split_terms(value):
            if len(term) > 80:
                continue
            if _caption_has_leakage(term):
                continue
            if term.lower() not in {existing.lower() for existing in terms}:
                terms.append(term)
    for dimension in coverage.get("missing") or []:
        repair = DIMENSION_REPAIR_TERMS.get(str(dimension), "")
        if repair and repair.lower() not in {existing.lower() for existing in terms}:
            terms.append(repair)
    if not terms:
        terms = ["modern pop", "steady groove", "clear lead vocal", "dynamic hook arrangement", "polished studio mix"]
    caption = ", ".join(terms[:18])
    return re.sub(r"\s+", " ", caption).strip(" ,.")[:ACE_STEP_CAPTION_CHAR_LIMIT]


def build_album_global_sonic_caption(concept: str, tracks: list[dict[str, Any]] | None = None, existing: Any = "") -> str:
    existing_text = str(existing or "").strip()
    if existing_text and not _caption_has_leakage(existing_text) and len(existing_text) <= ACE_STEP_CAPTION_CHAR_LIMIT:
        return existing_text
    terms: list[str] = []
    for track in tracks or []:
        for value in [track.get("tag_list"), track.get("tags"), track.get("style"), track.get("vibe")]:
            for term in _split_terms(value):
                if len(term) <= 54 and not _caption_has_leakage(term) and term.lower() not in {item.lower() for item in terms}:
                    terms.append(term)
    if not terms:
        terms = _split_terms(concept)[:8]
    if not terms:
        terms = ["cohesive album", "clear lead vocals", "dynamic arrangements", "polished studio mix"]
    coverage = tag_dimension_coverage(", ".join(terms), terms)
    for missing in coverage.get("missing") or []:
        repair = DIMENSION_REPAIR_TERMS.get(str(missing), "")
        if repair and repair.lower() not in {item.lower() for item in terms}:
            terms.append(repair)
    return ", ".join(terms[:16])[:ACE_STEP_CAPTION_CHAR_LIMIT]


def evaluate_album_payload_quality(
    payload: dict[str, Any],
    *,
    options: dict[str, Any] | None = None,
    repair: bool = True,
) -> dict[str, Any]:
    original = dict(payload or {})
    repaired = dict(original)
    opts = dict(options or {})
    issues: list[dict[str, Any]] = []
    repair_actions: list[str] = []

    caption = str(repaired.get("caption") or repaired.get("tags") or "")
    tag_list = repaired.get("tag_list") or ((repaired.get("album_metadata") or {}).get("tag_list") if isinstance(repaired.get("album_metadata"), dict) else [])
    tag_terms = _split_terms(tag_list)
    coverage = tag_dimension_coverage(caption, tag_list)
    caption_leaks = _caption_has_leakage(caption)
    tag_leaks = [term for term in tag_terms if _caption_has_leakage(term)]
    if caption_leaks or tag_leaks:
        issues.append({"id": "caption_leakage", "severity": "repairable", "detail": f"{len(caption_leaks) + len(tag_leaks)} leak marker(s)"})
    if coverage.get("missing"):
        issues.append({"id": "tag_dimension_coverage", "severity": "repairable", "detail": ", ".join(coverage["missing"])})
    if repair and (caption_leaks or tag_leaks or coverage.get("missing")):
        if tag_leaks:
            repaired["tag_list"] = [term for term in tag_terms if not _caption_has_leakage(term)]
            tag_list = repaired["tag_list"]
        repaired_caption = _clean_caption(repaired, coverage)
        repaired["caption"] = repaired_caption
        repaired["tags"] = repaired_caption
        repair_actions.append("caption_rebuilt_from_tag_dimensions")
        coverage = tag_dimension_coverage(repaired_caption, tag_list)

    global_caption = str(repaired.get("global_caption") or "")
    global_leaks = _caption_has_leakage(global_caption)
    if global_leaks:
        if repair:
            repaired["global_caption"] = build_album_global_sonic_caption(
                "",
                [{"tags": repaired.get("caption"), "tag_list": tag_list, "style": repaired.get("style"), "vibe": repaired.get("vibe")}],
                existing="",
            )
            repair_actions.append("global_caption_rebuilt")
        else:
            issues.append({"id": "global_caption_leakage", "severity": "fail", "detail": f"{len(global_leaks)} leak marker(s)"})

    duration = float(repaired.get("duration") or opts.get("track_duration") or 180)
    density = str(repaired.get("lyric_density") or opts.get("lyric_density") or "dense")
    structure_preset = str(repaired.get("structure_preset") or opts.get("structure_preset") or "auto")
    genre_hint = " ".join(
        str(repaired.get(key) or "")
        for key in ("caption", "description", "style", "vibe", "narrative")
    )
    plan = _lyric_plan(duration, density, structure_preset, genre_hint)
    lyrics = str(repaired.get("lyrics") or "")
    if repair and "\\n" in lyrics and lyrics.count("\\n") >= 3:
        lyrics = lyrics.replace("\\r\\n", "\n").replace("\\n", "\n")
        repaired["lyrics"] = lyrics
        repair_actions.append("lyrics_unescaped_newlines")
    instrumental = str(lyrics).strip().lower() == "[instrumental]" or bool(repaired.get("instrumental"))
    stats = _lyric_stats(lyrics)
    expected_keys = {_section_key(section) for section in plan.get("sections") or [] if section}
    actual_keys = {_section_key(section) for section in stats["sections"] if section}
    section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0

    if not instrumental:
        min_words = int(plan.get("min_words") or 0)
        min_lines = int(plan.get("min_lines") or 0)
        if repair and stats["line_count"] < min_lines and stats["word_count"] >= min_words and stats["char_count"] <= ACE_STEP_LYRICS_CHAR_LIMIT:
            reflowed, changed = _reflow_lyrics_to_min_lines(lyrics, min_lines, ACE_STEP_LYRICS_CHAR_LIMIT)
            if changed:
                old_lines = stats["line_count"]
                repaired["lyrics"] = reflowed
                lyrics = reflowed
                stats = _lyric_stats(lyrics)
                actual_keys = {_section_key(section) for section in stats["sections"] if section}
                section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0
                repair_actions.append(f"lyrics_reflowed_to_min_lines:{old_lines}->{stats['line_count']}")
        if not has_vocal_lyrics(lyrics):
            issues.append({"id": "lyrics_missing", "severity": "fail", "detail": "vocal track has no lyrics"})
        if stats["char_count"] > ACE_STEP_LYRICS_CHAR_LIMIT:
            issues.append({"id": "lyrics_over_budget", "severity": "fail", "detail": f"{stats['char_count']}/{ACE_STEP_LYRICS_CHAR_LIMIT} chars"})
        if stats["word_count"] < int(plan.get("min_words") or 0):
            issues.append({"id": "lyrics_under_length", "severity": "fail", "detail": f"{stats['word_count']}/{plan.get('min_words')} words"})
        if stats["line_count"] < int(plan.get("min_lines") or 0):
            issues.append({"id": "lyrics_too_few_lines", "severity": "fail", "detail": f"{stats['line_count']}/{plan.get('min_lines')} lines"})
        if section_coverage < 0.72:
            issues.append({"id": "section_coverage_low", "severity": "fail", "detail": f"{section_coverage} coverage"})
        if stats["hook_count"] < 1:
            issues.append({"id": "hook_missing", "severity": "fail", "detail": "no chorus/hook/refrain section"})
        if stats["fallback_artifact_count"]:
            issues.append({"id": "fallback_lyric_artifacts", "severity": "fail", "detail": f"{stats['fallback_artifact_count']} artifact(s)"})
        if stats["meta_leak_lines"]:
            issues.append({"id": "lyric_meta_leakage", "severity": "fail", "detail": f"{len(stats['meta_leak_lines'])} line(s)"})
        if stats["placeholder_count"]:
            issues.append({"id": "lyric_placeholders", "severity": "fail", "detail": f"{stats['placeholder_count']} placeholder(s)"})
        severe_repetition = (
            len(stats["repeated_lines"]) > max(20, stats["line_count"] // 2)
            or (stats["line_count"] >= 24 and stats["unique_line_ratio"] < 0.35)
        )
        notable_repetition = len(stats["repeated_lines"]) > max(6, stats["line_count"] // 4) or (
            stats["line_count"] >= 24 and stats["unique_line_ratio"] < 0.45
        )
        if severe_repetition:
            issues.append({"id": "too_many_repeated_lines", "severity": "fail", "detail": f"{len(stats['repeated_lines'])} repeated line(s)"})
        elif notable_repetition:
            issues.append({"id": "lyric_repetition_warning", "severity": "warning", "detail": f"{len(stats['repeated_lines'])} repeated line(s)"})

    required_phrases = [str(item).strip() for item in (repaired.get("required_phrases") or []) if str(item).strip()]
    missing_required = [phrase for phrase in required_phrases if phrase.lower() not in lyrics.lower()]
    if missing_required and not instrumental:
        append_block = "\n\n[Required Hook]\n" + "\n".join(missing_required[:6])
        if repair and len(lyrics) + len(append_block) <= ACE_STEP_LYRICS_CHAR_LIMIT:
            repaired["lyrics"] = (lyrics.rstrip() + append_block).strip()
            lyrics = str(repaired["lyrics"])
            stats = _lyric_stats(lyrics)
            repair_actions.append("missing_required_phrases_appended")
        else:
            issues.append({"id": "required_phrases_missing", "severity": "fail", "detail": ", ".join(missing_required[:4])})

    fail_issues = [issue for issue in issues if issue.get("severity") == "fail"]
    repairable_issues = [issue for issue in issues if issue.get("severity") == "repairable"]
    if fail_issues:
        status = "fail"
    elif repair_actions:
        status = "auto_repair"
    elif repairable_issues:
        status = "review_needed"
    else:
        status = "pass"

    final_caption_leaks = _caption_has_leakage(str(repaired.get("caption") or ""))
    caption_integrity = {
        "status": "pass" if not final_caption_leaks else "fail",
        "char_count": len(str(repaired.get("caption") or "")),
        "leakage_markers": final_caption_leaks,
        "repaired_leakage_markers": caption_leaks,
    }
    lyric_duration_fit = {
        "status": "pass" if not fail_issues else "fail",
        "duration": duration,
        "plan": plan,
        "stats": stats,
        "section_coverage": section_coverage,
        "expected_sections": sorted(expected_keys),
        "actual_sections": sorted(actual_keys),
    }
    report = {
        "version": ALBUM_PAYLOAD_GATE_VERSION,
        "status": status,
        "gate_passed": status in {"pass", "auto_repair"},
        "issues": issues,
        "blocking_issues": fail_issues,
        "repair_actions": repair_actions,
        "tag_coverage": coverage,
        "caption_integrity": caption_integrity,
        "lyric_duration_fit": lyric_duration_fit,
        "repaired_payload": repaired,
    }
    repaired["payload_gate_status"] = status
    repaired["payload_quality_gate"] = {key: value for key, value in report.items() if key != "repaired_payload"}
    repaired["tag_coverage"] = coverage
    repaired["caption_integrity"] = caption_integrity
    repaired["lyric_duration_fit"] = lyric_duration_fit
    repaired["repair_actions"] = repair_actions
    warnings = list(repaired.get("payload_warnings") or [])
    if status != "pass":
        warnings.append(f"album_payload_gate:{status}")
    repaired["payload_warnings"] = warnings
    report["repaired_payload"] = repaired
    return report
