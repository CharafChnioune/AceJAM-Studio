from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from studio_core import (
    ACE_STEP_CAPTION_CHAR_LIMIT,
    ACE_STEP_LYRICS_CHAR_LIMIT,
    ACE_STEP_LYRICS_SAFE_HEADROOM,
    ACE_STEP_LYRICS_WARNING_CHAR_LIMIT,
    has_vocal_lyrics,
)


ALBUM_PAYLOAD_GATE_VERSION = "album-payload-quality-gate-2026-04-29"
LYRIC_CRAFT_GATE_VERSION = "lyric-craft-gate-2026-05-02"

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
        "choir", "harmony", "harmonies", "vocal response", "hook", "chorus", "chant",
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

GENRE_STYLE_KEYWORDS = {
    "pop", "rap", "hip-hop", "hip hop", "trap", "drill", "g-funk", "west coast",
    "boom-bap", "boom bap", "r&b", "soul", "rock", "metal", "punk", "house",
    "techno", "trance", "dancehall", "reggaeton", "afro", "amapiano",
    "cinematic", "ambient", "schlager", "latin", "j-pop", "k-pop",
}

SONIC_CAPTION_TERM_RE = re.compile(
    r"\b(?:"
    r"pop|rap|hip-hop|hip hop|trap|drill|g-funk|r&b|soul|rock|metal|punk|house|techno|trance|"
    r"dancehall|reggaeton|afro|amapiano|cinematic|ambient|schlager|latin|j-pop|k-pop|"
    r"groove|bounce|swing|boom-bap|drums?|hi-hats?|snare|kick|percussion|shuffle|rhythm|"
    r"bass|808|sub-bass|low-end|synth|piano|guitar|strings|brass|horns?|sirens?|accordion|organ|"
    r"vocal|voice|sung|singer|choir|harmony|hook|chorus|chant|"
    r"dark|bright|uplifting|melancholic|euphoric|dreamy|nostalgic|aggressive|romantic|hopeful|"
    r"tense|warm|cold|gritty|emotional|intimate|confident|"
    r"build|drop|bridge|anthemic|breakdown|dynamic|riser|climax|intro|outro|stadium|"
    r"mix|master|polished|crisp|clean|wide stereo|high-fidelity|radio|studio|glossy|punchy|analog|"
    r"west coast"
    r")\b",
    re.I,
)

DIMENSION_REPAIR_TERMS: dict[str, str] = {
    "genre_style": "modern pop",
    "rhythm_groove": "steady groove",
    "instrumentation": "layered drums and melodic instruments",
    "vocal_style": "clear lead vocal",
    "mood_atmosphere": "emotional atmosphere",
    "arrangement_energy": "dynamic hook arrangement",
    "mix_production": "polished studio mix",
}

PRODUCER_GRADE_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "primary_genre": (
        "pop", "rap", "hip-hop", "hip hop", "trap", "drill", "g-funk", "boom-bap",
        "r&b", "soul", "rock", "house", "techno", "reggaeton", "afrobeats",
        "amapiano", "dancehall", "cinematic", "ambient", "schlager", "latin",
    ),
    "drum_groove": (
        "drum", "drums", "boom-bap", "kick", "snare", "hi-hat", "hi hats",
        "hats", "percussion", "groove", "bounce", "swing", "shuffle", "pocket",
        "breakbeat", "handclap", "clap", "rhythm",
    ),
    "low_end_bass": (
        "808", "bass", "bassline", "sub-bass", "sub bass", "low-end", "low end",
        "deep low end", "sliding 808", "synth bass", "bass guitar", "rubber bass",
    ),
    "melodic_identity": (
        "piano", "keys", "rhodes", "synth", "lead synth", "pad", "sample",
        "soul chop", "chop", "riff", "guitar", "brass", "horn", "strings",
        "organ", "melody", "melodic", "loop", "motif", "pluck", "bell", "sirens",
    ),
    "vocal_delivery": (
        "vocal", "voice", "rap vocal", "male rap", "female rap", "lead vocal",
        "clear vocal", "vocal pocket", "cadence", "ad-lib", "adlibs", "chant",
        "hook response", "chorus response", "stacked vocals", "harmony",
    ),
    "arrangement_movement": (
        "hook", "chorus", "bridge", "beat switch", "breakdown", "build", "drop",
        "call and response", "dynamic", "final hook", "intro", "outro", "switch-up",
        "section", "arrangement",
    ),
    "texture_space": (
        "gritty", "vinyl", "tape", "analog", "warm", "cold", "dark", "airy",
        "dry", "reverb", "space", "atmospheric", "wide", "dusty", "glossy",
        "saturated", "sirens", "neon",
    ),
    "mix_master": (
        "mix", "master", "polished", "crisp", "clean", "punchy", "studio",
        "high-fidelity", "radio-ready", "radio ready", "wide stereo", "club",
        "tight", "balanced", "glossy",
    ),
}

PRODUCER_GRADE_REPAIR_TERMS: dict[str, str] = {
    "primary_genre": "modern pop",
    "drum_groove": "tight punchy drums",
    "low_end_bass": "deep low-end bass",
    "melodic_identity": "memorable piano or synth motif",
    "vocal_delivery": "clear lead vocal pocket",
    "arrangement_movement": "dynamic hook arrangement",
    "texture_space": "warm analog texture",
    "mix_master": "crisp polished studio mix",
}

TAG_DIMENSION_KEYWORDS = PRODUCER_GRADE_DIMENSION_KEYWORDS
DIMENSION_REPAIR_TERMS = PRODUCER_GRADE_REPAIR_TERMS

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
    r"\b(?:morning finds the|morning lifts\b|light moves softly through the door|"
    r"midnight paints the\b|breeze reminds me time moves fast|now i build a future from the past|"
    r"every note becomes the reason|turn pressure into perfume|"
    r"light is leaning through the door|kept the receipt from the life before|"
    r"now i want the sound and nothing more|one more line keeps moving|one more breath holds steady|"
    r"final echo cuts through stone|last breath keeps the promise|one clear image brings release|"
    r"\w+\s+carries\s+\w+\s+through pressure\s+\d+|"
    r"\w+\s+answers\s+\w+\s+with steady breath\s+\d+|"
    r"the you|the was|the are|the is|the but|the end from the floor)\b",
    re.I,
)
PLACEHOLDER_RE = re.compile(r"\b(?:placeholder|repeat chorus|same as before|continue|tbd|todo|\.\.\.)\b", re.I)
SECTION_RE = re.compile(r"\[([^\]]+)\]")
WORD_RE = re.compile(r"[A-Za-z0-9À-ÿ\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff']+")
GENERIC_AI_LYRIC_RE = re.compile(
    r"\b(?:"
    r"neon dreams?|fire inside|rise up|we rise|we fly|chase the light|touch the sky|"
    r"never gonna break|break these chains|heart on fire|lost in the night|shining bright|"
    r"take me higher|feel alive|dreams come true|stars align|find our way|"
    r"through the storm|we are strong|stronger than before|nothing can stop us"
    r")\b",
    re.I,
)
ADJECTIVE_STACK_RE = re.compile(
    r"\b(?:dark|bright|broken|empty|endless|lonely|silent|golden|electric|neon|"
    r"burning|cold|warm|wild|heavy|soft)\s+"
    r"(?:dark|bright|broken|empty|endless|lonely|silent|golden|electric|neon|"
    r"burning|cold|warm|wild|heavy|soft)\s+"
    r"(?:dreams?|nights?|skies|fire|hearts?|streets?|souls?|shadows?)\b",
    re.I,
)
CONCRETE_SCENE_RE = re.compile(
    r"\b(?:street|streets|block|blocks|corner|kitchen|window|windows|door|doors|"
    r"table|train|station|car|cars|engine|rain|glass|concrete|brick|ledger|receipt|"
    r"knife|coin|coins|sirens?|smoke|dust|floor|hands?|mouth|teeth|coat|suit|"
    r"church|market|city|skyline|river|bridge|phone|screen|wire|crown|archive|"
    r"drums?|bass|piano|guitar|horns?|sample|vinyl|tape)\b",
    re.I,
)
ACTION_VERB_RE = re.compile(
    r"\b(?:cuts?|cracks?|leans?|paves?|buries|whispers?|counts?|signs?|shakes?|"
    r"bleeds?|drags?|pulls?|pushes?|opens?|locks?|breaks?|carries|answers?|"
    r"remembers?|swallows?|spills?|burns?|lands?|moves?|knocks?|turns?|folds?)\b",
    re.I,
)
ABSTRACT_SLOGAN_RE = re.compile(
    r"\b(?:hope|dreams?|destiny|freedom|truth|love|pain|soul|heart|fear|light|"
    r"darkness|strength|power|legacy|rebirth|pressure|loyalty|money)\b",
    re.I,
)
METAPHOR_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "weather": ("storm", "rain", "thunder", "lightning", "cloud", "wind"),
    "fire": ("fire", "flame", "burn", "ashes", "smoke", "spark"),
    "ocean": ("ocean", "wave", "tide", "ship", "sail", "drown"),
    "space": ("stars", "moon", "galaxy", "orbit", "comet", "cosmic"),
    "war": ("war", "battle", "soldier", "gun", "armor", "blood"),
    "machine": ("machine", "engine", "wire", "circuit", "gear", "data"),
    "religion": ("angel", "devil", "heaven", "god", "prayer", "altar"),
    "money": ("ledger", "coin", "cash", "bank", "debt", "receipt"),
}


def _producer_credit_aliases(value: Any) -> list[str]:
    aliases: list[str] = []
    for item in re.split(r"[,/&;]|\b(?:and|with|prod\.?|produced by)\b", str(value or ""), flags=re.I):
        text = re.sub(r"\s+", " ", item).strip(" .:-")
        if len(text) >= 3 and text.lower() not in {"unknown", "none", "producer"}:
            aliases.append(text)
    return aliases


def _lyrics_contain_any(lyrics: str, aliases: list[str]) -> list[str]:
    found: list[str] = []
    lowered = str(lyrics or "").lower()
    for alias in aliases:
        if alias.lower() in lowered and alias not in found:
            found.append(alias)
    return found


def _repair_producer_credit_lyrics(lyrics: str, aliases: list[str]) -> tuple[str, bool]:
    if not aliases:
        return lyrics, False
    changed = False
    repaired_lines: list[str] = []
    for line in str(lyrics or "").splitlines():
        if any(alias.lower() in line.lower() for alias in aliases):
            if line.strip().startswith("["):
                repaired_lines.append(line)
            else:
                repaired_lines.append("The beat lands heavy, crisp and clean,")
            changed = True
        else:
            repaired_lines.append(line)
    return "\n".join(repaired_lines), changed


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


def _phrase_norm(text: Any) -> str:
    value = str(text or "").lower()
    value = (
        value.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("…", " ")
        .replace("—", " ")
        .replace("–", " ")
    )
    return " ".join(_words(value))


def _phrase_present(phrase: Any, lyrics: str) -> bool:
    raw_phrase = str(phrase or "").strip()
    if not raw_phrase:
        return True
    lyric_text = str(lyrics or "")
    if raw_phrase.lower() in lyric_text.lower():
        return True
    normalized_phrase = _phrase_norm(raw_phrase)
    if not normalized_phrase:
        return True
    return normalized_phrase in _phrase_norm(lyric_text)


def _caption_term_is_genre(term: Any) -> bool:
    lowered = str(term or "").lower()
    if re.search(r"\b(?:drums?|vocal|voice|male|female|bass|808|hook|chorus|mix|master|texture|pocket)\b", lowered):
        return False
    return any(keyword in lowered for keyword in GENRE_STYLE_KEYWORDS)


def _prune_caption_terms(terms: list[str]) -> list[str]:
    pruned: list[str] = []
    genre_count = 0
    for term in terms:
        if _caption_term_is_genre(term):
            if genre_count >= 2:
                continue
            genre_count += 1
        if term.lower() not in {existing.lower() for existing in pruned}:
            pruned.append(term)
    return pruned


def _effective_min_lines(raw_min_lines: int, min_words: int) -> int:
    """Cap line targets so lyrics stay singable instead of over-fragmented."""
    raw = int(raw_min_lines or 0)
    if raw <= 0:
        return 0
    cap = max(36, int((int(min_words or 0) / 5.7) + 0.999))
    return min(raw, cap)


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


def _caption_term_allowed(term: Any) -> bool:
    text = re.sub(r"\s+", " ", str(term or "")).strip(" .")
    if not text:
        return False
    if len(text) > 64:
        return False
    if _caption_has_leakage(text):
        return False
    if re.search(r"[\n\r{}]|[.!?]", text):
        return False
    words = WORD_RE.findall(text)
    if len(words) > 6:
        return False
    lowered_words = {word.lower() for word in words}
    if lowered_words & {"and", "while", "with", "that", "this"} and not re.search(
        r"\b(?:r&b|drum and bass|call and response|wide stereo|warm analog|male and female)\b",
        text,
        re.I,
    ):
        return False
    if not SONIC_CAPTION_TERM_RE.search(text):
        return False
    if re.search(r"\b(?:album|track|verse|lyrics|prompt|json|metadata|narrative|naming drop|produced by|prod\.)\b", text, re.I):
        return False
    return True


RAP_INTENT_RE = re.compile(r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b", re.I)
RAP_CORE_RE = re.compile(r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b", re.I)
NON_RAP_PRIMARY_RE = re.compile(
    r"\b(?:house|techno|trance|edm|dance|r&b|soul|rock|metal|punk|country|"
    r"reggaeton|afro|afrobeats|amapiano|dancehall|latin|cinematic|ambient|instrumental|pop)\b",
    re.I,
)
RAP_VOCAL_RE = re.compile(
    r"\b(?:rap vocal|rapper|rapped|rap lead|male rap|female rap|spoken[-\s]?word|rhythmic vocal|"
    r"flow|cadence|vocal pocket|tight pocket)\b",
    re.I,
)
RAP_GROOVE_RE = re.compile(
    r"\b(?:hip[-\s]?hop drums|boom[-\s]?bap|trap hi[-\s]?hats?|drill drums|rap groove|"
    r"steady groove|hard[-\s]?hitting drums|drum patterns?|hard drums|kick|snare|"
    r"hi[-\s]?hats?|rhythmic pocket|bounce)\b",
    re.I,
)
RAP_LOW_END_RE = re.compile(r"\b(?:808|sub[-\s]?bass|deep bass|low[-\s]?end|bassline|bass|g[-\s]?funk bass)\b", re.I)
CINEMATIC_OVERLAY_RE = re.compile(
    r"\b(?:cinematic|orchestral|orchestra|strings?|brass|horns?|taiko|choir|score|trailer|epic)\b",
    re.I,
)
ARRANGEMENT_LYRIC_RE = re.compile(
    r"\b(?:instrumental break|(?:the\s+)?orchestra(?:l)?\s+swells?|strings?\s+(?:fade|fades|swell|swells|enter|enters)|"
    r"brass\s+(?:swells?|hits?|enter|enters)|taiko\s+drums?|"
    r"production\s+(?:drops|builds)|mix\s+(?:opens|widens))\b",
    re.I,
)
ARRANGEMENT_RAP_BAR_ALLOWED_RE = re.compile(
    r"\b(?:drums?\s+(?:hit|hits|knock|knocks|slap|slaps|crack|cracks)\s+hard|"
    r"(?:every|each)\s+drum\s+hit|bassline\s+drops?|bass\s+(?:drops?|hits?|knocks?|slaps?)|"
    r"low[-\s]?end\s+(?:shadows?|rumbles?|moves?|talks?|knocks?|hits?)|"
    r"808s?\s+(?:drop|drops|hit|hits|knock|knocks|roll|rolls))\b",
    re.I,
)


def _nested_album_options(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("album_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("album_options"), dict):
        return dict(metadata.get("album_options") or {})
    return {}


def _genre_source_text(
    payload: dict[str, Any] | None,
    options: dict[str, Any] | None = None,
    *,
    primary_only: bool = False,
) -> str:
    parts: list[str] = []
    for source in (options or {}, _nested_album_options(payload), payload or {}):
        if not isinstance(source, dict):
            continue
        for key in (
            "album_agent_genre_prompt",
            "genre_prompt",
            "album_agent_vocal_type",
            "vocal_type",
            "custom_tags",
            "style",
            "locked_style",
            "user_prompt",
            "raw_user_prompt",
            "prompt",
            "concept",
        ):
            value = source.get(key)
            if value:
                parts.append(" ".join(str(item) for item in value) if isinstance(value, list) else str(value))
        if primary_only:
            continue
        for key in ("genre_profile", "performance_brief", "tags", "caption", "description", "vibe", "narrative"):
            value = source.get(key)
            if value:
                parts.append(" ".join(str(item) for item in value) if isinstance(value, list) else str(value))
        tag_list = source.get("tag_list")
        if not primary_only and isinstance(tag_list, list):
            parts.extend(str(item) for item in tag_list if str(item).strip())
    return re.sub(r"\s+", " ", "\n".join(parts)).strip()


def build_genre_intent_contract(payload: dict[str, Any] | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = (payload or {}).get("genre_intent_contract") if isinstance(payload, dict) else None
    if isinstance(existing, dict) and existing.get("family"):
        return dict(existing)
    primary_source_text = _genre_source_text(payload, options, primary_only=True)
    source_text = _genre_source_text(payload, options, primary_only=False)
    track_source_text = " ".join(
        str((payload or {}).get(key) or "")
        for key in ("style", "locked_style", "caption", "tags", "genre_profile", "description", "vibe")
    )
    global_intent_text = " ".join(
        str((options or {}).get(key) or "")
        for key in ("album_agent_genre_prompt", "genre_prompt", "album_agent_vocal_type", "vocal_type", "custom_tags")
    )
    track_rap = bool(RAP_INTENT_RE.search(track_source_text))
    global_rap = bool(RAP_INTENT_RE.search(global_intent_text))
    track_non_rap_primary = bool(NON_RAP_PRIMARY_RE.search(track_source_text)) and not track_rap
    if track_non_rap_primary and not global_rap:
        return {
            "version": "genre-intent-contract-2026-05-02",
            "active": False,
            "family": "",
            "strict": False,
            "required": [],
            "allowed_secondary_color": [],
            "source_preview": source_text[:260],
        }
    primary_rap = bool(RAP_INTENT_RE.search(primary_source_text))
    secondary_rap = bool(RAP_INTENT_RE.search(source_text))
    if primary_rap or secondary_rap:
        return {
            "version": "genre-intent-contract-2026-05-02",
            "active": True,
            "family": "rap",
            "strict": primary_rap,
            "required": ["rap_core", "rap_vocal", "rap_groove", "low_end"],
            "allowed_secondary_color": ["cinematic", "orchestral"],
            "primary_source_preview": primary_source_text[:260],
            "source_preview": source_text[:260],
        }
    return {
        "version": "genre-intent-contract-2026-05-02",
        "active": False,
        "family": "",
        "strict": False,
        "required": [],
        "allowed_secondary_color": [],
        "source_preview": source_text[:260],
    }


def _regex_hits(pattern: re.Pattern[str], text: str) -> list[str]:
    hits: list[str] = []
    for match in pattern.finditer(text or ""):
        value = re.sub(r"\s+", " ", match.group(0).strip().lower())
        if value and value not in hits:
            hits.append(value)
    return hits


def _non_section_lines(lyrics: str) -> list[str]:
    return [
        _normalize_lyric_section_line(line).strip()
        for line in str(lyrics or "").splitlines()
        if line.strip() and not _section_only_line(line)
    ]


def _arrangement_lyric_scan(lines: list[str]) -> list[dict[str, str]]:
    scan: list[dict[str, str]] = []
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        blocked = ARRANGEMENT_LYRIC_RE.search(text)
        if blocked:
            scan.append({"line": text, "status": "blocked", "match": blocked.group(0)})
            continue
        allowed = ARRANGEMENT_RAP_BAR_ALLOWED_RE.search(text)
        if allowed:
            scan.append({"line": text, "status": "allowed_rap_bar", "match": allowed.group(0)})
    return scan


def evaluate_genre_adherence(payload: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(payload or {})
    contract = build_genre_intent_contract(payload, options)
    issues: list[dict[str, Any]] = []
    if contract.get("family") != "rap":
        return {
            "version": "genre-adherence-2026-05-02",
            "status": "pass",
            "gate_passed": True,
            "contract": contract,
            "issues": [],
            "issue_ids": [],
        }

    caption_terms = _split_terms(payload.get("caption") or payload.get("tags") or "")
    tag_terms = _split_terms(payload.get("tag_list") or [])
    performance_terms = _split_terms(payload.get("performance_brief") or "") + _split_terms(payload.get("genre_profile") or "")
    sonic_text = " ".join([*caption_terms, *tag_terms, *performance_terms, str(payload.get("style") or "")])
    sonic_text_lower = sonic_text.lower()
    first_terms = [term.lower() for term in [*tag_terms, *caption_terms][:4]]

    rap_hits = _regex_hits(RAP_CORE_RE, sonic_text_lower)
    vocal_hits = _regex_hits(RAP_VOCAL_RE, sonic_text_lower)
    groove_hits = _regex_hits(RAP_GROOVE_RE, sonic_text_lower)
    low_end_hits = _regex_hits(RAP_LOW_END_RE, sonic_text_lower)
    cinematic_hits = _regex_hits(CINEMATIC_OVERLAY_RE, sonic_text_lower)

    strict = bool(contract.get("strict"))
    if not strict:
        return {
            "version": "genre-adherence-2026-05-02",
            "status": "pass",
            "gate_passed": True,
            "contract": contract,
            "issues": [],
            "issue_ids": [],
            "stats": {
                "rap_hits": rap_hits,
                "rap_vocal_hits": vocal_hits,
                "rap_groove_hits": groove_hits,
                "low_end_hits": low_end_hits,
                "cinematic_overlay_hits": cinematic_hits,
                "rap_front_loaded": any(RAP_CORE_RE.search(term) or RAP_GROOVE_RE.search(term) for term in first_terms),
            },
        }

    if not rap_hits:
        issues.append({"id": "genre_intent_missing_rap_core", "severity": "fail", "detail": "caption/tags need rap or hip-hop as primary genre"})
    if not vocal_hits:
        issues.append({"id": "genre_intent_missing_rap_vocal", "severity": "fail", "detail": "caption/tags need rap vocal delivery"})
    if not groove_hits:
        issues.append({"id": "genre_intent_missing_rap_groove", "severity": "fail", "detail": "caption/tags need hip-hop drums or rap groove"})
    if not low_end_hits:
        issues.append({"id": "genre_intent_missing_low_end", "severity": "fail", "detail": "caption/tags need 808, bassline, sub-bass, or low-end anchor"})

    rap_front_loaded = any(RAP_CORE_RE.search(term) or RAP_GROOVE_RE.search(term) for term in first_terms)
    if cinematic_hits and (len(cinematic_hits) > len(rap_hits) + 1) and not rap_front_loaded:
        issues.append({"id": "rap_not_dominant", "severity": "fail", "detail": "cinematic/orchestral color appears before rap anchors"})
    if len(cinematic_hits) >= 4 and len(rap_hits) <= 2:
        issues.append({"id": "orchestral_overdominant", "severity": "fail", "detail": "orchestral/cinematic terms dominate the rap prompt"})

    lyric_lines = _non_section_lines(str(payload.get("lyrics") or "\n".join(payload.get("lyrics_lines") or [])))
    arrangement_scan = _arrangement_lyric_scan(lyric_lines)
    arrangement_lines = [item["line"] for item in arrangement_scan if item.get("status") == "blocked"]
    if arrangement_lines:
        issues.append({
            "id": "non_rap_arrangement_lyric_leakage",
            "severity": "fail",
            "detail": " | ".join(arrangement_lines[:3]),
        })
    if lyric_lines:
        long_lines = [line for line in lyric_lines if len(_words(line)) > 13]
        avg_words = round(sum(len(_words(line)) for line in lyric_lines) / max(1, len(lyric_lines)), 2)
        if avg_words > 10.5 or len(long_lines) > max(3, len(lyric_lines) // 4):
            issues.append({
                "id": "rap_lines_not_bar_like",
                "severity": "fail",
                "detail": f"avg_words={avg_words}; long_lines={len(long_lines)}",
            })

    return {
        "version": "genre-adherence-2026-05-02",
        "status": "pass" if not issues else "fail",
        "gate_passed": not issues,
        "contract": contract,
        "issues": issues,
        "issue_ids": [str(issue.get("id")) for issue in issues],
        "stats": {
            "rap_hits": rap_hits,
            "rap_vocal_hits": vocal_hits,
            "rap_groove_hits": groove_hits,
            "low_end_hits": low_end_hits,
            "cinematic_overlay_hits": cinematic_hits,
            "rap_front_loaded": rap_front_loaded,
            "arrangement_lyric_scan": arrangement_scan,
        },
    }


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
    avg_words_per_line = round(len(words) / max(1, len(lyric_lines)), 2)
    return {
        "word_count": len(words),
        "line_count": len(lyric_lines),
        "avg_words_per_line": avg_words_per_line,
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


def _lyric_section_line_counts(lyrics: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    section_lines: dict[str, list[str]] = {}
    current = ""
    for raw_line in str(lyrics or "").splitlines():
        line = _normalize_lyric_section_line(raw_line).strip()
        if not line:
            continue
        section = SECTION_RE.fullmatch(line)
        if section:
            current = f"[{section.group(1)}]"
            counts.setdefault(current, 0)
            section_lines.setdefault(current, [])
            continue
        if not current:
            current = "[untagged]"
        counts[current] = counts.get(current, 0) + 1
        section_lines.setdefault(current, []).append(line)
    return {"counts": counts, "lines": section_lines}


def _lyric_section_blocks(lyrics: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current = ""
    for raw_line in str(lyrics or "").splitlines():
        line = _normalize_lyric_section_line(raw_line).strip()
        if not line:
            continue
        section = re.fullmatch(r"\[[^\]]+\]", line)
        if section:
            current = line
            blocks.append({"tag": current, "key": _section_key(current), "lines": []})
            continue
        if not current:
            current = "[untagged]"
            blocks.append({"tag": current, "key": "untagged", "lines": []})
        blocks[-1]["lines"].append(line)
    return blocks


def _lyric_family_from_source(source_text: str, contract: dict[str, Any]) -> str:
    source = str(source_text or "").lower()
    if contract.get("family") == "rap":
        return "rap"
    if re.search(r"\b(?:instrumental|score|ambient|cinematic underscore|soundtrack|edm|techno|house|trance)\b", source):
        return "sparse_or_instrumental"
    if re.search(r"\b(?:r&b|soul|pop|sung|vocal|ballad|country|rock|latin|afro|dancehall|reggaeton)\b", source):
        return "sung"
    return "vocal"


def _craft_source_preview(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"\([^)]*\bproduced\s+by\b[^)]*\)", "(producer reference redacted)", value, flags=re.I)
    value = re.sub(r"\b(?:produced\s+by|producer|prod\.)\s*[:=-]?\s*[^.;,\n|)]+", "producer reference redacted", value, flags=re.I)
    value = re.sub(r"\bLyrics?\s*:\s*.*$", "Lyrics: [redacted]", value, flags=re.I)
    value = re.sub(r"\bNaming\s+Drop\s*:\s*.*$", "Naming Drop: [redacted]", value, flags=re.I)
    return value[:260]


def build_lyrical_craft_contract(
    payload: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = (payload or {}).get("lyrical_craft_contract") if isinstance(payload, dict) else None
    if isinstance(existing, dict) and existing.get("family"):
        return dict(existing)
    source_text = _genre_source_text(payload, options, primary_only=False)
    genre_contract = build_genre_intent_contract(payload, options)
    family = _lyric_family_from_source(source_text, genre_contract)
    allow_surreal = bool(re.search(r"\b(?:surreal|abstract|psychedelic|experimental|dream logic|avant[-\s]?garde)\b", source_text, re.I))
    if family == "rap":
        required = [
            "rap verse minimum 16 bars on tracks >=120s",
            "multisyllabic mosaic rhymes stacked in begin/middle/end of bar",
            "slant-dominant flow with perfect-rhyme landings on emphasis lines",
            "8-15 syllables per bar working range; pocket beats acrobatics",
            "cadence and breath control",
            "concrete punchlines and Nas-style sensory scene anchors per line",
            "every verse changes something (scene, POV, time jump, escalation, revelation)",
            "hook passes hum-test (stranger grasps thesis from chorus alone)",
        ]
    elif family == "sung":
        required = [
            "vowel-friendly phrasing",
            "emotional clarity",
            "title-connected memorable hook",
            "singable line lengths",
            "Pat Pattison prosody match (form supports content)",
            "concrete sensory detail over abstract emotional labels",
            "every verse changes something (new scene, time, POV)",
        ]
    elif family == "sparse_or_instrumental":
        required = [
            "concise structure tags",
            "motif or energy movement",
            "no forced literary verses when the genre is sparse",
        ]
    else:
        required = [
            "concrete imagery",
            "one coherent metaphor world",
            "clear section contrast",
            "memorable hook promise",
            "every verse moves the story forward",
        ]
    return {
        "version": "lyrical-craft-contract-2026-05-08",
        "family": family,
        "repair_first": True,
        "pass_score": 82,
        "repair_score_min": 65,
        "allow_surreal_or_abstract": allow_surreal,
        "required": required,
        "hard_blockers": [
            "placeholders",
            "metadata or producer credits in lyrics",
            "deterministic fallback artifacts",
            "missing hook in vocal tracks",
            "extreme repetition",
            "mid-line truncation",
            "AI-cliche image bank phrases (neon dreams, fire inside, shattered dreams, endless night, empty streets, embers, whispers, silhouettes, echoes of, we rise, let it burn, chasing the night, broken heart, rising from the ashes, stars aligned, fade away, into the void, burning bright, frozen in time)",
            "telling-not-showing emotional labels (I feel sad, my heart is broken, this is sad)",
            "generic POV without a named situated speaker (we all, the world, everyone, the people)",
            "explanation lines (in other words, what I mean is, to be clear)",
        ],
        "source_preview": _craft_source_preview(source_text),
    }


def _metaphor_domain_hits(text: str) -> dict[str, list[str]]:
    lowered = str(text or "").lower()
    hits: dict[str, list[str]] = {}
    for domain, keywords in METAPHOR_DOMAIN_KEYWORDS.items():
        matched = sorted({keyword for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", lowered)})
        if matched:
            hits[domain] = matched
    return hits


def _hook_section_lines(blocks: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        if re.search(r"chorus|hook|refrain", str(block.get("tag") or ""), re.I):
            lines.extend(str(line) for line in block.get("lines") or [] if str(line).strip())
    return lines


def _weak_craft_sections(blocks: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    weak: list[dict[str, Any]] = []
    for block in blocks:
        tag = str(block.get("tag") or "[untagged]")
        lines = [str(line) for line in block.get("lines") or [] if str(line).strip()]
        if not lines:
            continue
        section_text = "\n".join(lines)
        reasons: list[str] = []
        if GENERIC_AI_LYRIC_RE.search(section_text):
            reasons.append("generic_ai_phrase")
        if ADJECTIVE_STACK_RE.search(section_text):
            reasons.append("adjective_stacking")
        long_limit = 14 if family == "rap" else 13
        long_lines = [line for line in lines if len(_words(line)) > long_limit]
        if long_lines and len(long_lines) > max(1, len(lines) // 3):
            reasons.append("line_breathability")
        concrete_count = sum(1 for line in lines if CONCRETE_SCENE_RE.search(line) or ACTION_VERB_RE.search(line))
        if len(lines) >= 4 and concrete_count == 0 and ABSTRACT_SLOGAN_RE.search(section_text):
            reasons.append("no_concrete_scene")
        if reasons:
            weak.append({"section": tag, "reasons": reasons, "preview": " | ".join(lines[:2])[:220]})
    return weak


def lyric_craft_gate(
    lyrics: str,
    payload: dict[str, Any] | None = None,
    *,
    options: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    duration: float | int | None = None,
    genre_hint: str = "",
    instrumental: bool = False,
    partial: bool = False,
) -> dict[str, Any]:
    payload = dict(payload or {})
    options = dict(options or {})
    source_payload = dict(payload)
    if genre_hint:
        source_payload["style"] = " ".join([str(source_payload.get("style") or ""), str(genre_hint or "")]).strip()
    contract = build_lyrical_craft_contract(source_payload, options)
    family = str(contract.get("family") or "vocal")
    lyric_text = str(lyrics or "")
    stats = _lyric_stats(lyric_text)
    blocks = _lyric_section_blocks(lyric_text)
    lines = _non_section_lines(lyric_text)
    issue_map: dict[str, dict[str, Any]] = {}
    score = 100

    def add_issue(issue_id: str, severity: str, detail: str = "", *, sections: list[str] | None = None, penalty: int | None = None) -> None:
        nonlocal score
        if issue_id in issue_map:
            if detail and detail not in str(issue_map[issue_id].get("detail") or ""):
                issue_map[issue_id]["detail"] = (str(issue_map[issue_id].get("detail") or "") + " | " + detail).strip(" |")
            return
        default_penalty = 28 if severity == "fail" else 13 if severity == "repairable" else 4
        score -= int(default_penalty if penalty is None else penalty)
        issue_map[issue_id] = {
            "id": issue_id,
            "severity": severity,
            "detail": detail,
            **({"sections": sections} if sections else {}),
        }

    if instrumental or lyric_text.strip().lower() == "[instrumental]":
        return {
            "version": LYRIC_CRAFT_GATE_VERSION,
            "status": "pass",
            "gate_passed": True,
            "score": 100,
            "issues": [],
            "issue_ids": [],
            "weak_sections": [],
            "contract": contract,
            "stats": stats,
        }
    if not lyric_text.strip() or not lines:
        add_issue("lyric_craft_missing_lyrics", "fail", "vocal track has no lyric lines", penalty=45)
    if stats["placeholder_count"]:
        add_issue("lyric_craft_placeholder", "fail", f"{stats['placeholder_count']} placeholder marker(s)", penalty=40)
    if stats["fallback_artifact_count"]:
        add_issue("lyric_craft_fallback_artifact", "fail", f"{stats['fallback_artifact_count']} fallback artifact(s)", penalty=40)
    if stats["meta_leak_lines"]:
        add_issue("lyric_craft_metadata_leakage", "fail", f"{len(stats['meta_leak_lines'])} metadata line(s)", penalty=40)
    if not partial and family != "sparse_or_instrumental" and stats["hook_count"] < 1:
        add_issue("lyric_craft_hook_weak", "fail", "vocal lyric has no hook/chorus/refrain section", penalty=35)
    if stats["line_count"] >= 24 and stats["unique_line_ratio"] < 0.38:
        add_issue("lyric_craft_extreme_repetition", "fail", f"{stats['unique_line_ratio']} unique line ratio", penalty=35)

    generic_hits = _regex_hits(GENERIC_AI_LYRIC_RE, lyric_text)
    generic_total = len(GENERIC_AI_LYRIC_RE.findall(lyric_text))
    if generic_hits and (generic_total >= 2 or len(generic_hits) >= 2):
        add_issue(
            "lyric_craft_generic_ai_phrase",
            "repairable",
            ", ".join(generic_hits[:6]),
            sections=[item["section"] for item in _weak_craft_sections(blocks, family) if "generic_ai_phrase" in item["reasons"]],
            penalty=20,
        )
    adjective_hits = _regex_hits(ADJECTIVE_STACK_RE, lyric_text)
    if adjective_hits:
        add_issue("lyric_craft_adjective_stacking", "repairable", ", ".join(adjective_hits[:4]), penalty=13)

    domain_hits = _metaphor_domain_hits(lyric_text)
    if not contract.get("allow_surreal_or_abstract") and len(domain_hits) >= 4:
        add_issue(
            "lyric_craft_mixed_metaphor",
            "repairable",
            ", ".join(f"{key}:{'/'.join(value[:3])}" for key, value in list(domain_hits.items())[:5]),
            penalty=15,
        )

    if lines:
        concrete_count = sum(1 for line in lines if CONCRETE_SCENE_RE.search(line) or ACTION_VERB_RE.search(line))
        concrete_ratio = round(concrete_count / max(1, len(lines)), 2)
        abstract_total = len(ABSTRACT_SLOGAN_RE.findall(lyric_text))
        if len(lines) >= 8 and concrete_ratio < 0.18 and (abstract_total >= max(6, len(lines) // 3) or generic_hits):
            add_issue("lyric_craft_no_concrete_scene", "repairable", f"concrete_ratio={concrete_ratio}", penalty=15)
        long_limit = 14 if family == "rap" else 13
        avg_limit = 10.8 if family == "rap" else 9.8
        long_lines = [line for line in lines if len(_words(line)) > long_limit]
        avg_words = round(sum(len(_words(line)) for line in lines) / max(1, len(lines)), 2)
        if long_lines and (avg_words > avg_limit or len(long_lines) > max(2, len(lines) // 4)):
            add_issue("lyric_craft_line_breathability", "repairable", f"avg_words={avg_words}; long_lines={len(long_lines)}", penalty=15)

    if not partial and family != "sparse_or_instrumental":
        hook_lines = _hook_section_lines(blocks)
        hook_text = " ".join(hook_lines).lower()
        title_words = [
            word.lower()
            for word in _words(str(payload.get("title") or payload.get("locked_title") or ""))
            if len(word) > 3
        ]
        promise_words = [
            word.lower()
            for word in _words(" ".join(str(payload.get(key) or "") for key in ("hook_promise", "description", "narrative")))
            if len(word) > 4
        ]
        anchor_words = set(title_words[:4] + promise_words[:6])
        has_anchor = bool(anchor_words and any(re.search(rf"\b{re.escape(word)}\b", hook_text) for word in anchor_words))
        emotional_or_concrete = bool(
            re.search(r"\b(?:want|need|home|truth|free|lost|found|love|hurt|blood|name|remember|promise|cost|alive)\b", hook_text)
            or any(CONCRETE_SCENE_RE.search(line) or ACTION_VERB_RE.search(line) for line in hook_lines)
        )
        if hook_lines:
            hook_generic = len(GENERIC_AI_LYRIC_RE.findall("\n".join(hook_lines)))
            if (
                (len(hook_lines) <= 1 and int(float(duration or 0)) >= 90)
                or (hook_generic and not has_anchor)
                or (len(hook_lines) <= 2 and not has_anchor and not emotional_or_concrete)
            ):
                add_issue("lyric_craft_hook_weak", "repairable", "hook lacks title/emotional promise or concrete anchor", sections=[block["tag"] for block in blocks if re.search(r"chorus|hook|refrain", str(block.get("tag") or ""), re.I)], penalty=18)

    if family == "rap" and len(lines) >= 12:
        line_endings = [(_words(line)[-1].lower()[-3:] if _words(line) else "") for line in lines]
        repeated_endings = sum(1 for _ending, count in Counter(item for item in line_endings if item).items() if count > 1)
        internal_rhyme_markers = sum(1 for line in lines if re.search(r"\b\w+(?:ing|ight|old|one|own|all|ack|ore)\b.*\b\w+(?:ing|ight|old|one|own|all|ack|ore)\b", line, re.I))
        if repeated_endings < 2 and internal_rhyme_markers < 2 and (generic_hits or stats["unique_line_ratio"] < 0.58):
            add_issue("lyric_craft_rhyme_chaos", "repairable", "rap section lacks audible rhyme momentum", penalty=10)

    weak_sections = _weak_craft_sections(blocks, family)
    for issue in issue_map.values():
        for section in issue.get("sections") or []:
            if not any(item.get("section") == section for item in weak_sections):
                weak_sections.append({"section": section, "reasons": [str(issue.get("id"))], "preview": ""})
    if score < 65:
        add_issue("lyric_craft_low_score", "fail", f"score={max(0, score)}", penalty=0)
    issues = list(issue_map.values())
    score = max(0, min(100, score))
    hard_fail = any(issue.get("severity") == "fail" for issue in issues)
    if hard_fail or score < 65:
        status = "fail"
    elif issues or score < int(contract.get("pass_score") or 82):
        status = "repair_needed"
    else:
        status = "pass"
    return {
        "version": LYRIC_CRAFT_GATE_VERSION,
        "status": status,
        "gate_passed": status == "pass",
        "score": score,
        "issues": issues,
        "issue_ids": [str(issue.get("id")) for issue in issues],
        "weak_sections": weak_sections,
        "contract": contract,
        "stats": {
            **stats,
            "metaphor_domains": domain_hits,
            "generic_phrase_hits": generic_hits,
        },
        "plan": plan or {},
    }


def lyric_density_gate(
    lyrics: str,
    plan: dict[str, Any],
    *,
    duration: float,
    genre_hint: str = "",
    instrumental: bool = False,
) -> dict[str, Any]:
    stats = _lyric_stats(lyrics)
    section_detail = _lyric_section_line_counts(lyrics)
    issues: list[dict[str, Any]] = []
    if instrumental:
        return {
            "version": "lyric-density-gate-2026-05-02",
            "status": "pass",
            "gate_passed": True,
            "issues": [],
            "plan": plan,
            "stats": stats,
            "section_line_counts": section_detail["counts"],
        }
    dur = int(float(duration or 0))
    rap = bool(re.search(r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b", genre_hint or "", re.I))
    target_words = int(plan.get("target_words") or 0)
    target_lines = int(plan.get("target_lines") or 0)
    # Density floors loosened from 82% → 70%. The plan's target_lines is
    # an aspirational hit-density figure; 70% matches the per-section
    # minimums (3 verses × 16 + 4 hooks + bridge + intros = ~72 lines for
    # a 180s rap, which is target_lines ≈ 96 × 70%). 82% caused tracks
    # with full content to fail just because the planner aimed high.
    if target_words and stats["word_count"] < int(target_words * 0.70):
        issues.append({
            "id": "lyrics_under_hit_density",
            "severity": "fail",
            "detail": f"{stats['word_count']}/{target_words} target words",
        })
    if target_lines and stats["line_count"] < int(target_lines * 0.70):
        issues.append({
            "id": "lyrics_under_hit_line_density",
            "severity": "fail",
            "detail": f"{stats['line_count']}/{target_lines} target lines",
        })
    counts = section_detail["counts"]
    hook_counts = {
        section: count
        for section, count in counts.items()
        if re.search(r"chorus|hook|refrain", section, re.I)
    }
    if stats["hook_count"] < 1:
        issues.append({"id": "hook_missing", "severity": "fail", "detail": "no hook/chorus/refrain section"})
        issues.append({"id": "missing_hook", "severity": "fail", "detail": "no hook/chorus/refrain section"})
    elif dur >= 120 and hook_counts and max(hook_counts.values()) < 2:
        issues.append({"id": "hook_underwritten", "severity": "fail", "detail": f"hook section lines={hook_counts}"})
    if rap and dur >= 180:
        verse_counts = {
            section: count
            for section, count in counts.items()
            if re.search(r"verse", section, re.I)
        }
        verse_total = sum(verse_counts.values())
        min_total = 36 if dur >= 240 else 28
        min_each = 8 if dur >= 240 else 6
        short_verses = {section: count for section, count in verse_counts.items() if count < min_each}
        if verse_total < min_total or short_verses:
            detail = f"verse_total={verse_total}/{min_total}; short_verses={short_verses}"
            issues.append({"id": "rap_verses_underfilled", "severity": "fail", "detail": detail})
    if stats["line_count"] >= 48 and stats["unique_line_ratio"] < (0.50 if rap else 0.42):
        issues.append({
            "id": "lyric_unique_line_ratio_low",
            "severity": "fail",
            "detail": f"{stats['unique_line_ratio']} unique line ratio",
        })
    if str(lyrics or "").lower().count("final chorus - reprise") > 1:
        issues.append({"id": "fallback_reprise_overuse", "severity": "fail", "detail": "too many deterministic reprise blocks"})
    return {
        "version": "lyric-density-gate-2026-05-02",
        "status": "pass" if not issues else "fail",
        "gate_passed": not issues,
        "issues": issues,
        "plan": plan,
        "stats": stats,
        "section_line_counts": counts,
    }


def _duplicate_section_keys(sections: list[str]) -> dict[str, int]:
    counts = Counter(_section_key(section) for section in sections if section)
    return {key: count for key, count in counts.items() if count > 1}


def _lyric_terminal_fragment_lines(lyrics: str) -> list[str]:
    lines = [_normalize_lyric_section_line(line).strip() for line in str(lyrics or "").splitlines() if line.strip()]
    lyric_lines = [line for line in lines if not _section_only_line(line)]
    suspect: list[str] = []
    tail = lyric_lines[-10:]
    connector_re = re.compile(
        r"\b(?:a|an|the|to|through|with|while|when|where|and|but|or|from|of|in|on|under|over|into|for|by|as|that|this)\s*$",
        re.I,
    )
    for line in tail:
        clean = re.sub(r"[\"'“”‘’]+", "", line).strip()
        words = WORD_RE.findall(clean)
        if not words:
            continue
        if len(words[-1]) <= 1 and re.search(r"[A-Za-zÀ-ÿ]", words[-1]):
            suspect.append(line)
            continue
        if len(words) <= 2 and connector_re.search(clean):
            suspect.append(line)
            continue
        if connector_re.search(clean) and not re.search(r"[,.!?…)]\s*$", clean):
            suspect.append(line)
    return suspect


def _lyric_ending_is_complete(lyrics: str) -> bool:
    lines = [_normalize_lyric_section_line(line).strip() for line in str(lyrics or "").splitlines() if line.strip()]
    lyric_lines = [line for line in lines if not _section_only_line(line)]
    if not lyric_lines:
        return False
    if _lyric_terminal_fragment_lines(lyrics):
        return False
    tail = lyric_lines[-1]
    words = WORD_RE.findall(tail)
    return len(words) >= 2


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
    if len(pieces) <= 1 and len(words) >= max(target_words + 1, 7):
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


def _protected_phrase_line(line: str, protected_phrases: list[str]) -> bool:
    if not protected_phrases:
        return False
    normalized_line = _phrase_norm(line)
    if not normalized_line:
        return False
    for phrase in protected_phrases:
        normalized_phrase = _phrase_norm(phrase)
        if normalized_phrase and (normalized_phrase in normalized_line or normalized_line in normalized_phrase):
            return True
    return False


def _reflow_lyrics_to_min_lines(
    lyrics: str,
    min_lines: int,
    max_chars: int = ACE_STEP_LYRICS_CHAR_LIMIT,
    protected_phrases: list[str] | None = None,
) -> tuple[str, bool]:
    """Increase line count by splitting long lines, without adding filler words."""
    text = str(lyrics or "").strip()
    stats = _lyric_stats(text)
    target = int(min_lines or 0)
    if not text or stats["line_count"] >= target:
        return text, False
    deficit = target - stats["line_count"]
    target_words = 6
    output: list[str] = []
    changed = False
    protected = [str(item) for item in (protected_phrases or []) if str(item).strip()]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if deficit <= 0 or _section_only_line(line) or not line.strip() or _protected_phrase_line(line, protected):
            output.append(line)
            continue
        words = _words(line)
        if len(words) <= target_words + 1 and len(line) < 48:
            output.append(line)
            continue
        chunks = _line_word_chunks(line, target_words=target_words)
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


def _extract_reprise_lines(lyrics: str, max_lines: int = 8) -> list[str]:
    lines = [_normalize_lyric_section_line(line).strip() for line in str(lyrics or "").splitlines()]
    collected: list[str] = []
    in_hook = False
    for line in lines:
        if not line:
            if in_hook and collected:
                break
            continue
        section = SECTION_RE.fullmatch(line)
        if section:
            key = _section_key(section.group(1))
            if any(token in key for token in ("chorus", "hook", "refrain")):
                in_hook = True
                continue
            if in_hook and collected:
                break
            in_hook = False
            continue
        if in_hook and not META_LEAK_LINE_RE.search(_metadata_probe(line)):
            collected.append(line)
            if len(collected) >= max_lines:
                break
    return collected


def _extend_lyrics_to_min_words(
    lyrics: str,
    payload: dict[str, Any],
    min_words: int,
    max_chars: int = ACE_STEP_LYRICS_CHAR_LIMIT,
) -> tuple[str, bool]:
    """Fix small word-count misses by reusing musical material, not filler."""
    text = str(lyrics or "").strip()
    stats = _lyric_stats(text)
    missing = int(min_words or 0) - int(stats.get("word_count") or 0)
    if not text or missing <= 0:
        return text, False
    if missing > 36:
        return text, False
    reprise_lines = _extract_reprise_lines(text, max_lines=8)
    extension_lines: list[str] = []
    added_words = 0
    for line in reprise_lines:
        clean = re.sub(r"\s+", " ", str(line or "")).strip()
        if not clean or _section_only_line(clean):
            continue
        extension_lines.append(clean)
        added_words += len(_words(clean))
        if added_words >= missing:
            break
        prospective = (text.rstrip() + "\n\n[Final Chorus - reprise]\n" + "\n".join(extension_lines)).strip()
        if len(prospective) > max_chars:
            extension_lines.pop()
            break
    if added_words < missing or not extension_lines:
        return text, False
    block = "\n\n[Final Chorus - reprise]\n" + "\n".join(extension_lines)
    repaired = (text.rstrip() + block).strip()
    if len(repaired) <= max_chars and _lyric_stats(repaired)["word_count"] >= int(min_words or 0):
        return repaired, True
    return text, False


def _density_repair_bars(payload: dict[str, Any]) -> list[str]:
    title_token = re.sub(r"[^A-Za-z0-9 ]+", " ", str((payload or {}).get("title") or "pressure")).strip().split()
    motif = title_token[0].lower() if title_token else "pressure"
    return [
        f"{motif.title()} moves heavy where the kick drum lands",
        "Bassline talks clear while the streetlight bends",
        "Every bar cuts sharper than the polished plans",
        "Cold doors open when the hook comes in",
        "Low end rolling with the truth in front",
        "Snare cracks clean through the crowded room",
        "Piano loop flickers like a warning sign",
        "Clear rap cadence keeps the whole block tuned",
        "Dust on the sample but the pocket stays tight",
        "Hands stay steady when the pressure climbs",
        "Deep drums answer every quiet doubt",
        "Words hit forward with the bass locked down",
        "City glass shakes when the chorus lifts",
        "Street voice centered in the modern mix",
        "No filler lines when the stakes get high",
        "Every breath lands where the sirens fly",
    ]


def _repair_lyrics_density_sections(
    lyrics: str,
    payload: dict[str, Any],
    plan: dict[str, Any],
    *,
    duration: float,
    genre_hint: str,
    max_chars: int = ACE_STEP_LYRICS_CHAR_LIMIT,
) -> tuple[str, bool, str]:
    text = str(lyrics or "").strip()
    if not text:
        return text, False, ""
    lines = [_normalize_lyric_section_line(line).strip() for line in text.splitlines() if str(line).strip()]
    tags = [line for line in lines if _section_only_line(line)]
    if not tags:
        return text, False, ""
    rap = bool(re.search(r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b", genre_hint or "", re.I))
    density_gate = lyric_density_gate(text, plan, duration=duration, genre_hint=genre_hint, instrumental=False)
    issue_ids = {str(issue.get("id")) for issue in (density_gate.get("issues") or []) if isinstance(issue, dict)}
    stats = _lyric_stats(text)
    target_words = int(plan.get("target_words") or 0)
    target_lines = int(plan.get("target_lines") or 0)
    min_words = max(int(plan.get("min_words") or 0), int(target_words * 0.82))
    min_lines = max(int(plan.get("min_lines") or 0), int(target_lines * 0.82))
    if stats["word_count"] < max(24, int(min_words * 0.20)) or stats["line_count"] < 6:
        return text, False, ""
    if not (
        issue_ids
        & {
            "lyrics_under_hit_density",
            "lyrics_under_hit_line_density",
            "rap_verses_underfilled",
            "hook_underwritten",
        }
        or stats["word_count"] < min_words
        or stats["line_count"] < min_lines
    ):
        return text, False, ""
    bars = _density_repair_bars(payload)
    used = {line.casefold() for line in lines}
    added: list[dict[str, str]] = []
    bar_index = 0

    def _counts() -> dict[str, int]:
        result: dict[str, int] = {}
        active = ""
        for item in lines:
            if _section_only_line(item):
                active = item
                result.setdefault(active, 0)
            elif active:
                result[active] = result.get(active, 0) + 1
        return result

    def _next_bar() -> str:
        nonlocal bar_index
        for _ in range(len(bars) * 3):
            bar = bars[bar_index % len(bars)]
            bar_index += 1
            if bar.casefold() not in used:
                used.add(bar.casefold())
                return bar
        bar = f"Pressure line lands clean in pocket {bar_index}"
        bar_index += 1
        used.add(bar.casefold())
        return bar

    def _insert(section: str, line: str) -> bool:
        try:
            start = lines.index(section)
        except ValueError:
            return False
        insert_at = len(lines)
        for pos in range(start + 1, len(lines)):
            if _section_only_line(lines[pos]):
                insert_at = pos
                break
        candidate = lines[:insert_at] + [line] + lines[insert_at:]
        if len("\n".join(candidate)) > max_chars:
            return False
        lines[:] = candidate
        added.append({"section": section, "line": line})
        return True

    preferred = [tag for tag in tags if "verse" in tag.lower()]
    if not preferred:
        preferred = [tag for tag in tags if not re.search(r"intro|outro|break|instrumental", tag, re.I)] or tags
    if rap and int(duration or 0) >= 180 and preferred:
        minimum_each = 8
        total_min = 36 if int(duration or 0) >= 240 else 28
        counts = _counts()
        for section in preferred:
            while counts.get(section, 0) < minimum_each:
                if not _insert(section, _next_bar()):
                    break
                counts = _counts()
        while sum(_counts().get(section, 0) for section in preferred) < total_min:
            section = preferred[len(added) % len(preferred)]
            if not _insert(section, _next_bar()):
                break
    stats = _lyric_stats("\n".join(lines))
    round_index = 0
    while (stats["word_count"] < min_words or stats["line_count"] < min_lines) and round_index < 96:
        section = preferred[round_index % len(preferred)]
        if not _insert(section, _next_bar()):
            break
        stats = _lyric_stats("\n".join(lines))
        round_index += 1
    if not added:
        return text, False, ""
    return "\n".join(lines).strip(), True, f"lyric_density_sections_extended:{len(added)}"


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


def build_producer_grade_sonic_contract(payload: dict[str, Any] | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = {**(options or {}), **(payload or {})}
    genre_text = " ".join(
        str(merged.get(key) or "")
        for key in (
            "caption",
            "tags",
            "style",
            "vibe",
            "genre_prompt",
            "album_agent_genre_prompt",
            "album_agent_vocal_type",
            "custom_tags",
        )
    )
    rap_locked = bool(re.search(r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b", genre_text, re.I))
    required = list(PRODUCER_GRADE_DIMENSION_KEYWORDS)
    suggested_terms = dict(PRODUCER_GRADE_REPAIR_TERMS)
    if rap_locked:
        suggested_terms.update(
            {
                "primary_genre": "rap-dominant hip-hop",
                "drum_groove": "boom-bap or trap drum pocket",
                "low_end_bass": "808 bass or deep low-end",
                "melodic_identity": "piano sample, synth motif, or soul chop",
                "vocal_delivery": "clear male rap vocal pocket",
                "arrangement_movement": "hook response and beat switch movement",
                "texture_space": "gritty street texture with controlled space",
                "mix_master": "punchy crisp modern rap mix",
            }
        )
    return {
        "version": "producer-grade-sonic-contract-2026-05-02",
        "required_dimensions": required,
        "rap_locked": rap_locked,
        "suggested_terms": suggested_terms,
    }


def sonic_dna_coverage(
    caption: str,
    tag_list: Any = None,
    *,
    payload: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = build_producer_grade_sonic_contract(payload, options)
    combined = " ".join([str(caption or ""), " ".join(_split_terms(tag_list))]).lower()
    dimensions: list[dict[str, Any]] = []
    missing: list[str] = []
    for dimension in contract["required_dimensions"]:
        keywords = PRODUCER_GRADE_DIMENSION_KEYWORDS.get(dimension, ())
        matched = sorted({keyword for keyword in keywords if keyword in combined})
        status = "pass" if matched else "missing"
        dimensions.append({"dimension": dimension, "status": status, "matched": matched[:8]})
        if status == "missing":
            missing.append(dimension)
    return {
        "version": "sonic-dna-coverage-2026-05-02",
        "status": "pass" if not missing else "repair_needed",
        "contract": contract,
        "dimensions": dimensions,
        "missing": missing,
    }


def producer_grade_readiness(
    payload: dict[str, Any],
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = sonic_dna_coverage(
        str((payload or {}).get("caption") or (payload or {}).get("tags") or ""),
        (payload or {}).get("tag_list"),
        payload=payload,
        options=options,
    )
    missing = list(coverage.get("missing") or [])
    score = max(0, 100 - 13 * len(missing))
    return {
        "version": "producer-grade-readiness-2026-05-02",
        "status": "pass" if not missing else "fail",
        "gate_passed": not missing,
        "score": score,
        "sonic_dna_coverage": coverage,
        "issues": [
            {
                "id": f"producer_grade_missing_{dimension}",
                "severity": "repairable",
                "detail": PRODUCER_GRADE_REPAIR_TERMS.get(dimension, dimension),
            }
            for dimension in missing
        ],
    }


def _clean_caption(payload: dict[str, Any], coverage: dict[str, Any]) -> str:
    terms: list[str] = []
    for value in [payload.get("style"), payload.get("vibe"), payload.get("caption"), payload.get("tag_list")]:
        for term in _split_terms(value):
            if not _caption_term_allowed(term):
                continue
            if term.lower() not in {existing.lower() for existing in terms}:
                terms.append(term)
    terms = _prune_caption_terms(terms)
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
                if _caption_term_allowed(term) and term.lower() not in {item.lower() for item in terms}:
                    terms.append(term)
    terms = _prune_caption_terms(terms)
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
    caption_terms_for_check = [
        term
        for term in [*_split_terms(caption), *tag_terms]
        if _caption_term_allowed(term)
    ]
    genre_terms = []
    for term in caption_terms_for_check:
        if _caption_term_is_genre(term) and term.lower() not in {item.lower() for item in genre_terms}:
            genre_terms.append(term)
    if caption_leaks or tag_leaks:
        issues.append({"id": "caption_leakage", "severity": "repairable", "detail": f"{len(caption_leaks) + len(tag_leaks)} leak marker(s)"})
    if len(genre_terms) > 2:
        issues.append({"id": "caption_genre_overload", "severity": "repairable", "detail": ", ".join(genre_terms[:6])})
    if coverage.get("missing"):
        issues.append({"id": "tag_dimension_coverage", "severity": "repairable", "detail": ", ".join(coverage["missing"])})
    if repair and (caption_leaks or tag_leaks or coverage.get("missing") or len(genre_terms) > 2):
        if tag_leaks or len(genre_terms) > 2:
            repaired["tag_list"] = _prune_caption_terms([
                term for term in tag_terms if _caption_term_allowed(term) and not _caption_has_leakage(term)
            ])
            tag_list = repaired["tag_list"]
            coverage_source_caption = "" if (caption_leaks or tag_leaks or len(genre_terms) > 2) else caption
            coverage = tag_dimension_coverage(coverage_source_caption, tag_list)
        repaired_caption = _clean_caption(repaired, coverage)
        repaired["caption"] = repaired_caption
        repaired["tags"] = repaired_caption
        caption = repaired_caption
        repair_actions.append("caption_rebuilt_from_tag_dimensions")
        repaired["tag_list"] = _split_terms(tag_list) + [
            term for term in _split_terms(repaired_caption) if term.lower() not in {item.lower() for item in _split_terms(tag_list)}
        ]
        tag_list = repaired["tag_list"]
        coverage = tag_dimension_coverage(repaired_caption, tag_list)
        if coverage.get("missing"):
            issues.append({"id": "tag_dimension_coverage_unrepaired", "severity": "fail", "detail": ", ".join(coverage["missing"])})

    producer_readiness = producer_grade_readiness(repaired, options=opts)
    producer_missing = list((producer_readiness.get("sonic_dna_coverage") or {}).get("missing") or [])
    if producer_missing:
        for issue in producer_readiness.get("issues") or []:
            issues.append(dict(issue))
        if repair:
            terms = _split_terms(repaired.get("tag_list") or tag_list)
            for dimension in producer_missing:
                repair_term = (producer_readiness.get("sonic_dna_coverage", {}).get("contract", {}).get("suggested_terms", {}) or {}).get(
                    dimension,
                    PRODUCER_GRADE_REPAIR_TERMS.get(dimension, ""),
                )
                if repair_term and repair_term.lower() not in {term.lower() for term in terms}:
                    terms.append(repair_term)
            repaired["tag_list"] = terms
            caption_terms = _prune_caption_terms(terms)
            repaired["caption"] = ", ".join(caption_terms[:18])[:ACE_STEP_CAPTION_CHAR_LIMIT]
            repaired["tags"] = repaired["caption"]
            caption = repaired["caption"]
            tag_list = repaired["tag_list"]
            repair_actions.append("caption_rebuilt_from_producer_grade_sonic_dna")
            producer_readiness = producer_grade_readiness(repaired, options=opts)
            producer_missing = list((producer_readiness.get("sonic_dna_coverage") or {}).get("missing") or [])
            if producer_missing:
                issues.append({
                    "id": "producer_grade_sonic_dna_unrepaired",
                    "severity": "fail",
                    "detail": ", ".join(producer_missing),
                })

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
    elif global_caption and repair:
        caption_keys = {term.lower() for term in _split_terms(caption)}
        global_terms = _split_terms(global_caption)
        deduped_terms = [term for term in global_terms if term.lower() not in caption_keys]
        if len(deduped_terms) < len(global_terms):
            repaired["global_caption"] = ", ".join(deduped_terms[:8])
            repair_actions.append("global_caption_deduped_against_track_caption")

    duration = float(repaired.get("duration") or opts.get("track_duration") or 180)
    density = str(repaired.get("lyric_density") or opts.get("lyric_density") or "dense")
    structure_preset = str(repaired.get("structure_preset") or opts.get("structure_preset") or "auto")
    genre_hint = " ".join(
        str(repaired.get(key) or "")
        for key in ("caption", "description", "style", "vibe", "narrative")
    )
    plan = _lyric_plan(duration, density, structure_preset, genre_hint)
    lyrics = str(repaired.get("lyrics") or "")
    producer_credit = (
        str(repaired.get("producer_credit") or "")
        or str(((repaired.get("album_metadata") or {}).get("producer_credit") if isinstance(repaired.get("album_metadata"), dict) else "") or "")
    )
    producer_aliases = _producer_credit_aliases(producer_credit)
    producer_hits = _lyrics_contain_any(lyrics, producer_aliases)
    if producer_hits:
        issues.append({"id": "producer_credit_in_lyrics", "severity": "repairable", "detail": ", ".join(producer_hits)})
        if repair:
            repaired_lyrics, changed = _repair_producer_credit_lyrics(lyrics, producer_hits)
            if changed:
                lyrics = repaired_lyrics
                repaired["lyrics"] = lyrics
                repair_actions.append("producer_credit_removed_from_lyrics")
    if repair and "\\n" in lyrics and lyrics.count("\\n") >= 3:
        lyrics = lyrics.replace("\\r\\n", "\n").replace("\\n", "\n")
        repaired["lyrics"] = lyrics
        repair_actions.append("lyrics_unescaped_newlines")
    has_vocal_script = bool(lyrics.strip() and lyrics.strip().lower() != "[instrumental]")
    instrumental = str(lyrics).strip().lower() == "[instrumental]" or (bool(repaired.get("instrumental")) and not has_vocal_script)
    stats = _lyric_stats(lyrics)
    expected_keys = {_section_key(section) for section in plan.get("sections") or [] if section}
    actual_keys = {_section_key(section) for section in stats["sections"] if section}
    section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0

    if not instrumental:
        min_words = int(plan.get("min_words") or 0)
        raw_min_lines = int(plan.get("min_lines") or 0)
        min_lines = _effective_min_lines(raw_min_lines, min_words)
        if raw_min_lines and min_lines != raw_min_lines:
            plan = dict(plan)
            plan["raw_min_lines"] = raw_min_lines
            plan["effective_min_lines"] = min_lines
        required_phrases = [str(item).strip() for item in (repaired.get("required_phrases") or []) if str(item).strip()]
        missing_required = [phrase for phrase in required_phrases if not _phrase_present(phrase, lyrics)]
        can_extend_near_miss = (
            stats["word_count"] >= max(80, int(min_words * 0.65))
            and stats["fallback_artifact_count"] == 0
            and not stats["meta_leak_lines"]
            and stats["placeholder_count"] == 0
        )
        if missing_required and repair:
            append_block = "\n\n[Required Hook]\n" + "\n".join(missing_required[:6])
            if len(lyrics) + len(append_block) <= ACE_STEP_LYRICS_CHAR_LIMIT:
                repaired["lyrics"] = (lyrics.rstrip() + append_block).strip()
                lyrics = str(repaired["lyrics"])
                stats = _lyric_stats(lyrics)
                actual_keys = {_section_key(section) for section in stats["sections"] if section}
                section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0
                repair_actions.append("missing_required_phrases_appended")
        if repair and can_extend_near_miss and stats["word_count"] < min_words and stats["char_count"] <= ACE_STEP_LYRICS_CHAR_LIMIT:
            extended, changed = _extend_lyrics_to_min_words(lyrics, repaired, min_words, ACE_STEP_LYRICS_CHAR_LIMIT)
            if changed:
                old_words = stats["word_count"]
                repaired["lyrics"] = extended
                lyrics = extended
                stats = _lyric_stats(lyrics)
                actual_keys = {_section_key(section) for section in stats["sections"] if section}
                section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0
                repair_actions.append(f"lyrics_extended_to_min_words:{old_words}->{stats['word_count']}")
        if repair and stats["line_count"] < min_lines and stats["char_count"] <= ACE_STEP_LYRICS_CHAR_LIMIT:
            reflowed, changed = _reflow_lyrics_to_min_lines(lyrics, min_lines, ACE_STEP_LYRICS_CHAR_LIMIT, required_phrases)
            if changed:
                old_lines = stats["line_count"]
                repaired["lyrics"] = reflowed
                lyrics = reflowed
                stats = _lyric_stats(lyrics)
                actual_keys = {_section_key(section) for section in stats["sections"] if section}
                section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0
                repair_actions.append(f"lyrics_reflowed_to_min_lines:{old_lines}->{stats['line_count']}")
        can_extend_near_miss = (
            stats["word_count"] >= max(80, int(min_words * 0.65))
            and stats["fallback_artifact_count"] == 0
            and not stats["meta_leak_lines"]
            and stats["placeholder_count"] == 0
        )
        if repair and can_extend_near_miss and stats["word_count"] < min_words and stats["char_count"] <= ACE_STEP_LYRICS_CHAR_LIMIT:
            extended, changed = _extend_lyrics_to_min_words(lyrics, repaired, min_words, ACE_STEP_LYRICS_CHAR_LIMIT)
            if changed:
                old_words = stats["word_count"]
                repaired["lyrics"] = extended
                lyrics = extended
                stats = _lyric_stats(lyrics)
                actual_keys = {_section_key(section) for section in stats["sections"] if section}
                section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0
                repair_actions.append(f"lyrics_extended_to_min_words_after_reflow:{old_words}->{stats['word_count']}")
        if not has_vocal_lyrics(lyrics):
            issues.append({"id": "lyrics_missing", "severity": "fail", "detail": "vocal track has no lyrics"})
        if stats["char_count"] > ACE_STEP_LYRICS_CHAR_LIMIT:
            issues.append({"id": "lyrics_over_budget", "severity": "fail", "detail": f"{stats['char_count']}/{ACE_STEP_LYRICS_CHAR_LIMIT} chars"})
        elif stats["char_count"] > ACE_STEP_LYRICS_WARNING_CHAR_LIMIT:
            issues.append({"id": "lyrics_near_budget_limit", "severity": "warning", "detail": f"{stats['char_count']}/{ACE_STEP_LYRICS_CHAR_LIMIT} chars"})
        lyric_repaired = any(str(action).startswith(("lyric", "lyrics_")) for action in repair_actions)
        if stats["char_count"] > ACE_STEP_LYRICS_CHAR_LIMIT - ACE_STEP_LYRICS_SAFE_HEADROOM and lyric_repaired:
            issues.append({
                "id": "unsafe_budget_margin",
                "severity": "fail",
                "detail": f"{ACE_STEP_LYRICS_CHAR_LIMIT - stats['char_count']} chars headroom after repair",
            })
        duplicate_sections = _duplicate_section_keys(stats["sections"])
        if duplicate_sections.get("outro", 0) > 1:
            issues.append({"id": "duplicate_outro", "severity": "fail", "detail": f"{duplicate_sections['outro']} outro sections"})
        terminal_fragments = _lyric_terminal_fragment_lines(lyrics)
        if terminal_fragments:
            issues.append({"id": "lyrics_mid_line_truncation", "severity": "fail", "detail": " | ".join(terminal_fragments[:3])})
        if not _lyric_ending_is_complete(lyrics):
            issues.append({"id": "ending_not_complete", "severity": "fail", "detail": "last lyric line is incomplete or missing"})
        if stats["word_count"] < int(plan.get("min_words") or 0):
            issues.append({"id": "lyrics_under_length", "severity": "fail", "detail": f"{stats['word_count']}/{plan.get('min_words')} words"})
        if stats["line_count"] < min_lines:
            issues.append({"id": "lyrics_too_few_lines", "severity": "fail", "detail": f"{stats['line_count']}/{min_lines} lines"})
        if section_coverage < 0.72:
            issues.append({"id": "section_coverage_low", "severity": "fail", "detail": f"{section_coverage} coverage"})
        if stats["hook_count"] < 1:
            issues.append({"id": "hook_missing", "severity": "fail", "detail": "no chorus/hook/refrain section"})
            issues.append({"id": "missing_hook", "severity": "fail", "detail": "no chorus/hook/refrain section"})
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

    if repair and not instrumental and lyrics.strip():
        density_repaired, changed, action = _repair_lyrics_density_sections(
            lyrics,
            repaired,
            plan,
            duration=duration,
            genre_hint=genre_hint,
            max_chars=ACE_STEP_LYRICS_CHAR_LIMIT - ACE_STEP_LYRICS_SAFE_HEADROOM,
        )
        if changed:
            repaired["lyrics"] = density_repaired
            lyrics = density_repaired
            stats = _lyric_stats(lyrics)
            actual_keys = {_section_key(section) for section in stats["sections"] if section}
            section_coverage = round((len(expected_keys & actual_keys) / max(1, len(expected_keys))), 2) if expected_keys else 1.0
            resolved = {
                "lyrics_under_hit_density",
                "lyrics_under_hit_line_density",
                "rap_verses_underfilled",
                "hook_underwritten",
            }
            if stats["word_count"] >= int(plan.get("min_words") or 0):
                resolved.add("lyrics_under_length")
            if stats["line_count"] >= min_lines:
                resolved.add("lyrics_too_few_lines")
            issues = [issue for issue in issues if str(issue.get("id")) not in resolved]
            if action:
                repair_actions.append(action)

    density_gate = lyric_density_gate(
        lyrics,
        plan,
        duration=duration,
        genre_hint=genre_hint,
        instrumental=instrumental,
    )
    existing_issue_ids = {str(issue.get("id")) for issue in issues}
    for issue in density_gate.get("issues") or []:
        if str(issue.get("id")) not in existing_issue_ids:
            issues.append(dict(issue))
            existing_issue_ids.add(str(issue.get("id")))

    craft_gate = lyric_craft_gate(
        lyrics,
        repaired,
        options=opts,
        plan=plan,
        duration=duration,
        genre_hint=genre_hint,
        instrumental=instrumental,
    )
    for issue in craft_gate.get("issues") or []:
        issue_id = str(issue.get("id") or "")
        if issue_id and issue_id not in existing_issue_ids:
            issues.append(dict(issue))
            existing_issue_ids.add(issue_id)
    if not instrumental and craft_gate.get("status") != "pass" and "lyric_craft_gate_failed" not in existing_issue_ids:
        issues.append({
            "id": "lyric_craft_gate_failed",
            "severity": "fail",
            "detail": ", ".join(craft_gate.get("issue_ids") or []) or str(craft_gate.get("score")),
        })
        existing_issue_ids.add("lyric_craft_gate_failed")
    repaired["lyrical_craft_contract"] = craft_gate.get("contract") or {}
    repaired["lyric_craft_gate"] = craft_gate
    repaired["lyric_craft_score"] = craft_gate.get("score")
    repaired["lyric_craft_issues"] = craft_gate.get("issue_ids") or []

    required_phrases = [str(item).strip() for item in (repaired.get("required_phrases") or []) if str(item).strip()]
    missing_required = [phrase for phrase in required_phrases if not _phrase_present(phrase, lyrics)]
    if missing_required and not instrumental:
        issues.append({"id": "required_phrases_missing", "severity": "fail", "detail": ", ".join(missing_required[:4])})

    genre_adherence = evaluate_genre_adherence(repaired, opts)
    for issue in genre_adherence.get("issues") or []:
        issues.append(dict(issue))
    repaired["genre_intent_contract"] = genre_adherence.get("contract") or {}
    repaired["genre_adherence"] = {key: value for key, value in genre_adherence.items() if key != "contract"}
    repaired["genre_validation_issues"] = genre_adherence.get("issue_ids") or []

    final_caption_leaks = _caption_has_leakage(str(repaired.get("caption") or ""))
    final_coverage = tag_dimension_coverage(str(repaired.get("caption") or ""), repaired.get("tag_list") or tag_list)
    if final_coverage.get("missing") and not any(issue.get("id") == "tag_dimension_coverage_unrepaired" for issue in issues):
        issues.append({"id": "tag_dimension_coverage_unrepaired", "severity": "fail", "detail": ", ".join(final_coverage["missing"])})
    elif not final_coverage.get("missing"):
        issues = [
            issue for issue in issues
            if str(issue.get("id")) not in {"tag_dimension_coverage", "tag_dimension_coverage_unrepaired"}
        ]
    coverage = final_coverage
    final_producer_readiness = producer_grade_readiness(repaired, options=opts)
    final_sonic_dna_coverage = final_producer_readiness.get("sonic_dna_coverage") or {}
    if final_sonic_dna_coverage.get("missing") and not any(issue.get("id") == "producer_grade_sonic_dna_unrepaired" for issue in issues):
        issues.append({
            "id": "producer_grade_sonic_dna_unrepaired",
            "severity": "fail",
            "detail": ", ".join(final_sonic_dna_coverage["missing"]),
        })
    elif not final_sonic_dna_coverage.get("missing"):
        issues = [
            issue for issue in issues
            if not str(issue.get("id")).startswith("producer_grade_missing_")
            and str(issue.get("id")) != "producer_grade_sonic_dna_unrepaired"
        ]
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
        "density_gate": density_gate,
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
        "lyric_density_gate": density_gate,
        "lyrical_craft_contract": craft_gate.get("contract") or {},
        "lyric_craft_gate": craft_gate,
        "lyric_craft_score": craft_gate.get("score"),
        "lyric_craft_issues": craft_gate.get("issue_ids") or [],
        "genre_intent_contract": genre_adherence.get("contract") or {},
        "genre_adherence": {key: value for key, value in genre_adherence.items() if key != "contract"},
        "producer_grade_sonic_contract": final_sonic_dna_coverage.get("contract") or {},
        "sonic_dna_coverage": final_sonic_dna_coverage,
        "producer_grade_readiness": final_producer_readiness,
        "repaired_payload": repaired,
    }
    repaired["payload_gate_status"] = status
    repaired["payload_quality_gate"] = {key: value for key, value in report.items() if key != "repaired_payload"}
    repaired["tag_coverage"] = coverage
    repaired["caption_integrity"] = caption_integrity
    repaired["lyric_duration_fit"] = lyric_duration_fit
    repaired["lyric_density_gate"] = density_gate
    repaired["lyrical_craft_contract"] = craft_gate.get("contract") or {}
    repaired["lyric_craft_gate"] = craft_gate
    repaired["lyric_craft_score"] = craft_gate.get("score")
    repaired["lyric_craft_issues"] = craft_gate.get("issue_ids") or []
    repaired["genre_intent_contract"] = genre_adherence.get("contract") or {}
    repaired["genre_adherence"] = {key: value for key, value in genre_adherence.items() if key != "contract"}
    repaired["producer_grade_sonic_contract"] = final_sonic_dna_coverage.get("contract") or {}
    repaired["sonic_dna_coverage"] = final_sonic_dna_coverage
    repaired["producer_grade_readiness"] = final_producer_readiness
    repaired["repair_actions"] = repair_actions
    warnings = list(repaired.get("payload_warnings") or [])
    if status != "pass":
        warnings.append(f"album_payload_gate:{status}")
    repaired["payload_warnings"] = warnings
    report["repaired_payload"] = repaired
    return report
