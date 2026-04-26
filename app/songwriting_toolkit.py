from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from typing import Any

from prompt_kit import (
    ADVANCED_GENERATION_ADVISORY,
    GENRE_MODULES,
    LANGUAGE_PRESETS,
    PROMPT_KIT_METADATA_FIELDS,
    PROMPT_KIT_VERSION,
    TROUBLESHOOTING_MATRIX,
    VALIDATION_CHECKLIST,
    infer_genre_modules,
    is_sparse_lyric_genre,
    kit_metadata_defaults,
    language_preset,
    negative_control_for,
    prompt_kit_payload,
    section_map_for,
)
from studio_core import docs_best_model_settings
from user_album_contract import (
    USER_ALBUM_CONTRACT_VERSION,
    apply_user_album_contract_to_tracks,
    apply_user_album_contract_to_track,
    extract_user_album_contract,
    tracks_from_user_album_contract,
)


OFFICIAL_SOURCES = [
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo",
]

ALBUM_FINAL_MODEL = "acestep-v15-xl-sft"


def _portfolio_item(model: str, slug: str, label: str, summary: str) -> dict[str, Any]:
    settings = docs_best_model_settings(model)
    return {
        "model": model,
        "slug": slug,
        "label": label,
        "summary": summary,
        "default_steps": settings["inference_steps"],
        "default_guidance_scale": settings["guidance_scale"],
        "default_shift": settings["shift"],
        "default_infer_method": settings["infer_method"],
        "default_sampler_mode": settings["sampler_mode"],
        "default_audio_format": settings["audio_format"],
        "quality_preset": settings["quality_preset"],
    }


ALBUM_MODEL_PORTFOLIO: list[dict[str, Any]] = [
    _portfolio_item("acestep-v15-turbo", "turbo", "Turbo", "Best default, fast"),
    _portfolio_item("acestep-v15-turbo-shift3", "turbo-shift3", "Turbo Shift3", "Clear timbre, dry"),
    _portfolio_item("acestep-v15-sft", "sft", "SFT", "CFG detail tuning"),
    _portfolio_item("acestep-v15-base", "base", "Base", "All tasks, fine-tuning"),
    _portfolio_item("acestep-v15-xl-turbo", "xl-turbo", "XL Turbo", "Best 20GB+ daily driver"),
    _portfolio_item(ALBUM_FINAL_MODEL, "xl-sft", "XL SFT", "Highest detail, CFG"),
    _portfolio_item("acestep-v15-xl-base", "xl-base", "XL Base", "All tasks, XL quality"),
]
ALBUM_MODEL_PORTFOLIO_MODELS = [item["model"] for item in ALBUM_MODEL_PORTFOLIO]

MODEL_STRATEGIES: dict[str, dict[str, Any]] = {
    "all_models_album": {
        "label": "All 7 model albums",
        "summary": "Render one complete album for every official ACE-Step 1.5 model in AceJAM's album portfolio.",
        "order": ALBUM_MODEL_PORTFOLIO_MODELS,
        "multi_album": True,
    },
    "xl_sft_final": {
        "label": "XL SFT final",
        "summary": "Locked album final-render policy: every track uses XL SFT with Docs-best detail and CFG control.",
        "order": [ALBUM_FINAL_MODEL],
        "locked_final_model": True,
    },
    "best_installed": {
        "label": "Best installed",
        "summary": "Prefer XL Turbo for album quality, then default Turbo for practical full-album runs.",
        "order": ["acestep-v15-xl-turbo", "acestep-v15-turbo", "acestep-v15-xl-sft", "acestep-v15-sft"],
    },
    "fast_installed": {
        "label": "Fast installed",
        "summary": "Prefer default Turbo first so multi-track albums stay fast and reliable.",
        "order": ["acestep-v15-turbo", "acestep-v15-xl-turbo", "acestep-v15-turbo-shift3"],
    },
    "maximum_detail": {
        "label": "Maximum detail",
        "summary": "Prefer installed SFT/XL SFT for Docs-best CFG detail, then fall back to Turbo.",
        "order": ["acestep-v15-xl-sft", "acestep-v15-sft", "acestep-v15-xl-turbo", "acestep-v15-turbo"],
    },
    "selected": {
        "label": "Selected model",
        "summary": "Use the exact model selected in the global ACE-Step model dropdown.",
        "order": [],
    },
}

DENSITY_PRESETS = {
    "sparse": {"label": "Sparse", "word_factor": 0.9, "line_factor": 0.9},
    "balanced": {"label": "Balanced", "word_factor": 1.15, "line_factor": 1.0},
    "dense": {"label": "Dense", "word_factor": 1.35, "line_factor": 1.12},
    "rap_dense": {"label": "Rap dense", "word_factor": 1.55, "line_factor": 1.25},
}

TAG_TAXONOMY: dict[str, list[str]] = {
    "genre_style": [
        "pop", "hip-hop", "rap", "trap", "drill", "melodic rap", "R&B", "soul", "afrobeat",
        "afrobeats", "amapiano", "afrohouse", "dancehall", "reggaeton", "garage", "house", "tech house",
        "techno", "trance", "drum and bass", "dubstep", "EDM", "synthwave", "indie pop", "indie folk",
        "rock", "alt rock", "punk", "punk rock", "metal", "jazz", "lo-fi hip hop",
        "classical", "cinematic", "orchestral", "ambient", "gospel", "country", "latin pop", "K-pop", "J-pop",
    ],
    "mood_atmosphere": [
        "melancholic", "uplifting", "euphoric", "dark", "dreamy", "nostalgic", "intimate",
        "aggressive", "confident", "romantic", "cinematic", "tense", "hopeful", "bittersweet",
        "luxurious", "gritty", "warm", "cold", "neon-lit", "late night", "sunlit",
    ],
    "instruments": [
        "piano", "grand piano", "Rhodes", "electric piano", "organ", "acoustic guitar",
        "clean electric guitar", "distorted guitar", "nylon guitar", "bass guitar", "upright bass",
        "808 bass", "sub-bass", "synth bass", "trap hi-hats", "808 kick", "punchy snare",
        "breakbeat", "drum machine", "brush drums", "synth pads", "analog synth", "lead synth",
        "arpeggiated synth", "strings", "violin", "cello", "brass", "trumpet", "saxophone",
        "flute", "choir", "turntable scratches", "risers", "glitch effects",
    ],
    "timbre_texture": [
        "warm", "bright", "crisp", "airy", "punchy", "lush", "raw", "polished", "gritty",
        "wide stereo", "close-mic", "tape saturation", "vinyl texture", "deep low end",
        "silky top end", "dry vocal", "wet reverb", "analog warmth",
    ],
    "era_reference": [
        "70s soul", "80s synth pop", "90s boom bap", "90s R&B", "2000s pop punk",
        "2010s EDM", "modern trap", "future garage", "vintage soul", "classic house",
    ],
    "production_style": [
        "high-fidelity", "studio polished", "crisp modern mix", "lo-fi texture", "warm analog mix",
        "club master", "radio ready", "atmospheric", "minimal arrangement", "layered production",
        "cinematic build", "hard-hitting drums", "sidechain pulse",
    ],
    "vocal_character": [
        "male vocal", "female vocal", "male rap vocal", "female rap vocal", "melodic rap vocal",
        "autotune vocal", "breathy vocal", "raspy vocal", "powerful belt", "falsetto",
        "stacked harmonies", "choir vocals", "spoken vocal", "whispered vocal",
    ],
    "speed_rhythm": [
        "slow tempo", "mid-tempo", "fast-paced", "groovy", "driving rhythm", "laid-back groove",
        "swing feel", "four-on-the-floor", "half-time drums", "syncopated rhythm",
    ],
    "structure_hints": [
        "building intro", "catchy chorus", "anthemic hook", "dramatic bridge", "explosive drop",
        "breakdown", "beat switch", "fade-out ending", "stripped outro", "call and response",
    ],
    "track_stems": [
        "woodwinds", "brass", "fx", "synth", "strings", "percussion", "keyboard", "guitar",
        "bass", "drums", "backing_vocals", "vocals",
    ],
}

LYRIC_META_TAGS: dict[str, list[str]] = {
    "basic_structure": ["[Intro]", "[Verse]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Post-Chorus]", "[Bridge]", "[Final Chorus]", "[Outro]"],
    "dynamic_sections": ["[Build]", "[Build-Up]", "[Drop]", "[Final Drop]", "[Breakdown]", "[Climax]", "[Fade Out]", "[Silence]"],
    "instrumental_sections": ["[Instrumental]", "[Instrumental Break]", "[Synth Solo]", "[Guitar Solo]", "[Piano Interlude]", "[Brass Break]", "[Drum Break]"],
    "performance_modifiers": [
        "[Verse - rap]", "[Chorus - rap]", "[Verse - melodic rap]", "[Chorus - anthemic]",
        "[Bridge - whispered]", "[Chorus - layered vocals]", "[Intro - dreamy]",
    ],
    "energy_markers": ["[building energy]", "[explosive drop]", "[calm]", "[intense]", "[Final chord fades out]"],
}

CRAFT_TOOLS: list[dict[str, str]] = [
    {"name": "ModelAdvisorTool", "summary": "Chooses only installed ACE-Step models for the album strategy."},
    {"name": "ModelPortfolioTool", "summary": "Plans the full 7-model album render portfolio."},
    {"name": "PerModelSettingsTool", "summary": "Returns per-model steps, guidance, shift, speed, quality, and warnings."},
    {"name": "AlbumRenderMatrixTool", "summary": "Calculates track-by-model render counts and album grouping."},
    {"name": "FilenamePlannerTool", "summary": "Plans safe track/model filenames for downloads and album ZIPs."},
    {"name": "XLModelPolicyTool", "summary": "Locks final album renders to ACE-Step XL SFT and explains download/runtime requirements."},
    {"name": "TagLibraryTool", "summary": "Provides ACE-Step caption dimensions and lyric meta tags."},
    {"name": "LyricLengthTool", "summary": "Plans sections, words, and lines for the chosen duration."},
    {"name": "GenerationSettingsTool", "summary": "Builds editable per-track seed, steps, guidance, shift, sampler, and format settings."},
    {"name": "ArrangementTool", "summary": "Plans intro, verses, hooks, bridge, outro, BPM/key/time, and energy movement."},
    {"name": "VocalPerformanceTool", "summary": "Creates persona, cadence, ad-lib, harmony, and lyric performance tags."},
    {"name": "RhymeFlowTool", "summary": "Turns artist references into rhyme and flow technique briefs."},
    {"name": "MetaphorWorldTool", "summary": "Builds one coherent metaphor world per track."},
    {"name": "HookDoctorTool", "summary": "Checks hooks for contrast, repeatability, and title connection."},
    {"name": "ClicheGuardTool", "summary": "Flags repeated lines and generic lyric cliches."},
    {"name": "AlbumArcTool", "summary": "Maps opener, escalation, climax, cooldown, and closer."},
    {"name": "AlbumContinuityTool", "summary": "Keeps sequencing, recurring motifs, key movement, and emotional contrast coherent."},
    {"name": "InspirationRadarTool", "summary": "Fetches current inspiration snippets when enabled."},
    {"name": "CaptionPolisherTool", "summary": "Builds compact ACE-Step captions from style dimensions."},
    {"name": "ConflictCheckerTool", "summary": "Flags caption, tag, BPM, key, and lyric contradictions."},
    {"name": "MixMasterTool", "summary": "Recommends output, score/LRC/audio-code flags, and mix/master safety checks."},
    {"name": "HitScoreTool", "summary": "Scores hook, lyric sufficiency, uniqueness, and production readiness."},
    {"name": "TrackRepairTool", "summary": "Repairs missing lyrics, weak hooks, tag conflicts, and under-specified generation fields."},
    {"name": "LanguagePresetTool", "summary": "Returns the selected language/script policy and romanization guidance."},
    {"name": "GenreModuleTool", "summary": "Routes prompts through the closest prompt-kit genre module or fusion."},
    {"name": "SectionMapTool", "summary": "Builds duration-realistic section maps for vocal or instrumental workflows."},
    {"name": "IterationPlanTool", "summary": "Plans listen-adjust-regenerate passes for human-centered ACE-Step iteration."},
    {"name": "TroubleshootingTool", "summary": "Maps common ACE-Step failures to compact repair instructions."},
    {"name": "ValidationChecklistTool", "summary": "Returns prompt-kit quality gates for captions, lyrics, metadata, safety, and runtime support."},
    {"name": "NegativeControlTool", "summary": "Builds genre-aware negative control phrases for generation payloads."},
]

BANNED_CLICHES = [
    "echoes of", "shattered dreams", "empty streets", "fading light", "stories untold",
    "lost in time", "forgotten memories", "endless night", "unseen tears", "whispers in the dark",
    "waves crashing", "endless road", "burning bridges", "fading away", "broken chains",
    "heart on fire", "dancing in the rain", "light at the end of the tunnel", "stars aligned",
    "wings to fly", "ocean of emotions", "paint the sky", "chase the sun", "against all odds",
]

ARTIST_TECHNIQUES = {
    "nas": "cinematic street narrative, precise detail, internal rhyme, reflective authority",
    "eminem": "multisyllabic rhyme, breath-control bursts, sharp pivots, punchline density",
    "jay-z": "economical confidence, double meanings, conversational pocket, luxury detail",
    "kendrick lamar": "character perspective shifts, moral tension, rhythmic variation, layered hooks",
    "drake": "melodic rap contrast, conversational hooks, nightlife detail, emotional directness",
    "travis scott": "psychedelic ad-libs, atmospheric trap, texture-first hooks, floating cadence",
    "frank ocean": "fragmented memories, sensory intimacy, unusual chord mood, understated hooks",
    "the weeknd": "neon noir atmosphere, falsetto tension, nocturnal pop drama, glossy synths",
}

KEY_CYCLE = ["C minor", "Eb major", "G minor", "Bb major", "F minor", "Ab major", "D minor", "F major"]
VALID_KEY_ROOTS = {"C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"}
SUPPORTED_AUDIO_FORMATS = {"wav", "flac", "ogg", "mp3", "opus", "aac", "wav32"}
SUPPORTED_INFER_METHODS = {"ode", "sde"}
SUPPORTED_SAMPLERS = {"euler", "heun"}
NULLISH_TEXT = {"", "auto", "none", "null", "nil", "n/a", "na"}


def _is_nullish(value: Any) -> bool:
    return value is None or str(value).strip().lower() in NULLISH_TEXT


def _number_or_default(value: Any, default: Any) -> Any:
    return default if _is_nullish(value) else value


def _float_or_default(value: Any, default: Any) -> float:
    raw = _number_or_default(value, default)
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"-?\d+(?:\.\d+)?", str(raw or ""))
    if match:
        return float(match.group(0))
    if raw is not default:
        return _float_or_default(default, 0.0)
    return 0.0


def _int_or_default(value: Any, default: Any) -> int:
    return int(_float_or_default(value, default))


def parse_duration_seconds(value: Any, default: float = 120.0) -> float:
    """Parse album durations from seconds, m:ss, h:mm:ss, or loose text."""
    if _is_nullish(value):
        return float(default)
    if isinstance(value, (int, float)):
        return max(10.0, min(600.0, float(value)))
    text = str(value).strip().lower()
    if not text:
        return float(default)
    colon = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", text)
    if colon:
        hours = int(colon.group(1) or 0)
        minutes = int(colon.group(2))
        seconds = int(colon.group(3))
        return max(10.0, min(600.0, float(hours * 3600 + minutes * 60 + seconds)))
    minute_second = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|minutes?)\D+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?", text)
    if minute_second:
        return max(10.0, min(600.0, float(minute_second.group(1)) * 60.0 + float(minute_second.group(2))))
    minute_only = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|minutes?)\b", text)
    if minute_only:
        return max(10.0, min(600.0, float(minute_only.group(1)) * 60.0))
    number = re.search(r"-?\d+(?:\.\d+)?", text)
    if number:
        return max(10.0, min(600.0, float(number.group(0))))
    return float(default)


def normalize_key_scale(value: Any, index: int = 0, strategy: str = "related") -> str:
    text = str(value or "").strip()
    match = re.search(r"\b([A-G](?:#|b)?)\s*(major|minor|maj|min|m)\b", text, flags=re.I)
    if match:
        root = match.group(1)
        if len(root) == 1:
            root = root.upper()
        else:
            root = root[0].upper() + root[1:]
        if root in VALID_KEY_ROOTS:
            mode_raw = match.group(2).lower()
            mode = "minor" if mode_raw in {"minor", "min", "m"} else "major"
            return f"{root} {mode}"
    if strategy == "related":
        return KEY_CYCLE[index % len(KEY_CYCLE)]
    return ""


def normalize_time_signature(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "4"
    matches = re.findall(r"\b([2364])(?:\s*/\s*(?:4|8))?\b", text)
    for candidate in matches:
        if candidate in {"2", "3", "4", "6"}:
            return candidate
    return "4"


def normalize_album_audio_format(value: Any) -> str:
    text = str(value or "wav").lower()
    if "wav32" in text or "32-bit" in text or "24-bit" in text:
        return "wav32"
    for fmt in SUPPORTED_AUDIO_FORMATS:
        if re.search(rf"\b{re.escape(fmt)}\b", text):
            return fmt
    return "wav"


def normalize_infer_method(value: Any) -> str:
    text = str(value or "ode").lower()
    if "sde" in text:
        return "sde"
    return "ode"


def normalize_sampler_mode(value: Any) -> str:
    text = str(value or "heun").strip().lower()
    if "heun" in text:
        return "heun"
    if text == "euler":
        return "euler"
    return "heun"


def normalize_seed_value(value: Any) -> tuple[str, str]:
    if _is_nullish(value):
        return "-1", ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(?:\s*,\s*-?\d+)*", text):
        return text, ""
    stable = zlib.crc32(text.encode("utf-8")) % 2_147_483_647
    return str(stable or 42), f"Converted descriptive seed '{text[:80]}' to deterministic numeric seed {stable or 42}."


def split_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,;\n|]+", str(value))
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        term = str(item).strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.lower(), flags=re.UNICODE)


def _subject_terms(text: str) -> list[str]:
    stop = {
        "a", "an", "and", "the", "to", "of", "in", "on", "with", "for", "about", "song",
        "album", "track", "music", "make", "like", "style", "ft", "feat", "featuring",
    }
    terms = [w for w in _words(text) if len(w) > 2 and w not in stop]
    counts = Counter(terms)
    return [word for word, _ in counts.most_common(12)]


def sanitize_artist_references(text: str) -> tuple[str, list[str]]:
    raw = str(text or "")
    return raw.strip(), []


ARTIST_NAME_SUFFIXES = [
    "Signal",
    "Harbor",
    "Voltage",
    "Cinema",
    "Archive",
    "Crown",
    "Pulse",
    "North",
    "Velvet",
    "Static",
]


def normalize_artist_name(value: Any, fallback: str = "AceJAM") -> str:
    text = str(value or "").strip()
    if re.match(r"(?i)^\s*(?:in the style of|zoals|like|feat\.?|featuring|ft\.)\b", text):
        return str(fallback or "AceJAM").strip()[:48] or "AceJAM"
    cleaned, notes = sanitize_artist_references(text)
    if notes:
        cleaned = ""
    cleaned = re.sub(r"(?i)\b(in the style of|zoals|like|feat\.?|featuring|ft\.)\b", " ", cleaned)
    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    if not words:
        return str(fallback or "AceJAM").strip()[:48] or "AceJAM"
    return " ".join(words[:4]).title()[:48].strip() or str(fallback or "AceJAM").strip()[:48] or "AceJAM"


def derive_artist_name(title: Any = "", concept: Any = "", tags: Any = "", index: int = 0) -> str:
    text = " ".join(str(item or "") for item in [title, concept, tags])
    cleaned, _notes = sanitize_artist_references(text)
    terms = [term for term in _subject_terms(cleaned) if len(term) > 2]
    if not terms:
        return "AceJAM"
    suffix = ARTIST_NAME_SUFFIXES[int(index or 0) % len(ARTIST_NAME_SUFFIXES)]
    return normalize_artist_name(f"{terms[0]} {suffix}", "AceJAM")


def section_sequence(duration: float, preset: str = "auto", rap: bool = False) -> list[str]:
    if preset in {"short_hook", "single"}:
        return ["Verse - rap" if rap else "Verse", "Chorus - rap" if rap else "Chorus"]
    if preset == "club":
        return ["Intro", "Verse - rap" if rap else "Verse", "Build", "Chorus", "Drop", "Verse", "Bridge", "Final Chorus", "Outro"]
    if preset == "cinematic":
        return ["Intro", "Verse 1", "Pre-Chorus", "Chorus - anthemic", "Verse 2", "Bridge", "Final Chorus", "Outro"]
    dur = _int_or_default(duration, 120)
    if dur <= 45:
        return ["Verse - rap" if rap else "Verse", "Chorus - rap" if rap else "Chorus", "Outro"]
    if dur <= 90:
        return ["Intro", "Verse - rap" if rap else "Verse", "Chorus - rap" if rap else "Chorus", "Verse 2", "Final Chorus"]
    if dur <= 150:
        return ["Intro", "Verse 1 - rap" if rap else "Verse 1", "Pre-Chorus", "Chorus - rap" if rap else "Chorus", "Verse 2", "Bridge", "Final Chorus", "Outro"]
    if dur <= 240:
        return ["Intro", "Verse 1 - rap" if rap else "Verse 1", "Pre-Chorus", "Chorus - rap" if rap else "Chorus", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Verse 3" if rap else "Breakdown", "Final Chorus", "Outro"]
    if dur <= 360:
        return ["Intro", "Verse 1 - rap" if rap else "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Verse 3", "Breakdown", "Final Chorus", "Outro"]
    return ["Intro", "Verse 1 - rap" if rap else "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Verse 3", "Instrumental Break", "Verse 4", "Final Chorus", "Outro"]


def lyric_length_plan(duration: float, density: str = "balanced", structure_preset: str = "auto", genre_hint: str = "") -> dict[str, Any]:
    dur = int(parse_duration_seconds(duration, 120))
    preset = DENSITY_PRESETS.get(density, DENSITY_PRESETS["balanced"])
    rap = bool(re.search(r"\b(rap|hip.?hop|trap|drill|grime)\b", genre_hint or "", re.I))
    sparse_genre = bool(is_sparse_lyric_genre(genre_hint) and not rap)
    if sparse_genre:
        section_map = section_map_for(dur, genre_hint, instrumental=True)
        sections = [str(item.get("tag") or "").strip("[]") for item in section_map if item.get("tag")]
        density = "sparse"
        preset = DENSITY_PRESETS["sparse"]
    else:
        sections = section_sequence(dur, structure_preset, rap=rap)
    bands = [
        (30, 45, 58, 70),
        (60, 90, 110, 130),
        (120, 190, 225, 260),
        (180, 300, 360, 420),
        (240, 430, 500, 560),
        (300, 520, 630, 720),
        (600, 560, 720, 780),
    ]
    min_words, base_words, max_words = bands[-1][1:]
    for limit, low, base, high in bands:
        if dur <= limit:
            min_words, base_words, max_words = low, base, high
            break
    if sparse_genre:
        sparse_bands = [
            (60, 0, 16, 40),
            (120, 8, 32, 70),
            (210, 16, 54, 110),
            (300, 24, 78, 150),
            (600, 32, 96, 180),
        ]
        for limit, low, base, high in sparse_bands:
            if dur <= limit:
                min_words, base_words, max_words = low, base, high
                break
    density_factor = {"sparse": 0.9, "balanced": 1.0, "dense": 1.1, "rap_dense": 1.16}.get(density, 1.0)
    target_words = max(min_words, min(max_words, int(base_words * density_factor)))
    target_lines = max(len(sections) * 3, int(target_words / (4.6 if rap else 5.4) * float(preset["line_factor"])))
    min_lines = max(len(sections) * 2, int(target_lines * 0.72))
    if sparse_genre:
        target_lines = max(len(sections), min(len(sections) * 3, target_lines))
        min_lines = max(0 if min_words == 0 else len(sections), int(target_lines * 0.5))
    return {
        "duration": dur,
        "density": density,
        "structure_preset": structure_preset,
        "sections": sections,
        "structure": ", ".join(f"[{section}]" for section in sections),
        "target_words": target_words,
        "min_words": min_words,
        "max_words": max_words,
        "target_lines": target_lines,
        "min_lines": min_lines,
        "max_lyrics_chars": 4096,
        "duration_coverage_note": (
            "At very long durations ACE-Step's lyric cap limits continuous vocals; use enough sections plus intentional instrumental breaks."
            if dur > 360
            else "Sparse or instrumental genres should cover the duration with section tags, builds, drops, and short motifs instead of forced full verses."
            if sparse_genre
            else "Lyrics should cover the full selected duration with verses, hooks, bridge, and final chorus."
        ),
        "note": "ACE-Step lyrics are the temporal script; keep sonic tags in caption and use lyric/performance tags inside lyrics.",
    }


def choose_song_model(
    installed_models: set[str] | list[str],
    strategy: str = "best_installed",
    requested_model: str | None = None,
) -> dict[str, Any]:
    installed = {m for m in installed_models if str(m).startswith("acestep-v15-")}
    requested = (requested_model or "auto").strip()
    if requested and requested != "auto":
        if requested not in installed:
            return {"ok": False, "model": requested, "error": f"{requested} is not installed."}
        return {"ok": True, "model": requested, "strategy": "selected", "reason": "Using selected installed model."}
    selected_strategy = MODEL_STRATEGIES.get(strategy, MODEL_STRATEGIES["best_installed"])
    for candidate in selected_strategy["order"]:
        if candidate in installed:
            return {
                "ok": True,
                "model": candidate,
                "strategy": strategy,
                "reason": selected_strategy["summary"],
                "locked_final_model": bool(selected_strategy.get("locked_final_model")),
            }
    target = selected_strategy["order"][0] if selected_strategy.get("order") else ""
    return {
        "ok": False,
        "model": target,
        "strategy": strategy,
        "locked_final_model": bool(selected_strategy.get("locked_final_model")),
        "error": (
            f"{target} is not installed."
            if target
            else "No installed ACE-Step text2music model matches this album strategy."
        ),
    }


def album_model_portfolio(installed_models: set[str] | list[str] | None = None) -> list[dict[str, Any]]:
    installed = set(installed_models or [])
    portfolio: list[dict[str, Any]] = []
    for index, item in enumerate(ALBUM_MODEL_PORTFOLIO, start=1):
        enriched = dict(item)
        model = str(enriched["model"])
        enriched.update(
            {
                "index": index,
                "installed": model in installed,
                "download_required": model not in installed,
                "downloadable": True,
                "task_type": "text2music",
                "album_policy": "render_this_full_album_once_per_track",
            }
        )
        portfolio.append(enriched)
    return portfolio


def album_models_for_strategy(strategy: str, installed_models: set[str] | list[str] | None = None) -> list[dict[str, Any]]:
    if strategy == "all_models_album":
        return album_model_portfolio(installed_models)
    if strategy == "xl_sft_final":
        return [item for item in album_model_portfolio(installed_models) if item["model"] == ALBUM_FINAL_MODEL]
    info = choose_song_model(installed_models or [], strategy, "auto")
    model = str(info.get("model") or "").strip()
    if not model:
        return []
    for item in album_model_portfolio(installed_models):
        if item["model"] == model:
            return [item]
    return [
        {
            **_portfolio_item(
                model,
                re.sub(r"^acestep-v15-", "", model).replace("_", "-"),
                model,
                str(info.get("reason") or "Selected album model"),
            ),
            "installed": model in set(installed_models or []),
            "download_required": model not in set(installed_models or []),
            "downloadable": True,
            "task_type": "text2music",
            "album_policy": "single_model_album",
        }
    ]


def album_arc(num_tracks: int) -> list[str]:
    labels = []
    for index in range(max(1, int(num_tracks))):
        pos = index / max(1, num_tracks - 1)
        if index == 0:
            labels.append("opener - immediate identity and strongest first impression")
        elif index == num_tracks - 1:
            labels.append("closer - resolution, callback, or final twist")
        elif 0.45 <= pos <= 0.7:
            labels.append("climax - highest stakes and biggest hook")
        elif pos < 0.45:
            labels.append("escalation - new scene, sharper rhythm, more pressure")
        else:
            labels.append("cooldown - emotional consequence and contrast")
    return labels


def tag_pack_values(packs: list[str]) -> list[str]:
    values: list[str] = []
    for pack in packs:
        key = pack.strip()
        if key in TAG_TAXONOMY:
            values.extend(TAG_TAXONOMY[key][:12])
    return values


def infer_core_tags(concept: str, track_index: int = 0) -> list[str]:
    lowered = concept.lower()
    if re.search(r"\b(rap|hip.?hop|trap|drill|bars)\b", lowered):
        base = ["hip-hop", "808 bass", "trap hi-hats", "male rap vocal", "crisp modern mix"]
    elif "r&b" in lowered or "soul" in lowered:
        base = ["R&B", "Rhodes", "sub-bass", "breathy vocal", "warm analog mix"]
    elif "rock" in lowered:
        base = ["rock", "electric guitar", "bass guitar", "punchy snare", "high-fidelity"]
    elif "house" in lowered or "club" in lowered:
        base = ["house", "four-on-the-floor", "synth bass", "club master", "euphoric"]
    elif "cinematic" in lowered or "orchestral" in lowered:
        base = ["cinematic", "strings", "brass", "dramatic bridge", "wide stereo"]
    else:
        base = ["pop", "catchy chorus", "synth pads", "polished", "uplifting"]
    moods = TAG_TAXONOMY["mood_atmosphere"]
    instruments = TAG_TAXONOMY["instruments"]
    production = TAG_TAXONOMY["production_style"]
    base.extend([moods[(track_index * 3) % len(moods)], instruments[(track_index * 5) % len(instruments)], production[(track_index * 2) % len(production)]])
    return base


def build_track_tags(
    concept: str,
    track_index: int,
    tag_packs: Any = None,
    custom_tags: Any = None,
    negative_tags: Any = None,
    max_tags: int = 12,
) -> list[str]:
    tags = infer_core_tags(concept, track_index)
    tags.extend(tag_pack_values(split_terms(tag_packs)))
    tags.extend(split_terms(custom_tags))
    negatives = [n.lower() for n in split_terms(negative_tags)]
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = re.sub(r"\s+", " ", str(tag).strip())
        if not normalized:
            continue
        key = normalized.lower()
        if any(neg in key for neg in negatives):
            continue
        if key in seen:
            continue
        seen.add(key)
        clean.append(normalized)
        if len(clean) >= max_tags:
            break
    return clean


def polish_caption(tags: Any, description: str = "", global_caption: str = "") -> str:
    tag_text = ", ".join(split_terms(tags))
    parts = [tag_text]
    if description:
        parts.append(str(description).strip())
    if global_caption:
        parts.append(str(global_caption).strip())
    caption = ". ".join(part for part in parts if part)
    caption = re.sub(r"\b\d{2,3}\s*bpm\b", "", caption, flags=re.I)
    caption = re.sub(r"\s{2,}", " ", caption).strip(" ,.")
    return caption[:512]


def _fallback_lines(language: str, section: str, title: str, terms: list[str], metaphor: str, rap: bool, count: int) -> list[str]:
    subject = terms[count % len(terms)] if terms else "moment"
    accent = terms[(count + 1) % len(terms)] if len(terms) > 1 else "city"
    hook = re.sub(r"[^A-Za-z0-9 ']", "", title).strip() or "All The Way Up"
    if language == "nl":
        if "Chorus" in section:
            return [
                f"{hook} blijft hangen in de nacht",
                f"Elke stap klinkt harder dan gedacht",
                f"Wij bouwen vuur uit koude steen",
                f"Tot de hele straat met ons mee beweegt",
            ]
        if "Bridge" in section:
            return [
                f"Even stil, de kamer ademt mee",
                f"{metaphor.capitalize()} trekt een lijn door wat ik deed",
            ]
        return [
            f"Half drie, {subject} op mijn jas",
            f"{accent.capitalize()} in mijn hoofd, ik geef geen pas",
            f"Elke bar heeft tanden in de beat",
            f"Ik maak van druk een kroon die niemand ziet",
        ]
    if "Chorus" in section:
        return [
            f"{hook} rings out in the room",
            f"We turn pressure into perfume",
            f"Hands up when the low end blooms",
            f"One more night and we break through",
        ]
    if "Bridge" in section:
        return [
            f"The {metaphor} bends but it never breaks",
            f"I hear the truth in the breath it takes",
        ]
    if rap:
        return [
            f"Back door click with the {subject} on tilt",
            f"Cold chain swing where the old doubt built",
            f"Big dream stitched in a small room quilt",
            f"Every rhyme hits clean, no filler, no guilt",
        ]
    return [
        f"Morning finds the {subject} on the floor",
        f"{accent.capitalize()} light is leaning through the door",
        f"I kept the receipt from the life before",
        f"Now I want the sound and nothing more",
    ]


def build_fallback_lyrics(
    title: str,
    concept: str,
    duration: float,
    language: str,
    density: str,
    structure_preset: str,
) -> str:
    terms = _subject_terms(concept + " " + title) or ["night", "motion", "signal"]
    rap = bool(re.search(r"\b(rap|hip.?hop|trap|drill|bars)\b", concept, re.I))
    plan = lyric_length_plan(duration, density, structure_preset, genre_hint=concept)
    metaphor = ["signal", "architecture", "weather", "currency", "gravity"][len(title) % 5]
    chunks: list[str] = []
    section_counts: Counter[str] = Counter()
    for section in plan["sections"]:
        lines_needed = max(2, int(plan["target_lines"] / max(1, len(plan["sections"]))))
        if "Chorus" in section:
            lines_needed = max(4, lines_needed)
        if "Verse" in section:
            lines_needed = max(6 if rap else 5, lines_needed)
        count = section_counts[section]
        section_counts[section] += 1
        lines: list[str] = []
        while len(lines) < lines_needed:
            lines.extend(_fallback_lines(language, section, title, terms, metaphor, rap, count + len(lines)))
        lines = lines[:lines_needed]
        chunks.append(f"[{section}]\n" + "\n".join(lines))
    lyrics = "\n\n".join(chunks)
    return trim_lyrics_to_limit(lyrics, plan["max_lyrics_chars"])


def _section_key(section: str) -> str:
    text = re.sub(r"[\[\]]", "", section).lower()
    text = re.sub(r"\b\d+\b", "", text)
    if "chorus" in text:
        return "chorus"
    if "verse" in text:
        return "verse"
    if "bridge" in text:
        return "bridge"
    if "pre" in text:
        return "pre-chorus"
    if "intro" in text:
        return "intro"
    if "outro" in text:
        return "outro"
    if "break" in text or "drop" in text:
        return "break"
    return text.strip()


def trim_lyrics_to_limit(lyrics: str, limit: int = 4096) -> str:
    text = str(lyrics or "").strip()
    if len(text) <= limit:
        return text
    chunks: list[str] = []
    total = 0
    for block in text.split("\n\n"):
        addition = ("\n\n" if chunks else "") + block
        if total + len(addition) > limit - 48:
            break
        chunks.append(block)
        total += len(addition)
    trimmed = "\n\n".join(chunks).strip()
    return (trimmed or text[: limit - 32].rstrip()) + "\n\n[Outro]\nLet it breathe."


def expand_lyrics_for_duration(
    title: str,
    concept: str,
    lyrics: str,
    duration: float,
    language: str,
    density: str,
    structure_preset: str,
) -> str:
    plan = lyric_length_plan(duration, density, structure_preset, genre_hint=concept)
    current = str(lyrics or "").strip()
    if not current:
        return build_fallback_lyrics(title, concept, duration, language, density, structure_preset)
    stats = lyric_stats(current)
    covered = {_section_key(section) for section in stats["sections"]}
    needed = [_section_key(section) for section in plan["sections"]]
    missing_keys = [key for key in needed if key and key not in covered]
    extras: list[str] = []
    fallback = build_fallback_lyrics(title, concept, duration, language, density, structure_preset)
    fallback_blocks = re.split(r"\n\s*\n", fallback)
    for block in fallback_blocks:
        match = re.match(r"\[([^\]]+)\]", block.strip())
        if not match:
            continue
        key = _section_key(match.group(1))
        if key in missing_keys:
            extras.append(block.strip())
            covered.add(key)
            missing_keys = [item for item in missing_keys if item != key]
    repaired = current
    if extras:
        repaired = repaired + "\n\n" + "\n\n".join(extras)
    stats = lyric_stats(repaired)
    extension_index = 1
    while (
        (stats["word_count"] < plan["min_words"] or stats["line_count"] < plan["min_lines"])
        and len(repaired) < plan["max_lyrics_chars"] - 260
        and extension_index <= 8
    ):
        section = "Verse - extension" if extension_index % 2 else "Chorus - reprise"
        filler = build_fallback_lyrics(
            f"{title} {extension_index}",
            concept,
            min(90, duration),
            language,
            "rap_dense" if "rap" in density else density,
            "short_hook" if "Chorus" in section else "auto",
        )
        block = re.sub(r"^\[[^\]]+\]", f"[{section}]", filler.strip(), count=1)
        repaired = repaired + "\n\n" + block
        repaired = trim_lyrics_to_limit(repaired, plan["max_lyrics_chars"])
        stats = lyric_stats(repaired)
        extension_index += 1
    return trim_lyrics_to_limit(repaired, plan["max_lyrics_chars"])


def lyric_stats(lyrics: str) -> dict[str, Any]:
    nonempty = [line.strip() for line in str(lyrics or "").splitlines() if line.strip()]
    lyric_lines = [re.sub(r"\[[^\]]+\]", "", line).strip() for line in nonempty]
    lyric_lines = [line for line in lyric_lines if line]
    words = _words("\n".join(lyric_lines))
    sections = re.findall(r"\[[^\]]+\]", str(lyrics or ""))
    repeats = [line for line, count in Counter(line.lower() for line in lyric_lines).items() if count > 1]
    return {
        "word_count": len(words),
        "line_count": len(lyric_lines),
        "section_count": len(sections),
        "sections": sections,
        "repeated_lines": repeats[:6],
        "char_count": len(str(lyrics or "")),
    }


def quality_report(track: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    plan = lyric_length_plan(
        parse_duration_seconds(track.get("duration") or options.get("track_duration") or 120, 120),
        str(options.get("lyric_density") or "balanced"),
        str(options.get("structure_preset") or "auto"),
        str(track.get("tags") or options.get("concept") or ""),
    )
    stats = lyric_stats(str(track.get("lyrics") or ""))
    lowered = str(track.get("lyrics") or "").lower()
    cliches = [phrase for phrase in BANNED_CLICHES if phrase in lowered]
    tags = [t.lower() for t in split_terms(track.get("tags"))]
    conflicts = []
    if any("slow" in t for t in tags) and any("fast" in t for t in tags):
        conflicts.append("Caption tags contain both slow and fast tempo cues.")
    if any("instrumental" in t for t in tags) and stats["word_count"] > 12:
        conflicts.append("Caption says instrumental but lyrics contain sung words.")
    expected_keys = {_section_key(section) for section in plan["sections"] if section}
    actual_keys = {_section_key(section) for section in stats["sections"]}
    missing_sections = sorted(key for key in expected_keys if key and key not in actual_keys)
    section_coverage = round(
        (len(expected_keys) - len(missing_sections)) / max(1, len(expected_keys)),
        2,
    )
    length_ok = stats["word_count"] >= plan["min_words"] and stats["line_count"] >= plan["min_lines"]
    char_ok = stats["char_count"] <= plan["max_lyrics_chars"]
    lyric_meta_tag_score = round(min(1.0, stats["section_count"] / max(1, len(plan["sections"]))), 2)
    duration_coverage_score = round(
        min(
            1.0,
            min(
                stats["word_count"] / max(1, plan["target_words"]),
                stats["line_count"] / max(1, plan["target_lines"]),
            ),
        ),
        2,
    )
    troubleshooting_hints: list[str] = []
    if conflicts:
        troubleshooting_hints.append(TROUBLESHOOTING_MATRIX["muddy_mix"])
    if missing_sections:
        troubleshooting_hints.append(TROUBLESHOOTING_MATRIX["overlong_lines"])
    if cliches or stats["repeated_lines"]:
        troubleshooting_hints.append(TROUBLESHOOTING_MATRIX["too_generic"])
    if "caption_metadata_leak" in str(track.get("tags") or "").lower():
        troubleshooting_hints.append(TROUBLESHOOTING_MATRIX["caption_metadata_leak"])
    quality_checks = {
        "caption_lyrics_metadata_separated": "pass" if not conflicts else "review",
        "target_language_script_respected": "pending",
        "no_placeholders": "pass" if "placeholder" not in lowered else "fail",
        "hook_has_title_or_emotional_promise": "pending",
        "section_map_matches_duration": "pass" if section_coverage >= 0.72 else "review",
        "genre_module_matches_caption": "pending",
        "negative_control_present": "pass",
        "runtime_fields_supported_or_advisory": "pass",
    }
    return {
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "length_plan": plan,
        "lyric_stats": stats,
        "length_ok": length_ok and char_ok and section_coverage >= 0.72,
        "length_score": round(min(1.0, stats["word_count"] / max(1, plan["target_words"])), 2),
        "duration_coverage_score": duration_coverage_score,
        "section_coverage": section_coverage,
        "missing_sections": missing_sections,
        "lyric_meta_tag_score": lyric_meta_tag_score,
        "char_ok": char_ok,
        "cliches": cliches,
        "repeated_lines": stats["repeated_lines"],
        "conflicts": conflicts,
        "quality_checks": quality_checks,
        "negative_control": negative_control_for(track.get("tags") or options.get("concept") or "", instrumental=bool(track.get("instrumental"))),
        "troubleshooting_hints": troubleshooting_hints,
        "hit_gate_passed": bool(length_ok and char_ok and section_coverage >= 0.72 and not cliches and len(stats["repeated_lines"]) <= 2),
    }


def production_team_report(track: dict[str, Any], options: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    tags = split_terms(track.get("tags") or track.get("caption") or "")
    duration = parse_duration_seconds(track.get("duration") or options.get("track_duration") or 120, 120)
    genre_hint = " ".join([str(track.get("tags") or ""), str(track.get("description") or ""), str(options.get("concept") or "")])
    modules = infer_genre_modules(genre_hint, max_modules=2)
    language = str(track.get("language") or options.get("language") or "en")
    preset = language_preset(language)
    instrumental = bool(track.get("instrumental")) or is_sparse_lyric_genre(genre_hint)
    lyric_plan = lyric_length_plan(
        duration,
        str(options.get("lyric_density") or "dense"),
        str(options.get("structure_preset") or "auto"),
        genre_hint,
    )
    return {
        "prompt_kit": {
            "version": PROMPT_KIT_VERSION,
            "language_preset": preset,
            "genre_modules": modules,
            "section_map": section_map_for(duration, genre_hint, instrumental=instrumental),
            "validation_checklist": VALIDATION_CHECKLIST,
            "advanced_generation_settings": ADVANCED_GENERATION_ADVISORY,
        },
        "final_model_policy": {
            "model": ALBUM_FINAL_MODEL if str(options.get("song_model_strategy") or "") == "xl_sft_final" else "per-model portfolio",
            "portfolio": album_model_portfolio(options.get("installed_models")),
            "locked": str(options.get("song_model_strategy") or "") in {"xl_sft_final", "all_models_album"},
            "reason": "Album renders use the selected policy: XL SFT legacy final or the full 7-model portfolio.",
        },
        "executive_producer": {
            "album_role": track.get("description") or "",
            "quality_target": options.get("quality_target") or "hit",
        },
        "artist_performer": {
            "vocal_language": language,
            "language_notes": preset["notes"],
            "performance_tags": [tag for tag in tags if "vocal" in tag.lower() or "rap" in tag.lower()][:6],
            "delivery_note": "Original persona, clear cadence, hook contrast.",
        },
        "songwriter": {
            "target_sections": lyric_plan["sections"],
            "target_words": lyric_plan["target_words"],
            "hook_intensity": options.get("hook_intensity", 0.85),
        },
        "rhyme_metaphor_editor": {
            "rhyme_density": options.get("rhyme_density", 0.8),
            "metaphor_density": options.get("metaphor_density", 0.7),
            "rules": ["dense internal/slant rhyme when appropriate", "one coherent image world", "remove generic filler"],
        },
        "beat_producer": {
            "bpm": track.get("bpm"),
            "key_scale": track.get("key_scale"),
            "time_signature": track.get("time_signature"),
            "arrangement": lyric_plan["structure"],
        },
        "ace_prompt_engineer": {
            "caption": track.get("tags") or "",
            "tag_list": track.get("tag_list") or [],
            "model_advice": model_info,
            "model_portfolio": album_model_portfolio(options.get("installed_models")),
        },
        "studio_engineer": {
            "inference_steps": track.get("inference_steps"),
            "guidance_scale": track.get("guidance_scale"),
            "shift": track.get("shift"),
            "infer_method": track.get("infer_method"),
            "sampler_mode": track.get("sampler_mode"),
            "audio_format": track.get("audio_format"),
            "auto_score": track.get("auto_score"),
            "auto_lrc": track.get("auto_lrc"),
            "return_audio_codes": track.get("return_audio_codes"),
        },
        "ar_quality_gate": {
            "checks": ["lyrics sufficient for duration", "section/performance tags present", "unique hook", "tag conflicts", "cliche guard", "strict generation JSON"],
            "status": "ready_for_album_render" if track.get("tool_report", {}).get("length_ok", True) else "repaired_or_needs_review",
        },
    }


def normalize_track(track: dict[str, Any], index: int, options: dict[str, Any]) -> dict[str, Any]:
    concept = str(options.get("sanitized_concept") or options.get("concept") or "")
    contract = options.get("user_album_contract")
    if not isinstance(contract, dict):
        contract = extract_user_album_contract(concept, options.get("num_tracks"), str(options.get("language") or "en"), options)
    contract_logs: list[str] = []
    track = apply_user_album_contract_to_track(track, contract, index, contract_logs)
    title = str(track.get("title") or f"Track {index + 1}").strip()[:80]
    duration = parse_duration_seconds(track.get("duration") or track.get("duration_seconds") or options.get("track_duration") or 120, 120)
    language = str(track.get("language") or options.get("language") or "en").strip().lower()
    tags = build_track_tags(
        " ".join([concept, str(track.get("tags") or ""), str(track.get("description") or "")]),
        index,
        options.get("tag_packs"),
        " ".join(split_terms(options.get("custom_tags")) + split_terms(track.get("custom_tags"))),
        options.get("negative_tags"),
    )
    raw_tags = split_terms(track.get("tags"))
    for raw in raw_tags:
        if raw.lower() not in {t.lower() for t in tags} and len(tags) < 12:
            tags.append(raw)
    bpm_strategy = str(options.get("bpm_strategy") or "varied")
    if not _is_nullish(track.get("bpm")):
        bpm = _int_or_default(track.get("bpm"), 92)
    elif bpm_strategy == "slow":
        bpm = 72 + (index % 3) * 4
    elif bpm_strategy == "club":
        bpm = 120 + (index % 4) * 3
    elif bpm_strategy == "rap":
        bpm = 86 + (index % 5) * 7
    else:
        bpm = 92 + (index * 9) % 48
    bpm = max(30, min(300, bpm))
    key_strategy = str(options.get("key_strategy") or "related")
    key_scale = normalize_key_scale(track.get("key_scale"), index, key_strategy)
    time_signature = normalize_time_signature(track.get("time_signature") or track.get("timesignature") or "4")
    lyrics = str(track.get("lyrics") or "").strip()
    lyrics = expand_lyrics_for_duration(
        title,
        concept,
        lyrics,
        duration,
        language,
        str(options.get("lyric_density") or "balanced"),
        str(options.get("structure_preset") or "auto"),
    )
    caption = polish_caption(tags, str(track.get("description") or ""), str(options.get("global_caption") or ""))
    installed_models = set(options.get("installed_models") or [])
    requested_track_model = str(track.get("song_model") or "").strip()
    final_locked = str(options.get("song_model_strategy") or "") == "xl_sft_final"
    multi_album = str(options.get("song_model_strategy") or "") == "all_models_album"
    if requested_track_model and not final_locked:
        model_info = choose_song_model(installed_models, "selected", requested_track_model)
    else:
        model_info = choose_song_model(
            installed_models,
            str(options.get("song_model_strategy") or "best_installed"),
            str(options.get("requested_song_model") or "auto"),
        )
    song_model = ALBUM_FINAL_MODEL if final_locked else (requested_track_model or model_info.get("model") or str(options.get("requested_song_model") or ""))
    if multi_album:
        song_model = str(options.get("planning_song_model") or requested_track_model or ALBUM_FINAL_MODEL)
        model_info = {
            "ok": True,
            "model": "per-model portfolio",
            "strategy": "all_models_album",
            "reason": MODEL_STRATEGIES["all_models_album"]["summary"],
            "album_models": album_model_portfolio(installed_models),
            "multi_album": True,
        }
    model_defaults = docs_best_model_settings(song_model)
    inference_steps = _int_or_default(track.get("inference_steps"), _number_or_default(options.get("inference_steps"), model_defaults["inference_steps"]))
    guidance_scale = _float_or_default(track.get("guidance_scale"), _number_or_default(options.get("guidance_scale"), model_defaults["guidance_scale"]))
    shift = _float_or_default(track.get("shift"), _number_or_default(options.get("shift"), model_defaults["shift"]))
    seed_value, seed_note = normalize_seed_value(_number_or_default(track.get("seed"), options.get("seed") or "-1"))
    tool_notes = " ".join(split_terms([note for note in [track.get("tool_notes"), seed_note] if note]))
    normalized = {
        "track_number": int(track.get("track_number") or index + 1),
        "artist_name": normalize_artist_name(
            track.get("artist_name") or track.get("artist"),
            derive_artist_name(title, concept, " ".join(tags), index),
        ),
        "title": title,
        "description": str(track.get("description") or album_arc(int(options.get("num_tracks") or 1))[min(index, int(options.get("num_tracks") or 1) - 1)]),
        "tags": caption,
        "tag_list": tags,
        "lyrics": lyrics,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "language": language,
        "duration": duration,
        "song_model": song_model,
        "seed": seed_value,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "infer_method": normalize_infer_method(track.get("infer_method") or options.get("infer_method") or model_defaults["infer_method"]),
        "sampler_mode": normalize_sampler_mode(track.get("sampler_mode") or options.get("sampler_mode") or model_defaults["sampler_mode"]),
        "audio_format": normalize_album_audio_format(track.get("audio_format") or options.get("audio_format") or model_defaults["audio_format"]),
        "auto_score": bool(track.get("auto_score", options.get("auto_score", False))),
        "auto_lrc": bool(track.get("auto_lrc", options.get("auto_lrc", False))),
        "return_audio_codes": bool(track.get("return_audio_codes", options.get("return_audio_codes", False))),
        "save_to_library": bool(track.get("save_to_library", options.get("save_to_library", True))),
        "use_format": bool(track.get("use_format", options.get("use_format", False))),
        "model_advice": model_info,
        "tool_notes": tool_notes,
        "duration_seconds": duration,
    }
    for field in [
        "locked_title",
        "source_title",
        "producer_credit",
        "engineer_credit",
        "artist_role",
        "style",
        "vibe",
        "narrative",
        "required_phrases",
        "content_policy_status",
        "input_contract_applied",
        "input_contract_version",
        "contract_repaired_fields",
        "contract_compliance",
    ]:
        if field in track:
            normalized[field] = track.get(field)
    if normalized.get("input_contract_applied"):
        normalized["input_contract_version"] = normalized.get("input_contract_version") or USER_ALBUM_CONTRACT_VERSION
        normalized["tool_notes"] = " ".join(
            split_terms(
                [
                    normalized.get("tool_notes"),
                    "Input contract applied; locked user-provided title and production brief preserved.",
                ]
            )
        )
    kit_hint = " ".join([caption, str(normalized.get("description") or ""), concept])
    instrumental = bool(track.get("instrumental")) or is_sparse_lyric_genre(kit_hint)
    kit_defaults = kit_metadata_defaults(
        mode=str(options.get("workflow_mode") or "text2music"),
        language=language,
        genre_hint=kit_hint,
        duration=duration,
        instrumental=instrumental,
    )
    kit_defaults["concept_summary"] = str(track.get("concept_summary") or concept or normalized.get("description") or "")[:300]
    kit_defaults["ace_caption"] = caption
    kit_defaults["lyrics"] = lyrics
    kit_defaults["metadata"] = {
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "duration": duration,
        "vocal_language": language,
    }
    kit_defaults["generation_settings"] = {
        "song_model": song_model,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "infer_method": normalize_infer_method(track.get("infer_method") or options.get("infer_method") or model_defaults["infer_method"]),
        "sampler_mode": normalize_sampler_mode(track.get("sampler_mode") or options.get("sampler_mode") or model_defaults["sampler_mode"]),
    }
    for field in PROMPT_KIT_METADATA_FIELDS:
        if field == "copy_paste_block":
            normalized[field] = str(track.get(field) or "")
        elif field in {"lyrics", "ace_caption", "metadata", "generation_settings"}:
            normalized[field] = kit_defaults.get(field)
        elif field in track and track.get(field) not in (None, ""):
            normalized[field] = track.get(field)
        elif field in kit_defaults:
            normalized[field] = kit_defaults[field]
    normalized["vocal_language"] = str(track.get("vocal_language") or kit_defaults.get("vocal_language") or language)
    normalized["instrumental"] = instrumental
    normalized["tool_report"] = quality_report(normalized, options)
    normalized["quality_checks"] = {
        **dict(normalized.get("quality_checks") or {}),
        **dict(normalized["tool_report"].get("quality_checks") or {}),
    }
    normalized["troubleshooting_hints"] = list(
        dict.fromkeys(
            list(normalized.get("troubleshooting_hints") or [])
            + list(normalized["tool_report"].get("troubleshooting_hints") or [])
        )
    )
    normalized["production_team"] = production_team_report(normalized, options, model_info)
    normalized["tool_report"]["production_team"] = normalized["production_team"]
    return normalized


def normalize_album_tracks(tracks: list[dict[str, Any]], options: dict[str, Any]) -> list[dict[str, Any]]:
    return [normalize_track(track, index, options) for index, track in enumerate(tracks)]


def build_album_plan(concept: str, num_tracks: int, track_duration: float, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = dict(options or {})
    sanitized, artist_notes = sanitize_artist_references(concept)
    contract = opts.get("user_album_contract")
    if not isinstance(contract, dict):
        contract = extract_user_album_contract(concept, num_tracks, str(opts.get("language") or "en"), opts)
    opts.update(
        {
            "concept": concept,
            "sanitized_concept": sanitized,
            "num_tracks": num_tracks,
            "track_duration": track_duration,
            "user_album_contract": contract,
        }
    )
    arcs = album_arc(num_tracks)
    terms = _subject_terms(sanitized) or ["signal", "night", "arrival", "pressure", "crown"]
    contract_tracks = tracks_from_user_album_contract(contract)
    tracks: list[dict[str, Any]] = [dict(track) for track in contract_tracks[: max(0, int(num_tracks))]]
    for index in range(max(1, int(num_tracks))):
        if index < len(tracks):
            tracks[index].setdefault("track_number", index + 1)
            tracks[index].setdefault(
                "description",
                tracks[index].get("narrative") or tracks[index].get("vibe") or arcs[min(index, len(arcs) - 1)],
            )
            continue
        core = terms[index % len(terms)].title()
        title = f"{core} Protocol" if index == 0 else f"{core} Season"
        description = f"{arcs[index]}; scene built around {terms[index % len(terms)]} and {terms[(index + 1) % len(terms)]}."
        tracks.append({"track_number": index + 1, "title": title, "description": description})
    tracks = apply_user_album_contract_to_tracks(tracks[: max(1, int(num_tracks))], contract)
    normalized = normalize_album_tracks(tracks, opts)
    toolkit_report = {
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "prompt_kit": prompt_kit_payload(),
        "artist_reference_notes": artist_notes,
        "tag_snapshot": {key: values[:12] for key, values in TAG_TAXONOMY.items()},
        "lyric_plan": lyric_length_plan(track_duration, str(opts.get("lyric_density") or "balanced"), str(opts.get("structure_preset") or "auto"), sanitized),
        "language_preset": language_preset(opts.get("language") or "en"),
        "genre_modules": infer_genre_modules(sanitized, max_modules=2),
        "section_map": section_map_for(track_duration, sanitized, instrumental=is_sparse_lyric_genre(sanitized)),
        "model_strategy": MODEL_STRATEGIES.get(str(opts.get("song_model_strategy") or "best_installed"), MODEL_STRATEGIES["best_installed"]),
        "album_model_portfolio": album_model_portfolio(opts.get("installed_models")),
        "final_model_policy": {
            "model": ALBUM_FINAL_MODEL if str(opts.get("song_model_strategy") or "") == "xl_sft_final" else "all_models_album",
            "locked": str(opts.get("song_model_strategy") or "") in {"xl_sft_final", "all_models_album"},
            "default_steps": docs_best_model_settings(ALBUM_FINAL_MODEL)["inference_steps"],
            "default_guidance_scale": docs_best_model_settings(ALBUM_FINAL_MODEL)["guidance_scale"],
            "default_shift": docs_best_model_settings(ALBUM_FINAL_MODEL)["shift"],
            "quality_preset": docs_best_model_settings(ALBUM_FINAL_MODEL)["quality_preset"],
        },
        "user_album_contract": contract,
        "input_contract_applied": bool(contract.get("applied")) if isinstance(contract, dict) else False,
        "input_contract_version": USER_ALBUM_CONTRACT_VERSION,
        "blocked_unsafe_count": int(contract.get("blocked_unsafe_count") or 0) if isinstance(contract, dict) else 0,
    }
    return {"tracks": normalized, "toolkit_report": toolkit_report}


def toolkit_payload(installed_models: set[str] | list[str] | None = None) -> dict[str, Any]:
    installed = set(installed_models or [])
    return {
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "prompt_kit": prompt_kit_payload(),
        "prompt_kit_metadata_fields": PROMPT_KIT_METADATA_FIELDS,
        "language_presets": LANGUAGE_PRESETS,
        "genre_modules": GENRE_MODULES,
        "troubleshooting_matrix": TROUBLESHOOTING_MATRIX,
        "validation_checklist": VALIDATION_CHECKLIST,
        "advanced_generation_settings": ADVANCED_GENERATION_ADVISORY,
        "sources": OFFICIAL_SOURCES,
        "tag_taxonomy": TAG_TAXONOMY,
        "lyric_meta_tags": LYRIC_META_TAGS,
        "craft_tools": CRAFT_TOOLS,
        "density_presets": DENSITY_PRESETS,
        "model_strategies": {
            key: {k: v for k, v in value.items() if k != "order"}
            for key, value in MODEL_STRATEGIES.items()
        },
        "model_strategy_order": {key: value["order"] for key, value in MODEL_STRATEGIES.items()},
        "album_model_portfolio": album_model_portfolio(installed),
        "installed_song_models": sorted(installed),
        "cliche_guard": BANNED_CLICHES,
        "artist_reference_policy": "Artist references are passed through freely.",
        "tag_policy": "ACE-Step captions are free-form; these official-style categories are a starter library, and Custom Tags/Negative Tags extend them.",
    }


def make_crewai_tools(context: dict[str, Any]) -> list[Any]:
    try:
        from crewai.tools import tool
    except Exception:
        return []

    @tool("ModelAdvisorTool")
    def model_advisor(query: str = "") -> str:
        """Choose the best installed ACE-Step model for album text2music generation."""
        return json.dumps(
            choose_song_model(
                set(context.get("installed_models") or []),
                str(context.get("song_model_strategy") or "best_installed"),
                str(context.get("requested_song_model") or "auto"),
            )
        )

    @tool("XLModelPolicyTool")
    def xl_model_policy(query: str = "") -> str:
        """Return the locked XL SFT final-render policy for album generation."""
        installed = ALBUM_FINAL_MODEL in set(context.get("installed_models") or [])
        final_defaults = docs_best_model_settings(ALBUM_FINAL_MODEL)
        return json.dumps(
            {
                "final_model": ALBUM_FINAL_MODEL,
                "locked": True,
                "installed": installed,
                "download_required": not installed,
                "default_steps": final_defaults["inference_steps"],
                "default_guidance_scale": final_defaults["guidance_scale"],
                "default_shift": final_defaults["shift"],
                "quality_preset": final_defaults["quality_preset"],
                "rule": "Plan any creative settings you want, but final album audio renders with XL SFT only.",
            }
        )

    @tool("ModelPortfolioTool")
    def model_portfolio_tool(query: str = "") -> str:
        """Return the full 7-model album portfolio and install state."""
        return json.dumps(
            {
                "strategy": "all_models_album",
                "render_policy": "For each model, render every planned track once as a complete model-specific album.",
                "models": album_model_portfolio(context.get("installed_models")),
            }
        )

    @tool("PerModelSettingsTool")
    def per_model_settings(model_name: str = "") -> str:
        """Return recommended generation settings for one ACE-Step album model."""
        requested = str(model_name or "").strip()
        models = album_model_portfolio(context.get("installed_models"))
        if requested:
            models = [item for item in models if item["model"] == requested or item["slug"] == requested] or models
        return json.dumps({"models": models})

    @tool("AlbumRenderMatrixTool")
    def album_render_matrix(track_count: str = "") -> str:
        """Return the track-by-model render matrix for a full album generation."""
        try:
            count = int(track_count or context.get("num_tracks") or 1)
        except ValueError:
            count = int(context.get("num_tracks") or 1)
        models = album_model_portfolio(context.get("installed_models"))
        return json.dumps(
            {
                "tracks": count,
                "models": len(models),
                "total_renders": count * len(models),
                "album_groups": [{"album_model": item["model"], "album_slug": item["slug"]} for item in models],
                "rule": "Same album plan and lyrics, rendered once per track per model for fair comparison.",
            }
        )

    @tool("FilenamePlannerTool")
    def filename_planner(spec: str = "") -> str:
        """Return safe filename patterns for track/model downloads."""
        return json.dumps(
            {
                "pattern": "01-track-title--acestep-v15-xl-sft--v1.wav",
                "rules": [
                    "include zero-padded track number",
                    "include sanitized track title",
                    "include ACE-Step model slug",
                    "include variant number",
                    "keep extensions from requested audio_format",
                ],
            }
        )

    @tool("TagLibraryTool")
    def tag_library(query: str = "") -> str:
        """Return ACE-Step caption dimensions, lyric meta tags, and curated tags."""
        return json.dumps({"tag_taxonomy": TAG_TAXONOMY, "lyric_meta_tags": LYRIC_META_TAGS})

    @tool("LyricLengthTool")
    def lyric_length(duration: str = "") -> str:
        """Return duration-aware lyric word, line, and section targets."""
        dur = parse_duration_seconds(duration or context.get("track_duration") or 120, 120)
        return json.dumps(
            lyric_length_plan(
                dur,
                str(context.get("lyric_density") or "balanced"),
                str(context.get("structure_preset") or "auto"),
                str(context.get("sanitized_concept") or context.get("concept") or ""),
            )
        )

    @tool("GenerationSettingsTool")
    def generation_settings(track_brief: str = "") -> str:
        """Return editable ACE-Step generation settings for one album track."""
        final_defaults = docs_best_model_settings(ALBUM_FINAL_MODEL)
        return json.dumps(
            {
                "song_model": ALBUM_FINAL_MODEL,
                "album_model_portfolio": album_model_portfolio(context.get("installed_models")),
                "seed": str(context.get("seed") or "-1"),
                "inference_steps": _int_or_default(context.get("inference_steps"), final_defaults["inference_steps"]),
                "guidance_scale": _float_or_default(context.get("guidance_scale"), final_defaults["guidance_scale"]),
                "shift": _float_or_default(context.get("shift"), final_defaults["shift"]),
                "infer_method": str(context.get("infer_method") or final_defaults["infer_method"]),
                "sampler_mode": str(context.get("sampler_mode") or final_defaults["sampler_mode"]),
                "audio_format": str(context.get("audio_format") or final_defaults["audio_format"]),
                "auto_score": bool(context.get("auto_score", False)),
                "auto_lrc": bool(context.get("auto_lrc", False)),
                "return_audio_codes": bool(context.get("return_audio_codes", False)),
                "save_to_library": bool(context.get("save_to_library", True)),
                "note": "Agents may tune every setting here except the locked final song_model.",
                "portfolio_note": "With all_models_album, these settings become per-track defaults and AceJAM applies model-specific steps/shift while rendering every model album.",
            }
        )

    @tool("ArrangementTool")
    def arrangement_tool(track_brief: str = "") -> str:
        """Plan song sections, energy movement, BPM, key, and time signature."""
        dur = parse_duration_seconds(context.get("track_duration") or 120, 120)
        sections = section_sequence(
            dur,
            str(context.get("structure_preset") or "auto"),
            rap=bool(re.search(r"\b(rap|hip.?hop|trap|drill)\b", str(track_brief or context.get("concept") or ""), re.I)),
        )
        return json.dumps(
            {
                "sections": sections,
                "structure": ", ".join(f"[{section}]" for section in sections),
                "bpm_strategy": context.get("bpm_strategy") or "varied",
                "key_strategy": context.get("key_strategy") or "related",
                "time_signature": "4",
                "arrangement_rule": "Give every track a different lift, drop, bridge, or outro moment.",
            }
        )

    @tool("VocalPerformanceTool")
    def vocal_performance(goal: str = "") -> str:
        """Return persona, cadence, ad-lib, harmony, and performance tag guidance."""
        cleaned, notes = sanitize_artist_references(goal)
        return json.dumps(
            {
                "technique_brief": cleaned,
                "artist_reference_notes": notes,
                "performance_tags": [
                    "[Verse - rap]",
                    "[Chorus - anthemic]",
                    "[Bridge - whispered]",
                    "[Chorus - layered vocals]",
                ],
                "delivery_rules": [
                    "clear original persona",
                    "breath-control planning",
                    "ad-libs only where they support the hook",
                ],
            }
        )

    @tool("RhymeFlowTool")
    def rhyme_flow(artist_or_goal: str = "") -> str:
        """Return flow and rhyme technique guidance."""
        return json.dumps({"technique_brief": str(artist_or_goal or ""), "notes": []})

    @tool("MetaphorWorldTool")
    def metaphor_world(seed: str = "") -> str:
        """Build a coherent metaphor world for one track."""
        source = " ".join([str(seed or ""), str(context.get("sanitized_concept") or context.get("concept") or "")])
        terms = _subject_terms(source) or ["signal", "city", "pressure", "weather"]
        anchors = terms[:4]
        return json.dumps(
            {
                "anchor_terms": anchors,
                "image_field": [
                    f"{anchors[0]} as architecture",
                    f"{anchors[min(1, len(anchors) - 1)]} as weather",
                    f"{anchors[min(2, len(anchors) - 1)]} as currency",
                ],
                "rules": [
                    "Use one image system per song.",
                    "Make verses concrete and hooks simple enough to remember.",
                    "Avoid mixing random nature, space, and city metaphors in the same hook.",
                ],
            }
        )

    @tool("HookDoctorTool")
    def hook_doctor(hook_or_title: str = "") -> str:
        """Score hook memorability, title connection, and repetition risk."""
        text = str(hook_or_title or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        words = _words(text)
        repeated = [line for line, count in Counter(line.lower() for line in lines).items() if count > 1]
        score = 0.35
        if 4 <= len(words) <= 42:
            score += 0.25
        if lines and max(len(line.split()) for line in lines) <= 10:
            score += 0.2
        if not repeated:
            score += 0.2
        return json.dumps(
            {
                "hook_score": round(min(1.0, score), 2),
                "repeated_lines": repeated[:4],
                "advice": "Keep the hook short, title-connected, contrastive, and rhythmically easy to repeat.",
            }
        )

    @tool("CaptionPolisherTool")
    def caption_polisher(tags: str = "") -> str:
        """Polish a tag list into a compact ACE-Step caption."""
        return polish_caption(tags, str(context.get("sanitized_concept") or ""))

    @tool("ClicheGuardTool")
    def cliche_guard(lyrics: str = "") -> str:
        """Flag lyric cliches and repeated lines."""
        return json.dumps(quality_report({"lyrics": lyrics, "duration": context.get("track_duration"), "tags": ""}, context))

    @tool("AlbumArcTool")
    def album_arc_tool(count: str = "") -> str:
        """Return opener, escalation, climax, cooldown, and closer roles."""
        try:
            track_count = int(count or context.get("num_tracks") or 1)
        except ValueError:
            track_count = int(context.get("num_tracks") or 1)
        return json.dumps({"arc": album_arc(track_count)})

    @tool("AlbumContinuityTool")
    def album_continuity(count: str = "") -> str:
        """Return sequencing, motifs, key movement, and contrast guidance."""
        try:
            track_count = int(count or context.get("num_tracks") or 1)
        except ValueError:
            track_count = int(context.get("num_tracks") or 1)
        return json.dumps(
            {
                "arc": album_arc(track_count),
                "key_cycle": KEY_CYCLE[: max(1, min(track_count, len(KEY_CYCLE)))],
                "motif_rules": [
                    "repeat one emotional motif across the album",
                    "change drum pocket or vocal delivery every track",
                    "reserve the biggest hook for the opener, midpoint, or closer",
                ],
            }
        )

    @tool("InspirationRadarTool")
    def inspiration_radar(query: str = "") -> str:
        """Return inspiration notes for the given query."""
        raw = query or context.get("inspiration_queries") or context.get("concept") or ""
        return json.dumps(
            {
                "query": str(raw),
                "artist_reference_notes": [],
                "web_notes": str(context.get("web_inspiration") or "")[:1200],
                "usage": "Use trends, scenes, topics, sonic cues, and artist references freely.",
            }
        )

    @tool("ConflictCheckerTool")
    def conflict_checker(spec: str = "") -> str:
        """Find tag, lyric, and metadata contradictions."""
        try:
            payload = json.loads(spec or "{}")
        except json.JSONDecodeError:
            payload = {"tags": spec, "lyrics": ""}
        return json.dumps(quality_report(payload if isinstance(payload, dict) else {"tags": spec}, context))

    @tool("MixMasterTool")
    def mix_master(spec: str = "") -> str:
        """Return mix/master, output, score, LRC, and audio-code recommendations."""
        return json.dumps(
            {
                "production_tags": ["high-fidelity", "crisp modern mix", "radio ready", "wide stereo"],
                "output": {
                    "audio_format": str(context.get("audio_format") or "wav"),
                    "auto_score": bool(context.get("auto_score", False)),
                    "auto_lrc": bool(context.get("auto_lrc", False)),
                    "return_audio_codes": bool(context.get("return_audio_codes", False)),
                },
                "qa": [
                    "lyrics are not in caption",
                    "BPM/key/time are metadata",
                    "caption is compact but specific",
                    "all_models_album renders the same album plan through every listed ACE-Step model",
                ],
            }
        )

    @tool("HitScoreTool")
    def hit_score(spec: str = "") -> str:
        """Score hook, lyric length, cliche risk, and generation readiness."""
        try:
            payload = json.loads(spec or "{}")
        except json.JSONDecodeError:
            payload = {"lyrics": spec, "tags": ""}
        report = quality_report(payload if isinstance(payload, dict) else {"lyrics": spec}, context)
        hook_score = 0.5
        if report.get("length_ok"):
            hook_score += 0.2
        if not report.get("cliches"):
            hook_score += 0.15
        if not report.get("repeated_lines"):
            hook_score += 0.15
        report["hit_score"] = round(min(1.0, hook_score), 2)
        return json.dumps(report)

    @tool("TrackRepairTool")
    def track_repair(spec: str = "") -> str:
        """Repair missing lyrics, weak hooks, tag conflicts, and absent generation fields."""
        try:
            payload = json.loads(spec or "{}")
        except json.JSONDecodeError:
            payload = {"title": "Untitled", "tags": spec}
        if not isinstance(payload, dict):
            payload = {"title": "Untitled", "tags": str(spec or "")}
        title = str(payload.get("title") or "Untitled")
        lyrics = str(payload.get("lyrics") or "")
        lyrics = expand_lyrics_for_duration(
            title,
            str(context.get("sanitized_concept") or context.get("concept") or ""),
            lyrics,
            parse_duration_seconds(context.get("track_duration") or 120, 120),
            str(context.get("language") or "en"),
            str(context.get("lyric_density") or "dense"),
            str(context.get("structure_preset") or "auto"),
        )
        final_defaults = docs_best_model_settings(ALBUM_FINAL_MODEL)
        repaired = {
            "title": title,
            "tags": polish_caption(payload.get("tags") or infer_core_tags(str(context.get("concept") or ""), 0)),
            "lyrics": lyrics,
            "song_model": ALBUM_FINAL_MODEL,
            "album_model_portfolio": album_model_portfolio(context.get("installed_models")),
            "inference_steps": _int_or_default(payload.get("inference_steps"), _number_or_default(context.get("inference_steps"), final_defaults["inference_steps"])),
            "guidance_scale": _float_or_default(payload.get("guidance_scale"), _number_or_default(context.get("guidance_scale"), final_defaults["guidance_scale"])),
            "shift": _float_or_default(payload.get("shift"), _number_or_default(context.get("shift"), final_defaults["shift"])),
        }
        repaired["quality_report"] = quality_report(repaired, context)
        return json.dumps(repaired)

    @tool("LanguagePresetTool")
    def language_preset_tool(language_code: str = "") -> str:
        """Return language/script policy and romanization guidance."""
        return json.dumps(language_preset(language_code or context.get("language") or "en"), ensure_ascii=True)

    @tool("GenreModuleTool")
    def genre_module_tool(genre_hint: str = "") -> str:
        """Return prompt-kit genre modules for a style hint."""
        hint = genre_hint or context.get("sanitized_concept") or context.get("concept") or ""
        return json.dumps({"modules": infer_genre_modules(hint, max_modules=3)}, ensure_ascii=True)

    @tool("SectionMapTool")
    def section_map_tool(spec: str = "") -> str:
        """Return a duration-realistic section map for vocal or instrumental music."""
        hint = " ".join([str(spec or ""), str(context.get("sanitized_concept") or context.get("concept") or "")])
        duration = parse_duration_seconds(context.get("track_duration") or context.get("duration") or 180, 180)
        return json.dumps(
            {
                "duration": int(duration),
                "section_map": section_map_for(duration, hint, instrumental=is_sparse_lyric_genre(hint)),
            },
            ensure_ascii=True,
        )

    @tool("IterationPlanTool")
    def iteration_plan_tool(goal: str = "") -> str:
        """Return a compact human-centered ACE-Step iteration plan."""
        return json.dumps(
            {
                "version": PROMPT_KIT_VERSION,
                "iteration_plan": [
                    "Generate one focused first pass.",
                    "Listen for hook clarity, language/script drift, vocal intelligibility, low-end balance, and section realism.",
                    "Revise one axis at a time: caption density, lyrics, section map, or generation settings.",
                    "Use Cover/Repaint/Complete when source audio structure should be preserved.",
                ],
                "goal": str(goal or context.get("quality_target") or "hit"),
            },
            ensure_ascii=True,
        )

    @tool("TroubleshootingTool")
    def troubleshooting_tool(issue: str = "") -> str:
        """Return prompt-kit troubleshooting hints."""
        key = str(issue or "").strip().lower().replace("-", "_").replace(" ", "_")
        if key and key in TROUBLESHOOTING_MATRIX:
            hints = {key: TROUBLESHOOTING_MATRIX[key]}
        else:
            hints = TROUBLESHOOTING_MATRIX
        return json.dumps({"version": PROMPT_KIT_VERSION, "hints": hints}, ensure_ascii=True)

    @tool("ValidationChecklistTool")
    def validation_checklist_tool(query: str = "") -> str:
        """Return prompt-kit validation checks."""
        return json.dumps(
            {
                "version": PROMPT_KIT_VERSION,
                "validation_checklist": VALIDATION_CHECKLIST,
                "metadata_fields": PROMPT_KIT_METADATA_FIELDS,
            },
            ensure_ascii=True,
        )

    @tool("NegativeControlTool")
    def negative_control_tool(genre_hint: str = "") -> str:
        """Return genre-aware negative control phrases."""
        hint = genre_hint or context.get("sanitized_concept") or context.get("concept") or ""
        return json.dumps(
            {
                "version": PROMPT_KIT_VERSION,
                "negative_control": negative_control_for(hint, instrumental=is_sparse_lyric_genre(hint)),
            },
            ensure_ascii=True,
        )

    return [
        model_advisor,
        model_portfolio_tool,
        per_model_settings,
        album_render_matrix,
        filename_planner,
        xl_model_policy,
        tag_library,
        lyric_length,
        generation_settings,
        arrangement_tool,
        vocal_performance,
        rhyme_flow,
        metaphor_world,
        hook_doctor,
        caption_polisher,
        cliche_guard,
        album_arc_tool,
        album_continuity,
        inspiration_radar,
        conflict_checker,
        mix_master,
        hit_score,
        track_repair,
        language_preset_tool,
        genre_module_tool,
        section_map_tool,
        iteration_plan_tool,
        troubleshooting_tool,
        validation_checklist_tool,
        negative_control_tool,
    ]
