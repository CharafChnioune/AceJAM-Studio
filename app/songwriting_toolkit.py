from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
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
from acestep.constants import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    TASK_TYPES,
    TRACK_NAMES,
    VALID_KEYSCALES,
    VALID_LANGUAGES,
    VALID_TIME_SIGNATURES,
)
from album_quality_gate import evaluate_album_payload_quality
from studio_core import (
    ACE_STEP_LYRICS_SAFE_HEADROOM,
    ACE_STEP_LYRICS_SOFT_TARGET_MAX,
    ACE_STEP_LYRICS_WARNING_CHAR_LIMIT,
    ace_step_settings_compliance,
    ace_step_settings_registry,
    docs_best_model_settings,
    fit_ace_step_lyrics_to_limit,
    hit_readiness_report,
    official_downloadable_model_ids,
    official_model_registry,
    official_render_model_ids,
    parse_bool,
    pro_quality_policy,
    runtime_planner_report,
)
from user_album_contract import (
    USER_ALBUM_CONTRACT_VERSION,
    apply_user_album_contract_to_tracks,
    apply_user_album_contract_to_track,
    extract_user_album_contract,
    tracks_from_user_album_contract,
)


OFFICIAL_SOURCES = [
    "https://github.com/ace-step/ACE-Step-1.5",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md",
    "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md",
    "https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5/blob/main/docs/en/API.md",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-sft",
    "https://huggingface.co/ACE-Step/acestep-v15-xl-base",
    "https://arxiv.org/abs/2602.00744",
]

ALBUM_FINAL_MODEL = "acestep-v15-xl-sft"
SONG_INTENT_SCHEMA_VERSION = "2026-05-03"


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
    _portfolio_item("acestep-v15-turbo-shift1", "turbo-shift1", "Turbo Shift1", "Richer details, looser semantics"),
    _portfolio_item("acestep-v15-turbo-shift3", "turbo-shift3", "Turbo Shift3", "Clear timbre, dry"),
    _portfolio_item("acestep-v15-turbo-continuous", "turbo-continuous", "Turbo Continuous", "Smooth continuous turbo"),
    _portfolio_item("acestep-v15-sft", "sft", "SFT", "CFG detail tuning"),
    _portfolio_item("acestep-v15-base", "base", "Base", "All tasks, fine-tuning"),
    _portfolio_item("acestep-v15-xl-turbo", "xl-turbo", "XL Turbo", "Best 20GB+ daily driver"),
    _portfolio_item(ALBUM_FINAL_MODEL, "xl-sft", "XL SFT", "Highest detail, CFG"),
    _portfolio_item("acestep-v15-xl-base", "xl-base", "XL Base", "All tasks, XL quality"),
]
ALBUM_MODEL_PORTFOLIO_MODELS = [item["model"] for item in ALBUM_MODEL_PORTFOLIO]

MODEL_STRATEGIES: dict[str, dict[str, Any]] = {
    "all_models_album": {
        "label": "All official render model albums",
        "summary": "Render one complete album for every official render-usable ACE-Step 1.5 model in AceJAM's album portfolio.",
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
    "sparse": {"label": "Sparse", "word_factor": 0.95, "line_factor": 0.95},
    "balanced": {"label": "Balanced", "word_factor": 1.30, "line_factor": 1.10},
    "dense": {"label": "Dense", "word_factor": 1.50, "line_factor": 1.20},
    "rap_dense": {"label": "Rap dense", "word_factor": 1.65, "line_factor": 1.30},
}

TAG_TAXONOMY: dict[str, list[str]] = {
    "genre_style": [
        "pop", "hip-hop", "rap", "trap", "drill", "drill UK", "drill NY", "melodic rap",
        "boom bap", "G-funk", "West Coast hip hop", "East Coast hip hop", "NYC street rap",
        "Detroit rap", "Memphis rap", "cloud rap", "phonk", "phonk drift", "afrodrill",
        "jersey club", "trap soul", "neo-soul", "R&B", "soul", "afrobeat", "afrobeats",
        "amapiano", "afrohouse", "dancehall", "reggaeton", "dembow", "garage", "house",
        "tech house", "techno", "melodic techno", "trance", "drum and bass", "dubstep",
        "EDM", "deconstructed club", "hyperpop", "synthwave", "vaporwave", "chillwave",
        "indie pop", "indie folk", "dream pop", "art pop", "electropop", "dark pop",
        "country pop", "rock", "alt rock", "punk", "punk rock", "metal", "shoegaze",
        "post-rock", "psychedelic", "grunge", "country", "bluegrass", "folk", "jazz",
        "modal jazz", "free jazz", "bebop", "chamber jazz", "lo-fi hip hop", "downtempo",
        "IDM", "classical", "chamber pop", "cinematic", "orchestral", "ambient", "gospel",
        "latin pop", "salsa", "son", "Afro-Cuban", "K-pop", "J-pop", "kawaii pop",
        "musical theatre", "spoken word",
    ],
    "mood_atmosphere": [
        "melancholic", "uplifting", "euphoric", "dark", "dreamy", "nostalgic", "intimate",
        "aggressive", "confident", "romantic", "cinematic", "tense", "hopeful", "bittersweet",
        "luxurious", "gritty", "warm", "cold", "neon-lit", "late night", "sunlit",
        "motivational", "inspirational", "empowering", "cheerful", "deadpan", "sarcastic",
        "ironic", "menacing", "triumphant", "vulnerable", "rebellious", "haunted",
        "pulsating", "urban", "bold", "playful", "dramatic", "urgent", "chaotic",
        "high energy", "low energy", "explosive", "building energy", "calm", "intense",
        "exciting", "soulful", "sad", "happy", "energetic",
    ],
    "instruments": [
        "piano", "grand piano", "Rhodes", "electric piano", "organ", "clavinet", "mellotron",
        "wurlitzer", "harpsichord", "acoustic guitar", "clean electric guitar",
        "distorted guitar", "nylon guitar", "bass guitar", "upright bass", "808 bass",
        "sub-bass", "synth bass", "303 acid bass", "trap hi-hats", "808 kick",
        "punchy snare", "rim click", "cowbell", "808 cowbell", "shaker", "tambourine",
        "breakbeat", "drum machine", "brush drums", "synth pads", "analog synth",
        "lead synth", "arpeggiated synth", "arpeggiator", "strings", "violin", "cello",
        "harp", "mandolin", "brass", "trumpet", "saxophone", "trombone", "french horn",
        "flute", "oboe", "clarinet", "choir", "turntable scratches", "risers",
        "glitch effects", "talkbox", "vocoder", "kalimba", "accordion", "congas",
        "bongos", "timpani", "vibraphone", "soul sample chops", "dusty piano sample",
        "horn stab", "string stab",
    ],
    "timbre_texture": [
        "warm", "bright", "crisp", "airy", "punchy", "lush", "raw", "polished", "gritty",
        "wide stereo", "close-mic", "tape saturation", "vinyl texture", "deep low end",
        "silky top end", "dry vocal", "wet reverb", "analog warmth", "muddy", "dusty",
        "smoky", "metallic", "resonant", "hollow", "velvety", "saturated", "glossy",
    ],
    "era_reference": [
        "60s soul", "70s soul", "70s funk", "80s synth pop", "80s pop polish",
        "90s boom bap", "early 90s boom bap", "90s G-funk", "90s R&B", "90s grunge",
        "2000s pop punk", "early 2000s crunk", "2000s rap", "2010s EDM",
        "late 2010s trap", "modern trap", "2020s phonk drift", "future garage",
        "vintage soul", "classic house",
    ],
    "production_style": [
        "high-fidelity", "studio polished", "crisp modern mix", "lo-fi texture",
        "warm analog mix", "club master", "radio ready", "atmospheric",
        "minimal arrangement", "layered production", "cinematic build",
        "hard-hitting drums", "sidechain pulse", "bedroom pop", "live recording",
        "tape worn", "vinyl crackle", "head-nod groove", "summer banger polish",
        "dusty mix", "stripped mix", "raw demo feel", "big reverb tail",
        "gated reverb", "telephone EQ vocal",
    ],
    "vocal_character": [
        "male vocal", "female vocal", "male rap vocal", "female rap vocal",
        "melodic rap vocal", "autotune vocal", "auto-tune", "breathy vocal",
        "raspy vocal", "powerful belt", "falsetto", "stacked harmonies",
        "choir vocals", "spoken vocal", "whispered vocal",
        "mumble rap", "chopper rap", "lyrical rap", "trap flow", "double-time rap",
        "syncopated flow", "melodic flow", "storytelling flow", "punchline rap",
        "freestyle flow", "deadpan delivery", "comedic rap vocal",
        "bright vocal", "dark vocal", "warm vocal", "cold vocal", "nasal vocal",
        "gritty vocal", "smooth vocal", "husky vocal", "metallic vocal",
        "whispery vocal", "resonant vocal", "smoky vocal", "sultry vocal",
        "ethereal vocal", "hollow vocal", "velvety vocal", "shrill vocal",
        "mellow vocal", "thin vocal", "thick vocal", "reedy vocal", "silvery vocal",
        "twangy vocal", "vocoder vocal", "chopped vocal", "pitched-up vocal",
        "pitched-down vocal", "ad-libs", "shouted vocal", "narration", "spoken word",
        "auto-tune trap vocal", "Chinese-language vocal", "Spanish-language vocal",
        "Dutch-language vocal", "Arabic-language vocal",
        "whispered", "shouted", "harmonies", "harmonized", "call-and-response",
        "layered vocals", "raspy", "breathy", "soft", "powerful belting",
        "soft vocal", "powerful vocal",
    ],
    "speed_rhythm": [
        "slow tempo", "mid-tempo", "fast-paced", "fast", "groovy", "driving rhythm",
        "laid-back groove", "swing feel", "four-on-the-floor", "half-time drums",
        "syncopated rhythm", "head-nod groove", "trap bounce", "drill bounce",
        "double-time hi-hats", "shuffled hi-hats", "swung sixteenths",
        "behind-the-beat groove", "ahead-of-the-beat groove", "dembow groove",
        "afrohouse groove", "building tempo",
    ],
    "structure_hints": [
        "building intro", "catchy chorus", "anthemic hook", "dramatic bridge",
        "explosive drop", "breakdown", "beat switch", "fade-out ending",
        "stripped outro", "call and response", "chant hook", "headline hook",
        "punchline outro", "crowd chant", "cinematic bridge", "intimate verse",
        "explosive chorus", "final chorus lift",
    ],
    "track_stems": [
        "woodwinds", "brass", "fx", "synth", "strings", "percussion", "keyboard", "guitar",
        "bass", "drums", "backing_vocals", "vocals",
    ],
}

LYRIC_META_TAGS: dict[str, list[str]] = {
    "basic_structure": [
        "[Intro]", "[Verse]", "[Verse 1]", "[Verse 2]", "[Verse 3]",
        "[Pre-Chorus]", "[Chorus]", "[Post-Chorus]", "[Hook]", "[Hook/Chorus]",
        "[Refrain]", "[Bridge]", "[Final Chorus]", "[Outro]", "[Interlude]",
    ],
    "dynamic_sections": [
        "[Build]", "[Build-Up]", "[Drop]", "[Final Drop]", "[Breakdown]",
        "[Climax]", "[Fade Out]", "[Silence]", "[Beat Switch]",
    ],
    "instrumental_sections": [
        "[Instrumental]", "[inst]", "[Instrumental Break]",
        "[Synth Solo]", "[Guitar Solo]", "[Piano Solo]", "[Piano Interlude]",
        "[Brass Break]", "[Saxophone Solo]", "[Violin Solo]", "[Drum Break]",
    ],
    "performance_modifiers": [
        "[Verse - rap]", "[Verse - melodic rap]", "[Verse - double time rap]",
        "[Verse - whispered]", "[Verse - spoken]", "[Verse - shouted]",
        "[Verse - powerful]", "[Verse - falsetto]", "[Verse - crooned]",
        "[Chorus - anthemic]", "[Chorus - rap]", "[Chorus - layered vocals]",
        "[Chorus - chant]", "[Chorus - whispered]", "[Chorus - call and response]",
        "[Bridge - whispered]", "[Bridge - spoken]", "[Bridge - emotional]",
        "[Intro - dreamy]", "[Intro - dark]", "[Intro - spoken]",
        "[Intro - ambient]", "[Intro - piano]", "[Intro - talkbox]",
        "[Outro - fade out]", "[Outro - spoken]", "[Outro - acapella]",
        "[Outro - talkbox]", "[Climax - powerful]", "[Hook - sung]",
        "[Hook - chant]",
    ],
}

CRAFT_TOOLS: list[dict[str, str]] = [
    {"name": "ModelAdvisorTool", "summary": "Chooses only installed ACE-Step models for the album strategy."},
    {"name": "ModelPortfolioTool", "summary": "Plans the full official render-model album portfolio."},
    {"name": "PerModelSettingsTool", "summary": "Returns per-model steps, guidance, shift, speed, quality, and warnings."},
    {"name": "AlbumRenderMatrixTool", "summary": "Calculates track-by-model render counts and album grouping."},
    {"name": "FilenamePlannerTool", "summary": "Plans safe track/model filenames for downloads and album ZIPs."},
    {"name": "XLModelPolicyTool", "summary": "Locks final album renders to ACE-Step XL SFT and explains download/runtime requirements."},
    {"name": "TagLibraryTool", "summary": "Provides ACE-Step caption dimensions and lyric meta tags."},
    {"name": "AceStepPromptContractTool", "summary": "Returns the strict ACE-Step track payload prompt contract for album agents."},
    {"name": "LyricCounterTool", "summary": "Deterministically counts words, lyric lines, chars, sections, and hooks."},
    {"name": "TagCoverageTool", "summary": "Checks that caption/tags cover all required ACE-Step sound dimensions."},
    {"name": "CaptionIntegrityTool", "summary": "Checks captions for lyric, prompt, JSON, and metadata leakage."},
    {"name": "PayloadGateTool", "summary": "Runs the same album payload quality gate used before ACE-Step renders."},
    {"name": "LyricLengthTool", "summary": "Plans sections, words, and lines for the chosen duration."},
    {"name": "GenerationSettingsTool", "summary": "Builds editable per-track seed, steps, guidance, shift, sampler, and format settings."},
    {"name": "ArrangementTool", "summary": "Plans intro, verses, hooks, bridge, outro, BPM/key/time, and energy movement."},
    {"name": "VocalPerformanceTool", "summary": "Creates persona, cadence, vocal-response, harmony, and lyric performance tags."},
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
    {"name": "AceStepSettingsPolicyTool", "summary": "Returns official/default/docs-recommended/AceJAM settings policy for ACE-Step payloads."},
    {"name": "ChartMasterProfileTool", "summary": "Returns final-render Chart Master defaults and take-count policy."},
    {"name": "AceStepCoverageAuditTool", "summary": "Audits ACE-Step params, config, API endpoints, runtime controls, helpers, and result fields."},
    {"name": "EffectiveSettingsTool", "summary": "Shows effective active/ignored/unsupported settings for a proposed payload."},
    {"name": "HitReadinessTool", "summary": "Checks caption, lyrics, metadata, and runtime budgets before ACE-Step renders."},
    {"name": "RuntimePlannerTool", "summary": "Estimates model/backend/take-count runtime risk before final generation."},
    {"name": "AandRVariantPlanTool", "summary": "Plans multiple single-song takes for A&R selection while keeping album renders bounded."},
    {"name": "TaskApplicabilityTool", "summary": "Explains which ACE-Step controls are active, ignored, or source-locked for a task."},
    {"name": "ModelCompatibilityTool", "summary": "Checks model/task/settings compatibility before finalizing generation controls."},
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
    "travis scott": "psychedelic vocal textures, atmospheric trap, texture-first hooks, floating cadence",
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


PROMPT_KIT_DICT_FIELDS = {
    "genre_profile",
    "metadata",
    "generation_settings",
    "runtime_profile",
    "advanced_generation_settings",
    "quality_checks",
    "settings_compliance",
    "hit_readiness",
    "effective_settings",
    "settings_coverage",
    "audio_quality_audit",
    "metadata_adherence",
    "recommended_take",
    "tag_coverage",
    "caption_integrity",
    "lyric_duration_fit",
    "payload_quality_gate",
}

PROMPT_KIT_LIST_FIELDS = {
    "section_map",
    "iteration_plan",
    "community_risk_notes",
    "troubleshooting_hints",
    "variations",
    "negative_control",
    "genre_modules",
    "repair_actions",
}


def _coerce_prompt_kit_field(field: str, value: Any, default: Any = None) -> Any:
    if field == "prompt_kit_version":
        return PROMPT_KIT_VERSION
    if field in PROMPT_KIT_DICT_FIELDS:
        return _dict_or_empty(value) or _dict_or_empty(default)
    if field in PROMPT_KIT_LIST_FIELDS:
        items = _list_or_empty(value)
        if not items and not isinstance(value, (dict, list, tuple, set)):
            items = split_terms(value)
        return items or _list_or_empty(default)
    if field in {"lyrics", "ace_caption", "copy_paste_block"}:
        return str(value or default or "")
    if value in (None, ""):
        return default
    return str(value) if field in PROMPT_KIT_METADATA_FIELDS else value


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.lower(), flags=re.UNICODE)


def _subject_terms(text: str) -> list[str]:
    stop = {
        "a", "an", "and", "the", "to", "of", "in", "on", "with", "for", "about", "song",
        "album", "track", "music", "make", "like", "style", "ft", "feat", "featuring",
        "you", "your", "yours", "we", "our", "they", "them", "their", "was", "were",
        "are", "is", "this", "that", "these", "those", "vibe", "verse", "chorus",
        "lyrics", "narrative", "produced", "prod", "man", "woman", "one", "two",
        "pop", "funk", "city", "upbeat", "polished", "radio", "ready", "clean",
        "lead", "vocal", "stacked", "harmonies", "groovy", "bright", "crisp",
        "must", "feel", "complete", "singable", "placeholder", "copied",
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
    if rap and dur <= 150:
        return ["Intro", "Verse 1 - rap", "Chorus - rap", "Verse 2", "Bridge", "Final Chorus", "Outro"]
    if dur <= 150:
        return ["Intro", "Verse 1 - rap" if rap else "Verse 1", "Pre-Chorus", "Chorus - rap" if rap else "Chorus", "Verse 2", "Bridge", "Final Chorus", "Outro"]
    if rap and dur <= 240:
        return ["Intro", "Verse 1 - rap", "Chorus - rap", "Verse 2", "Second Chorus", "Verse 3 - Beat Switch", "Bridge", "Final Chorus", "Outro"]
    if dur <= 240:
        return ["Intro", "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Verse 3", "Final Chorus", "Outro"]
    if rap and dur <= 360:
        return ["Intro", "Verse 1 - rap", "Chorus - rap", "Verse 2", "Second Chorus", "Verse 3 - Beat Switch", "Bridge", "Final Chorus", "Outro"]
    if dur <= 360:
        return ["Intro", "Verse 1 - rap" if rap else "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Verse 3", "Breakdown", "Final Chorus", "Outro"]
    return ["Intro", "Verse 1 - rap" if rap else "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Verse 3", "Instrumental Break", "Verse 4", "Final Chorus", "Outro"]


def lyric_length_plan(duration: float, density: str = "balanced", structure_preset: str = "auto", genre_hint: str = "") -> dict[str, Any]:
    dur = int(parse_duration_seconds(duration, 120))
    rap = bool(re.search(r"\b(rap|hip.?hop|trap|drill|grime)\b", genre_hint or "", re.I))
    if rap and str(density or "").strip().lower() not in {"sparse", "instrumental"}:
        density = "rap_dense"
    preset = DENSITY_PRESETS.get(density, DENSITY_PRESETS["balanced"])
    sparse_genre = bool(is_sparse_lyric_genre(genre_hint) and not rap)
    if sparse_genre:
        section_map = section_map_for(dur, genre_hint, instrumental=True)
        sections = [str(item.get("tag") or "").strip("[]") for item in section_map if item.get("tag")]
        density = "sparse"
        preset = DENSITY_PRESETS["sparse"]
    else:
        sections = section_sequence(dur, structure_preset, rap=rap)
    # Bands tuned to push lyrics fuller while keeping char count under
    # ACE_STEP_LYRICS_SOFT_TARGET_MAX (3600) -> ~600 words at ~6 chars/word.
    bands = [
        (30, 55, 75, 95),
        (60, 120, 155, 195),
        (120, 240, 300, 360),
        (180, 340, 420, 500),
        (240, 420, 510, 580),
        (300, 480, 570, 640),
        (600, 540, 620, 660),
    ]
    if rap:
        bands = [
            (30, 70, 90, 110),
            (60, 150, 190, 230),
            (120, 290, 360, 430),
            (180, 410, 500, 580),
            (240, 490, 570, 630),
            (300, 540, 600, 650),
            (600, 580, 630, 660),
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
    density_factor = {"sparse": 0.9, "balanced": 1.0, "dense": 1.1, "rap_dense": 1.08}.get(density, 1.0)
    target_words = max(min_words, min(max_words, int(base_words * density_factor)))
    target_lines = max(len(sections) * 3, int(target_words / (4.6 if rap else 5.4) * float(preset["line_factor"])))
    min_lines = max(len(sections) * 2, int(target_lines * 0.72))
    if rap:
        rap_line_bands = [
            (60, 30, 42),
            (120, 56, 74),
            (180, 78, 96),
            (240, 92, 112),
            (300, 104, 124),
            (600, 116, 136),
        ]
        rap_min_lines, rap_target_lines = rap_line_bands[-1][1:]
        for limit, low, high in rap_line_bands:
            if dur <= limit:
                rap_min_lines, rap_target_lines = low, high
                break
        target_lines = max(len(sections) * 3, min(rap_target_lines, max(rap_min_lines, int(target_words / 5.6))))
        min_lines = max(len(sections) * 2, rap_min_lines)
    if sparse_genre:
        target_lines = max(len(sections), min(len(sections) * 3, target_lines))
        min_lines = max(0 if min_words == 0 else len(sections), int(target_lines * 0.5))
    # Bar allocation per section type. 1 bar = 4 beats. Rap verses lock to a
    # 16-bar floor on tracks >=120s (the standard hip-hop verse length); on
    # shorter tracks fall back to a target_lines-derived practical floor.
    rap_verse_bar_target = 16 if dur >= 120 else max(8, min(16, target_lines // max(1, len(sections) - 1)))
    sung_verse_bar_target = max(8, min(16, target_words // 60)) if not sparse_genre else max(4, min(8, target_lines // 2))
    hook_bar_target = 8 if dur > 120 else 4
    bars_per_section_template = {
        "Verse_rap": rap_verse_bar_target,
        "Verse_sung": sung_verse_bar_target,
        "Hook": hook_bar_target,
        "Chorus": hook_bar_target,
        "Pre-Chorus": 4,
        "Bridge": 4,
        "Intro": 2,
        "Outro": 2,
        "Instrumental": 4,
    }
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
        "bars_per_section": bars_per_section_template,
        "min_bars_per_rap_verse": rap_verse_bar_target,
        "bars_per_line_factor": 1.0 if rap else 1.4,
        "max_lyrics_chars": 4096,
        "safe_lyrics_char_target": ACE_STEP_LYRICS_SOFT_TARGET_MAX,
        "warning_lyrics_chars": ACE_STEP_LYRICS_WARNING_CHAR_LIMIT,
        "safe_headroom_chars": ACE_STEP_LYRICS_SAFE_HEADROOM,
        "duration_coverage_note": (
            "At very long durations ACE-Step's lyric cap limits continuous vocals; use enough sections plus intentional instrumental breaks."
            if dur >= 240
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


def album_models_for_strategy(
    strategy: str,
    installed_models: set[str] | list[str] | None = None,
    requested_model: str | None = None,
) -> list[dict[str, Any]]:
    if strategy == "all_models_album":
        return album_model_portfolio(installed_models)
    if strategy == "xl_sft_final":
        return [item for item in album_model_portfolio(installed_models) if item["model"] == ALBUM_FINAL_MODEL]
    requested = str(requested_model or "").strip()
    selection_strategy = "selected" if strategy in {"selected", "single_model_album"} else strategy
    info = choose_song_model(installed_models or [], selection_strategy, requested if selection_strategy == "selected" else "auto")
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
    if re.search(r"\b(schlager|accordion|akkordeon|brass|polka|volksmusik)\b", lowered):
        base = [
            "German schlager pop",
            "steady dance groove",
            "sparkling accordion",
            "bright brass stabs",
            "warm lead vocal",
            "uplifting singalong chorus",
            "clean radio-ready mix",
        ]
    elif re.search(r"\b(rap|hip.?hop|trap|drill|bars)\b", lowered):
        base = ["hip-hop", "808 bass", "trap hi-hats", "male rap vocal", "crisp modern mix"]
    elif "r&b" in lowered or "soul" in lowered:
        base = ["R&B", "Rhodes", "sub-bass", "breathy vocal", "warm analog mix"]
    elif "city-pop" in lowered or "city pop" in lowered or "funk" in lowered:
        base = ["pop", "funk", "groovy", "rubber bass", "bright piano", "clean lead vocal", "stacked harmonies", "radio ready"]
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


SONIC_CAPTION_TERM_RE = re.compile(
    r"\b(?:pop|rap|hip-hop|hip hop|trap|drill|g-funk|r&b|soul|rock|metal|punk|house|techno|trance|"
    r"dancehall|reggaeton|afro|amapiano|cinematic|ambient|schlager|latin|j-pop|k-pop|"
    r"groove|bounce|swing|boom-bap|drums?|hi-hats?|snare|kick|percussion|shuffle|rhythm|"
    r"bass|808|sub-bass|low-end|synth|piano|guitar|strings|brass|horns?|sirens?|accordion|organ|"
    r"vocal|voice|sung|singer|choir|harmony|hook|chorus|chant|dark|bright|uplifting|melancholic|"
    r"euphoric|dreamy|nostalgic|aggressive|romantic|hopeful|tense|warm|cold|gritty|emotional|"
    r"build|drop|bridge|anthemic|breakdown|dynamic|riser|climax|intro|outro|stadium|mix|master|"
    r"polished|crisp|clean|wide stereo|high-fidelity|radio|studio|glossy|punchy|analog|west coast)\b",
    re.I,
)


def _caption_term_is_compact(term: Any) -> bool:
    text = re.sub(r"\s+", " ", str(term or "")).strip(" .")
    if not text:
        return False
    if len(text) > 64:
        return False
    if re.search(r"[\n\r{}]|\b(?:album|track\s+\d+|verse|lyrics|metadata|json|bpm|key|duration|seed)\s*:", text, re.I):
        return False
    if re.search(r"[.!?]", text):
        return False
    words = re.findall(r"[A-Za-z0-9À-ÿ\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff']+", text)
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
    return True


def polish_caption(tags: Any, description: str = "", global_caption: str = "", *, strict: bool = False) -> str:
    terms: list[str] = []
    sources = [tags]
    if not strict:
        sources.extend([description, global_caption])
    for source in sources:
        for term in split_terms(source):
            if not _caption_term_is_compact(term):
                continue
            key = term.lower()
            if key not in {existing.lower() for existing in terms}:
                terms.append(term)
    caption = ", ".join(terms)
    caption = re.sub(r"\b\d{2,3}\s*bpm\b", "", caption, flags=re.I)
    caption = re.sub(r"\s{2,}", " ", caption).strip(" ,.")
    return caption[:512]


def _fallback_lines(language: str, section: str, title: str, terms: list[str], metaphor: str, rap: bool, count: int) -> list[str]:
    subject = terms[count % len(terms)] if terms else "moment"
    accent = terms[(count + 1) % len(terms)] if len(terms) > 1 else "city"
    hook = re.sub(r"[^A-Za-z0-9 ']", "", title).strip()
    if not _subject_terms(hook):
        hook = "The Moment"
    hook = hook or "All The Way Up"
    # Use count to rotate through different line templates for variety across tracks
    variant = count % 4
    if language == "nl":
        nl_chorus = [
            [f"{hook} blijft hangen in de nacht", f"Elke stap klinkt harder dan gedacht", f"Wij bouwen vuur uit koude steen", f"Tot de hele straat met ons mee beweegt"],
            [f"{hook} breekt door het plafond", f"Geen weg terug van deze grond", f"De bass slaat aan, het dak gaat af", f"Tot de speakers barsten in de stad"],
            [f"{hook} snijdt door de ruis", f"Vanuit het donker naar het licht thuis", f"Wij dragen goud in elke wond", f"En de hele buurt voelt wat ik vond"],
            [f"{hook} rolt door het beton", f"Geen slaap tot de volgende zon", f"De mic staat aan, de wereld stopt", f"Tot de laatste bar is afgedropt"],
        ]
        nl_verse = [
            [f"Half drie, {subject} op mijn jas", f"{accent.capitalize()} in mijn hoofd, ik geef geen pas", f"Elke bar heeft tanden in de beat", f"Ik maak van druk een kroon die niemand ziet"],
            [f"Vijf uur, de stad voelt dicht en koud", f"{subject.capitalize()} ligt zwaar maar ik hou vast", f"Elke lijn wordt scherper dan de vorige", f"Van {accent} bouw ik iets onmogelijks"],
            [f"De nacht breekt open, {subject} brandt", f"{accent.capitalize()} echo's door een leeg land", f"Pen op papier, de ink is bloed", f"Elk woord bewijst dat ik het nog kan"],
            [f"Zeven hoog, {subject} in de wind", f"{accent.capitalize()} droomt van wat ik ooit begin", f"De straat is stil maar ik schreeuw door", f"Van onderaf recht naar het koor"],
        ]
        if "Chorus" in section:
            return nl_chorus[variant]
        if "Bridge" in section:
            return [f"Even stil, de kamer ademt mee", f"{metaphor.capitalize()} trekt een lijn door wat ik deed"]
        return nl_verse[variant]
    en_chorus = [
        [f"{hook} rings out in the room", f"We turn pressure into perfume", f"Hands up when the low end blooms", f"One more night and we break through"],
        [f"{hook} echoes off the wall", f"We built this thing from nothing at all", f"Feel the ground shake when the bass calls", f"We don't stop until the curtain falls"],
        [f"{hook} cuts right through the noise", f"We traded silence for a louder voice", f"Every scar became a choice", f"Now the whole block hears us rejoice"],
        [f"{hook} hits different in the dark", f"We lit a fire from a single spark", f"No turning back once we made our mark", f"The city knows us by the art"],
    ]
    en_bridge = [
        [f"The {metaphor} bends but it never breaks", f"I hear the truth in the breath it takes"],
        [f"Somewhere between the {metaphor} and the rain", f"I found the part that kills the pain"],
        [f"They said the {metaphor} would crack the frame", f"But I rebuilt it and changed the game"],
        [f"Under the {metaphor}, the silence spoke", f"And every chain around me broke"],
    ]
    en_rap = [
        [f"Back door click with the {subject} on tilt", f"Cold chain swing where the old doubt built", f"Big dream stitched in a small room quilt", f"Every rhyme hits clean, no filler, no guilt"],
        [f"Street lamp buzz and the {subject} glows", f"Pocket full of bars that nobody knows", f"Stack the lines up, watch the pressure grow", f"From the concrete up, that's how we flow"],
        [f"Three AM, {subject} on my mind", f"Pen hits paper, leave the doubt behind", f"Every verse I spit is one of a kind", f"From the bottom up, we climb and grind"],
        [f"Dead end road but the {subject} speaks", f"Mic check one two, hear the floorboard creak", f"No handouts, built this week by week", f"Crown heavy but I never feel weak"],
    ]
    en_vocal = [
        [f"Morning lifts {subject} from the floor", f"{accent.capitalize()} light moves softly through the door", f"I turn the page and ask for more", f"Every note becomes the reason"],
        [f"Midnight paints the {subject} in the glass", f"{accent.capitalize()} breeze reminds me time moves fast", f"I held on tight but I couldn't last", f"Now I build a future from the past"],
        [f"Sunrise burns the {subject} clean", f"{accent.capitalize()} shadows fade from what I've seen", f"Every wound becomes a place I've been", f"Now I write the ending to this scene"],
        [f"Twilight carries {subject} through the air", f"{accent.capitalize()} noise dissolves and I don't care", f"I stripped it down, left the bones laid bare", f"Now the music is my only prayer"],
    ]
    if "Chorus" in section:
        return en_chorus[variant]
    if "Bridge" in section:
        return en_bridge[variant]
    if rap:
        return en_rap[variant]
    return en_vocal[variant]


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
    for section_index, section in enumerate(plan["sections"]):
        lines_needed = max(2, int(plan["target_lines"] / max(1, len(plan["sections"]))))
        if "Chorus" in section:
            lines_needed = max(4, lines_needed)
        if "Verse" in section:
            lines_needed = max(6 if rap else 5, lines_needed)
        section_key = _section_key(section)
        count = section_counts[section_key] + section_index
        section_counts[section_key] += 1
        lines: list[str] = []
        variant_round = 0
        while len(lines) < lines_needed:
            lines.extend(_fallback_lines(language, section, title, terms, metaphor, rap, count + variant_round))
            variant_round += 1
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


def _section_label_key(section: str) -> str:
    text = re.sub(r"[\[\]]", "", str(section or "")).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*-\s*(?:rap|anthemic|extension|reprise)\b.*$", "", text)
    return text


def trim_lyrics_to_limit(lyrics: str, limit: int = 4096) -> str:
    text = str(lyrics or "").strip()
    if len(text) <= limit:
        return text
    return fit_ace_step_lyrics_to_limit(text, limit)


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
    if stats["section_count"] == 0 and (stats["word_count"] < plan["min_words"] or stats["line_count"] < plan["min_lines"]):
        return build_fallback_lyrics(title, concept, duration, language, density, structure_preset)
    covered_labels = {_section_label_key(section) for section in stats["sections"]}
    missing_labels = [_section_label_key(section) for section in plan["sections"] if _section_label_key(section) not in covered_labels]
    extras: list[str] = []
    fallback = build_fallback_lyrics(title, concept, duration, language, density, structure_preset)
    fallback_blocks = re.split(r"\n\s*\n", fallback)
    for block in fallback_blocks:
        match = re.match(r"\[([^\]]+)\]", block.strip())
        if not match:
            continue
        label_key = _section_label_key(match.group(1))
        if label_key in missing_labels:
            extras.append(block.strip())
            covered_labels.add(label_key)
            missing_labels = [item for item in missing_labels if item != label_key]
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
    settings_compliance = ace_step_settings_compliance(
        track,
        task_type=str(track.get("task_type") or options.get("task_type") or "text2music"),
        song_model=str(track.get("song_model") or options.get("song_model") or ALBUM_FINAL_MODEL),
        runner_plan=str(track.get("runner_plan") or "official"),
    )
    if not settings_compliance.get("valid", True) or settings_compliance.get("unsupported"):
        quality_checks["runtime_fields_supported_or_advisory"] = "review"
    hit_readiness = hit_readiness_report(
        {**track, "settings_compliance": settings_compliance},
        task_type=str(track.get("task_type") or options.get("task_type") or "text2music"),
        song_model=str(track.get("song_model") or options.get("song_model") or ALBUM_FINAL_MODEL),
        runner_plan=str(track.get("runner_plan") or "official"),
    )
    runtime_plan = runtime_planner_report(
        {**track, "settings_compliance": settings_compliance},
        task_type=str(track.get("task_type") or options.get("task_type") or "text2music"),
        song_model=str(track.get("song_model") or options.get("song_model") or ALBUM_FINAL_MODEL),
        quality_profile=str(track.get("quality_profile") or options.get("quality_profile") or "chart_master"),
    )
    if hit_readiness.get("status") in {"review", "fail"}:
        quality_checks["hook_has_title_or_emotional_promise"] = "review"
    payload_gate = evaluate_album_payload_quality(
        track,
        options=options,
        repair=False,
    )
    payload_gate_public = {key: value for key, value in payload_gate.items() if key != "repaired_payload"}
    return {
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "settings_policy_version": settings_compliance["version"],
        "settings_compliance": settings_compliance,
        "pro_quality_policy": pro_quality_policy(),
        "hit_readiness": hit_readiness,
        "payload_quality_gate": payload_gate_public,
        "payload_gate_status": payload_gate.get("status"),
        "tag_coverage": payload_gate.get("tag_coverage"),
        "caption_integrity": payload_gate.get("caption_integrity"),
        "lyric_duration_fit": payload_gate.get("lyric_duration_fit"),
        "lyrical_craft_contract": payload_gate.get("lyrical_craft_contract"),
        "lyric_craft_gate": payload_gate.get("lyric_craft_gate"),
        "lyric_craft_score": payload_gate.get("lyric_craft_score"),
        "lyric_craft_issues": payload_gate.get("lyric_craft_issues"),
        "runtime_planner": runtime_plan,
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
        "negative_control": negative_control_for(
            track.get("tags") or options.get("concept") or "",
            instrumental=parse_bool(track.get("instrumental"), False),
        ),
        "troubleshooting_hints": troubleshooting_hints,
        "hit_gate_passed": bool(
            length_ok
            and char_ok
            and section_coverage >= 0.72
            and not cliches
            and len(stats["repeated_lines"]) <= 2
            and ((payload_gate.get("lyric_craft_gate") or {}).get("status") in {None, "pass"})
        ),
    }


def production_team_report(track: dict[str, Any], options: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    tags = split_terms(track.get("tags") or track.get("caption") or "")
    duration = parse_duration_seconds(track.get("duration") or options.get("track_duration") or 120, 120)
    genre_hint = " ".join([str(track.get("tags") or ""), str(track.get("description") or ""), str(options.get("concept") or "")])
    modules = infer_genre_modules(genre_hint, max_modules=2)
    language = str(track.get("language") or options.get("language") or "en")
    preset = language_preset(language)
    has_vocal_script = bool(str(track.get("lyrics") or "").strip() and str(track.get("lyrics") or "").strip().lower() != "[instrumental]")
    explicit_instrumental = parse_bool(track.get("instrumental"), False)
    instrumental = (explicit_instrumental and not has_vocal_script) or (not has_vocal_script and is_sparse_lyric_genre(genre_hint))
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
            "reason": "Album renders use the selected policy: XL SFT legacy final or the full official render-model portfolio.",
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
    genre_hint = str(options.get("genre_hint") or concept)
    contract = options.get("user_album_contract")
    if not isinstance(contract, dict):
        contract = extract_user_album_contract(concept, options.get("num_tracks"), str(options.get("language") or "en"), options)
    contract_logs: list[str] = []
    track = apply_user_album_contract_to_track(track, contract, index, contract_logs)
    title = str(track.get("title") or f"Track {index + 1}").strip()[:80]
    duration = parse_duration_seconds(track.get("duration") or track.get("duration_seconds") or options.get("track_duration") or 120, 120)
    language = str(track.get("language") or options.get("language") or "en").strip().lower()
    strict_album_agents = bool(options.get("strict_album_agents") or str(options.get("agent_engine") or "") == "acejam_agents")
    if strict_album_agents:
        genre_hint = " ".join(
            str(track.get(key) or "")
            for key in ("title", "tags", "style", "vibe", "narrative", "description")
        ).strip() or genre_hint
    micro_agent_tags = []
    if strict_album_agents and track.get("agent_micro_settings_flow"):
        micro_agent_tags = (
            split_terms(track.get("tag_list"))
            or split_terms(track.get("tags"))
            or split_terms(track.get("caption"))
        )
    if micro_agent_tags:
        tags = micro_agent_tags[:12]
    else:
        tags = build_track_tags(
            " ".join([genre_hint, str(track.get("tags") or ""), str(track.get("description") or "")]),
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
    if options.get("disable_auto_lyric_expansion") or strict_album_agents:
        lyrics = trim_lyrics_to_limit(lyrics)
    else:
        lyrics = expand_lyrics_for_duration(
            title,
            genre_hint,
            lyrics,
            duration,
            language,
            str(options.get("lyric_density") or "balanced"),
            str(options.get("structure_preset") or "auto"),
        )
    if micro_agent_tags:
        caption = str(track.get("caption") or track.get("tags") or ", ".join(tags)).strip()
        caption = caption[:512]
    else:
        caption = polish_caption(
            tags,
            str(track.get("description") or ""),
            str(options.get("global_caption") or ""),
            strict=strict_album_agents,
        )
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
        "quality_profile": str(track.get("quality_profile") or options.get("quality_profile") or "chart_master"),
        "seed": seed_value,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "infer_method": normalize_infer_method(track.get("infer_method") or options.get("infer_method") or model_defaults["infer_method"]),
        "sampler_mode": normalize_sampler_mode(track.get("sampler_mode") or options.get("sampler_mode") or model_defaults["sampler_mode"]),
        "audio_format": normalize_album_audio_format(track.get("audio_format") or options.get("audio_format") or model_defaults["audio_format"]),
        "auto_score": bool(options.get("auto_score", False)),
        "auto_lrc": bool(options.get("auto_lrc", False)),
        "return_audio_codes": bool(options.get("return_audio_codes", False)),
        "save_to_library": bool(track.get("save_to_library", options.get("save_to_library", True))),
        "use_format": bool(track.get("use_format", options.get("use_format", False))),
        "model_advice": model_info,
        "tool_notes": tool_notes,
        "duration_seconds": duration,
    }
    dict_fields = {"contract_compliance"}
    list_fields = {"required_phrases", "contract_repaired_fields"}
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
            if field in dict_fields:
                normalized[field] = _dict_or_empty(track.get(field))
            elif field in list_fields:
                normalized[field] = _list_or_empty(track.get(field)) or split_terms(track.get(field))
            else:
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
    kit_hint = " ".join(
        [
            caption,
            str(normalized.get("description") or ""),
            genre_hint,
            str(normalized.get("style") or ""),
            str(normalized.get("vibe") or ""),
            str(normalized.get("narrative") or ""),
        ]
    )
    has_vocal_script = bool(lyrics.strip() and lyrics.strip().lower() != "[instrumental]")
    explicit_instrumental = parse_bool(track.get("instrumental"), False)
    instrumental = (explicit_instrumental and not has_vocal_script) or (not has_vocal_script and is_sparse_lyric_genre(kit_hint))
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
    strict_album_agents = parse_bool(options.get("strict_album_agents"), False) or str(options.get("agent_engine") or "").strip().lower() == "acejam_agents"
    strict_kit_override_fields = {
        "genre_profile",
        "genre_modules",
        "section_map",
        "lyric_density_notes",
    }
    for field in PROMPT_KIT_METADATA_FIELDS:
        if field == "copy_paste_block":
            normalized[field] = _coerce_prompt_kit_field(field, track.get(field), "")
        elif field in {"lyrics", "ace_caption", "metadata", "generation_settings"}:
            normalized[field] = _coerce_prompt_kit_field(field, kit_defaults.get(field), kit_defaults.get(field))
        elif strict_album_agents and field in strict_kit_override_fields:
            normalized[field] = _coerce_prompt_kit_field(field, kit_defaults.get(field), kit_defaults.get(field))
        elif field in track and track.get(field) not in (None, ""):
            normalized[field] = _coerce_prompt_kit_field(field, track.get(field), kit_defaults.get(field))
        elif field in kit_defaults:
            normalized[field] = _coerce_prompt_kit_field(field, kit_defaults[field], kit_defaults[field])
    normalized["vocal_language"] = str(track.get("vocal_language") or kit_defaults.get("vocal_language") or language)
    normalized["instrumental"] = instrumental
    normalized["settings_compliance"] = ace_step_settings_compliance(
        normalized,
        task_type=str(options.get("task_type") or "text2music"),
        song_model=song_model,
        runner_plan=str(normalized.get("runner_plan") or "official"),
    )
    normalized["settings_policy_version"] = normalized["settings_compliance"]["version"]
    normalized["tool_report"] = quality_report(normalized, options)
    normalized["quality_checks"] = {
        **_dict_or_empty(normalized.get("quality_checks")),
        **_dict_or_empty(normalized["tool_report"].get("quality_checks")),
    }
    normalized["payload_quality_gate"] = normalized["tool_report"].get("payload_quality_gate", {})
    normalized["payload_gate_status"] = normalized["tool_report"].get("payload_gate_status", "")
    normalized["tag_coverage"] = normalized["tool_report"].get("tag_coverage", {})
    normalized["caption_integrity"] = normalized["tool_report"].get("caption_integrity", {})
    normalized["lyric_duration_fit"] = normalized["tool_report"].get("lyric_duration_fit", {})
    normalized["repair_actions"] = _list_or_empty(normalized.get("repair_actions")) or split_terms(normalized.get("repair_actions"))
    normalized["troubleshooting_hints"] = list(
        dict.fromkeys(
            (_list_or_empty(normalized.get("troubleshooting_hints")) or split_terms(normalized.get("troubleshooting_hints")))
            + (_list_or_empty(normalized["tool_report"].get("troubleshooting_hints")) or split_terms(normalized["tool_report"].get("troubleshooting_hints")))
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


def _dedupe_strings(values: list[Any] | tuple[Any, ...] | set[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _intent_option(
    value: str,
    label: str | None = None,
    *,
    description: str = "",
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    source: str = "acejam",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "value": str(value),
        "label": str(label or value).replace("_", " ").title() if label is None else str(label),
        "source": source,
    }
    if description:
        payload["description"] = description
    if aliases:
        payload["aliases"] = _dedupe_strings(aliases)
    if tags:
        payload["tags"] = _dedupe_strings(tags)
    if meta:
        payload["meta"] = meta
    return payload


def _taxonomy_options(category: str) -> list[dict[str, Any]]:
    return [
        _intent_option(value, value, source=f"tag_taxonomy.{category}")
        for value in TAG_TAXONOMY.get(category, [])
    ]


def _matching_taxonomy_terms(categories: list[str], patterns: list[str], extras: list[str] | None = None) -> list[str]:
    pattern = re.compile("|".join(re.escape(item) for item in patterns), re.IGNORECASE)
    terms: list[str] = []
    for category in categories:
        terms.extend(value for value in TAG_TAXONOMY.get(category, []) if pattern.search(value))
    terms.extend(extras or [])
    return _dedupe_strings(terms)


def _render_model_summary(model_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    tasks = list(meta.get("tasks") or [])
    return _intent_option(
        model_id,
        model_id,
        description=f"{meta.get('quality') or 'ACE-Step'} quality, {meta.get('steps') or 'variable'} steps",
        tags=tasks,
        meta={
            "repo_id": meta.get("repo_id"),
            "role": meta.get("role"),
            "tasks": tasks,
            "quality": meta.get("quality"),
            "steps": meta.get("steps"),
            "cfg": meta.get("cfg"),
            "downloadable": bool(meta.get("downloadable")),
            "render_usable": bool(meta.get("render_usable")),
        },
        source=str(meta.get("source") or "official_model_registry"),
    )


def song_intent_schema(installed_models: set[str] | list[str] | None = None) -> dict[str, Any]:
    """Return the complete, UI-ready Song Intent Builder contract."""
    installed = set(installed_models or [])
    registry = official_model_registry()
    render_models = [
        _render_model_summary(model_id, registry.get(model_id, {}))
        for model_id in official_render_model_ids()
        if model_id in registry
    ]
    lm_models = [
        _render_model_summary(model_id, meta)
        for model_id, meta in registry.items()
        if meta.get("role") == "ace_lm"
    ]
    downloadable_models = [
        _render_model_summary(model_id, registry.get(model_id, {}))
        for model_id in official_downloadable_model_ids()
        if model_id in registry
    ]
    task_modes = [
        _intent_option("text2music", "Text to music", description="Generate from caption, lyrics, metadata, and optional reference audio.", source="ace_step.tasks"),
        _intent_option("cover", "Cover / style transfer", description="Use source audio as song material and optional reference audio for style.", source="ace_step.tasks"),
        _intent_option("cover-nofsq", "Cover without FSQ", description="AceJAM cover variant for source-audio workflows.", source="acejam.tasks"),
        _intent_option("repaint", "Repaint / edit section", description="Modify a selected source-audio region while preserving surrounding context.", source="ace_step.tasks"),
        _intent_option("extract", "Extract stem", description="Base/XL-base only: isolate a selected track stem from a mix.", source="ace_step.tasks"),
        _intent_option("lego", "Lego add stem", description="Base/XL-base only: generate a selected track from source-audio context.", source="ace_step.tasks"),
        _intent_option("complete", "Complete arrangement", description="Base/XL-base only: complete partial tracks with selected stems.", source="ace_step.tasks"),
    ]
    all_task_ids = [option["value"] for option in task_modes]
    model_support = {
        task: [
            model["value"]
            for model in render_models
            if task in (model.get("meta", {}).get("tasks") or [])
        ]
        for task in all_task_ids
    }
    source_audio_modes = [
        _intent_option("none", "No source audio", source="ace_step.audio_control"),
        _intent_option("reference_audio", "Reference audio", description="Style reference for text2music.", source="ace_step.audio_control"),
        _intent_option("src_audio", "Source audio", description="Required for cover, repaint, extract, lego, and complete.", source="ace_step.audio_control"),
        _intent_option("src_plus_reference", "Source plus reference", description="Edit source audio with optional style reference.", source="ace_step.audio_control"),
        _intent_option("audio_codes", "Audio semantic codes", description="Advanced control from extracted audio codes.", source="ace_step.audio_control"),
    ]
    model_strategies = [
        _intent_option("auto", "Auto", description="Use AceJAM's selected model and quality profile.", source="acejam.model_strategy"),
        _intent_option("preview_fast", "Preview fast", description="Turbo draft settings.", tags=["turbo", "8 steps"], source="acejam.model_strategy"),
        _intent_option("chart_master", "Chart master", description="High-quality final render settings.", tags=["XL SFT", "CFG"], source="acejam.model_strategy"),
        _intent_option("base_control", "Base control", description="Use base/XL-base for extract, lego, and complete.", tags=["extract", "lego", "complete"], source="acejam.model_strategy"),
        _intent_option("all_models_song", "All render models", description="Render comparison across official render-usable models.", source="acejam.model_strategy"),
    ]
    personalization = [
        _intent_option("none", "No adapter", source="ace_step.personalization"),
        _intent_option("lora", "LoRA adapter", description="Use a trained adapter during inference.", source="ace_step.personalization"),
        _intent_option("lokr", "LoKr adapter", description="Use a LoKr-trained adapter exported for inference.", source="ace_step.personalization"),
        _intent_option("activation_tag", "Activation tag", description="Add a trained style trigger tag to the caption.", source="ace_step.personalization"),
    ]
    drums = _matching_taxonomy_terms(
        ["speed_rhythm", "instruments", "production_style", "structure_hints"],
        ["drum", "kick", "snare", "hat", "breakbeat", "percussion", "four-on", "rhythm", "groove", "sidechain"],
        ["no drums", "dembow rhythm", "log drum", "off-beat drums", "brush drums"],
    )
    bass = _matching_taxonomy_terms(
        ["instruments", "timbre_texture"],
        ["bass", "808", "sub", "low end", "low-end", "upright"],
        ["reese bass", "controlled sub"],
    )
    melodic = _dedupe_strings(
        [
            value
            for value in TAG_TAXONOMY["instruments"]
            if not re.search(r"drum|kick|snare|hat|bass|percussion|breakbeat", value, re.IGNORECASE)
        ]
        + ["soul chop", "piano motif", "guitar riff", "vocal chop"]
    )
    genre_modules = [
        _intent_option(
            key,
            module.get("label") or key,
            description=str(module.get("structure") or ""),
            aliases=list(module.get("aliases") or []),
            tags=list(module.get("caption_dna") or []),
            meta={
                "bpm": module.get("bpm"),
                "keys": module.get("keys"),
                "hook_strategy": module.get("hook_strategy"),
                "avoid": list(module.get("avoid") or []),
                "density": module.get("density"),
                "instrumental": bool(module.get("instrumental")),
            },
            source="prompt_kit.genre_modules",
        )
        for key, module in GENRE_MODULES.items()
    ]
    languages = [
        _intent_option(
            code,
            str(LANGUAGE_PRESETS.get(code, {}).get("language") or code),
            meta={
                "prompt_kit_preset": code in LANGUAGE_PRESETS,
                **({k: v for k, v in LANGUAGE_PRESETS.get(code, {}).items() if k != "language"} if code in LANGUAGE_PRESETS else {}),
            },
            source="ace_step.valid_languages",
        )
        for code in VALID_LANGUAGES
    ]
    groups = {
        "genre_modules": genre_modules,
        "genre_style": _taxonomy_options("genre_style"),
        "mood_atmosphere": _taxonomy_options("mood_atmosphere"),
        "drums_groove": [_intent_option(value, value, source="derived.drums_groove") for value in drums],
        "bass_low_end": [_intent_option(value, value, source="derived.bass_low_end") for value in bass],
        "melodic_identity": [_intent_option(value, value, source="derived.melodic_identity") for value in melodic],
        "instruments": _taxonomy_options("instruments"),
        "stems": _taxonomy_options("track_stems"),
        "vocal_character": _taxonomy_options("vocal_character"),
        "speed_rhythm": _taxonomy_options("speed_rhythm"),
        "structure_hints": _taxonomy_options("structure_hints"),
        "lyric_meta_tags": [
            _intent_option(value, value, meta={"category": category}, source=f"lyric_meta_tags.{category}")
            for category, values in LYRIC_META_TAGS.items()
            for value in values
        ],
        "timbre_texture": _taxonomy_options("timbre_texture"),
        "era_reference": _taxonomy_options("era_reference"),
        "production_style": _taxonomy_options("production_style"),
        "negative_control": [
            _intent_option(value, value, source="acejam.negative_control")
            for value in _dedupe_strings(
                [
                    "muddy mix",
                    "generic lyrics",
                    "weak hook",
                    "empty lyrics",
                    "off-key vocal",
                    "unclear vocal",
                    "noisy artifacts",
                    "flat drums",
                    "harsh high end",
                    "overcompressed",
                    "boring arrangement",
                    "contradictory style",
                    *negative_control_for("generic"),
                ]
            )
        ],
        "task_modes": task_modes,
        "source_audio_modes": source_audio_modes,
        "model_strategies": model_strategies,
        "personalization": personalization,
        "render_models": render_models,
        "lm_models": lm_models,
        "downloadable_models": downloadable_models,
        "languages": languages,
    }
    tab_layout = [
        {"id": "style", "label": "Style", "groups": ["genre_modules", "genre_style", "mood_atmosphere", "era_reference"]},
        {"id": "groove", "label": "Groove", "groups": ["drums_groove", "bass_low_end", "speed_rhythm"]},
        {"id": "arrangement", "label": "Arrangement", "groups": ["melodic_identity", "instruments", "stems", "structure_hints", "lyric_meta_tags"]},
        {"id": "vocals", "label": "Vocals", "groups": ["vocal_character", "languages"]},
        {"id": "ace_step_mode", "label": "ACE-Step Mode", "groups": ["task_modes", "source_audio_modes", "stems"]},
        {"id": "model_quality", "label": "Model/Quality", "groups": ["model_strategies", "render_models", "lm_models", "personalization"]},
        {"id": "negative", "label": "Negative", "groups": ["negative_control"]},
    ]
    return {
        "version": SONG_INTENT_SCHEMA_VERSION,
        "sources": OFFICIAL_SOURCES,
        "groups": groups,
        "tabs": tab_layout,
        "metadata": {
            "valid_languages": list(VALID_LANGUAGES),
            "language_presets": LANGUAGE_PRESETS,
            "valid_keyscales": sorted(VALID_KEYSCALES),
            "valid_time_signatures": list(VALID_TIME_SIGNATURES),
            "ranges": {"bpm": [BPM_MIN, BPM_MAX], "duration": [DURATION_MIN, DURATION_MAX]},
            "track_names": list(TRACK_NAMES),
            "installed_song_models": sorted(installed),
        },
        "capabilities": {
            "official_tasks": list(TASK_TYPES),
            "local_task_variants": ["cover-nofsq"],
            "all_task_modes": all_task_ids,
            "base_only_tasks": ["extract", "lego", "complete"],
            "source_audio_tasks": ["cover", "cover-nofsq", "repaint", "extract", "lego", "complete"],
            "lm_skipped_tasks": ["cover", "cover-nofsq", "repaint", "extract"],
            "model_support": model_support,
        },
        "legacy_field_map": {
            "genre_family": "genre_modules",
            "subgenre": "genre_style",
            "mood": "mood_atmosphere",
            "energy": "speed_rhythm",
            "vocal_type": "vocal_character",
            "drum_groove": "drums_groove",
            "bass_low_end": "bass_low_end",
            "melodic_identity": "melodic_identity",
            "texture_space": "timbre_texture",
            "mix_master": "production_style",
        },
        "counts": {
            "genre_modules": len(GENRE_MODULES),
            "tag_taxonomy_groups": len(TAG_TAXONOMY),
            "tag_taxonomy_terms": sum(len(values) for values in TAG_TAXONOMY.values()),
            "lyric_meta_tags": sum(len(values) for values in LYRIC_META_TAGS.values()),
            "valid_languages": len(VALID_LANGUAGES),
            "valid_keyscales": len(VALID_KEYSCALES),
            "official_tasks": len(TASK_TYPES),
            "task_modes": len(all_task_ids),
            "track_stems": len(TRACK_NAMES),
            "render_models": len(render_models),
            "lm_models": len(lm_models),
        },
    }


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
        "ace_step_settings_registry": ace_step_settings_registry(),
        "sources": OFFICIAL_SOURCES,
        "song_intent_schema": song_intent_schema(installed),
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


def _compact_tool_debug(value: Any, limit: int = 6000) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_tool_debug(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_compact_tool_debug(item, limit) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _write_tool_debug(context: dict[str, Any], tool_name: str, tool_input: Any, tool_output: Any) -> None:
    debug_dir = str(context.get("album_debug_dir") or "").strip()
    if not debug_dir:
        return
    try:
        path = Path(debug_dir) / "03_crewai_tool_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "input": _compact_tool_debug(tool_input),
            "output": _compact_tool_debug(tool_output),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    except Exception:
        return


def _tool_return(context: dict[str, Any], tool_name: str, tool_input: Any, payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=True, default=str)
    _write_tool_debug(context, tool_name, tool_input, payload)
    return text


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
        """Return the full official render-model album portfolio and install state."""
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
        payload = {
            "caption_dimensions": [
                "genre_style",
                "rhythm_groove",
                "instrumentation",
                "vocal_style",
                "mood_atmosphere",
                "arrangement_energy",
                "mix_production",
            ],
            "tag_taxonomy": TAG_TAXONOMY,
            "lyric_meta_tags": LYRIC_META_TAGS,
            "selection_rule": "Use this full library to choose compact caption/tag terms; do not dump the entire library into final captions.",
        }
        return _tool_return(context, "TagLibraryTool", query, payload)

    @tool("AceStepPromptContractTool")
    def ace_step_prompt_contract_tool(query: str = "") -> str:
        """Return the strict ACE-Step track payload contract and self-check rules."""
        payload = {
            "contract": context.get("ace_step_track_payload_contract") or {},
            "prompt_template_version": context.get("ace_step_track_prompt_template_version") or "",
            "rules": [
                "Caption is <=512 chars and contains only sound/style/production cues.",
                "Lyrics are <=4096 chars and contain section tags plus actual lines only.",
                "Metadata stays in bpm/key_scale/time_signature/duration fields.",
                "Run LyricCounterTool, TagCoverageTool, CaptionIntegrityTool, and PayloadGateTool before final JSON.",
                "If PayloadGateTool reports fail, repair before final JSON.",
            ],
        }
        return _tool_return(context, "AceStepPromptContractTool", query, payload)

    @tool("LyricCounterTool")
    def lyric_counter_tool(lyrics: str = "") -> str:
        """Deterministically count lyric words, lines, chars, sections, and hooks."""
        stats = lyric_stats(str(lyrics or ""))
        hooks = [
            section
            for section in stats.get("sections") or []
            if re.search(r"chorus|hook|refrain", str(section), re.I)
        ]
        payload = {
            "lyrics_word_count": int(stats.get("word_count") or 0),
            "lyrics_line_count": int(stats.get("line_count") or 0),
            "lyrics_char_count": int(stats.get("char_count") or 0),
            "section_count": int(stats.get("section_count") or 0),
            "hook_count": len(hooks),
            "sections": stats.get("sections") or [],
            "repeated_lines": stats.get("repeated_lines") or [],
            "copy_counts_to_final_json": True,
        }
        return _tool_return(context, "LyricCounterTool", {"lyrics_chars": len(str(lyrics or ""))}, payload)

    @tool("TagCoverageTool")
    def tag_coverage_tool(spec: str = "") -> str:
        """Check that caption/tags/tag_list cover every required ACE-Step sound dimension."""
        try:
            payload_in = json.loads(spec or "{}") if str(spec or "").strip().startswith("{") else {"caption": spec}
        except json.JSONDecodeError:
            payload_in = {"caption": spec}
        if not isinstance(payload_in, dict):
            payload_in = {"caption": str(spec or "")}
        report = evaluate_album_payload_quality(payload_in, options=context, repair=False)
        payload = {
            "status": (report.get("tag_coverage") or {}).get("status"),
            "missing": (report.get("tag_coverage") or {}).get("missing") or [],
            "dimensions": (report.get("tag_coverage") or {}).get("dimensions") or [],
            "repair_instruction": "Add one compact caption/tag term for every missing dimension before final JSON.",
        }
        return _tool_return(context, "TagCoverageTool", payload_in, payload)

    @tool("CaptionIntegrityTool")
    def caption_integrity_tool(spec: str = "") -> str:
        """Check caption text for lyrics, prompt, JSON, and metadata leakage."""
        try:
            payload_in = json.loads(spec or "{}") if str(spec or "").strip().startswith("{") else {"caption": spec}
        except json.JSONDecodeError:
            payload_in = {"caption": spec}
        if not isinstance(payload_in, dict):
            payload_in = {"caption": str(spec or "")}
        report = evaluate_album_payload_quality(payload_in, options=context, repair=False)
        payload = {
            "status": (report.get("caption_integrity") or {}).get("status"),
            "char_count": (report.get("caption_integrity") or {}).get("char_count"),
            "leakage_markers": (report.get("caption_integrity") or {}).get("leakage_markers") or [],
            "repair_instruction": "Rewrite caption as comma-separated sound terms only; move lyrics and metadata to their fields.",
        }
        return _tool_return(context, "CaptionIntegrityTool", payload_in, payload)

    @tool("PayloadGateTool")
    def payload_gate_tool(spec: str = "") -> str:
        """Run AceJAM's album payload quality gate and return compact retry instructions."""
        try:
            payload_in = json.loads(spec or "{}") if str(spec or "").strip().startswith("{") else {"lyrics": spec}
        except json.JSONDecodeError:
            payload_in = {"lyrics": spec}
        if not isinstance(payload_in, dict):
            payload_in = {"lyrics": str(spec or "")}
        report = evaluate_album_payload_quality(payload_in, options=context, repair=True)
        public_report = {key: value for key, value in report.items() if key != "repaired_payload"}
        blocking = report.get("blocking_issues") or []
        payload = {
            "status": report.get("status"),
            "gate_passed": bool(report.get("gate_passed")),
            "issues": report.get("issues") or [],
            "repair_actions": report.get("repair_actions") or [],
            "retry_instructions": [
                f"{issue.get('id')}: {issue.get('detail')}"
                for issue in blocking[:8]
            ] or [
                f"{issue.get('id')}: {issue.get('detail')}"
                for issue in (report.get("issues") or [])[:8]
            ],
            "report": public_report,
        }
        return _tool_return(context, "PayloadGateTool", payload_in, payload)

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
        """Return persona, cadence, vocal-response, harmony, and performance tag guidance."""
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
                    "backing vocal responses only where they support the hook",
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

    @tool("AceStepSettingsPolicyTool")
    def ace_step_settings_policy_tool(query: str = "") -> str:
        """Return the ACE-Step docs parity settings registry."""
        registry = ace_step_settings_registry()
        compact = {
            "version": registry["version"],
            "default_quality_profile": registry.get("default_quality_profile"),
            "profiles": registry["profiles"],
            "coverage": registry.get("coverage"),
            "task_policy": registry["task_policy"],
            "runner_support": registry["runner_support"],
        }
        return json.dumps(compact, ensure_ascii=True)

    @tool("ChartMasterProfileTool")
    def chart_master_profile_tool(song_model: str = "") -> str:
        """Return AceJAM's final-render Chart Master defaults."""
        registry = ace_step_settings_registry()
        model = str(song_model or context.get("song_model") or ALBUM_FINAL_MODEL)
        chart = registry["profiles"]["chart_master"]
        model_settings = chart["models"].get(model) or chart["models"].get(ALBUM_FINAL_MODEL) or {}
        return json.dumps(
            {
                "version": registry["version"],
                "quality_profile": "chart_master",
                "model": model,
                "settings": model_settings,
                "single_song_takes": chart.get("single_song_takes", 3),
                "album_takes": chart.get("album_takes", 1),
                "note": "Use Chart Master for final renders; use Preview Fast only for drafts.",
            },
            ensure_ascii=True,
        )

    @tool("AceStepCoverageAuditTool")
    def ace_step_coverage_audit_tool(query: str = "") -> str:
        """Return the full ACE-Step settings/API/helper/result coverage audit."""
        registry = ace_step_settings_registry()
        return json.dumps(registry.get("coverage", {}), ensure_ascii=True)

    @tool("EffectiveSettingsTool")
    def effective_settings_tool(settings_json: str = "") -> str:
        """Return effective settings compliance for a proposed generation payload."""
        try:
            payload = json.loads(settings_json) if settings_json.strip().startswith("{") else {}
        except Exception:
            payload = {}
        task = str(payload.get("task_type") or context.get("task_type") or "text2music")
        model = str(payload.get("song_model") or context.get("song_model") or ALBUM_FINAL_MODEL)
        runner = str(payload.get("runner_plan") or "official")
        compliance = ace_step_settings_compliance(payload, task_type=task, song_model=model, runner_plan=runner)
        return json.dumps(
            {
                "quality_profile": payload.get("quality_profile") or context.get("quality_profile") or "chart_master",
                "compliance": compliance,
                "note": "Fields marked ignored_for_task, source_locked, read_only_lm_output, reserved, official_only, or unsupported must not be treated as active controls.",
            },
            ensure_ascii=True,
        )

    @tool("HitReadinessTool")
    def hit_readiness_tool(settings_json: str = "") -> str:
        """Return pre-render hit readiness gates for a proposed ACE-Step payload."""
        try:
            payload = json.loads(settings_json) if settings_json.strip().startswith("{") else {}
        except Exception:
            payload = {}
        task = str(payload.get("task_type") or context.get("task_type") or "text2music")
        model = str(payload.get("song_model") or context.get("song_model") or ALBUM_FINAL_MODEL)
        runner = str(payload.get("runner_plan") or "official")
        return json.dumps(
            hit_readiness_report(payload, task_type=task, song_model=model, runner_plan=runner),
            ensure_ascii=True,
        )

    @tool("RuntimePlannerTool")
    def runtime_planner_tool(settings_json: str = "") -> str:
        """Return render ETA and memory-risk guidance for a proposed ACE-Step payload."""
        try:
            payload = json.loads(settings_json) if settings_json.strip().startswith("{") else {}
        except Exception:
            payload = {}
        task = str(payload.get("task_type") or context.get("task_type") or "text2music")
        model = str(payload.get("song_model") or context.get("song_model") or ALBUM_FINAL_MODEL)
        quality_profile = str(payload.get("quality_profile") or context.get("quality_profile") or "chart_master")
        return json.dumps(
            {
                "runtime_planner": runtime_planner_report(
                    payload,
                    task_type=task,
                    song_model=model,
                    quality_profile=quality_profile,
                ),
                "pro_quality_policy": pro_quality_policy(),
            },
            ensure_ascii=True,
        )

    @tool("AandRVariantPlanTool")
    def ar_variant_plan_tool(song_model: str = "") -> str:
        """Return take-count guidance for professional A&R selection."""
        registry = ace_step_settings_registry()
        chart = registry["profiles"]["chart_master"]
        return json.dumps(
            {
                "quality_profile": "chart_master",
                "single_song_takes": chart.get("single_song_takes", 3),
                "album_takes": chart.get("album_takes", 1),
                "selection_notes": [
                    "Generate three single-song takes for final selection.",
                    "Albums stay one take per track/model by default to avoid runaway render time.",
                    "Choose the take with the strongest hook clarity, groove, vocal presence, and artifact-free mix.",
                ],
                "model": str(song_model or context.get("song_model") or ALBUM_FINAL_MODEL),
            },
            ensure_ascii=True,
        )

    @tool("TaskApplicabilityTool")
    def task_applicability_tool(task_type: str = "") -> str:
        """Return task-specific ACE-Step setting applicability."""
        task = str(task_type or context.get("task_type") or "text2music").strip() or "text2music"
        registry = ace_step_settings_registry()
        return json.dumps(
            {
                "version": registry["version"],
                "task_type": task,
                "required_fields": registry["task_policy"]["required_fields"].get(task, []),
                "lm_ignored": task in set(registry["task_policy"]["lm_skips"]),
                "duration_source_locked": task in set(registry["task_policy"]["source_locked_duration"]),
                "note": "Do not ask agents to fill fields marked read-only, reserved, ignored, or unsupported for the task.",
            },
            ensure_ascii=True,
        )

    @tool("ModelCompatibilityTool")
    def model_compatibility_tool(settings_json: str = "") -> str:
        """Validate model/task/settings compatibility from compact JSON."""
        try:
            payload = json.loads(settings_json) if settings_json.strip().startswith("{") else {}
        except Exception:
            payload = {}
        task = str(payload.get("task_type") or context.get("task_type") or "text2music")
        model = str(payload.get("song_model") or context.get("song_model") or ALBUM_FINAL_MODEL)
        runner = str(payload.get("runner_plan") or "official")
        return json.dumps(
            ace_step_settings_compliance(payload, task_type=task, song_model=model, runner_plan=runner),
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
        ace_step_prompt_contract_tool,
        lyric_counter_tool,
        tag_coverage_tool,
        caption_integrity_tool,
        payload_gate_tool,
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
        ace_step_settings_policy_tool,
        chart_master_profile_tool,
        ace_step_coverage_audit_tool,
        effective_settings_tool,
        hit_readiness_tool,
        runtime_planner_tool,
        ar_variant_plan_tool,
        task_applicability_tool,
        model_compatibility_tool,
    ]
