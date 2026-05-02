from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


PROMPT_KIT_VERSION = "ace-step-multilingual-hit-kit-2026-04-26"

PROMPT_KIT_OUTPUT_CONTRACT_FIELDS = [
    "concept_summary",
    "target_language",
    "vocal_language",
    "language_notes",
    "genre_profile",
    "song_length_mode",
    "section_map",
    "lyric_density_notes",
    "ace_caption",
    "lyrics",
    "metadata",
    "generation_settings",
    "runtime_profile",
    "workflow_mode",
    "source_audio_mode",
    "advanced_generation_settings",
    "iteration_plan",
    "community_risk_notes",
    "troubleshooting_hints",
    "variations",
    "negative_control",
    "quality_checks",
    "anti_ai_rewrite_notes",
    "copy_paste_block",
]

PROMPT_KIT_METADATA_FIELDS = [
    "prompt_kit_version",
    *PROMPT_KIT_OUTPUT_CONTRACT_FIELDS,
    "genre_modules",
]

MASTER_RULES = [
    "Keep ACE-Step caption, lyrics, and metadata separate: caption is sonic intent, lyrics are the temporal script, metadata carries BPM/key/duration/runtime controls.",
    "Never put BPM, key, time signature, model names, seed, or duration clutter inside the caption.",
    "Use compact section and performance tags inside lyrics, for example [Verse], [Chorus], [Build], [Drop], [Outro].",
    "No placeholders, no prompt leakage.",
    "Anchor every song in one emotional promise, one coherent image world, and a hook short enough to remember.",
    "Lyric craft must be human and specific: concrete scenes, earned metaphors, natural cadence, no generic AI phrases such as neon dreams / fire inside / we rise.",
    "Adapt craft to the genre: rap needs cadence and internal rhyme; sung music needs vowel-friendly emotional clarity; EDM/instrumental needs motif and energy movement rather than forced verses.",
    "Write in the requested language/script unless the user explicitly asks for romanization or code switching.",
    "Use full polished output without ACE-Step format_input unless the workflow is a rough draft, repair, or explicit formatting task.",
    "For source-audio workflows, mark duration and structure as source-locked advisory metadata unless the handler supports a runtime knob.",
]

LANGUAGE_PRESETS: dict[str, dict[str, Any]] = {
    "en": {
        "name": "English",
        "vocal_language": "en",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Natural idiomatic English, short hook phrases, avoid filler rhyme padding.",
    },
    "nl": {
        "name": "Dutch",
        "vocal_language": "nl",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Dutch is primary; Dutch-English blend is allowed when it sounds like modern pop or rap.",
    },
    "fr": {
        "name": "French",
        "vocal_language": "fr",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Keep phrasing singable and avoid overly formal textbook French.",
    },
    "es": {
        "name": "Spanish",
        "vocal_language": "es",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Spanish lyric phrasing and concise vowel-forward hooks.",
    },
    "pt": {
        "name": "Portuguese",
        "vocal_language": "pt",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Portuguese; keep line lengths short enough for vocal clarity.",
    },
    "de": {
        "name": "German",
        "vocal_language": "de",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use compact German phrasing; split long compounds into singable lines when needed.",
    },
    "ar": {
        "name": "Arabic",
        "vocal_language": "ar",
        "script": "Arabic script",
        "romanization": "native script by default",
        "notes": "Use Arabic script by default; romanize only when explicitly requested.",
    },
    "tr": {
        "name": "Turkish",
        "vocal_language": "tr",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Turkish vowel flow and keep hooks rhythmically simple.",
    },
    "hi": {
        "name": "Hindi",
        "vocal_language": "hi",
        "script": "Devanagari or user-selected Hinglish",
        "romanization": "native script unless Hinglish is requested",
        "notes": "Use Devanagari by default; Hinglish/code switch only when requested or implied.",
    },
    "ur": {
        "name": "Urdu",
        "vocal_language": "ur",
        "script": "Arabic-derived Urdu script",
        "romanization": "native script by default",
        "notes": "Use Urdu script by default; keep couplet-like phrasing clear and singable.",
    },
    "pa": {
        "name": "Punjabi",
        "vocal_language": "pa",
        "script": "Gurmukhi or Shahmukhi by user context",
        "romanization": "native script by default",
        "notes": "Use the script implied by the request; keep refrain lines short and chantable.",
    },
    "ja": {
        "name": "Japanese",
        "vocal_language": "ja",
        "script": "Japanese characters",
        "romanization": "no romaji by default",
        "notes": "Use Japanese writing by default; no romaji unless explicitly requested.",
    },
    "ko": {
        "name": "Korean",
        "vocal_language": "ko",
        "script": "Hangul",
        "romanization": "no romanization by default",
        "notes": "Use Hangul by default; keep hook syllable counts tight and repeatable.",
    },
    "zh": {
        "name": "Chinese",
        "vocal_language": "zh",
        "script": "Chinese characters",
        "romanization": "no Pinyin by default",
        "notes": "Use Chinese characters by default; no Pinyin unless the user asks.",
    },
    "yue": {
        "name": "Cantonese",
        "vocal_language": "yue",
        "script": "Traditional Chinese characters unless requested otherwise",
        "romanization": "no Jyutping by default",
        "notes": "Use written Cantonese or user-specified register; no Jyutping unless requested.",
    },
    "id": {
        "name": "Indonesian",
        "vocal_language": "id",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Indonesian and avoid overly literal English syntax.",
    },
    "ms": {
        "name": "Malay",
        "vocal_language": "ms",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Malay; short melodic lines usually work best.",
    },
    "sw": {
        "name": "Swahili",
        "vocal_language": "sw",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Swahili with clear vowel endings and uncluttered hooks.",
    },
    "pcm": {
        "name": "Nigerian Pidgin",
        "vocal_language": "pcm",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Pidgin phrasing; code switching with English is acceptable.",
    },
    "it": {
        "name": "Italian",
        "vocal_language": "it",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use natural Italian vowels and avoid dense consonant clusters in hooks.",
    },
    "pl": {
        "name": "Polish",
        "vocal_language": "pl",
        "script": "Latin",
        "romanization": "native Latin script",
        "notes": "Use concise Polish lines; split dense grammar across shorter phrases.",
    },
    "he": {
        "name": "Hebrew",
        "vocal_language": "he",
        "script": "Hebrew",
        "romanization": "native script by default",
        "notes": "Use Hebrew script by default; romanize only when explicitly requested.",
    },
    "ru": {
        "name": "Russian",
        "vocal_language": "ru",
        "script": "Cyrillic",
        "romanization": "native script by default",
        "notes": "Use Cyrillic by default; keep long lines broken into clear vocal phrases.",
    },
    "unknown": {
        "name": "Unknown or mixed language",
        "vocal_language": "en",
        "script": "User specified",
        "romanization": "follow user request",
        "notes": "Ask the generation prompt to preserve the user's requested language/script and avoid guessing romanization.",
    },
}

GENRE_MODULES: dict[str, dict[str, Any]] = {
    "hiphop": {
        "label": "Hip-hop",
        "aliases": ["hip hop", "hip-hop", "rap"],
        "caption_dna": ["hip-hop drums", "deep low end", "confident vocal pocket", "crisp modern mix"],
        "structure": "Intro, Verse, Hook, Verse, Hook, Bridge or beat switch, Final Hook, Outro.",
        "bpm": "78-102 or double-time equivalent",
        "keys": "minor keys, modal tension, sparse melodic loops",
        "hook_strategy": "title-centered chant or short melodic phrase with clear repeat point",
        "avoid": ["generic flex lines", "overcrowded caption"],
        "density": "dense",
    },
    "boom_bap": {
        "label": "Boom-bap",
        "aliases": ["boom bap", "90s rap"],
        "caption_dna": ["dusty drums", "vinyl texture", "piano or soul chop", "turntable scratches"],
        "structure": "DJ intro, Verse, Hook, Verse, Hook, Bridge or scratch break, Final Hook.",
        "bpm": "82-96",
        "keys": "minor or blues-inflected loops",
        "hook_strategy": "short scratched phrase or crowd-call hook",
        "avoid": ["sterile drums", "too much reverb", "modern trap clutter"],
        "density": "dense",
    },
    "trap": {
        "label": "Trap",
        "aliases": ["trap rap", "modern trap"],
        "caption_dna": ["808 bass", "trap hi-hats", "dark synth lead", "wide vocal responses"],
        "structure": "Intro, Hook, Verse, Hook, Verse, Hook, Outro with backing vocal responses.",
        "bpm": "120-160 with half-time feel",
        "keys": "minor, phrygian color, sparse top lines",
        "hook_strategy": "minimal phrase, repeated with vocal-response contrast",
        "avoid": ["muddy 808", "too many metaphors", "weak consonant hook"],
        "density": "balanced",
    },
    "drill": {
        "label": "Drill",
        "aliases": ["uk drill", "ny drill"],
        "caption_dna": ["sliding 808s", "syncopated hats", "cold piano", "half-time menace"],
        "structure": "Intro, Verse, Hook, Verse, Hook, cold outro.",
        "bpm": "136-150",
        "keys": "minor, icy chord movement",
        "hook_strategy": "tight rhythmic hook with repeated end rhyme",
        "avoid": ["messy subdivisions"],
        "density": "dense",
    },
    "melodic_rap": {
        "label": "Melodic rap",
        "aliases": ["melodic rap", "emo rap"],
        "caption_dna": ["melodic rap vocal", "autotune texture", "808 bass", "dreamy pads"],
        "structure": "Intro, melodic Hook, Verse, Hook, Bridge, Final Hook.",
        "bpm": "78-98 or 140-160 half-time",
        "keys": "minor, relative major lift for hooks",
        "hook_strategy": "singable title phrase with emotional turn",
        "avoid": ["flat monotone", "overlong verses", "copied melodic contour"],
        "density": "balanced",
    },
    "rnb": {
        "label": "R&B",
        "aliases": ["r&b", "rnb", "contemporary r&b"],
        "caption_dna": ["silky vocal", "Rhodes", "sub-bass", "stacked harmonies"],
        "structure": "Intro, Verse, Pre-Chorus, Chorus, Verse, Chorus, Bridge, Final Chorus.",
        "bpm": "62-92",
        "keys": "minor seventh, major seventh, rich extensions",
        "hook_strategy": "emotional promise with vowel-rich melodic lift",
        "avoid": ["stiff phrasing", "crowded lyrics", "dry harmony stacks"],
        "density": "balanced",
    },
    "afrobeats": {
        "label": "Afrobeats",
        "aliases": ["afrobeats", "afrobeat"],
        "caption_dna": ["afrobeats groove", "warm percussion", "guitar plucks", "sunlit vocal"],
        "structure": "Intro, Hook, Verse, Hook, Dance break, Verse, Final Hook.",
        "bpm": "92-116",
        "keys": "major or minor with warm pentatonic feel",
        "hook_strategy": "simple chantable phrase with rhythmic bounce",
        "avoid": ["overly dense rap bars", "stiff drums", "cold mix"],
        "density": "sparse",
    },
    "amapiano": {
        "label": "Amapiano",
        "aliases": ["amapiano"],
        "caption_dna": ["log drum", "shuffling percussion", "deep piano chords", "club groove"],
        "structure": "Intro, groove build, vocal motif, drop, breakdown, final drop, outro.",
        "bpm": "110-115",
        "keys": "minor or soulful house-adjacent progressions",
        "hook_strategy": "short vocal motif, call and response, room for log drum",
        "avoid": ["too many full verses", "overly bright supersaws", "flat percussion"],
        "density": "sparse",
    },
    "dancehall": {
        "label": "Dancehall",
        "aliases": ["dancehall"],
        "caption_dna": ["dancehall drums", "syncopated bass", "bright percussion", "toasting vocal"],
        "structure": "Intro, Hook, Verse, Hook, Verse, Bridge, Final Hook.",
        "bpm": "90-110",
        "keys": "minor or bright modal loops",
        "hook_strategy": "call-response phrase with percussive consonants",
        "avoid": ["generic island cliches", "stiff backbeat", "crowded hook"],
        "density": "balanced",
    },
    "reggaeton": {
        "label": "Reggaeton",
        "aliases": ["reggaeton"],
        "caption_dna": ["dembow rhythm", "sub-bass", "latin percussion", "glossy vocal"],
        "structure": "Intro, Pre-Hook, Hook, Verse, Hook, Bridge, Final Hook.",
        "bpm": "88-106",
        "keys": "minor with bright hook lift",
        "hook_strategy": "repeatable dance phrase with strong vowel landing",
        "avoid": ["weak dembow", "overwritten verses", "flat vocal stacks"],
        "density": "balanced",
    },
    "pop": {
        "label": "Pop",
        "aliases": ["pop", "radio pop"],
        "caption_dna": ["radio-ready mix", "catchy chorus", "polished drums", "stacked harmonies"],
        "structure": "Intro, Verse, Pre-Chorus, Chorus, Verse, Pre-Chorus, Chorus, Bridge, Final Chorus.",
        "bpm": "90-128",
        "keys": "major, relative minor, clear lift into chorus",
        "hook_strategy": "title phrase in chorus with repeat and emotional payoff",
        "avoid": ["vague inspiration slogans", "weak title connection", "overlong bridge"],
        "density": "balanced",
    },
    "indie_pop": {
        "label": "Indie pop",
        "aliases": ["indie pop", "bedroom pop"],
        "caption_dna": ["jangly guitar", "soft synths", "intimate vocal", "warm tape texture"],
        "structure": "Intro, Verse, Chorus, Verse, Chorus, Bridge, Final Chorus, outro.",
        "bpm": "80-120",
        "keys": "major/minor bittersweet loops",
        "hook_strategy": "small concrete image that turns emotionally",
        "avoid": ["too glossy", "generic nostalgia", "overproduced drums"],
        "density": "balanced",
    },
    "rock": {
        "label": "Rock",
        "aliases": ["rock", "alt rock"],
        "caption_dna": ["electric guitars", "live drums", "bass guitar", "powerful vocal"],
        "structure": "Intro riff, Verse, Chorus, Verse, Chorus, Solo or Bridge, Final Chorus.",
        "bpm": "90-150",
        "keys": "minor pentatonic, modal rock, power-chord movement",
        "hook_strategy": "big chorus phrase with guitar response",
        "avoid": ["thin drums", "weak riff identity", "lyric overcrowding"],
        "density": "balanced",
    },
    "punk_rock": {
        "label": "Punk rock",
        "aliases": ["punk", "punk rock", "pop punk"],
        "caption_dna": ["fast guitars", "driving drums", "shouted hook", "raw energy"],
        "structure": "Count-in, Verse, Chorus, Verse, Chorus, Bridge, Final Chorus, hard stop.",
        "bpm": "150-210",
        "keys": "major/minor power-chord progressions",
        "hook_strategy": "one-line slogan with immediate chorus payoff",
        "avoid": ["overpolished ballad mix", "long abstract verses", "slow intro"],
        "density": "dense",
    },
    "metal": {
        "label": "Metal",
        "aliases": ["metal", "metalcore"],
        "caption_dna": ["heavy guitars", "double kick", "dark bass", "aggressive vocal"],
        "structure": "Intro riff, Verse, Pre-Chorus, Chorus, Verse, Breakdown, Solo, Final Chorus.",
        "bpm": "100-180",
        "keys": "minor, phrygian, chromatic tension",
        "hook_strategy": "contrast harsh verse with clean or chantable chorus",
        "avoid": ["muddy low mids", "uncontrolled screaming", "weak breakdown cue"],
        "density": "balanced",
    },
    "soul_funk": {
        "label": "Soul/Funk",
        "aliases": ["soul", "funk", "soul funk"],
        "caption_dna": ["groove bass", "tight drums", "brass stabs", "warm vocal"],
        "structure": "Intro groove, Verse, Chorus, Verse, Chorus, Breakdown, Final Chorus.",
        "bpm": "86-118",
        "keys": "dominant sevenths, minor pentatonic, gospel turns",
        "hook_strategy": "call-and-response with pocket and space",
        "avoid": ["stiff quantization", "weak bass riff", "too many words"],
        "density": "balanced",
    },
    "lofi": {
        "label": "Lo-fi",
        "aliases": ["lo-fi", "lofi", "lo-fi hip hop"],
        "caption_dna": ["dusty drums", "warm tape hiss", "soft keys", "vinyl texture"],
        "structure": "Intro, loop A, short vocal motif, loop B, breakdown, outro.",
        "bpm": "70-92",
        "keys": "jazzy minor, seventh chords, soft loops",
        "hook_strategy": "minimal phrase or instrumental motif",
        "avoid": ["bright harsh mix", "full pop lyric overload", "over-clean drums"],
        "density": "sparse",
    },
    "house": {
        "label": "House",
        "aliases": ["house", "deep house", "tech house"],
        "caption_dna": ["four-on-the-floor", "sidechain pulse", "warm bass", "club mix"],
        "structure": "Intro, groove build, vocal hook, drop, breakdown, final drop, outro.",
        "bpm": "118-128",
        "keys": "minor or soulful chord loops",
        "hook_strategy": "short phrase over groove, repeat with filter movement",
        "avoid": ["verse-heavy structure", "weak kick", "muddy sidechain"],
        "density": "sparse",
        "instrumental": True,
    },
    "techno": {
        "label": "Techno",
        "aliases": ["techno", "minimal techno"],
        "caption_dna": ["driving kick", "hypnotic synth sequence", "industrial percussion", "dark club mix"],
        "structure": "Intro, build, drop, breakdown, second build, final drop, outro.",
        "bpm": "124-140",
        "keys": "minor, modal, one-note tension",
        "hook_strategy": "instrumental motif or one processed vocal fragment",
        "avoid": ["full verses", "pop chorus overload", "soft kick"],
        "density": "sparse",
        "instrumental": True,
    },
    "trance": {
        "label": "Trance",
        "aliases": ["trance", "uplifting trance"],
        "caption_dna": ["supersaw lead", "rolling bass", "big breakdown", "euphoric risers"],
        "structure": "DJ intro, build, breakdown, lead reveal, drop, final lift, outro.",
        "bpm": "128-140",
        "keys": "minor to major lift, emotional progressions",
        "hook_strategy": "melodic lead hook, sparse vocal lift if used",
        "avoid": ["dense verses", "flat build", "weak risers"],
        "density": "sparse",
        "instrumental": True,
    },
    "drum_and_bass": {
        "label": "Drum and bass",
        "aliases": ["dnb", "drum and bass", "drum & bass"],
        "caption_dna": ["fast breakbeats", "reese bass", "sub pressure", "club energy"],
        "structure": "Intro, build, drop, verse or motif, second drop, breakdown, final drop.",
        "bpm": "160-176",
        "keys": "minor, dark bass movement, atmospheric pads",
        "hook_strategy": "short vocal line or bass motif before drop",
        "avoid": ["slow pop verse layout", "weak sub", "overcrowded vocal"],
        "density": "sparse",
        "instrumental": True,
    },
    "dubstep": {
        "label": "Dubstep",
        "aliases": ["dubstep", "brostep"],
        "caption_dna": ["half-time drums", "wobble bass", "riser impacts", "aggressive drop"],
        "structure": "Intro, build, drop, breakdown, second build, final drop, outro.",
        "bpm": "140-150",
        "keys": "minor, chromatic bass tension",
        "hook_strategy": "short vocal chop into bass-drop answer",
        "avoid": ["full rap verses unless requested", "weak impacts", "muddy bass layers"],
        "density": "sparse",
        "instrumental": True,
    },
    "cinematic": {
        "label": "Cinematic",
        "aliases": ["cinematic", "orchestral", "score"],
        "caption_dna": ["orchestral strings", "brass swells", "taiko drums", "wide stereo"],
        "structure": "Intro motif, build, peak, quiet contrast, final climax, resolving outro.",
        "bpm": "60-110 flexible",
        "keys": "minor, modal, heroic major lift",
        "hook_strategy": "leitmotif rather than pop chorus unless vocals requested",
        "avoid": ["random pop hook", "thin orchestration", "unearned climax"],
        "density": "sparse",
        "instrumental": True,
    },
    "ambient": {
        "label": "Ambient",
        "aliases": ["ambient", "drone", "soundscape"],
        "caption_dna": ["slow pads", "field texture", "soft evolving drones", "wide reverb"],
        "structure": "Fade in, texture A, harmonic bloom, sparse motif, fade out.",
        "bpm": "no strict pulse or 50-80",
        "keys": "modal, suspended, open harmony",
        "hook_strategy": "texture identity instead of lyric hook",
        "avoid": ["busy drums", "full verses", "sharp transients"],
        "density": "sparse",
        "instrumental": True,
    },
    "latin": {
        "label": "Latin",
        "aliases": ["latin", "latin pop", "salsa", "bachata"],
        "caption_dna": ["latin percussion", "nylon guitar", "warm bass", "romantic vocal"],
        "structure": "Intro, Verse, Pre-Hook, Chorus, Verse, Chorus, bridge or dance break, Final Chorus.",
        "bpm": "90-130 depending substyle",
        "keys": "minor/major with bright chorus lift",
        "hook_strategy": "vowel-rich repeated phrase and percussion response",
        "avoid": ["generic tourist cliches", "flat percussion", "overwritten chorus"],
        "density": "balanced",
    },
    "kpop_jpop": {
        "label": "K-pop/J-pop",
        "aliases": ["k-pop", "kpop", "j-pop", "jpop"],
        "caption_dna": ["polished pop production", "section switch-ups", "stacked harmonies", "bright synths"],
        "structure": "Intro, Verse, Pre-Chorus, Chorus, Post-Chorus, Verse/Rap, Bridge, Final Chorus.",
        "bpm": "100-150",
        "keys": "major/minor with bold pre-chorus lift",
        "hook_strategy": "short title chant plus post-chorus motif",
        "avoid": ["same energy every section", "weak pre-chorus", "language/script mismatch"],
        "density": "balanced",
    },
}

GENRE_ALIAS_TO_KEY: dict[str, str] = {}
for _key, _module in GENRE_MODULES.items():
    GENRE_ALIAS_TO_KEY[_key] = _key
    for _alias in _module.get("aliases", []):
        GENRE_ALIAS_TO_KEY[str(_alias).lower()] = _key

SECTION_TAGS = {
    "song": ["[Intro]", "[Verse]", "[Verse 1]", "[Pre-Chorus]", "[Chorus]", "[Post-Chorus]", "[Bridge]", "[Final Chorus]", "[Outro]"],
    "rap": ["[Intro]", "[Verse - rap]", "[Hook]", "[Chorus - rap]", "[Bridge - spoken]", "[Outro]"],
    "edm": ["[Intro]", "[Build]", "[Drop]", "[Breakdown]", "[Build-Up]", "[Final Drop]", "[Outro]"],
    "instrumental": ["[Intro]", "[Instrumental]", "[Build]", "[Drop]", "[Breakdown]", "[Climax]", "[Outro]"],
    "performance": ["[Vocal responses]", "[Call and response]", "[Layered vocals]", "[Whispered]", "[Spoken]"],
}

TROUBLESHOOTING_MATRIX = {
    "muddy_mix": "Reduce conflicting low-end tags, use clearer instrument hierarchy, and avoid piling multiple bass descriptors.",
    "weak_hook": "Make the title phrase shorter, repeat it once, and contrast the hook energy against the verse.",
    "language_drift": "Restate target_language and script policy; remove unintended romanization or code switching.",
    "overlong_lines": "Split long phrases into shorter vocal lines and avoid paragraph-style lyrics.",
    "lyrics_in_caption": "Move all words meant to be sung into lyrics; keep caption as sonic description only.",
    "caption_metadata_leak": "Move BPM, key, duration, seed, model, and time signature into metadata fields.",
    "too_generic": "Add concrete scene details, one metaphor world, and a more specific rhythm/instrument cue.",
    "source_audio_mismatch": "Mark source_audio_mode as duration_locked and preserve source structure in advisory metadata.",
    "unsafe_reference": "No content restrictions applied.",
}

VALIDATION_CHECKLIST = [
    "caption_lyrics_metadata_separated",
    "target_language_script_respected",
    "no_placeholders",
    "hook_has_title_or_emotional_promise",
    "section_map_matches_duration",
    "genre_module_matches_caption",
    "negative_control_present",
    "runtime_fields_supported_or_advisory",
]

DEFAULT_NEGATIVE_CONTROL = [
    "muddy mix",
    "generic lyrics",
    "weak hook",
    "placeholder lyrics",
    "off-key vocal",
    "unclear vocal",
    "noisy artifacts",
    "contradictory style tags",
    "prompt leakage",
]

ADVANCED_GENERATION_ADVISORY = {
    "dcw": "advisory_only_until_handler_supports_runtime_control",
    "scragvae": "advisory_only_until_handler_supports_runtime_control",
    "alternate_vae": "advisory_only_until_handler_supports_runtime_control",
    "lm_cfg": "available only when the selected ACE-Step route supports LM thinking controls",
    "source_audio_duration_lock": "metadata advisory for cover/repaint/extract/lego/complete routes",
}

SPARSE_GENRE_KEYS = {
    "house",
    "techno",
    "trance",
    "drum_and_bass",
    "dubstep",
    "cinematic",
    "ambient",
    "lofi",
    "amapiano",
}


def slugify_genre(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def language_preset(language: Any) -> dict[str, Any]:
    key = str(language or "unknown").strip().lower().replace("-", "_")
    if key in {"cn", "mandarin"}:
        key = "zh"
    if key in {"cant", "cantonese"}:
        key = "yue"
    if key in {"english"}:
        key = "en"
    preset = deepcopy(LANGUAGE_PRESETS.get(key) or LANGUAGE_PRESETS["unknown"])
    preset["code"] = key if key in LANGUAGE_PRESETS else "unknown"
    return preset


def genre_module(name: Any) -> dict[str, Any]:
    raw = str(name or "").strip().lower()
    key = GENRE_ALIAS_TO_KEY.get(raw) or GENRE_ALIAS_TO_KEY.get(slugify_genre(raw)) or "pop"
    module = deepcopy(GENRE_MODULES[key])
    module["slug"] = key
    return module


def infer_genre_modules(text: Any, max_modules: int = 2) -> list[dict[str, Any]]:
    haystack = str(text or "").lower()
    found: list[str] = []
    for alias, key in sorted(GENRE_ALIAS_TO_KEY.items(), key=lambda item: len(item[0]), reverse=True):
        if key in found:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(alias).replace(r"\ ", r"[\s_-]+") + r"(?![a-z0-9])"
        if re.search(pattern, haystack):
            found.append(key)
        if len(found) >= max(1, int(max_modules or 1)):
            break
    if not found:
        found = ["pop"]
    return [genre_module(key) for key in found[: max(1, int(max_modules or 1))]]


def is_sparse_lyric_genre(text: Any) -> bool:
    modules = infer_genre_modules(text, max_modules=3)
    return any(module.get("slug") in SPARSE_GENRE_KEYS or module.get("instrumental") for module in modules)


def section_map_for(duration: Any, genre_hint: Any = "", instrumental: bool = False) -> list[dict[str, Any]]:
    try:
        dur = max(30, min(600, int(float(duration or 180))))
    except Exception:
        dur = 180
    sparse = bool(instrumental)
    if sparse:
        if dur <= 90:
            sections = [("Intro", 0, 18), ("Build", 18, 42), ("Drop", 42, 72), ("Outro", 72, dur)]
        elif dur <= 210:
            sections = [
                ("Intro", 0, 24),
                ("Build", 24, 58),
                ("Drop", 58, 104),
                ("Breakdown", 104, 136),
                ("Final Drop", 136, max(150, dur - 18)),
                ("Outro", max(150, dur - 18), dur),
            ]
        else:
            sections = [
                ("DJ Intro", 0, 32),
                ("Build", 32, 70),
                ("Drop", 70, 128),
                ("Breakdown", 128, 170),
                ("Second Build", 170, 210),
                ("Final Drop", 210, max(225, dur - 26)),
                ("Outro", max(225, dur - 26), dur),
            ]
        return [
            {"tag": f"[{name}]", "start_sec": int(start), "end_sec": int(min(end, dur)), "vocal_role": "instrumental_or_sparse_motif"}
            for name, start, end in sections
            if end > start
        ]
    rap_locked = bool(re.search(r"\b(?:rap|hip[-\s]?hop|trap|drill|boom[-\s]?bap|g[-\s]?funk|west coast)\b", str(genre_hint or ""), re.I))
    if rap_locked:
        if dur <= 90:
            sections = [("Intro", 0, 8), ("Verse", 8, 44), ("Chorus", 44, 72), ("Outro", 72, dur)]
        elif dur <= 180:
            sections = [
                ("Intro", 0, 10),
                ("Verse 1", 10, 52),
                ("Chorus", 52, 82),
                ("Verse 2", 82, 124),
                ("Bridge", 124, 150),
                ("Final Chorus", 150, dur),
            ]
        elif dur <= 210:
            sections = [
                ("Intro", 0, 12),
                ("Verse 1", 12, 56),
                ("Chorus", 56, 82),
                ("Verse 2", 82, 128),
                ("Bridge", 128, 160),
                ("Final Chorus", 160, max(176, dur - 12)),
                ("Outro", max(176, dur - 12), dur),
            ]
        else:
            sections = [
                ("Intro", 0, 12),
                ("Verse 1", 12, 58),
                ("Chorus", 58, 84),
                ("Verse 2", 84, 130),
                ("Second Chorus", 130, 154),
                ("Verse 3 - Beat Switch", 154, max(188, dur - 52)),
                ("Bridge", max(188, dur - 52), max(210, dur - 30)),
                ("Final Chorus", max(210, dur - 30), max(226, dur - 12)),
                ("Outro", max(224, dur - 12), dur),
            ]
        return [
            {"tag": f"[{name}]", "start_sec": int(start), "end_sec": int(min(end, dur)), "vocal_role": "rap_lyrics"}
            for name, start, end in sections
            if end > start
        ]
    if dur <= 90:
        sections = [("Intro", 0, 8), ("Verse", 8, 42), ("Chorus", 42, 72), ("Outro", 72, dur)]
    elif dur <= 180:
        sections = [
            ("Intro", 0, 10),
            ("Verse 1", 10, 45),
            ("Pre-Chorus", 45, 62),
            ("Chorus", 62, 92),
            ("Verse 2", 92, 125),
            ("Bridge", 125, 150),
            ("Final Chorus", 150, dur),
        ]
    else:
        sections = [
            ("Intro", 0, 12),
            ("Verse 1", 12, 52),
            ("Pre-Chorus", 52, 72),
            ("Chorus", 72, 106),
            ("Verse 2", 106, 146),
            ("Pre-Chorus", 146, 166),
            ("Chorus", 166, 200),
            ("Bridge", 200, max(212, dur - 48)),
            ("Final Chorus", max(212, dur - 48), max(224, dur - 12)),
            ("Outro", max(224, dur - 12), dur),
        ]
    return [
        {"tag": f"[{name}]", "start_sec": int(start), "end_sec": int(min(end, dur)), "vocal_role": "lyrics"}
        for name, start, end in sections
        if end > start
    ]


def negative_control_for(genre_hint: Any = "", instrumental: bool = False) -> list[str]:
    controls = list(DEFAULT_NEGATIVE_CONTROL)
    if instrumental or is_sparse_lyric_genre(genre_hint):
        controls.extend(["forced full verses", "crowded vocals", "pop chorus overload"])
    if re.search(r"\b(rap|hip.?hop|trap|drill)\b", str(genre_hint or ""), re.I):
        controls.extend(["generic flexing"])
    seen: set[str] = set()
    result: list[str] = []
    for item in controls:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _quality_checks_template() -> dict[str, str]:
    return {name: "pending" for name in VALIDATION_CHECKLIST}


def kit_metadata_defaults(
    mode: str = "text2music",
    language: Any = "en",
    genre_hint: Any = "",
    duration: Any = 180,
    instrumental: bool = False,
) -> dict[str, Any]:
    preset = language_preset(language)
    modules = infer_genre_modules(genre_hint, max_modules=2)
    genre_slugs = [module["slug"] for module in modules]
    sparse = bool(instrumental)
    return {
        "prompt_kit_version": PROMPT_KIT_VERSION,
        "concept_summary": "",
        "target_language": preset["code"],
        "vocal_language": preset["vocal_language"],
        "language_notes": preset["notes"],
        "genre_profile": {
            "primary": modules[0]["slug"],
            "modules": genre_slugs,
            "hook_strategy": modules[0]["hook_strategy"],
            "caption_dna": modules[0]["caption_dna"],
        },
        "genre_modules": genre_slugs,
        "song_length_mode": "full_hit_210_270" if 210 <= int(float(duration or 180)) <= 270 else "custom_duration",
        "section_map": section_map_for(duration, genre_hint, instrumental=sparse),
        "lyric_density_notes": "Sparse vocal motifs and instrumental timeline." if sparse else "Full hit lyric coverage with verses, hook, contrast, and outro.",
        "ace_caption": "",
        "lyrics": "",
        "metadata": {
            "duration": int(float(duration or 180)),
            "vocal_language": preset["vocal_language"],
        },
        "generation_settings": {},
        "workflow_mode": mode,
        "source_audio_mode": "source_locked" if mode in {"cover", "repaint", "extract", "lego", "complete"} else "none",
        "runtime_profile": {
            "duration_seconds": int(float(duration or 180)),
            "handler_supported_runtime_knobs_only": True,
        },
        "advanced_generation_settings": deepcopy(ADVANCED_GENERATION_ADVISORY),
        "iteration_plan": [
            "Generate first pass.",
            "Listen for hook, vocal clarity, language/script drift, and low-end balance.",
            "If needed, adjust caption density or section map before regenerating.",
        ],
        "community_risk_notes": [],
        "troubleshooting_hints": [],
        "variations": [
            {"name": "safer_radio", "change": "cleaner language, brighter hook, less aggressive low end"},
            {"name": "club_energy", "change": "stronger drums, shorter lyric lines, more movement in the hook"},
        ],
        "negative_control": negative_control_for(genre_hint, instrumental=sparse),
        "quality_checks": _quality_checks_template(),
        "anti_ai_rewrite_notes": "Rewrite generic lines into concrete scenes; keep one metaphor world and avoid formulaic AI phrasing.",
        "copy_paste_block": "",
    }


def prompt_kit_payload() -> dict[str, Any]:
    return {
        "version": PROMPT_KIT_VERSION,
        "output_contract_fields": PROMPT_KIT_OUTPUT_CONTRACT_FIELDS,
        "metadata_fields": PROMPT_KIT_METADATA_FIELDS,
        "master_rules": MASTER_RULES,
        "language_presets": LANGUAGE_PRESETS,
        "genre_modules": GENRE_MODULES,
        "section_tags": SECTION_TAGS,
        "troubleshooting_matrix": TROUBLESHOOTING_MATRIX,
        "validation_checklist": VALIDATION_CHECKLIST,
        "negative_control": DEFAULT_NEGATIVE_CONTROL,
        "advanced_generation_advisory": ADVANCED_GENERATION_ADVISORY,
    }


def prompt_kit_system_block(mode: str = "custom") -> str:
    contract = ", ".join(PROMPT_KIT_OUTPUT_CONTRACT_FIELDS)
    metadata = ", ".join(PROMPT_KIT_METADATA_FIELDS)
    languages = ", ".join(LANGUAGE_PRESETS.keys())
    genres = ", ".join(GENRE_MODULES.keys())
    return (
        f"ACE-Step Multilingual Hit Prompt Kit\n"
        f"Version: {PROMPT_KIT_VERSION}\n"
        "Use this kit as the final contract for every AceJAM prompt assistant route.\n"
        "Core rules: keep caption, lyrics, and metadata separate; no placeholders; "
        "write in the requested language/script; keep hooks memorable and section maps realistic.\n"
        f"Mode: {mode}. The 24-part output contract is: {contract}.\n"
        f"Add these AceJAM storage metadata fields when applicable: {metadata}.\n"
        f"Language presets available: {languages}.\n"
        f"Genre modules available: {genres}.\n"
        "When filling Simple, Custom, Cover, Repaint, Lego, or Complete, also populate song_intent with "
        "genre_family, subgenre, mood, energy, vocal_type, language, drum_groove, bass_low_end, "
        "melodic_identity, texture_space, mix_master, custom_tags, and caption. "
        "Caption must be a concrete sonic portrait built from those menu choices, not a vague genre sentence.\n"
        "For polished Simple, Custom, Song, Album, and News output, set use_format=false unless the user explicitly asks for raw format/rewrite. "
        "For rough Improve, Cover, Repaint, Lego, Complete, and source-audio workflows, keep source_audio_mode and runtime_profile clear, and mark unsupported runtime controls as advisory metadata. "
        "Never hardcode planner_lm_provider to ollama; use the selected local provider and planner_model.\n"
        "Return valid JSON with the existing ACEJAM_PAYLOAD_JSON, ACEJAM_ALBUM_SETTINGS_JSON, or ACEJAM_DATASET_JSON contract plus kit metadata."
    )
