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
    "rap": ["[Intro]", "[Verse - rap]", "[Hook]", "[Hook/Chorus]", "[Chorus - rap]", "[Bridge - spoken]", "[Outro]"],
    "boom_bap": ["[Intro]", "[Verse - rap]", "[Hook]", "[Verse - rap]", "[Hook]", "[Bridge - spoken]", "[Verse - rap]", "[Hook]", "[Outro]"],
    "g_funk": ["[Intro - talkbox]", "[Verse - rap]", "[Hook - sung]", "[Verse - rap]", "[Hook - sung]", "[Bridge - g-funk solo]", "[Verse - rap]", "[Hook - sung]", "[Outro - talkbox]"],
    "trap": ["[Intro]", "[Verse - rap]", "[Hook]", "[Verse - rap]", "[Hook]", "[Bridge - melodic rap]", "[Hook]", "[Outro]"],
    "drill": ["[Intro - dark]", "[Verse - rap]", "[Hook - chant]", "[Verse - rap]", "[Hook - chant]", "[Outro]"],
    "edm": ["[Intro]", "[Build]", "[Drop]", "[Breakdown]", "[Build-Up]", "[Final Drop]", "[Outro]"],
    "instrumental": ["[Intro]", "[Instrumental]", "[Build]", "[Drop]", "[Breakdown]", "[Climax]", "[Outro]"],
    "performance": ["[Verse - whispered]", "[Chorus - layered vocals]", "[Chorus - call and response]", "[Bridge - spoken]", "[Outro - acapella]"],
}

ACE_STEP_AUTHORING_RULES: list[str] = [
    "Modifier syntax: write tags as [Section - modifier] with one dash and ONE modifier max. Stacking modifiers confuses ACE-Step and the tag content may be sung as lyrics.",
    "Lyric meta tags use square brackets only for sections ([Verse], [Verse - rap], [Hook], [Bridge], [Outro], [inst]). Background vocals use parentheses around the words: 'main line (echo)' is rendered as a backing vocal and not as a tag.",
    "Vocal techniques (whispered, ad-libs, harmonies, falsetto, spoken word, call-and-response, powerful belting, shouted, layered vocals, narration) and energy/emotion descriptors (high energy, low energy, melancholic, explosive, building energy, euphoric, dreamy, calm, intense) go in the comma-separated tags field. Use them inside square brackets ONLY as section modifiers like [Verse - whispered], [Chorus - layered vocals], [Climax - powerful], not as standalone bracket lines in the lyrics. ACE-Step's official examples never use standalone vocal-technique or energy brackets.",
    "ALL CAPS lyric lines render as shouted intensity. Use sparingly for hook accents like 'WE RUN THIS!' or one-word chants, never for whole verses.",
    "BPM, key/scale, time signature, and duration go in the dedicated metadata fields. ACE-Step also reads them from tag prose ('130 bpm', 'G major'), but our policy keeps them only in metadata to avoid double sources.",
    "Caption is sound-only: genre, mood, instruments, timbre, era, production style, vocal type, rhythm, structure energy. No song titles, no producer credits as prose, no lyrics, no JSON, no BPM/key/time text.",
    "Lyrics are a temporal script: a section tag header per block, then performable lines (rap can run 6-14 syllables/line, sung 6-10). Internal rhyme and ad-libs go inside the lyric text or in (parens) on the same line, not as separate tags.",
    "Producer references: never put a producer or label name in the caption. Map the request to the matching Producer-Format Cookbook entry below and use that tag stack instead. ACE-Step does not recognise producer names; it only responds to genre+era+drum+timbre vocabulary.",
    "Six-dimension beat-spec rule: every caption stack must cover at least five of these six dimensions: drum kit (kick + snare + hat triad), bass character, sample/source material, mix treatment, era marker, groove word. Missing one is acceptable; missing two reads as 'names a few instruments' and underspecifies the beat.",
    "Mono-bass / wide-pad split rule: when the producer is Dre / Metro / Mike Dean / Pharrell or any modern-trap or cinematic style, include both halves: 'mono 808 sub' + 'wide stereo pads' or 'mono low end' + 'wide stereo synths'. A single 'wide stereo' tag without the mono counterpart pulls the low end apart.",
    "Sample-source rule: never use the bare word 'sample' alone. Pair with origin genre + treatment: 'soul sample chops', 'jazz sample loop', 'gospel chop', 'film score string sample', 'replayed funk interpolation', 'chopped horn sample'. Highest-leverage upgrade for production specificity.",
    "Drum kit triad rule: specify kick + snare + hat as separate tokens ('punchy kick, dusty snare, shuffled hi-hat') instead of a single 'drums' token. ACE-Step's official prompts use compound terms ('Heavy Riffs', 'Blast Beats', 'punchy snare') and rewards specificity in the drum slot.",
    "Era + groove pairing rule: every cookbook entry must include exactly one era token ('90s G-funk', '2010s trap', 'NYC east coast warmth') AND one groove token ('head-nod groove', 'behind-the-beat', 'trap bounce'). Era anchors timbre era; groove anchors timing feel. They are not interchangeable.",
    "Section modifier dash: use ASCII hyphen `-` ([Verse - rap]). The model also accepts em-dash `–` (canonical in official examples like [Intro – Spoken], [Hook/Chorus – Reprise]). Pick one style per song; do not mix.",
    "Rap verse minimum: every [Verse - rap] section is at least 16 bars. 1 bar = 4 beats. At 90 BPM, 16 bars ≈ 42 seconds. Pack 8-15 syllables per bar (push to ~20 only on emotional spikes). 16 lines is the floor for a rap verse on tracks ≥120 seconds. Long-form story tracks (Nas / Eminem) can push to 32 bars.",
    "Songwriter craft: rhyme on sounds not words; stack multisyllabic mosaic rhymes in begin/middle/end of bars; slant-dominant with perfect-rhyme landings on emphasis; show concrete scene details over abstract slogans; every verse must change something (scene, POV, time, escalation, revelation). See appended SONGWRITER CRAFT block for the full 12-rule cookbook plus Eminem / 2Pac / Kendrick / Nas signatures.",
    "Anti-pattern guard: forbid AI-cliché phrases (neon dreams, fire inside, shattered dreams, endless night, empty streets, embers, whispers, silhouettes, echoes of, we rise, let it burn, chasing the night, broken heart, rising from the ashes, stars aligned, fade away, into the void, burning bright, stolen kisses, tears like rain, frozen in time). Reject any draft that contains telling-not-showing emotional labels ('I feel sad', 'my heart is broken'), generic POV ('we all', 'the world', 'everyone'), or explanation lines ('in other words', 'what I mean is'). See appended ANTI-PATTERNS block.",
]

PRODUCER_FORMAT_COOKBOOK: dict[str, str] = {
    # ACE-Step does NOT recognize producer names directly. Each entry covers
    # six dimensions: drums (kick + snare + hat triad), bass character, sample
    # source + treatment, mix treatment, era marker, groove word. Stays under
    # 512 chars (ACE-Step caption hard cap) when combined with mood/genre tags.
    "Dr. Dre / G-funk era": (
        "G-funk, West Coast hip hop, 90s G-funk, talkbox lead, whistle synth lead, "
        "Minimoog synth bass, replayed funk interpolation, sub-heavy kick, "
        "layered chest-hit snare, smooth closed hi-hat, mono low end, "
        "wide stereo synths, polished mix, head-nod groove, laid-back groove, "
        "summer banger polish"
    ),
    "Dr. Dre / Chronic 2001 + Get Rich era": (
        "Chronic 2001 era, post-Aftermath polish, early 2000s West Coast rap, "
        "cinematic string arrangement, sweeping orchestral strings, "
        "glassy synth piano lead, sparkling Rhodes lead, Mike Elizondo live bass guitar, "
        "mono punchy sub, sub-heavy tight kick, layered clap-snare combo, "
        "crisp programmed hi-hat, brass swell accents, Million Dollar Mix polish, "
        "loud mastered polish, wide stereo pads, mono low end, menacing anthemic energy, "
        "head-nod groove, Aftermath production polish, In Da Club bounce"
    ),
    "No I.D. / Common-era boom bap": (
        "boom bap, soul sample chops, jazzy chord loop, muted piano sample, "
        "soft horn under sample, SP-1200 played bassline, warm round sub, "
        "soft kick, tight backbeat snare, shuffled closed hi-hat, dusty drums, "
        "vinyl texture, warm analog mix, tape warmth, behind-the-beat groove, "
        "head-nod groove, 90s boom bap, NYC east coast warmth"
    ),
    "Metro Boomin / dark trap": (
        "modern trap, 2010s trap, dark atmospheric, sub-heavy 808, mono 808 sub, "
        "harmonic-distorted 808 mids, sparse drums, half-time snare, hard-hitting kick, "
        "fast triplet hi-hat rolls, hi-hat rolls, minor-key piano, filtered choir, "
        "haunting bell melody, eerie bell, wide stereo synth pads, cinematic tension, "
        "trap bounce"
    ),
    "Quincy Jones / 80s pop polish": (
        "80s pop polish, R&B/funk fusion, layered flugelhorn, alto horn, French horn warmth, "
        "soft tuba foundation, dual Minimoog synth bass, slap bass, Afro-Cuban percussion, "
        "congas, shekere, cowbell, cinematic dissonant string tension, tight punchy kick, "
        "gated reverb snare, layered backing vocals, wide stereo, mono bass, studio-polished"
    ),
    "Mobb Deep / NYC street rap": (
        "90s boom bap, NYC street rap, claustrophobic mood, eerie minor-key piano loop, "
        "soft ominous low strings, gritty MPC60 drums, hard sparse kick, hard-snare backbeat, "
        "rumbling sub, dusty mix, dry vocal, raw, head-nod groove, behind-the-beat"
    ),
    "J Dilla / Soulquarian feel": (
        "boom bap, soulquarian feel, drunken swing, MPC quantize-off feel, swung drums, "
        "dragged hi-hat, rushed snare, dusty kit samples, jazzy soul sample loop, "
        "pitched mid-loop chop, round warm bass, slightly out-of-tune tension, "
        "vinyl crackle, lo-fi mix, dusty mix, behind-the-beat groove, head-nod groove, "
        "2000s underground hip-hop"
    ),
    "Pete Rock / golden-age soul boom bap": (
        "boom bap, 90s NYC golden-age, soul horn stabs, tenor sax loop, jazzy sample chops, "
        "filtered multi-layer sample, SP-1200 'pointy' drums, hard kick, deep snare, "
        "prominent breakbeat hi-hat, walking bassline, live-bass character, summery mood, "
        "warm analog mix, head-nod groove, NYC east coast warmth"
    ),
    "Timbaland / early 2000s R&B-rap": (
        "2000s R&B polish, syncopated polyrhythmic drums, off-grid drum patterns, "
        "vocal beatbox layer, mouth percussion, vocal stabs, pitched percussion tuned to key, "
        "tabla, Middle Eastern percussion, clean sparse sub-bass, kick panned wide, "
        "central voice unobstructed, exotic percussion, glossy mix, head-nod groove"
    ),
    "Pharrell / Neptunes minimal": (
        "Neptunes minimal, 2000s pop-rap polish, minimal arrangement, syncopated rhythm, "
        "808 cowbell, syncopated shaker, robotic kick-clap, tight dry kick, processed clap, "
        "clavinet, electric piano, synthetic guitar, sci-fi FX, falsetto vocal accents, "
        "glossy mix, percussive bounce, head-nod groove, 3-4 sounds total"
    ),
    "Kanye West / 808s era": (
        "808s and Heartbreak era, electro-pop minimalism, TR-808, LinnDrum LM-2, "
        "step-input drum programming, sparse drums, distorted heartbreak 808, "
        "compressed saturated 808 bass, droning synth pads, lengthy strings, "
        "somber piano, minor-key, auto-tune as instrument, dense reverbed drums, "
        "austere mix, 2000s rap"
    ),
    "Mike Dean / cinematic rap": (
        "cinematic rap, 2010s trap polish, Jupiter-8 synth lead, CS-80-like lead, "
        "MemoryMoog screaming overdrive lead, smoke-toned analog monosynth, "
        "tape saturation, Decapitator-style 808 grit, saturated 808 bass, "
        "EchoBoy delay tails, big ambient hall reverb, wide stereo synth pads, "
        "mono punchy bass, atmospheric, cinematic tension, head-nod groove"
    ),
    "DJ Premier / 90s boom bap": (
        "90s boom bap, NYC street rap, scratched vocal-stab hook, chopped speech sample, "
        "loop discipline, weird background sax stabs, vinyl noise, hard programmed drums, "
        "sparse kick, deep bass, looped chord stab, raw mix, slightly behind-the-beat groove, "
        "head-nod groove"
    ),
    "Rick Rubin / stripped rap-rock": (
        "stripped rap-rock, naked TR-808, no melodic loop, single-take guitar, "
        "raw unprocessed riff, dry present vocal, uncomfortably sparse arrangement, "
        "punchy kick, hard snare, reductionist mix, imperfection retained, "
        "raw demo feel, 80s rap-rock crossover"
    ),
    "Madlib / loop-driven boom bap": (
        "loop-driven boom bap, micro-chopped jazz sample, world-music sample, "
        "pitched stitched loops, deep-in-the-mix dusty drums, non-quantized loose drums, "
        "drunken swing, SP-303 vinyl-sim crackle, exaggerated dust texture, "
        "round warm analog bass, psychedelic woozy pitch-shifted vocal interludes, "
        "behind-the-beat groove, lo-fi mix, 2000s underground hip-hop"
    ),
    "Just Blaze / triumphant soul hip-hop": (
        "triumphant soul hip-hop, anthemic stadium feel, Lafayette Afro Rock-style horns, "
        "Johnny Pate-style orchestral horn stabs, chopped scratched soul vocal hook, "
        "live-drum re-mic'd break, booming kick, marching rim-shot snare, polished punchy mix, "
        "NYC east coast warmth, head-nod groove, building energy, 2000s rap"
    ),
    "Havoc / Mobb Deep production": (
        "90s boom bap, Queensbridge street-noir, claustrophobic mood, eerie detuned piano, "
        "single-note minor-key piano loop, soft ominous low strings, MPC3000 drums, "
        "gritty hard snare, sparse kick, heavy rumbling sub, dusty mix, dry vocal, "
        "raw vinyl-cracked sample, head-nod groove, behind-the-beat"
    ),
    "Stoupe / cinematic hardcore hip-hop": (
        "cinematic hardcore hip-hop, 2000s underground hip-hop, horror score string sample, "
        "film score sample, Latin choir chop, operatic vocal snippet, Greek folk sample, "
        "world-music source, dramatic orchestral stab, taiko drums layered with boom bap, "
        "hard-hitting MPC drums, deep rumbling sub, dusty piano loop, ominous low strings, "
        "film dialogue snippet transitions, big reverb tail, filmic atmosphere, dark, "
        "claustrophobic, head-nod groove"
    ),
    # MODERN (2023-2026 chart-topper) producer entries.
    "Mustard / West Coast diss-track ratchet": (
        "ratchet West Coast hip hop, 2020s rap, Mustard signature, hyphy bass, "
        "finger-snap percussion, sliding sub-bass, dembow-tinted kick pattern, "
        "violin sample, sparse drums, half-time snare, 4/4 hand claps, mono 808, "
        "anthem-shout hook, polished commercial mix, head-nod groove, trap bounce, "
        "Compton 2024 sound, 'Not Like Us' template"
    ),
    "Pi'erre Bourne / plugg + rage hybrid": (
        "plugg, rage rap, 2020s underground rap, melodic synth lead, "
        "stereo-widened EDM-leaning lead, distorted overdriven 808, "
        "sparse trap drums, fast triplet hi-hat rolls, sub-bass swells, "
        "dreamy atmospheric pad, glossy mix, hyperpop-tinged melody, "
        "Yeat / Carti / Lil Uzi Vert era, modern trap polish"
    ),
    "Tay Keith / Memphis trap": (
        "Memphis trap, 2010s-2020s trap, Tay Keith bounce, hard 808 kick, "
        "rolling triplet hi-hat, snappy snare on 3, sparse melody, "
        "distorted Memphis sample, sliding 808 bass, dark menacing pad, "
        "anthemic chant hook, hard-hitting mix, GloRilla / BlocBoy template, "
        "trap bounce, head-nod groove"
    ),
    "AXL Beats / Brooklyn drill": (
        "Brooklyn drill, NY drill, 2020s drill, AXL Beats template, "
        "sliding 808 bass gliding scale-degree-to-scale-degree, sparse pianos, "
        "icy synth pads, stuttering hi-hat triplet rolls, bell melody, "
        "hard half-time kick, snare on 3, dark cinematic, drill bounce, "
        "Pop Smoke / Fivio Foreign era"
    ),
    "Central Cee / UK drill melodic-rap": (
        "UK drill, 140 BPM half-time, sliding 808s gliding scale-degree-to-scale-degree, "
        "chopped acoustic guitar loop, operatic vocal sample chop, melodic-rap hybrid, "
        "auto-tune retune around 25ms, sing-rap blend, stuttering hi-hat rolls, "
        "money-imagery hook, polished UK mix, head-nod groove, Sprinter / BAND4BAND template"
    ),
    "Jersey Club / PinkPantheress era": (
        "Jersey club, 2020s alt R&B, PinkPantheress template, four-on-the-floor club kick, "
        "syncopated hand claps on the off-beats, breakbeat sample chops, "
        "bedroom pop polish, breathy female vocal, Y2K nostalgia synths, "
        "pitched-up vocal chops, sliding 808, glossy mix, fast tempo around 140 BPM, "
        "Boy's a Liar Pt. 2 template"
    ),
    "Finneas / Billie Eilish bedroom pop": (
        "bedroom pop, alt pop, breathy whisper-aesthetic, close-mic vocal, "
        "minimal 4-chord palette across whole song, sub-bass-heavy minimalism, "
        "soft Rhodes piano, cinematic strings comeback for emotional climax, "
        "introspective mood, conversational diction, ASMR vocal texture, "
        "plate reverb on snare only, Birds of a Feather / Lunch template"
    ),
    "Carter Lang + Julian Bunetta / Sabrina Carpenter retro-disco pop": (
        "retro disco-pop, yacht-rock revival, vintage disco bass walk, "
        "2020s pop polish, syncopated guitar chops, electric piano stabs, "
        "tight punchy kick, snare on 2 and 4, hand-claps stacked, "
        "displaced-downbeat melody starting on beat 2, glossy commercial mix, "
        "Espresso / Please Please Please template, sex-positive humor topline"
    ),
    "Mike WiLL Made It / 2010s-2020s anthemic trap": (
        "anthemic trap, 2010s trap, Mike WiLL bounce, big 808 kick, "
        "snappy snare on 3, fast triplet hi-hat rolls, ominous synth lead, "
        "stadium-anthem hook, hard mix, half-time drums, sparse melody, "
        "Rae Sremmurd / Future / 21 Savage template, modern trap polish"
    ),
    "Bruno Mars / retro pop-rock revival": (
        "retro pop-rock, 80s pop-funk revival, 2020s pop polish, "
        "tight slap bass, syncopated funk guitar, brass stab section, "
        "live-drum kit, snare on 2 and 4, vintage Wurlitzer organ, "
        "anthemic shout hook, polished radio mix, APT. / Die With a Smile template, "
        "head-nod groove, dance-pop bounce"
    ),
    "Mustard x Kendrick / 2024 hyphy revival": (
        "ratchet West Coast 2024, hyphy bass + sliding sub, snap-clap percussion, "
        "violin string sample, finger-snap pattern, sparse drums, "
        "anthemic 4-syllable chant hook, polished commercial 2024 mix, "
        "Compton revival, mono 808 + wide stereo synths, head-nod groove, "
        "Not Like Us template"
    ),
}

# SONGWRITER_CRAFT_COOKBOOK distills lyric craft principles from Eminem, 2Pac,
# Kendrick Lamar, Nas, MF DOOM and Pat Pattison-style songwriting pedagogy into
# concrete rules the local LLM can actually apply. Sources cited via research
# audits stored in ~/.claude/plans/. These get appended to every music-mode
# system block so the writer reaches for craft moves instead of generic phrasing.
SONGWRITER_CRAFT_COOKBOOK: dict[str, str] = {
    "rhyme_stacking": (
        "Rhyme on sounds, not whole words. Stack rhymes in beginning/middle/end of each bar — "
        "Eminem's 'Lose Yourself' verse stacks sweaty/knees weak/heavy/spaghetti/ready as "
        "multiple multisyllabic chains across 4 lines. Default to 2-4 syllable mosaic rhymes; "
        "reserve perfect end-rhyme for emotional resolution."
    ),
    "rhyme_density": (
        "Slant-dominant with periodic perfect-rhyme landings on emphasis words. Pure rhyme "
        "everywhere reads like Dr. Seuss; pure prose has no song. Sweet spot is ~70% slant / "
        "internal / multisyllabic plus ~30% perfect end-rhymes on payoff lines."
    ),
    "bar_anatomy": (
        "1 bar = 4 beats. Standard rap verse = 16 bars (~64 beats; ~42 seconds at 90 BPM). "
        "Long-form storytelling (Nas Illmatic, Eminem Renegade) pushes to 32 bars. "
        "8-15 syllables per bar is the working range; push to ~20 only for emotional spike "
        "(Eminem furious on Godzilla = 11.3 syllables/sec)."
    ),
    "flow_pocket_vs_grid": (
        "Lock to the grid for verses 1-2. Break the grid (drift across bar lines, triplets, "
        "double-time) only for emotional spikes. Kendrick uses triplets only at 'high tension' "
        "moments, not as default flow. Pocket beats acrobatics."
    ),
    "show_dont_tell": (
        "No 'this is bad, don't do this' lines. Tell it as a scene. Concrete sensory detail — "
        "Nas writes trap doors, rooftop snipers, lobby kids; Tom Waits writes 'a stain on your "
        "bedroom wall, the flavour of a soda they stopped making, a girl's name you made up'. "
        "Specific is universal."
    ),
    "specific_over_abstract": (
        "One small specific beats ten generic. A discontinued soda flavour, a real first name, "
        "a specific street corner, a brand of cigarettes. Banned generic POV: 'the world', "
        "'everyone', 'the people', 'we all', 'this generation'."
    ),
    "punchline_construction": (
        "Setup + payoff. Plant a phrase in line 1 whose meaning flips in line 2 (homophone, "
        "double entendre, conceptual pivot). Eminem and Big L pair entendres with mind-rhyme. "
        "Ban explanation lines like 'in other words...' or 'what I mean is'."
    ),
    "section_purpose": (
        "Verse = scene. Chorus = single phrase that survives repetition. Bridge = NEW angle "
        "(zoom out, time shift, confession, counter-argument). Each section occupies a different "
        "sonic and emotional space. A bridge that just restates verse 3 is dead weight."
    ),
    "hook_hum_test": (
        "If a stranger can't hum the hook after one listen, rewrite. Keep hooks to ONE phrase "
        "or a few notes. The hook must work without the verses — someone hearing only the chorus "
        "should grasp the song's thesis."
    ),
    "prosody_match": (
        "Pat Pattison's rule: form supports content. Stable content (resolution, calm) = AABB or "
        "ABAB, even line count, perfect rhyme. Unstable content (longing, tension) = ABBA, odd "
        "line count, slant rhyme. Don't rhyme a tense lyric like a love song."
    ),
    "ad_libs_punctuation": (
        "Ad-libs are punctuation, not decoration. Use them to mark a payoff line, not every "
        "4 bars. Travis Scott reserves 'It's lit!' for moments that deserve that energy. "
        "Constant ad-libs feel cheap and drain meaning."
    ),
    "verse_must_change_something": (
        "Every verse must change something — new scene, new POV, time jump, escalation, or "
        "revelation. A verse that just restates the chorus is dead weight. Cut it or rewrite "
        "with a new angle."
    ),
    "eminem_signature": (
        "Stacks rhymes across beginning/middle/end columns of every bar. Rhymes 'sounds with "
        "sounds', not whole words. Late-career adds homophone + multi-entendre + mind-rhyme. "
        "Flow accelerates/decelerates with character emotion (Lose Yourself: hesitant → "
        "rapid-fire); pocket holds over half-time."
    ),
    "tupac_signature": (
        "Storytelling first, technique second. Says exactly what he means, literally and clearly. "
        "Brenda's Got a Baby = single-verse three-act narrative (community → incident → "
        "consequence). Character voicing: third-person omniscient (Brenda), direct address "
        "(Dear Mama). Empathy over technical density."
    ),
    "kendrick_signature": (
        "Concept-album narrative threading (good kid m.A.A.d city = single day, To Pimp a "
        "Butterfly = single poem revealed line-by-line across tracks, fully on Mortal Man). "
        "Switches between music-rhythmic (locked grid, King Kunta) and speech-rhythmic "
        "(off-grid, Momma v2). Triplets reserved for high-tension moments."
    ),
    "nas_signature": (
        "'Hip-hop Hemingway' — each line a short story connecting to a larger narrative. "
        "Concrete sensory anchors per bar (trap doors, rooftop snipers, lobby kids). Mid-verse "
        "pivot from braggadocio to immersive scene to moral reflection. One Love = epistolary "
        "form, verses written as letters to incarcerated friends."
    ),
    # MODERN HIT-CRAFT LAYER (2023-2026 chart anatomy). Sourced from Hit Songs
    # Deconstructed, Berklee Songwriting, MusicRadar Antonoff/Finneas/Max Martin
    # interviews, Billboard YE reports. We write to chart in 2025-2026, not 1995.
    "modern_title_drop_vowel_lock": (
        "Title-drop on line 1 of the chorus and again as the last vowel-locked syllable. "
        "Rhyme that vowel through 3-4 successive end-words. 'Espresso' loops 'that's that "
        "me espresso' on /oʊ/ then stacks six verbs all locking to /uː/. 'Flowers' opens "
        "'I can buy myself flowers' as the title-drop. Title is the first thing the listener "
        "hears in the chorus, not buried in the second half."
    ),
    "modern_three_chorus_framework": (
        "Default 2024-2026 song shape: V1 -> Pre-Chorus -> Chorus -> V2 -> Pre-Chorus -> "
        "Chorus -> Bridge-or-Tag -> Final Chorus. 69% of 2024 #1s use three choruses "
        "(up from 31% in 2022). Two-chorus is now minority; a missing third chorus reads "
        "as unfinished. Hit Songs Deconstructed 2024 #1 Focus Report."
    ),
    "modern_30_second_test": (
        "TikTok 30-second test: by 0:15 the listener must hear a sticky fragment, by 0:30 "
        "the hook must have landed at least once. Modern openings lead with a pre-hook "
        "fragment or jump straight into chorus. Average time-to-first-chorus is 33-45s on "
        "2024 #1s; only 25% delay past 1:00. Berklee 2024 hit-DNA research."
    ),
    "modern_displaced_downbeat": (
        "Modern hook melodies start on beat 2 or the 'and' of beat 1, not the downbeat. "
        "Espresso chorus delays every phrase to beat 2; verses to beat 4. That off-kilter "
        "syncopated pull is the earworm. Don't land the most important syllable on the "
        "downbeat — push it back half a beat."
    ),
    "modern_micro_hook_stacking": (
        "Stack micro-hooks: a pre-chorus lift (one ascending line), a post-chorus tag "
        "(repeated nano-phrase), AND a chantable nano-hook (1-3 syllable shout). 'Flowers' "
        "stacks: 'up-top/down-below' chorus + 'I can love me better' post-chorus + bass riff "
        "hook. 'Not Like Us' stacks: hook + 'they not like us' chant + Mustard's hyphy bass."
    ),
    "modern_anthem_hook": (
        "The chorus core must be a 4-syllable shout a stadium of strangers can chant on "
        "first listen. 'They not like us', 'BAND for BAND', 'that's that me espresso', "
        "'I can buy myself flowers'. Test by saying it out loud once — if it feels "
        "performable to a crowd, it's an anthem hook. If it requires explanation, rewrite."
    ),
    "modern_rant_bridge": (
        "If you write a bridge in 2024-2026, write the 'rant bridge' (Antonoff/Swift). "
        "Stream-of-consciousness, conversational diction, intrusive thoughts blended with "
        "metaphor, end on a shouted single-line thesis. Cruel Summer has TWO rant-bridges. "
        "Birds of a Feather has none — pick one approach, commit. Bridge is optional in "
        "modern pop, never mandatory."
    ),
    "modern_rap_verse_count": (
        "Modern rap verse count is 12 bars more often than 16 (2024-2026 mainstream). "
        "Reserve 16+ bars for storytelling tracks (Kendrick concept tracks, Nas-style "
        "narratives). For radio rap 'Not Like Us' / 'BAND4BAND' / Sexyy Red / Glorilla, "
        "12 staccato bars locked on snare beats both length and impact."
    ),
    "modern_concrete_proper_nouns": (
        "Force one concrete proper noun per verse: a brand, a place, a time, a person, an "
        "object. 'Espresso' has Mountain Dew, jet-lag, Dior. 'Texas Hold 'Em' has Texas, "
        "whiskey, dance hall. 'Not Like Us' name-drops 3 specific people. AI lyric tells "
        "ALL fail this test — they fear specificity. Force it on the first draft."
    ),
    "modern_metric_overflow": (
        "Allow ONE deliberate metric overflow per song — a line longer than the others, "
        "conversational, anti-symmetric. Antonoff/Swift 'rant' technique. 'Cruel Summer' "
        "bridge breaks meter on 'Devils roll the dice, angels roll their eyes'. Symmetric "
        "syllable counts on every line read as AI; one breath-overflow signals human."
    ),
    "modern_v2_must_escalate": (
        "Verse 2 NEVER paraphrases verse 1. V2 must add: a new scene, a new witness, a "
        "time jump, OR a reversal. 'Cruel Summer' V2 zooms to drunk-in-back-of-car detail. "
        "'Birds of a Feather' V2 is where Eilish/Finneas land 'our best writing lives "
        "here'. AI tools love restating V1 in V2 with synonyms — reject that draft."
    ),
    "modern_genre_blend_anchor": (
        "Genre is post-genre in 2024-2026 — pop+country, pop+drill, pop+afrobeats, "
        "K-pop+retro-rock all chart. But cross-genre still demands ONE production anchor. "
        "Pick a lane (sliding-808 drill, retro-disco, breathy-bedroom ballad, "
        "acoustic-percussion country-pop, rage synth-lead) and commit; the cross-pollination "
        "lives in the topline, not the production. Texas Hold 'Em = country anchor + R&B "
        "topline. APT. = retro pop-rock anchor + K-pop topline."
    ),
    # PER-MODERN-ARTIST signatures (2023-2026 era) to balance the classic ones above.
    "sabrina_carpenter_signature": (
        "Humor + sex-positive double-entendre layered over yacht-rock/disco-revival "
        "production. Title-drop with vowel-lock. Uses brand drops + personal-life specificity "
        "(jet-lag, Dior, ex-flames) over vague emotion. Confidence + vulnerability in same "
        "verse — never one note."
    ),
    "kendrick_2024_signature": (
        "Diss-track-as-hit anatomy: hook first (anthem-shout), 12-bar staccato verses "
        "locked to snare, name-checks of 3+ specific people, double-entendre 'A-Minor' "
        "wordplay, West Coast Mustard production. No bridge needed. Hook ×8 to outro. "
        "'Not Like Us' template."
    ),
    "billie_eilish_finneas_signature": (
        "Whisper-aesthetic + bass-heavy minimalism. 4-chord palette across the whole song "
        "(D / Bm / Em / A on Birds of a Feather). Conversational diction, one idiom-flip "
        "in the title, V2 is where the writers land their thesis. Close-mic'd vocal, "
        "introspection-themed (44% of 2025 top 10s are introspection). Cinematic strings "
        "comeback for emotional climax."
    ),
    "central_cee_drill_signature": (
        "UK drill template: 140 BPM half-time, sliding 808s gliding scale-degree-to-scale-"
        "degree, chopped acoustic-guitar/operatic-vocal loops. Auto-tune retune ~25ms "
        "for melodic-rap hybrid. Sing-rap blend, money imagery, 4-syllable repeat shout "
        "as hook ('BAND-for-BAND', 'Doja-Doja-Doja')."
    ),
    "morgan_wallen_country_signature": (
        "Country crossover: acoustic-percussion forward (37% of 2024 #1s use it), "
        "storytelling first (specific places, names, moments), genre-blur with "
        "hip-hop cadence on verses + traditional country topline on chorus. "
        "'Last Night' / 'I Had Some Help' / 'A Bar Song (Tipsy)' template."
    ),
}

# LYRIC_ANTI_PATTERNS lists the cliché phrases and structural failure modes the
# writer must avoid. Substring matches against the cliché_image_bank trigger a
# repair pass. The form_antipatterns are LLM-side reminders, not regex matches.
LYRIC_ANTI_PATTERNS: dict[str, list[str]] = {
    "cliche_image_bank": [
        # Classic AI-slop image bank (Audiocipher / AISongFix research)
        "neon dreams", "fire inside", "shattered dreams", "endless night",
        "empty streets", "city lights", "embers", "whispers in the dark",
        "silhouettes", "echoes of", "we rise", "let it burn",
        "chasing the night", "broken heart", "rising from the ashes",
        "stars aligned", "fade away", "into the void", "burning bright",
        "stolen kisses", "tears like rain", "frozen in time",
        "dancing in the dark", "running through my mind",
        # Modern AI-slop additions (Nick Cave Red Hand Files critique + 2024-2026 ChatGPT lyric tells)
        "soul on fire", "heart of gold", "light of my life",
        "wings to fly", "mountains to climb", "rivers to cross",
        "beautiful disaster", "love is the answer", "find my way",
        "out of the darkness", "into the light", "feel the rhythm",
        "ride or die", "until the end", "forever and always",
        "dancing with destiny",
    ],
    "telling_not_showing": [
        "I feel sad", "my heart is broken", "I'm in pain", "we're all in pain",
        "this is sad", "this is hard", "we suffer", "I'm hurting inside",
        "I'm so happy", "I'm so in love", "I feel alive",
        "this is real", "this is everything",
    ],
    "generic_pov": [
        "we all", "everyone feels", "the world is", "the people need",
        "society today", "this generation", "the youth of today",
        "humanity is", "mankind",
    ],
    "explanation_lines": [
        "in other words", "what I mean is", "to be clear",
        "let me explain", "in summary", "basically",
        "what I'm trying to say",
    ],
    # Nick Cave's GPT-4 critique flagged the polar-binary "I am X / I am Y" trick
    # as the unmistakable AI tell. ChatGPT loves it. Hits never use it.
    "polar_binary_reversals": [
        "i am the saint i am the sinner", "i am the angel i am the demon",
        "i am the light i am the dark", "i am the king i am the slave",
        "i am the hunter i am the prey", "i'm the fire i'm the flame",
        "i'm the rose i'm the thorn",
    ],
    "form_antipatterns": [
        "Every line same end-rhyme scheme — puts listeners to sleep.",
        "Padding syllables ('yeah', 'you know') to hit a bar count.",
        "Punchlines that explain themselves.",
        "Choruses that just repeat the verse's idea instead of distilling a new memorable phrase.",
        "Bridges that don't shift perspective (verse 3 in disguise).",
        "Flowery 8th-grade-poetry register without concrete imagery.",
        "Generic POV ('we', 'the world') with no named, situated speaker.",
        "Sterile syllable counts that don't sing — every line same length, no breath-overflow.",
        "Over-perfect rhymes that read like greeting cards.",
        "Ad-libs sprinkled every line — drains them of meaning.",
        # Modern (2024-2026) anti-patterns from chart research
        "Polar-binary 'I am X / I am Y' reversal — Nick Cave called this the AI tell.",
        "No proper nouns at all — AI fears specifics. Force one brand/place/name per verse.",
        "Theme stated but not embodied — cut the line that names the emotion, keep the line that proves it.",
        "Hook buried past 0:30 — 2024-2026 hits land the hook by 0:15-0:30 (TikTok test).",
        "Verse 2 paraphrases verse 1 — V2 must add new scene, witness, or reversal.",
        "No contradictions — real songs argue with themselves (confidence + jet-lag in same verse).",
        "Title-drop missing or buried — modern hits drop the title in chorus line 1, vowel-locked.",
        "Two-chorus structure — 2024 #1s use three choruses (69%, up from 31% in 2022).",
        "Symmetric meter throughout — allow ONE deliberate metric overflow per song (Antonoff/Swift rant).",
    ],
}

RAP_MODE_COOKBOOK: dict[str, str] = {
    "ad-libs / background vocals": (
        "Write ad-libs in parentheses on the same line as the main lyric: "
        "'I came up from the bottom (yeah!) / now they want a feature (uh!)'. "
        "Common ad-libs: (yeah), (uh), (huh), (skrrt), (woo), (let's go), (alright), (come on)."
    ),
    "rap section structure": (
        "Use [Verse - rap] for rapped verses, [Hook] or [Hook/Chorus] for the main repeating hook, "
        "[Chorus - rap] only when the chorus is itself rapped. Bridges in rap usually become [Bridge - spoken] "
        "or [Bridge - melodic rap]. Place 2-3 hook passes total per song."
    ),
    "rap line length": (
        "Rap lines run 6-14 syllables; keep syllable count consistent line-to-line for cadence. "
        "Internal rhyme inside the lyric text drives flow; do not write 'internal rhyme' as a tag."
    ),
    "shouted intensity": (
        "ALL CAPS = shouted. Use for hook accents like 'WE RUN THIS' or one-word chants. "
        "Do not capitalise whole verses."
    ),
    "language flag": (
        "Caption-side vocal cue (Rap, Trap Flow, Spoken Word, Melodic Rap) PLUS a section tag like [Verse - rap] "
        "is the most reliable way to switch ACE-Step into rap mode. Sung captions with [Verse - rap] tags "
        "produce inconsistent results."
    ),
    "rap caption stack template": (
        "Stack 6-9 caption tags in this order: subgenre (boom bap / G-funk / drill / trap / cloud rap), "
        "era (90s / 2010s / modern), drum signature (head-nod groove / trap bounce / drill bounce), "
        "low end (808 bass / heavy synthesizer bassline / sub-bass), melody (soul sample chops / talkbox lead / dark synth lead), "
        "vocal (male rap vocal / melodic rap vocal / mumble rap), texture (vinyl texture / glossy mix / dusty mix), "
        "energy (gritty / triumphant / menacing). Do not include BPM, key, or song titles in caption."
    ),
}

# Worked examples — concrete request -> caption + lyric structure pairs.
# These exist because LLMs follow patterns far better than rules. Each example
# shows the exact shape ACE-Step responds to, including ad-libs in parentheses,
# section tag modifiers, and how a producer-name request collapses into the
# Producer-Format Cookbook stack rather than the producer's literal name.
WORKED_EXAMPLES: list[dict[str, str]] = [
    {
        "request": "Make me a Dr. Dre G-funk banger about coming up from nothing",
        "caption": (
            "G-funk, West Coast hip hop, talkbox lead, heavy synthesizer bassline, "
            "laid-back groove, polished mix, deep low end, syncopated kick, smooth high hat, "
            "head-nod groove, male rap vocal, summer banger polish"
        ),
        "lyrics": (
            "[Intro - talkbox]\n"
            "From the bottom of the block to the penthouse view\n"
            "(yeah, yeah, alright)\n"
            "\n"
            "[Verse - rap]\n"
            "I came up where the streetlights flicker through the screen door (uh)\n"
            "Mama working doubles, I was sleeping on the floor (yeah)\n"
            "Now the candy paint glide on a Sunday afternoon (skrrt)\n"
            "Talkbox singing low, I'm conducting my own tune\n"
            "\n"
            "[Hook - sung]\n"
            "We came up, we came up (we came up)\n"
            "Top down on the West side, we came up\n"
            "We came up, we came up (we came up)\n"
            "Whole hood see the shine 'cause we came up\n"
            "\n"
            "[Verse - rap]\n"
            "Used to dream about the keys to a six-fo' Impala (let's go)\n"
            "Now I'm parking in the lot where the suit-and-tie holler\n"
            "Bassline kissing concrete, hi-hat skipping in the smoke\n"
            "Same block I came from, same block I provoke\n"
            "\n"
            "[Hook - sung]\n"
            "We came up, we came up (we came up)\n"
            "Top down on the West side, we came up\n"
            "\n"
            "[Outro - talkbox]\n"
            "From the bottom (yeah)\n"
            "From the bottom (alright)"
        ),
        "notes": (
            "Producer name 'Dr. Dre' does NOT appear in caption. The cookbook stack carries the sound. "
            "Ad-libs are in (parens) on the same line. [Intro - talkbox] and [Outro - talkbox] echo the cookbook's talkbox cue."
        ),
    },
    {
        "request": "Write something with No I.D. boom-bap soul flip energy, conscious lyrics",
        "caption": (
            "boom bap, soul sample chops, dusty drums, jazzy chord loop, vinyl texture, "
            "warm analog mix, head-nod groove, 90s boom bap, NYC east coast warmth, "
            "muted piano sample, soft kick, tight snare, male rap vocal, lyrical rap"
        ),
        "lyrics": (
            "[Intro]\n"
            "Vinyl crackle, muted keys (check it)\n"
            "\n"
            "[Verse - rap]\n"
            "Pulled the curtain back on what they sold us as a dream\n"
            "Soul flip on the loop, I can hear it through the seam\n"
            "Pop coloring the lie that we drink up like a stream\n"
            "I'm the question in the room, I'm the elephant unseen (uh)\n"
            "\n"
            "[Hook]\n"
            "Wake up, wake up, the record's still spinning\n"
            "Wake up, wake up, the truth in the beginning\n"
            "\n"
            "[Verse - rap]\n"
            "Brother on the corner with a story in his eyes\n"
            "Sister in the office with a lifetime in disguise\n"
            "Same beat keep playing 'til we recognise the lies (yeah)\n"
            "Same kick, same snare, same patient little rise\n"
            "\n"
            "[Bridge - spoken]\n"
            "It's a long road. Keep your head up.\n"
            "\n"
            "[Hook]\n"
            "Wake up, wake up, the record's still spinning\n"
            "\n"
            "[Outro]\n"
            "(wake up, wake up)"
        ),
        "notes": (
            "No I.D. is not in caption. Stack is the cookbook's boom-bap entry. "
            "[Bridge - spoken] anchors the conscious-rap pause. Hook line is short and repeats verbatim per Tutorial.md hook rule."
        ),
    },
    {
        "request": "Metro Boomin dark trap with a melodic hook, late-night vibe",
        "caption": (
            "modern trap, dark atmospheric, 808 bass, trap hi-hats, sparse melody, "
            "ominous synth lead, gritty, hard-hitting drums, half-time drums, "
            "hi-hat rolls, 808 swells, cinematic tension, melodic rap vocal, glossy mix"
        ),
        "lyrics": (
            "[Intro]\n"
            "(uh, uh) (Metro on the night flight, lights low)\n"
            "\n"
            "[Verse - rap]\n"
            "City sleeping but the 808 awake (yeah)\n"
            "Hi-hat dancing on the snare like a snake (skrrt)\n"
            "I been counting all my brothers and the moves they make (uh)\n"
            "Half the room a mirror and the other half a fake\n"
            "\n"
            "[Hook]\n"
            "Late night, lights low, 808 talk slow (slow)\n"
            "Late night, lights low, only the real know (real)\n"
            "Late night, lights low, 808 talk slow (slow)\n"
            "Late night, lights low, only the real know\n"
            "\n"
            "[Verse - rap]\n"
            "I been on the highway with my dreams in the trunk (woo)\n"
            "808 keep walking like the city in a funk\n"
            "Cold side of the moon when the morning come, hunh\n"
            "Tell 'em hold the silence, leave the rest of it to drum\n"
            "\n"
            "[Bridge - melodic rap]\n"
            "Lights low, lights low, 808 in slow motion\n"
            "\n"
            "[Hook]\n"
            "Late night, lights low, 808 talk slow (slow)\n"
            "Late night, lights low, only the real know"
        ),
        "notes": (
            "Metro Boomin not in caption. 'Metro on the night flight' is allowed inside lyrics as flavour, not as caption tag. "
            "Half-time drums + 808 swells + hi-hat rolls captured via cookbook stack."
        ),
    },
    # MODERN (2024-2026 chart) worked examples — Espresso-style retro disco-pop,
    # Not Like Us-style West Coast diss, Birds of a Feather-style ballad. These
    # show the modern hit anatomy: title-drop with vowel-lock, hook-first opens,
    # 12-bar verses, displaced downbeat, concrete proper nouns.
    {
        "request": "Make me a Sabrina Carpenter Espresso-style flirty retro disco-pop banger",
        "caption": (
            "retro disco-pop, yacht-rock revival, vintage disco bass walk, syncopated guitar chops, "
            "electric piano stabs, tight punchy kick, snare on 2 and 4, hand-claps stacked, "
            "displaced-downbeat melody, glossy commercial mix, female pop vocal, head-nod groove, "
            "2020s pop polish, sex-positive humor"
        ),
        "lyrics": (
            "[Intro]\n"
            "(uh-huh)\n"
            "\n"
            "[Hook]\n"
            "That's that me, espresso\n"
            "Mountain Dew on the dresser, I'm a problem you can't measure\n"
            "Move it like espresso, move it like espresso\n"
            "Move it like espresso, oh\n"
            "\n"
            "[Verse 1]\n"
            "Met him at the after-party, jet-lag in my Dior heels\n"
            "Asked if I'm exclusive, told him 'baby, I'm a deal'\n"
            "He been textin' twenty-twenty, I been readin' twenty-three\n"
            "Boy you're cute but I'm the brand, you're the marketing\n"
            "\n"
            "[Pre-Chorus]\n"
            "(say it back, say it back)\n"
            "He said baby say it back\n"
            "(say it back, say it back)\n"
            "\n"
            "[Hook]\n"
            "That's that me, espresso\n"
            "Mountain Dew on the dresser, I'm a problem you can't measure\n"
            "Move it like espresso, move it like espresso\n"
            "Move it like espresso, oh\n"
            "\n"
            "[Verse 2]\n"
            "Sunday morning chapel, Monday morning ICU\n"
            "Tuesday night I'm walkin' barefoot through a CVS for two\n"
            "He keeps askin' 'where you headed', I keep sayin' 'after-shift'\n"
            "Boy I'm bigger than the bar tab, I'm the bigger plot to flip\n"
            "\n"
            "[Pre-Chorus]\n"
            "(say it back, say it back)\n"
            "He said baby say it back\n"
            "\n"
            "[Hook]\n"
            "That's that me, espresso\n"
            "Move it like espresso, oh\n"
            "\n"
            "[Outro]\n"
            "(that's that me)"
        ),
        "notes": (
            "Title-drop on chorus line 1 ('that's that me espresso') with vowel-lock /oʊ/. "
            "Three-chorus framework. Concrete proper nouns per verse: Mountain Dew, Dior, CVS. "
            "V2 escalates (chapel -> ICU -> barefoot CVS). Pre-chorus 'say it back' is the nano-hook tag. "
            "Displaced-downbeat melody. Producer name (Carter Lang / Julian Bunetta) NOT in caption — cookbook stack carries it."
        ),
    },
    {
        "request": "Write a Kendrick Not Like Us-style West Coast diss anthem",
        "caption": (
            "ratchet West Coast hip hop, hyphy bass, finger-snap percussion, "
            "sliding sub-bass, violin sample, sparse drums, half-time snare, "
            "4/4 hand claps, mono 808, anthem-shout hook, polished commercial mix, "
            "male rap vocal, head-nod groove, 2020s rap, Compton 2024 sound"
        ),
        "lyrics": (
            "[Intro]\n"
            "(yeah, yeah)\n"
            "\n"
            "[Hook]\n"
            "They not like us\n"
            "They not like us\n"
            "They not like us\n"
            "They not like us\n"
            "\n"
            "[Verse - rap]\n"
            "Boy I see you on the timeline askin' how it's done (yeah)\n"
            "I been in the lab while you been chasin' clout for fun (uh)\n"
            "DJ Mustard on the violin, the city know the sound (Mustard)\n"
            "Concrete know my footprint, Compton know who run the town\n"
            "You said you was top three, top five, top ten (psh)\n"
            "I been countin' twenty Grammys while you countin' Instagram friends\n"
            "Stage at the Coliseum lit, the choir in the booth (uh)\n"
            "Half a million in the front row, every line a proof\n"
            "\n"
            "[Hook]\n"
            "They not like us\n"
            "They not like us\n"
            "They not like us\n"
            "They not like us\n"
            "\n"
            "[Verse - rap]\n"
            "Boy you doin' interviews to clean up what you said (yeah)\n"
            "I been doin' albums every line a body bag instead\n"
            "You was at the Drake show, I was at the Forum sold (skrrt)\n"
            "Sixty thousand chant the chorus louder than the radio\n"
            "Tell yo' team to call my team, my team don't make a deal (uh)\n"
            "Crown ain't been moved since two-thousand-twelve, that's how it feel\n"
            "\n"
            "[Hook]\n"
            "They not like us\n"
            "They not like us\n"
            "They not like us\n"
            "They not like us\n"
            "\n"
            "[Outro]\n"
            "(West side, West side)\n"
            "(they not like us)"
        ),
        "notes": (
            "Hook FIRST (anthem 4-syllable shout 'they not like us'). 12-bar verses (modern norm) "
            "locked to snare. Mustard self-reference allowed inside lyrics ('DJ Mustard on the violin') "
            "as flavour, NEVER in caption. Concrete proper nouns: Coliseum, Compton, Drake, Forum, "
            "Grammys. No bridge — straight Hook x4 to outro. Diss-track-as-hit anatomy."
        ),
    },
    {
        "request": "Write a Billie Eilish Birds of a Feather-style intimate alt-pop ballad",
        "caption": (
            "bedroom pop, alt pop, breathy whisper-aesthetic, close-mic vocal, "
            "minimal 4-chord palette, sub-bass-heavy minimalism, soft Rhodes piano, "
            "cinematic strings, introspective mood, conversational diction, ASMR vocal texture, "
            "plate reverb on snare, female alt-pop vocal, slow tempo around 100 BPM, "
            "2020s alt-pop polish"
        ),
        "lyrics": (
            "[Intro]\n"
            "(close, close)\n"
            "\n"
            "[Verse 1]\n"
            "I want you to stay (yeah)\n"
            "Till my second-floor balcony lights are gone\n"
            "Till the upstairs neighbor's dog stops barkin'\n"
            "Till the morning kettle's whistle quiet on\n"
            "\n"
            "[Pre-Chorus]\n"
            "And I don't know what's after, but if you ask me\n"
            "I'd say it's you, just you, just you\n"
            "\n"
            "[Chorus]\n"
            "Birds of a feather, we should stick together I know\n"
            "Said I'd never think I wasn't better alone\n"
            "Can't change the weather, might not be forever\n"
            "But if it's forever, it's even better, oh\n"
            "\n"
            "[Verse 2]\n"
            "I been the kind of girl who flies at the first turn of the key (mm)\n"
            "But you been workin' on me quiet like nobody but me see\n"
            "Got me writin' in the notebook, like our best stuff lives in here\n"
            "Got me thinkin' 'bout the long way 'round and tryin' to disappear\n"
            "\n"
            "[Pre-Chorus]\n"
            "And I don't know what's after, but if you ask me\n"
            "I'd say it's you, just you, just you\n"
            "\n"
            "[Chorus]\n"
            "Birds of a feather, we should stick together I know\n"
            "Said I'd never think I wasn't better alone\n"
            "Can't change the weather, might not be forever\n"
            "But if it's forever, it's even better, oh\n"
            "\n"
            "[Outro]\n"
            "(birds of a feather)\n"
            "(stick together)"
        ),
        "notes": (
            "Whisper-aesthetic close-mic. 4-chord palette (D / Bm / Em / A) across whole song. "
            "Idiom-flip in title ('birds of a feather' reframed as romantic commitment). "
            "V2 lands the writers' thesis ('our best stuff lives in here' = Eilish/Finneas songwriting "
            "moment from interviews). Concrete sensory details (second-floor balcony, neighbor's dog, "
            "morning kettle, notebook). Producer (Finneas) NOT in caption — bedroom-pop stack carries it. "
            "No bridge — modern alt-pop ballad doesn't need it."
        ),
    },
]

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
    "flat drums",
    "harsh high end",
    "overcompressed",
    "boring arrangement",
    "clichéd AI lyrics",
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
                ("Verse 1", 12, 62),
                ("Chorus", 62, 90),
                ("Verse 2", 90, 140),
                ("Second Chorus", 140, 164),
                ("Bridge", 164, max(210, dur - 30)),
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


MUSIC_MODES_FOR_RAP_COOKBOOK = {"simple", "custom", "song", "album", "news", "improve"}


def _format_tag_taxonomy_block() -> str:
    from songwriting_toolkit import LYRIC_META_TAGS, TAG_TAXONOMY

    lines: list[str] = ["## ACE-Step Tag Library (full reference)"]
    lines.append("Caption tag taxonomy — pick a compact stack that covers genre, mood, instruments, vocal, rhythm, era, production, structure energy:")
    for dimension, tags in TAG_TAXONOMY.items():
        lines.append(f"- **{dimension}**: {', '.join(tags)}")
    lines.append("")
    lines.append("Lyric meta tags — square brackets only; one-section-per-block; modifier syntax `[Section - modifier]` with one dash and ONE modifier max:")
    for category, tags in LYRIC_META_TAGS.items():
        lines.append(f"- **{category}**: {', '.join(tags)}")
    return "\n".join(lines)


def _format_authoring_rules_block() -> str:
    lines = ["## ACE-Step Authoring Rules (must follow)"]
    for index, rule in enumerate(ACE_STEP_AUTHORING_RULES, 1):
        lines.append(f"{index}. {rule}")
    return "\n".join(lines)


def _format_section_template_block() -> str:
    lines = ["## ACE-Step Section Templates (reference structures)"]
    for family, tags in SECTION_TAGS.items():
        lines.append(f"- **{family}**: {' -> '.join(tags)}")
    return "\n".join(lines)


def _format_producer_cookbook_block() -> str:
    lines = [
        "## Producer-Format Cookbook",
        "ACE-Step does NOT recognise producer names directly. When the user asks for a producer's sound, "
        "do NOT put the name in the caption. Translate the request into the matching tag stack below "
        "and stack 6-9 of those tags in the caption.",
    ]
    for label, stack in PRODUCER_FORMAT_COOKBOOK.items():
        lines.append(f"- **{label}**: {stack}")
    return "\n".join(lines)


def _format_rap_cookbook_block() -> str:
    lines = ["## Rap-Mode Cookbook"]
    for label, body in RAP_MODE_COOKBOOK.items():
        lines.append(f"- **{label}**: {body}")
    return "\n".join(lines)


def _format_worked_examples_block() -> str:
    lines = [
        "## Worked Examples (request -> caption + lyrics shape)",
        "Pattern-match these when the user asks for a producer-format or rap-mode song. "
        "Notice the producer name NEVER appears in caption, ad-libs sit in (parens) inside lyric lines, "
        "and section tags follow the single-dash modifier rule.",
    ]
    for index, example in enumerate(WORKED_EXAMPLES, 1):
        lines.append("")
        lines.append(f"### Example {index}: {example['request']}")
        lines.append("**caption** (no producer name, no BPM):")
        lines.append(f"`{example['caption']}`")
        lines.append("**lyrics** (section tags + ad-libs in parens):")
        lines.append("```")
        lines.append(example["lyrics"])
        lines.append("```")
        lines.append(f"_Notes: {example['notes']}_")
    return "\n".join(lines)


def _format_songwriter_cookbook_block() -> str:
    lines = [
        "## Songwriter Craft Cookbook",
        "Apply these to every verse, hook, and bridge. They distill craft moves from "
        "Eminem, 2Pac, Kendrick Lamar, Nas, MF DOOM, plus Pat Pattison-style songwriting "
        "pedagogy. Reach for these BEFORE writing — generic phrasing is the result of "
        "skipping this layer.",
    ]
    for label, body in SONGWRITER_CRAFT_COOKBOOK.items():
        lines.append(f"- **{label.replace('_', ' ')}**: {body}")
    return "\n".join(lines)


def _format_anti_patterns_block() -> str:
    lines = [
        "## Lyric Anti-Patterns (forbid; rewrite if drafted)",
        "If your draft contains any of the cliché phrases below, rewrite the line with "
        "concrete sensory detail before output. Form anti-patterns are reminders for the "
        "shape of your writing.",
        "",
        "### Cliché image bank — never use these phrases:",
    ]
    lines.append(", ".join(LYRIC_ANTI_PATTERNS["cliche_image_bank"]))
    lines.append("")
    lines.append("### Telling-not-showing labels — never use these phrases:")
    lines.append(", ".join(LYRIC_ANTI_PATTERNS["telling_not_showing"]))
    lines.append("")
    lines.append("### Generic POV — never use these phrases:")
    lines.append(", ".join(LYRIC_ANTI_PATTERNS["generic_pov"]))
    lines.append("")
    lines.append("### Explanation lines — never use these phrases:")
    lines.append(", ".join(LYRIC_ANTI_PATTERNS["explanation_lines"]))
    lines.append("")
    lines.append("### Form anti-patterns — restructure if drafted:")
    for item in LYRIC_ANTI_PATTERNS["form_antipatterns"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def prompt_kit_system_block(mode: str = "custom") -> str:
    contract = ", ".join(PROMPT_KIT_OUTPUT_CONTRACT_FIELDS)
    metadata = ", ".join(PROMPT_KIT_METADATA_FIELDS)
    languages = ", ".join(LANGUAGE_PRESETS.keys())
    genres = ", ".join(GENRE_MODULES.keys())
    base = (
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
    if str(mode or "").strip().lower() in {"image", "video", "trainer"}:
        return base
    blocks = [
        base,
        _format_authoring_rules_block(),
        _format_tag_taxonomy_block(),
        _format_section_template_block(),
    ]
    if str(mode or "").strip().lower() in MUSIC_MODES_FOR_RAP_COOKBOOK:
        blocks.append(_format_producer_cookbook_block())
        blocks.append(_format_rap_cookbook_block())
        blocks.append(_format_songwriter_cookbook_block())
        blocks.append(_format_anti_patterns_block())
        blocks.append(_format_worked_examples_block())
    return "\n\n".join(blocks)
