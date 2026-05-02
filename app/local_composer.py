from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from local_llm import (
    chat_completion,
    model_catalog,
    normalize_provider,
    planner_llm_options_for_provider,
    planner_llm_settings_from_payload,
    provider_label,
)
from songwriting_toolkit import derive_artist_name, normalize_artist_name


SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "artist_name": {"type": "string"},
        "title": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 6,
        },
        "song_intent": {
            "type": "object",
            "properties": {
                "genre_family": {"type": "string"},
                "subgenre": {"type": "string"},
                "mood": {"type": "string"},
                "energy": {"type": "string"},
                "vocal_type": {"type": "string"},
                "drum_groove": {"type": "string"},
                "bass_low_end": {"type": "string"},
                "melodic_identity": {"type": "string"},
                "texture_space": {"type": "string"},
                "mix_master": {"type": "string"},
                "custom_tags": {"type": "array", "items": {"type": "string"}},
                "caption": {"type": "string"},
            },
        },
        "caption": {"type": "string"},
        "quality_profile": {"type": "string"},
        "song_model": {"type": "string"},
        "audio_format": {"type": "string"},
        "inference_steps": {"type": "integer"},
        "guidance_scale": {"type": "number"},
        "shift": {"type": "number"},
        "infer_method": {"type": "string"},
        "sampler_mode": {"type": "string"},
        "bpm": {"type": "integer"},
        "key_scale": {"type": "string"},
        "time_signature": {"type": "integer"},
        "language": {"type": "string"},
        "lyrics": {"type": "string"},
    },
    "required": ["artist_name", "title", "tags", "bpm", "key_scale", "time_signature", "language", "lyrics"],
}

SYSTEM_PROMPT = """You are an expert songwriter and music producer crafting input for ACE-Step, a state-of-the-art AI music generator. Your job is to take any description—even a vague one—and expand it into a rich, professional-grade song specification that will produce the highest quality music.

Reply with exactly one JSON object and nothing else.

## Tag rules (the `tags` array)
Tags are the MOST important factor for music quality. Follow this layered format:
1. Primary genre + optional era (e.g. "indie folk", "80s synth pop", "modern R&B")
2. 2-3 key instruments (e.g. "acoustic guitar", "synth pads", "808 drums")
3. Mood / atmosphere (e.g. "melancholic", "uplifting", "dreamy", "energetic")
4. Vocal character if not instrumental (e.g. "female breathy vocal", "male raspy vocal", "powerful belt")
5. Production style (e.g. "high-fidelity", "lo-fi", "warm analog", "crisp modern mix")
- Use 4 to 6 focused, complementary tags. Never mix contradictory styles (e.g. "ambient" + "hardcore metal"). For genre fusion, use primary + influence format: "jazz, electronic elements".
- Include tempo descriptor in tags when relevant (e.g. "slow tempo", "driving rhythm", "groovy").

## Song Intent Builder fields
Also return `song_intent`, the structured UI menu contract AceJAM shows to the user:
- genre_family: one of pop, rap, rnb, rock, edm, latin, country, cinematic, jazz, instrumental, custom
- subgenre, mood, energy, vocal_type, drum_groove, bass_low_end, melodic_identity, texture_space, mix_master
- custom_tags: concise extra descriptors
- caption: one concrete ACE-Step sonic portrait combining genre, drums/groove, bass, melodic identity, vocal delivery, arrangement movement, texture/space and mix/master.
The top-level `caption` must match song_intent.caption and be editable by the user.
Default quality: quality_profile "chart_master", song_model "acestep-v15-xl-sft", inference_steps 64, guidance_scale 8, shift 3, infer_method "ode", sampler_mode "heun", audio_format "wav32".

Available instruments for tags (pick 2-3 that fit the genre):
Keys: piano, Rhodes, organ, electric piano, grand piano, clavinet, celesta
Guitar: acoustic guitar, electric guitar, clean guitar, distorted guitar, nylon guitar, fingerpicked guitar, slide guitar, power chords
Bass: bass guitar, upright bass, synth bass, 808 bass, sub-bass, slap bass, fretless bass
Drums: drums, trap hi-hats, 808 kick, snare, claps, shaker, tambourine, congas, timpani, punchy snare, gated drums, breakbeat
Synth: synth pads, arpeggiated synth, analog synth, lead synth, warm synth, dark synths, evolving drones
Strings: strings, violin, viola, cello, orchestral strings, pulsing strings, pizzicato
Brass/Wind: brass, trumpet, trombone, saxophone, alto sax, flute, clarinet
World: sitar, tabla, koto, djembe, steel drums, accordion, banjo, mandolin, ukulele, harmonica
Production: vinyl texture, tape hiss, wide stereo mix, warm mix, crisp mix, lo-fi texture

## BPM rules
- `bpm` must be a plausible tempo integer (30-300).
- Slow ambient/cinematic: 30-60. Ballads: 60-80. Mid-tempo pop/R&B: 90-120. Upbeat dance/rock: 120-150. Fast punk/drum&bass: 150-200. Extreme speedcore: 200-300.
- Match BPM to the mood and genre implied by the description.

## Key and scale rules
- `key_scale` must be a valid musical key: a note letter (A-G), optional sharp (#) or flat (b), followed by "major" or "minor".
- Examples: "C major", "Ab minor", "F# major", "Bb minor", "D major", "Em".
- Match the key to the genre and mood:
  - Major keys for uplifting, happy, bright music (C major, G major, D major, F major)
  - Minor keys for sad, dark, intense, emotional music (Am, Em, Dm, Cm)
- Common genre associations: C major (neutral pop), Am (melancholic ballad), G major (folk/country), Em (rock/alternative), Bb major (jazz/R&B/soul), F# minor (electronic/synthwave), Eb major (epic/cinematic).

## Time signature rules
- `time_signature` must be one of: 2, 3, 4, 6.
- 4 is standard for most pop, rock, hip-hop, electronic, R&B.
- 3 is for waltzes, some ballads, folk songs.
- 6 is for compound time (6/8 folk, blues, some ballads).
- 2 is for marches, polka.
- Default to 4 unless the genre or style specifically calls for another.

## Lyrics rules
ACE-Step sings approximately 2-3 words per second. Lyrics act as a temporal script controlling how music unfolds.
- Each line should be 4-8 words for natural singability. Avoid tongue twisters, complex vocabulary, or overly long sentences.
- Every section marker must be followed by actual sung lyric lines (except instrumental markers).
- Mood and emotion of lyrics MUST match the tags.

## Lyric craft — WRITE LIKE A PROFESSIONAL SONGWRITER, NOT AN AI

SHOW DON'T TELL: Never state emotions directly. Instead of "I am sad", write "The coffee's gone cold on the counter again". Instead of "I love you", write "Your laugh still echoes in the passenger seat". Paint scenes the listener can see, hear, smell, taste, touch.

USE SPECIFIC DETAILS: Replace generic words with precise ones.
- NOT "car" → "rusted Civic" or "midnight Uber"
- NOT "flower" → "dandelion" or "wilted orchid"
- NOT "city" → "the 3 AM bodega on Fifth" or "rain-slick Rotterdam cobblestones"
- NOT "drink" → "lukewarm sake" or "flat champagne"

SENSORY WRITING: Engage ALL five senses, not just sight.
- Sound: "the hum of the highway through thin motel walls"
- Smell: "gasoline and jasmine on the August wind"
- Touch: "your cold ring finger pressed against my neck"
- Taste: "copper on my tongue from biting back the words"

EMOTIONAL TENSION: The best lyrics hold two feelings at once.
- Joy + loss: celebrating what's already gone
- Love + anger: wanting someone you shouldn't
- Freedom + fear: leaving everything behind

IN MEDIAS RES: Start verses in the middle of action, not with setup.
- NOT "I was walking down the street one day" → "Halfway through the red light, you called"

METAPHOR RULES: Use 1-2 strong metaphors per song, not scattered random ones. Pick ONE metaphor world (ocean, fire, astronomy, weather) and stay in it.

BANNED CLICHÉ PHRASES — NEVER use these:
"echoes of", "shattered dreams", "empty streets", "fading light", "stories untold",
"lost in time", "forgotten memories", "endless night", "unseen tears", "whispers in the dark",
"waves crashing", "endless road", "burning bridges", "fading away", "broken chains",
"heart on fire", "dancing in the rain", "light at the end of the tunnel", "stars aligned",
"wings to fly", "ocean of emotions", "paint the sky", "chase the sun", "against all odds".

Available section markers: [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro], [Instrumental], [inst].
Vocal style modifiers: [Verse - rap], [Verse - whispered], [Chorus - anthemic], [Verse - melodic rap], [Bridge - Whispering], [Chorus - Layered vocals], [Intro - Dreamy].
Instrumental sections: [Guitar Solo], [Drum Break].
Energy markers: [building energy], [explosive drop].
Text formatting: UPPERCASE = emphasis/power. (parentheses) = background vocals/harmonies.
Vocal tags: female vocal, male vocal, female breathy vocal, male raspy vocal, male rap vocal, female rap vocal, powerful belt, falsetto.

## CRITICAL: Rap and hip-hop vocal delivery
When the description mentions rap, hip-hop, trap, drill, grime, or similar genres:
- ALWAYS use [Verse - rap], [Chorus - rap] section markers.
- Tags MUST include "male rap vocal" or "female rap vocal" (NOT just "male vocal").
- Lines: 4-6 words, punchy, rhythmic internal rhymes.
- Trap: "trap, 808 bass, dark, aggressive rap vocal, hi-hats".
- Melodic rap: [Verse - melodic rap], "melodic rap, autotune vocal, atmospheric".

## Enhancing vague prompts
If the description is vague (e.g. "a happy song"), you MUST:
- Invent a specific scene, character, or moment (NOT abstract emotions)
- Choose concrete imagery: a place, a time of day, an object, a sensory detail
- Build an emotional arc across sections (tension → release, or joy → doubt)
- Write lyrics that could only belong to THIS song, not any generic song

## Other rules
- `artist_name` must be a short original stage/project name that fits the song persona. Never use a real artist name, even when the user mentions one as a style reference.
- `title` must be a short, catchy, evocative song title that captures the essence of the song.
- `language` must be one of: en, zh, ja, ko, instrumental, unknown.
- If the request is instrumental, set `language` to `instrumental` and `lyrics` to `[Instrumental]`.
- Never return empty sections or placeholder markers such as [END], [LYRICS], [LYRITIC], or repeated labels without lyrics.
- Never wrap the JSON in markdown fences.
"""


STOP_WORDS = {
    "a", "about", "an", "and", "any", "are", "at", "for", "from", "in",
    "into", "is", "it", "its", "lyrics", "music", "of", "on", "song",
    "that", "the", "their", "this", "to", "with",
}


def _lyric_plan(audio_duration: float) -> dict[str, Any]:
    dur = int(audio_duration)
    target_words = int(dur * 1.3)
    word_min = int(dur * 0.9)
    word_max = int(dur * 1.8)
    target_lines = max(4, int(target_words / 5.5))

    if dur <= 45:
        num_verses, structure = 1, "[Verse], [Chorus]"
    elif dur <= 90:
        num_verses, structure = 2, "[Verse], [Chorus], [Verse], [Chorus]"
    elif dur <= 150:
        num_verses, structure = 2, "[Verse], [Chorus], [Verse], [Chorus], [Bridge], [Chorus]"
    elif dur <= 210:
        num_verses, structure = 3, "[Verse], [Chorus], [Verse], [Chorus], [Bridge], [Verse], [Chorus]"
    elif dur <= 270:
        num_verses, structure = 3, "[Intro], [Verse], [Chorus], [Verse], [Chorus], [Bridge], [Verse], [Chorus], [Outro]"
    else:
        num_verses, structure = 4, "[Intro], [Verse], [Chorus], [Verse], [Chorus], [Bridge], [Verse], [Chorus], [Verse], [Chorus], [Outro]"

    return {
        "structure": structure, "num_verses": num_verses,
        "target_words": target_words, "word_range": f"{word_min}-{word_max}",
        "target_lines": target_lines, "line_range": f"{target_lines - 2} to {target_lines + 2}",
        "min_lines": max(4, target_lines - 3), "min_words": max(16, word_min),
        "sections": tuple(s.strip("[] ") for s in structure.replace("optional ", "").split(",")),
    }


def _subject_terms(description: str) -> list[str]:
    source = description.strip().lower()
    if " about " in source:
        source = source.split(" about ", 1)[1]
    words = re.findall(r"[A-Za-z0-9']+", source)
    terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in STOP_WORDS or word.isdigit():
            continue
        if word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) == 4:
            break
    return terms


def _fallback_lines(section: str, section_index: int, hook: str, theme: str, accent: str) -> list[str]:
    if section == "Verse":
        variants = [
            [f"{hook} in the air while the {theme} starts to rise",
             "We lean into the feeling and let it color the night"],
            [f"Every little spark of {accent} keeps the whole room bright",
             "We sing it like a secret that finally found the light"],
            ["Another wave of heat makes the windows start to shake",
             "We laugh into the echo of every move we make"],
        ]
        return variants[min(section_index, len(variants) - 1)]
    if section == "Chorus":
        variants = [
            [f"{hook}, keep the fire moving through the night",
             f"{theme.title()}, in the rhythm everything feels right"],
            [f"{hook}, turn the hunger into something we can sing",
             "Hold the heat a little higher, let the whole room ring"],
        ]
        return variants[min(section_index, len(variants) - 1)]
    if section == "Bridge":
        return [f"We ride the taste of {theme} like a midnight wave",
                "Let the beat go softer just before it breaks"]
    return [f"{hook} on our lips as the final lights grow thin",
            "We carry that flavor with us when the next song begins"]


def _fallback_lyrics(title: str, description: str, audio_duration: float, instrumental: bool) -> str:
    if instrumental:
        return "[Instrumental]"
    hook = title or "Midnight Echo"
    terms = _subject_terms(description)
    theme = " ".join(terms[:2]).strip() or "midnight heat"
    accent = terms[2] if len(terms) >= 3 else (terms[0] if terms else "rhythm")
    plan = _lyric_plan(audio_duration)
    section_counts: dict[str, int] = {}
    chunks: list[str] = []
    for section in plan["sections"]:
        count = section_counts.get(section, 0)
        section_counts[section] = count + 1
        lines = _fallback_lines(section, count, hook, theme, accent)
        chunks.append(f"[{section}]\n" + "\n".join(lines))
    return "\n\n".join(chunks)


def _normalize_tags(tags: Any, description: str) -> list[str]:
    if isinstance(tags, str):
        candidates = re.split(r"[,/;|]", tags)
    elif isinstance(tags, list):
        candidates = tags
    else:
        candidates = []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        tag = str(item).strip().lower()
        if not tag:
            continue
        if len(tag) > 28:
            tag = tag[:28].strip()
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
        if len(normalized) == 6:
            break

    if len(normalized) >= 4:
        return normalized

    desc_lower = description.lower()
    if "lofi" in desc_lower or "lo-fi" in desc_lower:
        fallback = ["lo-fi hip hop", "mellow piano", "vinyl texture", "chill", "warm analog mix"]
    elif "rock" in desc_lower:
        fallback = ["rock", "electric guitar", "powerful drums", "energetic", "high-fidelity"]
    elif "rap" in desc_lower or "hip hop" in desc_lower or "trap" in desc_lower or "drill" in desc_lower:
        fallback = ["hip-hop", "808 bass", "trap hi-hats", "male rap vocal", "crisp modern mix"]
    elif "jazz" in desc_lower:
        fallback = ["jazz", "saxophone", "upright bass", "smooth", "warm analog"]
    elif "electronic" in desc_lower or "edm" in desc_lower:
        fallback = ["electronic", "synth pads", "driving rhythm", "energetic", "high-fidelity"]
    elif "sad" in desc_lower or "melanchol" in desc_lower:
        fallback = ["ballad", "piano", "soft strings", "melancholic", "intimate"]
    elif "classical" in desc_lower:
        fallback = ["orchestral", "strings", "piano", "cinematic", "high-fidelity"]
    else:
        fallback = ["pop", "synth", "catchy melody", "uplifting", "high-fidelity"]

    for tag in fallback:
        if tag not in seen:
            normalized.append(tag)
            seen.add(tag)
        if len(normalized) == 5:
            break
    return normalized


def _normalize_lyrics(lyrics: Any, instrumental: bool) -> str:
    if instrumental:
        return "[Instrumental]"
    # Handle lyrics as array (some models return lines as JSON array)
    if isinstance(lyrics, list):
        text = "\n".join(str(line) for line in lyrics)
    else:
        text = str(lyrics or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    if "[" not in text:
        text = f"[Verse]\n{text}"
    return text


def _has_meaningful_lyrics(text: str, audio_duration: float) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("[end]", "[lyrics]", "[lyritic]", "[end song]")):
        return False
    plan = _lyric_plan(audio_duration)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lyric_lines: list[str] = []
    for line in nonempty_lines:
        stripped = re.sub(r"\[[^\]]+\]", "", line).strip()
        if stripped:
            lyric_lines.append(stripped)
    if len(lyric_lines) < plan["min_lines"]:
        return False
    body = "\n".join(lyric_lines)
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", body, re.UNICODE)
    return len(words) >= plan["min_words"]


def _duration_prompt(audio_duration: float, instrumental: bool) -> str:
    if instrumental:
        return "Keep the output instrumental. Set lyrics to [Instrumental]."
    plan = _lyric_plan(audio_duration)
    target_words = int(audio_duration * 1.3)
    word_min = int(audio_duration * 0.9)
    word_max = int(audio_duration * 1.8)
    return (
        f"Use this section plan: {plan['structure']}.\n"
        f"Write {plan['line_range']} non-empty lyric lines total.\n"
        f"Target approximately {target_words} sung words total ({word_min}-{word_max} range).\n"
        f"Reference: real hit songs average ~1.3 words/sec including instrumental breaks. "
        f"For {int(audio_duration)} seconds, {word_min}-{word_max} words is realistic. "
        "Use more words for rap, fewer for ballads.\n"
        "Keep each line 4-8 words for natural singability.\n"
        "Every section must include actual sung lines, not empty labels.\n"
        "Do not emit placeholder tokens such as [END], [LYRICS], or [Instrumental]."
    )


def _guess_title(description: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", description)
    if not words:
        return "Untitled"
    return " ".join(words[:5]).title()[:48].strip() or "Untitled"


def _strip_wrappers(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _loads_json_lenient(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        repaired: list[str] = []
        in_string = False
        escape = False
        for char in text:
            if in_string:
                if escape:
                    repaired.append(char)
                    escape = False
                    continue
                if char == "\\":
                    repaired.append(char)
                    escape = True
                    continue
                if char == '"':
                    repaired.append(char)
                    in_string = False
                    continue
                if char == "\n":
                    repaired.append("\\n")
                    continue
                if char == "\r":
                    continue
                if char == "\t":
                    repaired.append("\\t")
                    continue
                repaired.append(char)
                continue
            repaired.append(char)
            if char == '"':
                in_string = True
        payload = json.loads("".join(repaired))
    if not isinstance(payload, dict):
        raise ValueError("model returned non-object JSON")
    return payload


def _extract_json(raw: str) -> dict[str, Any]:
    # Strip <think>...</think> tags
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    cleaned = _strip_wrappers(cleaned)
    try:
        return _loads_json_lenient(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("model did not return JSON")
    return _loads_json_lenient(match.group(0))


def _log_block(label: str, text: str) -> None:
    print(f"[{label}] ---")
    cleaned = (text or "").rstrip()
    print(cleaned if cleaned else "<empty>")
    print(f"[/{label}] ---")


# ── Local LLM Composer ───────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class LocalComposer:
    def __init__(self, ollama_model: str | None = None, planner_lm_provider: str = "ollama", planner_model: str | None = None):
        self.ollama_model = ollama_model
        self.planner_lm_provider = normalize_provider(planner_lm_provider)
        self.planner_model = planner_model

    def compose(
        self,
        description: str,
        audio_duration: float = 60.0,
        profile: str = "auto",
        instrumental: bool = False,
        ollama_model: str | None = None,
        planner_lm_provider: str = "ollama",
        planner_model: str | None = None,
        planner_llm_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider = normalize_provider(planner_lm_provider or self.planner_lm_provider)
        model = planner_model or (ollama_model if provider == "ollama" else "") or self.planner_model or self.ollama_model
        if not model:
            # Try to pick the first available local chat model.
            try:
                catalog = model_catalog(provider)
                models = catalog.get("chat_models") or catalog.get("models") or []
                if models:
                    model = str(models[0])
                    print(f"[composer] auto-selected {provider_label(provider)} model: {model}")
                else:
                    raise RuntimeError(f"No {provider_label(provider)} chat models available.")
            except Exception as exc:
                raise RuntimeError(f"Cannot connect to {provider_label(provider)}: {exc}") from exc

        planner_settings = planner_llm_settings_from_payload(
            planner_llm_settings or {},
            default_max_tokens=2048,
            default_timeout=600.0,
        )
        compose_started_at = time.perf_counter()
        print(
            f"[composer] starting provider={provider} model={model} duration={audio_duration} "
            f"instrumental={instrumental} planner_settings={planner_settings}"
        )

        user_prompt = (
            f"Description: {description.strip()}\n"
            f"Instrumental: {'yes' if instrumental else 'no'}\n"
            f"Target duration: {int(audio_duration)} seconds\n"
            f"{_duration_prompt(audio_duration, instrumental)}\n"
            "If the description is vague, expand it creatively into a fully detailed, "
            "professional specification with rich tags, specific instrumentation, "
            "and vivid lyrics. Aim for radio-quality output.\n"
            "Write the song spec now."
        )
        _log_block("composer.prompt", user_prompt)

        try:
            generation_started_at = time.perf_counter()
            print(f"[composer] generating song spec via {provider_label(provider)}...")
            content = chat_completion(
                provider,
                model,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options=planner_llm_options_for_provider(
                    provider,
                    planner_settings,
                    default_max_tokens=2048,
                    default_timeout=600.0,
                ),
                json_format=True,
            )
            content = content or "{}"
            generation_elapsed = time.perf_counter() - generation_started_at
            print(f"[composer] response received elapsed={generation_elapsed:.2f}s chars={len(content)}")
            _log_block("composer.raw_response", content)
            payload = _extract_json(content)
            print(f"[composer] parsed response keys={sorted(payload.keys())}")
        except Exception as exc:
            print(f"[composer] ERROR: {exc}")
            payload = {}

        title = str(payload.get("title") or _guess_title(description)).strip()[:60] or "Untitled"
        tags = _normalize_tags(payload.get("tags"), description)
        intent_payload = payload.get("song_intent") if isinstance(payload.get("song_intent"), dict) else {}
        caption = str(payload.get("caption") or intent_payload.get("caption") or ", ".join(tags)).strip()
        if not intent_payload:
            intent_payload = {
                "genre_family": "rap" if any("rap" in tag or "hip" in tag for tag in tags) else ("instrumental" if instrumental else "pop"),
                "subgenre": tags[0] if tags else "modern pop",
                "mood": next((tag for tag in tags if tag in {"dark", "uplifting", "melancholic", "energetic", "smooth"}), ""),
                "energy": "mid-tempo, controlled energy",
                "vocal_type": "instrumental, no lead vocal" if instrumental else next((tag for tag in tags if "vocal" in tag), "clear lead vocal"),
                "drum_groove": next((tag for tag in tags if any(token in tag for token in ("drum", "hi-hat", "kick", "snare"))), "crisp pocket drums"),
                "bass_low_end": next((tag for tag in tags if "bass" in tag or "808" in tag or "sub" in tag), "controlled low-end"),
                "melodic_identity": next((tag for tag in tags if any(token in tag for token in ("piano", "guitar", "synth", "sample", "strings", "keys"))), "clear melodic motif"),
                "texture_space": "wide stereo space",
                "mix_master": next((tag for tag in tags if "mix" in tag or "fidelity" in tag), "polished modern mix"),
                "custom_tags": [],
                "caption": caption,
            }
        intent_payload.setdefault("caption", caption)
        artist_name = normalize_artist_name(
            payload.get("artist_name") or payload.get("artist"),
            derive_artist_name(title, description, ", ".join(tags)),
        )
        bpm = payload.get("bpm")
        try:
            bpm_value = int(bpm)
        except (TypeError, ValueError):
            bpm_value = 120
        bpm_value = min(300, max(30, bpm_value))

        key_scale = str(payload.get("key_scale") or "").strip()
        if not re.match(r'^[A-G][#b]?\s+(major|minor)$', key_scale, re.IGNORECASE):
            key_scale = ""

        time_sig = payload.get("time_signature")
        try:
            time_sig_value = int(time_sig)
        except (TypeError, ValueError):
            time_sig_value = 4
        if time_sig_value not in (2, 3, 4, 6):
            time_sig_value = 4

        language = str(payload.get("language") or ("instrumental" if instrumental else "en")).strip().lower()
        if language not in {"en", "zh", "ja", "ko", "instrumental", "unknown"}:
            language = "instrumental" if instrumental else "en"

        lyrics = _normalize_lyrics(payload.get("lyrics"), instrumental)
        used_fallback_lyrics = False
        if not instrumental and (language == "instrumental" or not _has_meaningful_lyrics(lyrics, audio_duration)):
            language = "en"
            lyrics = _fallback_lyrics(title, description, audio_duration, instrumental=False)
            used_fallback_lyrics = True

        total_elapsed = time.perf_counter() - compose_started_at
        print(
            f"[composer] done model={model} language={language} bpm={bpm_value} "
            f"key_scale={key_scale or 'N/A'} time_signature={time_sig_value} "
            f"fallback_lyrics={used_fallback_lyrics} total={total_elapsed:.2f}s"
        )
        print(f"[composer] artist_name={artist_name}")
        print(f"[composer] title={title}")
        print(f"[composer] tags={', '.join(tags)}")
        _log_block("composer.final_lyrics", lyrics)

        return {
            "artist_name": artist_name,
            "title": title,
            "tags": ", ".join(tags),
            "caption": caption,
            "song_intent": intent_payload,
            "bpm": bpm_value,
            "key_scale": key_scale,
            "time_signature": str(time_sig_value),
            "language": language,
            "lyrics": lyrics,
            "quality_profile": str(payload.get("quality_profile") or "chart_master"),
            "song_model": str(payload.get("song_model") or "acestep-v15-xl-sft"),
            "audio_format": str(payload.get("audio_format") or "wav32"),
            "inference_steps": int(payload.get("inference_steps") or 64),
            "guidance_scale": float(payload.get("guidance_scale") or 8.0),
            "shift": float(payload.get("shift") or 3.0),
            "infer_method": str(payload.get("infer_method") or "ode"),
            "sampler_mode": str(payload.get("sampler_mode") or "heun"),
            "composer_model": model,
            "composer_provider": provider,
        }
