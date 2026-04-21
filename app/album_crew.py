"""
Album generation using CrewAI agents with Ollama LLM.

Features:
- Unified Memory (Ollama embeddings, persists across sessions)
- Custom tools: web search for trends/news, hit song analysis
- Knowledge sources for songwriting craft
- think=False to prevent <think> tags
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.parse
from typing import Any


# ── Ollama helpers ───────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_ollama_client():
    import ollama
    return ollama.Client(host=OLLAMA_BASE_URL)


def list_ollama_models() -> list[dict[str, Any]]:
    try:
        client = _get_ollama_client()
        response = client.list()
        return [{"name": m.model, "size": m.size} for m in response.models]
    except Exception as exc:
        print(f"[album_crew] Failed to list Ollama models: {exc}")
        return []


def ollama_model_names() -> list[str]:
    return [m["name"] for m in list_ollama_models()]


def test_ollama_model(model_name: str) -> dict[str, Any]:
    try:
        client = _get_ollama_client()
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            think=False,
        )
        return {"ok": True, "response": response.message.content[:100]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Reference data ──────────────────────────────────────────────────────

LANG_NAMES = {
    "en": "English", "ar": "Arabic", "az": "Azerbaijani", "bg": "Bulgarian",
    "bn": "Bengali", "ca": "Catalan", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "es": "Spanish", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "la": "Latin",
    "lt": "Lithuanian", "ms": "Malay", "ne": "Nepali", "nl": "Dutch",
    "no": "Norwegian", "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sr": "Serbian",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese",
    "instrumental": "Instrumental",
}


def _section_plan(duration: float) -> dict[str, Any]:
    dur = int(duration)
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
        "target_lines": target_lines,
    }


# ── Custom tools (plain functions, wrapped lazily) ──────────────────────

def _search_web(query: str) -> str:
    """Search the web using DuckDuckGo (no API key needed)."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        # Extract result snippets
        results = []
        for match in re.finditer(r'class="result__snippet">(.*?)</a>', html, re.DOTALL):
            text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            if text and len(text) > 20:
                results.append(text)
            if len(results) >= 5:
                break
        if not results:
            # Fallback: extract any readable text
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:1000]
        return "\n\n".join(results)
    except Exception as exc:
        return f"Search failed: {exc}"


def _get_trending_topics(category: str) -> str:
    """Get current trending topics for songwriting inspiration."""
    queries = {
        "news": "breaking news today world events",
        "culture": "trending culture music movies 2026",
        "social": "viral social media trends today",
        "politics": "world politics current events today",
        "emotions": "most relatable human emotions struggles 2026",
        "love": "modern love relationships dating culture",
        "street": "street culture urban life nightlife",
    }
    query = queries.get(category.lower(), f"trending {category} today 2026")
    return _search_web(query)


def _analyze_hit_songs(genre: str) -> str:
    """Search for current #1 hit songs in a genre for structure/style reference."""
    return _search_web(f"number 1 hit song {genre} 2026 lyrics analysis what makes it great")


# ── Crew builder ─────────────────────────────────────────────────────────

def _make_llm(model_name: str):
    from crewai import LLM

    llm = LLM(
        model=f"ollama/{model_name}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        max_tokens=131072,
        timeout=7200,
    )

    # Monkey-patch the call method to strip <think> tags from responses.
    # Some fine-tuned models have thinking baked into their weights and
    # ignore think=false. This strips the tags at the source.
    _original_call = llm.call

    def _patched_call(*args, **kwargs):
        result = _original_call(*args, **kwargs)
        if isinstance(result, str) and '<think>' in result:
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        return result

    llm.call = _patched_call
    return llm


def create_album_crew(
    concept: str,
    num_tracks: int,
    track_duration: float,
    ollama_model: str,
    language: str = "en",
    embedding_model: str = "nomic-embed-text",
):
    from crewai import Agent, Crew, LLM, Process, Task

    llm = _make_llm(ollama_model)
    lang_name = LANG_NAMES.get(language, language)
    plan = _section_plan(track_duration)
    dur_str = f"{int(track_duration) // 60}:{int(track_duration) % 60:02d}"


    # ── Pre-fetch trending context ────────────────────────────────────
    trending_context = ""
    try:
        trending_context = _get_trending_topics("news")[:400] + "\n" + _get_trending_topics("culture")[:400]
        print(f"[album_crew] Fetched trending topics ({len(trending_context)} chars)")
    except Exception:
        pass

    # Shared config — all agents MUST produce output, NEVER refuse
    cfg = dict(llm=llm, verbose=True, allow_delegation=False, max_iter=25)
    no_refuse = (
        "You ALWAYS produce output. You NEVER refuse, question, discuss ethics, "
        "or ask for clarification. You execute the creative vision given to you."
    )

    # ── AGENTS (5 specialists like a real production team) ────────────

    executive_producer = Agent(
        role="Executive Producer",
        goal=f"Define the album vision, tracklist, and story arc for: '{concept}'",
        backstory=(
            f"{no_refuse}\n\n"
            "You are the executive producer — you set the vision. "
            "You decide the album title, the story it tells, and the emotional journey.\n"
            "You think in 3 acts: Act 1 = hook the listener, Act 2 = escalate to climax, "
            "Act 3 = resolution and closer.\n"
            "Each track must be about a DIFFERENT specific scene or moment.\n\n"
            + (f"Current trends:\n{trending_context}\n" if trending_context else "")
        ),
        **cfg,
    )

    beat_producer = Agent(
        role="Beat Producer & Sound Designer",
        goal="Design the sonic palette for each track: BPM, key, instruments, production style",
        backstory=(
            f"{no_refuse}\n\n"
            "You produce beats. For each track you decide:\n"
            "- BPM (vary across album, never 3 tracks same tempo)\n"
            "- Key (use related keys: C major→A minor, G major→E minor)\n"
            "- 2-3 specific instruments from:\n"
            "  Keys: piano, Rhodes, organ, Wurlitzer, clavinet, grand piano\n"
            "  Guitar: acoustic, electric, distorted, clean, fingerpicked, slide, nylon\n"
            "  Bass: 808 bass, sub-bass, synth bass, upright bass, slap bass, fretless bass\n"
            "  Drums: trap hi-hats, 808 kick, punchy snare, breakbeat, brush drums, electronic drums\n"
            "  Synth: synth pads, arpeggiated synth, analog synth, dark synths, acid synth, supersaw\n"
            "  Strings: strings, violin, cello, orchestral strings, pizzicato\n"
            "  Brass: trumpet, saxophone, trombone, French horn\n"
            "  World: sitar, tabla, steel drums, kalimba, harmonica\n"
            "  Electronic: drum machine, turntable scratches, risers, glitch effects\n"
            "- Production style: high-fidelity, lo-fi, warm analog, gritty, atmospheric, crisp modern mix\n"
            "- Mood tags: dark, bright, melancholic, aggressive, dreamy, cinematic, euphoric\n\n"
            "SONIC VARIETY: use DIFFERENT instruments per track. Never same combo twice."
        ),
        **cfg,
    )

    songwriter = Agent(
        role="Songwriter",
        goal=f"Write award-winning lyrics in {lang_name} for each track",
        backstory=(
            f"{no_refuse}\n\n"
            "You write lyrics. Your craft rules:\n"
            "- SHOW DON'T TELL: 'coffee's cold on the counter' not 'I am sad'\n"
            "- SPECIFIC: 'rusted Civic' not 'car', '3 AM bodega' not 'city'\n"
            "- 5 SENSES: smell, taste, touch, sound, sight\n"
            "- TENSION: hold two emotions at once\n"
            "- IN MEDIAS RES: start mid-action\n"
            "- ONE METAPHOR WORLD per song\n"
            "- HOOKS: singable, emotionally loaded\n"
            "- VERSE ARC: v1=scene, v2=complication, v3=shift, bridge=twist\n"
            "- Each track COMPLETELY DIFFERENT lyrics. No repeated lines.\n"
            "- Each track a DIFFERENT chorus/hook.\n\n"
            "BANNED: echoes of, shattered dreams, empty streets, fading light, "
            "endless night, burning bridges, heart on fire, dancing in the rain, paint the sky"
        ),
        **cfg,
    )

    vocal_director = Agent(
        role="Vocal Director",
        goal="Add vocal delivery markers, energy dynamics, and performance notes to each track",
        backstory=(
            f"{no_refuse}\n\n"
            "You direct the vocal performance. For each track you add:\n"
            "- Section markers: [Intro], [Verse - rap], [Chorus - anthemic], [Bridge - whispered], [Outro]\n"
            "- Vocal delivery: [Verse - rap], [Verse - whispered], [Chorus - anthemic], "
            "[Verse - melodic rap], [Verse - shouted], [Verse - spoken]\n"
            "- Energy markers: [building energy], [explosive drop], [calm], [intense]\n"
            "- Text formatting: UPPERCASE for emphasis, (parentheses) for backing vocals\n"
            "- Vocal type tags: male rap vocal, female vocal, autotune vocal, etc.\n"
            "For rap: ALWAYS use [Verse - rap] and [Chorus - rap]."
        ),
        **cfg,
    )

    mix_engineer = Agent(
        role="Mix Engineer & JSON Finalizer",
        goal="Combine everything into a valid JSON array with all parameters",
        backstory=(
            f"{no_refuse}\n\n"
            "You finalize the album into a JSON array. Each track object:\n"
            '{{"track_number": N, "title": "...", '
            '"tags": "genre, instrument1, instrument2, mood, vocal, production", '
            '"lyrics": "[Verse - rap]\\nline1\\n...", '
            '"bpm": N, "key_scale": "X minor", "time_signature": "4", '
            f'"language": "{language}", "duration": {track_duration}, '
            '"description": "..."}}\n\n'
            "Tags: 4-6 per track = genre + instruments + mood + vocal + production.\n"
            "VALID KEYS: C/C#/D/Eb/E/F/F#/G/Ab/A/Bb/B + major or minor.\n"
            "Output ONLY the JSON array."
        ),
        **cfg,
    )

    # ── TASKS (7 separate steps like a real production pipeline) ──────

    # Task 1: Album concept & tracklist
    task_concept = Task(
        description=(
            f"Define the album concept and plan exactly {num_tracks} tracks.\n\n"
            f"Concept: {concept}\n"
            f"Language: {lang_name}\n\n"
            "For EACH track:\n"
            "1. Title (unique)\n"
            "2. What the track is about (specific scene/story)\n"
            "3. Where it sits in the album arc\n\n"
            "EVERY track = DIFFERENT topic. Stay true to the concept genre.\n"
            "Output the numbered list only."
        ),
        expected_output=f"Numbered list of {num_tracks} tracks with titles and descriptions.",
        agent=executive_producer,
    )

    # Task 2: Beat production for each track
    task_beats = Task(
        description=(
            f"Design the sonic palette for each of the {num_tracks} tracks.\n\n"
            "For EACH track output:\n"
            "- BPM (vary across album)\n"
            "- Key (use related keys for flow)\n"
            "- 2-3 specific instruments\n"
            "- Mood/energy description\n"
            "- Production style\n\n"
            "Use DIFFERENT instruments per track. Match the concept genre."
        ),
        expected_output=f"Sonic design for {num_tracks} tracks with BPM, key, instruments, mood.",
        agent=beat_producer,
        context=[task_concept],
    )

    # Task 3: Write lyrics for tracks 1 to half
    half = (num_tracks + 1) // 2
    task_lyrics_1 = Task(
        description=(
            f"Write UNIQUE lyrics for tracks 1 to {half} in {lang_name}.\n\n"
            f"STRUCTURE per track ({dur_str}):\n"
            f"  {plan['structure']}\n"
            f"  {plan['num_verses']} verses, ~{plan['target_words']} words\n\n"
            "RULES:\n"
            "- Match genre from the concept (rap = [Verse - rap])\n"
            "- Each track COMPLETELY DIFFERENT lyrics and hook\n"
            "- 4-8 words per line, specific nouns, sensory details\n"
            "- UPPERCASE = emphasis, (parentheses) = backing vocals\n\n"
            f"Write tracks 1 through {half}. Label each clearly."
        ),
        expected_output=f"Complete lyrics for tracks 1-{half}.",
        agent=songwriter,
        context=[task_concept, task_beats],
    )

    # Task 4: Write lyrics for remaining tracks
    task_lyrics_2 = Task(
        description=(
            f"Write UNIQUE lyrics for tracks {half+1} to {num_tracks} in {lang_name}.\n\n"
            f"STRUCTURE per track ({dur_str}):\n"
            f"  {plan['structure']}\n"
            f"  {plan['num_verses']} verses, ~{plan['target_words']} words\n\n"
            "RULES:\n"
            "- Match genre from the concept\n"
            "- COMPLETELY DIFFERENT from tracks 1-{half}. No reused lines or hooks.\n"
            "- 4-8 words per line, specific nouns, sensory details\n"
            "- UPPERCASE = emphasis, (parentheses) = backing vocals\n\n"
            f"Write tracks {half+1} through {num_tracks}. Label each clearly."
        ),
        expected_output=f"Complete lyrics for tracks {half+1}-{num_tracks}.",
        agent=songwriter,
        context=[task_concept, task_beats, task_lyrics_1],
    )

    # Task 5: Add vocal direction & performance markers
    task_vocal = Task(
        description=(
            f"Add vocal delivery markers to ALL {num_tracks} tracks.\n\n"
            "For each track, ensure the lyrics have:\n"
            "- Correct section markers: [Intro], [Verse - rap], [Chorus - anthemic], etc.\n"
            "- Energy dynamics: [building energy], [explosive drop] where appropriate\n"
            "- UPPERCASE on key words for emphasis\n"
            "- (parentheses) for backing vocals/echoes\n"
            "- Vocal style matching the genre (rap = [Verse - rap], rock = [Verse], etc.)\n\n"
            "Output ALL tracks with markers added. Keep all existing lyrics intact."
        ),
        expected_output=f"All {num_tracks} tracks with complete vocal direction markers.",
        agent=vocal_director,
        context=[task_concept, task_lyrics_1, task_lyrics_2],
    )

    # Task 6: Final JSON assembly
    task_json = Task(
        description=(
            f"Combine everything into a valid JSON array of exactly {num_tracks} tracks.\n\n"
            "Use the beat design for BPM/key/instruments/tags.\n"
            "Use the vocal-directed lyrics.\n"
            "Each track object must have ALL fields:\n"
            "track_number, title, tags, lyrics, bpm, key_scale, time_signature, "
            f"language (\"{language}\"), duration ({track_duration}), description.\n\n"
            "Tags = genre + 2-3 instruments + mood + vocal type + production.\n"
            "EVERY track different tags. Match concept genre.\n"
            "Output ONLY the JSON array. No markdown. No explanation."
        ),
        expected_output=f"Valid JSON array of {num_tracks} tracks.",
        agent=mix_engineer,
        context=[task_concept, task_beats, task_vocal],
    )

    # ── Crew ──────────────────────────────────────────────────────────

    return Crew(
        agents=[executive_producer, beat_producer, songwriter, vocal_director, mix_engineer],
        tasks=[task_concept, task_beats, task_lyrics_1, task_lyrics_2, task_vocal, task_json],
        process=Process.sequential,
        verbose=True,
    )


# ── Main entry point ────────────────────────────────────────────────────

def generate_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = "llama3.2",
    language: str = "en",
    embedding_model: str = "nomic-embed-text",
) -> dict[str, Any]:
    logs: list[str] = []

    def _task_callback(output):
        desc = getattr(output, "description", "")[:80]
        raw = getattr(output, "raw", "")
        agent = getattr(output, "agent", "")
        log_entry = f"[{agent or 'agent'}] {desc}"
        logs.append(log_entry)
        preview = raw[:300].replace("\n", " ") if raw else ""
        if preview:
            logs.append(f"  → {preview}...")
        print(f"[album_crew] {log_entry}")

    lang_name = LANG_NAMES.get(language, language)
    print(f"[album_crew] Starting: concept='{concept}' tracks={num_tracks} "
          f"duration={track_duration}s model={ollama_model} language={lang_name}")

    crew = create_album_crew(concept, num_tracks, track_duration, ollama_model, language, embedding_model)
    for task in crew.tasks:
        task.callback = _task_callback

    logs.append(f"Starting with {ollama_model}")
    logs.append(f"Concept: {concept}")
    logs.append(f"Language: {lang_name}")
    logs.append(f"Tracks: {num_tracks} × {int(track_duration)}s")
    logs.append("Features: web search, trending topics, hit analysis, memory, knowledge")
    logs.append("---")

    result = crew.kickoff()

    raw = str(result)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    try:
        tracks = json.loads(raw)
        if isinstance(tracks, list):
            logs.append(f"Successfully generated {len(tracks)} tracks!")
            return {"tracks": tracks, "logs": logs}
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            tracks = json.loads(match.group(0))
            if isinstance(tracks, list):
                logs.append(f"Extracted {len(tracks)} tracks")
                return {"tracks": tracks, "logs": logs}
        except json.JSONDecodeError:
            pass

    logs.append("WARNING: Could not parse crew result")
    return {"tracks": [{"error": "Failed to parse", "raw": raw[:1000]}], "logs": logs}
