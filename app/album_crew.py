"""
Album generation using CrewAI agents with Ollama plus AceJAM songwriting tools.

The CrewAI layer is backed by deterministic post-processing tools. If an LLM
returns weak or malformed JSON, the same toolbelt repairs the plan so album
generation still has usable tags, lyrics, metadata, and model advice.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

from songwriting_toolkit import (
    build_album_plan,
    choose_song_model,
    lyric_length_plan,
    make_crewai_tools,
    normalize_album_tracks,
    sanitize_artist_references,
    toolkit_payload,
)


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

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


def _search_web(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        results = []
        for match in re.finditer(r'class="result__snippet">(.*?)</a>', html, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            text = re.sub(r"\s+", " ", text)
            if text and len(text) > 20:
                results.append(text)
            if len(results) >= 5:
                break
        if results:
            return "\n\n".join(results)
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()[:1000]
    except Exception as exc:
        return f"Search failed: {exc}"


def _make_llm(model_name: str):
    from crewai import LLM

    llm = LLM(
        model=f"ollama/{model_name}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.72,
        max_tokens=131072,
        timeout=7200,
    )
    original_call = llm.call

    def _patched_call(*args, **kwargs):
        result = original_call(*args, **kwargs)
        if isinstance(result, str) and "<think>" in result:
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result

    llm.call = _patched_call
    return llm


def _json_from_text(raw: str) -> list[dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("tracks"), list):
            return [item for item in parsed["tracks"] if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    raise ValueError("Crew result did not contain a valid JSON track array")


def _coerce_options(
    concept: str,
    num_tracks: int,
    track_duration: float,
    language: str,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    opts = dict(options or {})
    sanitized, artist_notes = sanitize_artist_references(concept)
    opts.setdefault("song_model_strategy", "best_installed")
    opts.setdefault("quality_target", "hit")
    opts.setdefault("lyric_density", "dense")
    opts.setdefault("rhyme_density", 0.8)
    opts.setdefault("metaphor_density", 0.7)
    opts.setdefault("hook_intensity", 0.85)
    opts.setdefault("structure_preset", "auto")
    opts.setdefault("bpm_strategy", "varied")
    opts.setdefault("key_strategy", "related")
    opts.setdefault("use_web_inspiration", False)
    opts.setdefault("track_variants", 1)
    opts.update(
        {
            "concept": concept,
            "sanitized_concept": sanitized,
            "artist_reference_notes": artist_notes,
            "num_tracks": int(num_tracks),
            "track_duration": float(track_duration),
            "language": language,
        }
    )
    return opts


def songwriting_toolkit(installed_models: set[str] | list[str] | None = None) -> dict[str, Any]:
    return toolkit_payload(installed_models)


def create_album_crew(
    concept: str,
    num_tracks: int,
    track_duration: float,
    ollama_model: str,
    language: str = "en",
    embedding_model: str = "nomic-embed-text",
    options: dict[str, Any] | None = None,
):
    from crewai import Agent, Crew, Process, Task

    opts = _coerce_options(concept, num_tracks, track_duration, language, options)
    llm = _make_llm(ollama_model)
    lang_name = LANG_NAMES.get(language, language)
    length_plan = lyric_length_plan(
        track_duration,
        str(opts.get("lyric_density") or "dense"),
        str(opts.get("structure_preset") or "auto"),
        opts["sanitized_concept"],
    )
    model_info = choose_song_model(
        set(opts.get("installed_models") or []),
        str(opts.get("song_model_strategy") or "best_installed"),
        str(opts.get("requested_song_model") or "auto"),
    )
    inspiration = ""
    if opts.get("use_web_inspiration"):
        queries = opts.get("inspiration_queries") or opts["sanitized_concept"]
        inspiration = _search_web(str(queries))[:1200]
    tool_context = dict(opts)
    tool_context["web_inspiration"] = inspiration
    tools = make_crewai_tools(tool_context)

    tool_summary = json.dumps(
        {
            "lyric_length_plan": length_plan,
            "model_advice": model_info,
            "quality_target": opts.get("quality_target"),
            "tag_packs": opts.get("tag_packs"),
            "custom_tags": opts.get("custom_tags"),
            "artist_reference_notes": opts.get("artist_reference_notes"),
        },
        ensure_ascii=True,
    )

    shared_rules = (
        "Create original songs only. Do not imitate a living artist. "
        "If artist names appear, convert them into broad technique briefs: internal rhyme, narrative detail, "
        "metaphor discipline, punchlines, hook contrast, breath control, and cinematic imagery. "
        "Use the provided tools when useful. Output concrete, editable production data."
    )
    cfg = dict(llm=llm, verbose=True, allow_delegation=False, max_iter=20)

    executive_producer = Agent(
        role="Executive Producer",
        goal="Design a cohesive album arc with hit-level contrast between tracks",
        backstory=(
            f"{shared_rules}\n\n"
            "You plan albums in acts: opener, escalation, climax, cooldown, closer. "
            "Every track needs a distinct scene, emotional job, title, and hook promise."
        ),
        tools=tools,
        **cfg,
    )
    beat_producer = Agent(
        role="Beat Producer and ACE-Step Tag Architect",
        goal="Create ACE-Step captions, tags, BPM, key, time signature, and model choices",
        backstory=(
            f"{shared_rules}\n\n"
            "Use caption dimensions from ACE-Step: genre, emotion, instruments, timbre, era, "
            "production, vocal character, speed, rhythm, and structure hints. "
            "Keep BPM/key/time in metadata instead of repeating them in captions."
        ),
        tools=tools,
        **cfg,
    )
    songwriter = Agent(
        role="Rhyme, Hook, and Lyric Writer",
        goal=f"Write original, duration-matched lyrics in {lang_name}",
        backstory=(
            f"{shared_rules}\n\n"
            "Write vivid lyrics with specific nouns, internal/slant rhyme, one coherent metaphor world, "
            "clear section tags, hook contrast, and no repeated filler. "
            "Use concise meta tags such as [Verse - rap], [Chorus - anthemic], [Bridge - whispered]."
        ),
        tools=tools,
        **cfg,
    )
    quality_editor = Agent(
        role="A&R Quality Editor and JSON Finalizer",
        goal="Repair weak songs and output strict JSON for AceJAM generation",
        backstory=(
            f"{shared_rules}\n\n"
            "You reject generic lyrics, cliches, repeated hooks, tag conflicts, and under-length songs. "
            "Final output must be a JSON array only."
        ),
        tools=tools,
        **cfg,
    )

    task_concept = Task(
        description=(
            f"Plan exactly {num_tracks} tracks for this album.\n"
            f"Concept: {opts['sanitized_concept']}\n"
            f"Language: {lang_name}\n"
            f"Tool context: {tool_summary}\n"
            + (f"Current inspiration snippets:\n{inspiration}\n" if inspiration else "")
            + "Return track titles, role in album arc, unique scene, and hook promise."
        ),
        expected_output=f"Numbered plan for {num_tracks} distinct tracks.",
        agent=executive_producer,
    )
    task_sonic = Task(
        description=(
            "For each planned track, assign ACE-Step-ready caption tags, BPM, key_scale, time_signature, "
            "vocal character, and the chosen installed song_model. Use ModelAdvisorTool and TagLibraryTool. "
            "Every track must have different tags and a clear production reason."
        ),
        expected_output="Sonic specification for every track.",
        agent=beat_producer,
        context=[task_concept],
    )
    task_lyrics = Task(
        description=(
            f"Write complete lyrics for all tracks in {lang_name}.\n"
            f"Duration per track: {int(track_duration)} seconds.\n"
            f"Required plan: {length_plan['structure']}; target {length_plan['target_words']} words, "
            f"minimum {length_plan['min_words']} words and {length_plan['min_lines']} lyric lines.\n"
            "Use enough lyrics for the duration. Keep each hook unique. Do not include placeholder lines."
        ),
        expected_output="Complete duration-matched lyrics for every track.",
        agent=songwriter,
        context=[task_concept, task_sonic],
    )
    task_json = Task(
        description=(
            "Combine the album plan, sonic specs, and lyrics into a strict JSON array only. "
            "Each object must include: track_number, title, description, tags, lyrics, bpm, key_scale, "
            "time_signature, language, duration, song_model. "
            "Also include tool_notes when you changed an artist reference into technique language."
        ),
        expected_output="Valid JSON array of album tracks only.",
        agent=quality_editor,
        context=[task_concept, task_sonic, task_lyrics],
    )

    return Crew(
        agents=[executive_producer, beat_producer, songwriter, quality_editor],
        tasks=[task_concept, task_sonic, task_lyrics, task_json],
        process=Process.sequential,
        verbose=True,
    )


def plan_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = "llama3.2",
    language: str = "en",
    embedding_model: str = "nomic-embed-text",
    options: dict[str, Any] | None = None,
    use_crewai: bool = True,
    input_tracks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    logs: list[str] = []
    opts = _coerce_options(concept, num_tracks, track_duration, language, options)
    lang_name = LANG_NAMES.get(language, language)
    logs.append(f"Concept: {opts['sanitized_concept']}")
    logs.append(f"Language: {lang_name}")
    logs.append(f"Tracks: {num_tracks} x {int(track_duration)}s")
    logs.append(f"Song model strategy: {opts.get('song_model_strategy')}")
    if opts.get("artist_reference_notes"):
        logs.extend(str(note) for note in opts["artist_reference_notes"])

    model_info = choose_song_model(
        set(opts.get("installed_models") or []),
        str(opts.get("song_model_strategy") or "best_installed"),
        str(opts.get("requested_song_model") or "auto"),
    )
    if not model_info.get("ok"):
        logs.append(f"ERROR: {model_info.get('error')}")
        return {"tracks": [], "logs": logs, "success": False, "error": model_info.get("error"), "toolkit": toolkit_payload(opts.get("installed_models"))}

    if input_tracks:
        tracks = normalize_album_tracks(input_tracks, opts)
        logs.append(f"Using editable album plan with {len(tracks)} tracks.")
        return {
            "tracks": tracks,
            "logs": logs,
            "success": True,
            "toolkit": toolkit_payload(opts.get("installed_models")),
            "toolkit_report": {"model_advice": model_info, "artist_reference_notes": opts.get("artist_reference_notes", [])},
        }

    if use_crewai:
        try:
            logs.append(f"Planning with CrewAI and Ollama model {ollama_model}...")
            crew = create_album_crew(concept, num_tracks, track_duration, ollama_model, language, embedding_model, opts)

            def _task_callback(output):
                desc = getattr(output, "description", "")[:90]
                raw = getattr(output, "raw", "")
                agent = getattr(output, "agent", "")
                logs.append(f"[{agent or 'agent'}] {desc}")
                if raw:
                    logs.append("  " + raw[:240].replace("\n", " ") + "...")

            for task in crew.tasks:
                task.callback = _task_callback
            result = crew.kickoff()
            parsed = _json_from_text(str(result))
            tracks = normalize_album_tracks(parsed[:num_tracks], opts)
            logs.append(f"CrewAI planned {len(tracks)} tracks.")
            return {
                "tracks": tracks,
                "logs": logs,
                "success": True,
                "toolkit": toolkit_payload(opts.get("installed_models")),
                "toolkit_report": {"model_advice": model_info, "artist_reference_notes": opts.get("artist_reference_notes", [])},
            }
        except Exception as exc:
            logs.append(f"CrewAI planning fell back to deterministic toolbelt: {exc}")

    fallback = build_album_plan(concept, num_tracks, track_duration, opts)
    logs.append(f"Toolbelt fallback planned {len(fallback['tracks'])} tracks.")
    return {
        "tracks": fallback["tracks"],
        "logs": logs,
        "success": True,
        "toolkit": toolkit_payload(opts.get("installed_models")),
        "toolkit_report": fallback.get("toolkit_report", {}),
    }


def generate_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = "llama3.2",
    language: str = "en",
    embedding_model: str = "nomic-embed-text",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return plan_album(
        concept=concept,
        num_tracks=num_tracks,
        track_duration=track_duration,
        ollama_model=ollama_model,
        language=language,
        embedding_model=embedding_model,
        options=options,
        use_crewai=True,
    )
