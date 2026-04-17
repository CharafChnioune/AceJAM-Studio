from __future__ import annotations

import base64
import gc
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

for name in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(name, None)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
DATA_DIR = BASE_DIR / "data"
SONGS_DIR = DATA_DIR / "songs"
KNOWN_ACE_STEP_MODELS = [
    "acestep-v15-turbo",
    "acestep-v15-xl-turbo",
]

MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SONGS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_MODULES_CACHE", str(MODEL_CACHE_DIR / "hf_modules"))
os.environ.setdefault("MPLCONFIGDIR", str(MODEL_CACHE_DIR / "matplotlib"))

NANO_VLLM_DIR = BASE_DIR / "acestep" / "third_parts" / "nano-vllm"
if NANO_VLLM_DIR.exists():
    sys.path.insert(0, str(NANO_VLLM_DIR))

import torch
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from gradio import Server

from acestep.handler import AceStepHandler
from local_composer import LocalComposer


def _cleanup_accelerator_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _default_acestep_checkpoint() -> str:
    override = os.environ.get("ACE_STEP_MODEL", "").strip()
    if override:
        return override
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "acestep-v15-turbo"
    return "acestep-v15-xl-turbo"


def _song_model_label(name: str) -> str:
    labels = {
        "acestep-v15-turbo": "Turbo",
        "acestep-v15-xl-turbo": "XL Turbo",
    }
    return labels.get(name, name)


def _available_acestep_models() -> list[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    available = {name for name in KNOWN_ACE_STEP_MODELS}
    if checkpoint_dir.exists():
        for child in checkpoint_dir.iterdir():
            if child.is_dir() and child.name.startswith("acestep-v15-"):
                available.add(child.name)
    preferred_order = {name: index for index, name in enumerate(KNOWN_ACE_STEP_MODELS)}
    return sorted(available, key=lambda name: (preferred_order.get(name, len(KNOWN_ACE_STEP_MODELS)), name))


def _normalize_song_model(requested: str | None) -> str:
    value = (requested or "").strip()
    if not value or value == "auto":
        return _default_acestep_checkpoint()
    if value.startswith("acestep-v15-"):
        return value
    return _default_acestep_checkpoint()


def _log_block(label: str, text: str) -> None:
    print(f"[{label}] ---")
    cleaned = (text or "").rstrip()
    print(cleaned if cleaned else "<empty>")
    print(f"[/{label}] ---")


def _get_storage_path() -> str:
    storage_root = MODEL_CACHE_DIR
    checkpoint_dir = storage_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = _default_acestep_checkpoint()

    try:
        from huggingface_hub import snapshot_download

        target = checkpoint_dir / checkpoint_name
        if not target.exists():
            cached = Path(snapshot_download(f"ACE-Step/{checkpoint_name}", local_files_only=True))
            try:
                target.symlink_to(cached, target_is_directory=True)
                print(f"[startup] Linked {checkpoint_name} -> {cached}")
            except FileExistsError:
                pass
            except OSError as exc:
                print(f"[startup] Could not link {checkpoint_name}: {exc}")

        shared_cache = Path(snapshot_download("ACE-Step/Ace-Step1.5", local_files_only=True))
        for child in shared_cache.iterdir():
            dst = checkpoint_dir / child.name
            if dst.exists() or not child.is_dir():
                continue
            try:
                dst.symlink_to(child, target_is_directory=True)
                print(f"[startup] Linked {child.name} -> {child}")
            except OSError as exc:
                print(f"[startup] Could not link {child.name}: {exc}")
    except Exception as exc:
        print(f"[startup] Cache warm links skipped: {exc}")

    return str(storage_root)


STORAGE_PATH = _get_storage_path()
print(f"[startup] Model storage: {STORAGE_PATH}")
ACE_STEP_CHECKPOINT = _default_acestep_checkpoint()
print(f"[startup] ACE-Step checkpoint: {ACE_STEP_CHECKPOINT}")

handler = AceStepHandler(persistent_storage_path=STORAGE_PATH)
handler_lock = threading.Lock()
ACTIVE_ACE_STEP_MODEL = ACE_STEP_CHECKPOINT


def _release_handler_state() -> None:
    handler.model = None
    handler.config = None
    handler.vae = None
    handler.text_encoder = None
    handler.text_tokenizer = None
    handler.silence_latent = None
    gc.collect()
    _cleanup_accelerator_memory()


def _initialize_acestep_handler(config_path: str) -> tuple[str, bool]:
    return handler.initialize_service(
        project_root=str(BASE_DIR),
        config_path=config_path,
        device="auto",
        use_flash_attention=handler.is_flash_attention_available(),
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )


def _ensure_song_model(requested: str | None) -> str:
    global ACTIVE_ACE_STEP_MODEL

    target_model = _normalize_song_model(requested)
    if handler.model is not None and ACTIVE_ACE_STEP_MODEL == target_model:
        return ACTIVE_ACE_STEP_MODEL

    previous_model = ACTIVE_ACE_STEP_MODEL
    if handler.model is None:
        print(f"[song-model] initializing {target_model}")
    else:
        print(f"[song-model] switching {previous_model} -> {target_model}")

    _release_handler_state()
    status, ready = _initialize_acestep_handler(target_model)
    if ready:
        ACTIVE_ACE_STEP_MODEL = target_model
        print(f"[song-model] active={ACTIVE_ACE_STEP_MODEL}")
        print(status)
        return ACTIVE_ACE_STEP_MODEL

    print(f"[song-model] failed to load {target_model}")
    print(status)
    if previous_model != target_model:
        print(f"[song-model] restoring previous model {previous_model}")
        _release_handler_state()
        restore_status, restore_ready = _initialize_acestep_handler(previous_model)
        if restore_ready:
            ACTIVE_ACE_STEP_MODEL = previous_model
            print(f"[song-model] restored active={ACTIVE_ACE_STEP_MODEL}")
            print(restore_status)
        else:
            print("[song-model] restore failed")
            print(restore_status)

    raise RuntimeError(f"failed to initialize ACE-Step model: {target_model}")


status, ready = _initialize_acestep_handler(ACE_STEP_CHECKPOINT)
print(f"[startup] Handler ready={ready} status={status}")

composer = LocalComposer(BASE_DIR / "composer_models")


def _language_for_generation(language: str) -> str:
    if language in {"en", "zh", "ja", "ko"}:
        return language
    return "en"


def _run_inference(
    prompt: str,
    lyrics: str,
    audio_duration: float,
    infer_steps: int,
    seed: int,
    language: str,
    song_model: str | None = None,
) -> tuple[str, str]:
    use_random_seed = seed < 0
    with handler_lock:
        active_song_model = _ensure_song_model(song_model)
        result = handler.generate_music(
            captions=prompt,
            lyrics=lyrics,
            audio_duration=audio_duration,
            inference_steps=infer_steps,
            guidance_scale=7.0,
            use_random_seed=use_random_seed,
            seed=None if use_random_seed else seed,
            infer_method="ode",
            shift=1.0,
            use_adg=False,
            vocal_language=_language_for_generation(language),
            # The UI returns a single track, so avoid generating an unused second sample.
            batch_size=1,
        )

    if not result.get("success"):
        raise RuntimeError(result.get("error", "generation failed"))

    audio_dict = result["audios"][0]
    tensor = audio_dict["tensor"]
    sample_rate = audio_dict["sample_rate"]

    data = tensor.cpu().float().numpy()
    if data.ndim == 2:
        data = data.T
        if data.shape[1] == 1:
            data = data[:, 0]

    peak = float(np.abs(data).max())
    if peak > 1e-4:
        data = (data / peak * 0.95).astype(np.float32)

    out_path = Path(tempfile.mkdtemp()) / "output.wav"
    sf.write(str(out_path), data, sample_rate)
    return str(out_path), active_song_model


def _song_public_url(song_id: str, filename: str) -> str:
    return f"/media/songs/{song_id}/{filename}"


def _decorate_song(meta: dict) -> dict:
    entry = dict(meta)
    audio_file = entry.get("audio_file")
    if audio_file:
        entry["audio_url"] = _song_public_url(entry["id"], audio_file)
    thumb_file = entry.get("thumb_file")
    if thumb_file:
        entry["thumb_url"] = _song_public_url(entry["id"], thumb_file)
    return entry


def _load_feed_from_disk() -> list[dict]:
    songs: list[dict] = []
    if not SONGS_DIR.exists():
        return songs

    for song_dir in SONGS_DIR.iterdir():
        meta_path = song_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            songs.append(_decorate_song(meta))
        except Exception:
            continue

    songs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    print(f"[feed] Loaded {len(songs)} saved songs")
    return songs


_feed_songs = _load_feed_from_disk()

app = Server(title="AceJAM")


@app.api(name="create", concurrency_limit=1, time_limit=420)
def create(
    description: str,
    audio_duration: float = 60.0,
    seed: int = -1,
    community: bool = False,
    composer_profile: str = "auto",
    song_model: str = "auto",
    instrumental: bool = False,
) -> str:
    started_at = time.perf_counter()
    try:
        print(
            "[create] "
            f"request duration={audio_duration} "
            f"seed={seed} "
            f"community={community} "
            f"composer_profile={composer_profile} "
            f"song_model={song_model} "
            f"instrumental={instrumental}"
        )
        _log_block("create.description", description)
        compose_started_at = time.perf_counter()
        composed = composer.compose(
            description=description,
            audio_duration=audio_duration,
            profile=composer_profile,
            instrumental=instrumental,
        )
        compose_elapsed = time.perf_counter() - compose_started_at
        print(
            "[create] "
            f"profile={composed['composer_profile']} "
            f"model={composed['composer_model']} "
            f"title={composed['title']} "
            f"language={composed['language']} "
            f"bpm={composed['bpm']} "
            f"tags={composed['tags'][:80]} "
            f"compose_time={compose_elapsed:.2f}s"
        )
        _log_block("create.generated_lyrics", composed["lyrics"])
        _cleanup_accelerator_memory()

        print(
            "[create->acestep] "
            f"requested_song_model={song_model} "
            f"audio_duration={audio_duration} "
            f"infer_steps=8 "
            f"seed={seed} "
            f"language={composed['language']}"
        )
        _log_block("create.acestep_prompt", composed["tags"])
        _log_block("create.acestep_lyrics", composed["lyrics"])
        inference_started_at = time.perf_counter()
        wav_path, active_song_model = _run_inference(
            prompt=composed["tags"],
            lyrics=composed["lyrics"],
            audio_duration=audio_duration,
            infer_steps=8,
            seed=seed,
            language=composed["language"],
            song_model=song_model,
        )
        inference_elapsed = time.perf_counter() - inference_started_at
        total_elapsed = time.perf_counter() - started_at
        print(
            "[create timing] "
            f"compose={compose_elapsed:.2f}s "
            f"generate={inference_elapsed:.2f}s "
            f"total={total_elapsed:.2f}s"
        )
        wav_bytes = Path(wav_path).read_bytes()
        audio_b64 = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode()}"

        result = {
            "audio": audio_b64,
            "title": composed["title"],
            "tags": composed["tags"],
            "lyrics": composed["lyrics"],
            "bpm": composed["bpm"],
            "language": composed["language"],
            "composer_profile": composed["composer_profile"],
            "composer_model": composed["composer_model"],
            "song_model": active_song_model,
        }

        if community:
            song_id = uuid.uuid4().hex[:12]
            song_dir = SONGS_DIR / song_id
            song_dir.mkdir(parents=True, exist_ok=True)

            audio_file = f"{song_id}.wav"
            (song_dir / audio_file).write_bytes(wav_bytes)

            meta = {
                "id": song_id,
                "title": composed["title"],
                "description": description,
                "tags": composed["tags"],
                "lyrics": composed["lyrics"],
                "bpm": composed["bpm"],
                "language": composed["language"],
                "duration": audio_duration,
                "audio_file": audio_file,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (song_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            entry = _decorate_song(meta)
            _feed_songs.insert(0, entry)
            result["community_url"] = entry["audio_url"]

        return json.dumps(result)
    except Exception as exc:
        print(f"[create ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
    finally:
        _cleanup_accelerator_memory()


@app.api(name="generate", concurrency_limit=1, time_limit=240)
def generate(
    prompt: str,
    lyrics: str,
    audio_duration: float = 60.0,
    infer_step: int = 8,
    guidance_scale: float = 7.0,
    seed: int = -1,
    song_model: str = "auto",
    lora_name_or_path: str = "",
    lora_weight: float = 0.8,
) -> str:
    del guidance_scale, lora_name_or_path, lora_weight
    try:
        wav_path, _ = _run_inference(prompt, lyrics, audio_duration, infer_step, seed, "en", song_model=song_model)
        encoded = base64.b64encode(Path(wav_path).read_bytes()).decode()
        return f"data:audio/wav;base64,{encoded}"
    except Exception as exc:
        print(f"[generate ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
    finally:
        _cleanup_accelerator_memory()


@app.api(name="community", concurrency_limit=4)
def community() -> str:
    return json.dumps(_feed_songs[:50])


@app.api(name="config", concurrency_limit=8)
def config() -> str:
    return json.dumps(
        {
            "active_song_model": ACTIVE_ACE_STEP_MODEL,
            "default_song_model": _default_acestep_checkpoint(),
            "available_song_models": _available_acestep_models(),
        }
    )


@app.get("/media/songs/{song_id}/{filename}")
async def media(song_id: str, filename: str):
    songs_root = SONGS_DIR.resolve()
    song_dir = (SONGS_DIR / song_id).resolve()
    target = (song_dir / filename).resolve()
    if songs_root not in song_dir.parents or not song_dir.is_dir() or song_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


demo = app


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False)
