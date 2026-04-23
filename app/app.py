from __future__ import annotations

import base64
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

for name in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(name, None)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
DATA_DIR = BASE_DIR / "data"
SONGS_DIR = DATA_DIR / "songs"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
LORA_DATASETS_DIR = DATA_DIR / "lora_datasets"
LORA_EXPORTS_DIR = DATA_DIR / "loras"
OFFICIAL_ACE_STEP_DIR = BASE_DIR / "vendor" / "ACE-Step-1.5"
OFFICIAL_RUNNER_SCRIPT = BASE_DIR / "official_runner.py"

MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SONGS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LORA_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_MODULES_CACHE", str(MODEL_CACHE_DIR / "hf_modules"))
os.environ.setdefault("MPLCONFIGDIR", str(MODEL_CACHE_DIR / "matplotlib"))

NANO_VLLM_DIR = BASE_DIR / "acestep" / "third_parts" / "nano-vllm"
if NANO_VLLM_DIR.exists():
    sys.path.insert(0, str(NANO_VLLM_DIR))

import torch
from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from gradio import Server

from acestep.constants import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    TASK_TYPES,
    TRACK_NAMES,
    VALID_LANGUAGES,
    VALID_TIME_SIGNATURES,
)
from acestep.handler import AceStepHandler
from lora_trainer import AceTrainingManager
from local_composer import LocalComposer
from songwriting_toolkit import MODEL_STRATEGIES, choose_song_model, normalize_album_tracks, split_terms, toolkit_payload
from studio_core import (
    ACE_STEP_LM_MODELS,
    ALLOWED_AUDIO_EXTENSIONS,
    KNOWN_ACE_STEP_MODELS,
    MAX_BATCH_SIZE,
    build_task_instruction,
    clamp_float,
    clamp_int,
    ensure_task_supported,
    get_param,
    lm_model_profiles_for_models,
    model_label,
    model_profiles_for_models,
    normalize_audio_format,
    normalize_task_type,
    normalize_track_names,
    official_fields_used,
    ordered_models,
    parse_bool,
    parse_timesteps,
    recommended_lm_model,
    recommended_song_model,
    safe_filename,
    safe_id,
    studio_ui_schema,
    supported_tasks_for_model,
)


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
    return model_label(name)


def _download_job_active(model_name: str) -> bool:
    jobs = globals().get("_model_download_jobs", {})
    job = jobs.get(model_name) if isinstance(jobs, dict) else None
    return bool(job and job.get("state") in {"queued", "running"})


def _checkpoint_dir_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else {}
            shards = sorted({str(name) for name in weight_map.values()}) if isinstance(weight_map, dict) else []
            if shards:
                return all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards)
        except Exception:
            return False
    return any(
        child.is_file() and child.stat().st_size > 0 and child.suffix in {".safetensors", ".bin", ".pt"}
        for child in path.iterdir()
    )


def _available_acestep_models() -> list[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    available = set(KNOWN_ACE_STEP_MODELS)
    if checkpoint_dir.exists():
        for child in checkpoint_dir.iterdir():
            if child.is_dir() and child.name.startswith("acestep-v15-"):
                available.add(child.name)
    return ordered_models(list(available))


def _installed_acestep_models() -> set[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    if not checkpoint_dir.exists():
        return set()
    return {
        child.name
        for child in checkpoint_dir.iterdir()
        if child.name.startswith("acestep-v15-") and not _download_job_active(child.name) and _checkpoint_dir_ready(child)
    }


def _installed_lm_models() -> set[str]:
    checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
    installed = {"auto", "none"}
    if checkpoint_dir.exists():
        installed.update(
            child.name
            for child in checkpoint_dir.iterdir()
            if child.name.startswith("acestep-5Hz-lm-") and not _download_job_active(child.name) and _checkpoint_dir_ready(child)
        )
    return installed


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


def _release_models_for_training() -> None:
    with handler_lock:
        _release_handler_state()


training_manager = AceTrainingManager(
    base_dir=BASE_DIR,
    data_dir=DATA_DIR,
    model_cache_dir=MODEL_CACHE_DIR,
    release_models=_release_models_for_training,
)


def _ensure_training_idle() -> None:
    active_job = training_manager.active_job()
    if active_job:
        raise RuntimeError(
            f"ACE-Step trainer is busy with {active_job['kind']} job {active_job['id']}. "
            "Wait for it to finish or stop it before generation."
        )


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

composer = LocalComposer()


def _language_for_generation(language: str) -> str:
    value = (language or "unknown").strip().lower()
    if value == "instrumental":
        return "unknown"
    if value in VALID_LANGUAGES:
        return value
    return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "shape"):
        return {"shape": list(value.shape)}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _audio_tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    data = tensor.cpu().float().numpy()
    if data.ndim == 2:
        data = data.T
        if data.shape[1] == 1:
            data = data[:, 0]
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak > 1e-4:
        data = (data / peak * 0.95).astype(np.float32)
    return data


def _write_audio_file(audio_dict: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = _audio_tensor_to_array(audio_dict["tensor"])
    sf.write(str(out_path), data, int(audio_dict["sample_rate"]))


def _encode_audio_file(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:audio/{path.suffix.lstrip('.') or 'wav'};base64,{encoded}"


def _song_public_url(song_id: str, filename: str) -> str:
    return f"/media/songs/{song_id}/{filename}"


def _result_public_url(result_id: str, filename: str) -> str:
    return f"/media/results/{result_id}/{filename}"


def _save_song_entry(meta: dict[str, Any], audio_source: Path) -> dict[str, Any]:
    song_id = meta.get("id") or uuid.uuid4().hex[:12]
    song_dir = SONGS_DIR / song_id
    song_dir.mkdir(parents=True, exist_ok=True)

    extension = audio_source.suffix or ".wav"
    audio_file = f"{song_id}{extension}"
    shutil.copyfile(audio_source, song_dir / audio_file)

    saved_meta = dict(meta)
    saved_meta.update(
        {
            "id": song_id,
            "audio_file": audio_file,
            "created_at": saved_meta.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }
    )
    (song_dir / "meta.json").write_text(json.dumps(_jsonable(saved_meta), indent=2), encoding="utf-8")
    entry = _decorate_song(saved_meta)
    _feed_songs.insert(0, entry)
    return entry


def _run_inference(
    prompt: str,
    lyrics: str,
    audio_duration: float,
    infer_steps: int,
    seed: int,
    language: str,
    song_model: str | None = None,
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    guidance_scale: float = 7.0,
) -> tuple[str, str]:
    _ensure_training_idle()
    use_random_seed = seed < 0
    with handler_lock:
        active_song_model = _ensure_song_model(song_model)
        is_turbo = "turbo" in active_song_model
        is_sft = "sft" in active_song_model and not is_turbo
        is_base = "base" in active_song_model and not is_turbo
        model_shift = 3.0 if is_turbo else 1.0
        if infer_steps <= 8 and not is_turbo:
            infer_steps = 50 if is_sft else (32 if is_base else infer_steps)
        result = handler.generate_music(
            captions=prompt,
            lyrics=lyrics,
            audio_duration=audio_duration,
            inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            use_random_seed=use_random_seed,
            seed=None if use_random_seed else seed,
            infer_method="ode",
            shift=model_shift,
            use_adg=False,
            vocal_language=_language_for_generation(language),
            batch_size=1,
        )

    if not result.get("success"):
        raise RuntimeError(result.get("error", "generation failed"))

    out_path = Path(tempfile.mkdtemp()) / "output.wav"
    _write_audio_file(result["audios"][0], out_path)
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
_result_extra_cache: dict[str, dict[str, Any]] = {}
_model_download_jobs: dict[str, dict[str, Any]] = {}
_model_download_lock = threading.Lock()
_model_download_runner_lock = threading.Lock()


class ModelDownloadStarted(RuntimeError):
    def __init__(self, model_name: str, job: dict[str, Any], message: str):
        super().__init__(message)
        self.model_name = model_name
        self.job = job
        self.message = message


def _downloadable_model_names() -> set[str]:
    return set(KNOWN_ACE_STEP_MODELS) | {name for name in ACE_STEP_LM_MODELS if name not in {"auto", "none"}}


def _is_model_installed(model_name: str, ignore_active_job: bool = False) -> bool:
    if not ignore_active_job and _download_job_active(model_name):
        return False
    checkpoint_path = MODEL_CACHE_DIR / "checkpoints" / model_name
    if model_name.startswith("acestep-v15-"):
        return _checkpoint_dir_ready(checkpoint_path)
    if model_name.startswith("acestep-5Hz-lm-"):
        return model_name in {"auto", "none"} or _checkpoint_dir_ready(checkpoint_path)
    return False


def _model_download_job(model_name: str) -> dict[str, Any]:
    job = _model_download_jobs.get(model_name)
    if job:
        return dict(job)
    return {
        "id": "",
        "model_name": model_name,
        "state": "installed" if _is_model_installed(model_name) else "missing",
        "message": "Already installed" if _is_model_installed(model_name) else "Not installed",
        "started_at": None,
        "finished_at": None,
        "error": "",
    }


def _set_model_download_job(model_name: str, **updates: Any) -> dict[str, Any]:
    with _model_download_lock:
        job = _model_download_jobs.setdefault(
            model_name,
            {
                "id": uuid.uuid4().hex[:12],
                "model_name": model_name,
                "state": "queued",
                "message": "Queued",
                "started_at": None,
                "finished_at": None,
                "error": "",
            },
        )
        job.update(_jsonable(updates))
        return dict(job)


def _download_model_worker(model_name: str) -> None:
    with _model_download_runner_lock:
        _set_model_download_job(
            model_name,
            state="running",
            message=f"Downloading {model_name}...",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            error="",
        )
        try:
            checkpoint_dir = MODEL_CACHE_DIR / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            handler._ensure_model_downloaded(model_name, str(checkpoint_dir))
            if model_name.startswith("acestep-v15-") and model_name != "acestep-v15-turbo":
                if not (checkpoint_dir / "acestep-v15-turbo").exists():
                    _set_model_download_job(model_name, message="Downloading shared ACE-Step components...")
                    handler._ensure_model_downloaded("acestep-v15-turbo", str(checkpoint_dir))
            if not _is_model_installed(model_name, ignore_active_job=True):
                raise RuntimeError(f"{model_name} download finished but the checkpoint folder was not found.")
            _set_model_download_job(
                model_name,
                state="succeeded",
                message=f"{model_name} installed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="",
            )
        except Exception as exc:
            _set_model_download_job(
                model_name,
                state="failed",
                message=f"{model_name} download failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )


def _start_model_download(model_name: str) -> dict[str, Any]:
    model_name = str(model_name or "").strip()
    if model_name not in _downloadable_model_names():
        raise ValueError(f"{model_name or 'model'} is not a known downloadable ACE-Step model.")
    if _is_model_installed(model_name):
        return _set_model_download_job(
            model_name,
            state="succeeded",
            message=f"{model_name} already installed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error="",
        )
    existing = _model_download_jobs.get(model_name)
    if existing and existing.get("state") in {"queued", "running"}:
        return dict(existing)
    job = _set_model_download_job(
        model_name,
        id=uuid.uuid4().hex[:12],
        state="queued",
        message=f"Queued download for {model_name}",
        started_at=None,
        finished_at=None,
        error="",
    )
    thread = threading.Thread(target=_download_model_worker, args=(model_name,), daemon=True)
    thread.start()
    return job


def _start_model_download_or_raise(model_name: str, context: str = "generation") -> None:
    job = _start_model_download(model_name)
    raise ModelDownloadStarted(
        model_name,
        job,
        f"{model_name} is not installed yet. AceJAM started the download for {context}. "
        "Wait until the model is installed, then press Generate again.",
    )


def _download_started_payload(model_name: str, job: dict[str, Any], logs: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    message = (
        f"{model_name} is not installed yet. AceJAM started downloading it. "
        "Generate will be available when the download finishes."
    )
    payload = {
        "success": False,
        "download_started": True,
        "download_model": model_name,
        "download_job": _jsonable(job),
        "message": message,
        "error": "",
        "logs": list(logs or []) + [message],
    }
    payload.update(_jsonable(extra))
    return payload


def _album_download_candidate(model_info: dict[str, Any], album_options: dict[str, Any]) -> str:
    requested = str(album_options.get("requested_song_model") or "").strip()
    if requested and requested != "auto" and requested in _downloadable_model_names():
        return requested
    model = str(model_info.get("model") or "").strip()
    if model in _downloadable_model_names():
        return model
    strategy = str(album_options.get("song_model_strategy") or "best_installed")
    for candidate in MODEL_STRATEGIES.get(strategy, MODEL_STRATEGIES["best_installed"]).get("order", []):
        if candidate in _downloadable_model_names():
            return candidate
    return ""


def _resolve_child(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    target = root.joinpath(*parts).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise HTTPException(status_code=404, detail="File not found")
    return target


def _resolve_upload_file(upload_id: str | None) -> Path | None:
    if not upload_id:
        return None
    upload_dir = _resolve_child(UPLOADS_DIR, safe_id(upload_id))
    if not upload_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload not found")
    for item in upload_dir.iterdir():
        if item.is_file() and item.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
            return item
    raise HTTPException(status_code=404, detail="Upload has no audio file")


def _result_meta_path(result_id: str) -> Path:
    return _resolve_child(RESULTS_DIR, safe_id(result_id), "result.json")


def _load_result_meta(result_id: str) -> dict[str, Any]:
    meta_path = _result_meta_path(result_id)
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_result_audio(result_id: str | None, audio_id: str | None = None) -> Path | None:
    if not result_id:
        return None
    meta = _load_result_meta(result_id)
    selected = None
    for audio in meta.get("audios", []):
        if audio_id and audio.get("id") == audio_id:
            selected = audio
            break
        if selected is None:
            selected = audio
    if not selected:
        raise HTTPException(status_code=404, detail="Result has no audio")
    return _resolve_child(RESULTS_DIR, safe_id(result_id), selected["filename"])


def _resolve_audio_reference(payload: dict[str, Any], upload_key: str, result_key: str) -> Path | None:
    upload_path = _resolve_upload_file(payload.get(upload_key))
    if upload_path is not None:
        return upload_path
    return _resolve_result_audio(payload.get(result_key), payload.get(f"{result_key}_audio_id"))


def _model_capabilities() -> dict[str, Any]:
    installed = _installed_acestep_models()
    return {
        model: {
            "label": _song_model_label(model),
            "tasks": supported_tasks_for_model(model),
            "installed": model in installed,
        }
        for model in _available_acestep_models()
    }


def _official_runner_status() -> dict[str, Any]:
    missing = []
    if not OFFICIAL_ACE_STEP_DIR.exists():
        missing.append("app/vendor/ACE-Step-1.5")
    if not OFFICIAL_RUNNER_SCRIPT.exists():
        missing.append("app/official_runner.py")
    return {
        "available": not missing,
        "vendor_path": str(OFFICIAL_ACE_STEP_DIR),
        "runner_path": str(OFFICIAL_RUNNER_SCRIPT),
        "missing": missing,
        "routing_note": "Used when Custom enables official-only ACE-Step 1.5 controls.",
    }


def _songwriting_toolkit_payload() -> dict[str, Any]:
    return toolkit_payload(_installed_acestep_models())


def _json_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return split_terms(stripped)
    return [value]


def _album_options_from_payload(payload: dict[str, Any], song_model: str = "auto") -> dict[str, Any]:
    strategy = str(payload.get("song_model_strategy") or "best_installed")
    requested_song_model = song_model if strategy == "selected" else "auto"
    return {
        "requested_song_model": requested_song_model,
        "song_model_strategy": strategy,
        "quality_target": str(payload.get("quality_target") or "hit"),
        "tag_packs": _json_list(payload.get("tag_packs")),
        "custom_tags": payload.get("custom_tags") or "",
        "negative_tags": payload.get("negative_tags") or "",
        "lyric_density": str(payload.get("lyric_density") or "dense"),
        "rhyme_density": clamp_float(payload.get("rhyme_density"), 0.8, 0.0, 1.0),
        "metaphor_density": clamp_float(payload.get("metaphor_density"), 0.7, 0.0, 1.0),
        "hook_intensity": clamp_float(payload.get("hook_intensity"), 0.85, 0.0, 1.0),
        "structure_preset": str(payload.get("structure_preset") or "auto"),
        "bpm_strategy": str(payload.get("bpm_strategy") or "varied"),
        "key_strategy": str(payload.get("key_strategy") or "related"),
        "inspiration_queries": payload.get("inspiration_queries") or "",
        "use_web_inspiration": parse_bool(payload.get("use_web_inspiration"), False),
        "track_variants": clamp_int(payload.get("track_variants"), 1, 1, MAX_BATCH_SIZE),
        "installed_models": sorted(_installed_acestep_models()),
        "global_caption": str(payload.get("global_caption") or ""),
    }


def _merge_song_album_metadata(song_id: str, extra: dict[str, Any]) -> None:
    if not song_id:
        return
    song_dir = SONGS_DIR / safe_id(song_id)
    meta_path = song_dir / "meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(_jsonable(extra))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for index, song in enumerate(_feed_songs):
        if song.get("id") == song_id:
            _feed_songs[index] = _decorate_song(meta)
            break


def _parse_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    task_type = normalize_task_type(payload.get("task_type"))
    song_model = _normalize_song_model(get_param(payload, "song_model", payload.get("song_model")))
    ensure_task_supported(song_model, task_type)
    if song_model not in _installed_acestep_models():
        if song_model in _downloadable_model_names():
            _start_model_download_or_raise(song_model, context=f"{task_type} generation")
        raise ValueError(f"{song_model} is not installed and is not in the known ACE-Step download list.")
    batch_size = clamp_int(payload.get("batch_size"), 1, 1, MAX_BATCH_SIZE)
    duration = clamp_float(get_param(payload, "duration"), 60.0, DURATION_MIN, DURATION_MAX)
    is_turbo = "turbo" in song_model
    is_sft = "sft" in song_model and not is_turbo
    is_base = "base" in song_model and not is_turbo
    raw_steps = payload.get("inference_steps", payload.get("infer_step"))
    if raw_steps in [None, "", "auto"]:
        default_steps = 8 if is_turbo else (50 if is_sft else (32 if is_base else 8))
    else:
        default_steps = int(raw_steps)
    inference_steps = clamp_int(default_steps, default_steps, 1, 200)
    if is_turbo and inference_steps > 20:
        inference_steps = min(inference_steps, 20)

    bpm_value = payload.get("bpm")
    bpm = None if bpm_value in [None, "", "auto", "Auto"] else clamp_int(bpm_value, 120, BPM_MIN, BPM_MAX)
    time_signature = str(get_param(payload, "time_signature", "") or "").strip()
    if time_signature:
        try:
            if int(float(time_signature)) not in VALID_TIME_SIGNATURES:
                time_signature = ""
        except ValueError:
            time_signature = ""

    official_used = official_fields_used(payload)
    use_official = bool(official_used)
    requested_format = str(payload.get("audio_format") or "wav").strip().lower().lstrip(".")
    if use_official and requested_format == "ogg":
        raise ValueError("OGG is only available in the fast AceJAM runner. Use wav/flac/mp3/opus/aac/wav32 with official ACE-Step controls.")
    vocal_language = _language_for_generation(str(get_param(payload, "vocal_language", "unknown") or "unknown"))
    track_names = normalize_track_names(payload.get("track_names") or payload.get("track_name"))
    instruction = str(payload.get("instruction") or "").strip() or build_task_instruction(task_type, track_names)

    if task_type in {"cover", "repaint", "extract", "lego", "complete"}:
        has_source = bool(payload.get("src_audio_id") or payload.get("src_result_id") or payload.get("audio_code_string"))
        if not has_source:
            raise ValueError(f"{task_type} requires source audio, a source result, or audio codes")
    if task_type in {"extract", "lego"} and not track_names:
        raise ValueError(f"{task_type} requires a track name")
    if task_type == "complete" and not track_names:
        raise ValueError("complete requires one or more track names")

    return {
        "task_type": task_type,
        "caption": str(payload.get("caption") or payload.get("prompt") or ""),
        "global_caption": str(payload.get("global_caption") or ""),
        "lyrics": str(payload.get("lyrics") or ""),
        "instrumental": parse_bool(payload.get("instrumental"), False),
        "duration": duration,
        "bpm": bpm,
        "key_scale": str(get_param(payload, "key_scale", "") or "").strip(),
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "batch_size": batch_size,
        "seed": str(payload.get("seeds") or payload.get("seed") or "-1"),
        "song_model": song_model,
        "ace_lm_model": str(payload.get("ace_lm_model") or "auto").strip() or "auto",
        "reference_audio": _resolve_audio_reference(payload, "reference_audio_id", "reference_result_id"),
        "src_audio": _resolve_audio_reference(payload, "src_audio_id", "src_result_id"),
        "audio_code_string": str(get_param(payload, "audio_code_string", "") or ""),
        "repainting_start": clamp_float(payload.get("repainting_start"), 0.0, -DURATION_MAX, DURATION_MAX),
        "repainting_end": None if payload.get("repainting_end") in [None, "", "end"] else clamp_float(payload.get("repainting_end"), -1.0, -1.0, DURATION_MAX),
        "instruction": instruction,
        "audio_cover_strength": clamp_float(get_param(payload, "audio_cover_strength", 1.0), 1.0, 0.0, 1.0),
        "cover_noise_strength": clamp_float(payload.get("cover_noise_strength"), 0.0, 0.0, 1.0),
        "inference_steps": inference_steps,
        "guidance_scale": clamp_float(payload.get("guidance_scale"), 7.0, 1.0, 15.0),
        "shift": clamp_float(payload.get("shift"), 3.0 if "turbo" in song_model else 1.0, 1.0, 5.0),
        "infer_method": "sde" if str(payload.get("infer_method")).lower() == "sde" else "ode",
        "sampler_mode": "heun" if str(payload.get("sampler_mode")).lower() == "heun" else "euler",
        "velocity_norm_threshold": clamp_float(payload.get("velocity_norm_threshold"), 0.0, 0.0, 20.0),
        "velocity_ema_factor": clamp_float(payload.get("velocity_ema_factor"), 0.0, 0.0, 1.0),
        "use_adg": parse_bool(payload.get("use_adg"), False),
        "cfg_interval_start": clamp_float(payload.get("cfg_interval_start"), 0.0, 0.0, 1.0),
        "cfg_interval_end": clamp_float(payload.get("cfg_interval_end"), 1.0, 0.0, 1.0),
        "timesteps": parse_timesteps(payload.get("timesteps")),
        "audio_format": normalize_audio_format(payload.get("audio_format"), allow_official=use_official),
        "mp3_bitrate": str(payload.get("mp3_bitrate") or "128k").strip() or "128k",
        "mp3_sample_rate": clamp_int(payload.get("mp3_sample_rate"), 48000, 16000, 48000),
        "auto_score": parse_bool(payload.get("auto_score"), False),
        "auto_lrc": parse_bool(payload.get("auto_lrc"), False),
        "return_audio_codes": parse_bool(payload.get("return_audio_codes"), False),
        "save_to_library": parse_bool(payload.get("save_to_library"), False),
        "title": str(payload.get("title") or "").strip() or "Untitled",
        "description": str(payload.get("description") or "").strip(),
        "album_metadata": payload.get("album_metadata") if isinstance(payload.get("album_metadata"), dict) else {},
        "track_names": track_names,
        "thinking": parse_bool(payload.get("thinking"), False),
        "sample_mode": parse_bool(payload.get("sample_mode"), False),
        "sample_query": str(get_param(payload, "sample_query", "") or "").strip(),
        "use_format": parse_bool(get_param(payload, "use_format"), False),
        "lm_temperature": clamp_float(payload.get("lm_temperature"), 0.85, 0.0, 2.0),
        "lm_cfg_scale": clamp_float(payload.get("lm_cfg_scale"), 2.0, 0.0, 10.0),
        "lm_top_k": clamp_int(payload.get("lm_top_k"), 0, 0, 200),
        "lm_top_p": clamp_float(payload.get("lm_top_p"), 0.9, 0.0, 1.0),
        "lm_negative_prompt": str(payload.get("lm_negative_prompt") or "NO USER INPUT"),
        "lm_backend": str(payload.get("lm_backend") or "auto").strip().lower()
        if str(payload.get("lm_backend") or "auto").strip().lower() in {"auto", "vllm", "pt", "mlx"}
        else "auto",
        "use_cot_metas": parse_bool(payload.get("use_cot_metas"), True),
        "use_cot_caption": parse_bool(payload.get("use_cot_caption"), True),
        "use_cot_lyrics": parse_bool(payload.get("use_cot_lyrics"), False),
        "use_cot_language": parse_bool(payload.get("use_cot_language"), True),
        "allow_lm_batch": parse_bool(payload.get("allow_lm_batch"), False),
        "lm_batch_chunk_size": clamp_int(payload.get("lm_batch_chunk_size"), 8, 1, 64),
        "use_constrained_decoding": parse_bool(payload.get("use_constrained_decoding"), True),
        "constrained_decoding_debug": parse_bool(payload.get("constrained_decoding_debug"), False),
        "chunk_mask_mode": "explicit" if str(payload.get("chunk_mask_mode")).lower() == "explicit" else "auto",
        "repaint_latent_crossfade_frames": clamp_int(payload.get("repaint_latent_crossfade_frames"), 10, 0, 250),
        "repaint_wav_crossfade_sec": clamp_float(payload.get("repaint_wav_crossfade_sec"), 0.0, 0.0, 20.0),
        "repaint_mode": str(payload.get("repaint_mode") or "balanced").strip().lower()
        if str(payload.get("repaint_mode") or "balanced").strip().lower() in {"conservative", "balanced", "aggressive"}
        else "balanced",
        "repaint_strength": clamp_float(payload.get("repaint_strength"), 0.5, 0.0, 1.0),
        "enable_normalization": parse_bool(payload.get("enable_normalization"), True),
        "normalization_db": clamp_float(payload.get("normalization_db"), -1.0, -24.0, 0.0),
        "fade_in_duration": clamp_float(payload.get("fade_in_duration"), 0.0, 0.0, 20.0),
        "fade_out_duration": clamp_float(payload.get("fade_out_duration"), 0.0, 0.0, 20.0),
        "latent_shift": clamp_float(payload.get("latent_shift"), 0.0, -2.0, 2.0),
        "latent_rescale": clamp_float(payload.get("latent_rescale"), 1.0, 0.1, 3.0),
        "official_fields": official_used,
        "requires_official_runner": use_official,
    }


def _slice_batch_tensor(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > index:
        return value[index : index + 1]
    return value


def _extra_for_index(extra: dict[str, Any], index: int) -> dict[str, Any]:
    return {key: _slice_batch_tensor(value, index) for key, value in extra.items()}


def _calculate_lrc(extra: dict[str, Any], duration: float, language: str, inference_steps: int, seed: int) -> dict[str, Any]:
    required = ["pred_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents", "lyric_token_idss"]
    if any(extra.get(key) is None for key in required):
        return {"success": False, "error": "LRC tensors are unavailable for this result"}
    with handler_lock:
        return handler.get_lyric_timestamp(
            pred_latent=extra["pred_latents"],
            encoder_hidden_states=extra["encoder_hidden_states"],
            encoder_attention_mask=extra["encoder_attention_mask"],
            context_latents=extra["context_latents"],
            lyric_token_ids=extra["lyric_token_idss"],
            total_duration_seconds=duration,
            vocal_language=language,
            inference_steps=inference_steps,
            seed=seed,
        )


def _calculate_score(extra: dict[str, Any], language: str, inference_steps: int, seed: int) -> dict[str, Any]:
    required = ["pred_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents", "lyric_token_idss"]
    if any(extra.get(key) is None for key in required):
        return {"success": False, "error": "Score tensors are unavailable for this result"}
    with handler_lock:
        return handler.get_lyric_score(
            pred_latent=extra["pred_latents"],
            encoder_hidden_states=extra["encoder_hidden_states"],
            encoder_attention_mask=extra["encoder_attention_mask"],
            context_latents=extra["context_latents"],
            lyric_token_ids=extra["lyric_token_idss"],
            vocal_language=language,
            inference_steps=inference_steps,
            seed=seed,
        )


def _concrete_lm_model(requested: str) -> str | None:
    value = (requested or "auto").strip()
    if value == "none":
        return None
    installed = _installed_lm_models()
    if value == "auto":
        for candidate in ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"]:
            if candidate in installed:
                return candidate
        return None
    return value if value in installed else None


def _requires_lm(params: dict[str, Any]) -> bool:
    if params["task_type"] in {"cover", "repaint", "extract"}:
        return False
    lm_control_fields = {
        "allow_lm_batch",
        "constrained_decoding_debug",
        "lm_batch_chunk_size",
        "lm_backend",
        "lm_cfg_scale",
        "lm_negative_prompt",
        "lm_temperature",
        "lm_top_k",
        "lm_top_p",
        "use_constrained_decoding",
        "use_cot_caption",
        "use_cot_language",
        "use_cot_lyrics",
        "use_cot_metas",
    }
    return any(
        [
            params["thinking"],
            params["sample_mode"],
            bool(params["sample_query"]),
            params["use_format"],
            params["use_cot_lyrics"],
            bool(lm_control_fields.intersection(params.get("official_fields", []))),
        ]
    )


def _official_request_payload(params: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    needs_lm = _requires_lm(params)
    lm_model = _concrete_lm_model(params["ace_lm_model"]) if needs_lm else None
    if needs_lm and not lm_model:
        requested_lm = params["ace_lm_model"]
        download_target = recommended_lm_model(set()) if requested_lm == "auto" else requested_lm
        if download_target in _downloadable_model_names():
            _start_model_download_or_raise(download_target, context="official ACE-Step LM controls")
        raise RuntimeError(
            "Official ACE-Step LM controls require a locally installed 5Hz LM model. "
            "Choose an installed LM or install acestep-5Hz-lm-0.6B/1.7B/4B first."
        )
    if params["auto_lrc"] or params["auto_score"]:
        raise RuntimeError(
            "Auto score and Auto LRC need AceJAM's in-process tensor cache. "
            "Disable official-only controls or turn off Auto score/LRC for this run."
        )

    return {
        "base_dir": str(BASE_DIR),
        "vendor_dir": str(OFFICIAL_ACE_STEP_DIR),
        "model_cache_dir": str(MODEL_CACHE_DIR),
        "checkpoint_dir": str(MODEL_CACHE_DIR / "checkpoints"),
        "save_dir": str(save_dir),
        "song_model": params["song_model"],
        "lm_model": lm_model,
        "requires_lm": needs_lm,
        "params": {
            "task_type": params["task_type"],
            "instruction": params["instruction"],
            "reference_audio": str(params["reference_audio"]) if params["reference_audio"] else None,
            "src_audio": str(params["src_audio"]) if params["src_audio"] else None,
            "audio_codes": params["audio_code_string"],
            "caption": params["caption"],
            "global_caption": params["global_caption"],
            "lyrics": "[Instrumental]" if params["instrumental"] else params["lyrics"],
            "instrumental": params["instrumental"],
            "vocal_language": params["vocal_language"],
            "bpm": params["bpm"],
            "keyscale": params["key_scale"],
            "timesignature": params["time_signature"],
            "duration": params["duration"],
            "enable_normalization": params["enable_normalization"],
            "normalization_db": params["normalization_db"],
            "fade_in_duration": params["fade_in_duration"],
            "fade_out_duration": params["fade_out_duration"],
            "latent_shift": params["latent_shift"],
            "latent_rescale": params["latent_rescale"],
            "inference_steps": params["inference_steps"],
            "seed": -1,
            "guidance_scale": params["guidance_scale"],
            "use_adg": params["use_adg"],
            "cfg_interval_start": params["cfg_interval_start"],
            "cfg_interval_end": params["cfg_interval_end"],
            "shift": params["shift"],
            "infer_method": params["infer_method"],
            "sampler_mode": params["sampler_mode"],
            "velocity_norm_threshold": params["velocity_norm_threshold"],
            "velocity_ema_factor": params["velocity_ema_factor"],
            "timesteps": params["timesteps"],
            "repainting_start": params["repainting_start"],
            "repainting_end": params["repainting_end"],
            "chunk_mask_mode": params["chunk_mask_mode"],
            "repaint_latent_crossfade_frames": params["repaint_latent_crossfade_frames"],
            "repaint_wav_crossfade_sec": params["repaint_wav_crossfade_sec"],
            "repaint_mode": params["repaint_mode"],
            "repaint_strength": params["repaint_strength"],
            "audio_cover_strength": params["audio_cover_strength"],
            "cover_noise_strength": params["cover_noise_strength"],
            "thinking": params["thinking"],
            "lm_temperature": params["lm_temperature"],
            "lm_cfg_scale": params["lm_cfg_scale"],
            "lm_top_k": params["lm_top_k"],
            "lm_top_p": params["lm_top_p"],
            "lm_negative_prompt": params["lm_negative_prompt"],
            "use_cot_metas": params["use_cot_metas"],
            "use_cot_caption": params["use_cot_caption"],
            "use_cot_lyrics": params["use_cot_lyrics"],
            "use_cot_language": params["use_cot_language"],
            "use_constrained_decoding": params["use_constrained_decoding"],
            "sample_mode": params["sample_mode"],
            "sample_query": params["sample_query"],
            "use_format": params["use_format"],
        },
        "lm_backend": params["lm_backend"],
        "config": {
            "batch_size": params["batch_size"],
            "allow_lm_batch": params["allow_lm_batch"],
            "use_random_seed": params["seed"].strip() in {"", "-1"},
            "seeds": None if params["seed"].strip() in {"", "-1"} else params["seed"],
            "lm_batch_chunk_size": params["lm_batch_chunk_size"],
            "constrained_decoding_debug": params["constrained_decoding_debug"],
            "audio_format": params["audio_format"],
            "mp3_bitrate": params["mp3_bitrate"],
            "mp3_sample_rate": params["mp3_sample_rate"],
        },
    }


def _copy_official_audio(result_dir: Path, audio: dict[str, Any], index: int, requested_format: str) -> tuple[Path, str]:
    source = Path(str(audio.get("path") or ""))
    if not source.is_file():
        raise RuntimeError("Official ACE-Step runner did not return an audio file")
    ext = source.suffix.lstrip(".") or ("wav" if requested_format == "wav32" else requested_format)
    filename = f"take-{index + 1}.{ext}"
    target = result_dir / filename
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target, filename


def _run_official_generation(params: dict[str, Any]) -> dict[str, Any]:
    if not OFFICIAL_ACE_STEP_DIR.exists():
        raise RuntimeError("Official ACE-Step runner requires app/vendor/ACE-Step-1.5. Run Install/Update first.")
    if not OFFICIAL_RUNNER_SCRIPT.exists():
        raise RuntimeError("Official ACE-Step runner script is missing.")

    result_id = uuid.uuid4().hex[:12]
    result_dir = RESULTS_DIR / result_id
    official_dir = result_dir / "official"
    result_dir.mkdir(parents=True, exist_ok=True)
    official_dir.mkdir(parents=True, exist_ok=True)
    request_path = result_dir / "official_request.json"
    response_path = result_dir / "official_response.json"
    request_path.write_text(json.dumps(_official_request_payload(params, official_dir), indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OFFICIAL_ACE_STEP_DIR)
    env["HF_HOME"] = str(MODEL_CACHE_DIR / "huggingface")
    env["HF_MODULES_CACHE"] = str(MODEL_CACHE_DIR / "hf_modules")
    env["XDG_CACHE_HOME"] = str(MODEL_CACHE_DIR / "xdg")
    env["ACESTEP_DISABLE_TQDM"] = "1"

    with handler_lock:
        _release_handler_state()

    completed = subprocess.run(
        [sys.executable, str(OFFICIAL_RUNNER_SCRIPT), str(request_path), str(response_path)],
        cwd=str(OFFICIAL_ACE_STEP_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=3600,
        check=False,
    )
    (result_dir / "official_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (result_dir / "official_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout or "").splitlines()[-20:])
        raise RuntimeError(f"Official ACE-Step runner failed: {tail or completed.returncode}")
    if not response_path.is_file():
        raise RuntimeError("Official ACE-Step runner did not write a response file")

    official = json.loads(response_path.read_text(encoding="utf-8"))
    if not official.get("success"):
        raise RuntimeError(official.get("error") or "Official ACE-Step generation failed")

    audios: list[dict[str, Any]] = []
    for index, audio in enumerate(official.get("audios", [])):
        path, filename = _copy_official_audio(result_dir, audio, index, params["audio_format"])
        audio_id = f"take-{index + 1}"
        audio_params = audio.get("params") or {}
        seed_text = str(audio_params.get("seed") or params["seed"] or "-1")
        item = {
            "id": audio_id,
            "filename": filename,
            "audio_url": _result_public_url(result_id, filename),
            "download_url": _result_public_url(result_id, filename),
            "title": params["title"] if len(official.get("audios", [])) == 1 else f"{params['title']} {index + 1}",
            "seed": seed_text,
            "sample_rate": int(audio.get("sample_rate") or 48000),
        }
        if params["return_audio_codes"] and audio_params.get("audio_codes"):
            item["audio_codes"] = audio_params["audio_codes"]
        if params["save_to_library"]:
            entry = _save_song_entry(
                {
                    "title": item["title"],
                    "description": params["description"],
                    "tags": params["caption"],
                    "lyrics": "[Instrumental]" if params["instrumental"] else params["lyrics"],
                    "bpm": params["bpm"],
                    "key_scale": params["key_scale"],
                    "time_signature": params["time_signature"],
                    "language": params["vocal_language"],
                    "duration": params["duration"],
                    "task_type": params["task_type"],
                    "song_model": params["song_model"],
                    "ace_lm_model": params["ace_lm_model"],
                    "seed": seed_text,
                    "parameters": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
                    "album": _jsonable(params["album_metadata"]),
                    "album_concept": params["album_metadata"].get("album_concept"),
                    "album_id": params["album_metadata"].get("album_id"),
                    "track_number": params["album_metadata"].get("track_number"),
                    "track_variant": params["album_metadata"].get("track_variant"),
                    "result_id": result_id,
                    "runner": "official",
                },
                path,
            )
            item["song_id"] = entry["id"]
            item["library_url"] = entry["audio_url"]
        audios.append(item)

    meta = {
        "id": result_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_song_model": params["song_model"],
        "runner": "official",
        "official_features": params["official_fields"],
        "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
        "time_costs": _jsonable(official.get("time_costs", {})),
        "lm_metadata": _jsonable(official.get("lm_metadata")),
        "audios": audios,
    }
    (result_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "success": True,
        "result_id": result_id,
        "active_song_model": params["song_model"],
        "runner": "official",
        "official_features": params["official_fields"],
        "audios": audios,
    }


def _run_advanced_generation(raw_payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_training_idle()
    params = _parse_generation_payload(raw_payload)
    if params["instrumental"] and not params["lyrics"].strip():
        params["lyrics"] = "[Instrumental]"
    if params["requires_official_runner"]:
        return _run_official_generation(params)
    use_random_seed = params["seed"].strip() in {"", "-1"}
    with handler_lock:
        active_song_model = _ensure_song_model(params["song_model"])
        result = handler.generate_music(
            captions=params["caption"],
            lyrics="[Instrumental]" if params["instrumental"] else params["lyrics"],
            bpm=params["bpm"],
            key_scale=params["key_scale"],
            time_signature=params["time_signature"],
            vocal_language=params["vocal_language"],
            inference_steps=params["inference_steps"],
            guidance_scale=params["guidance_scale"],
            use_random_seed=use_random_seed,
            seed=None if use_random_seed else params["seed"],
            reference_audio=str(params["reference_audio"]) if params["reference_audio"] else None,
            audio_duration=params["duration"],
            batch_size=params["batch_size"],
            src_audio=str(params["src_audio"]) if params["src_audio"] else None,
            audio_code_string=params["audio_code_string"],
            repainting_start=params["repainting_start"],
            repainting_end=params["repainting_end"],
            instruction=params["instruction"],
            audio_cover_strength=params["audio_cover_strength"],
            task_type=params["task_type"],
            use_adg=params["use_adg"],
            cfg_interval_start=params["cfg_interval_start"],
            cfg_interval_end=params["cfg_interval_end"],
            shift=params["shift"],
            infer_method=params["infer_method"],
            timesteps=params["timesteps"],
        )

    if not result.get("success"):
        raise RuntimeError(result.get("error", "generation failed"))

    result_id = uuid.uuid4().hex[:12]
    result_dir = RESULTS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)
    extra = result.get("extra_outputs") or {}
    seed_values = [item.strip() for item in str(extra.get("seed_value") or params["seed"]).split(",")]
    audios = []

    for index, audio_dict in enumerate(result.get("audios", [])):
        audio_id = f"take-{index + 1}"
        filename = f"{audio_id}.{params['audio_format']}"
        path = result_dir / filename
        _write_audio_file(audio_dict, path)
        item_extra = _extra_for_index(extra, index)
        seed_text = seed_values[index] if index < len(seed_values) else (seed_values[0] if seed_values else "42")
        try:
            seed_int = int(seed_text)
        except (TypeError, ValueError):
            seed_int = 42

        item = {
            "id": audio_id,
            "filename": filename,
            "audio_url": _result_public_url(result_id, filename),
            "download_url": _result_public_url(result_id, filename),
            "title": params["title"] if len(result.get("audios", [])) == 1 else f"{params['title']} {index + 1}",
            "seed": seed_text,
            "sample_rate": int(audio_dict["sample_rate"]),
        }
        if params["auto_lrc"]:
            item["lrc"] = _calculate_lrc(item_extra, params["duration"], params["vocal_language"], params["inference_steps"], seed_int)
        if params["auto_score"]:
            item["score"] = _calculate_score(item_extra, params["vocal_language"], params["inference_steps"], seed_int)
        if params["return_audio_codes"]:
            with handler_lock:
                item["audio_codes"] = handler.convert_src_audio_to_codes(str(path))
        if params["save_to_library"]:
            entry = _save_song_entry(
                {
                    "title": item["title"],
                    "description": params["description"],
                    "tags": params["caption"],
                    "lyrics": params["lyrics"],
                    "bpm": params["bpm"],
                    "key_scale": params["key_scale"],
                    "time_signature": params["time_signature"],
                    "language": params["vocal_language"],
                    "duration": params["duration"],
                    "task_type": params["task_type"],
                    "song_model": active_song_model,
                    "seed": seed_text,
                    "parameters": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
                    "album": _jsonable(params["album_metadata"]),
                    "album_concept": params["album_metadata"].get("album_concept"),
                    "album_id": params["album_metadata"].get("album_id"),
                    "track_number": params["album_metadata"].get("track_number"),
                    "track_variant": params["album_metadata"].get("track_variant"),
                    "score": item.get("score"),
                    "lrc": item.get("lrc"),
                    "result_id": result_id,
                },
                path,
            )
            item["song_id"] = entry["id"]
            item["library_url"] = entry["audio_url"]
        audios.append(item)

    meta = {
        "id": result_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_song_model": active_song_model,
        "params": {k: _jsonable(v) for k, v in params.items() if k not in {"reference_audio", "src_audio"}},
        "time_costs": _jsonable(extra.get("time_costs", {})),
        "audios": audios,
    }
    (result_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _result_extra_cache[result_id] = extra
    while len(_result_extra_cache) > 8:
        _result_extra_cache.pop(next(iter(_result_extra_cache)))

    return {
        "success": True,
        "result_id": result_id,
        "active_song_model": active_song_model,
        "audios": audios,
        "params": meta["params"],
        "time_costs": meta["time_costs"],
    }


app = Server(title="AceJAM")


@app.api(name="compose", concurrency_limit=1, time_limit=120)
def compose(
    description: str,
    audio_duration: float = 60.0,
    composer_profile: str = "auto",
    instrumental: bool = False,
    ollama_model: str = "",
) -> str:
    """Compose song spec (title, tags, lyrics, etc.) without generating music."""
    try:
        composed = composer.compose(
            description=description,
            audio_duration=audio_duration,
            profile=composer_profile,
            instrumental=instrumental,
            ollama_model=ollama_model or None,
        )
        return json.dumps(composed)
    except Exception as exc:
        print(f"[compose ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise


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
        resolved_profile = composed.get("composer_profile", composer_profile)
        print(
            "[create] "
            f"profile={resolved_profile} "
            f"model={composed.get('composer_model', 'unknown')} "
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
            f"language={composed['language']} "
            f"bpm={composed['bpm']} "
            f"key_scale={composed.get('key_scale', '')} "
            f"time_signature={composed.get('time_signature', '')}"
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
            bpm=composed["bpm"],
            key_scale=composed.get("key_scale", ""),
            time_signature=composed.get("time_signature", ""),
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
            "key_scale": composed.get("key_scale", ""),
            "time_signature": composed.get("time_signature", ""),
            "language": composed["language"],
            "composer_profile": resolved_profile,
            "composer_model": composed.get("composer_model", "unknown"),
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
                "key_scale": composed.get("key_scale", ""),
                "time_signature": composed.get("time_signature", ""),
                "language": composed["language"],
                "duration": audio_duration,
                "audio_file": audio_file,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (song_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            entry = _decorate_song(meta)
            _feed_songs.insert(0, entry)
            result["song_id"] = song_id
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
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    lora_name_or_path: str = "",
    lora_weight: float = 0.8,
) -> str:
    del lora_weight
    try:
        if lora_name_or_path.strip():
            with handler_lock:
                status_msg = handler.load_lora(lora_name_or_path.strip())
                if status_msg.startswith("❌"):
                    raise RuntimeError(status_msg)
        wav_path, _ = _run_inference(
            prompt, lyrics, audio_duration, infer_step, seed, "en",
            song_model=song_model,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            guidance_scale=guidance_scale,
        )
        encoded = base64.b64encode(Path(wav_path).read_bytes()).decode()
        return f"data:audio/wav;base64,{encoded}"
    except Exception as exc:
        print(f"[generate ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
    finally:
        _cleanup_accelerator_memory()


@app.api(name="delete_song", concurrency_limit=4)
def delete_song(song_id: str) -> str:
    import shutil
    song_dir = SONGS_DIR / song_id
    if not song_dir.exists():
        return json.dumps({"success": False, "error": "not found"})
    # Remove from in-memory feed
    _feed_songs[:] = [s for s in _feed_songs if s.get("id") != song_id]
    # Remove from disk
    shutil.rmtree(song_dir, ignore_errors=True)
    print(f"[delete] removed song {song_id}")
    return json.dumps({"success": True})


@app.api(name="community", concurrency_limit=4)
def community() -> str:
    return json.dumps(_feed_songs[:50])


@app.api(name="config", concurrency_limit=8)
def config() -> str:
    available_models = _available_acestep_models()
    installed_models = _installed_acestep_models()
    installed_lms = _installed_lm_models()
    model_profiles = model_profiles_for_models(available_models, installed_models)
    lm_profiles = lm_model_profiles_for_models(ACE_STEP_LM_MODELS, installed_lms)
    return json.dumps(
        {
            "active_song_model": ACTIVE_ACE_STEP_MODEL,
            "default_song_model": _default_acestep_checkpoint(),
            "recommended_song_model": recommended_song_model(installed_models),
            "recommended_lm_model": recommended_lm_model(installed_lms),
            "available_song_models": available_models,
            "installed_song_models": sorted(installed_models),
            "installed_lm_models": sorted(installed_lms),
            "model_labels": {name: model_profiles[name]["label"] for name in available_models},
            "model_profiles": model_profiles,
            "lm_model_profiles": lm_profiles,
            "model_capabilities": _model_capabilities(),
            "model_downloads": {name: _model_download_job(name) for name in sorted(_downloadable_model_names())},
            "official_runner": _official_runner_status(),
            "ui_schema": studio_ui_schema(),
            "songwriting_toolkit": _songwriting_toolkit_payload(),
            "task_types": TASK_TYPES,
            "track_names": TRACK_NAMES,
            "valid_languages": VALID_LANGUAGES,
            "valid_time_signatures": VALID_TIME_SIGNATURES,
            "lm_models": ACE_STEP_LM_MODELS,
            "lora": handler.get_lora_status(),
            "trainer": training_manager.status(),
        }
    )


@app.api(name="ollama_models", concurrency_limit=8)
def ollama_models() -> str:
    """List available Ollama models using the official ollama library."""
    try:
        import ollama
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        client = ollama.Client(host=ollama_url)
        response = client.list()
        models = [
            {"name": m.model, "size_gb": round(m.size / 1e9, 1)}
            for m in response.models
        ]
        return json.dumps({"models": [m["name"] for m in models], "details": models})
    except Exception as exc:
        print(f"[ollama_models ERROR] {exc}")
        return json.dumps({"models": [], "error": str(exc)})


@app.api(name="generate_album", concurrency_limit=1, time_limit=3600)
def generate_album(
    concept: str,
    num_tracks: int = 5,
    track_duration: float = 180.0,
    ollama_model: str = "llama3.2",
    language: str = "en",
    song_model: str = "auto",
    embedding_model: str = "nomic-embed-text",
    ace_lm_model: str = "auto",
    request_json: str = "",
) -> str:
    """Plan album with tools/CrewAI, then generate through the advanced engine."""
    logs: list[str] = []
    try:
        request_payload = json.loads(request_json or "{}")
        album_options = _album_options_from_payload(request_payload, song_model=song_model)
        planned_tracks = _json_list(request_payload.get("tracks") or request_payload.get("planned_tracks"))
        model_info = choose_song_model(
            _installed_acestep_models(),
            str(album_options.get("song_model_strategy") or "best_installed"),
            str(album_options.get("requested_song_model") or "auto"),
        )
        if not model_info.get("ok"):
            download_model = _album_download_candidate(model_info, album_options)
            if download_model:
                job = _start_model_download(download_model)
                logs.append(str(model_info.get("error") or "Album model is not installed."))
                return json.dumps(
                    _download_started_payload(
                        download_model,
                        job,
                        logs,
                        tracks=planned_tracks,
                        album_model_strategy=album_options.get("song_model_strategy"),
                    )
                )
            raise RuntimeError(str(model_info.get("error") or "No installed album model available"))

        from album_crew import plan_album as _plan_album

        logs.append("Phase 1: Planning album with Hit Album Agent tools...")
        result = _plan_album(
            concept=concept,
            num_tracks=num_tracks,
            track_duration=track_duration,
            ollama_model=ollama_model,
            language=language,
            embedding_model=embedding_model,
            options=album_options,
            use_crewai=not planned_tracks,
            input_tracks=planned_tracks if planned_tracks else None,
        )
        tracks = result.get("tracks", [])
        logs.extend(result.get("logs", []))

        if not result.get("success", True) or not tracks or "error" in tracks[0]:
            logs.append("ERROR: Album planning failed")
            return json.dumps({"tracks": tracks, "logs": logs, "success": False, "error": result.get("error") or "Planning failed"})

        logs.append(f"Phase 1 complete: {len(tracks)} tracks planned")
        logs.append(f"ACE-Step LM profile: {ace_lm_model}")
        logs.append(f"Model strategy: {album_options.get('song_model_strategy')} -> {model_info.get('model')}")
        logs.append("---")
        logs.append("Phase 2: Generating music for each track with /generate_advanced...")

        album_id = uuid.uuid4().hex[:12]
        generated_audios: list[dict[str, Any]] = []
        variants = clamp_int(album_options.get("track_variants"), 1, 1, MAX_BATCH_SIZE)
        for i, track in enumerate(tracks):
            track_title = track.get("title", f"Track {i+1}")
            track_model = str(track.get("song_model") or model_info.get("model") or song_model)
            if track_model not in _installed_acestep_models():
                if track_model in _downloadable_model_names():
                    job = _start_model_download(track_model)
                    logs.append(f"Track {i+1} needs {track_model}. Starting download instead of failing.")
                    return json.dumps(
                        _download_started_payload(
                            track_model,
                            job,
                            logs,
                            tracks=tracks,
                            album_model_strategy=album_options.get("song_model_strategy"),
                        )
                    )
                raise RuntimeError(f"{track_model} is not installed and is not in the known ACE-Step download list.")
            logs.append(f"Generating track {i+1}/{len(tracks)}: {track_title} ({variants} variant{'s' if variants != 1 else ''})...")
            print(f"[generate_album] Generating track {i+1}/{len(tracks)}: {track_title}")

            try:
                _cleanup_accelerator_memory()
                generation_payload = {
                    "task_type": "text2music",
                    "title": track_title,
                    "description": track.get("description", ""),
                    "caption": track.get("tags", ""),
                    "lyrics": track.get("lyrics", ""),
                    "duration": track.get("duration", track_duration),
                    "bpm": track.get("bpm"),
                    "key_scale": track.get("key_scale", ""),
                    "time_signature": track.get("time_signature", "4"),
                    "vocal_language": track.get("language", language),
                    "batch_size": variants,
                    "seed": str(track.get("seed") or request_payload.get("seed") or request_payload.get("seeds") or "-1"),
                    "song_model": track_model,
                    "ace_lm_model": ace_lm_model,
                    "inference_steps": clamp_int(request_payload.get("inference_steps"), 8 if "turbo" in track_model else (50 if "sft" in track_model else 32), 1, 200),
                    "guidance_scale": clamp_float(request_payload.get("guidance_scale"), 7.0, 1.0, 15.0),
                    "shift": clamp_float(request_payload.get("shift"), 3.0 if "turbo" in track_model else 1.0, 1.0, 5.0),
                    "infer_method": str(request_payload.get("infer_method") or "ode"),
                    "use_adg": parse_bool(request_payload.get("use_adg"), False),
                    "cfg_interval_start": clamp_float(request_payload.get("cfg_interval_start"), 0.0, 0.0, 1.0),
                    "cfg_interval_end": clamp_float(request_payload.get("cfg_interval_end"), 1.0, 0.0, 1.0),
                    "audio_format": str(request_payload.get("audio_format") or "wav"),
                    "auto_score": parse_bool(request_payload.get("auto_score"), False),
                    "auto_lrc": parse_bool(request_payload.get("auto_lrc"), False),
                    "return_audio_codes": parse_bool(request_payload.get("return_audio_codes"), False),
                    "save_to_library": parse_bool(request_payload.get("save_to_library"), True),
                    "album_metadata": {
                        "album_id": album_id,
                        "album_concept": concept,
                        "album_options": _jsonable(album_options),
                        "album_toolkit_report": _jsonable(result.get("toolkit_report", {})),
                        "track_number": track.get("track_number", i + 1),
                        "track_variant": "batch",
                        "tool_report": _jsonable(track.get("tool_report", {})),
                        "tag_list": track.get("tag_list", []),
                    },
                }
                generation_result = _run_advanced_generation(generation_payload)
                if not generation_result.get("success"):
                    raise RuntimeError(generation_result.get("error") or "Track generation failed")

                track["result_id"] = generation_result.get("result_id")
                track["active_song_model"] = generation_result.get("active_song_model")
                track["audios"] = generation_result.get("audios", [])
                if track["audios"]:
                    first_audio = track["audios"][0]
                    track["song_id"] = first_audio.get("song_id")
                    track["audio_url"] = first_audio.get("audio_url") or first_audio.get("library_url")
                for audio_index, audio in enumerate(track["audios"]):
                    if audio.get("song_id"):
                        _merge_song_album_metadata(
                            audio["song_id"],
                            {
                                "album_concept": concept,
                                "album_id": album_id,
                                "track_number": track.get("track_number", i + 1),
                                "track_variant": audio_index + 1,
                                "album_toolkit_report": result.get("toolkit_report", {}),
                                "tool_report": track.get("tool_report", {}),
                                "tag_list": track.get("tag_list", []),
                            },
                        )
                generated_audios.extend(track["audios"])
                track["generated"] = True

                logs.append(f"  Track {i+1} done: {track_title}")
                print(f"[generate_album] Track {i+1} generated: {track.get('result_id')}")

            except Exception as track_exc:
                track["generated"] = False
                track["error"] = str(track_exc)
                logs.append(f"  Track {i+1} FAILED: {track_exc}")
                print(f"[generate_album] Track {i+1} failed: {track_exc}")
            finally:
                _cleanup_accelerator_memory()

        generated_count = sum(1 for t in tracks if t.get("generated"))
        logs.append("---")
        logs.append(f"Album complete: {generated_count}/{len(tracks)} tracks generated!")

        return json.dumps({
            "tracks": tracks,
            "audios": generated_audios,
            "album_id": album_id,
            "toolkit": result.get("toolkit", _songwriting_toolkit_payload()),
            "toolkit_report": result.get("toolkit_report", {}),
            "logs": logs,
            "success": True,
        })
    except ModelDownloadStarted as exc:
        print(f"[generate_album DOWNLOAD] {exc.message}")
        logs.append(exc.message)
        return json.dumps(_download_started_payload(exc.model_name, exc.job, logs))
    except Exception as exc:
        print(f"[generate_album ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        logs.append(f"ERROR: {exc}")
        return json.dumps({"tracks": [], "logs": logs, "success": False, "error": str(exc)})


@app.api(name="generate_advanced", concurrency_limit=1, time_limit=3600)
def generate_advanced(request_json: str) -> str:
    try:
        payload = json.loads(request_json or "{}")
        return json.dumps(_run_advanced_generation(payload))
    except ModelDownloadStarted as exc:
        print(f"[generate_advanced DOWNLOAD] {exc.message}")
        return json.dumps(_download_started_payload(exc.model_name, exc.job))
    except Exception as exc:
        print(f"[generate_advanced ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return json.dumps({"success": False, "error": str(exc)})
    finally:
        _cleanup_accelerator_memory()


@app.post("/api/config")
async def api_config():
    return JSONResponse(json.loads(config()))


@app.get("/api/config")
async def api_config_get():
    return JSONResponse(json.loads(config()))


@app.get("/api/songwriting_toolkit")
async def api_songwriting_toolkit():
    return JSONResponse(_songwriting_toolkit_payload())


@app.get("/api/models/downloads")
async def api_model_downloads():
    return JSONResponse({name: _model_download_job(name) for name in sorted(_downloadable_model_names())})


@app.get("/api/models/download/{model_name}")
async def api_model_download_status(model_name: str):
    if model_name not in _downloadable_model_names():
        return JSONResponse({"success": False, "error": f"{model_name} is not downloadable"}, status_code=404)
    return JSONResponse({"success": True, "job": _model_download_job(model_name), "installed": _is_model_installed(model_name)})


@app.post("/api/models/download")
async def api_model_download(request: Request):
    try:
        body = await request.json()
        model_name = str(body.get("model_name") or body.get("model") or "").strip()
        job = _start_model_download(model_name)
        return JSONResponse({"success": True, "job": job, "installed": _is_model_installed(model_name)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/album/plan")
async def api_album_plan(request: Request):
    body = await request.json()
    concept = str(body.get("concept") or "")
    num_tracks = int(body.get("num_tracks") or 5)
    track_duration = float(body.get("track_duration") or body.get("duration") or 180.0)
    language = str(body.get("language") or "en")
    song_model = str(body.get("song_model") or "auto")
    options = _album_options_from_payload(body, song_model=song_model)
    from album_crew import plan_album as _plan_album

    result = _plan_album(
        concept=concept,
        num_tracks=num_tracks,
        track_duration=track_duration,
        ollama_model=str(body.get("ollama_model") or "llama3.2"),
        language=language,
        embedding_model=str(body.get("embedding_model") or "nomic-embed-text"),
        options=options,
        use_crewai=not parse_bool(body.get("toolbelt_only"), False),
        input_tracks=_json_list(body.get("tracks")) or None,
    )
    return JSONResponse(result, status_code=200 if result.get("success", True) else 400)


@app.get("/api/community")
async def api_community():
    return JSONResponse(_feed_songs[:100])


@app.get("/api/ollama_models")
async def api_ollama_models():
    return JSONResponse(json.loads(ollama_models()))


@app.post("/api/delete_song")
async def api_delete_song(request: Request):
    body = await request.json()
    return JSONResponse(json.loads(delete_song(str(body.get("song_id") or ""))))


@app.post("/api/compose")
async def api_compose(request: Request):
    body = await request.json()
    raw = compose(
        description=str(body.get("description") or ""),
        audio_duration=float(body.get("audio_duration") or body.get("duration") or 60.0),
        composer_profile=str(body.get("composer_profile") or "auto"),
        instrumental=parse_bool(body.get("instrumental"), False),
        ollama_model=str(body.get("ollama_model") or ""),
    )
    return JSONResponse(json.loads(raw))


@app.post("/api/create_sample")
async def api_create_sample(request: Request):
    body = await request.json()
    raw = compose(
        description=str(body.get("query") or body.get("description") or body.get("caption") or ""),
        audio_duration=float(body.get("duration") or 60.0),
        composer_profile="auto",
        instrumental=parse_bool(body.get("instrumental"), False),
        ollama_model=str(body.get("ollama_model") or ""),
    )
    data = json.loads(raw)
    return JSONResponse({"success": True, **data})


@app.post("/api/format_sample")
async def api_format_sample(request: Request):
    body = await request.json()
    raw = compose(
        description=str(body.get("caption") or body.get("description") or "custom song"),
        audio_duration=float(body.get("duration") or 60.0),
        composer_profile="auto",
        instrumental=parse_bool(body.get("instrumental"), False),
        ollama_model=str(body.get("ollama_model") or ""),
    )
    data = json.loads(raw)
    if str(body.get("lyrics") or "").strip():
        data["lyrics"] = str(body["lyrics"]).strip()
    return JSONResponse({"success": True, **data})


@app.post("/api/generate_advanced")
async def api_generate_advanced(request: Request):
    try:
        payload = await request.json()
        return JSONResponse(_run_advanced_generation(payload))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[api_generate_advanced ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)
    finally:
        _cleanup_accelerator_memory()


@app.post("/api/uploads")
async def api_upload_audio(file: UploadFile = File(...)):
    original_name = file.filename or "audio.wav"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        return JSONResponse({"success": False, "error": f"Unsupported audio file: {suffix}"}, status_code=400)
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOADS_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_filename(original_name)}{suffix}"
    target = upload_dir / filename
    target.write_bytes(await file.read())
    return JSONResponse({"success": True, "id": upload_id, "filename": filename, "url": f"/api/uploads/{upload_id}"})


@app.get("/api/uploads/{upload_id}")
async def api_get_upload(upload_id: str):
    path = _resolve_upload_file(upload_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return FileResponse(path)


@app.get("/api/results/{result_id}")
async def api_get_result(result_id: str):
    return JSONResponse(_load_result_meta(result_id))


@app.post("/api/audio-codes")
async def api_audio_codes(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        audio_path = _resolve_upload_file(body.get("upload_id")) or _resolve_result_audio(body.get("result_id"), body.get("audio_id"))
        if audio_path is None:
            return JSONResponse({"success": False, "error": "No upload_id or result_id supplied"}, status_code=400)
        with handler_lock:
            _ensure_song_model(body.get("song_model"))
            codes = handler.convert_src_audio_to_codes(str(audio_path))
        return JSONResponse({"success": True, "audio_codes": codes})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


def _update_result_item(result_id: str, audio_id: str, field: str, value: Any) -> None:
    meta_path = _result_meta_path(result_id)
    meta = _load_result_meta(result_id)
    for item in meta.get("audios", []):
        if item.get("id") == audio_id:
            item[field] = _jsonable(value)
            break
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@app.post("/api/lrc")
async def api_lrc(request: Request):
    body = await request.json()
    result_id = safe_id(str(body.get("result_id") or ""))
    audio_id = str(body.get("audio_id") or "take-1")
    meta = _load_result_meta(result_id)
    extra = _result_extra_cache.get(result_id)
    if not extra:
        return JSONResponse({"success": False, "error": "LRC cache expired; regenerate with Auto LRC enabled"}, status_code=400)
    index = max(0, int(audio_id.split("-")[-1]) - 1) if "-" in audio_id else 0
    params = meta.get("params", {})
    seed = int((meta.get("audios", [{}])[index].get("seed") or 42))
    lrc = _calculate_lrc(_extra_for_index(extra, index), float(params.get("duration") or 60), str(params.get("vocal_language") or "unknown"), int(params.get("inference_steps") or 8), seed)
    _update_result_item(result_id, audio_id, "lrc", lrc)
    return JSONResponse(lrc)


@app.post("/api/score")
async def api_score(request: Request):
    body = await request.json()
    result_id = safe_id(str(body.get("result_id") or ""))
    audio_id = str(body.get("audio_id") or "take-1")
    meta = _load_result_meta(result_id)
    extra = _result_extra_cache.get(result_id)
    if not extra:
        return JSONResponse({"success": False, "error": "Score cache expired; regenerate with Auto Score enabled"}, status_code=400)
    index = max(0, int(audio_id.split("-")[-1]) - 1) if "-" in audio_id else 0
    params = meta.get("params", {})
    seed = int((meta.get("audios", [{}])[index].get("seed") or 42))
    score = _calculate_score(_extra_for_index(extra, index), str(params.get("vocal_language") or "unknown"), int(params.get("inference_steps") or 8), seed)
    _update_result_item(result_id, audio_id, "score", score)
    return JSONResponse(score)


@app.get("/api/lora/status")
async def api_lora_status():
    return JSONResponse(
        {
            "success": True,
            **handler.get_lora_status(),
            "trainer": training_manager.status(),
            "adapters": training_manager.list_adapters(),
        }
    )


@app.post("/api/lora/load")
async def api_lora_load(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        with handler_lock:
            status_msg = handler.load_lora(str(body.get("path") or ""))
        return JSONResponse({"success": not status_msg.startswith("❌"), "status": status_msg, **handler.get_lora_status()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/unload")
async def api_lora_unload():
    try:
        _ensure_training_idle()
        with handler_lock:
            status_msg = handler.unload_lora()
        return JSONResponse({"success": not status_msg.startswith("❌"), "status": status_msg, **handler.get_lora_status()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/use")
async def api_lora_use(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        with handler_lock:
            status_msg = handler.set_use_lora(parse_bool(body.get("use"), True))
        return JSONResponse({"success": not status_msg.startswith("❌"), "status": status_msg, **handler.get_lora_status()})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/dataset/scan")
async def api_lora_dataset_scan(request: Request):
    try:
        body = await request.json()
        data = training_manager.scan_dataset(Path(str(body.get("path") or "")))
        return JSONResponse({"success": True, **data})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/dataset/autolabel")
async def api_lora_dataset_autolabel(request: Request):
    body = await request.json()
    files = body.get("files") or []
    labels = []
    for item in files[: int(body.get("limit") or 24)]:
        path = Path(str(item.get("path") if isinstance(item, dict) else item))
        duration = None
        try:
            info = sf.info(str(path))
            duration = round(info.frames / info.samplerate, 2)
        except Exception:
            pass
        labels.append(
            {
                "path": str(path),
                "filename": path.name,
                "caption": (item.get("caption") if isinstance(item, dict) else "") or path.stem.replace("-", " ").replace("_", " "),
                "lyrics": (item.get("lyrics") if isinstance(item, dict) else "") or "[Instrumental]",
                "genre": item.get("genre", "") if isinstance(item, dict) else "",
                "bpm": (item.get("bpm") if isinstance(item, dict) else None) or None,
                "keyscale": (item.get("keyscale") if isinstance(item, dict) else "") or "",
                "timesignature": (item.get("timesignature") if isinstance(item, dict) else "") or "4",
                "language": (item.get("language") if isinstance(item, dict) else "") or "instrumental",
                "duration": duration or (item.get("duration") if isinstance(item, dict) else 0),
                "is_instrumental": True,
            }
        )
    return JSONResponse({"success": True, "labels": labels})


@app.post("/api/lora/train")
async def api_lora_train(request: Request):
    try:
        body = await request.json()
        job = training_manager.start_train(body)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/dataset/save")
async def api_lora_dataset_save(request: Request):
    try:
        body = await request.json()
        entries = body.get("entries") or body.get("labels") or body.get("files") or []
        data = training_manager.save_dataset(
            entries,
            dataset_id=str(body.get("dataset_id") or ""),
            metadata=body.get("metadata") or {},
        )
        return JSONResponse({"success": True, **data})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/preprocess")
async def api_lora_preprocess(request: Request):
    try:
        body = await request.json()
        job = training_manager.start_preprocess(body)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/lora/estimate")
async def api_lora_estimate(request: Request):
    try:
        body = await request.json()
        job = training_manager.start_estimate(body)
        return JSONResponse({"success": True, "job": job})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.get("/api/lora/jobs")
async def api_lora_jobs():
    return JSONResponse({"success": True, "jobs": training_manager.list_jobs()})


@app.get("/api/lora/jobs/{job_id}/log")
async def api_lora_job_log(job_id: str):
    try:
        return JSONResponse({"success": True, **training_manager.read_log(job_id)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)


@app.get("/api/lora/jobs/{job_id}")
async def api_lora_job(job_id: str):
    try:
        return JSONResponse({"success": True, "job": training_manager.get_job(job_id)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)


@app.post("/api/lora/jobs/{job_id}/stop")
async def api_lora_job_stop(job_id: str):
    try:
        return JSONResponse({"success": True, "job": training_manager.stop_job(job_id)})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=404)


@app.get("/api/lora/adapters")
async def api_lora_adapters():
    return JSONResponse({"success": True, "adapters": training_manager.list_adapters()})


@app.post("/api/lora/export")
async def api_lora_export(request: Request):
    body = await request.json()
    try:
        source_text = str(body.get("source_path") or "").strip()
        if not source_text:
            return JSONResponse({"success": False, "error": "LoRA source path is required"}, status_code=400)
        data = training_manager.export_adapter(
            Path(source_text),
            str(body.get("name") or ""),
        )
        return JSONResponse(data)
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/audio-understand")
async def api_audio_understand(request: Request):
    try:
        _ensure_training_idle()
        body = await request.json()
        audio_path = _resolve_upload_file(body.get("upload_id")) or _resolve_result_audio(body.get("result_id"), body.get("audio_id"))
        if audio_path is None:
            return JSONResponse({"success": False, "error": "No audio supplied"}, status_code=400)
        info = sf.info(str(audio_path))
        with handler_lock:
            _ensure_song_model(body.get("song_model"))
            codes = handler.convert_src_audio_to_codes(str(audio_path))
        return JSONResponse(
            {
                "success": True,
                "duration": round(info.frames / info.samplerate, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "audio_codes": codes,
            }
        )
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/generate_album")
async def api_generate_album(request: Request):
    body = await request.json()
    raw = generate_album(
        concept=str(body.get("concept") or ""),
        num_tracks=int(body.get("num_tracks") or 5),
        track_duration=float(body.get("track_duration") or body.get("duration") or 180.0),
        ollama_model=str(body.get("ollama_model") or "llama3.2"),
        language=str(body.get("language") or "en"),
        song_model=str(body.get("song_model") or "auto"),
        embedding_model=str(body.get("embedding_model") or "nomic-embed-text"),
        ace_lm_model=str(body.get("ace_lm_model") or "auto"),
        request_json=json.dumps(body),
    )
    return JSONResponse(json.loads(raw))


@app.get("/media/songs/{song_id}/{filename}")
async def media(song_id: str, filename: str):
    songs_root = SONGS_DIR.resolve()
    song_dir = (SONGS_DIR / song_id).resolve()
    target = (song_dir / filename).resolve()
    if songs_root not in song_dir.parents or not song_dir.is_dir() or song_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/media/results/{result_id}/{filename}")
async def result_media(result_id: str, filename: str):
    target = _resolve_child(RESULTS_DIR, safe_id(result_id), filename)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


demo = app


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False)
