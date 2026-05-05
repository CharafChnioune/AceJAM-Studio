from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
ART_DIR = DATA_DIR / "art"
MFLUX_RESULTS_DIR = DATA_DIR / "mflux" / "results"

MLX_VIDEO_DIR = DATA_DIR / "mlx_video"
MLX_VIDEO_RESULTS_DIR = MLX_VIDEO_DIR / "results"
MLX_VIDEO_JOBS_DIR = MLX_VIDEO_DIR / "jobs"
MLX_VIDEO_UPLOADS_DIR = MLX_VIDEO_DIR / "uploads"
MLX_VIDEO_LORAS_DIR = MLX_VIDEO_DIR / "loras"
MLX_VIDEO_MODEL_DIRS_DIR = MLX_VIDEO_DIR / "model_dirs"
MLX_VIDEO_MODEL_REGISTRY_PATH = MLX_VIDEO_DIR / "model_registry.json"
MLX_VIDEO_ATTACHMENTS_PATH = MLX_VIDEO_DIR / "attachments.json"

MLX_VIDEO_ENV_DIR = BASE_DIR / "video-env"
MLX_VIDEO_VENDOR_DIR = BASE_DIR / "vendor" / "mlx-video"
MLX_VIDEO_REPO_URL = "https://github.com/Blaizzy/mlx-video.git"
MLX_VIDEO_TARGET_REF = "main"

MLX_VIDEO_ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MLX_VIDEO_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
MLX_VIDEO_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
MLX_VIDEO_ALLOWED_UPLOAD_EXTENSIONS = (
    MLX_VIDEO_ALLOWED_IMAGE_EXTENSIONS | MLX_VIDEO_ALLOWED_AUDIO_EXTENSIONS | MLX_VIDEO_ALLOWED_VIDEO_EXTENSIONS
)

for _path in (
    MLX_VIDEO_RESULTS_DIR,
    MLX_VIDEO_JOBS_DIR,
    MLX_VIDEO_UPLOADS_DIR,
    MLX_VIDEO_LORAS_DIR,
    MLX_VIDEO_MODEL_DIRS_DIR,
):
    _path.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _safe_id(value: str | None) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-")
    return text[:96] or uuid.uuid4().hex[:12]


def _safe_slug(value: str | None, fallback: str = "mlx-video") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text or fallback)[:90]


def _resolve_child(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    target = root.joinpath(*parts).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise FileNotFoundError("Path escapes MLX video data directory")
    return target


def _env_python() -> Path:
    if sys.platform == "win32":
        return MLX_VIDEO_ENV_DIR / "Scripts" / "python.exe"
    return MLX_VIDEO_ENV_DIR / "bin" / "python"


def _env_bin(name: str) -> Path:
    if sys.platform == "win32":
        return MLX_VIDEO_ENV_DIR / "Scripts" / f"{name}.exe"
    return MLX_VIDEO_ENV_DIR / "bin" / name


def _run_probe(command: list[str], timeout: int = 8) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return False, str(exc)
    text = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, text[-1200:]


def _python_version_info(python: Path) -> dict[str, Any]:
    if not python.is_file():
        return {"available": False, "version": "", "ok": False, "reason": "video-env python is missing"}
    ok, out = _run_probe([str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"])
    version = out.strip().splitlines()[-1] if out.strip() else ""
    major_minor = tuple(int(part) for part in version.split(".")[:2] if part.isdigit())
    return {
        "available": ok,
        "version": version,
        "ok": bool(ok and major_minor >= (3, 11)),
        "reason": "" if ok and major_minor >= (3, 11) else "mlx-video requires Python >= 3.11",
    }


def _python_import_status(python: Path, module: str) -> dict[str, Any]:
    if not python.is_file():
        return {"available": False, "reason": "video-env python is missing"}
    ok, out = _run_probe([str(python), "-c", f"import {module}; print(getattr({module}, '__version__', 'ok'))"])
    return {"available": ok, "version": out.strip().splitlines()[-1] if ok and out.strip() else "", "reason": "" if ok else out}


def _module_available_current_env(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _command_candidate(name: str) -> str | None:
    local = _env_bin(name)
    if local.is_file():
        return str(local)
    return shutil.which(name)


def _engine_command(engine: str) -> list[str]:
    python = _env_python()
    if engine == "ltx":
        console = _command_candidate("mlx_video.ltx_2.generate")
        module = "mlx_video.models.ltx_2.generate"
    elif engine == "wan":
        console = _command_candidate("mlx_video.wan_2.generate")
        module = "mlx_video.models.wan_2.generate"
    else:
        raise RuntimeError(f"Unknown MLX video engine: {engine}")
    if console:
        return [console]
    if python.is_file():
        return [str(python), "-m", module]
    return [sys.executable, "-m", module]


def _engine_source_text(engine: str) -> str:
    if engine == "ltx":
        path = MLX_VIDEO_VENDOR_DIR / "mlx_video" / "models" / "ltx_2" / "generate.py"
    elif engine == "wan":
        path = MLX_VIDEO_VENDOR_DIR / "mlx_video" / "models" / "wan_2" / "generate.py"
    else:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
    except Exception:
        return ""


def _parse_help_capabilities(engine: str, text: str) -> dict[str, Any]:
    blob = text or ""
    if engine == "ltx":
        return {
            "image": "--image" in blob,
            "end_image": "--end-image" in blob,
            "audio_file": "--audio-file" in blob,
            "enhance_prompt": "--enhance-prompt" in blob,
            "spatial_upscaler": "--spatial-upscaler" in blob,
            "tiling": "--tiling" in blob,
            "negative_prompt": "--negative-prompt" in blob,
            "ltx_lora": "--lora-path" in blob,
            "output_path": "--output-path" in blob or "-o" in blob,
        }
    if engine == "wan":
        return {
            "image": "--image" in blob,
            "negative_prompt": "--negative-prompt" in blob,
            "no_negative_prompt": "--no-negative-prompt" in blob,
            "tiling": "--tiling" in blob,
            "wan_lora_shared": "--lora" in blob,
            "wan_lora_high": "--lora-high" in blob,
            "wan_lora_low": "--lora-low" in blob,
            "output_path": "--output-path" in blob,
        }
    return {}


def _engine_capabilities(engine: str, help_status: dict[str, Any] | None = None) -> dict[str, Any]:
    source_caps = _parse_help_capabilities(engine, _engine_source_text(engine))
    help_caps = dict((help_status or {}).get("capabilities") or {})
    caps = {**source_caps, **help_caps}
    if engine == "ltx":
        # Main exposes these today; keeping them true lets the UI show stable defaults
        # even before Install/Update has created video-env and `--help` can run.
        caps.setdefault("image", True)
        caps.setdefault("audio_file", True)
        caps.setdefault("enhance_prompt", True)
        caps.setdefault("spatial_upscaler", True)
        caps.setdefault("tiling", True)
        caps.setdefault("ltx_lora", True)
        caps.setdefault("output_path", True)
    if engine == "wan":
        caps.setdefault("image", True)
        caps.setdefault("negative_prompt", True)
        caps.setdefault("no_negative_prompt", True)
        caps.setdefault("tiling", True)
        caps.setdefault("wan_lora_shared", True)
        caps.setdefault("wan_lora_high", True)
        caps.setdefault("wan_lora_low", True)
        caps.setdefault("output_path", True)
    return caps


def _output_flag(engine: str, capabilities: dict[str, Any] | None = None) -> str:
    caps = capabilities or _engine_capabilities(engine)
    if engine in {"ltx", "wan"} and caps.get("output_path", True):
        return "--output-path"
    return "-o"


def _help_status(engine: str) -> dict[str, Any]:
    command = _engine_command(engine)
    if command[0] == sys.executable and not _env_python().is_file():
        caps = _engine_capabilities(engine)
        return {
            "available": False,
            "help_ok": False,
            "command": command,
            "reason": "video-env is not installed",
            "capabilities": caps,
            "output_flag": _output_flag(engine, caps),
        }
    ok, out = _run_probe(command + ["--help"], timeout=12)
    caps = _engine_capabilities(engine, {"capabilities": _parse_help_capabilities(engine, out if ok else "")})
    return {
        "available": True,
        "help_ok": ok,
        "command": command,
        "reason": "" if ok else out,
        "capabilities": caps,
        "output_flag": _output_flag(engine, caps),
        "help_excerpt": out[-1200:] if ok else "",
    }


def _patch_status() -> dict[str, Any]:
    video_vae = MLX_VIDEO_VENDOR_DIR / "mlx_video" / "models" / "ltx_2" / "video_vae" / "video_vae.py"
    sampling = MLX_VIDEO_VENDOR_DIR / "mlx_video" / "models" / "ltx_2" / "video_vae" / "sampling.py"
    ltx_generate = MLX_VIDEO_VENDOR_DIR / "mlx_video" / "models" / "ltx_2" / "generate.py"
    video_vae_text = video_vae.read_text(encoding="utf-8", errors="ignore") if video_vae.is_file() else ""
    sampling_text = sampling.read_text(encoding="utf-8", errors="ignore") if sampling.is_file() else ""
    ltx_text = ltx_generate.read_text(encoding="utf-8", errors="ignore") if ltx_generate.is_file() else ""
    pr27 = "max_channels" in video_vae_text and "min(in_channels * multiplier" in video_vae_text
    pr24 = "conv branch" in sampling_text.lower() and "x_conv" in sampling_text and "x_in" in sampling_text
    pr23 = "--end-image" in ltx_text and "end_image_strength" in ltx_text
    return {
        "vendor_dir": str(MLX_VIDEO_VENDOR_DIR),
        "commit": _current_vendor_commit(),
        "pr27_ltx23_vae_channel_cap": pr27,
        "pr24_ltx23_sampling_fallback": pr24,
        "vae_fix_active": bool(pr27 or pr24),
        "pr23_ltx_i2v_end_frame": pr23,
        "tokenizer_issue_26_guarded": True,
        "helios_pr21_enabled": False,
        "training_available": False,
        "training_reason": "mlx-video exposes no stable video training console script in pyproject.toml yet.",
    }



MLX_VIDEO_MODELS: list[dict[str, Any]] = [
    {
        "id": "ltx2-fast-draft",
        "label": "LTX-2 Fast Draft",
        "engine": "ltx",
        "preset": "fast_draft",
        "pipeline": "distilled",
        "model_repo": "Lightricks/LTX-2",
        "default_width": 512,
        "default_height": 320,
        "default_frames": 33,
        "default_fps": 24,
        "default_steps": 8,
        "supports_lora": True,
        "capabilities": ["t2v", "i2v", "a2v", "song_video", "final"],
        "description": "Snelle kleine preview op LTX-2 distilled. Goed voor veel proberen zonder minuten te wachten.",
    },
    {
        "id": "ltx2-final-hq",
        "label": "LTX-2 Final HQ",
        "engine": "ltx",
        "preset": "final_hq",
        "pipeline": "dev-two-stage-hq",
        "model_repo": "prince-canuma/LTX-2-dev",
        "default_width": 768,
        "default_height": 512,
        "default_frames": 97,
        "default_fps": 24,
        "default_steps": 30,
        "cfg_scale": 3.0,
        "supports_lora": True,
        "capabilities": ["t2v", "i2v", "a2v", "song_video", "final"],
        "description": "Langzamere final pass met dev-two-stage-hq en dezelfde seed/source als je draft.",
    },
    {
        "id": "wan21-reality-480p",
        "label": "Wan2.1 Reality 480P",
        "engine": "wan",
        "preset": "reality_480p",
        "requires_model_dir": True,
        "family": "wan21",
        "default_width": 832,
        "default_height": 480,
        "default_frames": 81,
        "default_steps": 50,
        "guide_scale": 5.0,
        "shift": 5.0,
        "supports_lora": True,
        "capabilities": ["t2v", "final"],
        "description": "Native 480P Wan2.1 1.3B style preset. Vereist een geconverteerde MLX model-dir.",
    },
    {
        "id": "wan22-lightning-draft",
        "label": "Wan2.2 Lightning Draft",
        "engine": "wan",
        "preset": "wan_lightning",
        "requires_model_dir": True,
        "family": "wan22",
        "default_width": 480,
        "default_height": 704,
        "default_frames": 41,
        "default_steps": 4,
        "guide_scale": 1.0,
        "trim_first_frames": 1,
        "supports_lora": True,
        "capabilities": ["t2v", "i2v", "final"],
        "description": "Snelle Wan2.2 preview voor Lightning high/low LoRA setups.",
    },
    {
        "id": "wan22-final-hq",
        "label": "Wan2.2 Final HQ",
        "engine": "wan",
        "preset": "wan_final",
        "requires_model_dir": True,
        "family": "wan22",
        "default_width": 1280,
        "default_height": 704,
        "default_frames": 81,
        "default_steps": 40,
        "guide_scale": "3.0,4.0",
        "shift": 12.0,
        "supports_lora": True,
        "capabilities": ["t2v", "i2v", "final"],
        "description": "Hogere Wan2.2 final render wanneer de draft goed genoeg is.",
    },
    {
        "id": "helios-experimental",
        "label": "Helios Experimental",
        "engine": "helios",
        "preset": "disabled",
        "disabled": True,
        "capabilities": [],
        "description": "Uitgeschakeld: upstream PR is niet mergeable en het minute-scale model past niet bij draft-first.",
    },
]

MLX_VIDEO_ACTIONS: dict[str, dict[str, Any]] = {
    "t2v": {"label": "Text to video", "requires_prompt": True, "requires_image": False, "requires_audio": False},
    "i2v": {"label": "Image to video", "requires_prompt": True, "requires_image": True, "requires_audio": False},
    "a2v": {"label": "Audio to video", "requires_prompt": True, "requires_image": False, "requires_audio": True},
    "song_video": {"label": "Song to video", "requires_prompt": True, "requires_image": False, "requires_audio": True},
    "final": {"label": "Final rerender", "requires_prompt": True, "requires_image": False, "requires_audio": False},
}


def _model_by_id(model_id: str | None) -> dict[str, Any]:
    wanted = str(model_id or "ltx2-fast-draft").strip()
    for model in MLX_VIDEO_MODELS:
        if model["id"] == wanted:
            return dict(model)
    raise RuntimeError(f"Unknown MLX video model preset: {wanted}")


def _action_models(action: str) -> list[dict[str, Any]]:
    cap = "song_video" if action == "song_video" else action
    if action == "final":
        cap = "final"
    return [dict(model) for model in MLX_VIDEO_MODELS if cap in model.get("capabilities", []) and not model.get("disabled")]


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")


def _infer_wan_family(path: Path, fallback: str = "") -> str:
    blob = f"{path.name} {fallback}".lower()
    if "wan2.2" in blob or "wan22" in blob:
        return "wan22"
    if "wan2.1" in blob or "wan21" in blob:
        return "wan21"
    cfg = path / "config.json"
    if cfg.is_file():
        try:
            text = cfg.read_text(encoding="utf-8").lower()
            if "wan2.2" in text or "wan22" in text:
                return "wan22"
            if "wan2.1" in text or "wan21" in text:
                return "wan21"
        except Exception:
            pass
    return fallback or "wan"


def mlx_video_registered_model_dirs() -> list[dict[str, Any]]:
    raw = _read_json(MLX_VIDEO_MODEL_REGISTRY_PATH, [])
    items: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            path = Path(str(entry.get("path") or "")).expanduser()
            if path and path.is_dir():
                family = str(entry.get("family") or _infer_wan_family(path))
                items.append(
                    {
                        "id": str(entry.get("id") or _safe_slug(path.name, "wan-model")),
                        "label": str(entry.get("label") or path.name),
                        "path": str(path.resolve()),
                        "family": family,
                        "exists": True,
                        "config_path": str(path / "config.json") if (path / "config.json").is_file() else "",
                    }
                )
    for child in sorted(MLX_VIDEO_MODEL_DIRS_DIR.iterdir() if MLX_VIDEO_MODEL_DIRS_DIR.is_dir() else []):
        if child.is_dir() and (child / "config.json").is_file() and not any(item["path"] == str(child.resolve()) for item in items):
            items.append(
                {
                    "id": _safe_slug(child.name, "wan-model"),
                    "label": child.name,
                    "path": str(child.resolve()),
                    "family": _infer_wan_family(child),
                    "exists": True,
                    "config_path": str(child / "config.json"),
                }
            )
    return items


def mlx_video_register_model_dir(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(payload.get("path") or "")).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise RuntimeError(f"Model directory does not exist: {path}")
    entry = {
        "id": _safe_slug(payload.get("id") or path.name, "wan-model"),
        "label": str(payload.get("label") or path.name),
        "path": str(path),
        "family": str(payload.get("family") or _infer_wan_family(path)),
        "registered_at": _now(),
    }
    current = [item for item in _read_json(MLX_VIDEO_MODEL_REGISTRY_PATH, []) if isinstance(item, dict)]
    current = [item for item in current if str(item.get("path")) != str(path) and str(item.get("id")) != entry["id"]]
    current.append(entry)
    _write_json(MLX_VIDEO_MODEL_REGISTRY_PATH, current)
    return entry


def _current_vendor_commit() -> str:
    if not (MLX_VIDEO_VENDOR_DIR / ".git").is_dir():
        return ""
    ok, out = _run_probe(["git", "-C", str(MLX_VIDEO_VENDOR_DIR), "rev-parse", "--short", "HEAD"], timeout=5)
    return out.strip() if ok else ""


def mlx_video_status(check_help: bool = True) -> dict[str, Any]:
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    python = _env_python()
    py_info = _python_version_info(python)
    mlx_info = _python_import_status(python, "mlx") if python.is_file() else {"available": _module_available_current_env("mlx"), "reason": ""}
    package_info = _python_import_status(python, "mlx_video") if python.is_file() else {"available": False, "reason": "video-env is not installed"}
    commands = {
        "ltx": _engine_command("ltx"),
        "wan": _engine_command("wan"),
    }
    help_status = {engine: _help_status(engine) for engine in commands} if check_help else {}
    ready = bool(apple and py_info.get("ok") and mlx_info.get("available") and package_info.get("available"))
    reason = ""
    if not apple:
        reason = "MLX Video Studio requires Apple Silicon (darwin arm64)."
    elif not py_info.get("ok"):
        reason = str(py_info.get("reason") or "Install/Update must create app/video-env with Python >= 3.11.")
    elif not mlx_info.get("available"):
        reason = "MLX is not importable in app/video-env."
    elif not package_info.get("available"):
        reason = "mlx-video is not installed in app/video-env. Re-run Install/Update."
    return {
        "success": True,
        "ready": ready,
        "platform": sys.platform,
        "arch": platform.machine(),
        "apple_silicon": apple,
        "video_env_dir": str(MLX_VIDEO_ENV_DIR),
        "video_python": str(python),
        "python": py_info,
        "mlx_available": bool(mlx_info.get("available")),
        "mlx": mlx_info,
        "mlx_video_available": bool(package_info.get("available")),
        "mlx_video": package_info,
        "commands": commands,
        "command_help": help_status,
        "data_dir": str(MLX_VIDEO_DIR),
        "results_dir": str(MLX_VIDEO_RESULTS_DIR),
        "uploads_dir": str(MLX_VIDEO_UPLOADS_DIR),
        "lora_dir": str(MLX_VIDEO_LORAS_DIR),
        "model_dirs_dir": str(MLX_VIDEO_MODEL_DIRS_DIR),
        "registered_model_dirs": mlx_video_registered_model_dirs(),
        "patch_status": _patch_status(),
        "blocking_reason": reason,
    }


def mlx_video_models() -> dict[str, Any]:
    by_action = {action: _action_models(action) for action in MLX_VIDEO_ACTIONS}
    presets: dict[str, list[dict[str, Any]]] = {}
    for model in MLX_VIDEO_MODELS:
        presets.setdefault(str(model.get("preset") or "other"), []).append(dict(model))
    return {
        "success": True,
        "models": [dict(model) for model in MLX_VIDEO_MODELS],
        "presets": presets,
        "actions": MLX_VIDEO_ACTIONS,
        "by_action": by_action,
        "defaults": {
            "t2v": "ltx2-fast-draft",
            "i2v": "ltx2-fast-draft",
            "a2v": "ltx2-fast-draft",
            "song_video": "ltx2-fast-draft",
            "final": "ltx2-final-hq",
            "wan_480p": "wan21-reality-480p",
            "wan_lightning": "wan22-lightning-draft",
        },
        "registered_model_dirs": mlx_video_registered_model_dirs(),
    }


def _nearest_ltx_frames(value: Any) -> int:
    try:
        frames = int(value)
    except (TypeError, ValueError):
        frames = 33
    frames = max(9, frames)
    k = max(1, round((frames - 1) / 8))
    return 1 + 8 * k


def _nearest_wan_frames(value: Any) -> int:
    try:
        frames = int(value)
    except (TypeError, ValueError):
        frames = 81
    frames = max(5, frames)
    k = max(1, round((frames - 1) / 4))
    return 1 + 4 * k


def _align_dim(value: Any, *, multiple: int = 64, fallback: int = 512) -> int:
    try:
        dim = int(value)
    except (TypeError, ValueError):
        dim = fallback
    dim = max(multiple, dim)
    return max(multiple, round(dim / multiple) * multiple)


def _media_url_to_path(text: str) -> Path | None:
    if text.startswith("/media/mlx-video/uploads/"):
        parts = text.split("/media/mlx-video/uploads/", 1)[1].split("/", 1)
        if len(parts) == 2:
            return _resolve_child(MLX_VIDEO_UPLOADS_DIR, _safe_id(parts[0]), Path(parts[1]).name)
    if text.startswith("/media/mlx-video/"):
        parts = text.split("/media/mlx-video/", 1)[1].split("/", 1)
        if len(parts) == 2:
            return _resolve_child(MLX_VIDEO_RESULTS_DIR, _safe_id(parts[0]), Path(parts[1]).name)
    if text.startswith("/media/results/"):
        parts = text.split("/media/results/", 1)[1].split("/", 1)
        if len(parts) == 2:
            return _resolve_child(RESULTS_DIR, _safe_id(parts[0]), Path(parts[1]).name)
    if text.startswith("/media/mflux/"):
        parts = text.split("/media/mflux/", 1)[1].split("/", 1)
        if len(parts) == 2:
            return _resolve_child(MFLUX_RESULTS_DIR, _safe_id(parts[0]), Path(parts[1]).name)
    if text.startswith("/media/art/"):
        parts = text.split("/media/art/", 1)[1].split("/", 1)
        if len(parts) == 2:
            return _resolve_child(ART_DIR, _safe_id(parts[0]), Path(parts[1]).name)
    return None


def mlx_video_public_upload_url(upload_id: str, filename: str) -> str:
    return f"/media/mlx-video/uploads/{_safe_id(upload_id)}/{Path(filename).name}"


def mlx_video_public_result_url(result_id: str, filename: str) -> str:
    return f"/media/mlx-video/{_safe_id(result_id)}/{Path(filename).name}"


def mlx_video_validate_media_path(value: str | None, media_kind: str, *, required: bool = False) -> Path | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise RuntimeError(f"{media_kind.capitalize()} input is required.")
        return None
    if text.startswith("data:"):
        path = _write_data_url_upload(text, media_kind)
        if path is None:
            raise RuntimeError(f"Unsupported {media_kind} data URL.")
        return path
    path = _media_url_to_path(text) if text.startswith("/media/") else None
    if path is None:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        else:
            path = path.resolve()
    allowed_roots = [
        DATA_DIR.resolve(),
        MLX_VIDEO_UPLOADS_DIR.resolve(),
        MLX_VIDEO_RESULTS_DIR.resolve(),
        RESULTS_DIR.resolve(),
        ART_DIR.resolve(),
        MFLUX_RESULTS_DIR.resolve(),
    ]
    if not any(root == path or root in path.parents for root in allowed_roots):
        raise RuntimeError("Media path must come from MLX Media uploads, results or data folders.")
    ext = path.suffix.lower()
    allowed = {
        "image": MLX_VIDEO_ALLOWED_IMAGE_EXTENSIONS,
        "audio": MLX_VIDEO_ALLOWED_AUDIO_EXTENSIONS,
        "video": MLX_VIDEO_ALLOWED_VIDEO_EXTENSIONS,
    }.get(media_kind, MLX_VIDEO_ALLOWED_UPLOAD_EXTENSIONS)
    if ext not in allowed:
        raise RuntimeError(f"Unsupported {media_kind} file type: {ext or '(none)'}")
    if not path.is_file():
        raise RuntimeError(f"{media_kind.capitalize()} file does not exist: {path}")
    return path


def _write_data_url_upload(text: str, media_kind: str) -> Path | None:
    match = re.match(r"^data:([^;,]+)(?:;[^,]*)?,(.*)$", text, re.DOTALL)
    if not match:
        return None
    mime = match.group(1).lower()
    payload = match.group(2)
    ext_map = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/aac": ".aac",
        "audio/mp4": ".m4a",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }
    ext = ext_map.get(mime)
    allowed = {
        "image": MLX_VIDEO_ALLOWED_IMAGE_EXTENSIONS,
        "audio": MLX_VIDEO_ALLOWED_AUDIO_EXTENSIONS,
        "video": MLX_VIDEO_ALLOWED_VIDEO_EXTENSIONS,
    }.get(media_kind, MLX_VIDEO_ALLOWED_UPLOAD_EXTENSIONS)
    if not ext or ext not in allowed:
        return None
    try:
        blob = base64.b64decode(payload, validate=False)
    except Exception:
        return None
    upload_id = _safe_id(f"data-url-{uuid.uuid4().hex[:10]}")
    path = _resolve_child(MLX_VIDEO_UPLOADS_DIR, upload_id, f"source{ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def _resolve_wan_model_dir(payload: dict[str, Any], model: dict[str, Any]) -> str:
    raw = str(payload.get("model_dir") or payload.get("wan_model_dir") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        else:
            path = path.resolve()
        if not path.is_dir():
            raise RuntimeError(f"Wan model directory does not exist: {path}")
        return str(path)
    family = str(model.get("family") or "")
    for item in mlx_video_registered_model_dirs():
        if not family or item.get("family") == family:
            return str(item["path"])
    raise RuntimeError(f"{model['label']} requires a registered converted Wan MLX model directory.")


def _normal_lora_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("lora_adapters") or payload.get("loras") or []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _ltx_lora_args(payload: dict[str, Any]) -> list[str]:
    items = _normal_lora_items(payload)
    if not items:
        return []
    first = items[0]
    path = str(first.get("path") or first.get("lora_path") or "").strip()
    if not path:
        return []
    path_obj = Path(path).expanduser()
    if not path_obj.exists() and not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
        raise RuntimeError(f"LTX LoRA path does not exist: {path}")
    scale = float(first.get("scale") or first.get("strength") or 1.0)
    args = ["--lora-path", path, "--lora-strength", str(max(0.0, min(2.0, scale)))]
    if str(first.get("stage_1_scale") or "").strip():
        args.extend(["--lora-strength-stage-1", str(first["stage_1_scale"])])
    if str(first.get("stage_2_scale") or "").strip():
        args.extend(["--lora-strength-stage-2", str(first["stage_2_scale"])])
    return args


def _wan_lora_args(payload: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for item in _normal_lora_items(payload):
        path = str(item.get("path") or item.get("lora_path") or "").strip()
        if not path:
            continue
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            raise RuntimeError(f"Wan LoRA path does not exist: {path}")
        scale = str(float(item.get("scale") or item.get("strength") or 1.0))
        role = str(item.get("role") or item.get("noise") or "shared").lower()
        if role == "high":
            args.extend(["--lora-high", str(path_obj.resolve()), scale])
        elif role == "low":
            args.extend(["--lora-low", str(path_obj.resolve()), scale])
        else:
            args.extend(["--lora", str(path_obj.resolve()), scale])
    return args


def _tokenizer_guard(payload: dict[str, Any], model: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    model_repo = str(payload.get("model_repo") or model.get("model_repo") or "")
    text_encoder_repo = str(payload.get("text_encoder_repo") or model.get("text_encoder_repo") or "")
    if "ltx-2.3" in model_repo.lower() and text_encoder_repo and "ltx-2.3" not in text_encoder_repo.lower():
        raise RuntimeError(
            "LTX-2.3 model/text-encoder mismatch blocked: issue #26 reports prompt-agnostic output when the wrong tokenizer is used."
        )
    if "ltx-2.3" in model_repo.lower() and not text_encoder_repo:
        warnings.append("LTX-2.3 selected; keep text_encoder_repo aligned with the model repo to avoid issue #26.")
    return warnings


def _build_ltx_command(payload: dict[str, Any], model: dict[str, Any], output: Path) -> tuple[list[str], list[str]]:
    warnings = _tokenizer_guard(payload, model)
    capabilities = _engine_capabilities("ltx")
    width = _align_dim(payload.get("width") or model.get("default_width"), fallback=int(model.get("default_width") or 512))
    height = _align_dim(payload.get("height") or model.get("default_height"), fallback=int(model.get("default_height") or 320))
    frames = _nearest_ltx_frames(payload.get("num_frames") or payload.get("frames") or model.get("default_frames"))
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Prompt is required for LTX video generation.")
    args = _engine_command("ltx") + [
        "--prompt",
        prompt,
        "--pipeline",
        str(payload.get("pipeline") or model.get("pipeline") or "distilled"),
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(frames),
        "--fps",
        str(int(payload.get("fps") or model.get("default_fps") or 24)),
        "--seed",
        str(payload.get("seed") if payload.get("seed") not in {None, ""} else -1),
        _output_flag("ltx", capabilities),
        str(output),
    ]
    model_repo = str(payload.get("model_repo") or model.get("model_repo") or "")
    if model_repo:
        args.extend(["--model-repo", model_repo])
    text_encoder_repo = str(payload.get("text_encoder_repo") or model.get("text_encoder_repo") or "")
    if text_encoder_repo:
        args.extend(["--text-encoder-repo", text_encoder_repo])
    if str(payload.get("negative_prompt") or "").strip():
        args.extend(["--negative-prompt", str(payload.get("negative_prompt")).strip()])
    if str(payload.get("steps") or model.get("default_steps") or "").strip():
        args.extend(["--steps", str(int(payload.get("steps") or model.get("default_steps")))])
    if str(payload.get("cfg_scale") or model.get("cfg_scale") or "").strip():
        args.extend(["--cfg-scale", str(payload.get("cfg_scale") or model.get("cfg_scale"))])
    if payload.get("enhance_prompt"):
        if not capabilities.get("enhance_prompt"):
            raise RuntimeError("Installed LTX command does not expose --enhance-prompt. Re-run Install/Update.")
        args.append("--enhance-prompt")
    if str(payload.get("spatial_upscaler") or "").strip():
        if not capabilities.get("spatial_upscaler"):
            raise RuntimeError("Installed LTX command does not expose --spatial-upscaler. Re-run Install/Update.")
        args.extend(["--spatial-upscaler", str(payload.get("spatial_upscaler"))])
    if payload.get("tiling"):
        if not capabilities.get("tiling"):
            raise RuntimeError("Installed LTX command does not expose --tiling. Re-run Install/Update.")
        args.append("--tiling")
    image_path = mlx_video_validate_media_path(
        payload.get("image_path") or payload.get("source_image_path") or payload.get("source_image_url"),
        "image",
        required=False,
    )
    end_image_path = mlx_video_validate_media_path(
        payload.get("end_image_path") or payload.get("source_end_image_path") or payload.get("source_end_image_url"),
        "image",
        required=False,
    )
    audio_path = mlx_video_validate_media_path(
        payload.get("audio_path") or payload.get("source_audio_path") or payload.get("audio_url"),
        "audio",
        required=False,
    )
    if image_path:
        args.extend(["--image", str(image_path)])
    if end_image_path:
        if not capabilities.get("end_image"):
            raise RuntimeError("LTX end-frame conditioning requires upstream PR #23 / --end-image support. Re-run Install/Update.")
        args.extend(["--end-image", str(end_image_path)])
        if str(payload.get("end_image_strength") or "").strip():
            args.extend(["--end-image-strength", str(payload.get("end_image_strength"))])
    if audio_path:
        args.extend(["--audio-file", str(audio_path)])
    if str(payload.get("audio_start_time") or "").strip():
        args.extend(["--audio-start-time", str(payload.get("audio_start_time"))])
    args.extend(_ltx_lora_args(payload))
    return args, warnings


def _build_wan_command(payload: dict[str, Any], model: dict[str, Any], output: Path) -> tuple[list[str], list[str]]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Prompt is required for Wan video generation.")
    width = _align_dim(payload.get("width") or model.get("default_width"), multiple=16, fallback=int(model.get("default_width") or 832))
    height = _align_dim(payload.get("height") or model.get("default_height"), multiple=16, fallback=int(model.get("default_height") or 480))
    frames = _nearest_wan_frames(payload.get("num_frames") or payload.get("frames") or model.get("default_frames"))
    args = _engine_command("wan") + [
        "--model-dir",
        _resolve_wan_model_dir(payload, model),
        "--prompt",
        prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(frames),
        _output_flag("wan"),
        str(output),
    ]
    image_path = mlx_video_validate_media_path(
        payload.get("image_path") or payload.get("source_image_path") or payload.get("source_image_url"),
        "image",
        required=False,
    )
    if image_path:
        args.extend(["--image", str(image_path)])
    if str(payload.get("negative_prompt") or "").strip():
        args.extend(["--negative-prompt", str(payload.get("negative_prompt")).strip()])
    if payload.get("no_negative_prompt"):
        args.append("--no-negative-prompt")
    if payload.get("tiling"):
        args.append("--tiling")
    if str(payload.get("steps") or model.get("default_steps") or "").strip():
        args.extend(["--steps", str(int(payload.get("steps") or model.get("default_steps")))])
    if str(payload.get("guide_scale") or model.get("guide_scale") or "").strip():
        args.extend(["--guide-scale", str(payload.get("guide_scale") or model.get("guide_scale"))])
    if str(payload.get("shift") or model.get("shift") or "").strip():
        args.extend(["--shift", str(payload.get("shift") or model.get("shift"))])
    if str(payload.get("seed") or "").strip():
        args.extend(["--seed", str(payload.get("seed"))])
    if str(payload.get("trim_first_frames") or model.get("trim_first_frames") or "").strip():
        args.extend(["--trim-first-frames", str(int(payload.get("trim_first_frames") or model.get("trim_first_frames")))])
    args.extend(_wan_lora_args(payload))
    return args, []


def _source_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source_id = str(payload.get("source_job_id") or payload.get("draft_job_id") or "").strip()
    if not source_id:
        return {}
    source = mlx_video_get_job(source_id) or {}
    if not source:
        raise RuntimeError(f"Source video job not found: {source_id}")
    base = dict(source.get("payload") or {})
    base.update({k: v for k, v in payload.items() if v not in {None, ""}})
    return base


def _build_mlx_video_command(payload: dict[str, Any], output: Path) -> tuple[list[str], dict[str, Any], list[str]]:
    if str(payload.get("action") or "t2v") == "final":
        payload = _source_job_payload(payload) or payload
        payload = {**payload, "action": "final", "model_id": payload.get("model_id") or "ltx2-final-hq"}
    model = _model_by_id(str(payload.get("model_id") or payload.get("model") or "ltx2-fast-draft"))
    if model.get("disabled"):
        raise RuntimeError(f"{model['label']} is disabled: {model.get('description')}")
    action = str(payload.get("action") or "t2v")
    spec = MLX_VIDEO_ACTIONS.get(action if action != "final" else "t2v") or MLX_VIDEO_ACTIONS["t2v"]
    if spec.get("requires_image"):
        mlx_video_validate_media_path(payload.get("image_path") or payload.get("source_image_path") or payload.get("source_image_url"), "image", required=True)
    if spec.get("requires_audio"):
        mlx_video_validate_media_path(payload.get("audio_path") or payload.get("source_audio_path") or payload.get("audio_url"), "audio", required=True)
    if model["engine"] == "ltx":
        command, warnings = _build_ltx_command(payload, model, output)
    elif model["engine"] == "wan":
        command, warnings = _build_wan_command(payload, model, output)
    else:
        raise RuntimeError(f"Unsupported MLX video engine: {model['engine']}")
    return command, model, warnings


_jobs_lock = threading.Lock()


def _job_file(job_id: str) -> Path:
    return MLX_VIDEO_JOBS_DIR / f"{_safe_id(job_id)}.json"


def _write_job(job: dict[str, Any]) -> dict[str, Any]:
    job["updated_at"] = _now()
    _job_file(str(job["id"])).write_text(json.dumps(_jsonable(job), indent=2), encoding="utf-8")
    return job


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_file(job_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _set_job(job_id: str, **patch: Any) -> dict[str, Any]:
    with _jobs_lock:
        job = _read_job(job_id) or {"id": _safe_id(job_id)}
        job.update(_jsonable(patch))
        return _write_job(job)


def mlx_video_get_job(job_id: str) -> dict[str, Any] | None:
    return _read_job(job_id)


def mlx_video_list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in sorted(MLX_VIDEO_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        job = _read_job(path.stem)
        if job:
            jobs.append(job)
        if len(jobs) >= limit:
            break
    return jobs


def _result_dir(result_id: str) -> Path:
    path = MLX_VIDEO_RESULTS_DIR / _safe_id(result_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_poster(video_path: Path, result_id: str) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not video_path.is_file():
        return ""
    poster = video_path.with_suffix(".jpg")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(poster)],
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return ""
    return mlx_video_public_result_url(result_id, poster.name) if poster.is_file() else ""


def _song_video_source_audio(payload: dict[str, Any], *, required: bool = True) -> Path | None:
    return mlx_video_validate_media_path(
        payload.get("audio_path") or payload.get("source_audio_path") or payload.get("audio_url") or payload.get("source_audio"),
        "audio",
        required=required,
    )


def _mux_source_audio(raw_video_path: Path, source_audio_path: Path, result_id: str) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to mux the original song audio into the video clip.")
    if not raw_video_path.is_file():
        raise RuntimeError(f"Raw generated video not found for audio mux: {raw_video_path}")
    if not source_audio_path.is_file():
        raise RuntimeError(f"Source audio not found for audio mux: {source_audio_path}")
    muxed = raw_video_path.with_name(f"{raw_video_path.stem}-source-audio.mp4")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_video_path),
        "-i",
        str(source_audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(muxed),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=900)
    if completed.returncode != 0 or not muxed.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()[-1600:]
        raise RuntimeError(f"Audio mux failed with ffmpeg exit {completed.returncode}: {detail}")
    return {
        "path": str(muxed),
        "url": mlx_video_public_result_url(result_id, muxed.name),
        "filename": muxed.name,
        "command": command,
        "logs": [f"$ {' '.join(command)}", (completed.stdout or "")[-4000:], (completed.stderr or "")[-4000:]],
    }


def _run_cli_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    result_id = _safe_id(str(payload.get("result_id") or f"mlx-video-{uuid.uuid4().hex[:12]}"))
    output_name = _safe_slug(payload.get("title") or payload.get("prompt") or result_id, "mlx-video") + ".mp4"
    output_path = _result_dir(result_id) / output_name
    command, model, warnings = _build_mlx_video_command(payload, output_path)
    capabilities = _engine_capabilities(str(model.get("engine") or ""))
    patch_status = _patch_status()
    _set_job(
        job_id,
        state="running",
        status="running",
        stage=f"{model['label']} running",
        progress=8,
        command=command,
        command_capabilities=capabilities,
        patch_status=patch_status,
        result_id=result_id,
        logs=[f"$ {' '.join(command)}"],
        warnings=warnings,
    )
    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    completed = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=max(600, int(payload.get("timeout_seconds") or 14400)),
    )
    logs = [f"$ {' '.join(command)}"]
    if completed.stdout:
        logs.append(completed.stdout[-8000:])
    if completed.stderr:
        logs.append(completed.stderr[-8000:])
    if completed.returncode != 0:
        raise RuntimeError(f"mlx-video exited with code {completed.returncode}: {completed.stderr[-1600:] or completed.stdout[-1600:]}")
    if not output_path.is_file():
        candidates = sorted(output_path.parent.glob("*.mp4")) + sorted(output_path.parent.glob("*.mov")) + sorted(output_path.parent.glob("*.webm"))
        if not candidates:
            raise RuntimeError("mlx-video finished but did not write a video file.")
        output_path = candidates[-1]
    action = str(payload.get("action") or "t2v")
    raw_video_path = output_path
    raw_video_url = mlx_video_public_result_url(result_id, raw_video_path.name)
    primary_video_path = raw_video_path
    primary_video_url = raw_video_url
    muxed_video_url = ""
    muxed_video_path = ""
    postprocess_status = "not_applicable"
    postprocess_error = ""
    mux_logs: list[str] = []
    mux_audio = action == "song_video" and str(payload.get("audio_policy") or "replace_with_source") == "replace_with_source"
    source_audio_path = _song_video_source_audio(payload, required=action == "song_video")
    if action == "song_video":
        if not mux_audio:
            raise RuntimeError("song_video requires audio_policy='replace_with_source' and mux_audio=true.")
        try:
            muxed = _mux_source_audio(raw_video_path, source_audio_path, result_id)  # type: ignore[arg-type]
            muxed_video_url = str(muxed["url"])
            muxed_video_path = str(muxed["path"])
            primary_video_url = muxed_video_url
            primary_video_path = Path(muxed_video_path)
            postprocess_status = "muxed"
            mux_logs = [line for line in (muxed.get("logs") or []) if line]
            logs.extend(mux_logs)
        except Exception as exc:
            postprocess_status = "failed"
            postprocess_error = str(exc)
            raise RuntimeError(f"song_video audio postprocess failed: {postprocess_error}") from exc
    poster_url = _make_poster(primary_video_path, result_id)
    metadata = {
        "result_id": result_id,
        "filename": primary_video_path.name,
        "path": str(primary_video_path),
        "url": primary_video_url,
        "video_url": primary_video_url,
        "primary_video_url": primary_video_url,
        "primary_video_path": str(primary_video_path),
        "raw_video_url": raw_video_url,
        "raw_video_path": str(raw_video_path),
        "muxed_video_url": muxed_video_url,
        "muxed_video_path": muxed_video_path,
        "audio_policy": "replace_with_source" if action == "song_video" else "none",
        "mux_audio": bool(action == "song_video"),
        "postprocess_status": postprocess_status,
        "postprocess_error": postprocess_error,
        "poster_url": poster_url,
        "model_id": model["id"],
        "model_label": model["label"],
        "engine": model["engine"],
        "preset": model.get("preset"),
        "action": action,
        "prompt": payload.get("prompt") or "",
        "seed": payload.get("seed"),
        "width": payload.get("width") or model.get("default_width"),
        "height": payload.get("height") or model.get("default_height"),
        "num_frames": payload.get("num_frames") or payload.get("frames") or model.get("default_frames"),
        "fps": payload.get("fps") or model.get("default_fps"),
        "source_image": payload.get("image_path") or payload.get("source_image_path") or payload.get("source_image_url") or "",
        "source_end_image": payload.get("end_image_path") or payload.get("source_end_image_path") or payload.get("source_end_image_url") or "",
        "source_audio": str(source_audio_path or payload.get("audio_path") or payload.get("source_audio_path") or payload.get("audio_url") or ""),
        "source_job_id": payload.get("source_job_id") or payload.get("draft_job_id") or "",
        "lora_adapters": payload.get("lora_adapters") or [],
        "enhance_prompt": bool(payload.get("enhance_prompt")),
        "spatial_upscaler": payload.get("spatial_upscaler") or "",
        "tiling": bool(payload.get("tiling")),
        "target_type": payload.get("target_type") or "",
        "target_id": payload.get("target_id") or "",
        "command": command,
        "command_capabilities": capabilities,
        "patch_status": patch_status,
        "replayable_payload": _jsonable(payload),
        "warnings": warnings,
        "attach_status": {},
        "created_at": _now(),
        "source": "mlx-video",
        "logs": logs,
    }
    (output_path.parent / "mlx_video_result.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return metadata


def mlx_video_create_job(payload: dict[str, Any], runner: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    status = mlx_video_status(check_help=False)
    if not status["ready"] and runner is None:
        raise RuntimeError(status["blocking_reason"])
    action = str(payload.get("action") or "t2v").strip()
    if action not in MLX_VIDEO_ACTIONS:
        raise RuntimeError(f"Unsupported MLX video action: {action}")
    if action == "song_video":
        payload = {**payload, "audio_policy": "replace_with_source", "mux_audio": True}
        if runner is None:
            _song_video_source_audio(payload, required=True)
    model = _model_by_id(str(payload.get("model_id") or payload.get("model") or ("ltx2-final-hq" if action == "final" else "ltx2-fast-draft")))
    if action not in model.get("capabilities", []) and not (action in {"a2v", "song_video"} and "a2v" in model.get("capabilities", [])):
        raise RuntimeError(f"{model['label']} does not support {action}.")
    if runner is None:
        _build_mlx_video_command(payload, MLX_VIDEO_RESULTS_DIR / "dry-run.mp4")
    job_id = _safe_id(str(payload.get("job_id") or f"mlx-video-{uuid.uuid4().hex[:12]}"))
    job = {
        "id": job_id,
        "kind": "mlx-video",
        "state": "queued",
        "status": "queued",
        "stage": "Queued",
        "progress": 0,
        "created_at": _now(),
        "payload": _jsonable({**payload, "action": action, "model_id": model["id"]}),
        "model": model,
        "logs": [],
    }
    _write_job(job)

    def worker() -> None:
        try:
            active = runner or _run_cli_job
            result = active(job_id, {**payload, "action": action, "model_id": model["id"]})
            _set_job(
                job_id,
                state="succeeded",
                status="succeeded",
                stage="Complete",
                progress=100,
                result=_jsonable(result),
                result_summary={
                    "result_id": result.get("result_id"),
                    "video_url": result.get("primary_video_url") or result.get("video_url") or result.get("url"),
                    "primary_video_url": result.get("primary_video_url") or result.get("video_url") or result.get("url"),
                    "raw_video_url": result.get("raw_video_url") or "",
                    "muxed_video_url": result.get("muxed_video_url") or "",
                    "poster_url": result.get("poster_url"),
                    "model_id": result.get("model_id"),
                    "action": result.get("action"),
                    "audio_policy": result.get("audio_policy"),
                    "mux_audio": result.get("mux_audio"),
                    "postprocess_status": result.get("postprocess_status"),
                    "postprocess_error": result.get("postprocess_error"),
                },
                finished_at=_now(),
                logs=result.get("logs") or (_read_job(job_id) or {}).get("logs", []),
            )
        except Exception as exc:
            current = _read_job(job_id) or {}
            logs = list(current.get("logs") or [])
            logs.append(str(exc))
            _set_job(
                job_id,
                state="failed",
                status="failed",
                stage="Failed",
                progress=max(1, int(current.get("progress") or 0)),
                error=str(exc),
                logs=logs,
                finished_at=_now(),
            )

    threading.Thread(target=worker, name=f"mlx-video-job-{job_id}", daemon=True).start()
    return _read_job(job_id) or job


def mlx_video_list_loras() -> list[dict[str, Any]]:
    adapters: list[dict[str, Any]] = []
    for child in sorted(MLX_VIDEO_LORAS_DIR.iterdir() if MLX_VIDEO_LORAS_DIR.is_dir() else [], key=lambda p: p.name.lower()):
        if not child.exists():
            continue
        meta: dict[str, Any] = {}
        if child.is_dir():
            for meta_name in ("mlx_video_lora.json", "metadata.json", "adapter_config.json"):
                path = child / meta_name
                if path.is_file():
                    try:
                        meta.update(json.loads(path.read_text(encoding="utf-8")))
                    except Exception:
                        pass
            files = [p for p in child.iterdir() if p.is_file() and p.suffix.lower() in {".safetensors", ".pt", ".ckpt"}]
            lora_path = files[0] if files else child
        elif child.suffix.lower() in {".safetensors", ".pt", ".ckpt"}:
            lora_path = child
        else:
            continue
        adapters.append(
            {
                "name": child.stem,
                "display_name": str(meta.get("display_name") or meta.get("name") or child.stem),
                "path": str(lora_path.resolve()),
                "adapter_type": "video_lora",
                "family": str(meta.get("family") or meta.get("model_family") or ""),
                "role": str(meta.get("role") or meta.get("noise") or "shared"),
                "model_id": str(meta.get("model_id") or meta.get("base_model") or ""),
                "generation_loadable": True,
                "is_loadable": True,
                "metadata": meta,
                "updated_at": datetime.fromtimestamp(child.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    return adapters


def mlx_video_attach(payload: dict[str, Any]) -> dict[str, Any]:
    result_id = _safe_id(str(payload.get("source_result_id") or payload.get("result_id") or ""))
    if not result_id:
        raise RuntimeError("source_result_id is required.")
    meta_path = _resolve_child(MLX_VIDEO_RESULTS_DIR, result_id, "mlx_video_result.json")
    meta = _read_json(meta_path, {})
    if not meta:
        raise RuntimeError("MLX video result metadata not found.")
    target_type = str(payload.get("target_type") or "video")
    target_id = str(payload.get("target_id") or "")
    attachment = {
        "video_id": f"mlx-video-{result_id}",
        "source": "mlx-video",
        "target_type": target_type,
        "target_id": target_id,
        "result_id": result_id,
        "url": meta.get("video_url") or meta.get("url"),
        "poster_url": meta.get("poster_url") or "",
        "path": meta.get("path") or "",
        "prompt": meta.get("prompt") or "",
        "model_label": meta.get("model_label") or "",
        "attached_at": _now(),
    }
    attachments = _read_json(MLX_VIDEO_ATTACHMENTS_PATH, [])
    if not isinstance(attachments, list):
        attachments = []
    attachments = [
        item
        for item in attachments
        if not (
            isinstance(item, dict)
            and item.get("target_type") == target_type
            and item.get("target_id") == target_id
            and item.get("result_id") == result_id
        )
    ]
    attachments.append(attachment)
    _write_json(MLX_VIDEO_ATTACHMENTS_PATH, attachments)
    meta.setdefault("attach_status", {})
    meta["attach_status"][target_type or "video"] = attachment
    _write_json(meta_path, meta)
    return attachment


def mlx_video_list_attachments(target_type: str | None = None, target_id: str | None = None) -> list[dict[str, Any]]:
    attachments = _read_json(MLX_VIDEO_ATTACHMENTS_PATH, [])
    if not isinstance(attachments, list):
        return []
    wanted_type = str(target_type or "").strip()
    wanted_id = str(target_id or "").strip()
    items: list[dict[str, Any]] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        if wanted_type and str(item.get("target_type") or "") != wanted_type:
            continue
        if wanted_id and str(item.get("target_id") or "") != wanted_id:
            continue
        items.append(item)
    return items


def mark_stale_mlx_video_jobs() -> None:
    for job in mlx_video_list_jobs(limit=200):
        if str(job.get("state") or "").lower() in {"queued", "running", "stopping"}:
            _set_job(
                str(job["id"]),
                state="failed",
                status="failed",
                stage="Interrupted by app restart",
                error="Interrupted by app restart",
                finished_at=_now(),
            )


if os.environ.get("ACEJAM_SKIP_MODEL_INIT_FOR_TESTS") != "1":
    try:
        mark_stale_mlx_video_jobs()
    except Exception:
        pass
